#include "./qoamax.h"
#include <stdckdint.h>
#include <string.h>

// <stdbit.h> is part of the C library (glibc since 2.39), which may be older than the compiler. gcc and clang get
// __builtin_ctz instead (see qoamax__trailing_zeros).
#if __has_include(<stdbit.h>) || !defined(__GNUC__)
#include <stdbit.h>
#define QOAMAX_HAS_STDBIT 1
#else
#define QOAMAX_HAS_STDBIT 0
#endif

///////////////////////////////////////////////////////////////////////////////
//                        Core De-/Encoding Functions                        //
///////////////////////////////////////////////////////////////////////////////

#define QOAMAX_MAX_BEAM_WIDTH 1024    //!< Largest beam width of the encoder efforts
#define QOAMAX_MAX_LOOKAHEAD_SLICES 4 //!< Largest lookahead of the encoder efforts (80 samples)
// The beam search's state is sized for the efforts' small beams, so that only wide beams (PLACEBO) need large stack
// frames (about 350 KB), in separate functions.
#define QOAMAX_SMALL_BEAM_WIDTH 24 //!< Largest beam width with a small search state (SUPERSLOW)
#define QOAMAX_SMALL_BEAM_NODES                                                                     \
  144 //!< Most nodes of all slices of a search kept in a small stack frame (SUPERSLOW: 3 * 2 * 24)

//! The dequantization values per scalefactor. Each row (a, b, c, d) expands to the
//! residuals {a, -a, b, -b, c, -c, d, -d}.
#define QOAMAX_DQT_ROWS(ROW)   \
  ROW(1, 3, 5, 7)              \
  ROW(5, 18, 32, 49)           \
  ROW(16, 53, 95, 147)         \
  ROW(34, 113, 203, 315)       \
  ROW(63, 210, 378, 588)       \
  ROW(104, 345, 621, 966)      \
  ROW(158, 528, 950, 1477)     \
  ROW(228, 760, 1368, 2128)    \
  ROW(316, 1053, 1895, 2947)   \
  ROW(422, 1405, 2529, 3934)   \
  ROW(548, 1828, 3290, 5117)   \
  ROW(696, 2320, 4176, 6496)   \
  ROW(868, 2893, 5207, 8099)   \
  ROW(1064, 3548, 6386, 9933)  \
  ROW(1286, 4288, 7718, 12005) \
  ROW(1536, 5120, 9216, 14336)

#define QOAMAX_DQT_ROW(a, b, c, d) {a, -a, b, -b, c, -c, d, -d},
static const int16_t qoamax_dqt[16][8] = {QOAMAX_DQT_ROWS(QOAMAX_DQT_ROW)};

//! Dequantization table entry with the precomputed LMS weight update.
typedef struct {
  int32_t dequantized;
  int32_t weight_delta[2]; //!< Indexed by 1 + (history >> 31): {-delta, +delta} for negative/positive history.
  int32_t padding;
} qoamax_decoder_entry;

#define QOAMAX_DECODER_ENTRY(dq) {(dq), {-((dq) >> 4), (dq) >> 4}, 0}
#define QOAMAX_DECODER_ROW(a, b, c, d)                                                                        \
  {QOAMAX_DECODER_ENTRY(a), QOAMAX_DECODER_ENTRY(-(a)), QOAMAX_DECODER_ENTRY(b), QOAMAX_DECODER_ENTRY(-(b)),  \
   QOAMAX_DECODER_ENTRY(c), QOAMAX_DECODER_ENTRY(-(c)), QOAMAX_DECODER_ENTRY(d), QOAMAX_DECODER_ENTRY(-(d))},
static const qoamax_decoder_entry qoamax_decoder_table[16][8] = {QOAMAX_DQT_ROWS(QOAMAX_DECODER_ROW)};

//! The quantization thresholds per scalefactor: the midpoints between neighbouring dequantization magnitudes.
#define QOAMAX_QT_ROW(a, b, c, d) {((a) + (b)) / 2, ((b) + (c)) / 2, ((c) + (d)) / 2},
static const int16_t qoamax_qt[16][3] = {QOAMAX_DQT_ROWS(QOAMAX_QT_ROW)};

#define qoamax_assume(x) ((x) ? (void)0 : unreachable())

// A branch hint where the compiler has __builtin_expect (__has_builtin isn't part of C23).
#if defined(__has_builtin)
#if __has_builtin(__builtin_expect)
#define QOAMAX_UNLIKELY(x) __builtin_expect((x), 0)
#endif
#endif
#ifndef QOAMAX_UNLIKELY
#define QOAMAX_UNLIKELY(x) (x)
#endif

#define QOAMAX_MIN(a, b) (((a) < (b)) ? (a) : (b))
#define QOAMAX_MAX(a, b) (((a) > (b)) ? (a) : (b))

// Fully unrolls the 20-sample loops, unless the compiler optimizes for size (-Os, -Oz).
#define QOAMAX_PRAGMA(x) _Pragma(#x)
#ifdef __OPTIMIZE_SIZE__
#define QOAMAX_UNROLL_SLICE
#else
#define QOAMAX_UNROLL_SLICE QOAMAX_PRAGMA(GCC unroll 20)
#endif

//! Saturates the value to the int16 range. Clipping is rare, so a branch keeps the clamp off the dependency chain.
[[gnu::always_inline]]
static inline int32_t qoamax_clamp_s16(int32_t value) {
  if (QOAMAX_UNLIKELY((int16_t)value != value)) value = value < 0 ? INT16_MIN : INT16_MAX;
  return value;
}

//! Returns the squared error value, used to calculate or minimize the RMS error.
[[gnu::always_inline]]
static inline uint32_t qoamax_error_squared(int16_t actual, int16_t decoded) {
  int32_t diff   = ((int32_t)actual) - decoded;
  uint32_t error = ((uint32_t)diff) * ((uint32_t)diff);
  return error;
}

//! Returns the squared error value, used to calculate or minimize the RMS error.
[[gnu::always_inline]]
static inline uint64_t qoamax_error_squared_unclamped(int16_t actual, int32_t decoded) {
  int32_t diff = ((int32_t)actual) - decoded;
  qoamax_assume(diff == ((int64_t)actual) - decoded);
  uint64_t error = ((uint64_t)diff) * ((uint64_t)diff);
  return error;
}

//! Returns the number of trailing zero bits of a non-zero value.
[[gnu::always_inline]]
static inline int qoamax__trailing_zeros(unsigned value) {
  qoamax_assume(value != 0);
#if QOAMAX_HAS_STDBIT
  return (int)stdc_trailing_zeros(value);
#else
  return __builtin_ctz(value);
#endif
}

//! Returns a + b, saturating at UINT64_MAX.
[[gnu::always_inline]]
static inline uint64_t qoamax_add_saturating(uint64_t a, uint64_t b) {
  uint64_t sum;
  return ckd_add(&sum, a, b) ? UINT64_MAX : sum;
}

//! The LMS filter's state as local variables. The hot loops work on this copy instead of a qoamax_lms, which keeps
//! the state in registers (clang keeps *lms or small arrays in memory, roughly halving the throughput).
typedef struct {
  int32_t h0, h1, h2, h3; //!< History
  int32_t w0, w1, w2, w3; //!< Weights
} qoamax_lms_regs;

[[gnu::always_inline]]
static inline qoamax_lms_regs qoamax_lms_load(const qoamax_lms *lms) {
  return (qoamax_lms_regs){
      lms->history[0], lms->history[1], lms->history[2], lms->history[3],
      lms->weights[0], lms->weights[1], lms->weights[2], lms->weights[3],
  };
}

[[gnu::always_inline]]
static inline void qoamax_lms_store(qoamax_lms *lms, const qoamax_lms_regs *r) {
  lms->history[0] = r->h0, lms->history[1] = r->h1, lms->history[2] = r->h2, lms->history[3] = r->h3;
  lms->weights[0] = r->w0, lms->weights[1] = r->w1, lms->weights[2] = r->w2, lms->weights[3] = r->w3;
}

//! Returns the LMS filter's prediction for the next sample.
[[gnu::always_inline]]
static inline int32_t qoamax_lms_predict(const qoamax_lms_regs *r) {
  // Unsigned to keep overflow defined: the weights can grow large (in the encoder's search or in crafted files).
  uint32_t sum = (uint32_t)r->w0 * (uint32_t)r->h0 + (uint32_t)r->w1 * (uint32_t)r->h1 +
                 (uint32_t)r->w2 * (uint32_t)r->h2 + (uint32_t)r->w3 * (uint32_t)r->h3;
  return (int32_t)sum >> 13;
}

//! Updates the LMS filter with the next reconstructed sample and the table entry of its residual.
[[gnu::always_inline]]
static inline void qoamax_lms_update(qoamax_lms_regs *r, const qoamax_decoder_entry *entry, int32_t sample) {
  r->w0 += entry->weight_delta[1 + (r->h0 >> 31)];
  r->w1 += entry->weight_delta[1 + (r->h1 >> 31)];
  r->w2 += entry->weight_delta[1 + (r->h2 >> 31)];
  r->w3 += entry->weight_delta[1 + (r->h3 >> 31)];
  r->h0 = r->h1, r->h1 = r->h2, r->h2 = r->h3, r->h3 = sample;
}

//! Returns the table entry of the slice's i-th residual (&row[(slice >> (57 - 3 * i)) & 7]), shifting the residual
//! directly into a byte offset. Must be called for i = 0, 1, 2, ... in order, as it may consume the slice.
[[gnu::always_inline]]
static inline const qoamax_decoder_entry *qoamax_decoder_entry_at(const qoamax_decoder_entry row[static 8],
                                                                  uint64_t *slice, int i) {
  static_assert(sizeof(qoamax_decoder_entry) == 16);
#ifdef __OPTIMIZE_SIZE__
  // The loop is not unrolled, so constant shifts are cheaper than variable ones (especially on 32-bit targets).
  (void)i;
  uint64_t offset = (*slice >> (57 - 4)) & (7 << 4);
  *slice <<= 3;
#else
  int shift       = 57 - 3 * i - 4;
  uint64_t offset = (shift >= 0 ? *slice >> shift : *slice << -shift) & (7 << 4);
#endif
  return (const qoamax_decoder_entry *)((const char *)row + offset);
}

//! Decodes a slice like @ref qoamax_decode_slice, writing the i-th sample to out[i * stride].
[[gnu::always_inline]]
static inline void qoamax_decode_slice__impl(qoamax_lms *restrict lms, uint64_t slice, int16_t *restrict out,
                                             int stride) {
  const qoamax_decoder_entry *row = qoamax_decoder_table[slice >> 60];
  qoamax_lms_regs regs            = qoamax_lms_load(lms);

  QOAMAX_UNROLL_SLICE
  for (int i = 0; i < 20; i++) {
    const qoamax_decoder_entry *entry = qoamax_decoder_entry_at(row, &slice, i);
    int32_t sample                    = qoamax_clamp_s16(qoamax_lms_predict(&regs) + entry->dequantized);
    qoamax_lms_update(&regs, entry, sample);
    out[i * stride] = sample;
  }

  qoamax_lms_store(lms, &regs);
}

void qoamax_decode_slice(qoamax_lms *restrict lms, uint64_t slice, int16_t out[static restrict 20]) {
  qoamax_decode_slice__impl(lms, slice, out, 1);
}

//! Decodes a slice directly into an interleaved buffer (out[i * stride]), see @ref qoamax_decode_slice.
static void qoamax_decode_slice__interleaved(qoamax_lms *restrict lms, uint64_t slice, int16_t *restrict out,
                                             int stride) {
  qoamax_decode_slice__impl(lms, slice, out, stride);
}

//! The rank-th value of the search order center, center - 1, center + 1, center - 2, center + 2, ...
//! The encoder searches scalefactors outward from the previous slice's one, as it usually is close to the best one.
static inline int qoamax_outward(int center, int rank) { return center + ((rank >> 1) ^ -(rank & 1)); }

//! Reads a big-endian 64-bit word and advances *data past it.
static inline uint64_t qoamax__read_u64be(const char *restrict *restrict data) {
  uint64_t value = 0;
  for (int i = 0; i < 8; i++) value = (value << 8) | (uint8_t)(*data)[i];
  *data += 8;
  return value;
}

//! Writes a big-endian 64-bit word and advances *out past it.
static inline void qoamax__write_u64be(char *restrict *restrict out, uint64_t value) {
  for (int i = 0; i < 8; i++) (*out)[i] = (char)(value >> (56 - 8 * i));
  *out += 8;
}

///////////////////////////////////////////////////////////////////////////////
//                                  Streams                                  //
///////////////////////////////////////////////////////////////////////////////

// The encoder and decoder work on streams: independent sequences of slices, e.g. a channel of the current frame, or
// (as every frame header stores the LMS state) a channel of one of several frames, or the streams of the public
// qoamax_encode_streams / qoamax_decode_streams. Independent streams are encoded or decoded in parallel
// (qoamax__encode_streams, qoamax__decode_streams), which only the file format code and the public API set up.

//! An independent sequence of slices and their samples.
typedef struct {
  qoamax_encoder_channel state; //!< The LMS state (the decoder uses only this), the error and the last scalefactor
  char *slices;                 //!< The next slice (the decoder only reads it)
  int slice_step;               //!< Bytes from one slice to the next
  bool big_endian;              //!< Whether the slices are big-endian words (the file format) or uint64_t values
  int sample_stride;            //!< Sample i is samples[i * sample_stride]
  int16_t *samples;             //!< The next sample (the encoder only reads it)
  size_t samples_size;          //!< Samples from `samples` on: the end of the stream, and the encoder's lookahead
  size_t slice_count;           //!< The number of slices left to encode or decode
} qoamax__stream;

//! Returns the stream's next slice.
static inline uint64_t qoamax__stream_get(const qoamax__stream *stream) {
  const char *slice = stream->slices;
  if (stream->big_endian) return qoamax__read_u64be(&slice);
  uint64_t value;
  memcpy(&value, slice, sizeof value);
  return value;
}

//! Advances the stream past a slice of samples_count samples.
static inline void qoamax__stream_next(qoamax__stream *stream, int samples_count) {
  stream->slices += stream->slice_step;
  stream->samples += samples_count * stream->sample_stride;
  stream->samples_size -= samples_count;
  stream->slice_count -= 1;
}

//! Stores the stream's next slice, which encodes samples_count samples, and advances the stream past it.
static inline void qoamax__stream_put(qoamax__stream *stream, uint64_t slice, int samples_count) {
  char *out = stream->slices;
  if (stream->big_endian)
    qoamax__write_u64be(&out, slice);
  else
    memcpy(out, &slice, sizeof slice);
  qoamax__stream_next(stream, samples_count);
}

//! Returns the number of full slices (20 samples) the stream has left to encode or decode.
static inline size_t qoamax__stream_full_slices(const qoamax__stream *stream) {
  return QOAMAX_MIN(stream->slice_count, stream->samples_size / 20);
}

///////////////////////////////////////////////////////////////////////////////
//                        Hillclimb Encoding Algorithm                       //
///////////////////////////////////////////////////////////////////////////////

//! One sample of @ref qoamax_encode_hillclimb_search_impl. Returns false as soon as the error reaches error_hint.
[[gnu::always_inline]]
static inline bool qoamax_hillclimb_step(qoamax_lms_regs *regs, uint64_t *slice, uint64_t *error, int32_t sample,
                                         const qoamax_decoder_entry *row, const int16_t *qt, uint64_t error_hint,
                                         bool unclamped) {
  int32_t predicted    = qoamax_lms_predict(regs);
  int32_t residual     = sample - predicted;
  int32_t residual_abs = residual < 0 ? -residual : residual;

  // This line computes the best codepoint unless the signal is clipping.
  // However, even if the signal is clipping, choosing the non-optimal codepoint may be beneficial.
  int quant = (((residual_abs > qt[0]) + (residual_abs > qt[1]) + (residual_abs > qt[2])) << 1) + (residual <= 0);
  *slice    = (*slice << 3) | quant;

  const qoamax_decoder_entry *entry = &row[quant];
  int32_t reconstructed             = predicted + entry->dequantized;
  int32_t clamped                   = qoamax_clamp_s16(reconstructed);
  if (unclamped)
    *error = qoamax_add_saturating(*error, qoamax_error_squared_unclamped(sample, reconstructed));
  else
    *error += qoamax_error_squared(sample, clamped);
  if (*error >= error_hint) return false;

  qoamax_lms_update(regs, entry, clamped);
  return true;
}

//! Performs a hill-climb search (equivalent to beam search with width=1).
//!
//! The hill-climb search encodes the audio samples with the given scalefactor by choosing the
//! closest codepoint at each point, never reconsidering past choices.
//!
//! The normal variant measures the error of the decoded samples; it is the encoder of the hill-climb efforts.
//! The unclamped variant measures the error before clamping the reconstructed samples to int16 (like the beam
//! search), and provides the error bound that lets the beam search skip most bad options.
//!
//! @returns the cumulative squared error, or -1 as soon as it reaches error_hint (lms is left unchanged then).
[[gnu::always_inline]]
static inline uint64_t qoamax_encode_hillclimb_search_impl(qoamax_lms lms[static restrict 1], int samples_count,
                                                           const int16_t samples[static restrict samples_count],
                                                           int scalefactor, uint64_t error_hint,
                                                           uint64_t slice_out[static restrict 1], bool unclamped) {
  qoamax_assume(samples_count >= 1 && samples_count <= 20);
  qoamax_assume(scalefactor >= 0 && scalefactor <= 15);

  const qoamax_decoder_entry *row = qoamax_decoder_table[scalefactor];
  const int16_t *qt               = qoamax_qt[scalefactor];
  qoamax_lms_regs regs            = qoamax_lms_load(lms);

  uint64_t slice = scalefactor;
  uint64_t error = 0;

  // Only full slices are unrolled. Partial slices (the end of the stream) get a compact loop.
  if (samples_count == 20) {
    QOAMAX_UNROLL_SLICE
    for (int i = 0; i < 20; i++)
      if (!qoamax_hillclimb_step(&regs, &slice, &error, samples[i], row, qt, error_hint, unclamped)) return -1;
  } else {
    QOAMAX_PRAGMA(GCC unroll 1)
    for (int i = 0; i < samples_count; i++)
      if (!qoamax_hillclimb_step(&regs, &slice, &error, samples[i], row, qt, error_hint, unclamped)) return -1;
  }

  slice <<= (20 - samples_count) * 3;
  *slice_out = slice;
  qoamax_lms_store(lms, &regs);
  return error;
}

//! See @ref qoamax_encode_hillclimb_search_impl. The common 20-sample case gets its own copy, which the compiler can
//! fully unroll without a loop exit check after every sample. Partial slices use a compact loop.
static uint64_t qoamax_encode_hillclimb_search(qoamax_lms lms[static restrict 1], int samples_count,
                                               const int16_t samples[static restrict samples_count], int scalefactor,
                                               uint64_t error_hint, uint64_t slice_out[static restrict 1]) {
  if (samples_count == 20)
    return qoamax_encode_hillclimb_search_impl(lms, 20, samples, scalefactor, error_hint, slice_out, false);
  qoamax_assume(samples_count < 20); // so that the unrolled loop isn't emitted again
  return qoamax_encode_hillclimb_search_impl(lms, samples_count, samples, scalefactor, error_hint, slice_out, false);
}

//! See @ref qoamax_encode_hillclimb_search_impl. The common 20-sample case gets its own copy.
static uint64_t qoamax_encode_hillclimb_search_unclamped(qoamax_lms lms[static restrict 1], int samples_count,
                                                         const int16_t samples[static restrict samples_count],
                                                         int scalefactor, uint64_t error_hint,
                                                         uint64_t slice_out[static restrict 1]) {
  if (samples_count == 20)
    return qoamax_encode_hillclimb_search_impl(lms, 20, samples, scalefactor, error_hint, slice_out, true);
  qoamax_assume(samples_count < 20);
  return qoamax_encode_hillclimb_search_impl(lms, samples_count, samples, scalefactor, error_hint, slice_out, true);
}

///////////////////////////////////////////////////////////////////////////////
//                         Ultrafast Encoding Algorithm                      //
///////////////////////////////////////////////////////////////////////////////

// ULTRAFAST doesn't search the scalefactor. It estimates it from the largest residual of an open-loop prediction (from
// the input samples instead of the reconstructed ones, so without a dependency chain), and runs the hill-climb for the
// estimate and the next larger scalefactor. The SIMD version encodes several streams (channels, possibly of different
// frames) at once, one lane per stream and scalefactor.
//
// The hill-climb is a chain of dependent operations (about 17 vector operations per sample), so one slice per call
// would leave the lanes mostly idle: SUPERFAST runs 16 scalefactors in barely more time than one. To have enough
// independent work, the stream encoder encodes up to QOAMAX_ULTRAFAST_GROUP_FRAMES frames at once (see
// qoamax_encoder_write__impl). The frames of a group start from the four input samples before them and the LMS
// weights at the end of the previous group, instead of the state where the previous frame ended (every frame header
// holds the state the decoder starts from). The weights change slowly with the signal's spectrum, and this costs about
// 0.05 dB (resetting the weights to their initial values would cost 0.7 dB). The first group's frames are encoded one
// after another.
//
// Measured (2026-11, Ryzen 5900X, clang 23, AVX2 by dispatch, 9 music and sound effect samples): SUPERFAST 48.94 dB at
// 85 M samples/s, ULTRAFAST 48.30 dB at 316 M samples/s (the decoder: 380). The old ULTRAFAST (the hill-climb with the
// previous scalefactor -1 and +1) had 40.54 dB at 112 M samples/s. With 1 or 4 candidates instead of 2, ULTRAFAST
// would be about 1.6 dB worse at 1.6x the speed, or 0.6 dB better at 0.6x.

//! Frames per group of ULTRAFAST (the output depends on it). 4 fill AVX2 vectors (4 streams) for mono.
#define QOAMAX_ULTRAFAST_GROUP_FRAMES 4

//! The estimated scalefactor is the number of thresholds the largest open-loop residual exceeds. Threshold s - 1 is
//! 7.5 times the geometric mean of the scales of scalefactors s - 1 and s (the dequantization steps are 0.75, 2.5, 4.5
//! and 7 times the scale (s + 1)^2.75): floor(7.5 * sqrt(s^2.75 * (s + 1)^2.75)). There is no threshold for 15, as the
//! next larger scalefactor is tried as well. The factor was tuned on music samples.
static const int32_t qoamax_ultrafast_thresholds[14] = {19,   88,   228,  461,  805,  1279, 1900,
                                                        2684, 3648, 4808, 6178, 7773, 9608, 11698};

///////////////////////////////////////////////////////////////////////////////
//                      Hillclimb Scalefactor Search                         //
///////////////////////////////////////////////////////////////////////////////

//! Searches the scalefactor with the lowest hill-climb error over the lookahead (recursively choosing the best
//! scalefactors of the following slices), considering only results better than error_hint.
//! Ties are resolved in favor of the scalefactor closest to first_scalefactor.
//!
//! @returns the error over the whole lookahead, or error_hint if no result is better. The first slice, its error and
//!          the LMS state after it are stored in out_slice, out_error and lms.
//!
//! Comes in a normal and an unclamped variant like @ref qoamax_encode_hillclimb_search_impl. Both are thin wrappers
//! around one always_inline implementation, so the variant is known at compile time.
static uint64_t
qoamax_encode_slice__hillclimb_scalefactor_search_scalar(qoamax_lms *lms, int first_scalefactor, int lookahead,
                                                         const int16_t samples[static restrict lookahead],
                                                         uint64_t error_hint, uint64_t *out_slice, uint64_t *out_error);
static uint64_t qoamax_encode_slice__hillclimb_scalefactor_search_scalar_unclamped(
    qoamax_lms *lms, int first_scalefactor, int lookahead, const int16_t samples[static restrict lookahead],
    uint64_t error_hint, uint64_t *out_slice, uint64_t *out_error);

[[gnu::always_inline]]
static inline uint64_t qoamax_encode_slice__hillclimb_scalefactor_search_scalar_impl(
    qoamax_lms *lms, int first_scalefactor, int lookahead, const int16_t samples[static restrict lookahead],
    uint64_t error_hint, uint64_t *out_slice, uint64_t *out_error, bool unclamped) {
  int samples_count   = QOAMAX_MIN(20, lookahead);
  uint64_t best_slice = 0;
  uint64_t best_error = 0;
  qoamax_lms best_lms = *lms;

  for (int rank = 0; rank < 31; rank++) {
    int scalefactor = qoamax_outward(first_scalefactor, rank);
    if ((unsigned)scalefactor >= 16) continue;

    qoamax_lms local_lms = *lms;
    uint64_t local_slice;
    uint64_t local_error;
    if (unclamped)
      local_error = qoamax_encode_hillclimb_search_unclamped(&local_lms, samples_count, samples, scalefactor,
                                                             error_hint, &local_slice);
    else
      local_error =
          qoamax_encode_hillclimb_search(&local_lms, samples_count, samples, scalefactor, error_hint, &local_slice);
    if (local_error >= error_hint) continue;

    uint64_t total_error = local_error;
    if (lookahead > 20) {
      // Only the error of the following slices matters here, their slices and LMS states are discarded.
      qoamax_lms next_lms = local_lms;
      uint64_t next_slice, next_first_error;
      if (unclamped)
        total_error += qoamax_encode_slice__hillclimb_scalefactor_search_scalar_unclamped(
            &next_lms, scalefactor, lookahead - 20, &samples[20], error_hint - local_error, &next_slice,
            &next_first_error);
      else
        total_error += qoamax_encode_slice__hillclimb_scalefactor_search_scalar(&next_lms, scalefactor, lookahead - 20,
                                                                                &samples[20], error_hint - local_error,
                                                                                &next_slice, &next_first_error);
    }

    if (total_error < error_hint) {
      error_hint = total_error;
      best_slice = local_slice;
      best_error = local_error;
      best_lms   = local_lms;
    }
  }

  *out_slice = best_slice;
  *out_error = best_error;
  *lms       = best_lms;
  return error_hint;
}

//! See @ref qoamax_encode_slice__hillclimb_scalefactor_search_scalar_impl.
static uint64_t qoamax_encode_slice__hillclimb_scalefactor_search_scalar(
    qoamax_lms *lms, int first_scalefactor, int lookahead, const int16_t samples[static restrict lookahead],
    uint64_t error_hint, uint64_t *out_slice, uint64_t *out_error) {
  return qoamax_encode_slice__hillclimb_scalefactor_search_scalar_impl(lms, first_scalefactor, lookahead, samples,
                                                                       error_hint, out_slice, out_error, false);
}

//! See @ref qoamax_encode_slice__hillclimb_scalefactor_search_scalar_impl.
static uint64_t qoamax_encode_slice__hillclimb_scalefactor_search_scalar_unclamped(
    qoamax_lms *lms, int first_scalefactor, int lookahead, const int16_t samples[static restrict lookahead],
    uint64_t error_hint, uint64_t *out_slice, uint64_t *out_error) {
  return qoamax_encode_slice__hillclimb_scalefactor_search_scalar_impl(lms, first_scalefactor, lookahead, samples,
                                                                       error_hint, out_slice, out_error, true);
}

// The SIMD search runs the hill-climb for several scalefactors at once, one lane per scalefactor (at most 16). It is
// compiled for the vector width of the compile target (QOAMAX_LANES, 1 = no SIMD, use the scalar search), and on x86
// and RISC-V also for wider vector units, one variant per lane count, which is chosen at runtime by the CPU's features
// (function multi-versioning, see qoamax__lanes_supported). Defining QOAMAX_LANES or QOAMAX_DISPATCH=0 builds only
// one variant.
//
// SIMD and dispatch use GNU C extensions (vector types, __builtin_cpu_supports, target attributes), which gcc and
// clang (and compilers based on them) indicate by defining __GNUC__. Other C23 compilers get the scalar code only.
#ifndef __GNUC__
#if defined(QOAMAX_LANES) && QOAMAX_LANES > 1
#error "QOAMAX_LANES > 1 needs GNU C vector extensions"
#endif
#undef QOAMAX_DISPATCH
#define QOAMAX_DISPATCH 0
#ifndef QOAMAX_LANES
#define QOAMAX_LANES 1
#endif
#endif

#ifndef QOAMAX_DISPATCH
#ifdef QOAMAX_LANES
#define QOAMAX_DISPATCH 0
#else
#define QOAMAX_DISPATCH 1
#endif
#endif

#ifndef QOAMAX_LANES
#if defined(__AVX512F__)
#define QOAMAX_LANES 16
#elif defined(__AVX2__)
#define QOAMAX_LANES 8
#elif defined(__ARM_FEATURE_SVE_BITS) && __ARM_FEATURE_SVE_BITS >= 128 // -msve-vector-bits=...
#define QOAMAX_LANES (__ARM_FEATURE_SVE_BITS >= 512 ? 16 : __ARM_FEATURE_SVE_BITS / 32)
#elif defined(__riscv_v_fixed_vlen) // -mrvv-vector-bits=zvl
#define QOAMAX_LANES (__riscv_v_fixed_vlen >= 512 ? 16 : __riscv_v_fixed_vlen / 32)
#elif defined(__riscv_v_min_vlen) && __riscv_v_min_vlen >= 64
#define QOAMAX_LANES (__riscv_v_min_vlen >= 512 ? 16 : __riscv_v_min_vlen / 32)
#elif defined(__SSE2__) || defined(__ARM_NEON) || defined(__ARM_FEATURE_MVE)
#define QOAMAX_LANES 4
#else
#define QOAMAX_LANES 1
#endif
#endif
static_assert(QOAMAX_LANES >= 1 && QOAMAX_LANES <= 16 && 16 % QOAMAX_LANES == 0);

// The target attributes of the variants wider than QOAMAX_LANES (QOAMAX_TARGET_X<lanes>), and a function that returns
// the widest lane count the CPU supports. AArch64 has none: NEON is always there, and the compilers only use SVE for
// fixed-size vectors with -msve-vector-bits, which can't be set per function.
#if QOAMAX_DISPATCH && (defined(__x86_64__) || defined(__i386__))
#if QOAMAX_LANES < 8
#define QOAMAX_TARGET_X8 "avx2"
#define QOAMAX_TARGET_BEAM "avx2" // the beam search's variant, chosen if qoamax__lanes_supported() >= 8
#endif
#if QOAMAX_LANES < 16
#define QOAMAX_TARGET_X16 "avx512f"
#endif

static int qoamax__lanes_supported(void) {
  if (__builtin_cpu_supports("avx512f")) return 16;
  if (__builtin_cpu_supports("avx2")) return 8;
  return 4;
}

// RISC-V: only with clang (19 for __builtin_cpu_supports, which needs __riscv_feature_bits from compiler-rt or libgcc
// 15), not tested with gcc. If the compile target already has the V extension, clang (23) crashes on the zvl256b and
// zvl512b variants, so only the target's default variant is built then (use -mrvv-vector-bits=zvl for wider ones).
#elif QOAMAX_DISPATCH && defined(__riscv) && defined(__linux__) && __clang_major__ >= 19 && !defined(__riscv_vector)
#define QOAMAX_TARGET_X4 "arch=+v"
#define QOAMAX_TARGET_X8 "arch=+v,+zvl256b"
#define QOAMAX_TARGET_X16 "arch=+v,+zvl512b"

static int qoamax__lanes_supported(void) {
  if (!__builtin_cpu_supports("v")) return 1;
  unsigned long vlenb; // the vector register size in bytes (reading it without the V extension would trap)
  __asm__("csrr %0, 0xc22" : "=r"(vlenb));
  return vlenb >= 64 ? 16 : vlenb >= 32 ? 8 : 4;
}
#endif

#if defined(QOAMAX_TARGET_X4) || defined(QOAMAX_TARGET_X8) || defined(QOAMAX_TARGET_X16)
#define QOAMAX_HAS_VARIANTS
#endif

//! The functions of a SIMD variant (qoamax_lanes.h defines one table per variant, qoamax__simd_get chooses one).
typedef struct {
  int lanes;
  //! See @ref qoamax_encode_slice__hillclimb_scalefactor_search_scalar_impl (normal and unclamped variant).
  uint64_t (*search)(qoamax_lms *lms, int first_scalefactor, int lookahead, const int16_t *samples, uint64_t error_hint,
                     uint64_t *out_slice, uint64_t *out_error);
  uint64_t (*search_unclamped)(qoamax_lms *lms, int first_scalefactor, int lookahead, const int16_t *samples,
                               uint64_t error_hint, uint64_t *out_slice, uint64_t *out_error);
  //! ULTRAFAST for full slices of up to lanes / 2 streams, see @ref qoamax_encode_ultrafast_scalar.
  void (*encode_ultrafast)(qoamax__stream *streams, int count, int slice_count);
  //! Decodes full slices of up to `lanes` streams.
  void (*decode)(qoamax__stream *streams, int count, int slice_count);
} qoamax__simd;

// Pastes QOAMAX_VARIANT_SUFFIX to a name defined in qoamax_lanes.h or qoamax_beam.h.
#define QOAMAX_VARIANT_NAME(name) QOAMAX_VARIANT_NAME_(name, QOAMAX_VARIANT_SUFFIX)
#define QOAMAX_VARIANT_NAME_(name, suffix) QOAMAX_VARIANT_NAME__(name, suffix)
#define QOAMAX_VARIANT_NAME__(name, suffix) name##suffix

#if QOAMAX_LANES > 1 || defined(QOAMAX_HAS_VARIANTS)

//! qoamax_dqt (positive values) and qoamax_qt transposed, so that a vector can hold one scalefactor per lane.
//! The 16 columns are repeated, so that a lane window wrapping around from scalefactor 15 to 0 is one load.
#define QOAMAX_COLUMN_A(a, b, c, d) a,
#define QOAMAX_COLUMN_B(a, b, c, d) (b) - (a),
#define QOAMAX_COLUMN_C(a, b, c, d) (c) - (b),
#define QOAMAX_COLUMN_D(a, b, c, d) (d) - (c),
#define QOAMAX_COLUMN_AB(a, b, c, d) ((a) + (b)) / 2,
#define QOAMAX_COLUMN_BC(a, b, c, d) ((b) + (c)) / 2,
#define QOAMAX_COLUMN_CD(a, b, c, d) ((c) + (d)) / 2,
static const int32_t qoamax_lanes_dqt[4][32] = {
    {QOAMAX_DQT_ROWS(QOAMAX_COLUMN_A) QOAMAX_DQT_ROWS(QOAMAX_COLUMN_A)},
    {QOAMAX_DQT_ROWS(QOAMAX_COLUMN_B) QOAMAX_DQT_ROWS(QOAMAX_COLUMN_B)},
    {QOAMAX_DQT_ROWS(QOAMAX_COLUMN_C) QOAMAX_DQT_ROWS(QOAMAX_COLUMN_C)},
    {QOAMAX_DQT_ROWS(QOAMAX_COLUMN_D) QOAMAX_DQT_ROWS(QOAMAX_COLUMN_D)},
};
static const int32_t qoamax_lanes_qt[3][32] = {
    {QOAMAX_DQT_ROWS(QOAMAX_COLUMN_AB) QOAMAX_DQT_ROWS(QOAMAX_COLUMN_AB)},
    {QOAMAX_DQT_ROWS(QOAMAX_COLUMN_BC) QOAMAX_DQT_ROWS(QOAMAX_COLUMN_BC)},
    {QOAMAX_DQT_ROWS(QOAMAX_COLUMN_CD) QOAMAX_DQT_ROWS(QOAMAX_COLUMN_CD)},
};

//! Inverse of @ref qoamax_outward: the position of value in the search order around center.
static inline int qoamax_outward_rank(int value, int center) {
  int distance = value - center;
  return distance < 0 ? -2 * distance - 1 : 2 * distance;
}

//! Hill-climb results of all 16 scalefactors after encoding the same samples. They are indexed by position
//! (scalefactor - base) & 15 in the search's rotated order, so that every lane window is a contiguous block.
typedef struct {
  int32_t history[4][16];
  int32_t weights[4][16];
  int32_t quant[20][16];
  uint64_t errors[16];
} qoamax_scalefactor_results;

//! Returns the slice of the result at position pos (whose scalefactor is scalefactor), and sets lms to its LMS state.
static inline uint64_t qoamax_scalefactor_results_get(const qoamax_scalefactor_results *results, int pos,
                                                      int scalefactor, int samples_count, qoamax_lms *lms) {
  uint64_t slice = scalefactor;
  for (int i = 0; i < samples_count; i++) slice = (slice << 3) | results->quant[i][pos];
  for (int k = 0; k < 4; k++) lms->history[k] = results->history[k][pos], lms->weights[k] = results->weights[k][pos];
  return slice << ((20 - samples_count) * 3);
}

#if QOAMAX_LANES > 1
#define QOAMAX_VARIANT_LANES QOAMAX_LANES
#define QOAMAX_VARIANT_SUFFIX
#define QOAMAX_VARIANT_TARGET
#include "./qoamax_lanes.h"
#endif

#ifdef QOAMAX_TARGET_X4
#define QOAMAX_VARIANT_LANES 4
#define QOAMAX_VARIANT_SUFFIX __x4
#define QOAMAX_VARIANT_TARGET [[gnu::target(QOAMAX_TARGET_X4)]]
#include "./qoamax_lanes.h"
#endif

#ifdef QOAMAX_TARGET_X8
#define QOAMAX_VARIANT_LANES 8
#define QOAMAX_VARIANT_SUFFIX __x8
#define QOAMAX_VARIANT_TARGET [[gnu::target(QOAMAX_TARGET_X8)]]
#include "./qoamax_lanes.h"
#endif

#ifdef QOAMAX_TARGET_X16
#define QOAMAX_VARIANT_LANES 16
#define QOAMAX_VARIANT_SUFFIX __x16
#define QOAMAX_VARIANT_TARGET [[gnu::target(QOAMAX_TARGET_X16)]]
#include "./qoamax_lanes.h"
#endif

#endif

//! Returns the functions of the widest SIMD variant the CPU supports, or null without SIMD.
static const qoamax__simd *qoamax__simd_get(void) {
#ifdef QOAMAX_HAS_VARIANTS
  switch (qoamax__lanes_supported()) {
#ifdef QOAMAX_TARGET_X16
  case 16:
    return &qoamax_simd__x16;
#endif
#ifdef QOAMAX_TARGET_X8
  case 8:
    return &qoamax_simd__x8;
#endif
#ifdef QOAMAX_TARGET_X4
  case 4:
    return &qoamax_simd__x4;
#endif
  }
#endif
#if QOAMAX_LANES > 1
  return &qoamax_simd;
#else
  return NULL; // not nullptr: cproc (as of 2026-11) crashes on returning nullptr as a pointer
#endif
}

//! The scalefactor search of SUPERFAST/VERYFAST, using the widest SIMD variant the CPU supports (or the scalar search).
//! See @ref qoamax_encode_slice__hillclimb_scalefactor_search_scalar_impl.
// The dispatchers aren't inlined: inlined into qoamax_encode_slice, they made ULTRAFAST execute 1.5% more instructions.
[[gnu::noinline]]
static uint64_t qoamax_encode_slice__hillclimb_scalefactor_search(qoamax_lms *lms, int first_scalefactor, int lookahead,
                                                                  const int16_t samples[static restrict lookahead],
                                                                  uint64_t error_hint, uint64_t *out_slice,
                                                                  uint64_t *out_error) {
  const qoamax__simd *simd = qoamax__simd_get();
  if (simd) return simd->search(lms, first_scalefactor, lookahead, samples, error_hint, out_slice, out_error);
  return qoamax_encode_slice__hillclimb_scalefactor_search_scalar(lms, first_scalefactor, lookahead, samples,
                                                                  error_hint, out_slice, out_error);
}

//! See @ref qoamax_encode_slice__hillclimb_scalefactor_search (unclamped variant, the beam search's error bound).
[[gnu::noinline]]
static uint64_t qoamax_encode_slice__hillclimb_scalefactor_search_unclamped(
    qoamax_lms *lms, int first_scalefactor, int lookahead, const int16_t samples[static restrict lookahead],
    uint64_t error_hint, uint64_t *out_slice, uint64_t *out_error) {
  const qoamax__simd *simd = qoamax__simd_get();
  if (simd) return simd->search_unclamped(lms, first_scalefactor, lookahead, samples, error_hint, out_slice, out_error);
  return qoamax_encode_slice__hillclimb_scalefactor_search_scalar_unclamped(lms, first_scalefactor, lookahead, samples,
                                                                            error_hint, out_slice, out_error);
}

///////////////////////////////////////////////////////////////////////////////
//                           Beam Encoding Algorithm                         //
///////////////////////////////////////////////////////////////////////////////

//! A node of the beam search: one candidate encoding of the samples so far.
typedef struct {
  uint64_t error; //!< Cumulative (unclamped) squared error
  uint64_t slice; //!< The first slice of this node's path; only this slice is used as the result
  qoamax_lms lms;
} qoamax_beam_node;

typedef struct {
  uint64_t error;
  uint32_t slot;
} qoamax_beam_heap_entry;

//! The candidates of one beam search step, stored by slot: arrays with an entry per slot (at least the beam width).
//! They point to local arrays of the caller; once inlined, the compiler addresses those directly.
typedef struct {
  qoamax_beam_heap_entry *heap; //!< Max-heap of the used slots, ordered by error
  uint64_t *error;
  uint16_t *parent; //!< Index of the source node
  uint8_t *codepoint;
} qoamax_beam_slots;

//! Adds a candidate to the beam, replacing the worst one if all `width` slots are already used.
[[gnu::always_inline]]
static inline void qoamax_beam_insert(qoamax_beam_slots slots, int *used, int width, uint64_t error, int parent,
                                      int codepoint) {
  qoamax_beam_heap_entry *heap = slots.heap;
  int slot, heap_pos;
  if (*used < width) {
    slot = heap_pos = (*used)++;
    while (heap_pos > 0) {
      int parent_pos = (heap_pos - 1) >> 1;
      if (heap[parent_pos].error >= error) break;
      heap[heap_pos] = heap[parent_pos];
      heap_pos       = parent_pos;
    }
  } else {
    slot = heap[0].slot, heap_pos = 0;
    while (true) {
      int left = 2 * heap_pos + 1, right = 2 * heap_pos + 2;
      int largest            = heap_pos;
      uint64_t largest_error = error;
      if (left < width && heap[left].error > largest_error) largest = left, largest_error = heap[left].error;
      if (right < width && heap[right].error > largest_error) largest = right, largest_error = heap[right].error;
      if (largest == heap_pos) break;
      heap[heap_pos] = heap[largest];
      heap_pos       = largest;
    }
  }
  heap[heap_pos].error = error, heap[heap_pos].slot = slot;

  slots.error[slot]     = error;
  slots.parent[slot]    = parent;
  slots.codepoint[slot] = codepoint;
}

//! The errors of the candidates of a beam search step for small beams (up to 4), by slot. Once inlined, they are kept
//! in registers. Free slots hold the bound (`error_hint`), so they are taken first and in order, and the used slots
//! always come first.
typedef struct {
  uint64_t error[4];
  uint64_t max_error; //!< The highest error of all slots: candidates need a lower error to enter the beam
  int worst;          //!< The first slot holding max_error, which the next candidate replaces
} qoamax_beam_small;

//! Adds a candidate (with an error below max_error) to a small beam, replacing the worst one.
//! @param candidates By slot: index of the source node << 3 | codepoint. (A separate array, because it is indexed by
//!                   slot, which would keep the errors in memory.)
[[gnu::always_inline]]
static inline void qoamax_beam_small_insert(qoamax_beam_small *beam, uint16_t *candidates, int width, uint64_t error,
                                            int parent, int codepoint) {
  // Conditional moves instead of indexing error[worst], so that the errors stay in registers.
  for (int k = 0; k < width; k++) beam->error[k] = k == beam->worst ? error : beam->error[k];
  candidates[beam->worst] = (uint16_t)(parent << 3 | codepoint);

  beam->max_error = beam->error[0], beam->worst = 0;
  for (int k = 1; k < width; k++)
    if (beam->error[k] > beam->max_error) beam->max_error = beam->error[k], beam->worst = k;
}

//! See @ref qoamax_encode_beam_search_impl, for beams of width 1 to 4. Instead of a heap, the candidates are kept by
//! slot (see @ref qoamax_beam_small). That is faster (FAST 8-13%, MEDIUM and SLOW 13-17%), but keeps a different one
//! of several candidates with the same error, so its results differ slightly (not worse on average).
[[gnu::always_inline]]
static inline int qoamax_encode_beam_search_small(const qoamax_beam_node *restrict src, int src_count,
                                                  qoamax_beam_node *restrict buffers, int width, bool first_slice,
                                                  int samples_size, const int16_t samples[static restrict samples_size],
                                                  int scalefactor, uint64_t error_hint,
                                                  qoamax_beam_node *restrict *restrict out, int32_t *restrict predicted,
                                                  bool branchless) {
  qoamax_assume(width > 0 && width <= 4);

  const int16_t *restrict dqt     = qoamax_dqt[scalefactor];
  const qoamax_decoder_entry *row = qoamax_decoder_table[scalefactor];
  qoamax_beam_small beam;
  uint16_t slot_candidates[4];

  for (int i = 0; i < samples_size; i++) {
    int32_t actual_sample = samples[i];
    for (int k = 0; k < width; k++) beam.error[k] = error_hint;
    beam.max_error = error_hint, beam.worst = 0;

    for (int j = 0; j < src_count; j++) {
      uint64_t current_error = src[j].error;
      qoamax_lms_regs regs   = qoamax_lms_load(&src[j].lms);
      int32_t prediction     = qoamax_lms_predict(&regs);
      predicted[j]           = prediction;

      if (width <= 2 && !branchless) {
        // With tiny beams, most candidates pass the bound, so checking them all upfront doesn't pay off.
        for (int codepoint = 0; codepoint < 8; codepoint++) {
          uint64_t error     = qoamax_error_squared_unclamped(actual_sample, prediction + dqt[codepoint]);
          uint64_t new_error = qoamax_add_saturating(current_error, error);
          if (new_error < beam.max_error)
            qoamax_beam_small_insert(&beam, slot_candidates, width, new_error, j, codepoint);
        }
      } else {
        // Compute all candidates without branches, then insert the ones below the bound (in codepoint order).
        uint64_t new_errors[8];
        unsigned candidates = 0;
        for (int codepoint = 0; codepoint < 8; codepoint++) {
          uint64_t error        = qoamax_error_squared_unclamped(actual_sample, prediction + dqt[codepoint]);
          new_errors[codepoint] = qoamax_add_saturating(current_error, error);
          candidates |= (unsigned)(new_errors[codepoint] < beam.max_error) << codepoint;
        }

        for (; candidates; candidates &= candidates - 1) {
          int codepoint      = qoamax__trailing_zeros(candidates);
          uint64_t new_error = new_errors[codepoint];
          // the bound may have decreased in the meantime
          if (new_error < beam.max_error)
            qoamax_beam_small_insert(&beam, slot_candidates, width, new_error, j, codepoint);
        }
      }
    }

    // Construct the surviving nodes (the used slots)
    qoamax_beam_node *dst = &buffers[(i & 1) * width];
    int used              = 0;
    for (int slot = 0; slot < width && beam.error[slot] < error_hint; slot++, used++) {
      int parent_index                  = slot_candidates[slot] >> 3;
      int codepoint                     = slot_candidates[slot] & 7;
      const qoamax_beam_node *parent    = &src[parent_index];
      const qoamax_decoder_entry *entry = &row[codepoint];

      qoamax_lms_regs regs = qoamax_lms_load(&parent->lms);
      qoamax_lms_update(&regs, entry, qoamax_clamp_s16(predicted[parent_index] + entry->dequantized));
      qoamax_lms_store(&dst[slot].lms, &regs);
      dst[slot].error = beam.error[slot];

      if (!first_slice)
        dst[slot].slice = parent->slice;
      else if (i == 0)
        dst[slot].slice = ((uint64_t)scalefactor << 3) | codepoint;
      else
        dst[slot].slice = (parent->slice << 3) | codepoint;
    }
    if (used == 0) return 0;

    src       = dst;
    src_count = used;
  }

  *out = (qoamax_beam_node *)src;
  return src_count;
}

//! Runs a beam search with the given scalefactor over the samples, starting from the `src` nodes (not modified).
//! The error is computed before clamping the reconstructed sample to 16 bits.
//!
//! Every step keeps the `width` best candidates (8 codepoints per node). Candidates are first collected in a max-heap
//! of slots (beams up to width 4: see @ref qoamax_encode_beam_search_small), and the surviving nodes are only
//! constructed once the step is complete.
//!
//! @param buffers   Room for 2 * width nodes, alternately used by the steps.
//! @param slots     Room for `width` candidates.
//! @param predicted Room for `width` predictions.
//! @param branchless Whether beams of width 1 and 2 also compute all candidates first. That only pays off where the
//!                   compiler vectorizes it well (64-bit multiplies and comparisons: AVX2).
//! @returns the number of resulting nodes, which are stored in buffers (*out).
[[gnu::always_inline]]
static inline int qoamax_encode_beam_search_impl(const qoamax_beam_node *restrict src, int src_count,
                                                 qoamax_beam_node *restrict buffers, int width, bool first_slice,
                                                 int samples_size, const int16_t samples[static restrict samples_size],
                                                 int scalefactor, uint64_t error_hint,
                                                 qoamax_beam_node *restrict *restrict out, qoamax_beam_slots slots,
                                                 int32_t *restrict predicted, bool branchless) {
  qoamax_assume(samples_size >= 1 && samples_size <= 20);
  qoamax_assume(width > 0 && width <= QOAMAX_MAX_BEAM_WIDTH);
  qoamax_assume(scalefactor >= 0 && scalefactor <= 15);

  if (width <= 4)
    return qoamax_encode_beam_search_small(src, src_count, buffers, width, first_slice, samples_size, samples,
                                           scalefactor, error_hint, out, predicted, branchless);

  const int16_t *restrict dqt     = qoamax_dqt[scalefactor];
  const qoamax_decoder_entry *row = qoamax_decoder_table[scalefactor];

  for (int i = 0; i < samples_size; i++) {
    int32_t actual_sample = samples[i];
    int used              = 0;
    uint64_t max_error    = error_hint; // Candidates need a lower error to enter the beam

    for (int j = 0; j < src_count; j++) {
      uint64_t current_error = src[j].error;
      qoamax_lms_regs regs   = qoamax_lms_load(&src[j].lms);
      int32_t prediction     = qoamax_lms_predict(&regs);
      predicted[j]           = prediction;

      // Compute all candidates without branches, then insert the ones below the bound (in codepoint order).
      uint64_t new_errors[8];
      unsigned candidates = 0;
      for (int codepoint = 0; codepoint < 8; codepoint++) {
        uint64_t error        = qoamax_error_squared_unclamped(actual_sample, prediction + dqt[codepoint]);
        new_errors[codepoint] = qoamax_add_saturating(current_error, error);
        candidates |= (unsigned)(new_errors[codepoint] < max_error) << codepoint;
      }

      for (; candidates; candidates &= candidates - 1) {
        int codepoint      = qoamax__trailing_zeros(candidates);
        uint64_t new_error = new_errors[codepoint];
        if (new_error >= max_error) continue; // the bound may have decreased in the meantime

        qoamax_beam_insert(slots, &used, width, new_error, j, codepoint);
        if (used == width) max_error = QOAMAX_MIN(max_error, slots.heap[0].error);
      }
    }

    if (used == 0) return 0;

    // Construct the surviving nodes (in slot order)
    qoamax_beam_node *dst = &buffers[(i & 1) * width];
    for (int slot = 0; slot < used; slot++) {
      const qoamax_beam_node *parent    = &src[slots.parent[slot]];
      int codepoint                     = slots.codepoint[slot];
      const qoamax_decoder_entry *entry = &row[codepoint];

      qoamax_lms_regs regs = qoamax_lms_load(&parent->lms);
      qoamax_lms_update(&regs, entry, qoamax_clamp_s16(predicted[slots.parent[slot]] + entry->dequantized));
      qoamax_lms_store(&dst[slot].lms, &regs);
      dst[slot].error = slots.error[slot];

      if (!first_slice)
        dst[slot].slice = parent->slice;
      else if (i == 0)
        dst[slot].slice = ((uint64_t)scalefactor << 3) | codepoint;
      else
        dst[slot].slice = (parent->slice << 3) | codepoint;
    }

    src       = dst;
    src_count = used;
  }

  *out = (qoamax_beam_node *)src;
  return src_count;
}

// The functions of the beam search, compiled for the compile target and, on x86 without AVX2, also for AVX2 (see
// qoamax_beam.h). The AVX2 variant makes FAST 7% faster and MEDIUM to VERYSLOW 9-10%.
#define QOAMAX_VARIANT_SUFFIX
#define QOAMAX_VARIANT_TARGET
#ifdef __AVX2__
#define QOAMAX_BEAM_BRANCHLESS true
#else
#define QOAMAX_BEAM_BRANCHLESS false
#endif
#include "./qoamax_beam.h"

#ifdef QOAMAX_TARGET_BEAM
#define QOAMAX_VARIANT_SUFFIX __avx2
#define QOAMAX_VARIANT_TARGET [[gnu::target(QOAMAX_TARGET_BEAM)]]
#define QOAMAX_BEAM_BRANCHLESS true
#include "./qoamax_beam.h"
#endif

///////////////////////////////////////////////////////////////////////////////
//                               Final Encoder                               //
///////////////////////////////////////////////////////////////////////////////

//! Returns the scalefactor ULTRAFAST estimates for the samples: the number of thresholds the largest residual of the
//! prediction from the input samples (with the LMS weights, which aren't updated) exceeds.
static int qoamax_ultrafast_estimate(const qoamax_lms *lms, int samples_count, const int16_t samples[samples_count]) {
  const int32_t *weights = lms->weights;
  int32_t x0 = lms->history[0], x1 = lms->history[1], x2 = lms->history[2], x3 = lms->history[3];
  int32_t max_residual = 0;
  for (int i = 0; i < samples_count; i++) {
    uint32_t sum     = (uint32_t)weights[0] * (uint32_t)x0 + (uint32_t)weights[1] * (uint32_t)x1 +
                       (uint32_t)weights[2] * (uint32_t)x2 + (uint32_t)weights[3] * (uint32_t)x3;
    int32_t residual = samples[i] - ((int32_t)sum >> 13);
    int32_t abs      = residual < 0 ? -residual : residual;
    max_residual     = QOAMAX_MAX(max_residual, abs);
    x0 = x1, x1 = x2, x2 = x3, x3 = samples[i];
  }

  int estimate = 0;
  for (int k = 0; k < 14; k++) estimate += max_residual > qoamax_ultrafast_thresholds[k];
  return estimate;
}

//! ULTRAFAST without SIMD: encodes the stream's slices with the hill-climb, using the better of the estimated
//! scalefactor and the next larger one (the estimated one if both are equal).
static void qoamax_encode_ultrafast_scalar(qoamax__stream *stream) {
  while (stream->slice_count > 0) {
    int samples_count = (int)QOAMAX_MIN(stream->samples_size, 20);
    int16_t samples[20];
    for (int i = 0; i < samples_count; i++) samples[i] = stream->samples[i * stream->sample_stride];

    int estimate        = qoamax_ultrafast_estimate(&stream->state.lms, samples_count, samples);
    uint64_t best_error = UINT64_MAX, best_slice = 0;
    qoamax_lms best_lms = stream->state.lms;
    // A loop rather than two calls, so the compiler doesn't duplicate the (inlined) hill-climb.
    for (int scalefactor = estimate; scalefactor <= estimate + 1; scalefactor++) {
      qoamax_lms lms = stream->state.lms;
      uint64_t slice;
      uint64_t error = qoamax_encode_hillclimb_search(&lms, samples_count, samples, scalefactor, best_error, &slice);
      if (error < best_error) best_error = error, best_lms = lms, best_slice = slice;
    }

    stream->state.lms = best_lms;
    stream->state.error += best_error;
    stream->state.prev_scalefactor = best_slice >> 60;
    qoamax__stream_put(stream, best_slice, samples_count);
  }
}

//! Processes the slices of every stream: the full slices with a SIMD kernel, which takes up to `per_call` streams at
//! once and advances all of them by the same number of slices, while at least `min_count` streams have full slices
//! left; everything else (partial slices, too few streams) one stream at a time with `scalar`, which gives the same
//! results. The streams may have different lengths: the kernel's streams are refilled as they run out.
static void qoamax__run_streams(qoamax__stream *streams, size_t count,
                                void (*kernel)(qoamax__stream *streams, int count, int slice_count), int per_call,
                                int min_count, void (*scalar)(qoamax__stream *stream)) {
  qoamax__stream batch[16]; // copies of the streams in the kernel's lanes
  size_t index[16];         // their positions in `streams`
  int used    = 0;
  size_t next = 0;
  qoamax_assume(per_call <= 16);

  while (kernel) {
    for (; used < per_call && next < count; next++) {
      if (qoamax__stream_full_slices(&streams[next]) == 0)
        scalar(&streams[next]);
      else
        batch[used] = streams[next], index[used++] = next;
    }
    if (used < min_count) break;

    // At most 65536 slices per call, so that ULTRAFAST's sums of errors (of 1.3 M samples) can't overflow.
    size_t slices = 1 << 16;
    for (int i = 0; i < used; i++) slices = QOAMAX_MIN(slices, qoamax__stream_full_slices(&batch[i]));
    kernel(batch, used, (int)slices);

    // Streams without full slices left make room for the next ones.
    for (int i = 0; i < used;) {
      if (qoamax__stream_full_slices(&batch[i]) > 0) {
        i++;
        continue;
      }
      streams[index[i]] = batch[i];
      scalar(&streams[index[i]]);
      used--, batch[i] = batch[used], index[i] = index[used];
    }
  }

  for (int i = 0; i < used; i++) streams[index[i]] = batch[i], scalar(&streams[index[i]]);
  for (; next < count; next++) scalar(&streams[next]);
}

//! ULTRAFAST: encodes the slices of every stream, half as many streams at once as the widest SIMD variant the CPU
//! supports has lanes.
static void qoamax_encode_ultrafast(qoamax__stream *streams, size_t count) {
  const qoamax__simd *simd = qoamax__simd_get();
  qoamax__run_streams(streams, count, simd ? simd->encode_ultrafast : NULL, simd ? simd->lanes / 2 : 0, 1,
                      qoamax_encode_ultrafast_scalar);
}

//! SUPERFAST, VERYFAST: hill-climb with the best scalefactors over the lookahead.
static uint64_t qoamax_encode_slice__hillclimb(qoamax_encoder_channel *restrict channel, int lookahead,
                                               const int16_t samples[static restrict lookahead]) {
  uint64_t slice;
  uint64_t error;
  qoamax_encode_slice__hillclimb_scalefactor_search(&channel->lms, channel->prev_scalefactor, lookahead, samples,
                                                    UINT64_MAX, &slice, &error);

  channel->prev_scalefactor = slice >> 60;
  channel->error += error;
  return slice;
}

//! FAST to PLACEBO: beam search for every scalefactor over the lookahead.
static uint64_t qoamax_encode_slice__beam(qoamax_encoder_channel *restrict channel, int beam_width, int lookahead,
                                          const int16_t samples[static restrict lookahead]) {
  // The (unclamped) hill-climb result bounds the beam search, which then only has to look for better results.
  qoamax_lms hillclimb_lms = channel->lms;
  uint64_t hillclimb_slice, hillclimb_first_error;
  uint64_t hillclimb_error = qoamax_encode_slice__hillclimb_scalefactor_search_unclamped(
      &hillclimb_lms, channel->prev_scalefactor, lookahead, samples, UINT64_MAX, &hillclimb_slice,
      &hillclimb_first_error);

  qoamax_beam_node root = {.error = 0, .slice = 0, .lms = channel->lms};
  qoamax_beam_node best = {.error = -1};
#ifdef QOAMAX_TARGET_BEAM
  if (qoamax__lanes_supported() >= 8)
    qoamax_encode_slice__beam_search__avx2(&root, beam_width, channel->prev_scalefactor, lookahead, samples,
                                           hillclimb_error, &best);
  else
#endif
    qoamax_encode_slice__beam_search(&root, beam_width, channel->prev_scalefactor, lookahead, samples, hillclimb_error,
                                     &best);

  // Use the hill-climb result unless the beam search found a better one.
  // A partial slice (end of the stream) must be left-aligned like a full one.
  int samples_count = QOAMAX_MIN(lookahead, 20);
  uint64_t slice    = best.error > hillclimb_error ? hillclimb_slice : best.slice << ((20 - samples_count) * 3);

  int16_t decoded[20];
  qoamax_decode_slice(&channel->lms, slice, decoded);
  for (int i = 0; i < samples_count; i++) channel->error += qoamax_error_squared(samples[i], decoded[i]);

  channel->prev_scalefactor = slice >> 60;
  return slice;
}

uint64_t qoamax_encode_slice(qoamax_encoder_channel *restrict channel, qoamax_encoder_effort effort, int samples_size,
                             const int16_t samples[static restrict samples_size]) {
  qoamax_assume(samples_size >= 1);

  int lookahead  = QOAMAX_MIN(samples_size, qoamax_encoder_effort_get_lookahead(effort));
  int beam_width = effort >> 2;

  if (effort < 0) {
    uint64_t slice;
    qoamax__stream stream = {.state         = *channel,
                             .slices        = (char *)&slice,
                             .slice_step    = 8,
                             .sample_stride = 1,
                             .samples       = (int16_t *)samples, // only read
                             .samples_size  = QOAMAX_MIN(samples_size, 20),
                             .slice_count   = 1};
    qoamax_encode_ultrafast(&stream, 1);
    *channel = stream.state;
    return slice;
  }
  if (beam_width == 0) return qoamax_encode_slice__hillclimb(channel, lookahead, samples);
  return qoamax_encode_slice__beam(channel, beam_width, lookahead, samples);
}

//! Encodes the slices of every stream (samples_size must hold the lookahead of every slice, unless the stream ends).
//! ULTRAFAST encodes the streams in parallel.
static void qoamax__encode_streams(qoamax_encoder_effort effort, qoamax__stream *streams, size_t count) {
  if (effort < 0) {
    qoamax_encode_ultrafast(streams, count);
    return;
  }

  for (size_t j = 0; j < count; j++) {
    qoamax__stream *stream = &streams[j];
    while (stream->slice_count > 0) {
      int samples_size = (int)QOAMAX_MIN(stream->samples_size, (size_t)qoamax_encoder_effort_get_lookahead(effort));
      const int16_t *samples = stream->samples;
      int16_t gathered[QOAMAX_MAX_LOOKAHEAD_SLICES * 20];
      if (stream->sample_stride != 1) {
        for (int i = 0; i < samples_size; i++) gathered[i] = samples[i * stream->sample_stride];
        samples = gathered;
      }
      uint64_t slice = qoamax_encode_slice(&stream->state, effort, samples_size, samples);
      qoamax__stream_put(stream, slice, QOAMAX_MIN(samples_size, 20));
    }
  }
}

//! Decodes the stream's slices without SIMD (the last one may end before the slice, see samples_size).
static void qoamax__decode_stream_scalar(qoamax__stream *stream) {
  int stride = stream->sample_stride;
  while (stream->slice_count > 0) {
    int samples_count = (int)QOAMAX_MIN(stream->samples_size, 20);
    uint64_t slice    = qoamax__stream_get(stream);
    if (samples_count < 20) {
      // Decode a partial slice into a temporary buffer, so as not to write out of bounds.
      int16_t decoded[20];
      qoamax_decode_slice(&stream->state.lms, slice, decoded);
      for (int i = 0; i < samples_count; i++) stream->samples[i * stride] = decoded[i];
    } else if (stride == 1) {
      qoamax_decode_slice(&stream->state.lms, slice, stream->samples);
    } else {
      qoamax_decode_slice__interleaved(&stream->state.lms, slice, stream->samples, stride);
    }
    qoamax__stream_next(stream, samples_count);
  }
}

//! Decodes the slices of every stream. Full slices are decoded in SIMD lanes, one stream per lane, using the widest
//! variant the CPU supports, while at least half of the lanes and 4 streams are in use: with fewer, the lanes are
//! slower than the scalar decoder (which the CPU runs for several streams in parallel too).
static void qoamax__decode_streams(qoamax__stream *streams, size_t count) {
  const qoamax__simd *simd = qoamax__simd_get();
  qoamax__run_streams(streams, count, simd ? simd->decode : NULL, simd ? simd->lanes : 0,
                      simd ? QOAMAX_MAX(simd->lanes / 2, 4) : 0, qoamax__decode_stream_scalar);
}

#define QOAMAX_API_STREAMS 32 //!< Streams of the public API converted at once (on the stack)

void qoamax_encode_streams(qoamax_encoder_effort effort, size_t stream_count,
                           qoamax_encode_stream streams[stream_count], bool samples_complete) {
  size_t lookahead = qoamax_encoder_effort_get_lookahead(effort);
  qoamax__stream internal[QOAMAX_API_STREAMS];

  for (size_t first = 0; first < stream_count; first += QOAMAX_API_STREAMS) {
    size_t count = QOAMAX_MIN(stream_count - first, QOAMAX_API_STREAMS);
    for (size_t j = 0; j < count; j++) {
      const qoamax_encode_stream *stream = &streams[first + j];
      size_t size                        = stream->samples_size;
      // Without the end of the stream, only slices with their whole lookahead.
      size_t slices = samples_complete ? (size + 19) / 20 : size < lookahead ? 0 : (size - lookahead) / 20 + 1;
      internal[j]   = (qoamax__stream){
          .state         = stream->channel,
          .slices        = (char *)stream->slices,
          .slice_step    = 8,
          .sample_stride = stream->sample_stride,
          .samples       = (int16_t *)stream->samples, // only read
          .samples_size  = size,
          .slice_count   = QOAMAX_MIN(slices, stream->slices_size),
      };
    }

    qoamax__encode_streams(effort, internal, count);

    for (size_t j = 0; j < count; j++) {
      qoamax_encode_stream *stream = &streams[first + j];
      uint64_t *slices             = (uint64_t *)internal[j].slices;
      stream->channel              = internal[j].state;
      stream->samples              = internal[j].samples;
      stream->samples_size         = internal[j].samples_size;
      stream->slices_size -= slices - stream->slices;
      stream->slices = slices;
    }
  }
}

void qoamax_decode_streams(size_t stream_count, qoamax_decode_stream streams[stream_count]) {
  qoamax__stream internal[QOAMAX_API_STREAMS];

  for (size_t first = 0; first < stream_count; first += QOAMAX_API_STREAMS) {
    size_t count = QOAMAX_MIN(stream_count - first, QOAMAX_API_STREAMS);
    for (size_t j = 0; j < count; j++) {
      const qoamax_decode_stream *stream = &streams[first + j];
      internal[j]                        = (qoamax__stream){
          .state.lms     = stream->lms,
          .slices        = (char *)stream->slices, // only read
          .slice_step    = 8,
          .sample_stride = stream->sample_stride,
          .samples       = stream->samples,
          .samples_size  = stream->samples_size,
          .slice_count   = QOAMAX_MIN(stream->slices_size, (stream->samples_size + 19) / 20),
      };
    }

    qoamax__decode_streams(internal, count);

    for (size_t j = 0; j < count; j++) {
      qoamax_decode_stream *stream = &streams[first + j];
      const uint64_t *slices       = (const uint64_t *)internal[j].slices;
      stream->lms                  = internal[j].state.lms;
      stream->samples              = internal[j].samples;
      stream->samples_size         = internal[j].samples_size;
      stream->slices_size -= slices - stream->slices;
      stream->slices = slices;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
//                                File Format                                //
///////////////////////////////////////////////////////////////////////////////

#define QOAMAX_MAGIC 0x716F6166   //!< "qoaf"
#define QOAMAX_DECODER_STREAMS 16 //!< Most streams (frames * channels) the decoder decodes at once (the widest SIMD)

//! The size of a frame in bytes: the frame header, every channel's LMS state (history, weights) and the slices.
static inline uint32_t qoamax__frame_size(int channel_count, int samples_per_channel) {
  return 8 * (1 + channel_count * (2 + (samples_per_channel + 19) / 20));
}

//! The streams of the slices of every channel in a frame, from the slice at `slices` and sample samples[ch] on (the
//! state is left to the caller).
static void qoamax__frame_streams(qoamax__stream *streams, int channel_count, char *slices, int16_t *const samples[],
                                  int stride, size_t samples_size, size_t slice_count) {
  for (int ch = 0; ch < channel_count; ch++) {
    streams[ch].slices        = slices + 8 * ch;
    streams[ch].slice_step    = 8 * channel_count;
    streams[ch].big_endian    = true;
    streams[ch].sample_stride = stride;
    streams[ch].samples       = samples[ch];
    streams[ch].samples_size  = samples_size;
    streams[ch].slice_count   = slice_count;
  }
}

typedef struct {
  uint8_t channel_count;
  int32_t samplerate;
  uint16_t samples_per_channel;
  uint16_t size; //!< In bytes
} qoamax_frame_header;

static inline qoamax_frame_header qoamax__parse_frame_header(uint64_t header) {
  return (qoamax_frame_header){
      .channel_count       = header >> 56,
      .samplerate          = (header >> 32) & 0xFFFFFF,
      .samples_per_channel = header >> 16,
      .size                = header,
  };
}

//! Reads a frame header and the LMS state of every channel that follows it (into lms[ch]).
//! @returns the frame's sample count per channel, or an error code.
static int qoamax_decoder_read_frame_header(const qoamax_decoder *restrict self, const char *restrict *restrict file,
                                            qoamax_lms lms[restrict]) {
  qoamax_frame_header header = qoamax__parse_frame_header(qoamax__read_u64be(file));
  if (header.channel_count != self->channel_count) return QOAMAX_ERROR_SAMPLE_FORMAT_BREAK;
  if (header.samplerate != self->samplerate) return QOAMAX_ERROR_SAMPLE_FORMAT_BREAK;

  qoamax_assume(header.channel_count >= 1 && header.channel_count <= QOAMAX_MAX_CHANNEL_COUNT);

  // The frame size is fully determined by the sample count; padding slices are not allowed.
  if (header.samples_per_channel == 0 || header.samples_per_channel > QOAMAX_FRAME_SAMPLES)
    return QOAMAX_ERROR_INVALID_FRAME_HEADER;
  if (header.size != qoamax__frame_size(header.channel_count, header.samples_per_channel))
    return QOAMAX_ERROR_INVALID_FRAME_HEADER;

  for (int ch = 0; ch < header.channel_count; ch++) {
    uint64_t history = qoamax__read_u64be(file);
    uint64_t weights = qoamax__read_u64be(file);
    for (int i = 0; i < 4; i++) {
      lms[ch].history[i] = (int16_t)(history >> 48);
      lms[ch].weights[i] = (int16_t)(weights >> 48);
      history <<= 16;
      weights <<= 16;
    }
  }

  return header.samples_per_channel;
}

int qoamax_decoder_init(qoamax_decoder *restrict self, size_t file_size, const char *restrict *restrict file,
                        qoamax_decoder_query *query) {
  if (file_size < 16) return QOAMAX_ERROR_INVALID_ARGUMENT;

  uint64_t file_header = qoamax__read_u64be(file);
  if ((file_header >> 32) != QOAMAX_MAGIC) return QOAMAX_ERROR_INVALID_FILE_HEADER;

  // Only peek at the first frame header to get the stream's format; qoamax_decoder_read reads it again.
  const char *frame          = *file;
  qoamax_frame_header header = qoamax__parse_frame_header(qoamax__read_u64be(&frame));

  if (header.channel_count < 1 || header.channel_count > 31) return QOAMAX_ERROR_INVALID_FILE_HEADER;
  if (header.channel_count > QOAMAX_MAX_CHANNEL_COUNT) return QOAMAX_ERROR_UNSUPPORTED_CHANNEL_COUNT;

  query->total_samples = (uint32_t)file_header;
  query->samplerate    = header.samplerate;
  query->channel_count = header.channel_count;
  self->frame_samples  = 0;
  self->channel_count  = header.channel_count;
  self->samplerate     = header.samplerate;

  return 0;
}

//! The decoder of @ref qoamax_decoder_read and @ref qoamax_decoder_read_interleaved: the samples of channel ch are
//! written to samples[ch], one every `stride` (the pointers are advanced past them).
static int qoamax_decoder_read__impl(qoamax_decoder *restrict self, size_t file_size,
                                     const char *restrict *restrict file, size_t samples_per_channel,
                                     int16_t *samples[], int stride) {
  int channel_count = self->channel_count;
  qoamax_assume(channel_count >= 1 && channel_count <= QOAMAX_MAX_CHANNEL_COUNT);
  const char *file_end = *file + file_size;

  while (true) {
    if (self->frame_samples == 0) {
      // The full frames that are complete in the file and fit into the output are decoded at once: as every frame
      // header holds the LMS states, the channels of all frames are independent streams.
      qoamax__stream streams[QOAMAX_DECODER_STREAMS];
      int frames = 0, max_frames = QOAMAX_MAX(QOAMAX_DECODER_STREAMS / channel_count, 1);
      const char *frame = *file;
      size_t frame_size = qoamax__frame_size(channel_count, QOAMAX_FRAME_SAMPLES);
      while (frames < max_frames && (size_t)(file_end - frame) >= frame_size &&
             samples_per_channel >= (size_t)(frames + 1) * QOAMAX_FRAME_SAMPLES) {
        qoamax_lms lms[QOAMAX_MAX_CHANNEL_COUNT];
        const char *slices = frame;
        // Other frames (the last one, invalid ones) are left to the frame-by-frame path below.
        if (qoamax_decoder_read_frame_header(self, &slices, lms) != QOAMAX_FRAME_SAMPLES) break;

        int16_t *frame_samples[QOAMAX_MAX_CHANNEL_COUNT];
        for (int ch = 0; ch < channel_count; ch++)
          frame_samples[ch] = samples[ch] + frames * QOAMAX_FRAME_SAMPLES * stride;
        qoamax__stream *frame_streams = &streams[frames * channel_count];
        qoamax__frame_streams(frame_streams, channel_count, (char *)slices, frame_samples, stride, QOAMAX_FRAME_SAMPLES,
                              QOAMAX_FRAME_SAMPLES / 20); // only read
        for (int ch = 0; ch < channel_count; ch++) frame_streams[ch].state.lms = lms[ch];
        frame += frame_size, frames++;
      }
      if (frames > 0) {
        qoamax__decode_streams(streams, frames * channel_count);
        for (int ch = 0; ch < channel_count; ch++) samples[ch] += frames * QOAMAX_FRAME_SAMPLES * stride;
        *file = frame;
        samples_per_channel -= frames * QOAMAX_FRAME_SAMPLES;
        continue;
      }

      if (file_end - *file < 8 * (1 + 2 * channel_count)) return 0;
      // Start the frame only if the output has room for its first slice. Otherwise the next call would have to decode
      // the whole frame slice by slice, instead of with the frames after it.
      const char *header = *file;
      int frame_samples  = qoamax_decoder_read_frame_header(self, &header, self->lms);
      if (frame_samples < 0) {
        *file += 8; // past the invalid header
        return frame_samples;
      }
      if (samples_per_channel < (size_t)QOAMAX_MIN(frame_samples, 20)) return 0;
      *file               = header;
      self->frame_samples = frame_samples;
    }

    // As many slices of the current frame as the file holds and the output has room for (a partial slice only at
    // the end of the frame).
    size_t frame_slices = (self->frame_samples + 19) / 20;
    size_t slices       = QOAMAX_MIN(frame_slices, (size_t)(file_end - *file) / (8 * channel_count));
    slices = QOAMAX_MIN(slices, samples_per_channel >= self->frame_samples ? frame_slices : samples_per_channel / 20);
    if (slices == 0) return 0;
    size_t samples_count = QOAMAX_MIN(20 * slices, self->frame_samples);

    qoamax__stream streams[QOAMAX_MAX_CHANNEL_COUNT];
    qoamax__frame_streams(streams, channel_count, (char *)*file, samples, stride, samples_count, slices); // only read
    for (int ch = 0; ch < channel_count; ch++) streams[ch].state.lms = self->lms[ch];
    qoamax__decode_streams(streams, channel_count);
    for (int ch = 0; ch < channel_count; ch++) self->lms[ch] = streams[ch].state.lms, samples[ch] = streams[ch].samples;

    *file += 8 * channel_count * slices;
    self->frame_samples -= samples_count;
    samples_per_channel -= samples_count;
  }
}

int qoamax_decoder_read(qoamax_decoder *restrict self, size_t file_size, const char *restrict *restrict file,
                        size_t samples_per_channel, int16_t *restrict samples[static restrict self->channel_count]) {
  return qoamax_decoder_read__impl(self, file_size, file, samples_per_channel, (int16_t **)samples, 1);
}

int qoamax_decoder_read_interleaved(qoamax_decoder *restrict self, size_t file_size,
                                    const char *restrict *restrict file, size_t samples_size,
                                    int16_t *restrict samples[static restrict 1]) {
  int16_t *channels[QOAMAX_MAX_CHANNEL_COUNT];
  for (int ch = 0; ch < self->channel_count; ch++) channels[ch] = *samples + ch;
  int err  = qoamax_decoder_read__impl(self, file_size, file, samples_size / self->channel_count, channels,
                                       self->channel_count);
  *samples = channels[0];
  return err;
}

int qoamax_encoder_init(qoamax_encoder *restrict self, qoamax_encoder_effort effort, int channel_count,
                        uint32_t total_samples, int32_t samplerate) {
  if (channel_count < 1 || channel_count > 31) return QOAMAX_ERROR_INVALID_ARGUMENT;
  if (channel_count > QOAMAX_MAX_CHANNEL_COUNT) return QOAMAX_ERROR_UNSUPPORTED_CHANNEL_COUNT;
  if ((samplerate >> 24) != 0) return QOAMAX_ERROR_INVALID_ARGUMENT;

  self->effort         = effort;
  self->channel_count  = channel_count;
  self->total_samples  = total_samples;
  self->samplerate     = samplerate;
  self->frame_slices   = 0;
  self->header_written = false;
  self->frame_index    = 0;

  for (int i = 0; i < channel_count; i++) {
    self->channels[i].lms              = QOAMAX_LMS_INITIAL;
    self->channels[i].error            = 0;
    self->channels[i].prev_scalefactor = 0;
    for (int k = 0; k < 4; k++) self->group_weights[i][k] = 0, self->last_samples[i][k] = 0;
  }

  return 0;
}

//! Starts a frame: writes the file header (along with the first frame, as a QOA file can't be empty), the frame
//! header and the LMS state every channel starts with, which is stored in streams[ch].state.
//!
//! That is the state where the previous frame ended, with the weights truncated to int16 (the frame header stores them
//! that way, and the decoder continues from the stored values). ULTRAFAST frames after the first group start from the
//! weights at the end of the previous group and the last four input samples instead (see
//! QOAMAX_ULTRAFAST_GROUP_FRAMES), which makes the frames of a group independent of each other.
static void qoamax_encoder_start_frame(qoamax_encoder *restrict self, char *restrict *restrict out,
                                       int samples_per_channel, qoamax__stream *streams) {
  if (!self->header_written) qoamax__write_u64be(out, (uint64_t)QOAMAX_MAGIC << 32 | self->total_samples);
  self->header_written = true;

  uint64_t frame_header = (uint64_t)self->channel_count << 56 | (uint64_t)self->samplerate << 32 |
                          (uint64_t)samples_per_channel << 16 |
                          qoamax__frame_size(self->channel_count, samples_per_channel);
  qoamax__write_u64be(out, frame_header);

  bool restart = self->effort < 0 && self->frame_index >= QOAMAX_ULTRAFAST_GROUP_FRAMES;
  for (int ch = 0; ch < self->channel_count; ch++) {
    qoamax_encoder_channel *state = &streams[ch].state;
    *state                        = self->channels[ch];
    uint64_t history = 0, weights = 0;
    for (int k = 0; k < 4; k++) {
      if (restart) state->lms.history[k] = self->last_samples[ch][k];
      state->lms.weights[k] = restart ? self->group_weights[ch][k] : (int16_t)state->lms.weights[k];
      history               = (history << 16) | (uint16_t)state->lms.history[k];
      weights               = (weights << 16) | (uint16_t)state->lms.weights[k];
    }
    qoamax__write_u64be(out, history);
    qoamax__write_u64be(out, weights);
  }
  self->frame_index += 1;
}

//! The encoder of @ref qoamax_encoder_write and @ref qoamax_encoder_write_interleaved: the samples of channel ch are
//! read from samples[ch], one every `stride` (the pointers are advanced past the encoded ones).
//!
//! The frames are encoded slice by slice, as far as the input and output allow, all channels at once. ULTRAFAST
//! encodes the full frames of a group (except for the first one) at once, if they are available and fit into the
//! output.
//! @returns the number of samples per channel that were encoded.
static size_t qoamax_encoder_write__impl(qoamax_encoder *restrict self, size_t file_size, char *restrict *restrict file,
                                         size_t samples_per_channel, const int16_t *samples[], int stride,
                                         bool samples_complete) {
  int channel_count = self->channel_count;
  size_t lookahead  = qoamax_encoder_effort_get_lookahead(self->effort);
  char *file_end    = *file + file_size;
  size_t remaining  = samples_per_channel;
  qoamax__stream streams[QOAMAX_ULTRAFAST_GROUP_FRAMES * QOAMAX_MAX_CHANNEL_COUNT];

  while (true) {
    size_t samples_count;
    if (self->frame_slices == 0) {
      // The frame header holds the frame's sample count, so all its samples must be available (or the stream ends).
      if (remaining < QOAMAX_FRAME_SAMPLES && (!samples_complete || remaining == 0)) break;
      ptrdiff_t space = file_end - *file - (self->header_written ? 0 : 8);

      // ULTRAFAST: the full frames of the current group that are available and fit into the output.
      size_t frames = 0, frame_size = qoamax__frame_size(channel_count, QOAMAX_FRAME_SAMPLES);
      if (self->effort < 0 && self->frame_index >= QOAMAX_ULTRAFAST_GROUP_FRAMES && space > 0) {
        frames = QOAMAX_ULTRAFAST_GROUP_FRAMES - self->frame_index % QOAMAX_ULTRAFAST_GROUP_FRAMES;
        frames = QOAMAX_MIN(frames, remaining / QOAMAX_FRAME_SAMPLES);
        frames = QOAMAX_MIN(frames, (size_t)space / frame_size);
      }

      if (frames == 0) {
        // Start a frame, which the next iterations encode slice by slice.
        if (space < 8 * (1 + 2 * channel_count)) break;
        int frame_samples = QOAMAX_MIN(remaining, QOAMAX_FRAME_SAMPLES);
        qoamax_encoder_start_frame(self, file, frame_samples, streams);
        for (int ch = 0; ch < channel_count; ch++) self->channels[ch] = streams[ch].state;
        self->frame_slices = (frame_samples + 19) / 20;
        continue;
      }

      for (size_t frame = 0; frame < frames; frame++) {
        int16_t *frame_samples[QOAMAX_MAX_CHANNEL_COUNT];
        for (int ch = 0; ch < channel_count; ch++) {
          frame_samples[ch] = (int16_t *)samples[ch] + frame * QOAMAX_FRAME_SAMPLES * stride; // only read
          if (frame > 0)
            for (int k = 0; k < 4; k++) self->last_samples[ch][k] = frame_samples[ch][(k - 4) * stride];
        }
        qoamax__stream *frame_streams = &streams[frame * channel_count];
        qoamax_encoder_start_frame(self, file, QOAMAX_FRAME_SAMPLES, frame_streams);
        for (int ch = 0; ch < channel_count; ch++) frame_streams[ch].state.error = 0; // summed up below
        qoamax__frame_streams(frame_streams, channel_count, *file, frame_samples, stride, QOAMAX_FRAME_SAMPLES,
                              QOAMAX_FRAME_SAMPLES / 20);
        *file += 8 * channel_count * (QOAMAX_FRAME_SAMPLES / 20);
      }
      qoamax__encode_streams(self->effort, streams, frames * channel_count);

      // The channels continue from the last frame.
      for (int ch = 0; ch < channel_count; ch++) {
        uint64_t error = self->channels[ch].error;
        for (size_t frame = 0; frame < frames; frame++) error += streams[frame * channel_count + ch].state.error;
        self->channels[ch]       = streams[(frames - 1) * channel_count + ch].state;
        self->channels[ch].error = error;
      }
      samples_count = frames * QOAMAX_FRAME_SAMPLES;
    } else {
      // As many slices of the current frame as the input (with lookahead) and output allow.
      size_t slices = QOAMAX_MIN(self->frame_slices, (size_t)(file_end - *file) / (8 * channel_count));
      if (samples_complete)
        slices = QOAMAX_MIN(slices, (remaining + 19) / 20);
      else
        slices = remaining < lookahead ? 0 : QOAMAX_MIN(slices, (remaining - lookahead) / 20 + 1);
      if (slices == 0) break;

      qoamax__frame_streams(streams, channel_count, *file, (int16_t **)samples, stride, remaining, slices); // only read
      for (int ch = 0; ch < channel_count; ch++) streams[ch].state = self->channels[ch];
      qoamax__encode_streams(self->effort, streams, channel_count);
      for (int ch = 0; ch < channel_count; ch++) self->channels[ch] = streams[ch].state;

      *file += 8 * channel_count * slices;
      self->frame_slices -= slices;
      samples_count = QOAMAX_MIN(remaining, 20 * slices);
    }

    for (int ch = 0; ch < channel_count; ch++) {
      // Keep the last four samples for the next ULTRAFAST frame.
      int16_t *last = self->last_samples[ch];
      for (size_t i = samples_count - QOAMAX_MIN(samples_count, 4); i < samples_count; i++) {
        last[0] = last[1], last[1] = last[2], last[2] = last[3];
        last[3] = samples[ch][i * stride];
      }
      samples[ch] += samples_count * stride;
    }
    remaining -= samples_count;

    // At the end of a group, store the LMS weights the next one starts with.
    bool group_end = (self->frame_index - 1) % QOAMAX_ULTRAFAST_GROUP_FRAMES == QOAMAX_ULTRAFAST_GROUP_FRAMES - 1;
    if (self->frame_slices == 0 && group_end)
      for (int ch = 0; ch < channel_count; ch++)
        for (int k = 0; k < 4; k++) self->group_weights[ch][k] = (int16_t)self->channels[ch].lms.weights[k];
  }

  return samples_per_channel - remaining;
}

int qoamax_encoder_write(qoamax_encoder *restrict self, size_t file_size, char *restrict *restrict file,
                         size_t samples_per_channel,
                         const int16_t *restrict samples[static restrict self->channel_count], bool samples_complete) {
  qoamax_encoder_write__impl(self, file_size, file, samples_per_channel, (const int16_t **)samples, 1,
                             samples_complete);
  return 0;
}

int qoamax_encoder_write_interleaved(qoamax_encoder *restrict self, size_t file_size, char *restrict *restrict file,
                                     size_t samples_size, const int16_t *restrict samples[static restrict 1],
                                     bool samples_complete) {
  const int16_t *channels[QOAMAX_MAX_CHANNEL_COUNT];
  for (int ch = 0; ch < self->channel_count; ch++) channels[ch] = *samples + ch;
  size_t encoded = qoamax_encoder_write__impl(self, file_size, file, samples_size / self->channel_count, channels,
                                              self->channel_count, samples_complete);
  *samples += encoded * self->channel_count;
  return 0;
}
