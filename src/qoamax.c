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

//! The scalefactor search of SUPERFAST/VERYFAST, using the widest SIMD variant the CPU supports (or the scalar search).
//! See @ref qoamax_encode_slice__hillclimb_scalefactor_search_scalar_impl.
// The dispatchers aren't inlined: inlined into qoamax_encode_slice, they made ULTRAFAST execute 1.5% more instructions.
[[gnu::noinline]]
static uint64_t qoamax_encode_slice__hillclimb_scalefactor_search(qoamax_lms *lms, int first_scalefactor, int lookahead,
                                                                  const int16_t samples[static restrict lookahead],
                                                                  uint64_t error_hint, uint64_t *out_slice,
                                                                  uint64_t *out_error) {
#ifdef QOAMAX_HAS_VARIANTS
  switch (qoamax__lanes_supported()) {
#ifdef QOAMAX_TARGET_X16
  case 16:
    return qoamax_encode_slice__hillclimb_scalefactor_search_lanes__x16(lms, first_scalefactor, lookahead, samples,
                                                                        error_hint, out_slice, out_error);
#endif
#ifdef QOAMAX_TARGET_X8
  case 8:
    return qoamax_encode_slice__hillclimb_scalefactor_search_lanes__x8(lms, first_scalefactor, lookahead, samples,
                                                                       error_hint, out_slice, out_error);
#endif
#ifdef QOAMAX_TARGET_X4
  case 4:
    return qoamax_encode_slice__hillclimb_scalefactor_search_lanes__x4(lms, first_scalefactor, lookahead, samples,
                                                                       error_hint, out_slice, out_error);
#endif
  }
#endif
#if QOAMAX_LANES > 1
  return qoamax_encode_slice__hillclimb_scalefactor_search_lanes(lms, first_scalefactor, lookahead, samples, error_hint,
                                                                 out_slice, out_error);
#else
  return qoamax_encode_slice__hillclimb_scalefactor_search_scalar(lms, first_scalefactor, lookahead, samples,
                                                                  error_hint, out_slice, out_error);
#endif
}

//! See @ref qoamax_encode_slice__hillclimb_scalefactor_search (unclamped variant, the beam search's error bound).
[[gnu::noinline]]
static uint64_t qoamax_encode_slice__hillclimb_scalefactor_search_unclamped(
    qoamax_lms *lms, int first_scalefactor, int lookahead, const int16_t samples[static restrict lookahead],
    uint64_t error_hint, uint64_t *out_slice, uint64_t *out_error) {
#ifdef QOAMAX_HAS_VARIANTS
  switch (qoamax__lanes_supported()) {
#ifdef QOAMAX_TARGET_X16
  case 16:
    return qoamax_encode_slice__hillclimb_scalefactor_search_lanes_unclamped__x16(
        lms, first_scalefactor, lookahead, samples, error_hint, out_slice, out_error);
#endif
#ifdef QOAMAX_TARGET_X8
  case 8:
    return qoamax_encode_slice__hillclimb_scalefactor_search_lanes_unclamped__x8(
        lms, first_scalefactor, lookahead, samples, error_hint, out_slice, out_error);
#endif
#ifdef QOAMAX_TARGET_X4
  case 4:
    return qoamax_encode_slice__hillclimb_scalefactor_search_lanes_unclamped__x4(
        lms, first_scalefactor, lookahead, samples, error_hint, out_slice, out_error);
#endif
  }
#endif
#if QOAMAX_LANES > 1
  return qoamax_encode_slice__hillclimb_scalefactor_search_lanes_unclamped(lms, first_scalefactor, lookahead, samples,
                                                                           error_hint, out_slice, out_error);
#else
  return qoamax_encode_slice__hillclimb_scalefactor_search_scalar_unclamped(lms, first_scalefactor, lookahead, samples,
                                                                            error_hint, out_slice, out_error);
#endif
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

//! ULTRAFAST without SIMD: hill-climb with the two scalefactors next to the previous one.
static uint64_t qoamax_encode_slice__ultrafast_scalar(qoamax_lms *lms, int prev_scalefactor, int samples_count,
                                                      const int16_t samples[static restrict samples_count],
                                                      uint64_t *out_error) {
  uint64_t best_slice = 0;
  uint64_t best_error = -1;
  qoamax_lms best_lms = *lms;

  // A loop rather than two calls, so the compiler doesn't duplicate the (inlined) hill-climb.
  for (int offset = -1; offset <= 1; offset += 2) {
    int scalefactor = (prev_scalefactor + offset) & 15;

    qoamax_lms local_lms = *lms;
    uint64_t slice;
    uint64_t error =
        qoamax_encode_hillclimb_search(&local_lms, samples_count, samples, scalefactor, best_error, &slice);
    if (error < best_error) {
      best_error = error;
      best_slice = slice;
      best_lms   = local_lms;
    }
  }

  *lms       = best_lms;
  *out_error = best_error;
  return best_slice;
}

//! ULTRAFAST: hill-climb with the scalefactors next to the previous one, using the widest SIMD variant the CPU
//! supports (or the scalar version).
static uint64_t qoamax_encode_slice__ultrafast(qoamax_encoder_channel *restrict channel, int samples_count,
                                               const int16_t samples[static restrict samples_count]) {
  qoamax_lms *lms = &channel->lms;
  int prev        = channel->prev_scalefactor;
  uint64_t slice, error;
#ifdef QOAMAX_HAS_VARIANTS
  int lanes = qoamax__lanes_supported();
#else
  int lanes = QOAMAX_LANES;
#endif
  switch (lanes) {
#ifdef QOAMAX_TARGET_X16
  case 16:
    slice = qoamax_encode_slice__ultrafast_lanes__x16(lms, prev, samples_count, samples, &error);
    break;
#endif
#ifdef QOAMAX_TARGET_X8
  case 8:
    slice = qoamax_encode_slice__ultrafast_lanes__x8(lms, prev, samples_count, samples, &error);
    break;
#endif
#ifdef QOAMAX_TARGET_X4
  case 4:
    slice = qoamax_encode_slice__ultrafast_lanes__x4(lms, prev, samples_count, samples, &error);
    break;
#endif
#if QOAMAX_LANES >= 4
  case QOAMAX_LANES:
    slice = qoamax_encode_slice__ultrafast_lanes(lms, prev, samples_count, samples, &error);
    break;
#endif
  default:
    slice = qoamax_encode_slice__ultrafast_scalar(lms, prev, samples_count, samples, &error);
    break;
  }

  channel->prev_scalefactor = slice >> 60;
  channel->error += error;
  return slice;
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

  if (effort < 0) return qoamax_encode_slice__ultrafast(channel, QOAMAX_MIN(samples_size, 20), samples);
  if (beam_width == 0) return qoamax_encode_slice__hillclimb(channel, lookahead, samples);
  return qoamax_encode_slice__beam(channel, beam_width, lookahead, samples);
}

///////////////////////////////////////////////////////////////////////////////
//                                File Format                                //
///////////////////////////////////////////////////////////////////////////////

#define QOAMAX_MAGIC 0x716F6166 //!< "qoaf"

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

//! The size of a frame in bytes: the frame header, every channel's LMS state (history, weights) and the slices.
static inline uint32_t qoamax__frame_size(int channel_count, int samples_per_channel) {
  return 8 * (1 + channel_count * (2 + (samples_per_channel + 19) / 20));
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

static int qoamax_decoder_read_frame_header(qoamax_decoder *restrict self, const char *restrict *restrict file) {
  qoamax_frame_header header = qoamax__parse_frame_header(qoamax__read_u64be(file));
  if (header.channel_count != self->channel_count) return QOAMAX_ERROR_SAMPLE_FORMAT_BREAK;
  if (header.samplerate != self->samplerate) return QOAMAX_ERROR_SAMPLE_FORMAT_BREAK;

  qoamax_assume(header.channel_count >= 1 && header.channel_count <= QOAMAX_MAX_CHANNEL_COUNT);

  // The frame size is fully determined by the sample count; padding slices are not allowed.
  if (header.samples_per_channel == 0 || header.samples_per_channel > QOAMAX_FRAME_SAMPLES)
    return QOAMAX_ERROR_INVALID_FRAME_HEADER;
  if (header.size != qoamax__frame_size(header.channel_count, header.samples_per_channel))
    return QOAMAX_ERROR_INVALID_FRAME_HEADER;

  self->frame_samples = header.samples_per_channel;

  for (int ch = 0; ch < header.channel_count; ch++) {
    uint64_t history = qoamax__read_u64be(file);
    uint64_t weights = qoamax__read_u64be(file);
    for (int i = 0; i < 4; i++) {
      self->lms[ch].history[i] = (int16_t)(history >> 48);
      self->lms[ch].weights[i] = (int16_t)(weights >> 48);
      history <<= 16;
      weights <<= 16;
    }
  }

  return 0;
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

//! Prepares decoding the next slice of every channel, reading a frame header first if necessary.
//! @returns the slice's sample count per channel, 0 if the file or the output space runs out, or an error code.
static int qoamax_decoder_begin_slice(qoamax_decoder *restrict self, const char *restrict *restrict file,
                                      const char *file_end, size_t samples_per_channel) {
  if (self->frame_samples == 0) {
    if (file_end - *file < 8 * (1 + 2 * self->channel_count)) return 0;

    int err = qoamax_decoder_read_frame_header(self, file);
    if (err) return err;
  }

  int slice_size = QOAMAX_MIN(20, self->frame_samples);
  if (file_end - *file < 8 * self->channel_count) return 0;
  if (samples_per_channel < (size_t)slice_size) return 0;

  self->frame_samples -= slice_size;
  return slice_size;
}

int qoamax_decoder_read(qoamax_decoder *restrict self, size_t file_size, const char *restrict *restrict file,
                        size_t samples_per_channel, int16_t *restrict samples[static restrict self->channel_count]) {
  qoamax_assume(self->channel_count >= 1 && self->channel_count <= QOAMAX_MAX_CHANNEL_COUNT);

  const char *file_end = *file + file_size;

  while (true) {
    int slice_size = qoamax_decoder_begin_slice(self, file, file_end, samples_per_channel);
    if (slice_size <= 0) return slice_size;

    for (int ch = 0; ch < self->channel_count; ch++) {
      uint64_t slice = qoamax__read_u64be(file);
      if (samples_per_channel >= 20) {
        qoamax_decode_slice(&self->lms[ch], slice, samples[ch]);
      } else {
        // Less than a full slice of space left: decode into a temporary buffer to not write out of bounds.
        int16_t tmp[20];
        qoamax_decode_slice(&self->lms[ch], slice, tmp);
        memcpy(samples[ch], tmp, slice_size * sizeof(int16_t));
      }
      samples[ch] += slice_size;
    }

    samples_per_channel -= slice_size;
  }
}

int qoamax_decoder_read_interleaved(qoamax_decoder *restrict self, size_t file_size,
                                    const char *restrict *restrict file, size_t samples_size,
                                    int16_t *restrict samples[static restrict 1]) {
  if (self->channel_count == 1) return qoamax_decoder_read(self, file_size, file, samples_size, samples);

  qoamax_assume(self->channel_count >= 1 && self->channel_count <= QOAMAX_MAX_CHANNEL_COUNT);

  const char *file_end       = *file + file_size;
  size_t samples_per_channel = samples_size / self->channel_count;

  while (true) {
    int slice_size = qoamax_decoder_begin_slice(self, file, file_end, samples_per_channel);
    if (slice_size <= 0) return slice_size;

    if (slice_size == 20) {
      // Full slices are decoded directly into the output.
      for (int ch = 0; ch < self->channel_count; ch++)
        qoamax_decode_slice__interleaved(&self->lms[ch], qoamax__read_u64be(file), &(*samples)[ch],
                                         self->channel_count);
      *samples += 20 * self->channel_count;
    } else {
      // The last slice of a frame may be shorter: decode it into a temporary buffer.
      int16_t decoded[QOAMAX_MAX_CHANNEL_COUNT][20];
      for (int ch = 0; ch < self->channel_count; ch++)
        qoamax_decode_slice(&self->lms[ch], qoamax__read_u64be(file), decoded[ch]);

      for (int i = 0; i < slice_size; i++)
        for (int ch = 0; ch < self->channel_count; ch++) *(*samples)++ = decoded[ch][i];
    }

    samples_per_channel -= slice_size;
  }
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

  for (int i = 0; i < channel_count; i++) {
    self->channels[i].lms              = QOAMAX_LMS_INITIAL;
    self->channels[i].error            = 0;
    self->channels[i].prev_scalefactor = 0;
  }

  return 0;
}

static void qoamax_encoder_write_frame_header(const qoamax_encoder *restrict self, char *restrict *restrict out,
                                              int samples_per_channel) {
  uint64_t frame_header = (uint64_t)self->channel_count << 56 | (uint64_t)self->samplerate << 32 |
                          (uint64_t)samples_per_channel << 16 |
                          qoamax__frame_size(self->channel_count, samples_per_channel);
  qoamax__write_u64be(out, frame_header);

  for (int ch = 0; ch < self->channel_count; ch++) {
    const qoamax_lms *lms = &self->channels[ch].lms;
    uint64_t history = 0, weights = 0;
    for (int i = 0; i < 4; i++) {
      history = (history << 16) | (uint16_t)lms->history[i];
      weights = (weights << 16) | (uint16_t)lms->weights[i];
    }
    qoamax__write_u64be(out, history);
    qoamax__write_u64be(out, weights);
  }
}

//! Prepares encoding the next slice of every channel, writing the file and frame headers first if necessary.
//! @returns whether the next slice can be encoded (false if the input or the output space runs out).
static bool qoamax_encoder_begin_slice(qoamax_encoder *restrict self, char *restrict *restrict file,
                                       const char *file_end, size_t samples_per_channel, bool samples_complete) {
  if (self->frame_slices == 0) {
    // The frame header holds the frame's sample count, so all its samples must be available (or the stream ends).
    // The file header is only written along with the first frame, as a QOA file can't be empty.
    if (samples_per_channel < QOAMAX_FRAME_SAMPLES && (!samples_complete || samples_per_channel == 0)) return false;
    if (file_end - *file < 8 * (1 + 2 * self->channel_count) + (self->header_written ? 0 : 8)) return false;
    if (!self->header_written) {
      qoamax__write_u64be(file, (uint64_t)QOAMAX_MAGIC << 32 | self->total_samples);
      self->header_written = true;
    }

    int frame_samples  = QOAMAX_MIN(samples_per_channel, QOAMAX_FRAME_SAMPLES);
    self->frame_slices = (frame_samples + 19) / 20;

    // The file stores the LMS weights as int16. The encoder must continue from the same state as the decoder.
    for (int ch = 0; ch < self->channel_count; ch++)
      for (int i = 0; i < 4; i++) self->channels[ch].lms.weights[i] = (int16_t)self->channels[ch].lms.weights[i];

    qoamax_encoder_write_frame_header(self, file, frame_samples);
  }

  size_t min_samples = samples_complete ? 1 : qoamax_encoder_effort_get_lookahead(self->effort);
  return file_end - *file >= 8 * self->channel_count && samples_per_channel >= min_samples;
}

int qoamax_encoder_write(qoamax_encoder *restrict self, size_t file_size, char *restrict *restrict file,
                         size_t samples_per_channel,
                         const int16_t *restrict samples[static restrict self->channel_count], bool samples_complete) {
  size_t lookahead = qoamax_encoder_effort_get_lookahead(self->effort);
  char *file_end   = *file + file_size;

  while (qoamax_encoder_begin_slice(self, file, file_end, samples_per_channel, samples_complete)) {
    int slice_size = QOAMAX_MIN(20, samples_per_channel);
    for (int ch = 0; ch < self->channel_count; ch++) {
      uint64_t slice = qoamax_encode_slice(&self->channels[ch], self->effort,
                                           QOAMAX_MIN(lookahead, samples_per_channel), samples[ch]);
      qoamax__write_u64be(file, slice);
      samples[ch] += slice_size;
    }

    self->frame_slices -= 1;
    samples_per_channel -= slice_size;
  }

  return 0;
}

int qoamax_encoder_write_interleaved(qoamax_encoder *restrict self, size_t file_size, char *restrict *restrict file,
                                     size_t samples_size, const int16_t *restrict samples[static restrict 1],
                                     bool samples_complete) {
  if (self->channel_count == 1)
    return qoamax_encoder_write(self, file_size, file, samples_size, samples, samples_complete);

  size_t lookahead           = qoamax_encoder_effort_get_lookahead(self->effort);
  size_t samples_per_channel = samples_size / self->channel_count;
  char *file_end             = *file + file_size;

  while (qoamax_encoder_begin_slice(self, file, file_end, samples_per_channel, samples_complete)) {
    int slice_size           = QOAMAX_MIN(20, samples_per_channel);
    int channel_samples_size = QOAMAX_MIN(lookahead, samples_per_channel);

    for (int ch = 0; ch < self->channel_count; ch++) {
      int16_t channel_samples[QOAMAX_MAX_LOOKAHEAD_SLICES * 20];
      for (int i = 0; i < channel_samples_size; i++) channel_samples[i] = (*samples)[i * self->channel_count + ch];

      uint64_t slice = qoamax_encode_slice(&self->channels[ch], self->effort, channel_samples_size, channel_samples);
      qoamax__write_u64be(file, slice);
    }

    *samples += slice_size * self->channel_count;
    self->frame_slices -= 1;
    samples_per_channel -= slice_size;
  }

  return 0;
}
