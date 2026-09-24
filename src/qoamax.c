#include <errno.h>
#include <stdbit.h>
#include <string.h>

#include "./qoamax.h"

//! This flag provides a tiny performance improvement,
//! but requires qoa files to be aligned to an 8-byte boundary.
#define QOAMAX_FILE_ALIGNED true

///////////////////////////////////////////////////////////////////////////////
//                        Core De-/Encoding Functions                        //
///////////////////////////////////////////////////////////////////////////////

typedef struct [[gnu::aligned(64)]] {
  uint64_t error;
  qoamax_lms lms;
  uint64_t slices[4]; //!< Maximum lookahead of 80 samples
} qoamax_encoder_node;

static const int16_t qoamax_dqt[16][8] = {
    {1, -1, 3, -3, 5, -5, 7, -7},
    {5, -5, 18, -18, 32, -32, 49, -49},
    {16, -16, 53, -53, 95, -95, 147, -147},
    {34, -34, 113, -113, 203, -203, 315, -315},
    {63, -63, 210, -210, 378, -378, 588, -588},
    {104, -104, 345, -345, 621, -621, 966, -966},
    {158, -158, 528, -528, 950, -950, 1477, -1477},
    {228, -228, 760, -760, 1368, -1368, 2128, -2128},
    {316, -316, 1053, -1053, 1895, -1895, 2947, -2947},
    {422, -422, 1405, -1405, 2529, -2529, 3934, -3934},
    {548, -548, 1828, -1828, 3290, -3290, 5117, -5117},
    {696, -696, 2320, -2320, 4176, -4176, 6496, -6496},
    {868, -868, 2893, -2893, 5207, -5207, 8099, -8099},
    {1064, -1064, 3548, -3548, 6386, -6386, 9933, -9933},
    {1286, -1286, 4288, -4288, 7718, -7718, 12005, -12005},
    {1536, -1536, 5120, -5120, 9216, -9216, 14336, -14336},
};

static const int16_t qoamax_qt[16][3] = {
    {2, 4, 6},          {11, 25, 40},       {34, 74, 121},      {73, 158, 259},
    {136, 294, 483},    {224, 483, 793},    {343, 739, 1213},   {494, 1064, 1748},
    {684, 1474, 2421},  {913, 1967, 3231},  {1188, 2559, 4203}, {1508, 3248, 5336},
    {1880, 4050, 6653}, {2306, 4967, 8159}, {2787, 6003, 9861}, {3328, 7168, 11776},
};

#define qoamax_assume(x)  \
  if (!(x)) unreachable()

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

//! Saturating conversion from int32_t to int16_t.
[[gnu::always_inline]]
static inline int16_t qoamax_clamp_s16(int32_t value) {
  return (((int16_t)value) == value) ? value : (value < 0 ? INT16_MIN : INT16_MAX);
}

//! Returns an error value that has the same sorting behavior as of @ref qoamax_error_squared, but is faster to compute.
//! This is intended to be used when the error result is only used to find a local minima and then immediately
//! discarded.
[[gnu::always_inline]]
static inline uint32_t qoamax_error_monotonic(int16_t actual, int16_t decoded) {
  int32_t diff = ((int32_t)actual) - decoded;
  uint32_t abs = diff < 0 ? -diff : +diff; //< On some platforms, squaring is faster than abs (e.g. Cortex-M0+)
  return abs;
}

//! Returns the squared error value, used to calculate or minimize the RMS error.
[[gnu::always_inline]]
static inline uint32_t qoamax_error_squared(int16_t actual, int16_t decoded) {
  int32_t diff   = ((int32_t)actual) - decoded;
  uint32_t error = ((uint32_t)diff) * ((uint32_t)diff);
  return error;
}

//! Returns the LMS filter's current prediction for the upcoming sample.
[[gnu::always_inline]]
static inline int32_t qoamax_lms_predict_next(const qoamax_lms *self) {
  int32_t prediction = 0;

  #pragma GCC unroll 4
  for (int i = 0; i < 4; i++) {
    prediction += self->weights[i] * self->history[i];
  }

  return prediction >> 13;
}

//! Updates the LMS filter's internal state, influencing the upcoming predictions.
[[gnu::always_inline]]
static inline void qoamax_lms_update(qoamax_lms *self, int16_t sample, int16_t residual) {
  int16_t delta = residual >> 4;

  #pragma GCC unroll 4
  for (int i = 0; i < 4; i++) {
    self->weights[i] += self->history[i] < 0 ? -delta : delta;
  }

  #pragma GCC unroll 3
  for (int i = 0; i < 3; i++) {
    self->history[i] = self->history[i + 1];
  }

  self->history[3] = sample;
}

void qoamax_decode_slice(qoamax_lms *restrict lms, uint64_t slice, int16_t out[static restrict 20]) {
  int scalefactor = slice >> 60;
  qoamax_assume(scalefactor >= 0 && scalefactor <= 15);
  const int16_t *restrict dqt = qoamax_dqt[scalefactor];
  slice <<= 4;

  #pragma GCC unroll 20
  for (int i = 0; i < 20; i++) {
    int residual = slice >> 61;
    qoamax_assume(residual >= 0 && residual <= 7);
    slice <<= 3;

    int16_t dequantized   = dqt[residual];
    int32_t predicted     = qoamax_lms_predict_next(lms);
    int16_t reconstructed = qoamax_clamp_s16(predicted + dequantized);
    qoamax_lms_update(lms, reconstructed, dequantized);
    out[i] = reconstructed;
  }
}

//! Performs a hill-climb search (equivalent to beam search with width=1).
//!
//! The hill-climb search encodes the audio samples by choosing the
//! most promising residual at each point, never reconsidering past choices.
//!
//! The resulting quality is lower than of the reference encoder.
//! This function is primarily used for calculating a quick error estimate,
//! allowing the beam search to ignore most bad options, increasing performance.
//!
//! @returns the cumulative squared error
static uint64_t qoamax_encode_hillclimb_search(qoamax_lms lms[static restrict 1], int samples_count,
                                               const int16_t samples[static restrict 20], int scalefactor,
                                               uint64_t error_hint, uint64_t slice_out[static restrict 1]) {
  qoamax_assume(samples_count >= 1 && samples_count <= 20);
  qoamax_assume(scalefactor >= 0 && scalefactor <= 15);

  const int16_t *dqt = qoamax_dqt[scalefactor];
  const int16_t *qt  = qoamax_qt[scalefactor];

  uint64_t slice = scalefactor;
  uint64_t error = 0;

  #pragma GCC unroll 20
  for (int i = 0; i < samples_count; i++) {
    int32_t predicted    = qoamax_lms_predict_next(lms);
    int16_t sample       = samples[i];
    int32_t residual     = ((int32_t)sample) - predicted;
    int32_t residual_abs = residual < 0 ? -residual : residual;

    int quant = (((residual_abs > qt[0]) + (residual_abs > qt[1]) + (residual_abs > qt[2])) << 1) + (residual <= 0);

    // Decode best sample again to update lms
    qoamax_assume(quant >= 0 && quant <= 7);
    slice = (slice << 3) | quant;

    int16_t dequantized   = dqt[quant];
    int16_t reconstructed = qoamax_clamp_s16(predicted + dequantized);
    uint32_t code_error   = qoamax_error_squared(sample, reconstructed);
    error += code_error;
    if (error >= error_hint) return -1;

    qoamax_lms_update(lms, reconstructed, dequantized);
  }

  slice <<= (20 - samples_count) * 3;
  *slice_out = slice;
  return error;
}

//! Performs or continues a beam search, adding a full 64-bit qoa slice to the encoder nodes
//!(lookahead of <=20 samples).
static int qoamax_encode_beam_search(qoamax_encoder_node *restrict nodes[static restrict 2], int initial_nodes_size,
                                     int width, int slice_pos, int samples_size,
                                     const int16_t samples[static restrict 20], int scalefactor, uint64_t error_hint) {
  constexpr int W = 1024;
  qoamax_assume(samples_size >= 1 && samples_size <= 20);
  qoamax_assume(width > 0 && width <= W);

  uint16_t heap[W]; // Reference-heap with indexes into nodes_src[]

  qoamax_encoder_node *nodes_src = nodes[0], *nodes_dst = nodes[1];
  int nodes_src_size = initial_nodes_size;

  for (int i = 0; i < nodes_src_size; i++) {
    nodes_src[i].slices[slice_pos] = scalefactor;
  }

  for (int i = 0; i < samples_size; i++) {
    int16_t actual_sample = samples[i];

    int nodes_dst_unused = width;
    uint64_t max_error   = error_hint;

    for (int j = 0; j < nodes_src_size; j++) {
      uint64_t current_error      = nodes_src[j].error;
      int32_t predicted           = qoamax_lms_predict_next(&nodes_src[j].lms);
      const int16_t *restrict dqt = qoamax_dqt[scalefactor];

      for (int codepoint = 0; codepoint < 8; codepoint++) {
        int16_t dequantized   = dqt[codepoint];
        int16_t reconstructed = qoamax_clamp_s16(predicted + dequantized);
        uint32_t error        = qoamax_error_squared(actual_sample, reconstructed);
        uint64_t new_error    = current_error + error;

        if (new_error >= max_error) continue;

        int target;
        if (nodes_dst_unused > 0) {
          target = width - nodes_dst_unused;
          nodes_dst_unused -= 1;
          nodes_dst[target].error = new_error;

          // Insert into max heap (still has unused capacity)
          int heap_pos   = target;
          heap[heap_pos] = target;
          while (heap_pos > 0) {
            int parent_pos = (heap_pos - 1) >> 1;
            if (nodes_dst[heap[parent_pos]].error >= nodes_dst[heap[heap_pos]].error) break;

            auto tmp         = heap[parent_pos];
            heap[parent_pos] = heap[heap_pos];
            heap[heap_pos]   = tmp;
            heap_pos         = parent_pos;
          }
        } else if (nodes_dst_unused == 0) {
          target                  = heap[0];
          nodes_dst[target].error = new_error;

          // Replace the max element inside the heap
          int heap_pos = 0;
          while (true) {
            int left = 2 * heap_pos + 1, right = 2 * heap_pos + 2;
            int largest = heap_pos;
            if (left < width && nodes_dst[heap[left]].error > nodes_dst[heap[largest]].error) largest = left;
            if (right < width && nodes_dst[heap[right]].error > nodes_dst[heap[largest]].error) largest = right;
            if (largest == heap_pos) break;

            auto tmp       = heap[largest];
            heap[largest]  = heap[heap_pos];
            heap[heap_pos] = tmp;
            heap_pos       = largest;
          }
        } else {
          unreachable();
        }
        if (nodes_dst_unused == 0) {
          max_error = MIN(max_error, nodes_dst[heap[0]].error);
        }

        for (int k = 0; k < 3; k++) {
          nodes_dst[target].slices[k] = nodes_src[j].slices[k];
        }
        nodes_dst[target].slices[slice_pos] = (nodes_src[j].slices[slice_pos] << 3) | codepoint;
        nodes_dst[target].lms               = nodes_src[j].lms;
        qoamax_lms_update(&nodes_dst[target].lms, reconstructed, dequantized);
      }
    }

    auto tmp       = nodes_dst;
    nodes_dst      = nodes_src;
    nodes_src      = tmp;
    nodes_src_size = width - nodes_dst_unused;
  }

  nodes[0] = nodes_src;
  nodes[1] = nodes_dst;
  return nodes_src_size;
}

static inline uint64_t
qoamax_encode_slice__hillclimb_scalefactor_search(qoamax_lms *lms, int first_scalefactor, int lookahead,
                                                  const int16_t samples[static restrict lookahead], uint64_t error_hint,
                                                  uint64_t *out_slice, uint64_t *out_error) {
  uint64_t best_slice;
  uint64_t best_error;
  qoamax_lms best_lms;

  for (int scalefactor = 0; scalefactor < 31; scalefactor += 1) {
    int chosen_scalefactor = first_scalefactor + ((scalefactor >> 1) ^ -(scalefactor & 1));
    if ((unsigned)chosen_scalefactor >= 16) continue;

    uint64_t local_slice;
    qoamax_lms local_lms = *lms;
    uint64_t local_error = qoamax_encode_hillclimb_search(&local_lms, MIN(20, lookahead), samples, chosen_scalefactor,
                                                          error_hint, &local_slice);
    if (local_error >= error_hint) continue;

    uint64_t total_error = local_error;

    if (lookahead > 20) {
      qoamax_lms recursive_lms = local_lms;
      uint64_t tmp_slice;
      uint64_t tmp_error;
      total_error += qoamax_encode_slice__hillclimb_scalefactor_search(
          &recursive_lms, chosen_scalefactor, lookahead - 20, &samples[20], error_hint - local_error, &tmp_slice,
          &tmp_error);
    }

    if (total_error < error_hint) {
      error_hint = total_error;
      best_error = local_error;
      best_slice = local_slice;
      best_lms   = local_lms;
    }
  }

  *out_slice = best_slice;
  *out_error = best_error;
  *lms       = best_lms;
  return error_hint;
}

static inline void qoamax_encode_slice__beam_scalefactor_search(qoamax_encoder_node **nodes, int32_t *nodes_sizes,
                                                                int beam_width, int depth, int first_scalefactor,
                                                                int lookahead,
                                                                const int16_t samples[static restrict lookahead],
                                                                uint64_t error_hint, qoamax_encoder_node *out_node) {
  for (int scalefactor = 0; scalefactor < 31; scalefactor += 1) {
    int chosen_scalefactor = first_scalefactor + ((scalefactor >> 1) ^ -(scalefactor & 1));
    if ((unsigned)chosen_scalefactor >= 16) continue;

    // Copy initial state (previous recursion)
    memcpy(nodes[depth], nodes[depth - 1], sizeof(qoamax_encoder_node) * nodes_sizes[depth - 1]);
    nodes_sizes[depth] = nodes_sizes[depth - 1];

    // TODO: Narrow error hint down
    if (lookahead <= 20) {
      // uint64_t local_slice;
      // qoamax_lms local_lms = *lms;
      // uint64_t local_error = qoamax_encode_hillclimb_search(&local_lms, MIN(20, lookahead), samples,
      // chosen_scalefactor,
      //                                                 error_hint, &local_slice);
    }

    nodes_sizes[depth] = qoamax_encode_beam_search(&nodes[depth], nodes_sizes[depth], beam_width, depth,
                                                   MIN(20, lookahead), samples, scalefactor, error_hint);
    if (nodes_sizes[depth] == 0) continue;

    int local_best_index      = 0;
    uint64_t local_best_error = -1;
    for (int i = 0; i < nodes_sizes[depth]; i++) {
      uint64_t error = nodes[depth][i].error;
      if (error >= local_best_error) continue;
      local_best_index = i;
      local_best_error = error;
    }

    if (local_best_error >= error_hint) continue;

    uint64_t total_error = local_best_error;

    if (lookahead > 20) {
      qoamax_encoder_node tmp_node;
      qoamax_encode_slice__beam_scalefactor_search(nodes, nodes_sizes, beam_width, depth + 1, chosen_scalefactor,
                                                   lookahead - 20, &samples[20], error_hint, &tmp_node);
      total_error = tmp_node.error;
    }

    if (total_error < error_hint) {
      error_hint = total_error;
      *out_node  = nodes[depth][local_best_index];
    }
  }
}

uint64_t qoamax_encode_slice(qoamax_encoder_channel *restrict channel, qoamax_encoder_effort effort, int lookahead,
                             const int16_t samples[static restrict lookahead]) {
  qoamax_assume(lookahead >= 1);

  const int slice_count = (effort & 3) + 1;
  const int beam_width  = effort >> 2;

  // // Fast hillclimb algorithm
  // if (beam_width == 0 && slice_count == 1) {
  //   uint64_t best_slice;
  //   uint64_t best_error = -1;
  //   qoamax_lms best_lms;

  //   for (int scalefactor = 0; scalefactor < 31; scalefactor += 1) {
  //     int chosen_scalefactor = prev_scalefactor + ((scalefactor >> 1) ^ -(scalefactor & 1));
  //     if ((unsigned)chosen_scalefactor >= 16) continue;
  //     // for(int scalefactor = 0; scalefactor <= 2; scalefactor += 2) {
  //     //   int chosen_scalefactor = prev_scalefactor + scalefactor - 1;

  //     qoamax_lms hillclimb_lms = channel->lms;
  //     uint64_t hillclimb_slice;
  //     uint64_t error = qoamax_encode_hillclimb_search(&hillclimb_lms, MIN(20, lookahead), samples,
  //     chosen_scalefactor,
  //                                                     best_error, &hillclimb_slice);

  //     if (error < best_error) {
  //       best_error = error;
  //       best_slice = hillclimb_slice;
  //       best_lms   = hillclimb_lms;
  //     }
  //   }

  //   channel->lms              = best_lms;
  //   channel->prev_scalefactor = best_slice >> 60;
  //   channel->error += best_error;
  //   return best_slice;
  // }

  if (beam_width == 0) {
    uint64_t slice;
    uint64_t error;
    qoamax_encode_slice__hillclimb_scalefactor_search(&channel->lms, channel->prev_scalefactor,
                                                      MIN(lookahead, 20 * slice_count), samples, -1, &slice, &error);

    channel->prev_scalefactor = slice >> 60;
    channel->error += error;
    return slice;
  } else {
    constexpr int W = 1024;
    constexpr int L = 5;

    qoamax_encoder_node nodes_storage[W * L + 1];
    qoamax_encoder_node *nodes[L + 1];
    int nodes_sizes[L];

    // Initialize encoder nodes
    for (int i = 1; i <= L; i++) nodes[i] = &nodes_storage[i * beam_width];

    // Initial node
    nodes_storage[W * L].error = 0;
    nodes_storage[W * L].lms   = channel->lms;
    nodes[0]                   = &nodes_storage[W * L];
    nodes_sizes[0]             = 1;

    qoamax_encoder_node best_node;
    qoamax_encode_slice__beam_scalefactor_search(&nodes[1], &nodes_sizes[1], beam_width, 0, channel->prev_scalefactor,
                                                 MIN(lookahead, 20 * slice_count), samples, -1, &best_node);

    int16_t out_samples[20];
    qoamax_decode_slice(&channel->lms, best_node.slices[0], out_samples);

    for (int i = 0; i < MIN(lookahead, 20); i++) {
      channel->error += qoamax_error_squared(samples[i], out_samples[i]);
    }

    channel->prev_scalefactor = best_node.slices[0] >> 60;
    return best_node.slices[0];
  }
}

///////////////////////////////////////////////////////////////////////////////
//                                File Format                                //
///////////////////////////////////////////////////////////////////////////////

static inline uint64_t qoamax__read_u64be(const char *data) {
#if QOAMAX_FILE_ALIGNED && __STDC_ENDIAN_NATIVE__ == __STDC_ENDIAN_LITTLE__
  return __builtin_bswap64(*((uint64_t *)data));
#elif QOAMAX_FILE_ALIGNED && __STDC_ENDIAN_NATIVE__ == __STDC_ENDIAN_BIG__
  return *((uint64_t *)data);
#else
  uint64_t ret = 0;
  for (int i = 0; i < 8; i++) ret = (ret << 8) | ((uint8_t)(*data++));
  return ret;
#endif
}

static inline void qoamax__write_u64be(char *out, uint64_t chunk) {
#if QOAMAX_FILE_ALIGNED && __STDC_ENDIAN_NATIVE__ == __STDC_ENDIAN_LITTLE__
  *((uint64_t *)out) = __builtin_bswap64(chunk);
#elif QOAMAX_FILE_ALIGNED && __STDC_ENDIAN_NATIVE__ == __STDC_ENDIAN_BIG__
  *((uint64_t *)out) = chunk;
#else
  for (int i = 0; i < 8; i++) {
    *out++ = chunk >> 56;
    chunk <<= 8;
  }
#endif
}

static int qoamax_decoder_read_file_header(qoamax_decoder *restrict self, const char *restrict *restrict header,
                                           uint32_t *total_sample_count) {
  uint64_t file_header = qoamax__read_u64be(*header);
  *header += 8;

  if ((file_header >> 32) != 0x716F6166) return -EINVAL;

  *total_sample_count = (uint32_t)file_header;

  return 0;
}

static int qoamax_decoder_read_frame_header(qoamax_decoder *restrict self, const char *restrict *restrict header) {
  uint64_t frame_header = qoamax__read_u64be(*header);
  *header += 8;

  uint8_t channel_count        = frame_header >> 56;
  uint32_t samplerate          = frame_header << 8 >> 40;
  uint32_t samples_per_channel = frame_header << 32 >> 48;
  uint32_t frame_bytesize      = frame_header << 48 >> 48;

  if (channel_count != self->channel_count) return -EINVAL;
  if (samplerate != self->samplerate) return -EINVAL;

  qoamax_assume(channel_count >= 1 && channel_count <= QOAMAX_MAX_CHANNEL_COUNT);

  int32_t total_slices = frame_bytesize / 8 - (2 * channel_count) - 1;
  self->frame_samples  = samples_per_channel;
  self->slice_pos      = total_slices / channel_count;

  if (total_slices != self->slice_pos * channel_count) return -EINVAL;
  if (samples_per_channel > (self->slice_pos * 20)) return -EINVAL;

  for (int i = 0; i < channel_count; i++) {
    uint64_t history = qoamax__read_u64be(*header);
    *header += 8;
    uint64_t weights = qoamax__read_u64be(*header);
    *header += 8;

    for (int j = 0; j < 4; j++) {
      self->lms[i].history[j] = (int16_t)(history >> 48);
      history <<= 16;
    }
    for (int j = 0; j < 4; j++) {
      self->lms[i].weights[j] = (int16_t)(weights >> 48);
      weights <<= 16;
    }
  }

  return 0;
}

int qoamax_decoder_init(qoamax_decoder *restrict self, int file_size, const char *restrict *restrict file,
                        qoamax_decoder_query *query) {
  uint32_t total_sample_count;
  int err = qoamax_decoder_read_file_header(self, file, &total_sample_count);
  if (err) return err;

  uint64_t frame_header = qoamax__read_u64be(*file);

  uint8_t channel_count        = frame_header >> 56;
  uint32_t samplerate          = frame_header << 8 >> 40;
  uint32_t samples_per_channel = frame_header << 32 >> 48;
  uint32_t frame_bytesize      = frame_header << 48 >> 48;

  if (channel_count > QOAMAX_MAX_CHANNEL_COUNT) return -ENOMEM;
  if (channel_count < 1) return -EINVAL;

  query->total_samples = total_sample_count;
  query->samplerate    = samplerate;
  query->channel_count = channel_count;
  self->slice_pos      = 0;
  self->channel_count  = channel_count;
  self->samplerate     = samplerate;
  (void)samples_per_channel;
  (void)frame_bytesize;

  return 0;
}

int qoamax_decoder_read(qoamax_decoder *restrict self, int file_size, const char *restrict *restrict file,
                        size_t samples_per_channel, int16_t *restrict samples[static restrict self->channel_count]) {
  qoamax_assume(self->channel_count >= 1 && self->channel_count <= QOAMAX_MAX_CHANNEL_COUNT);

  const char *file_end = *file + file_size;

  while (true) {
    if (self->slice_pos == 0) {
      if (*file + 8 * (1 + self->channel_count) > file_end) return 0;

      int err = qoamax_decoder_read_frame_header(self, file);
      if (err) return err;
    } else {
      int slice_size = MIN(20, self->frame_samples);
      if (((file_end - *file) / 8) < self->channel_count) return 0;
      if (samples_per_channel < slice_size) return 0;

      self->slice_pos -= 1;
      self->frame_samples -= slice_size;
      samples_per_channel -= slice_size;

      for (int channel = 0; channel < self->channel_count; channel++) {
        uint64_t slice = qoamax__read_u64be(*file);
        *file += 8;
        qoamax_decode_slice(&self->lms[channel], slice, samples[channel]);
        samples[channel] += slice_size;
      }
    }
  }
}

int qoamax_decoder_read_interleaved(qoamax_decoder *restrict self, int file_size, const char *restrict *restrict file,
                                    size_t samples_size, int16_t *restrict samples[static restrict 1]) {
  if (self->channel_count == 1) return qoamax_decoder_read(self, file_size, file, samples_size, samples);

  qoamax_assume(self->channel_count >= 1 && self->channel_count <= QOAMAX_MAX_CHANNEL_COUNT);

  const char *file_end    = *file + file_size;
  int samples_per_channel = samples_size / self->channel_count;

  while (true) {
    if (self->slice_pos == 0) {
      if ((*file + 8 * (1 + self->channel_count)) > file_end) return 0;

      int err = qoamax_decoder_read_frame_header(self, file);
      if (err) return err;
    } else {
      int slice_size = MIN(20, self->frame_samples);
      if (((file_end - *file) / 8) < self->channel_count) return 0;
      if (samples_per_channel < slice_size) return 0;

      self->slice_pos -= 1;
      self->frame_samples -= slice_size;
      samples_per_channel -= slice_size;

      int16_t interleaving[20 * QOAMAX_MAX_CHANNEL_COUNT];

      int16_t *interleaving_pos = interleaving;
      for (int channel = 0; channel < self->channel_count; channel++) {
        uint64_t slice = qoamax__read_u64be(*file);
        *file += 8;
        qoamax_decode_slice(&self->lms[channel], slice, interleaving_pos);
        interleaving_pos += 20;
      }

      for (int i = 0; i < slice_size; i++) {
        interleaving_pos = interleaving + i;
        for (int channel = 0; channel < self->channel_count; channel++) {
          *(*samples)++ = *interleaving_pos;
          interleaving_pos += 20;
        }
      }
    }
  }
}

int qoamax_encoder_init(qoamax_encoder *restrict self, qoamax_encoder_effort effort, int channel_count,
                        uint32_t total_samples, int32_t samplerate) {
  if (channel_count < 1 || channel_count > 8) return -EINVAL;
  if ((samplerate >> 24) != 0) return -EINVAL;

  self->effort         = effort;
  self->channel_count  = channel_count;
  self->total_samples  = total_samples;
  self->samplerate     = samplerate;
  self->slice_pos      = 0;
  self->header_written = false;

  for (int i = 0; i < channel_count; i++) {
    self->channels[i].lms              = QOAMAX_LMS_INITIAL;
    self->channels[i].error            = 0;
    self->channels[i].prev_scalefactor = 0;
    for (int j = 0; j < 3; j++) self->channels[i].cache[j] = 0;
  }

  return 0;
}

static void qoamax_encoder_write_file_header(const qoamax_encoder *restrict self, char *restrict *restrict out) {
  uint64_t file_header = (0x716F6166LLU << 32) | (self->total_samples << 0);
  qoamax__write_u64be(*out, file_header);
  *out += 8;
}

static void qoamax_encoder_write_frame_header(qoamax_encoder *restrict self, char *restrict *restrict out,
                                              int samples_per_channel) {
  uint64_t frame_header = (((uint64_t)self->channel_count) << 56) | (((uint64_t)self->samplerate) << 32) |
                          (((uint64_t)samples_per_channel) << 16) |
                          (((1 + self->channel_count * (self->slice_pos + 2)) * 8) << 0);
  qoamax__write_u64be(*out, frame_header);
  *out += 8;

  for (int ch = 0; ch < self->channel_count; ch++) {
    qoamax_lms *lms = &self->channels[ch].lms;

    uint64_t channel_history;
    for (int i = 0; i < 4; i++) channel_history = (channel_history << 16) | ((uint16_t)lms->history[i]);
    qoamax__write_u64be(*out, channel_history);
    *out += 8;

    uint64_t channel_weights;
    for (int i = 0; i < 4; i++) {
      lms->weights[i] = (lms->weights[i] << 16 >> 16);
      channel_weights = (channel_weights << 16) | ((uint16_t)lms->weights[i]);
    }
    qoamax__write_u64be(*out, channel_weights);
    *out += 8;
  }
}

int qoamax_encoder_write(qoamax_encoder *restrict self, size_t file_size, char *restrict *restrict file,
                         int samples_per_channel, const int16_t *restrict samples[static restrict self->channel_count],
                         bool samples_complete) {
  qoamax_assume(samples_per_channel >= 0);

  int lookahead   = qoamax_encoder_effort_get_lookahead(self->effort);
  int min_samples = samples_complete ? 1 : lookahead;

  char *file_end = *file + file_size;

  while (true) {
    if (self->slice_pos == 0) {
      if (!self->header_written) {
        if ((file_end - *file) < 8) return 0;
        qoamax_encoder_write_file_header(self, file);
        self->header_written = true;
      }

      if ((file_end - *file) < 8 * (1 + self->channel_count * 2)) return 0;
      if (samples_per_channel < 5120 && (!samples_complete || samples_per_channel <= 0)) return 0;
      int upcoming_samples = MIN(samples_per_channel, 5120);
      self->slice_pos      = (upcoming_samples + 19) / 20;

      qoamax_encoder_write_frame_header(self, file, upcoming_samples);
    } else {
      if (((file_end - *file) / 8) < self->channel_count) return 0;
      if (samples_per_channel < min_samples) return 0;

      for (int ch = 0; ch < self->channel_count; ch++) {
        uint64_t slice =
            qoamax_encode_slice(&self->channels[ch], self->effort, MIN(lookahead, samples_per_channel), samples[ch]);
        samples[ch] += MIN(20, samples_per_channel);

        qoamax__write_u64be(*file, slice);
        *file += 8;
      }

      self->slice_pos -= 1;
      samples_per_channel -= 20;
    }
  }

  return 0;
}

int qoamax_encoder_write_interleaved(qoamax_encoder *restrict self, size_t file_size, char *restrict *restrict file,
                                     int samples_size, const int16_t *restrict samples[static restrict 1],
                                     bool samples_complete) {
  qoamax_assume(samples_size >= 0);
  if (self->channel_count == 1)
    return qoamax_encoder_write(self, file_size, file, samples_size, samples, samples_complete);

  int lookahead           = qoamax_encoder_effort_get_lookahead(self->effort);
  int min_samples         = samples_complete ? 1 : lookahead;
  int samples_per_channel = samples_size / self->channel_count;

  char *file_end = *file + file_size;

  while (true) {
    if (self->slice_pos == 0) {
      if (!self->header_written) {
        if ((file_end - *file) < 8) return 0;
        qoamax_encoder_write_file_header(self, file);
        self->header_written = true;
      }

      if ((file_end - *file) < 8 * (1 + self->channel_count * 2)) return 0;
      if (samples_per_channel < 5120 && (!samples_complete || samples_per_channel <= 0)) return 0;
      int upcoming_samples = MIN(samples_per_channel, 5120);
      self->slice_pos      = (upcoming_samples + 19) / 20;

      qoamax_encoder_write_frame_header(self, file, upcoming_samples);
    } else {
      if (((file_end - *file) / 8) < self->channel_count) return 0;
      if (samples_per_channel < min_samples) return 0;

      for (int ch = 0; ch < self->channel_count; ch++) {
        int16_t channel_samples[80];
        const int16_t *samples_channel = &(*samples)[ch];

        for (int i = 0; i < MIN(lookahead, samples_per_channel); i++) {
          channel_samples[i] = *samples_channel;
          samples_channel += self->channel_count;
        }

        uint64_t slice = qoamax_encode_slice(&self->channels[ch], self->effort, MIN(lookahead, samples_per_channel),
                                             channel_samples);
        qoamax__write_u64be(*file, slice);
        *file += 8;
      }

      *samples += MIN(20, samples_per_channel) * self->channel_count;
      self->slice_pos -= 1;
      samples_per_channel -= 20;
    }
  }
}
