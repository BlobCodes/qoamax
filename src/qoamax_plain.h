// Plain declarations of the qoamax API's functions, included by qoamax.h for C++ and for C compilers without variably
// modified types (__STDC_NO_VLA__, optional in C11, e.g. MSVC). Include qoamax.h, not this file.
//
// The functions are the same as in qoamax.h, which also holds their documentation, but without [static n] array
// parameters, and without restrict in C++, which doesn't have it. In C, the declarations must stay compatible with
// the ones in qoamax.h, so they keep restrict below the top level (make test compiles qoamax.c against them).
// Keep both in sync.
#pragma once
#include "qoamax.h"

#ifdef __cplusplus
#define QOAMAX_RESTRICT
extern "C" {
#else
#define QOAMAX_RESTRICT restrict
#endif

////////////////////
//      Core      //
////////////////////

void qoamax_decode_slice(qoamax_lms *lms, uint64_t slice, int16_t out[20]);

uint64_t qoamax_encode_slice(qoamax_encoder_channel *channel, qoamax_encoder_effort effort, int samples_size,
                             const int16_t *samples);

static inline int qoamax_encoder_effort_get_lookahead(qoamax_encoder_effort effort) { return ((effort & 3) + 1) * 20; }

////////////////////
//    Decoder     //
////////////////////

int qoamax_decoder_init(qoamax_decoder *self, size_t file_size, const char *QOAMAX_RESTRICT *file,
                        qoamax_decoder_query *query);

int qoamax_decoder_read(qoamax_decoder *self, size_t file_size, const char *QOAMAX_RESTRICT *file,
                        size_t samples_per_channel, int16_t *QOAMAX_RESTRICT *samples);

int qoamax_decoder_read_interleaved(qoamax_decoder *self, size_t file_size, const char *QOAMAX_RESTRICT *file,
                                    size_t samples_size, int16_t *QOAMAX_RESTRICT *samples);

////////////////////
//    Encoder     //
////////////////////

static inline uint64_t qoamax_encoder_get_encoded_size(int channel_count, uint32_t samples_per_channel) {
  uint64_t frames = ((uint64_t)samples_per_channel + QOAMAX_FRAME_SAMPLES - 1) / QOAMAX_FRAME_SAMPLES;
  uint64_t slices = ((uint64_t)samples_per_channel + 19) / 20; // per channel

  // The file header, a frame header plus every channel's LMS state (history, weights) per frame, and the slices.
  return 8 * (1 + frames * (1 + 2 * channel_count) + slices * channel_count);
}

int qoamax_encoder_init(qoamax_encoder *self, qoamax_encoder_effort effort, int channel_count, uint32_t total_samples,
                        int32_t samplerate);

int qoamax_encoder_write(qoamax_encoder *self, size_t file_size, char *QOAMAX_RESTRICT *file,
                         size_t samples_per_channel, const int16_t *QOAMAX_RESTRICT *samples, bool samples_complete);

int qoamax_encoder_write_interleaved(qoamax_encoder *self, size_t file_size, char *QOAMAX_RESTRICT *file,
                                     size_t samples_size, const int16_t *QOAMAX_RESTRICT *samples,
                                     bool samples_complete);

static inline void qoamax_encoder_get_error(const qoamax_encoder *self, uint64_t *errors) {
  for (int i = 0; i < self->channel_count; i++) errors[i] = self->channels[i].error;
}

#ifdef __cplusplus
} // extern "C"
#endif
#undef QOAMAX_RESTRICT
