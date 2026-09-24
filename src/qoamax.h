#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define QOAMAX_MAX_CHANNEL_COUNT 8

typedef struct {
  int16_t history[4];
  int32_t weights[4];
} qoamax_lms;

typedef struct {
  qoamax_lms lms[QOAMAX_MAX_CHANNEL_COUNT];
  uint8_t channel_count;
  uint16_t frame_samples;
  int slice_pos;
  int samplerate;
} qoamax_decoder;

typedef struct {
  uint32_t total_samples;
  int32_t samplerate;
  uint8_t channel_count;
} qoamax_decoder_query;

typedef enum {
  QOAMAX_ENCODER_EFFORT_SUPERFAST = 0,                //!< Similar quality to reference encoder
  QOAMAX_ENCODER_EFFORT_VERYFAST  = 1,                //!< Moderate quality gains at small overhead
  QOAMAX_ENCODER_EFFORT_FAST      = 2,                //!< Moderate quality gains at medium overhead
  QOAMAX_ENCODER_EFFORT_MEDIUM    = 3,                //!< Significant intra-slice quality gains
  QOAMAX_ENCODER_EFFORT_SLOW      = 1 | (64 << 2),   //!< Increased horizon,
  QOAMAX_ENCODER_EFFORT_VERYSLOW  = 1 | (1024 << 2),  //!< Large horizon,
  QOAMAX_ENCODER_EFFORT_PLACEBO   = 3 | (1024 << 2),  //!< Maximum quality the current implementation can achieve
} qoamax_encoder_effort;

typedef struct {
  qoamax_lms lms;
  uint8_t prev_scalefactor;
  uint64_t cache[3];
  uint64_t error;
} qoamax_encoder_channel;

typedef struct {
  qoamax_encoder_effort effort;
  uint32_t total_samples;
  int32_t samplerate;
  int16_t slice_pos;
  uint8_t channel_count;
  bool header_written;

  qoamax_encoder_channel channels[QOAMAX_MAX_CHANNEL_COUNT];
} qoamax_encoder;

[[maybe_unused]] static const qoamax_lms QOAMAX_LMS_INITIAL = {
    .history = {0, 0, 0, 0},
    .weights = {0, 0, -(1 << 13), (1 << 14)},
};

////////////////////
//      Core      //
////////////////////

//! The core slice decoding function used in all other decoding mechanisms.
//! Decodes a 64-bit slice into 20 audio samples, modifying the LMS filter's state.
//!
//! The function can be used to implement custom file formats, streaming, etc..
void qoamax_decode_slice(qoamax_lms *restrict lms, uint64_t slice, int16_t out[static restrict 20]) [[unsequenced]];

//! The core slice encoding function used in all other encoding mechanisms.
//! Encodes the first 20 audio samples into a 64-bit slice, modifying the LMS filter's state.
//!
//! The function can be used to implement custom file formats, for streaming, etc..
//!
//! The required amount of samples depends on the audio position and encoder effort.
//! The encoder should always receive the lookahead required
//! by the encoder effort @ref qoamax_encoder_effort_get_lookahead.
//! Using a smaller samples_size is only allowed if the audio stream is about to end
//! and thus has fewer samples than would be required by the effort.
//!
//! @returns the encoded slice.
uint64_t qoamax_encode_slice(qoamax_encoder_channel *restrict channel, qoamax_encoder_effort effort, int samples_size,
                             const int16_t samples[static restrict samples_size])
    [[unsequenced]];

//! Returns the amount of continuous samples being looked at to produce a 64-bit slice for this effort option.
[[maybe_unused]] static inline int qoamax_encoder_effort_get_lookahead(qoamax_encoder_effort effort) {
  return ((effort & 3) + 1) * 20;
}

////////////////////
//    Decoder     //
////////////////////

//! Initializes the decoder, reading the file's header and returning playback information.
//!
//! @param self      The decoder, does not need to be zeroed.
//! @param file_size The amount of bytes that can be read from *file. Must be >=16.
//! @param file      Mutable pointer to the file's contents.
//! @param query     Destiny to write queried information (not null).
//!
//! @returns 0 on success
int qoamax_decoder_init(qoamax_decoder *restrict self, int file_size, const char *restrict *restrict file,
                        qoamax_decoder_query *query) [[unsequenced]];

//! Continues decoding the given file, writing as many decoded samples as possible into the output buffers.
//!
//! @param self                The decoder.
//! @param file_size           The amount of bytes that can be read from *file. Must be >= 8 * channel_count.
//! @param file                Mutable pointer to the file's contents.
//! @param samples_per_channel The amount of samples that can be written into each out buffer.
//! @param out                 Array of mutable pointers to individual buffers for each channel.
//!
//! @returns 0 on success
int qoamax_decoder_read(qoamax_decoder *restrict self, int file_size, const char *restrict *restrict file,
                        size_t samples_per_channel, int16_t *restrict samples[static restrict self->channel_count])
    [[unsequenced]];

//! Continues decoding the given file, writing as many decoded samples as possible into the interleaved output buffer.
//!
//! This function is slower than the non-interleaved function @ref qoamax_decoder_read.
//!
//! @param self         The decoder.
//! @param file_size    The amount of bytes that can be read from *file. Must be >= 8 * channel_count.
//! @param file         Mutable pointer to the file's contents.
//! @param samples_size The amount of samples that can be written into each out buffer.
//! @param out          Mutable pointer to interleaved audio buffer.
//!
//! @returns 0 on success
int qoamax_decoder_read_interleaved(qoamax_decoder *restrict self, int file_size, const char *restrict *restrict file,
                                    size_t samples_size, int16_t *restrict samples[static restrict 1]) [[unsequenced]];

////////////////////
//    Encoder     //
////////////////////

//! Returns the final encoded bytesize for an audio stream with the given length and channel count.
//!
//! NOTE: channel_count must be between 1 and 8, samples_per_channel must be > 0.
//!       The result is undefined otherwise.
[[maybe_unused]] static inline int32_t qoamax_encoder_get_encoded_size(int32_t channel_count,
                                                                       int32_t samples_per_channel) {
  return 8 * (1 + ((samples_per_channel + 5119) / 5120) + channel_count * ((samples_per_channel + 19) / 20 + 2 * ((samples_per_channel + 5119) / 5120)));
}

//! Initializes the encoder state, supplying it with the information
//! required to produce headers and prepare the encoder's state.
//!
//! @param
//! @param effort        The chosen performance/quality tradeoff.
//! @param channel_count The file's channel count (range 1..8).
//! @param total_samples Audio size hint written into the file's header, may be 0 for streaming.
//! @param samplerate    Samplerate hint written into the file's header, may be 0 (not recommended).
//!
//! @returns 0 on success
int qoamax_encoder_init(qoamax_encoder *self, qoamax_encoder_effort effort, int channel_count, uint32_t total_samples,
                        int32_t samplerate) [[unsequenced]];

//! Encodes as many audio samples as possible using the given output and input buffers.
//!
//! Depending on the chosen encoder effort, the encoder requires different amounts of lookahead.
//! See @ref qoamax_encoder_effort_get_lookahead to find out how large the input buffer needs to be.
//!
//! Note that the input/output buffer pointers are given by non-const references,
//! since the encoder moves them to signal encoder progress.
//!
//! @param file_size           Amount of writable bytes in *file.
//! @param file                Pointer to output buffer pointer.
//! @param samples_per_channel Size of a sample buffer pointed to from samples[].
//! @param samples             Array of audio sample buffers (one continuous sample buffer per channel).
//! @param samples_complete    Whether the given samples array has reached the audio file's end.
//!
//! @returns 0 on success
int qoamax_encoder_write(qoamax_encoder *restrict self, size_t file_size, char *restrict *restrict file,
                         int samples_per_channel, const int16_t *restrict samples[static restrict self->channel_count],
                         bool samples_complete) [[unsequenced]];

//! Encodes as many audio samples as possible using the given output and input buffers.
//!
//! Depending on the chosen encoder effort, the encoder requires different amounts of lookahead.
//! See @ref qoamax_encoder_effort_get_lookahead to find out how large the input buffer needs to be.
//!
//! Note that the input/output buffer pointers are given by non-const references,
//! since the encoder moves them to signal encoder progress.
//!
//! @param file_size           Amount of writable bytes in *file.
//! @param file                Pointer to output buffer pointer.
//! @param samples_per_channel Size of a sample buffer pointed to from samples[].
//! @param samples             Pointer to interleaved audio sample buffer.
//! @param samples_complete    Whether the given samples array has reached the audio file's end.
//!
//! @returns 0 on success
int qoamax_encoder_write_interleaved(qoamax_encoder *restrict self, size_t file_size, char *restrict *restrict file,
                                     int samples_size, const int16_t *restrict samples[static restrict 1],
                                     bool samples_complete) [[unsequenced]];

//! Returns the cumulated squared error up to the current encoding position for each channel.
[[maybe_unused]] static inline void qoamax_encoder_get_error(qoamax_encoder *self,
                                                             uint64_t errors[static restrict self->channel_count]) {
  for (int i = 0; i < self->channel_count; i++) errors[i] = self->channels[i].error;
}
