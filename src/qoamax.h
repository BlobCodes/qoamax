// The header is C11 (and C++11) without extensions. Only qoamax.c needs C23.
#pragma once
#ifndef __cplusplus
#include <stdalign.h>
#endif
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define QOAMAX_VERSION_MAJOR 0
#define QOAMAX_VERSION_MINOR 2
#define QOAMAX_VERSION_PATCH 0

#define QOAMAX_MAX_CHANNEL_COUNT 8
#define QOAMAX_FRAME_SAMPLES 5120 //!< Samples per channel in a full frame

#define QOAMAX_ERROR_INVALID_ARGUMENT -1
#define QOAMAX_ERROR_INVALID_FILE_HEADER -2
#define QOAMAX_ERROR_INVALID_FRAME_HEADER -3
#define QOAMAX_ERROR_SAMPLE_FORMAT_BREAK -4
#define QOAMAX_ERROR_UNSUPPORTED_CHANNEL_COUNT -5

typedef struct {
  // 8-byte aligned, so that RISC-V can load and store the state with 64-bit accesses. This is part of the ABI.
  alignas(8) int16_t history[4];
  int32_t weights[4];
} qoamax_lms;

typedef struct {
  int32_t samplerate;
  uint16_t frame_samples; //!< Samples per channel left in the current frame (0 = next read is a frame header).
  uint8_t channel_count;
  qoamax_lms lms[QOAMAX_MAX_CHANNEL_COUNT];
} qoamax_decoder;

typedef struct {
  uint32_t total_samples;
  int32_t samplerate;
  uint8_t channel_count;
} qoamax_decoder_query;

//! The encoder effort enum specifies which algorithm and what
//! parameters will be used for encoding the audio data.
//!
//! The available choices range from being as fast as the reference decoder
//! to taking multiple hours to encode a single minute of audio.
//!
//! A value packs the lookahead in slices minus one (bits 0-1) and the beam width (the remaining bits, 0 = hill-climb,
//! negative = ULTRAFAST).
typedef enum {
  QOAMAX_ENCODER_EFFORT_ULTRAFAST = -4,              //!< Roughly 1.0dB worse than SUPERFAST, but about 3x as fast
                                                     //!  (with SIMD). Chooses the scalefactor from an estimate and
                                                     //!  encodes several frames at once.
  QOAMAX_ENCODER_EFFORT_SUPERFAST = 0,               //!< Similar quality to reference encoder.
  QOAMAX_ENCODER_EFFORT_VERYFAST  = 1,               //!< Roughly 0.5dB better than SUPERFAST by increasing the
                                                     //!  lookahead to 40 samples when encoding scalefactors.
  QOAMAX_ENCODER_EFFORT_FAST      = 0 | (2 << 2),    //!< Roughly 1.0dB better than SUPERFAST
                                                     //!  by greedily encoding residuals.
  QOAMAX_ENCODER_EFFORT_MEDIUM    = 0 | (4 << 2),    //!< Roughly 1.5dB better than SUPERFAST
                                                     //!  by more greedily encoding residuals.
  QOAMAX_ENCODER_EFFORT_SLOW      = 1 | (4 << 2),    //!< Roughly 2.0dB better than SUPERFAST by greedily
                                                     //!  encoding both the residuals and the scalefactor.
  QOAMAX_ENCODER_EFFORT_VERYSLOW  = 1 | (16 << 2),   //!< Roughly 2.5dB better than SUPERFAST
                                                     //!  by more greedily encoding residuals.
  QOAMAX_ENCODER_EFFORT_SUPERSLOW = 2 | (24 << 2),   //!< Roughly 0.15dB better than VERYSLOW by increasing the
                                                     //!  lookahead to 60 samples and the beam width. 4-5x slower.
  QOAMAX_ENCODER_EFFORT_PLACEBO   = 3 | (1024 << 2), //!< Roughly 3.0dB better than SUPERFAST.
                                                     //!  Maximum quality the current implementation can achieve.
                                                     //!  Can take multiple hours to encode a single minute!
} qoamax_encoder_effort;

typedef struct {
  qoamax_lms lms;
  uint8_t prev_scalefactor;
  uint64_t error; //!< Cumulative squared error of the encoded samples
} qoamax_encoder_channel;

typedef struct {
  qoamax_encoder_effort effort;
  uint32_t total_samples;
  int32_t samplerate;
  uint16_t frame_slices; //!< Slices per channel left in the current frame (0 = next write starts a frame).
  uint8_t channel_count;
  bool header_written;
  uint32_t frame_index; //!< Number of frames started so far (wraps around)

  qoamax_encoder_channel channels[QOAMAX_MAX_CHANNEL_COUNT];

  // ULTRAFAST encodes groups of frames independently of each other (see qoamax.c), which start with these values:
  int16_t group_weights[QOAMAX_MAX_CHANNEL_COUNT][4]; //!< The LMS weights at the end of the previous group
  int16_t last_samples[QOAMAX_MAX_CHANNEL_COUNT][4];  //!< The last four input samples
} qoamax_encoder;

//! The LMS state at the start of a stream (history, weights).
static const qoamax_lms QOAMAX_LMS_INITIAL = {
    {0, 0, 0, 0},
    {0, 0, -(1 << 13), (1 << 14)},
};

//! One stream of samples for @ref qoamax_encode_streams, which encodes many streams at once: a channel of an audio
//! track, a separate track, or a block of a channel in your own container format (see qoamax_encode_streams).
//! Create it with @ref qoamax_encode_stream_init. The encoder advances `samples` and `slices` as it encodes, so a
//! stream can be passed again with more samples or more room for slices.
typedef struct {
  qoamax_encoder_channel channel; //!< The encoder's state (and the squared error of the samples encoded so far)
  const int16_t *samples;         //!< The next sample to encode; sample i is samples[i * sample_stride]
  size_t samples_size;            //!< The number of samples from `samples` on
  int sample_stride;              //!< 1 for a channel of its own, the channel count for interleaved channels
  uint64_t *slices;               //!< Where the next encoded slice is stored
  size_t slices_size;             //!< Room for slices from `slices` on
} qoamax_encode_stream;

//! One stream of slices for @ref qoamax_decode_streams, which decodes many streams at once (the counterpart of
//! @ref qoamax_encode_stream). Create it with @ref qoamax_decode_stream_init.
typedef struct {
  qoamax_lms lms;         //!< The decoder's state: QOAMAX_LMS_INITIAL, or the state the encoder started with
  const uint64_t *slices; //!< The next slice to decode
  size_t slices_size;     //!< The number of slices from `slices` on
  int16_t *samples;       //!< Where the next decoded sample is stored; sample i is samples[i * sample_stride]
  size_t samples_size;    //!< The number of samples left to decode (room from `samples` on, and the stream's length)
  int sample_stride;      //!< 1 for a channel of its own, the channel count for interleaved channels
} qoamax_decode_stream;

//! Returns a stream that encodes the samples (sample i is samples[i * sample_stride]) from the start of the audio, into
//! room for slices_size slices (one per 20 samples, rounded up, for all of them).
static inline qoamax_encode_stream qoamax_encode_stream_init(const int16_t *samples, size_t samples_size,
                                                             int sample_stride, uint64_t *slices, size_t slices_size) {
  qoamax_encode_stream stream;
  stream.channel.lms              = QOAMAX_LMS_INITIAL;
  stream.channel.prev_scalefactor = 0;
  stream.channel.error            = 0;
  stream.samples                  = samples;
  stream.samples_size             = samples_size;
  stream.sample_stride            = sample_stride;
  stream.slices                   = slices;
  stream.slices_size              = slices_size;
  return stream;
}

//! Returns a stream that decodes the slices of an encoded stream (from its start) into samples_size samples (sample i
//! is stored at samples[i * sample_stride]).
static inline qoamax_decode_stream qoamax_decode_stream_init(const uint64_t *slices, size_t slices_size,
                                                             int16_t *samples, size_t samples_size, int sample_stride) {
  qoamax_decode_stream stream;
  stream.lms           = QOAMAX_LMS_INITIAL;
  stream.slices        = slices;
  stream.slices_size   = slices_size;
  stream.samples       = samples;
  stream.samples_size  = samples_size;
  stream.sample_stride = sample_stride;
  return stream;
}

// C++ has no restrict or [static n] array parameters, and C11 compilers may not support the variable ones
// (__STDC_NO_VLA__, e.g. MSVC), so they get the function declarations from qoamax_plain.h.
#if defined(__cplusplus) || defined(__STDC_NO_VLA__)
#include "qoamax_plain.h"
#else

////////////////////
//      Core      //
////////////////////

//! The core slice decoding function used in all other decoding mechanisms.
//! Decodes a 64-bit slice into 20 audio samples, modifying the LMS filter's state.
//!
//! The function can be used to implement custom file formats, streaming, etc..
void qoamax_decode_slice(qoamax_lms *restrict lms, uint64_t slice, int16_t out[static restrict 20]);

//! The core slice encoding function used in all other encoding mechanisms.
//! Encodes the first 20 audio samples (fewer at the end of the stream) into a 64-bit slice,
//! modifying the LMS filter's state.
//!
//! The function can be used to implement custom file formats, for streaming, etc..
//!
//! The required amount of samples depends on the audio position and encoder effort.
//! The encoder should always receive the lookahead required
//! by the encoder effort @ref qoamax_encoder_effort_get_lookahead.
//! Using a smaller samples_size is only allowed if the audio stream is about to end
//! and thus has fewer samples than would be required by the effort.
//!
//! The encoder effort may not be changed during encoding. Doing so could result in a segmentation fault.
//!
//! @returns the encoded slice.
uint64_t qoamax_encode_slice(qoamax_encoder_channel *restrict channel, qoamax_encoder_effort effort, int samples_size,
                             const int16_t samples[static restrict samples_size]);

//! Encodes many independent streams at once (e.g. the channels of a track), which is faster than encoding them one
//! by one with @ref qoamax_encode_slice: ULTRAFAST encodes several streams in parallel in SIMD lanes (4 streams fill
//! AVX2). Every stream gives the same slices as encoding it slice by slice with qoamax_encode_slice, whichever streams
//! are passed together, in which order, and on any CPU.
//!
//! Each stream encodes as many slices as its samples and its room for slices allow, and advances its `samples` and
//! `slices` past them. The streams may have different lengths.
//! - samples_complete = true: the streams end after their samples_size samples. All of them are encoded; the last slice
//!   may hold fewer than 20 samples.
//! - samples_complete = false: more samples follow later. Only slices whose whole lookahead
//!   (@ref qoamax_encoder_effort_get_lookahead) is available are encoded, so a stream may have samples left. Pass it
//!   again with these and the new samples.
//!
//! Example: encode a stereo track (n interleaved samples per channel) into one array of slices per channel.
//! @code
//! size_t slice_count = (n + 19) / 20;
//! qoamax_encode_stream streams[2];
//! for (int ch = 0; ch < 2; ch++)
//!   streams[ch] = qoamax_encode_stream_init(&samples[ch], n, 2, slices[ch], slice_count);
//! qoamax_encode_streams(QOAMAX_ENCODER_EFFORT_ULTRAFAST, 2, streams, true);
//! @endcode
//!
//! A single channel can also be split into blocks that are encoded at once, if your container stores the LMS state
//! each block starts with (like the QOA frame headers): start the block's stream with channel.lms.history = the 4 input
//! samples before the block and channel.lms.weights = the weights at the end of an earlier block (QOAMAX_LMS_INITIAL's
//! weights cost about 0.7 dB), store that state, and start the block's decoder with it.
//!
//! @param effort       The effort of all streams. It may not change while a stream is encoded.
//! @param stream_count The number of streams, at least 1.
void qoamax_encode_streams(qoamax_encoder_effort effort, size_t stream_count, qoamax_encode_stream streams[stream_count],
                           bool samples_complete);

//! Decodes many independent streams at once, which is faster than decoding them one by one with
//! @ref qoamax_decode_slice: at least 4 streams (and half of the SIMD lanes, 4 of 8 with AVX2) are decoded in parallel.
//!
//! Each stream decodes as many slices as it has, up to its samples_size samples, and advances its `slices` and
//! `samples` past them. If samples_size isn't a multiple of 20, the last slice is decoded only partially (the end of
//! the stream). The streams may have different lengths.
//!
//! Example: decode the slices of a stereo track into interleaved samples.
//! @code
//! qoamax_decode_stream streams[2];
//! for (int ch = 0; ch < 2; ch++)
//!   streams[ch] = qoamax_decode_stream_init(slices[ch], slice_count, &samples[ch], samples_per_channel, 2);
//! qoamax_decode_streams(2, streams);
//! @endcode
//! @param stream_count The number of streams, at least 1.
void qoamax_decode_streams(size_t stream_count, qoamax_decode_stream streams[stream_count]);

//! Returns the amount of continuous samples @ref qoamax_encode_slice looks at to produce a 64-bit slice for this
//! effort option (its samples_size should be at least this large, except at the end of the stream).
static inline int qoamax_encoder_effort_get_lookahead(qoamax_encoder_effort effort) { return ((effort & 3) + 1) * 20; }

////////////////////
//    Decoder     //
////////////////////

//! Initializes the decoder, reading the file's header and returning playback information.
//!
//! @param self      The decoder, does not need to be zeroed.
//! @param file_size The amount of bytes that can be read from *file. Must be >= 16.
//! @param file      Mutable pointer to the file's contents.
//! @param query     Destination to write queried information (not null).
//!
//! @returns 0 on success
int qoamax_decoder_init(qoamax_decoder *restrict self, size_t file_size, const char *restrict *restrict file,
                        qoamax_decoder_query *query);

//! Continues decoding the given file, writing as many decoded samples as possible into the output buffers.
//! Returns once the file or the output buffers run out, advancing *file and the buffer pointers.
//!
//! @param self                The decoder.
//! @param file_size           The amount of bytes that can be read from *file.
//! @param file                Mutable pointer to the file's contents.
//! @param samples_per_channel The amount of samples that can be written into each buffer.
//! @param samples             Array of mutable pointers to individual buffers for each channel.
//!
//! @returns 0 on success
int qoamax_decoder_read(qoamax_decoder *restrict self, size_t file_size, const char *restrict *restrict file,
                        size_t samples_per_channel, int16_t *restrict samples[static restrict self->channel_count]);

//! Continues decoding the given file, writing as many decoded samples as possible into the interleaved output buffer.
//! Returns once the file or the output buffer runs out, advancing *file and *samples.
//!
//! This function is slightly slower than the non-interleaved function @ref qoamax_decoder_read.
//!
//! @param self         The decoder.
//! @param file_size    The amount of bytes that can be read from *file.
//! @param file         Mutable pointer to the file's contents.
//! @param samples_size The amount of total samples that can be written into the buffer.
//! @param samples      Mutable pointer to the interleaved audio buffer.
//!
//! @returns 0 on success
int qoamax_decoder_read_interleaved(qoamax_decoder *restrict self, size_t file_size,
                                    const char *restrict *restrict file, size_t samples_size,
                                    int16_t *restrict samples[static restrict 1]);

////////////////////
//    Encoder     //
////////////////////

//! Returns the final encoded bytesize for an audio stream with the given length and channel count.
//!
//! NOTE: channel_count must be between 1 and 8, samples_per_channel must be > 0.
//!       The result is undefined otherwise.
static inline uint64_t qoamax_encoder_get_encoded_size(int channel_count, uint32_t samples_per_channel) {
  uint64_t frames = ((uint64_t)samples_per_channel + QOAMAX_FRAME_SAMPLES - 1) / QOAMAX_FRAME_SAMPLES;
  uint64_t slices = ((uint64_t)samples_per_channel + 19) / 20; // per channel

  // The file header, a frame header plus every channel's LMS state (history, weights) per frame, and the slices.
  return 8 * (1 + frames * (1 + 2 * channel_count) + slices * channel_count);
}

//! Initializes the encoder state, supplying it with the information
//! required to produce headers and prepare the encoder's state.
//!
//! @param self          The encoder, does not need to be zeroed.
//! @param effort        The chosen performance/quality tradeoff.
//! @param channel_count The file's channel count (range 1..8).
//! @param total_samples Audio size hint written into the file's header, may be 0 for streaming.
//! @param samplerate    Samplerate hint written into the file's header, may be 0 (not recommended).
//!
//! @returns 0 on success
int qoamax_encoder_init(qoamax_encoder *self, qoamax_encoder_effort effort, int channel_count, uint32_t total_samples,
                        int32_t samplerate);

//! Encodes as many audio samples as possible using the given output and input buffers.
//!
//! Unless samples_complete is set, a frame is only started once all of its samples (QOAMAX_FRAME_SAMPLES per channel)
//! are available, as its header holds the sample count, and a slice is only encoded once the samples the effort
//! looks ahead are available. The encoder returns when it needs more input, so pass at least QOAMAX_FRAME_SAMPLES
//! samples per channel to ensure progress, and pass the samples that weren't consumed again in the next call.
//! Nothing is written for a stream without samples, as a QOA file can't be empty.
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
                         size_t samples_per_channel,
                         const int16_t *restrict samples[static restrict self->channel_count], bool samples_complete);

//! Encodes as many audio samples as possible using the given output and input buffers.
//!
//! Unless samples_complete is set, a frame is only started once all of its samples (QOAMAX_FRAME_SAMPLES per channel)
//! are available, as its header holds the sample count, and a slice is only encoded once the samples the effort
//! looks ahead are available. The encoder returns when it needs more input, so pass at least QOAMAX_FRAME_SAMPLES
//! samples per channel to ensure progress, and pass the samples that weren't consumed again in the next call.
//! Nothing is written for a stream without samples, as a QOA file can't be empty.
//!
//! Note that the input/output buffer pointers are given by non-const references,
//! since the encoder moves them to signal encoder progress.
//!
//! @param file_size           Amount of writable bytes in *file.
//! @param file                Pointer to output buffer pointer.
//! @param samples_size        Amount of total samples in the interleaved buffer.
//! @param samples             Pointer to interleaved audio sample buffer.
//! @param samples_complete    Whether the given samples array has reached the audio file's end.
//!
//! @returns 0 on success
int qoamax_encoder_write_interleaved(qoamax_encoder *restrict self, size_t file_size, char *restrict *restrict file,
                                     size_t samples_size, const int16_t *restrict samples[static restrict 1],
                                     bool samples_complete);

//! Returns the cumulated squared error up to the current encoding position for each channel.
static inline void qoamax_encoder_get_error(const qoamax_encoder *self,
                                            uint64_t errors[static restrict self->channel_count]) {
  for (int i = 0; i < self->channel_count; i++) errors[i] = self->channels[i].error;
}

#endif
