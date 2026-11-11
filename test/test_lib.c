// Library tests, run by `make test`. They are built with ASan and UBSan, with the default SIMD variants (chosen at
// runtime) and for several fixed values of QOAMAX_LANES, which must all produce identical output. All buffers passed to
// the library are allocated with their exact size, so that ASan catches reads or writes out of bounds.
//
// Usage: test_lib [--print-checksums]
//   --print-checksums  Prints the checksums of the encoder efforts' output, to update `effort_cases` after an
//                      intended change of the encoder's output (check the quality of the change first).
#include "../src/qoamax.c"
#include "signal.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

///////////////////////////////////////////////////////////////////////////////
//                                 Utilities                                 //
///////////////////////////////////////////////////////////////////////////////

static int checks, failures;

#define CHECK(condition, ...)                                \
  do {                                                       \
    checks++;                                                \
    if (!(condition)) {                                      \
      failures++;                                            \
      fprintf(stderr, "FAIL (%s:%d): ", __FILE__, __LINE__); \
      fprintf(stderr, __VA_ARGS__);                          \
      fputc('\n', stderr);                                   \
    }                                                        \
  } while (0)

static void *xmalloc(size_t size) {
  void *data = malloc(size ? size : 1);
  if (!data) {
    fputs("out of memory\n", stderr);
    exit(EXIT_FAILURE);
  }
  return data;
}

//! Returns a copy of the data in a buffer of exactly its size.
static char *copy(const char *data, size_t size) {
  char *result = xmalloc(size);
  memcpy(result, data, size);
  return result;
}

static size_t max_size(size_t a, size_t b) { return a > b ? a : b; }

static uint64_t fnv1a(const void *data, size_t size) {
  uint64_t hash = 0xCBF29CE484222325u;
  for (size_t i = 0; i < size; i++) hash = (hash ^ ((const unsigned char *)data)[i]) * 0x100000001B3u;
  return hash;
}

//! Returns the interleaved test signal.
static int16_t *make_signal(int channels, size_t frames, uint32_t seed) {
  int16_t *samples = xmalloc(frames * channels * sizeof *samples);
  for (size_t i = 0; i < frames; i++)
    for (int ch = 0; ch < channels; ch++) samples[i * channels + ch] = test_signal(seed, ch, i);
  return samples;
}

static uint64_t squared_error(const int16_t *a, const int16_t *b, size_t count) {
  uint64_t error = 0;
  for (size_t i = 0; i < count; i++) error += (uint64_t)((int32_t)a[i] - b[i]) * (uint64_t)((int32_t)a[i] - b[i]);
  return error;
}

typedef struct {
  char *data;
  size_t size;
  uint64_t error; //!< The encoder's error over all channels
} encoded;

//! Encodes the whole signal with one call into a buffer of exactly qoamax_encoder_get_encoded_size bytes.
static encoded encode_all(qoamax_encoder_effort effort, int channels, size_t frames, const int16_t *samples) {
  encoded result = {.size = frames ? qoamax_encoder_get_encoded_size(channels, frames) : 0};
  result.data    = xmalloc(result.size);

  qoamax_encoder encoder;
  CHECK(qoamax_encoder_init(&encoder, effort, channels, frames, 44100) == 0, "encoder_init");
  char *pos         = result.data;
  const int16_t *in = samples;
  int err           = qoamax_encoder_write_interleaved(&encoder, result.size, &pos, frames * channels, &in, true);
  CHECK(err == 0, "encoder_write_interleaved returned %d", err);
  CHECK(pos == result.data + result.size, "encoded %td bytes, expected %zu (%d channels, %zu samples)",
        pos - result.data, result.size, channels, frames);
  CHECK(in == samples + frames * channels, "encoder consumed %td of %zu samples", in - samples, frames * channels);

  uint64_t errors[QOAMAX_MAX_CHANNEL_COUNT];
  qoamax_encoder_get_error(&encoder, errors);
  for (int ch = 0; ch < channels; ch++) result.error += errors[ch];
  return result;
}

//! Decodes a whole file (interleaved) into a buffer of exactly the sample count of its header.
static int16_t *decode_all(const encoded *file, int channels, size_t frames) {
  char *data       = copy(file->data, file->size);
  int16_t *samples = xmalloc(frames * channels * sizeof *samples);
  const char *pos  = data;
  qoamax_decoder decoder;
  qoamax_decoder_query query;
  int err = qoamax_decoder_init(&decoder, file->size, &pos, &query);
  CHECK(err == 0, "decoder_init returned %d", err);
  CHECK(query.channel_count == channels && query.total_samples == frames && query.samplerate == 44100,
        "decoder_init: %d channels, %lu samples, %ld Hz", query.channel_count, (unsigned long)query.total_samples,
        (long)query.samplerate);

  int16_t *out = samples;
  err          = qoamax_decoder_read_interleaved(&decoder, data + file->size - pos, &pos, frames * channels, &out);
  CHECK(err == 0, "decoder_read_interleaved returned %d", err);
  CHECK(out == samples + frames * channels, "decoded %td of %zu samples", out - samples, frames * channels);
  CHECK(pos == data + file->size, "decoder stopped at byte %td of %zu", pos - data, file->size);
  free(data);
  return samples;
}

///////////////////////////////////////////////////////////////////////////////
//                                Encoder efforts                            //
///////////////////////////////////////////////////////////////////////////////

//! The expected checksums of the encoded and decoded test signal for each effort (and some other parameters packed
//! into an effort value). They make sure that optimizations don't change the encoder's output.
static const struct {
  const char *name;
  qoamax_encoder_effort effort;
  int channels;
  size_t frames;
  uint64_t encoded_checksum;
  uint64_t decoded_checksum;
} effort_cases[] = {
    {"ultrafast", QOAMAX_ENCODER_EFFORT_ULTRAFAST, 2, 10317, 0xAAD267AD2A857CC2, 0xDE9D6C7BD238FCC3},
    {"superfast", QOAMAX_ENCODER_EFFORT_SUPERFAST, 2, 10317, 0x43EAED04864AB5E5, 0x9BEF49125285D1A0},
    {"veryfast", QOAMAX_ENCODER_EFFORT_VERYFAST, 2, 10317, 0x8960E132589BCD4B, 0xCE2861A7992C052B},
    {"fast", QOAMAX_ENCODER_EFFORT_FAST, 2, 10317, 0xA6C362F7FF6761C7, 0x7E401339D86FEDC0},
    {"medium", QOAMAX_ENCODER_EFFORT_MEDIUM, 2, 10317, 0x23A838F6C26579DA, 0x8ECB5D8BF5B62D04},
    {"slow", QOAMAX_ENCODER_EFFORT_SLOW, 2, 10317, 0x3F83D1476F40952F, 0xA5B5429054B97DCD},
    {"veryslow", QOAMAX_ENCODER_EFFORT_VERYSLOW, 1, 6000, 0x0FAF1EE5C2914745, 0xF27ECCC8D35EB97F},
    {"placebo", QOAMAX_ENCODER_EFFORT_PLACEBO, 1, 100, 0xF92501D63341B662, 0x5332B08CB36D1716},
    {"hill-climb, 60-sample lookahead", (qoamax_encoder_effort)2, 2, 10317, 0x6D5D577D032DF79E, 0xD0FF5EBD73FB4FC6},
    {"hill-climb, 80-sample lookahead", (qoamax_encoder_effort)3, 2, 10317, 0x0D392AD93F818114, 0x12ABA06CDC916916},
    {"beam width 8 (wide search state)", (qoamax_encoder_effort)(8 << 2), 1, 3000, 0x17358E6822EC6EF1,
     0x0992061CE2B3FF9D},
    {"beam width 16, 80-sample lookahead (wide node buffers)", (qoamax_encoder_effort)(3 | 16 << 2), 1, 600,
     0x5CBB4E7800814D3E, 0xF5A00D1DF9A687A6},
};

static void test_efforts(bool print_checksums) {
  for (size_t i = 0; i < sizeof effort_cases / sizeof effort_cases[0]; i++) {
    int channels     = effort_cases[i].channels;
    size_t frames    = effort_cases[i].frames;
    int16_t *signal  = make_signal(channels, frames, 1);
    encoded file     = encode_all(effort_cases[i].effort, channels, frames, signal);
    int16_t *decoded = decode_all(&file, channels, frames);

    // The encoder tracks the exact error of the decoded samples.
    uint64_t error = squared_error(signal, decoded, frames * channels);
    CHECK(file.error == error, "%s: the encoder reports an error of %llu, the decoded samples have %llu",
          effort_cases[i].name, (unsigned long long)file.error, (unsigned long long)error);

    uint64_t encoded_checksum = fnv1a(file.data, file.size);
    uint64_t decoded_checksum = fnv1a(decoded, frames * channels * sizeof *decoded);
    if (print_checksums) {
      printf("    {\"%s\", ..., 0x%016llX, 0x%016llX}, // PSNR %.2f dB\n", effort_cases[i].name,
             (unsigned long long)encoded_checksum, (unsigned long long)decoded_checksum,
             error ? 10 * __builtin_log10(32767.0 * 32767.0 * frames * channels / error) : 999.0);
    } else {
      CHECK(encoded_checksum == effort_cases[i].encoded_checksum,
            "%s: the encoded output changed (checksum 0x%016llX). If intended, see --print-checksums",
            effort_cases[i].name, (unsigned long long)encoded_checksum);
      CHECK(decoded_checksum == effort_cases[i].decoded_checksum,
            "%s: the decoded output changed (checksum 0x%016llX). If intended, see --print-checksums",
            effort_cases[i].name, (unsigned long long)decoded_checksum);
    }
    free(signal), free(file.data), free(decoded);
  }
}

///////////////////////////////////////////////////////////////////////////////
//                        Channel counts and lengths                         //
///////////////////////////////////////////////////////////////////////////////

//! Decodes a file with the planar API into separate buffers, and interleaves the result.
static int16_t *decode_all_planar(const encoded *file, int channels, size_t frames) {
  char *data       = copy(file->data, file->size);
  int16_t *samples = xmalloc(frames * channels * sizeof *samples);
  int16_t *planes[QOAMAX_MAX_CHANNEL_COUNT], *out[QOAMAX_MAX_CHANNEL_COUNT];
  for (int ch = 0; ch < channels; ch++) out[ch] = planes[ch] = xmalloc(frames * sizeof(int16_t));

  const char *pos = data;
  qoamax_decoder decoder;
  qoamax_decoder_query query;
  CHECK(qoamax_decoder_init(&decoder, file->size, &pos, &query) == 0, "decoder_init");
  int err = qoamax_decoder_read(&decoder, data + file->size - pos, &pos, frames, out);
  CHECK(err == 0, "decoder_read returned %d", err);
  CHECK(out[0] == planes[0] + frames, "planar: decoded %td of %zu samples", out[0] - planes[0], frames);
  CHECK(pos == data + file->size, "planar: decoder stopped at byte %td of %zu", pos - data, file->size);

  for (int ch = 0; ch < channels; ch++) {
    for (size_t i = 0; i < frames; i++) samples[i * channels + ch] = planes[ch][i];
    free(planes[ch]);
  }
  free(data);
  return samples;
}

static void test_channels_and_lengths(void) {
  static const size_t lengths[]                = {1, 19, 20, 21, 5119, 5120, 5121, 10317};
  static const qoamax_encoder_effort efforts[] = {QOAMAX_ENCODER_EFFORT_ULTRAFAST, QOAMAX_ENCODER_EFFORT_VERYFAST};

  for (int channels = 1; channels <= QOAMAX_MAX_CHANNEL_COUNT; channels++) {
    for (size_t l = 0; l < sizeof lengths / sizeof lengths[0]; l++) {
      for (size_t e = 0; e < sizeof efforts / sizeof efforts[0]; e++) {
        size_t frames    = lengths[l];
        int16_t *signal  = make_signal(channels, frames, 2);
        encoded file     = encode_all(efforts[e], channels, frames, signal);
        int16_t *decoded = decode_all(&file, channels, frames);
        int16_t *planar  = decode_all_planar(&file, channels, frames);

        CHECK(memcmp(decoded, planar, frames * channels * sizeof *decoded) == 0,
              "%d channels, %zu samples: planar and interleaved decoding differ", channels, frames);
        uint64_t error = squared_error(signal, decoded, frames * channels);
        CHECK(file.error == error, "%d channels, %zu samples: the encoder reports an error of %llu, actual %llu",
              channels, frames, (unsigned long long)file.error, (unsigned long long)error);
        free(signal), free(file.data), free(decoded), free(planar);
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
//                              Streaming encoder                            //
///////////////////////////////////////////////////////////////////////////////

//! Encodes the signal in pieces of `piece` samples per channel (the samples that weren't consumed are passed again),
//! with at most `space` bytes of output per call.
static char *encode_chunked(qoamax_encoder_effort effort, int channels, size_t frames, const int16_t *samples,
                            size_t piece, size_t space, size_t size) {
  char *result = xmalloc(size), *pos = result;
  qoamax_encoder encoder;
  qoamax_encoder_init(&encoder, effort, channels, frames, 44100);

  size_t done = 0;
  while (done < frames || pos < result + size) {
    size_t available = QOAMAX_MIN(frames - done, piece);
    bool complete    = done + available == frames;
    size_t out_size  = QOAMAX_MIN(space, (size_t)(result + size - pos));
    char *out = xmalloc(out_size), *out_pos = out;
    int16_t *in = (int16_t *)copy((const char *)&samples[done * channels], available * channels * sizeof *samples);
    const int16_t *in_pos = in;

    int err = qoamax_encoder_write_interleaved(&encoder, out_size, &out_pos, available * channels, &in_pos, complete);
    CHECK(err == 0, "encoder_write_interleaved returned %d", err);
    memcpy(pos, out, out_pos - out);
    pos += out_pos - out;
    done += (in_pos - in) / channels;
    bool progress = out_pos != out || in_pos != in;
    free(out), free(in);
    if (!progress) {
      CHECK(false, "the encoder stopped at sample %zu of %zu, byte %td of %zu (pieces of %zu, %zu bytes of space)",
            done, frames, pos - result, size, piece, space);
      break;
    }
  }
  return result;
}

static void test_streaming_encoder(void) {
  static const qoamax_encoder_effort efforts[] = {QOAMAX_ENCODER_EFFORT_ULTRAFAST, QOAMAX_ENCODER_EFFORT_VERYFAST,
                                                  QOAMAX_ENCODER_EFFORT_FAST, (qoamax_encoder_effort)3};
  static const size_t pieces[]                 = {QOAMAX_FRAME_SAMPLES, QOAMAX_FRAME_SAMPLES + 33, 9000};
  static const size_t extra_spaces[]           = {0, 8, 4096}; // on top of the space for a frame header and a slice
  size_t frames                                = 3 * QOAMAX_FRAME_SAMPLES - 77;

  for (size_t e = 0; e < sizeof efforts / sizeof efforts[0]; e++) {
    for (int channels = 1; channels <= 3; channels++) {
      int16_t *signal = make_signal(channels, frames, 3);
      encoded file    = encode_all(efforts[e], channels, frames, signal);

      for (size_t p = 0; p < sizeof pieces / sizeof pieces[0]; p++) {
        for (size_t s = 0; s < sizeof extra_spaces / sizeof extra_spaces[0]; s++) {
          // Starting a frame needs space for the frame header with the LMS states, and the first slices.
          size_t space  = 8 * (1 + 2 * channels) + 8 * channels + extra_spaces[s];
          char *chunked = encode_chunked(efforts[e], channels, frames, signal, pieces[p], space, file.size);
          CHECK(memcmp(chunked, file.data, file.size) == 0,
                "effort %d, %d channels: encoding in pieces of %zu samples with %zu bytes of space differs", efforts[e],
                channels, pieces[p], space);
          free(chunked);
        }
      }

      // The planar API gives the same result.
      int16_t *planes[QOAMAX_MAX_CHANNEL_COUNT];
      const int16_t *in[QOAMAX_MAX_CHANNEL_COUNT];
      for (int ch = 0; ch < channels; ch++) {
        in[ch] = planes[ch] = xmalloc(frames * sizeof(int16_t));
        for (size_t i = 0; i < frames; i++) planes[ch][i] = signal[i * channels + ch];
      }
      char *planar = xmalloc(file.size), *pos = planar;
      qoamax_encoder encoder;
      qoamax_encoder_init(&encoder, efforts[e], channels, frames, 44100);
      CHECK(qoamax_encoder_write(&encoder, file.size, &pos, frames, in, true) == 0, "encoder_write");
      CHECK(pos == planar + file.size && memcmp(planar, file.data, file.size) == 0,
            "effort %d, %d channels: planar and interleaved encoding differ", efforts[e], channels);
      CHECK(in[0] == planes[0] + frames, "planar: consumed %td of %zu samples", in[0] - planes[0], frames);
      for (int ch = 0; ch < channels; ch++) free(planes[ch]);
      free(planar), free(signal), free(file.data);
    }
  }

  // Before the end of the stream, a frame is only started once all of its samples are available.
  qoamax_encoder encoder;
  int16_t *signal = make_signal(2, QOAMAX_FRAME_SAMPLES, 4);
  char *out = xmalloc(4096), *pos = out;
  const int16_t *in = signal;
  qoamax_encoder_init(&encoder, QOAMAX_ENCODER_EFFORT_FAST, 2, 0, 44100);
  CHECK(qoamax_encoder_write_interleaved(&encoder, 4096, &pos, 2 * (QOAMAX_FRAME_SAMPLES - 1), &in, false) == 0,
        "encoder_write_interleaved");
  CHECK(pos == out && in == signal, "the encoder started a frame without all of its samples");
  CHECK(qoamax_encoder_write_interleaved(&encoder, 4096, &pos, 2 * QOAMAX_FRAME_SAMPLES, &in, false) == 0,
        "encoder_write_interleaved");
  CHECK(in > signal, "the encoder made no progress with a full frame of samples");

  // An empty stream produces no output (a QOA file can't be empty).
  pos = out, in = signal;
  qoamax_encoder_init(&encoder, QOAMAX_ENCODER_EFFORT_FAST, 2, 0, 44100);
  CHECK(qoamax_encoder_write_interleaved(&encoder, 4096, &pos, 0, &in, true) == 0, "encoder_write_interleaved");
  CHECK(pos == out, "an empty stream produced %td bytes", pos - out);
  free(out), free(signal);

  // Invalid arguments
  CHECK(qoamax_encoder_init(&encoder, QOAMAX_ENCODER_EFFORT_FAST, 0, 0, 44100) == QOAMAX_ERROR_INVALID_ARGUMENT,
        "0 channels");
  CHECK(qoamax_encoder_init(&encoder, QOAMAX_ENCODER_EFFORT_FAST, 9, 0, 44100) ==
            QOAMAX_ERROR_UNSUPPORTED_CHANNEL_COUNT,
        "9 channels");
  CHECK(qoamax_encoder_init(&encoder, QOAMAX_ENCODER_EFFORT_FAST, 1, 0, 1 << 24) == QOAMAX_ERROR_INVALID_ARGUMENT,
        "samplerate 2^24");
}

///////////////////////////////////////////////////////////////////////////////
//                              Streaming decoder                            //
///////////////////////////////////////////////////////////////////////////////

//! Decodes the file into output buffers of `chunk` samples per channel, making `window` more bytes of input visible
//! whenever the decoder needs more (like reading a file in pieces). Returns the interleaved samples.
static int16_t *decode_chunked(const encoded *file, int channels, size_t frames, size_t chunk, size_t window,
                               bool planar) {
  int16_t *samples = xmalloc(frames * channels * sizeof *samples);
  size_t offset = 0, visible = max_size(window, 16), done = 0;

  qoamax_decoder decoder;
  qoamax_decoder_query query;
  char *head      = copy(file->data, 16);
  const char *pos = head;
  CHECK(qoamax_decoder_init(&decoder, 16, &pos, &query) == 0, "decoder_init");
  offset = pos - head;
  free(head);

  while (true) {
    size_t size     = QOAMAX_MIN(visible, file->size - offset);
    char *data      = copy(file->data + offset, size);
    const char *pos = data;
    size_t got;
    int err;
    if (planar) {
      int16_t *planes[QOAMAX_MAX_CHANNEL_COUNT], *out[QOAMAX_MAX_CHANNEL_COUNT];
      for (int ch = 0; ch < channels; ch++) out[ch] = planes[ch] = xmalloc(chunk * sizeof(int16_t));
      err = qoamax_decoder_read(&decoder, size, &pos, chunk, out);
      got = out[0] - planes[0];
      for (int ch = 0; ch < channels; ch++) {
        for (size_t i = 0; i < got && done + i < frames; i++) samples[(done + i) * channels + ch] = planes[ch][i];
        free(planes[ch]);
      }
    } else {
      int16_t *buffer = xmalloc(chunk * channels * sizeof(int16_t)), *out = buffer;
      err = qoamax_decoder_read_interleaved(&decoder, size, &pos, chunk * channels, &out);
      got = (out - buffer) / channels;
      memcpy(&samples[done * channels], buffer, QOAMAX_MIN(got, frames - done) * channels * sizeof(int16_t));
      free(buffer);
    }
    CHECK(err == 0, "decoder_read returned %d", err);
    size_t consumed = pos - data;
    free(data);

    offset += consumed;
    done += got;
    if (consumed || got) {
      visible = max_size(window, 16);
    } else if (size == file->size - offset) {
      break; // the whole rest of the file was visible
    } else {
      visible += window;
    }
  }
  CHECK(done == frames && offset == file->size, "decoded %zu of %zu samples, %zu of %zu bytes", done, frames, offset,
        file->size);
  return samples;
}

static void test_streaming_decoder(void) {
  static const size_t chunks[]  = {20, 21, 39, 100, QOAMAX_FRAME_SAMPLES, QOAMAX_FRAME_SAMPLES + 1};
  static const size_t windows[] = {16, 17, 100, 4096, SIZE_MAX / 2};
  size_t frames                 = 2 * QOAMAX_FRAME_SAMPLES + 77;

  for (int channels = 1; channels <= 3; channels++) {
    int16_t *signal    = make_signal(channels, frames, 5);
    encoded file       = encode_all(QOAMAX_ENCODER_EFFORT_ULTRAFAST, channels, frames, signal);
    int16_t *reference = decode_all(&file, channels, frames);

    for (int planar = 0; planar <= 1; planar++) {
      for (size_t c = 0; c < sizeof chunks / sizeof chunks[0]; c++) {
        for (size_t w = 0; w < sizeof windows / sizeof windows[0]; w++) {
          int16_t *decoded = decode_chunked(&file, channels, frames, chunks[c], windows[w], planar);
          CHECK(memcmp(decoded, reference, frames * channels * sizeof *decoded) == 0,
                "%d channels, %s: decoding with %zu samples of output and %zu bytes of input at once differs", channels,
                planar ? "planar" : "interleaved", chunks[c], windows[w]);
          free(decoded);
        }
      }
    }

    // Less than a slice of output space: the decoder may read the frame header, but no slices, and writes nothing.
    char *data      = copy(file.data, file.size);
    const char *pos = data;
    qoamax_decoder decoder;
    qoamax_decoder_query query;
    qoamax_decoder_init(&decoder, file.size, &pos, &query);
    const char *start = pos;
    int16_t *buffer = xmalloc(19 * channels * sizeof(int16_t)), *out = buffer;
    CHECK(qoamax_decoder_read_interleaved(&decoder, data + file.size - pos, &pos, 19 * channels, &out) == 0,
          "decoder_read_interleaved");
    CHECK(out == buffer && pos <= start + 8 * (1 + 2 * channels), "decoding into space for 19 samples made progress");

    free(buffer), free(data), free(reference), free(signal), free(file.data);
  }
}

///////////////////////////////////////////////////////////////////////////////
//                            Invalid and corrupt files                      //
///////////////////////////////////////////////////////////////////////////////

//! Decodes a whole (possibly corrupt) file. @returns the first error, or 0.
static int decode_file(const char *file, size_t size, size_t *decoded_frames) {
  char *data      = copy(file, size);
  const char *pos = data;
  qoamax_decoder decoder;
  qoamax_decoder_query query;
  int err         = qoamax_decoder_init(&decoder, size, &pos, &query);
  *decoded_frames = 0;

  int16_t *buffer = err ? NULL : xmalloc(QOAMAX_FRAME_SAMPLES * query.channel_count * sizeof(int16_t));
  while (!err) {
    const char *before = pos;
    int16_t *out       = buffer;
    err = qoamax_decoder_read_interleaved(&decoder, data + size - pos, &pos, QOAMAX_FRAME_SAMPLES * query.channel_count,
                                          &out);
    CHECK(pos >= before && pos <= data + size, "the decoder moved outside the file");
    *decoded_frames += (out - buffer) / query.channel_count;
    if (pos == before && out == buffer) break;
  }
  free(buffer), free(data);
  return err;
}

//! Sets a big-endian value in a file.
static void put_be(char *file, size_t offset, int bytes, uint32_t value) {
  for (int i = 0; i < bytes; i++) file[offset + i] = (char)(value >> (8 * (bytes - 1 - i)));
}

static void test_invalid_files(void) {
  size_t frames   = QOAMAX_FRAME_SAMPLES + 80; // two frames
  int16_t *signal = make_signal(2, frames, 6);
  encoded file    = encode_all(QOAMAX_ENCODER_EFFORT_ULTRAFAST, 2, frames, signal);
  size_t second   = 8 + qoamax__frame_size(2, QOAMAX_FRAME_SAMPLES); // offset of the second frame header
  size_t decoded;

  CHECK(decode_file(file.data, file.size, &decoded) == 0 && decoded == frames, "the valid file");
  CHECK(decode_file(file.data, 15, &decoded) == QOAMAX_ERROR_INVALID_ARGUMENT, "15 bytes");

  // Each case modifies a copy of the file and expects an error.
  static const struct {
    const char *name;
    int frame;         //!< 0 = the file header, 1 = the first frame header, 2 = the second one
    int offset, bytes; // in the header
    uint32_t value;
    int error;
  } cases[] = {
      {"magic", 0, 0, 4, 0x716F6167, QOAMAX_ERROR_INVALID_FILE_HEADER},
      {"0 channels", 1, 0, 1, 0, QOAMAX_ERROR_INVALID_FILE_HEADER},
      {"32 channels", 1, 0, 1, 32, QOAMAX_ERROR_INVALID_FILE_HEADER},
      {"9 channels", 1, 0, 1, 9, QOAMAX_ERROR_UNSUPPORTED_CHANNEL_COUNT},
      {"0 samples in a frame", 1, 4, 2, 0, QOAMAX_ERROR_INVALID_FRAME_HEADER},
      {"5121 samples in a frame", 1, 4, 2, 5121, QOAMAX_ERROR_INVALID_FRAME_HEADER},
      {"wrong frame size", 1, 6, 2, 0, QOAMAX_ERROR_INVALID_FRAME_HEADER},
      {"padding slice", 2, 6, 2, 8 * (1 + 2 * (2 + 4)) + 16, QOAMAX_ERROR_INVALID_FRAME_HEADER},
      {"samplerate changes", 2, 1, 3, 48000, QOAMAX_ERROR_SAMPLE_FORMAT_BREAK},
      {"channel count changes", 2, 0, 1, 1, QOAMAX_ERROR_SAMPLE_FORMAT_BREAK},
  };
  for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
    char *modified = copy(file.data, file.size);
    size_t header  = cases[i].frame == 0 ? 0 : cases[i].frame == 1 ? 8 : second;
    put_be(modified, header + cases[i].offset, cases[i].bytes, cases[i].value);
    int err = decode_file(modified, file.size, &decoded);
    CHECK(err == cases[i].error, "%s: error %d, expected %d", cases[i].name, err, cases[i].error);
    free(modified);
  }

  // Truncated files decode up to the cut without reading beyond it.
  for (size_t size = 16; size < file.size; size++) {
    int err = decode_file(file.data, size, &decoded);
    CHECK(err == 0 && decoded < frames, "truncated to %zu bytes: error %d, %zu samples", size, err, decoded);
  }

  // Random corruption must not crash the decoder or make it access memory out of bounds (checked by the sanitizers).
  uint32_t random = 12345;
  for (int i = 0; i < 3000; i++) {
    char *modified = copy(file.data, file.size);
    for (int n = 0; n < 1 + i % 4; n++) {
      random = random * 1664525 + 1013904223;
      // Mostly the headers and LMS states, where corruption changes the control flow.
      size_t position    = (random >> 8) % (i % 2 ? file.size : 8 + 8 * (1 + 2 * 2));
      modified[position] = (char)(random >> 24);
    }
    decode_file(modified, file.size, &decoded);
    free(modified);
  }
  free(signal), free(file.data);
}

///////////////////////////////////////////////////////////////////////////////
//                                 Stack usage                               //
///////////////////////////////////////////////////////////////////////////////

#ifndef TEST_STACK_SIZE
#define TEST_STACK_SIZE (32 * 1024) //!< Stack of the thread running ULTRAFAST to VERYSLOW (they need about 6 KB)
#endif

static void *encode_efforts_thread(void *unused) {
  (void)unused;
  static const qoamax_encoder_effort efforts[] = {
      QOAMAX_ENCODER_EFFORT_ULTRAFAST, QOAMAX_ENCODER_EFFORT_SUPERFAST, QOAMAX_ENCODER_EFFORT_VERYFAST,
      QOAMAX_ENCODER_EFFORT_FAST,      QOAMAX_ENCODER_EFFORT_MEDIUM,    QOAMAX_ENCODER_EFFORT_SLOW,
      QOAMAX_ENCODER_EFFORT_VERYSLOW,
  };
  int16_t *signal = make_signal(2, 300, 7);
  for (size_t i = 0; i < sizeof efforts / sizeof efforts[0]; i++) free(encode_all(efforts[i], 2, 300, signal).data);
  free(signal);
  return NULL;
}

//! The efforts up to VERYSLOW must run with little stack (e.g. on embedded targets or threads with small stacks).
//! A crash here means that the encoder uses too much stack.
static void test_stack_usage(void) {
  pthread_attr_t attributes;
  pthread_t thread;
  pthread_attr_init(&attributes);
  CHECK(pthread_attr_setstacksize(&attributes, TEST_STACK_SIZE) == 0, "pthread_attr_setstacksize");
  CHECK(pthread_create(&thread, &attributes, encode_efforts_thread, NULL) == 0, "pthread_create");
  pthread_join(thread, NULL);
  pthread_attr_destroy(&attributes);
}

///////////////////////////////////////////////////////////////////////////////
//                                    Main                                   //
///////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
  if (argc > 1 && strcmp(argv[1], "--print-checksums") == 0) {
    test_efforts(true);
    return EXIT_SUCCESS;
  }

  static const struct {
    const char *name;
    void (*run)(void);
  } tests[] = {
      {"channel counts and lengths", test_channels_and_lengths},
      {"streaming encoder", test_streaming_encoder},
      {"streaming decoder", test_streaming_decoder},
      {"invalid and corrupt files", test_invalid_files},
      {"stack usage", test_stack_usage},
  };

#ifdef QOAMAX_HAS_VARIANTS
  printf("test_lib (QOAMAX_LANES=%d, runtime dispatch to %d):\n", QOAMAX_LANES, qoamax__lanes_supported());
#else
  printf("test_lib (QOAMAX_LANES=%d):\n", QOAMAX_LANES);
#endif
  int before = failures;
  test_efforts(false);
  printf("  %-28s %s\n", "encoder efforts", failures == before ? "ok" : "FAILED");
  for (size_t i = 0; i < sizeof tests / sizeof tests[0]; i++) {
    before = failures;
    fflush(stdout);
    tests[i].run();
    printf("  %-28s %s\n", tests[i].name, failures == before ? "ok" : "FAILED");
  }

  printf("%d of %d checks failed\n", failures, checks);
  return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
