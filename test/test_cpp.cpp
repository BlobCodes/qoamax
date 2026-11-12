// Tests the plain declarations (src/qoamax_plain.h) used by C++: calls every API function from C++, linked against
// the library compiled as C. The functions have C linkage, so a wrong parameter type wouldn't be a link error; the
// results are checked against each other instead (planar vs. interleaved, the encoded size, the error, the first
// slice).
#include "signal.h"
#include <cstdio>
#include <cstring>
#include <qoamax.h>
#include <vector>

static int failures;

#define CHECK(condition)                                                                           \
  do {                                                                                             \
    if (!(condition)) {                                                                            \
      std::fprintf(stderr, "test_cpp: %s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
      failures++;                                                                                  \
    }                                                                                              \
  } while (0)

static constexpr int channels   = 2;
static constexpr uint32_t count = 12345; // samples per channel, not a multiple of a slice or a frame

int main() {
  std::vector<int16_t> interleaved(channels * count);
  std::vector<std::vector<int16_t>> planar(channels, std::vector<int16_t>(count));
  for (uint32_t i = 0; i < count; i++) {
    for (int c = 0; c < channels; c++) planar[c][i] = interleaved[i * channels + c] = test_signal(7, c, i);
  }
  size_t size = qoamax_encoder_get_encoded_size(channels, count);

  // Encode planar and interleaved.
  std::vector<char> file(size), file_interleaved(size);
  uint64_t errors[channels], errors_interleaved[channels];
  {
    qoamax_encoder encoder;
    CHECK(qoamax_encoder_init(&encoder, QOAMAX_ENCODER_EFFORT_FAST, channels, count, 44100) == 0);
    char *out                       = file.data();
    const int16_t *inputs[channels] = {planar[0].data(), planar[1].data()};
    CHECK(qoamax_encoder_write(&encoder, size, &out, count, inputs, true) == 0);
    CHECK(out == file.data() + size);
    CHECK(inputs[0] == planar[0].data() + count && inputs[1] == planar[1].data() + count);
    qoamax_encoder_get_error(&encoder, errors);
  }
  {
    qoamax_encoder encoder;
    CHECK(qoamax_encoder_init(&encoder, QOAMAX_ENCODER_EFFORT_FAST, channels, count, 44100) == 0);
    char *out            = file_interleaved.data();
    const int16_t *input = interleaved.data();
    CHECK(qoamax_encoder_write_interleaved(&encoder, size, &out, interleaved.size(), &input, true) == 0);
    CHECK(out == file_interleaved.data() + size);
    CHECK(input == interleaved.data() + interleaved.size());
    qoamax_encoder_get_error(&encoder, errors_interleaved);
  }
  CHECK(file == file_interleaved);
  CHECK(std::memcmp(errors, errors_interleaved, sizeof errors) == 0);

  // Decode planar and interleaved, and compare the error.
  std::vector<std::vector<int16_t>> decoded(channels, std::vector<int16_t>(count));
  std::vector<int16_t> decoded_interleaved(channels * count);
  {
    qoamax_decoder decoder;
    qoamax_decoder_query query;
    const char *in = file.data();
    CHECK(qoamax_decoder_init(&decoder, size, &in, &query) == 0);
    CHECK(query.total_samples == count && query.samplerate == 44100 && query.channel_count == channels);
    int16_t *outputs[channels] = {decoded[0].data(), decoded[1].data()};
    CHECK(qoamax_decoder_read(&decoder, file.data() + size - in, &in, count, outputs) == 0);
    CHECK(in == file.data() + size);
    CHECK(outputs[0] == decoded[0].data() + count && outputs[1] == decoded[1].data() + count);
  }
  {
    qoamax_decoder decoder;
    qoamax_decoder_query query;
    const char *in = file.data();
    CHECK(qoamax_decoder_init(&decoder, size, &in, &query) == 0);
    int16_t *out = decoded_interleaved.data();
    CHECK(qoamax_decoder_read_interleaved(&decoder, file.data() + size - in, &in, decoded_interleaved.size(), &out) ==
          0);
    CHECK(out == decoded_interleaved.data() + decoded_interleaved.size());
  }
  for (int c = 0; c < channels; c++) {
    uint64_t error = 0;
    for (uint32_t i = 0; i < count; i++) {
      CHECK(decoded[c][i] == decoded_interleaved[i * channels + c]);
      int64_t diff = decoded[c][i] - planar[c][i];
      error += (uint64_t)(diff * diff);
    }
    CHECK(error == errors[c]);
  }

  // The core functions reproduce the first slice of channel 0 (after the file header, the frame header and the
  // channels' LMS states).
  qoamax_encoder_channel channel = {QOAMAX_LMS_INITIAL, 0, 0};
  int lookahead                  = qoamax_encoder_effort_get_lookahead(QOAMAX_ENCODER_EFFORT_FAST);
  uint64_t slice  = qoamax_encode_slice(&channel, QOAMAX_ENCODER_EFFORT_FAST, lookahead, planar[0].data());
  uint64_t stored = 0;
  for (int i = 0; i < 8; i++) stored = stored << 8 | (uint8_t)file[16 + 16 * channels + i];
  CHECK(slice == stored);
  qoamax_lms lms = QOAMAX_LMS_INITIAL;
  int16_t samples[20];
  qoamax_decode_slice(&lms, slice, samples);
  CHECK(std::memcmp(samples, decoded[0].data(), sizeof samples) == 0);

  // The stream API: encode the interleaved samples (the first slice must match qoamax_encode_slice), decode them into
  // interleaved samples, whose error must match the encoder's.
  size_t slice_count = (count + 19) / 20;
  std::vector<std::vector<uint64_t>> slices(channels, std::vector<uint64_t>(slice_count));
  qoamax_encode_stream encode[channels];
  for (int c = 0; c < channels; c++)
    encode[c] = qoamax_encode_stream_init(&interleaved[c], count, channels, slices[c].data(), slice_count);
  qoamax_encode_streams(QOAMAX_ENCODER_EFFORT_FAST, channels, encode, true);
  CHECK(slices[0][0] == slice);
  std::vector<int16_t> stream_decoded(channels * count);
  qoamax_decode_stream decode[channels];
  for (int c = 0; c < channels; c++) {
    CHECK(encode[c].samples == &interleaved[c] + channels * count && encode[c].samples_size == 0);
    CHECK(encode[c].slices == slices[c].data() + slice_count && encode[c].slices_size == 0);
    decode[c] = qoamax_decode_stream_init(slices[c].data(), slice_count, &stream_decoded[c], count, channels);
  }
  qoamax_decode_streams(channels, decode);
  for (int c = 0; c < channels; c++) {
    CHECK(decode[c].samples_size == 0 && decode[c].slices_size == 0);
    uint64_t error = 0;
    for (uint32_t i = 0; i < count; i++) {
      int64_t diff = stream_decoded[i * channels + c] - interleaved[i * channels + c];
      error += (uint64_t)(diff * diff);
    }
    CHECK(error == encode[c].channel.error);
  }

  if (failures) return 1;
  std::printf("test_cpp: all checks passed (qoamax %d.%d.%d)\n", QOAMAX_VERSION_MAJOR, QOAMAX_VERSION_MINOR,
              QOAMAX_VERSION_PATCH);
  return 0;
}
