// Example: encodes one second of stereo audio to QOA in memory and decodes it again, then does the same without the
// file format, with the stream API (as in the README).
// test/build_systems.sh builds it against the installed library (as C11) to check the build systems.
#include <qoamax.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
  enum { channels = 2, samplerate = 44100, count = samplerate }; // samples per channel

  // Interleaved 16-bit samples (left, right, left, ...): two triangle waves.
  static int16_t samples[count * channels];
  for (int i = 0; i < count; i++) {
    samples[i * 2]     = (int16_t)(abs(i * 300 % 40000 - 20000) - 10000);
    samples[i * 2 + 1] = (int16_t)(abs(i * 450 % 40000 - 20000) - 10000);
  }

  // Encode. The functions advance the buffer pointers to report how much they used.
  size_t size = qoamax_encoder_get_encoded_size(channels, count);
  char *file  = malloc(size);
  if (!file) return 1;
  qoamax_encoder encoder;
  if (qoamax_encoder_init(&encoder, QOAMAX_ENCODER_EFFORT_MEDIUM, channels, count, samplerate) != 0) return 1;
  char *out         = file;
  const int16_t *in = samples;
  if (qoamax_encoder_write_interleaved(&encoder, size, &out, count * channels, &in, true) != 0) return 1;
  printf("Encoded %d samples into %zu bytes\n", count * channels, (size_t)(out - file));

  // Decode.
  static int16_t decoded[count * channels];
  qoamax_decoder decoder;
  qoamax_decoder_query query;
  const char *read = file;
  if (qoamax_decoder_init(&decoder, size, &read, &query) != 0) return 1;
  int16_t *write = decoded;
  if (qoamax_decoder_read_interleaved(&decoder, size - (size_t)(read - file), &read, count * channels, &write) != 0) {
    return 1;
  }
  printf("Decoded %u Hz, %u channels, %u samples per channel\n", (unsigned)query.samplerate,
         (unsigned)query.channel_count, (unsigned)query.total_samples);

  free(file);
  if (write != decoded + count * channels) return 1;

  // The stream API: one array of slices per channel, for your own container format.
  enum { slice_count = (count + 19) / 20 };
  static uint64_t slices[channels][slice_count];
  qoamax_encode_stream encode[channels];
  for (int ch = 0; ch < channels; ch++)
    encode[ch] = qoamax_encode_stream_init(&samples[ch], count, channels, slices[ch], slice_count);
  qoamax_encode_streams(QOAMAX_ENCODER_EFFORT_ULTRAFAST, channels, encode, true);

  qoamax_decode_stream decode[channels];
  for (int ch = 0; ch < channels; ch++)
    decode[ch] = qoamax_decode_stream_init(slices[ch], slice_count, &decoded[ch], count, channels);
  qoamax_decode_streams(channels, decode);
  printf("Encoded and decoded %d slices per channel with the stream API\n", slice_count);
  return decode[0].samples_size == 0 && decode[1].samples_size == 0 ? 0 : 1;
}
