// Writes the test signal (test/signal.h) as raw 16-bit PCM or a WAV file, for the CLI tests.
//
// Usage: wavgen FORMAT CHANNELS FRAMES OUTPUT [unknown-size]
//   FORMAT        raw (16-bit little-endian PCM without header), u8, s16, s24, s32, f32, f64, or s16x/s24x/f32x
//                 (WAVE_FORMAT_EXTENSIBLE)
//   OUTPUT        A file, or - for stdout
//   unknown-size  Writes 0xFFFFFFFF as the RIFF and data sizes (like ffmpeg does when writing to a pipe)
//
// The samples' low bytes are zero, so that every format holds exactly the same 16-bit signal: converted to 16 bits by
// the CLI, all formats give identical QOA files. The WAV files contain extra chunks with odd sizes, which readers have
// to skip (including their padding byte).
#include "signal.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void put_u16(FILE *file, uint16_t value) {
  fputc(value & 0xFF, file);
  fputc(value >> 8, file);
}

static void put_u32(FILE *file, uint32_t value) {
  put_u16(file, value & 0xFFFF);
  put_u16(file, value >> 16);
}

int main(int argc, char **argv) {
  if (argc < 5) {
    fputs("usage: wavgen FORMAT CHANNELS FRAMES OUTPUT [unknown-size]\n", stderr);
    return EXIT_FAILURE;
  }
  const char *format = argv[1];
  int channels       = atoi(argv[2]);
  long frames        = atol(argv[3]);
  bool unknown_size  = argc > 5 && strcmp(argv[5], "unknown-size") == 0;
  bool raw           = strcmp(format, "raw") == 0;
  bool extensible    = format[strlen(format) - 1] == 'x';
  bool is_float      = format[0] == 'f';
  int bits           = raw ? 16 : atoi(format + 1);
  int bytes          = bits / 8;
  if (channels < 1 || frames < 0 || (bits != 8 && bits != 16 && bits != 24 && bits != 32 && bits != 64)) {
    fprintf(stderr, "wavgen: invalid arguments\n");
    return EXIT_FAILURE;
  }

  FILE *file = strcmp(argv[4], "-") == 0 ? stdout : fopen(argv[4], "wb");
  if (!file) {
    perror("wavgen");
    return EXIT_FAILURE;
  }

  uint32_t data_size = (uint32_t)(frames * channels * bytes);
  if (!raw) {
    uint32_t format_size = extensible ? 40 : 16;
    uint32_t riff_size   = 4 + (8 + 4) + (8 + format_size) + (8 + 6) + 8 + data_size;
    fputs("RIFF", file);
    put_u32(file, unknown_size ? UINT32_MAX : riff_size);
    fputs("WAVE", file);
    fputs("JUNK", file); // an odd-sized chunk with a padding byte
    put_u32(file, 3);
    fwrite("abc\0", 1, 4, file);

    fputs("fmt ", file);
    put_u32(file, format_size);
    put_u16(file, extensible ? 0xFFFE : is_float ? 3 : 1);
    put_u16(file, channels);
    put_u32(file, 44100);
    put_u32(file, 44100 * channels * bytes);
    put_u16(file, channels * bytes);
    put_u16(file, bits);
    if (extensible) {
      put_u16(file, 22);   // size of the extension
      put_u16(file, bits); // valid bits per sample
      put_u32(file, 0);    // channel mask
      put_u16(file, is_float ? 3 : 1);
      fwrite("\x00\x00\x00\x00\x10\x00\x80\x00\x00\xAA\x00\x38\x9B\x71", 1, 14, file); // rest of the subformat GUID
    }

    fputs("LIST", file);
    put_u32(file, 5);
    fwrite("info\0\0", 1, 6, file);

    fputs("data", file);
    put_u32(file, unknown_size ? UINT32_MAX : data_size);
  }

  for (long i = 0; i < frames; i++) {
    for (int ch = 0; ch < channels; ch++) {
      int16_t sample = (int16_t)(test_signal(8, ch, i) & ~0xFF);
      if (is_float && bits == 32) {
        float value = sample / 32768.0f;
        uint32_t data;
        memcpy(&data, &value, sizeof data);
        put_u32(file, data);
      } else if (is_float) {
        double value = sample / 32768.0;
        uint64_t data;
        memcpy(&data, &value, sizeof data);
        put_u32(file, (uint32_t)data);
        put_u32(file, (uint32_t)(data >> 32));
      } else if (bits == 8) {
        fputc((sample >> 8) + 128, file);
      } else {
        // Integer samples, left-aligned: the 16-bit sample followed by zero bytes.
        for (int b = 0; b < bytes - 2; b++) fputc(0, file);
        put_u16(file, (uint16_t)sample);
      }
    }
  }

  if (fclose(file) != 0) {
    perror("wavgen");
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
