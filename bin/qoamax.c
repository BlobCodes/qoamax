#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <math.h>
#include <getopt.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdatomic.h>
#include <semaphore.h>

#define TINYWAV_USE_VLA 1
#include "../src/qoamax.c"

///////////////////////////////////////////////////////////////////////////////
//                              Progress Thread                              //
///////////////////////////////////////////////////////////////////////////////

// typedef struct {
//   atomic_bool active;
//   uint8_t channel_count;
//   atomic_uint_fast64_t position;
//   atomic_uint_fast64_t error;
//   uint64_t total_samples;
// } print_progress_data;

// static void *print_progress_thread_func(void *args) {
//   print_progress_data *data = args;

//   while(atomic_load_explicit(&data->active, memory_order_relaxed)) {
// sem_timedwait
//   }

//   return NULL;
// }

static void print_progress(int samplerate, uint64_t total_samples, uint64_t position, uint64_t error) {
  uint64_t audio_ms = total_samples * 1000 / samplerate;
  uint64_t pos_ms = position * 1000 / samplerate;

  constexpr int width = 40;
  char progressbar[width * 22 + 5];

  int filled_segments = pos_ms * width / audio_ms;

  char *progressbar_pos = progressbar;
  for(int i = 1; i <= width; i++) {
    if(i <= filled_segments) {
      memcpy(progressbar_pos, "\033[38;2", sizeof("\033[38;2") - 1);
      progressbar_pos += sizeof("\033[38;2") - 1;

      int rgb[3] = {180 - i * 2, 60, 40 + i * 2};
      for(int j = 0; j < 3; j++) {
        *progressbar_pos = ';';
        progressbar_pos += 1;
        *progressbar_pos = '0' + (rgb[j] / 100);
        progressbar_pos += 1;
        *progressbar_pos = '0' + (rgb[j] / 10 % 10);
        progressbar_pos += 1;
        *progressbar_pos = '0' + (rgb[j] % 10);
        progressbar_pos += 1;
      }

      memcpy(progressbar_pos, "m█", sizeof("m█") - 1);
      progressbar_pos += sizeof("m█") - 1;

      if(i == filled_segments) {
        memcpy(progressbar_pos, "\033[0m", sizeof("\033[0m") - 1);
        progressbar_pos += sizeof("\033[0m") - 1;
      }
    } else {
      memcpy(progressbar_pos, "░", sizeof("░") - 1);
      progressbar_pos += sizeof("░") - 1;
    }
  }
  *progressbar_pos = '\0';

  char *initial_string = (position == 0) ? "\n" : "\033[2F";

  fprintf(stdout,
    "%s"
    "    %02i:%02i:%02i.%03i / %02i:%02i:%02i.%03i (RMS %.3f dB)"
    "\033[0K\n"
    "    %s"
    "\033[0K\n",

    initial_string,


    (int)(pos_ms / (1000 * 60 * 60)),
    (int)(pos_ms / (1000 * 60) % 60),
    (int)(pos_ms / 1000 % 60),
    (int)(pos_ms % 1000),

    (int)(audio_ms / (1000 * 60 * 60)),
    (int)(audio_ms / (1000 * 60) % 60),
    (int)(audio_ms / 1000 % 60),
    (int)(audio_ms % 1000),

    (-20.0f * log10f(2.0f / sqrtf(error / 1073741824.0f / position))),

    progressbar
  );
}

static int read_file_fully(FILE *in, char **buffer, size_t *size) {
  int err = fseek(in, 0, SEEK_END);
  if(err) return err;

  size_t insize = ftell(in);
  rewind(in);
  if(insize > (1LL << 30)) return -ENOMEM;

  char *inbuf = malloc(insize);
  if (!inbuf) return -ENOMEM;

  size_t read = fread(inbuf, 1, insize, in);
  if(read != insize) {
    free(inbuf);
    return -EBADF;
  }

  *buffer = inbuf;
  *size = insize;
  return 0;
}

int codec_encode(const char *input_path, FILE *in, const char *output_path, FILE *out, qoamax_encoder_effort effort) {
  char *inbuf_begin;
  size_t insize;

  int err = read_file_fully(in, &inbuf_begin, &insize);
  if(err) return err;

  const int channel_count = 2;
  char *inbuf_enc = inbuf_begin;
  char *inbuf_end = inbuf_begin + insize;
  qoamax_encoder encoder;

  // TODO: Read WAV, extract channel count / samplerate / size

  err = qoamax_encoder_init(&encoder, effort, channel_count, insize / sizeof(int16_t) / channel_count, 48000);
  if(err) {
    free(inbuf_begin);
    return err;
  }

  size_t alloc_size = qoamax_encoder_get_encoded_size(channel_count, insize / sizeof(int16_t) / channel_count);
  char *outbuf_begin = malloc(alloc_size);
  char *outbuf_end = outbuf_begin + alloc_size;
  char *outbuf_enc = outbuf_begin;

  if(!outbuf_begin) {
    free(inbuf_begin);
    return -ENOMEM;
  }

  fprintf(stdout, "\e[0;37mEncoding \e[0;95m%s\e[0m\n", input_path);

  int min_inbuf = 5120 * channel_count;
  uint64_t errors[2];
  while(true) {
    char *outbuf_before = outbuf_enc;

    qoamax_encoder_get_error(&encoder, errors);
    print_progress(48000, insize / sizeof(int16_t) / channel_count, (inbuf_enc - inbuf_begin) / sizeof(int16_t) / channel_count, (errors[0] + errors[1]) / 2);

    if(inbuf_end == inbuf_enc) break;

    err = qoamax_encoder_write_interleaved(&encoder, (outbuf_end - outbuf_enc), &outbuf_enc, MIN(min_inbuf, (inbuf_end - inbuf_enc) / sizeof(int16_t)), (const int16_t **)&inbuf_enc, true);
    if(err) break;

    size_t written = fwrite(outbuf_before, 1, (outbuf_enc - outbuf_before), out);
    if(written != (outbuf_enc - outbuf_before)) return -EINVAL;
  }

  free(outbuf_begin);
  free(inbuf_begin);
  if(err) {
    return err;
  }
  if(outbuf_enc != outbuf_end) {
    return -EINVAL;
  }
  if(inbuf_enc != inbuf_end) {
    return -EINVAL;
  }

  return 0;
}

int codec_decode(FILE *in, FILE *out) {
  char *inbuf;
  size_t insize;

  int err = read_file_fully(in, &inbuf, &insize);
  if(err) return err;

  char *inbuf_enc = inbuf;
  char *inbuf_end = inbuf + insize;
  qoamax_decoder decoder;
  qoamax_decoder_query query;

  err = qoamax_decoder_init(&decoder, (inbuf_end - inbuf_enc), (const char **)&inbuf_enc, &query);
  if(err) {
    free(inbuf);
    return err;
  }

  size_t samples_size = query.total_samples * query.channel_count;
  size_t samples_alloc = (samples_size + 19) * sizeof(int16_t);
  if(samples_alloc > (1LL << 30)) {
    free(inbuf);
    return -ENOMEM;
  }

  int16_t *samples_begin = malloc(samples_alloc);
  int16_t *samples_enc = samples_begin;
  int16_t *samples_end = &samples_begin[samples_size];
  if(!samples_begin) {
    free(inbuf);
    return -ENOMEM;
  }

  err = qoamax_decoder_read_interleaved(&decoder, (inbuf_end - inbuf_enc), (const char **)&inbuf_enc, samples_end - samples_enc, &samples_enc);
  free(inbuf);
  if(err) {
    free(samples_begin);
    return err;
  }
  if(samples_enc != samples_end) {
    free(samples_begin);
    return -EINVAL;
  }
  if(inbuf_enc != inbuf_end) {
    free(samples_begin);
    return -EINVAL;
  }

  // TODO: Pack into WAV?

  size_t written = fwrite(samples_begin, 1, (samples_enc - samples_begin) * sizeof(int16_t), out);
  free(samples_begin);
  if(written != (samples_enc - samples_begin) * sizeof(int16_t)) return -EINVAL;

  return 0;
}

static void print_global_help(FILE *stream) {
  fprintf(stream, "Usage:\n"
                  "  qoamax encode [options] <input> <output>\n"
                  "  qoamax decode [options] <input> <output>\n"
                  "  qoamax --help\n"
                  "\n"
                  "Commands:\n"
                  "  encode       Encode an audio file\n"
                  "  decode       Decode an audio file\n"
                  "\n"
                  "Options:\n"
                  "  -h, --help   Show this help message\n");
}

static void print_encode_help(FILE *stream) {
  fprintf(stream, "Usage: qoamax encode [options] <input> <output>\n"
                  "\n"
                  "Encode an audio file.\n"
                  "\n"
                  "Options:\n"
                  "  -e, --effort=LEVEL  Encoding effort:\n"
                  "                      superfast, veryfast, fast, medium,\n"
                  "                      slow, veryslow, or placebo\n"
                  "  -h, --help          Show this help message\n"
                  "\n"
                  "Examples:\n"
                  "  qoamax encode input.wav output.qoa\n"
                  "  qoamax encode --effort=veryfast input.wav output.qoa\n"
                  "  qoamax encode --effort slow input.wav output.qoa\n"
                  "  qoamax encode -e placebo input.wav output.qoa\n");
}

static void print_decode_help(FILE *stream) {
  fprintf(stream, "Usage: qoamax decode [options] <input> <output>\n"
                  "\n"
                  "Decode an audio file.\n"
                  "\n"
                  "Options:\n"
                  "  -h, --help   Show this help message\n"
                  "\n"
                  "Examples:\n"
                  "  qoamax decode input.qoa output.wav\n");
}

static int parse_effort(const char *text, qoamax_encoder_effort *effort) {
  static const struct {
    const char *name;
    qoamax_encoder_effort value;
  } values[] = {
    {"superfast", QOAMAX_ENCODER_EFFORT_SUPERFAST},
    {"veryfast", QOAMAX_ENCODER_EFFORT_VERYFAST},
    {"fast", QOAMAX_ENCODER_EFFORT_FAST},
    {"medium", QOAMAX_ENCODER_EFFORT_MEDIUM},
    {"slow", QOAMAX_ENCODER_EFFORT_SLOW},
    {"veryslow", QOAMAX_ENCODER_EFFORT_VERYSLOW},
    {"placebo", QOAMAX_ENCODER_EFFORT_PLACEBO},
  };

  if (text == NULL || effort == NULL) {
    return -1;
  }

  for (size_t i = 0; i < sizeof(values) / sizeof(values[0]); ++i) {
    if (strcmp(text, values[i].name) == 0) {
      *effort = values[i].value;
      return 0;
    }
  }

  return -1;
}

static int paths_are_equal(const char *input_path, const char *output_path) {
  /*
   * This catches the common case where the same path is supplied twice.
   * The codec CLI should never truncate its input file.
   */
  return strcmp(input_path, output_path) == 0;
}

static int open_files(const char *input_path, const char *output_path, FILE **input, FILE **output) {
  *input = fopen(input_path, "rb");
  if (*input == NULL) {
    fprintf(stderr, "qoamax: cannot open input '%s': %s\n", input_path, strerror(errno));
    return -1;
  }

  *output = fopen(output_path, "wb");
  if (*output == NULL) {
    fprintf(stderr, "qoamax: cannot open output '%s': %s\n", output_path, strerror(errno));
    fclose(*input);
    return -1;
  }

  return 0;
}

static int run_encode(int argc, char **argv) {
  static const struct option options[] = {
      {"effort", required_argument, NULL, 'e'},
      {"help", no_argument, NULL, 'h'},
      {NULL, 0, NULL, 0}};

  qoamax_encoder_effort effort = QOAMAX_ENCODER_EFFORT_MEDIUM;
  const char *input_path;
  const char *output_path;
  FILE *input = NULL;
  FILE *output = NULL;
  int option;
  int codec_status;

  optind = 1;
  opterr = 0;

  while ((option = getopt_long(argc, argv, "e:h", options, NULL)) != -1) {
    switch (option) {
    case 'e':
      if (parse_effort(optarg, &effort) != 0) {
        fprintf(stderr,
                "qoamax: invalid effort '%s'\n"
                "qoamax: valid efforts are: "
                "superfast, veryfast, fast, medium, slow, veryslow, placebo\n",
                optarg);
        return EXIT_FAILURE;
      }
      break;

    case 'h':
      print_encode_help(stdout);
      return EXIT_SUCCESS;

    case '?':
    default:
      fprintf(stderr, "qoamax: invalid encode option\n");
      print_encode_help(stderr);
      return EXIT_FAILURE;
    }
  }

  if (argc - optind != 2) {
    fprintf(stderr, "qoamax: encode requires exactly two file arguments\n");
    print_encode_help(stderr);
    return EXIT_FAILURE;
  }

  input_path  = argv[optind];
  output_path = argv[optind + 1];

  if (paths_are_equal(input_path, output_path)) {
    fprintf(stderr, "qoamax: input and output files must be different\n");
    return EXIT_FAILURE;
  }

  if (open_files(input_path, output_path, &input, &output) != 0) {
    return EXIT_FAILURE;
  }

  codec_status = codec_encode(input_path, input, output_path, output, effort);

  if (fclose(input) != 0) {
    fprintf(stderr, "qoamax: error reading input '%s'\n", input_path);
    codec_status = EXIT_FAILURE;
  }

  if (fclose(output) != 0) {
    fprintf(stderr, "qoamax: error writing output '%s'\n", output_path);
    codec_status = EXIT_FAILURE;
  }

  if (codec_status != 0) {
    fprintf(stderr, "qoamax: encoding error %i\n", codec_status);
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

static int run_decode(int argc, char **argv) {
  static const struct option options[] = {{"help", no_argument, NULL, 'h'}, {NULL, 0, NULL, 0}};

  const char *input_path;
  const char *output_path;
  FILE *input  = NULL;
  FILE *output = NULL;
  int option;
  int codec_status;

  optind = 1;
  opterr = 0;

  while ((option = getopt_long(argc, argv, "h", options, NULL)) != -1) {
    switch (option) {
    case 'h':
      print_decode_help(stdout);
      return EXIT_SUCCESS;

    case '?':
    default:
      fprintf(stderr, "qoamax: invalid decode option\n");
      print_decode_help(stderr);
      return EXIT_FAILURE;
    }
  }

  if (argc - optind != 2) {
    fprintf(stderr, "qoamax: decode requires exactly two file arguments\n");
    print_decode_help(stderr);
    return EXIT_FAILURE;
  }

  input_path  = argv[optind];
  output_path = argv[optind + 1];

  if (paths_are_equal(input_path, output_path)) {
    fprintf(stderr, "qoamax: input and output files must be different\n");
    return EXIT_FAILURE;
  }

  if (open_files(input_path, output_path, &input, &output) != 0) {
    return EXIT_FAILURE;
  }

  codec_status = codec_decode(input, output);

  if (fclose(input) != 0) {
    fprintf(stderr, "qoamax: error reading input '%s'\n", input_path);
    codec_status = EXIT_FAILURE;
  }

  if (fclose(output) != 0) {
    fprintf(stderr, "qoamax: error writing output '%s'\n", output_path);
    codec_status = EXIT_FAILURE;
  }

  if (codec_status != 0) {
    fprintf(stderr, "qoamax: decoding error %i\n", codec_status);
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    print_global_help(stderr);
    return EXIT_FAILURE;
  }

  if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
    print_global_help(stdout);
    return EXIT_SUCCESS;
  }

  if (strcmp(argv[1], "encode") == 0) {
    return run_encode(argc - 1, argv + 1);
  }

  if (strcmp(argv[1], "decode") == 0) {
    return run_decode(argc - 1, argv + 1);
  }

  fprintf(stderr, "qoamax: unknown command '%s'\n", argv[1]);
  print_global_help(stderr);
  return EXIT_FAILURE;
}
