#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "../src/qoamax.c"

///////////////////////////////////////////////////////////////////////////////
//                                 Utilities                                 //
///////////////////////////////////////////////////////////////////////////////

//! What the terminal on stderr supports, see @ref terminal_detect.
static struct {
  bool interactive; //!< A terminal that supports cursor movement (for the progress display)
  bool color;       //!< ANSI colors are wanted
  bool truecolor;   //!< 24-bit colors are supported
  bool progress;    //!< The terminal shows the progress reported with OSC 9;4 (e.g. in its tab or taskbar button)
} terminal;

static bool env_equals(const char *name, const char *value) {
  const char *actual = getenv(name);
  return actual && strcmp(actual, value) == 0;
}

static void terminal_detect(void) {
  const char *term     = getenv("TERM");
  const char *no_color = getenv("NO_COLOR"); // https://no-color.org
  terminal.interactive = isatty(STDERR_FILENO) && term && *term && strcmp(term, "dumb") != 0;
  terminal.color       = terminal.interactive && !(no_color && *no_color);
  terminal.truecolor   = terminal.color && (env_equals("COLORTERM", "truecolor") || env_equals("COLORTERM", "24bit"));
  // Only known terminals, as others may misinterpret it (older iTerm2 versions show OSC 9 as a notification).
  terminal.progress =
      terminal.interactive && (getenv("VTE_VERSION") || getenv("WT_SESSION") || env_equals("ConEmuANSI", "ON") ||
                               env_equals("TERM_PROGRAM", "WezTerm") || env_equals("TERM_PROGRAM", "ghostty"));
}

//! Removes the progress reported to the terminal (OSC 9;4).
#define TERMINAL_PROGRESS_REMOVE "\033]9;4;0;0\033\\"

//! The state of the progress display on stderr. Messages clear it first, so that they stay on screen and the display
//! is redrawn below them.
static volatile sig_atomic_t progress_lines;    //!< Lines of the display on screen
static volatile sig_atomic_t progress_reported; //!< Whether progress was reported to the terminal (OSC 9;4)

//! Removes the progress display from the screen. The cursor is at the start of the line below it.
static void progress_clear(void) {
  if (progress_lines > 0) fprintf(stderr, "\033[%dF\033[J", (int)progress_lines);
  progress_lines = 0;
}

//! Removes the progress display and the progress reported to the terminal.
static void progress_end(void) {
  progress_clear();
  if (progress_reported) fputs(TERMINAL_PROGRESS_REMOVE, stderr);
  progress_reported = 0;
}

[[gnu::format(printf, 1, 2)]]
static void print_error(const char *format, ...) {
  progress_clear();
  va_list args;
  va_start(args, format);
  fputs("qoamax: ", stderr);
  vfprintf(stderr, format, args);
  fputc('\n', stderr);
  va_end(args);
}

[[gnu::format(printf, 1, 2)]]
static void print_warning(const char *format, ...) {
  progress_clear();
  va_list args;
  va_start(args, format);
  fputs("qoamax: warning: ", stderr);
  vfprintf(stderr, format, args);
  fputc('\n', stderr);
  va_end(args);
}

static const char *library_error_string(int err) {
  switch (err) {
  case QOAMAX_ERROR_INVALID_ARGUMENT:
    return "invalid argument";
  case QOAMAX_ERROR_INVALID_FILE_HEADER:
    return "not a QOA file";
  case QOAMAX_ERROR_INVALID_FRAME_HEADER:
    return "corrupt frame header";
  case QOAMAX_ERROR_SAMPLE_FORMAT_BREAK:
    return "the channel count or samplerate changes within the file";
  case QOAMAX_ERROR_UNSUPPORTED_CHANNEL_COUNT:
    return "more than 8 channels are not supported";
  default:
    return "unknown error";
  }
}

static double now_seconds(void) {
  struct timespec time;
  clock_gettime(CLOCK_MONOTONIC, &time);
  return time.tv_sec + time.tv_nsec * 1e-9;
}

//! Formats a duration as HH:MM:SS.mmm (or HH:MM:SS).
static void format_duration(char out[static 24], double seconds, bool milliseconds) {
  if (!(seconds >= 0)) seconds = 0; // also catches NaN
  if (seconds > 1e9) seconds = 1e9;
  uint64_t ms    = (uint64_t)(seconds * 1000 + 0.5);
  unsigned hours = ms / 3600000, minutes = ms / 60000 % 60, secs = ms / 1000 % 60;
  if (milliseconds)
    snprintf(out, 24, "%02u:%02u:%02u.%03u", hours, minutes, secs, (unsigned)(ms % 1000));
  else
    snprintf(out, 24, "%02u:%02u:%02u", hours, minutes, secs);
}

//! Formats the peak signal-to-noise ratio of the given squared error over the given amount of samples.
static void format_psnr(char out[static 24], uint64_t error, uint64_t samples) {
  if (samples == 0)
    snprintf(out, 24, "-- dB");
  else if (error == 0)
    snprintf(out, 24, "inf dB");
  else
    snprintf(out, 24, "%.2f dB", 10.0 * log10(32767.0 * 32767.0 * samples / error));
}

static bool has_extension(const char *path, const char *extension) {
  size_t path_length = strlen(path), extension_length = strlen(extension);
  return path_length >= extension_length && strcasecmp(path + path_length - extension_length, extension) == 0;
}

static uint16_t get_u16le(const unsigned char *p) { return p[0] | p[1] << 8; }

static uint32_t get_u32le(const unsigned char *p) { return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24; }

static void put_u16le(unsigned char *p, uint16_t value) { p[0] = value, p[1] = value >> 8; }

static void put_u32le(unsigned char *p, uint32_t value) {
  p[0] = value, p[1] = value >> 8, p[2] = value >> 16, p[3] = value >> 24;
}

///////////////////////////////////////////////////////////////////////////////
//                                 File I/O                                  //
///////////////////////////////////////////////////////////////////////////////

#define READ_CHUNK_SIZE 65536    //!< Bytes read from the input at once
#define WRITE_BUFFER_SIZE 262144 //!< Bytes buffered before writing to the output

//! An input file ("-" = stdin). It is read sequentially, so that pipes work.
typedef struct {
  FILE *file;
  const char *path;
  bool failed;              //!< A read error occurred (and was reported)
  int64_t size;             //!< The file's size, -1 if unknown (not a regular file)
  unsigned char peeked[12]; //!< The first bytes, read ahead to detect the format
  size_t peeked_size;
  size_t peeked_pos;
} input;

static bool input_open(input *in, const char *path) {
  *in      = (input){.path = path, .size = -1};
  in->file = strcmp(path, "-") == 0 ? stdin : fopen(path, "rb");
  if (!in->file) {
    print_error("cannot open '%s': %s", path, strerror(errno));
    return false;
  }

  struct stat info;
  if (fstat(fileno(in->file), &info) == 0 && S_ISREG(info.st_mode)) in->size = info.st_size;
  in->peeked_size = fread(in->peeked, 1, sizeof in->peeked, in->file);
  return true;
}

//! Reads up to `size` bytes. Returns fewer only at the end of the file, or on errors (which set in->failed).
static size_t input_read(input *in, void *data, size_t size) {
  size_t n = QOAMAX_MIN(size, in->peeked_size - in->peeked_pos);
  memcpy(data, &in->peeked[in->peeked_pos], n);
  in->peeked_pos += n;
  if (n < size) n += fread((unsigned char *)data + n, 1, size - n, in->file);

  if (n < size && ferror(in->file) && !in->failed) {
    print_error("cannot read '%s': %s", in->path, strerror(errno));
    in->failed = true;
  }
  return n;
}

static bool input_skip(input *in, uint64_t size) {
  unsigned char discarded[4096];
  while (size > 0) {
    size_t n = QOAMAX_MIN(size, sizeof discarded);
    if (input_read(in, discarded, n) != n) return false;
    size -= n;
  }
  return true;
}

static void input_close(input *in) {
  if (in->file && in->file != stdin) fclose(in->file);
}

//! Whether both paths refer to the same existing file.
static bool same_file(const char *a, const char *b) {
  struct stat info_a, info_b;
  if (strcmp(a, "-") == 0 || strcmp(b, "-") == 0) return false;
  return stat(a, &info_a) == 0 && stat(b, &info_b) == 0 && info_a.st_dev == info_b.st_dev &&
         info_a.st_ino == info_b.st_ino;
}

//! The partially written output file, which is removed if the program is interrupted.
static const char *volatile interrupted_cleanup_path;

static void handle_interrupt(int signal_number) {
  if (interrupted_cleanup_path) unlink(interrupted_cleanup_path);
  // Remove the progress display (only write() is allowed here). Line wrapping is re-enabled, in case the display was
  // being drawn.
  [[maybe_unused]] ssize_t ignored;
  if (progress_lines > 0 && progress_lines < 10) {
    char clear[] = "\033[?7h\033[0F\033[J";
    clear[7]     = '0' + progress_lines;
    ignored      = write(STDERR_FILENO, clear, sizeof clear - 1);
  }
  if (progress_reported) ignored = write(STDERR_FILENO, TERMINAL_PROGRESS_REMOVE, sizeof TERMINAL_PROGRESS_REMOVE - 1);
  signal(signal_number, SIG_DFL);
  raise(signal_number);
}

//! An output file ("-" = stdout). It is written directly, so it should only be opened once the input has been
//! checked. A partially written regular file is removed on failure.
typedef struct {
  FILE *file;
  const char *path;
  bool regular; //!< A regular file: it can be patched afterwards, and is removed on failure
} output;

static bool output_open(output *out, const char *path) {
  // glibc ignores setvbuf's size without a buffer, using st_blksize (often 4 KiB). Writes that small make decoding
  // to a file up to 60% slower (e.g. on btrfs).
  static char buffer[WRITE_BUFFER_SIZE];

  *out = (output){.path = path};
  if (strcmp(path, "-") == 0) {
    if (isatty(STDOUT_FILENO)) {
      print_error("refusing to write binary data to a terminal");
      return false;
    }
    out->file = stdout;
    setvbuf(out->file, buffer, _IOFBF, sizeof buffer);
    return true;
  }

  out->file = fopen(path, "wb");
  if (!out->file) {
    print_error("cannot create '%s': %s", path, strerror(errno));
    return false;
  }
  setvbuf(out->file, buffer, _IOFBF, sizeof buffer);

  struct stat info;
  out->regular = fstat(fileno(out->file), &info) == 0 && S_ISREG(info.st_mode); // not e.g. /dev/null or a pipe
  if (out->regular) interrupted_cleanup_path = path;
  return true;
}

//! Overwrites already written bytes (e.g. to update a header). Only possible for regular files.
static void output_patch(output *out, long offset, const void *data, size_t size) {
  if (fseek(out->file, offset, SEEK_SET) == 0) fwrite(data, 1, size, out->file);
  fseek(out->file, 0, SEEK_END);
}

//! Closes the output, checking for write errors. A partially written regular file is removed.
static bool output_close(output *out, bool ok) {
  if (fflush(out->file) != 0 || ferror(out->file)) {
    print_error("cannot write '%s': %s", out->path, strerror(errno));
    ok = false;
  }
  if (out->file != stdout && fclose(out->file) != 0 && ok) {
    print_error("cannot write '%s': %s", out->path, strerror(errno));
    ok = false;
  }

  if (out->regular && !ok) remove(out->path);
  interrupted_cleanup_path = NULL;
  return ok;
}

///////////////////////////////////////////////////////////////////////////////
//                              PCM Audio Formats                            //
///////////////////////////////////////////////////////////////////////////////

typedef enum {
  FORMAT_AUTO,
  FORMAT_WAV,
  FORMAT_RAW, //!< 16-bit little-endian PCM, interleaved
} pcm_format;

enum {
  WAV_PCM        = 1,
  WAV_FLOAT      = 3,
  WAV_EXTENSIBLE = 0xFFFE,
};

//! Checks the format against QOA's limits.
static bool check_format(int channels, int64_t samplerate) {
  if (channels < 1 || channels > QOAMAX_MAX_CHANNEL_COUNT) {
    print_error("%d channels are not supported (1 to %d)", channels, QOAMAX_MAX_CHANNEL_COUNT);
    return false;
  }
  if (samplerate < 1 || samplerate > 0xFFFFFF) {
    print_error("a samplerate of %lld Hz is not supported (1 to 16777215)", (long long)samplerate);
    return false;
  }
  return true;
}

//! Converts a WAV sample (integer PCM with 1 to 4 bytes, or float with 4 or 8 bytes) to 16 bits.
static int16_t wav_convert_sample(const unsigned char *p, int encoding, int bytes) {
  if (encoding == WAV_FLOAT) {
    double value;
    if (bytes == 4) {
      uint32_t bits = get_u32le(p);
      float single;
      memcpy(&single, &bits, sizeof single);
      value = single;
    } else {
      uint64_t bits = get_u32le(p) | (uint64_t)get_u32le(p + 4) << 32;
      memcpy(&value, &bits, sizeof value);
    }

    value *= 32768.0;
    if (isnan(value)) return 0;
    if (value >= INT16_MAX) return INT16_MAX;
    if (value <= INT16_MIN) return INT16_MIN;
    return (int16_t)lrint(value);
  }

  // Align the integer sample to 32 bits (8-bit samples are unsigned), then round it to 16 bits.
  uint32_t aligned;
  switch (bytes) {
  case 1:
    aligned = (uint32_t)(p[0] ^ 0x80) << 24;
    break;
  case 2:
    aligned = (uint32_t)get_u16le(p) << 16;
    break;
  case 3:
    aligned = (uint32_t)p[0] << 8 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 24;
    break;
  default:
    aligned = get_u32le(p);
    break;
  }
  int64_t rounded = ((int64_t)(int32_t)aligned + (1 << 15)) >> 16;
  return rounded > INT16_MAX ? INT16_MAX : (int16_t)rounded;
}

//! Reads interleaved 16-bit audio from a WAV or raw PCM input.
typedef struct {
  input in;
  int channels;
  int32_t samplerate;
  int encoding;       //!< WAV_PCM or WAV_FLOAT
  int bytes;          //!< Bytes per sample
  uint64_t remaining; //!< Bytes of audio data left, UINT64_MAX = until the end of the file
  uint32_t frames;    //!< Total samples per channel, 0 if unknown
} pcm_reader;

static bool wav_read_header(pcm_reader *reader) {
  input *in       = &reader->in;
  in->peeked_pos  = 12; // "RIFF", size, "WAVE"
  bool has_format = false;

  unsigned char chunk[8];
  while (input_read(in, chunk, sizeof chunk) == sizeof chunk) {
    uint32_t size = get_u32le(&chunk[4]);

    if (memcmp(chunk, "fmt ", 4) == 0) {
      unsigned char format[40] = {0};
      size_t format_size       = QOAMAX_MIN(size, sizeof format);
      if (size < 16 || input_read(in, format, format_size) != format_size ||
          !input_skip(in, size - format_size + (size & 1))) {
        if (!in->failed) print_error("invalid WAV format chunk");
        return false;
      }

      int encoding        = get_u16le(&format[0]);
      int channels        = get_u16le(&format[2]);
      uint32_t samplerate = get_u32le(&format[4]);
      int block_align     = get_u16le(&format[12]);
      int bits            = get_u16le(&format[14]);
      if (encoding == WAV_EXTENSIBLE && size >= 40) encoding = get_u16le(&format[24]); // GUID of the subformat
      int bytes = channels > 0 ? block_align / channels : 0;

      bool supported =
          (encoding == WAV_PCM && bytes >= 1 && bytes <= 4) || (encoding == WAV_FLOAT && (bytes == 4 || bytes == 8));
      if (!supported || block_align != bytes * channels) {
        print_error("unsupported WAV sample format (format tag 0x%04X, %d bits per sample); supported are 8 to "
                    "32-bit integer and 32/64-bit float PCM",
                    encoding, bits);
        return false;
      }
      if (!check_format(channels, samplerate)) return false;

      reader->encoding   = encoding;
      reader->bytes      = bytes;
      reader->channels   = channels;
      reader->samplerate = samplerate;
      has_format         = true;
    } else if (memcmp(chunk, "data", 4) == 0) {
      if (!has_format) {
        print_error("invalid WAV file (the data chunk comes before the format chunk)");
        return false;
      }

      // WAV files written to a pipe (e.g. by ffmpeg) don't know their size: their data goes until the end.
      if (size != UINT32_MAX) {
        reader->remaining = size;
        reader->frames    = size / ((uint32_t)reader->bytes * reader->channels);
      }
      return true;
    } else if (!input_skip(in, size + (size & 1))) {
      break;
    }
  }

  if (!in->failed) print_error("invalid WAV file (no data chunk)");
  return false;
}

//! Opens the input and reads its header. The format is detected unless it's given, raw input defaults to 48 kHz mono.
static bool pcm_reader_open(pcm_reader *reader, const char *path, pcm_format format, int channels, int32_t samplerate) {
  *reader = (pcm_reader){.encoding = WAV_PCM, .bytes = 2, .remaining = UINT64_MAX};
  if (!input_open(&reader->in, path)) return false;

  const input *in = &reader->in;
  bool is_wav = in->peeked_size == 12 && memcmp(in->peeked, "RIFF", 4) == 0 && memcmp(&in->peeked[8], "WAVE", 4) == 0;
  bool raw_options = channels != 0 || samplerate != 0;
  if (format == FORMAT_AUTO) {
    if (is_wav)
      format = FORMAT_WAV;
    else if (raw_options || has_extension(path, ".pcm") || has_extension(path, ".raw"))
      format = FORMAT_RAW;
    else {
      print_error("'%s' is not a WAV file; use --format=raw for raw 16-bit PCM, or convert it first "
                  "(e.g. ffmpeg -i <input> -f wav - | qoamax encode - <output>)",
                  path);
      return false;
    }
  }

  if (format == FORMAT_WAV) {
    if (raw_options) print_warning("--rate and --channels only apply to raw input");
    if (!is_wav) {
      print_error("'%s' is not a WAV file", path);
      return false;
    }
    return wav_read_header(reader);
  }

  reader->channels   = channels ? channels : 1;
  reader->samplerate = samplerate ? samplerate : 48000;
  if (in->size >= 0) {
    uint64_t frames = (uint64_t)in->size / (2 * reader->channels);
    reader->frames  = QOAMAX_MIN(frames, UINT32_MAX); // longer inputs are rejected while reading
  }
  return true;
}

//! Reads up to `frames` samples per channel (interleaved). Returns fewer only at the end of the audio, or on errors
//! (reader->in.failed).
static size_t pcm_reader_read(pcm_reader *reader, int16_t *samples, size_t frames) {
  static unsigned char data[READ_CHUNK_SIZE];
  size_t frame_size = (size_t)reader->bytes * reader->channels;
  size_t done       = 0;

  while (done < frames && reader->remaining > 0) {
    if (reader->remaining < frame_size) {
      print_warning("ignoring the last %zu bytes (incomplete sample frame)", (size_t)reader->remaining);
      reader->remaining = 0;
      break;
    }

    size_t size = QOAMAX_MIN((frames - done) * frame_size, sizeof data / frame_size * frame_size);
    size        = QOAMAX_MIN(size, reader->remaining / frame_size * frame_size);
    size_t read = input_read(&reader->in, data, size);
    if (reader->remaining != UINT64_MAX) reader->remaining -= read;

    size_t read_frames = read / frame_size;
    int16_t *converted = &samples[done * reader->channels];
    if (reader->encoding == WAV_PCM && reader->bytes == 2) { // the common case is 5% faster without the generic path
      for (size_t i = 0; i < read_frames * reader->channels; i++) converted[i] = (int16_t)get_u16le(&data[2 * i]);
    } else {
      for (size_t i = 0; i < read_frames * reader->channels; i++)
        converted[i] = wav_convert_sample(&data[i * reader->bytes], reader->encoding, reader->bytes);
    }
    done += read_frames;

    if (read < size) { // end of the file
      if (reader->in.failed)
        ;
      else if (reader->remaining != UINT64_MAX)
        print_warning("the WAV file is truncated");
      else if (read % frame_size != 0)
        print_warning("ignoring the last %zu bytes (incomplete sample frame)", read % frame_size);
      reader->remaining = 0;
    }
  }
  return done;
}

//! Writes the samples as 16-bit little-endian PCM. Write errors are reported by output_close.
static void write_s16le(FILE *file, const int16_t *samples, size_t count) {
  unsigned char chunk[8192];
  while (count > 0) {
    size_t n = QOAMAX_MIN(count, sizeof chunk / 2);
    for (size_t i = 0; i < n; i++) put_u16le(&chunk[2 * i], samples[i]);
    fwrite(chunk, 2, n, file);
    samples += n;
    count -= n;
  }
}

//! The WAV data size of 16-bit audio, or UINT32_MAX ("unknown") if it is too large for a WAV header.
static uint32_t wav_data_size(uint64_t frames, int channels) {
  uint64_t size = frames * channels * sizeof(int16_t);
  return size <= UINT32_MAX - 36 ? size : UINT32_MAX;
}

//! Builds the header of a 16-bit PCM WAV file. A data size of UINT32_MAX means "until the end of the file".
static void wav_build_header(unsigned char header[static 44], int channels, int32_t samplerate, uint32_t data_size) {
  memcpy(&header[0], "RIFF", 4);
  put_u32le(&header[4], data_size == UINT32_MAX ? UINT32_MAX : 36 + data_size);
  memcpy(&header[8], "WAVEfmt ", 8);
  put_u32le(&header[16], 16);
  put_u16le(&header[20], WAV_PCM);
  put_u16le(&header[22], channels);
  put_u32le(&header[24], samplerate);
  put_u32le(&header[28], samplerate * channels * sizeof(int16_t)); // bytes per second
  put_u16le(&header[32], channels * sizeof(int16_t));              // bytes per sample frame
  put_u16le(&header[34], 16);                                      // bits per sample
  memcpy(&header[36], "data", 4);
  put_u32le(&header[40], data_size);
}

///////////////////////////////////////////////////////////////////////////////
//                                 Progress                                  //
///////////////////////////////////////////////////////////////////////////////

#define PROGRESS_BAR_WIDTH 40
#define PROGRESS_INTERVAL 0.1 //!< Minimum seconds between redraws

//! Three-line progress display on stderr (title, position, PSNR, speed, ETA and a progress bar). It is removed again
//! by @ref progress_end, so that only a summary line is left once the file is done.
typedef struct {
  bool enabled;
  const char *title;
  uint32_t total; //!< 0 if unknown
  int32_t samplerate;
  double start;
  double last_draw;
  int reported; //!< The progress last reported to the terminal in percent, -1 = unknown length, -2 = nothing yet
} progress;

static void progress_begin(progress *self, bool enabled, const char *title, uint32_t total, int32_t samplerate) {
  *self = (progress){
      .enabled    = enabled,
      .title      = title,
      .total      = total,
      .samplerate = samplerate,
      .start      = now_seconds(),
      .last_draw  = -INFINITY,
      .reported   = -2,
  };
}

//! Appends formatted text to a string in a fixed-size buffer (truncating it if the buffer is full).
[[gnu::format(printf, 3, 4)]]
static void append(char *buffer, size_t size, const char *format, ...) {
  size_t length = strlen(buffer);
  va_list args;
  va_start(args, format);
  vsnprintf(&buffer[length], size - length, format, args);
  va_end(args);
}

//! Redraws the progress display, at most every PROGRESS_INTERVAL seconds.
static void progress_update(progress *self, uint64_t position, uint64_t error, int channels) {
  double now = now_seconds();
  if (!self->enabled || now - self->last_draw < PROGRESS_INTERVAL) return;
  self->last_draw = now;

  double elapsed = now - self->start;
  double speed   = elapsed > 0 ? position / (self->samplerate * elapsed) : 0; // audio seconds per second
  char position_text[24], psnr_text[24], line[PROGRESS_BAR_WIDTH * 24 + 160 + 1024] = "";
  format_duration(position_text, (double)position / self->samplerate, true);
  format_psnr(psnr_text, error, position * channels);

  append(line, sizeof line, "%s\033[K\n", self->title);
  int percent = -1; // unknown
  if (self->total == 0) {
    append(line, sizeof line, "    %s  PSNR %s  %.1fx\033[K\n    (total length unknown)", position_text, psnr_text,
           speed);
  } else {
    double fraction = QOAMAX_MIN((double)position / self->total, 1.0);
    percent         = (int)(100 * fraction);
    char total_text[24], eta_text[24];
    format_duration(total_text, (double)self->total / self->samplerate, true);
    format_duration(eta_text, position > 0 ? elapsed * (1 - fraction) / fraction : 0, false);
    append(line, sizeof line, "    %s / %s  %5.1f%%  PSNR %s  %.1fx  ETA %s\033[K\n    ", position_text, total_text,
           100 * fraction, psnr_text, speed, eta_text);

    // Filled cells get a color gradient (or just a color), empty cells are shaded.
    int filled = (int)(fraction * PROGRESS_BAR_WIDTH);
    if (terminal.color && !terminal.truecolor) append(line, sizeof line, "\033[35m");
    for (int i = 1; i <= filled; i++) {
      if (terminal.truecolor) append(line, sizeof line, "\033[38;2;%d;60;%dm", 180 - 2 * i, 40 + 2 * i);
      append(line, sizeof line, "█");
    }
    if (terminal.color) append(line, sizeof line, "\033[0m");
    for (int i = filled; i < PROGRESS_BAR_WIDTH; i++) append(line, sizeof line, "░");
  }

  // Line wrapping is disabled while drawing, so that every line takes up exactly one line on screen (a long title is
  // cut off), and the display can be overwritten.
  char up[16] = "";
  if (progress_lines > 0) snprintf(up, sizeof up, "\033[%dF", (int)progress_lines);
  fprintf(stderr, "\033[?7l%s%s\033[K\n\033[?7h", up, line);
  progress_lines = 3;

  // Report the progress to the terminal: a percentage, or an indeterminate state (3) if the length is unknown.
  if (terminal.progress && percent != self->reported) {
    if (percent >= 0)
      fprintf(stderr, "\033]9;4;1;%d\033\\", percent);
    else
      fputs("\033]9;4;3;0\033\\", stderr);
    self->reported    = percent;
    progress_reported = 1;
  }
}

///////////////////////////////////////////////////////////////////////////////
//                                   Codec                                   //
///////////////////////////////////////////////////////////////////////////////

#define ENCODE_BUFFER_FRAMES (2 * QOAMAX_FRAME_SAMPLES + 80) //!< At least a frame plus the longest lookahead
#define ENCODE_STEP_SLICES 32                                //!< Slices per channel encoded between progress updates
//! Output bytes per encoding step: file header, frame header and LMS states, and ENCODE_STEP_SLICES slices.
#define ENCODE_STEP_SIZE(channels) (8 * (2 + 2 * (channels)) + 8 * (channels) * ENCODE_STEP_SLICES)
#define DECODE_BUFFER_FRAMES QOAMAX_FRAME_SAMPLES

//! Encodes the reader's audio into the output in fixed-size chunks.
//! @returns the number of encoded samples per channel (0 on errors, which are reported), and their total squared
//!          error.
static uint64_t encode(pcm_reader *reader, output *out, qoamax_encoder_effort effort, bool show_progress,
                       const char *title, uint64_t *total_error) {
  static int16_t samples[ENCODE_BUFFER_FRAMES * QOAMAX_MAX_CHANNEL_COUNT];
  static char encoded[ENCODE_STEP_SIZE(QOAMAX_MAX_CHANNEL_COUNT)];
  int channels = reader->channels;

  qoamax_encoder encoder;
  int err = qoamax_encoder_init(&encoder, effort, channels, reader->frames, reader->samplerate);
  if (err) {
    print_error("cannot initialize the encoder: %s", library_error_string(err));
    return 0;
  }

  progress display;
  progress_begin(&display, show_progress, title, reader->frames, reader->samplerate);

  const int16_t *pos = samples, *end = samples; // the buffered samples
  bool input_complete = false;
  uint64_t position   = 0; // encoded samples per channel
  uint64_t buffered   = 0; // samples per channel read so far
  uint64_t errors[QOAMAX_MAX_CHANNEL_COUNT];

  while (!ferror(out->file)) {
    // The output is handed out in small steps, so that the progress display is updated regularly.
    char *encoded_end         = encoded;
    const int16_t *pos_before = pos;
    err = qoamax_encoder_write_interleaved(&encoder, ENCODE_STEP_SIZE(channels), &encoded_end, end - pos, &pos,
                                           input_complete);
    if (err) {
      print_error("encoding failed: %s", library_error_string(err));
      return 0;
    }
    fwrite(encoded, 1, encoded_end - encoded, out->file);
    position += (pos - pos_before) / channels;

    qoamax_encoder_get_error(&encoder, errors);
    *total_error = 0;
    for (int i = 0; i < channels; i++) *total_error += errors[i];
    progress_update(&display, position, *total_error, channels);
    if (encoded_end != encoded) continue;

    // The encoder needs more samples: move the remaining ones to the front of the buffer and refill it.
    if (input_complete) break;
    size_t remaining = end - pos;
    memmove(samples, pos, remaining * sizeof *samples);
    size_t wanted = ENCODE_BUFFER_FRAMES - remaining / channels;
    size_t read   = pcm_reader_read(reader, &samples[remaining], wanted);
    if (reader->in.failed) return 0;

    pos            = samples;
    end            = &samples[remaining + read * channels];
    input_complete = read < wanted;
    buffered += read;
    if (buffered > UINT32_MAX) {
      print_error("the audio is too long (QOA allows at most %lu samples per channel)", (unsigned long)UINT32_MAX);
      return 0;
    }
  }

  if (position == 0 && !ferror(out->file)) print_error("'%s' contains no audio", reader->in.path);
  return position;
}

//! Reads a QOA file in fixed-size chunks.
typedef struct {
  input in;
  qoamax_decoder decoder;
  qoamax_decoder_query query;
  unsigned char data[READ_CHUNK_SIZE];
  const char *pos, *end; //!< The buffered data
  bool input_complete;
} qoa_reader;

//! Opens the input and initializes the decoder, so that an invalid file is rejected before the output is opened.
static bool qoa_reader_open(qoa_reader *reader, const char *path) {
  if (!input_open(&reader->in, path)) return false;

  size_t size            = input_read(&reader->in, reader->data, sizeof reader->data);
  reader->pos            = (const char *)reader->data;
  reader->end            = reader->pos + size;
  reader->input_complete = size < sizeof reader->data;
  if (reader->in.failed) return false;

  int err = size < 16 ? QOAMAX_ERROR_INVALID_FILE_HEADER
                      : qoamax_decoder_init(&reader->decoder, size, &reader->pos, &reader->query);
  if (err) {
    print_error("cannot decode: %s", library_error_string(err));
    return false;
  }
  return true;
}

//! Decodes the QOA input into the output (WAV or raw PCM) in fixed-size chunks.
//! @returns whether it succeeded (errors are reported), and the number of decoded samples per channel.
static bool decode(qoa_reader *reader, output *out, pcm_format format, uint64_t *decoded_frames) {
  static int16_t samples[DECODE_BUFFER_FRAMES * QOAMAX_MAX_CHANNEL_COUNT];
  const qoamax_decoder_query *query = &reader->query;

  // The file header's sample count may be 0 (streaming) or wrong. The WAV header is updated at the end if needed.
  unsigned char header[44];
  uint32_t header_data_size =
      query->total_samples ? wav_data_size(query->total_samples, query->channel_count) : UINT32_MAX;
  wav_build_header(header, query->channel_count, query->samplerate, header_data_size);
  if (format == FORMAT_WAV) fwrite(header, 1, sizeof header, out->file);

  uint64_t frames = 0;
  while (!ferror(out->file)) {
    const char *pos_before = reader->pos;
    int16_t *samples_end   = samples;
    int err                = qoamax_decoder_read_interleaved(&reader->decoder, reader->end - reader->pos, &reader->pos,
                                                             DECODE_BUFFER_FRAMES * query->channel_count, &samples_end);
    if (err) {
      print_error("cannot decode: %s", library_error_string(err));
      return false;
    }
    write_s16le(out->file, samples, samples_end - samples);
    frames += (samples_end - samples) / query->channel_count;
    if (reader->pos != pos_before || samples_end != samples) continue;

    // The decoder needs more data: move the remaining bytes to the front of the buffer and refill it.
    if (reader->input_complete) break;
    size_t remaining = reader->end - reader->pos;
    memmove(reader->data, reader->pos, remaining);
    size_t read = input_read(&reader->in, &reader->data[remaining], sizeof reader->data - remaining);
    if (reader->in.failed) return false;
    reader->pos            = (const char *)reader->data;
    reader->end            = reader->pos + remaining + read;
    reader->input_complete = read < sizeof reader->data - remaining;
  }

  *decoded_frames = frames;
  if (ferror(out->file)) return false; // reported by output_close
  if (reader->pos != reader->end || reader->decoder.frame_samples != 0) {
    print_error("cannot decode: the file is truncated or corrupt");
    return false;
  }
  if (query->total_samples != 0 && query->total_samples != frames)
    print_warning("the file header claims %lu samples per channel, but the frames contain %llu",
                  (unsigned long)query->total_samples, (unsigned long long)frames);

  uint32_t data_size = wav_data_size(frames, query->channel_count);
  if (format == FORMAT_WAV && data_size == UINT32_MAX)
    print_warning("the audio is too long for a WAV header (4 GiB), so it has no size; use --format=raw instead");
  if (format == FORMAT_WAV && data_size != header_data_size) {
    if (out->regular) {
      wav_build_header(header, query->channel_count, query->samplerate, data_size);
      output_patch(out, 0, header, sizeof header);
    } else if (header_data_size != UINT32_MAX) {
      print_warning("the WAV header's size is wrong, as the output can't be updated afterwards");
    }
  }
  return true;
}

///////////////////////////////////////////////////////////////////////////////
//                                 Commands                                  //
///////////////////////////////////////////////////////////////////////////////

static const struct {
  const char *name;
  qoamax_encoder_effort value;
} efforts[] = {
    {"ultrafast", QOAMAX_ENCODER_EFFORT_ULTRAFAST}, {"superfast", QOAMAX_ENCODER_EFFORT_SUPERFAST},
    {"veryfast", QOAMAX_ENCODER_EFFORT_VERYFAST},   {"fast", QOAMAX_ENCODER_EFFORT_FAST},
    {"medium", QOAMAX_ENCODER_EFFORT_MEDIUM},       {"slow", QOAMAX_ENCODER_EFFORT_SLOW},
    {"veryslow", QOAMAX_ENCODER_EFFORT_VERYSLOW},   {"placebo", QOAMAX_ENCODER_EFFORT_PLACEBO},
};

typedef struct {
  const char *effort_name;
  qoamax_encoder_effort effort;
  pcm_format format;
  long channels;   //!< Raw input only, 0 = default
  long samplerate; //!< Raw input only, 0 = default
  bool quiet;
  const char *input;
  const char *output;
} options;

static void print_usage(FILE *stream) {
  fprintf(stream, "Usage:\n"
                  "  qoamax encode [options] <input> <output.qoa>\n"
                  "  qoamax decode [options] <input.qoa> <output>\n"
                  "  qoamax --help\n"
                  "\n"
                  "Commands:\n"
                  "  encode       Encode a WAV or raw PCM file into a QOA file\n"
                  "  decode       Decode a QOA file into a WAV or raw PCM file\n"
                  "\n"
                  "Run 'qoamax <command> --help' for the command's options.\n");
}

static void print_encode_help(FILE *stream) {
  fprintf(stream, "Usage: qoamax encode [options] <input> <output.qoa>\n"
                  "\n"
                  "Encode a WAV file (8 to 32-bit integer or 32/64-bit float PCM) or raw 16-bit little-endian\n"
                  "PCM into a QOA file. Use '-' to read from stdin or write to stdout. Other formats can be\n"
                  "piped in, e.g.: ffmpeg -i input.flac -f wav - | qoamax encode - output.qoa\n"
                  "\n"
                  "Options:\n"
                  "  -e, --effort=LEVEL    Encoding effort (default: medium):\n"
                  "                        ultrafast, superfast, veryfast, fast,\n"
                  "                        medium, slow, veryslow, or placebo\n"
                  "  -f, --format=FORMAT   Input format: wav or raw (default: wav, or raw for\n"
                  "                        .pcm/.raw files and when --rate or --channels is given)\n"
                  "  -r, --rate=HZ         Samplerate of raw input (default: 48000)\n"
                  "  -c, --channels=N      Channel count of raw (interleaved) input (default: 1)\n"
                  "  -q, --quiet           Don't show progress and the summary\n"
                  "  -h, --help            Show this help message\n"
                  "\n"
                  "Examples:\n"
                  "  qoamax encode input.wav output.qoa\n"
                  "  qoamax encode --effort=slow input.wav output.qoa\n"
                  "  qoamax encode -r 44100 -c 2 input.pcm output.qoa\n");
}

static void print_decode_help(FILE *stream) {
  fprintf(stream, "Usage: qoamax decode [options] <input.qoa> <output>\n"
                  "\n"
                  "Decode a QOA file into a 16-bit WAV file or raw 16-bit little-endian PCM.\n"
                  "Use '-' to read from stdin or write to stdout.\n"
                  "\n"
                  "Options:\n"
                  "  -f, --format=FORMAT   Output format: wav or raw (default: raw for\n"
                  "                        .pcm/.raw files, otherwise wav)\n"
                  "  -q, --quiet           Don't show the summary\n"
                  "  -h, --help            Show this help message\n"
                  "\n"
                  "Examples:\n"
                  "  qoamax decode input.qoa output.wav\n"
                  "  qoamax decode input.qoa - | aplay\n");
}

static bool parse_long(const char *text, long min, long max, long *out) {
  char *end;
  errno      = 0;
  long value = strtol(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || value < min || value > max) return false;
  *out = value;
  return true;
}

typedef enum { PARSE_OK, PARSE_EXIT_SUCCESS, PARSE_EXIT_FAILURE } parse_result;

static parse_result parse_options(int argc, char **argv, bool encode, options *out) {
  static const struct option encode_options[] = {
      {"effort", required_argument, NULL, 'e'},
      {"format", required_argument, NULL, 'f'},
      {"rate", required_argument, NULL, 'r'},
      {"channels", required_argument, NULL, 'c'},
      {"quiet", no_argument, NULL, 'q'},
      {"help", no_argument, NULL, 'h'},
      {NULL, 0, NULL, 0},
  };
  static const struct option decode_options[] = {
      {"format", required_argument, NULL, 'f'},
      {"quiet", no_argument, NULL, 'q'},
      {"help", no_argument, NULL, 'h'},
      {NULL, 0, NULL, 0},
  };
  const char *short_options         = encode ? ":e:f:r:c:qh" : ":f:qh";
  const struct option *long_options = encode ? encode_options : decode_options;
  void (*print_help)(FILE *)        = encode ? print_encode_help : print_decode_help;

  *out   = (options){.effort_name = "medium", .effort = QOAMAX_ENCODER_EFFORT_MEDIUM};
  opterr = 0;

  int option;
  while ((option = getopt_long(argc, argv, short_options, long_options, NULL)) != -1) {
    switch (option) {
    case 'e': {
      size_t i = 0;
      while (i < sizeof efforts / sizeof efforts[0] && strcmp(optarg, efforts[i].name) != 0) i++;
      if (i == sizeof efforts / sizeof efforts[0]) {
        print_error("invalid effort '%s' (valid: ultrafast, superfast, veryfast, fast, medium, slow, veryslow, "
                    "placebo)",
                    optarg);
        return PARSE_EXIT_FAILURE;
      }
      out->effort_name = efforts[i].name;
      out->effort      = efforts[i].value;
      break;
    }
    case 'f':
      if (strcmp(optarg, "wav") == 0)
        out->format = FORMAT_WAV;
      else if (strcmp(optarg, "raw") == 0)
        out->format = FORMAT_RAW;
      else {
        print_error("invalid format '%s' (valid: wav, raw)", optarg);
        return PARSE_EXIT_FAILURE;
      }
      break;
    case 'r':
      if (!parse_long(optarg, 1, 0xFFFFFF, &out->samplerate)) {
        print_error("invalid samplerate '%s' (1 to 16777215)", optarg);
        return PARSE_EXIT_FAILURE;
      }
      break;
    case 'c':
      if (!parse_long(optarg, 1, QOAMAX_MAX_CHANNEL_COUNT, &out->channels)) {
        print_error("invalid channel count '%s' (1 to %d)", optarg, QOAMAX_MAX_CHANNEL_COUNT);
        return PARSE_EXIT_FAILURE;
      }
      break;
    case 'q':
      out->quiet = true;
      break;
    case 'h':
      print_help(stdout);
      return PARSE_EXIT_SUCCESS;
    case ':':
      print_error("option '%s' requires a value", argv[optind - 1]);
      return PARSE_EXIT_FAILURE;
    default:
      if (optopt)
        print_error("unknown option '-%c'", optopt);
      else
        print_error("unknown option '%s'", argv[optind - 1]);
      print_help(stderr);
      return PARSE_EXIT_FAILURE;
    }
  }

  if (argc - optind != 2) {
    print_error("expected an input and an output file");
    print_help(stderr);
    return PARSE_EXIT_FAILURE;
  }
  out->input  = argv[optind];
  out->output = argv[optind + 1];

  if (same_file(out->input, out->output)) {
    print_error("the input and output files must be different");
    return PARSE_EXIT_FAILURE;
  }
  return PARSE_OK;
}

//! The name of an input file in messages.
static const char *display_name(const char *path) { return strcmp(path, "-") == 0 ? "stdin" : path; }

//! Prints the line that is left on screen once a file is done, e.g.
//! "Encoded input.wav in 00:00:01.234 (100.0x realtime)" followed by the details.
static void print_summary(const char *verb, const char *path, double elapsed, uint64_t frames, int32_t samplerate,
                          const char *details) {
  bool colors = terminal.color;
  char time_text[24];
  format_duration(time_text, elapsed, true);
  fprintf(stderr, "%s%s %s%s%s in %s (%.1fx realtime)%s\n", colors ? "\033[0;37m" : "", verb,
          colors ? "\033[0;95m" : "", display_name(path), colors ? "\033[0m" : "", time_text,
          elapsed > 0 ? frames / (samplerate * elapsed) : 0, details);
}

static int run_encode(const options *opts) {
  pcm_reader reader;
  output out;
  if (!pcm_reader_open(&reader, opts->input, opts->format, opts->channels, opts->samplerate) ||
      !output_open(&out, opts->output)) {
    input_close(&reader.in);
    return EXIT_FAILURE;
  }

  // The title of the progress display, which is only shown on a terminal.
  char title[1024], duration[24] = "unknown length";
  if (reader.frames > 0) format_duration(duration, (double)reader.frames / reader.samplerate, true);
  bool colors = terminal.color;
  snprintf(title, sizeof title, "%sEncoding %s%s%s (%d channels, %d Hz, %s, effort %s)", colors ? "\033[0;37m" : "",
           colors ? "\033[0;95m" : "", display_name(opts->input), colors ? "\033[0m" : "", reader.channels,
           reader.samplerate, duration, opts->effort_name);

  uint64_t error  = 0;
  double start    = now_seconds();
  uint64_t frames = encode(&reader, &out, opts->effort, !opts->quiet && terminal.interactive, title, &error);
  progress_end();
  bool ok = frames > 0 && !reader.in.failed && !ferror(out.file);
  if (ok && frames != reader.frames) {
    // The input's length wasn't known (or was wrong), so the file header needs the actual sample count.
    if (out.regular) {
      unsigned char count[4] = {frames >> 24, frames >> 16, frames >> 8, frames};
      output_patch(&out, 4, count, sizeof count);
    } else if (reader.frames == 0) {
      print_warning("the QOA header has no sample count (streaming mode), as the output can't be updated "
                    "afterwards; some decoders don't support this");
    } else {
      print_warning("the QOA header's sample count is wrong, as the output can't be updated afterwards");
    }
  }
  input_close(&reader.in);
  if (!output_close(&out, ok)) return EXIT_FAILURE;

  if (!opts->quiet) {
    char psnr_text[24], details[64];
    format_psnr(psnr_text, error, frames * reader.channels);
    snprintf(details, sizeof details, ", PSNR %s, effort %s", psnr_text, opts->effort_name);
    print_summary("Encoded", opts->input, now_seconds() - start, frames, reader.samplerate, details);
  }
  return EXIT_SUCCESS;
}

static int run_decode(const options *opts) {
  pcm_format format = opts->format;
  if (format == FORMAT_AUTO)
    format = has_extension(opts->output, ".pcm") || has_extension(opts->output, ".raw") ? FORMAT_RAW : FORMAT_WAV;

  qoa_reader reader;
  output out;
  if (!qoa_reader_open(&reader, opts->input) || !output_open(&out, opts->output)) {
    input_close(&reader.in);
    return EXIT_FAILURE;
  }

  uint64_t frames = 0;
  double start    = now_seconds();
  bool ok         = decode(&reader, &out, format, &frames);
  input_close(&reader.in);
  if (!output_close(&out, ok)) return EXIT_FAILURE;

  if (!opts->quiet) {
    char duration[24], details[96];
    format_duration(duration, (double)frames / reader.query.samplerate, true);
    snprintf(details, sizeof details, ", %s, %d channels, %d Hz", duration, reader.query.channel_count,
             reader.query.samplerate);
    print_summary("Decoded", opts->input, now_seconds() - start, frames, reader.query.samplerate, details);
  }
  return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    print_usage(stderr);
    return EXIT_FAILURE;
  }
  if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
    print_usage(stdout);
    return EXIT_SUCCESS;
  }

  bool encode = strcmp(argv[1], "encode") == 0;
  if (!encode && strcmp(argv[1], "decode") != 0) {
    print_error("unknown command '%s'", argv[1]);
    print_usage(stderr);
    return EXIT_FAILURE;
  }

  options opts;
  switch (parse_options(argc - 1, argv + 1, encode, &opts)) {
  case PARSE_EXIT_SUCCESS:
    return EXIT_SUCCESS;
  case PARSE_EXIT_FAILURE:
    return EXIT_FAILURE;
  case PARSE_OK:
    break;
  }

  terminal_detect();
  signal(SIGINT, handle_interrupt);
  signal(SIGTERM, handle_interrupt);
  signal(SIGHUP, handle_interrupt);
  return encode ? run_encode(&opts) : run_decode(&opts);
}
