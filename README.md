# QOAMAX - High-Quality QOA Encoder/Decoder Library and CLI tool

QOAMAX is an encoder + decoder implementation for the QOA codec and file format,
usable as a library inside other applications as well as a standalone CLI tool.

The library can be used stand-alone without any dependencies or even memory allocations,
but requires a recent gcc or clang (C23 with GNU extensions, see [Compiler support](#compiler-support)).

QOAMAX features more optimized encoder/decoder implementations,
often resulting in a 2x performance increase over qoaconv.

QOAMAX implements multiple parametrizable encoders,
ranging from being 5x as fast as the reference encoder to
trading hours of compute for large quality improvements.

On a pop song, the tradeoff currently looks like this (Ryzen 5900X, placebo and ultrafast missing):  
![Graph showing the speed-quality tradeoff](assets/qoa_pareto.svg)

## Compilation

Simply run `make` inside the repo root to build the CLI tool.
This builds the optimized `build/qoamax` and a sanitizer-enabled `build/qoamax-debug`;
use `make release` to only build `build/qoamax`, and `make test` to run the tests.
`make lib` builds the static and shared library (`build/libqoamax.a`, `build/libqoamax.so`), and
`make install` installs the CLI, the library, its headers and a pkg-config file (`PREFIX=/usr/local`, `DESTDIR`).
CMake and Meson builds work too, see [Using the library](#using-the-library).

Notice that some encoder implementations are noticeably faster when compiled with clang
instead of gcc.

### Compiler support

The library's source (`src/qoamax.c`) is C23 and also uses GNU extensions, without fallbacks, so it needs gcc or clang
(tested with gcc 16 and clang 23; older versions with C23 support may work):

- builtins: `__builtin_expect`, `__builtin_ctz`, `__builtin_add_overflow` and `__builtin_cpu_supports`
  (runtime SIMD detection),
- vector extensions (`__attribute__((vector_size))`) for the SIMD encoder searches,
- `[[gnu::...]]` attributes such as `always_inline`, `noinline` and `target` (SIMD variants of functions),
- variably modified (VLA) parameters, e.g. `samples[static restrict samples_count]`. They are standard in C23, but
  compilers without full C23 support may lack them (they were optional in C11).

MSVC can't compile it; clang-cl should work, but is untested. The public header `src/qoamax.h` has none of these
requirements: it is plain C11 and C++11.

## Usage

The tool features a `decode` and an `encode` command, which should both be self-explanatory:

```sh
$ qoamax -h
$ qoamax encode -h
$ qoamax encode input.wav output.qoa --effort=slow
$ qoamax decode output.qoa decoded.wav
```

`--effort` selects the speed/quality tradeoff of the encoder: `ultrafast`, `superfast`, `veryfast`, `fast`,
`medium` (default), `slow`, `veryslow` or `placebo`.

`encode` reads WAV files (8 to 32-bit integer or 32/64-bit float PCM) and raw 16-bit little-endian PCM.
Raw input is used for `.pcm`/`.raw` files, with `--format=raw`, or when `--rate`/`--channels` is given
(defaults: 48000 Hz, mono).
`decode` writes 16-bit WAV files, or raw PCM for `.pcm`/`.raw` files and with `--format=raw`.
Use `-` for stdin/stdout to convert other formats with e.g. ffmpeg:

```sh
$ ffmpeg -i input.flac -f wav - | qoamax encode - output.qoa
$ qoamax decode input.qoa - | ffmpeg -i - output.flac
```

## Using the library

The library has no dependencies and doesn't allocate memory. It consists of `src/qoamax.c`, which needs gcc or clang
(see [Compiler support](#compiler-support); clang is faster), and the header `src/qoamax.h`, which is plain C11 and
C++11, so the rest of your project can use any compiler and standard from those on. Build the library with
optimizations: the encoder is many times slower without them.

**Copy the source** into your project: `src/qoamax.h`, `src/qoamax_plain.h` (plain declarations for C++ and C
compilers without VLA parameters, included by `qoamax.h`), `src/qoamax.c`, `src/qoamax_lanes.h` and
`src/qoamax_beam.h`. Compile `qoamax.c` as its own file, with
e.g. `-std=c23 -O3`.

**CMake** (3.21 or later), with `FetchContent` or `add_subdirectory`:

```cmake
include(FetchContent)
FetchContent_Declare(qoamax GIT_REPOSITORY https://gitlab.com/BlobCodes/qoamax.git GIT_TAG main)
FetchContent_MakeAvailable(qoamax)
target_link_libraries(your_target PRIVATE qoamax::qoamax)
```

After installing it (`cmake -B build && cmake --build build && cmake --install build`), use `find_package(qoamax)`
and the same `qoamax::qoamax` target. `BUILD_SHARED_LIBS=ON` builds a shared library, `QOAMAX_BUILD_CLI` and
`QOAMAX_INSTALL` select whether the CLI is built and the files are installed (by default only if qoamax is the
top-level project). Its default build type is `Release`, but as a subproject it uses your project's build type.

**Meson** (1.4 or later), as a subproject with a wrap file `subprojects/qoamax.wrap`:

```ini
[wrap-git]
url = https://gitlab.com/BlobCodes/qoamax.git
revision = main

[provide]
dependency_names = qoamax
```

Then `dependency('qoamax')` finds it, installed or as a subproject. The CLI isn't built as a subproject
(option `cli`). On Windows, build it as a static library (`default_library=static`), as Meson doesn't export the
symbols from a DLL.

**Installed** (by `make install`, CMake or Meson): `pkg-config --cflags --libs qoamax`.
Before version 1.0, every minor version may change the ABI (the shared library's soname is `libqoamax.so.0.1`).

### Example

Encoding interleaved 16-bit samples into a memory buffer (the whole program, which decodes it again, is
[`test/example.c`](test/example.c)):

```c
#include <qoamax.h>

size_t size = qoamax_encoder_get_encoded_size(channels, count); // count = samples per channel
char *file  = malloc(size);
qoamax_encoder encoder;
if (qoamax_encoder_init(&encoder, QOAMAX_ENCODER_EFFORT_MEDIUM, channels, count, samplerate) != 0) return 1;
char *out         = file;
const int16_t *in = samples;
if (qoamax_encoder_write_interleaved(&encoder, size, &out, count * channels, &in, true) != 0) return 1;
// out - file bytes were written.
```

The encoder and decoder functions advance the buffer pointers to report their progress, and return when the input
or output runs out (0 = OK, negative values are `QOAMAX_ERROR_*` codes), so they can also stream in chunks.
`qoamax_encoder_write` and `qoamax_decoder_read` take one buffer per channel instead of interleaved samples, and
`qoamax_encode_slice`/`qoamax_decode_slice` encode/decode single 20-sample slices for custom formats.
See [`src/qoamax.h`](src/qoamax.h) for the documentation.
