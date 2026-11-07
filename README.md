# QOAMAX - High-Quality QOA Encoder/Decoder Library and CLI tool

QOAMAX is an encoder + decoder implementation for the QOA codec and file format,
usable as a library inside other applications as well as a standalone CLI tool.

The library can be used stand-alone without any dependencies or even memory allocations,
but requires a modern C23 compiler.

QOAMAX features more optimized encoder/decoder implementations,
often resulting in a 2x performance increase over qoaconv.

QOAMAX implements multiple parametrizable encoders,
ranging from being 5x as fast as the reference encoder to
trading hours of compute for large quality improvements.

## Compilation

Simply run `make` inside the repo root to build the CLI tool.
This builds the optimized `build/qoamax` and a sanitizer-enabled `build/qoamax-debug`;
use `make release` to only build `build/qoamax`, and `make test` to run the tests.

Notice that some encoder implementations are noticeably faster when compiled with clang
instead of gcc, but it works with any modern C23 compiler.

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
