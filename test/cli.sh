#!/usr/bin/env bash
# CLI tests, run by `make test`.
#
# Usage: test/cli.sh QOAMAX WAVGEN
#   QOAMAX  The CLI to test (make test uses the sanitizer build)
#   WAVGEN  The test signal generator (test/wavgen.c)
#
# Compares against the reference decoder (qoaconv) if it is installed.
set -u

if [ $# -ne 2 ]; then
  echo "usage: $0 QOAMAX WAVGEN" >&2
  exit 2
fi
Q=$(realpath "$1")
W=$(realpath "$2")
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
# The test files are small (below 1 MB). The limit keeps a broken build from filling the disk (or RAM, as /tmp is often
# a tmpfs) with endless output.
ulimit -f 10240 # KB
cd "$T" || exit 2

checks=0
failures=0

fail() {
  echo "FAIL: $*" >&2
  failures=$((failures + 1))
}

# Commands get no stdin (so they can't wait for input) and a time limit.
if command -v timeout >/dev/null; then
  LIMIT=(timeout 60)
else
  LIMIT=()
fi

#! Runs a command, saving its stdout, stderr and exit status.
run() {
  "${LIMIT[@]}" "$@" </dev/null >stdout 2>stderr
  status=$?
  [ $status -eq 124 ] && echo "(timed out)" >>stderr
}

#! expect_ok DESCRIPTION COMMAND...: the command succeeds.
expect_ok() {
  local description=$1
  shift
  checks=$((checks + 1))
  run "$@"
  if [ $status -ne 0 ]; then
    fail "$description: exit status $status: $(head -c 300 stderr)"
  elif grep -qE "ERROR: AddressSanitizer|runtime error" stderr; then
    fail "$description: sanitizer report: $(head -c 300 stderr)"
  fi
}

#! expect_error DESCRIPTION MESSAGE COMMAND...: the command fails with exit status 1, printing the message.
expect_error() {
  local description=$1 message=$2
  shift 2
  checks=$((checks + 1))
  run "$@"
  if [ $status -ne 1 ]; then
    fail "$description: exit status $status, expected 1: $(head -c 300 stderr)"
  elif ! grep -qF -- "$message" stderr; then
    fail "$description: expected the message '$message', got: $(head -c 300 stderr)"
  fi
}

#! expect_message DESCRIPTION MESSAGE: the last command printed the message to stderr.
expect_message() {
  checks=$((checks + 1))
  grep -qF -- "$2" stderr || fail "$1: expected the message '$2', got: $(head -c 300 stderr)"
}

#! expect_same DESCRIPTION FILE1 FILE2: both files exist and are identical.
expect_same() {
  checks=$((checks + 1))
  cmp -s "$2" "$3" || fail "$1: '$2' and '$3' differ"
}

#! expect_bytes DESCRIPTION FILE OFFSET HEX: the file contains the bytes at the offset.
expect_bytes() {
  local actual
  checks=$((checks + 1))
  actual=$(od -An -tx1 -j "$3" -N $(($(echo "$4" | wc -w))) "$2" | tr -s ' \n' ' ' | sed 's/^ //; s/ $//')
  [ "$actual" = "$4" ] || fail "$1: bytes $3.. of '$2' are '$actual', expected '$4'"
}

#! expect_missing DESCRIPTION FILE: the file doesn't exist.
expect_missing() {
  checks=$((checks + 1))
  [ ! -e "$2" ] || fail "$1: '$2' exists"
}

FRAMES=12000 # 2 channels, 44100 Hz (wavgen's samplerate)

#################################################################################
# Arguments and help

expect_ok "help" "$Q" --help
grep -q "Usage:" stdout || fail "help: no usage on stdout"
expect_ok "version" "$Q" -v
grep -qx "qoamax [0-9]*\.[0-9]*\.[0-9]*" stdout || fail "version: no version on stdout"
expect_ok "long version" "$Q" --version
grep -qx "qoamax [0-9]*\.[0-9]*\.[0-9]*" stdout || fail "long version: no version on stdout"
expect_ok "encode help" "$Q" encode -h
grep -q -- "--effort" stdout || fail "encode help: no --effort"
expect_ok "decode help" "$Q" decode --help
grep -q -- "--format" stdout || fail "decode help: no --format"

expect_error "no arguments" "Usage:" "$Q"
expect_error "unknown command" "unknown command 'transcode'" "$Q" transcode a b
expect_error "unknown option" "unknown option '-x'" "$Q" encode -x a b
expect_error "unknown long option" "unknown option '--loud'" "$Q" decode --loud a b
expect_error "invalid effort" "invalid effort 'turbo'" "$Q" encode -e turbo a b
expect_error "missing value" "requires a value" "$Q" encode a b -e
expect_error "invalid channel count" "invalid channel count '9'" "$Q" encode -c 9 a b
expect_error "invalid samplerate" "invalid samplerate '0'" "$Q" encode -r 0 a b
expect_error "invalid format" "invalid format 'mp3'" "$Q" decode -f mp3 a b
expect_error "one file" "expected an input and an output file" "$Q" encode a
expect_error "missing input" "cannot open 'missing.wav'" "$Q" encode missing.wav out.qoa
expect_missing "missing input" out.qoa

"$W" raw 2 $FRAMES in.pcm || exit 2
expect_error "same input and output" "must be different" "$Q" encode -c 2 in.pcm in.pcm

#################################################################################
# Input formats: every WAV format holds the same signal, so all give the same QOA file as the raw input.

expect_ok "encode raw" "$Q" encode -e superfast -c 2 -r 44100 in.pcm raw.qoa
for format in u8 s16 s24 s32 f32 f64 s16x s24x f32x; do
  "$W" $format 2 $FRAMES in_$format.wav || exit 2
  expect_ok "encode $format WAV" "$Q" encode -e superfast in_$format.wav $format.qoa
  expect_same "$format WAV" raw.qoa $format.qoa
done
expect_ok "raw by --format" "$Q" encode -e superfast -f raw -c 2 -r 44100 in_s16.wav forced.qoa
cmp -s raw.qoa forced.qoa && fail "--format=raw: the WAV header was not encoded as audio"
cp in.pcm in.raw
expect_ok "raw by extension .raw" "$Q" encode -e superfast -c 2 -r 44100 in.raw ext.qoa
expect_same "raw by extension .raw" raw.qoa ext.qoa

for effort in ultrafast superfast veryfast fast medium slow veryslow superslow; do
  "$W" raw 1 300 short.pcm || exit 2
  expect_ok "effort $effort" "$Q" encode -e $effort -c 1 short.pcm short_$effort.qoa
done
: >empty.pcm
expect_error "effort placebo (accepted)" "contains no audio" "$Q" encode -e placebo empty.pcm placebo.qoa

#################################################################################
# Decoding and output formats

expect_ok "encode WAV" "$Q" encode in_s16.wav a.qoa
expect_ok "decode to WAV" "$Q" decode a.qoa a.wav
expect_ok "decode to raw" "$Q" decode a.qoa a.pcm
checks=$((checks + 1))
[ "$(wc -c <a.pcm)" -eq $((FRAMES * 4)) ] || fail "decode to raw: $(wc -c <a.pcm) bytes"
tail -c +45 a.wav >a_data.pcm
expect_same "WAV data equals raw output" a.pcm a_data.pcm
expect_bytes "WAV header" a.wav 0 "52 49 46 46"                  # RIFF
expect_bytes "WAV data size" a.wav 40 "80 bb 00 00"              # 48000 bytes
expect_bytes "WAV format" a.wav 20 "01 00 02 00 44 ac 00 00"     # PCM, 2 channels, 44100 Hz
expect_ok "decode --format=raw" "$Q" decode -f raw a.qoa a_forced.wav
expect_same "decode --format=raw" a.pcm a_forced.wav
expect_ok "decode --format=wav" "$Q" decode -f wav a.qoa a_forced.pcm
expect_same "decode --format=wav" a.wav a_forced.pcm

if command -v qoaconv >/dev/null; then
  checks=$((checks + 1))
  if qoaconv a.qoa reference.wav >/dev/null 2>&1; then
    tail -c +45 reference.wav >reference.pcm
    expect_same "the reference decoder (qoaconv) gives the same samples" a.pcm reference.pcm
  else
    fail "qoaconv can't decode the encoded file"
  fi
else
  echo "  (qoaconv not found, skipping the comparison with the reference decoder)"
fi

#################################################################################
# Pipes

expect_ok "encode from stdin" bash -c "cat in_s16.wav | '$Q' encode - stdin.qoa"
expect_same "encode from stdin" a.qoa stdin.qoa

# A WAV of unknown size (like ffmpeg's piped output): the sample count is patched into the header afterwards.
expect_ok "encode unknown size" bash -c "'$W' s16 2 $FRAMES - unknown-size | '$Q' encode - unknown.qoa"
expect_same "encode unknown size" a.qoa unknown.qoa

# Unknown size into a pipe: the header can't be patched, so the file is in streaming mode (sample count 0).
expect_ok "encode to stdout" bash -c "'$W' s16 2 $FRAMES - unknown-size | '$Q' encode - - | cat >streaming.qoa"
expect_message "encode to stdout" "streaming mode"
expect_bytes "streaming mode header" streaming.qoa 4 "00 00 00 00"
tail -c +9 a.qoa >a_frames
tail -c +9 streaming.qoa >streaming_frames
expect_same "streaming mode frames" a_frames streaming_frames

expect_ok "decode streaming mode" "$Q" decode streaming.qoa streaming.wav
expect_same "decode streaming mode (WAV size patched)" a.wav streaming.wav
expect_ok "decode to stdout" bash -c "'$Q' decode a.qoa - | cat >stdout.wav"
expect_same "decode to stdout" a.wav stdout.wav
expect_ok "decode from stdin" bash -c "cat a.qoa | '$Q' decode - stdin.wav"
expect_same "decode from stdin" a.wav stdin.wav
expect_ok "decode streaming mode to stdout" bash -c "cat streaming.qoa | '$Q' decode - - | cat >unknown.wav"
expect_bytes "WAV of unknown size" unknown.wav 40 "ff ff ff ff"
tail -c +45 unknown.wav >unknown.pcm
expect_same "WAV of unknown size" a.pcm unknown.pcm

#################################################################################
# Invalid input: existing outputs are kept, partial outputs are removed.

echo "keep" >keep.qoa
echo "keep" >keep.wav
echo "not audio" >garbage.txt
expect_error "encode invalid input" "is not a WAV file" "$Q" encode garbage.txt keep.qoa
expect_error "decode invalid input" "not a QOA file" "$Q" decode garbage.txt keep.wav
checks=$((checks + 1))
[ "$(cat keep.qoa keep.wav)" = "$(printf 'keep\nkeep')" ] || fail "an existing output was overwritten"

head -c $(($(wc -c <a.qoa) - 100)) a.qoa >truncated.qoa
expect_error "decode truncated file" "truncated or corrupt" "$Q" decode truncated.qoa truncated.wav
expect_missing "decode truncated file" truncated.wav

expect_error "encode empty input" "contains no audio" "$Q" encode empty.pcm empty.qoa
expect_missing "encode empty input" empty.qoa

head -c 30000 in_s16.wav >truncated.wav
expect_ok "encode truncated WAV" "$Q" encode truncated.wav truncated.qoa
expect_message "encode truncated WAV" "the WAV file is truncated"

head -c 4001 in.pcm >odd.pcm
expect_ok "encode odd raw size" "$Q" encode odd.pcm odd.qoa
expect_message "encode odd raw size" "ignoring the last 1 bytes"

#################################################################################
# Messages: a summary line (without escape sequences, as stderr isn't a terminal), nothing with --quiet

expect_ok "encode summary" "$Q" encode in.pcm -c 2 summary.qoa
expect_message "encode summary" "Encoded in.pcm in"
expect_ok "decode summary" "$Q" decode summary.qoa summary.wav
expect_message "decode summary" "Decoded summary.qoa in"
checks=$((checks + 1))
grep -q $'\033' stderr && fail "escape sequences on stderr, which is not a terminal"

expect_ok "encode --quiet" "$Q" encode -q -c 2 in.pcm quiet.qoa
checks=$((checks + 1))
[ -s stderr ] && fail "encode --quiet printed: $(head -c 300 stderr)"
expect_ok "decode --quiet" "$Q" decode --quiet quiet.qoa quiet.wav
checks=$((checks + 1))
[ -s stderr ] && fail "decode --quiet printed: $(head -c 300 stderr)"

echo "test/cli.sh: $failures of $checks checks failed"
[ $failures -eq 0 ]
