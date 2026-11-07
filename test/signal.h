// Deterministic test signal, shared by the library tests and the WAV generator of the CLI tests.
//
// It only uses integer math, so the signal (and the expected checksums of the encoded files) are the same on every
// platform and compiler. It cycles through sections that stress different parts of the encoder: a chirp, a chirp with
// noise, noise, a square wave plus the chirp (clipping) and near silence with clicks.
#pragma once
#include <stddef.h>
#include <stdint.h>

#define TEST_SECTION_FRAMES 2000 //!< Length of each signal section in samples per channel

//! Returns the test signal's sample of the given channel at the given position.
static inline int16_t test_signal(uint32_t seed, int channel, size_t position) {
  // Noise from a hash of the position, so that samples can be generated in any order.
  uint32_t noise = (uint32_t)position * 2654435761u ^ (seed + 0x9E3779B9u * (uint32_t)(channel + 1));
  noise ^= noise >> 15;
  noise *= 0x2C1B3C6Du;
  noise ^= noise >> 12;
  int32_t white = (int32_t)(noise & 0xFFFF) - 32768; // -32768..32767

  // A chirp: a parabolic approximation of a sine wave, with a slowly rising frequency.
  uint32_t phase = (uint32_t)((uint64_t)position * (60000 + 9000 * (uint32_t)channel) +
                              (uint64_t)position * position / 64 + seed * 977u);
  int32_t x      = (int32_t)(phase >> 16) - 32768;       // -32768..32767
  int32_t tone   = x * (32768 - (x < 0 ? -x : x)) >> 14; // about -16384..16384

  int32_t sample;
  switch ((position / TEST_SECTION_FRAMES + (size_t)channel) % 5) {
  case 0:
    sample = tone;
    break;
  case 1:
    sample = tone + white / 16;
    break;
  case 2:
    sample = white / 4;
    break;
  case 3: // a square wave plus the chirp, clipping at the peaks
    sample = ((position / (37 + 11 * (size_t)channel)) % 2 ? 24000 : -24000) + tone;
    break;
  default:
    sample = white / 8192 + (position % 500 == 0 ? 20000 : 0);
    break;
  }
  return sample < INT16_MIN ? INT16_MIN : sample > INT16_MAX ? INT16_MAX : (int16_t)sample;
}
