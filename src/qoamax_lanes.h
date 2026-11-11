// The SIMD scalefactor search (one lane per scalefactor). Not a public header: qoamax.c includes this file once per
// SIMD variant, after defining
// - QOAMAX_VARIANT_LANES:  the number of lanes (2, 4, 8 or 16),
// - QOAMAX_VARIANT_SUFFIX: appended to every name defined here (empty for the compile target's default variant),
// - QOAMAX_VARIANT_TARGET: an attribute for every function, e.g. [[gnu::target("avx2")]] (or empty).
// Every variant produces identical results.

static_assert(QOAMAX_VARIANT_LANES >= 2 && QOAMAX_VARIANT_LANES <= 16 && 16 % QOAMAX_VARIANT_LANES == 0);

// The names defined here, renamed per variant (e.g. qoamax_hillclimb_lanes__x8), so that the code below reads as usual.
#define qoamax_vi32 QOAMAX_VARIANT_NAME(qoamax_vi32)
#define qoamax_vu32 QOAMAX_VARIANT_NAME(qoamax_vu32)
#define qoamax_lanes_table QOAMAX_VARIANT_NAME(qoamax_lanes_table)
#define qoamax_lanes_state QOAMAX_VARIANT_NAME(qoamax_lanes_state)
#define qoamax_hillclimb_lanes_step QOAMAX_VARIANT_NAME(qoamax_hillclimb_lanes_step)
#define qoamax_hillclimb_lanes_impl QOAMAX_VARIANT_NAME(qoamax_hillclimb_lanes_impl)
#define qoamax_hillclimb_lanes__partial QOAMAX_VARIANT_NAME(qoamax_hillclimb_lanes__partial)
#define qoamax_hillclimb_lanes_unclamped__partial QOAMAX_VARIANT_NAME(qoamax_hillclimb_lanes_unclamped__partial)
#define qoamax_hillclimb_lanes QOAMAX_VARIANT_NAME(qoamax_hillclimb_lanes)
#define qoamax_hillclimb_lanes_unclamped QOAMAX_VARIANT_NAME(qoamax_hillclimb_lanes_unclamped)
#define qoamax_encode_slice__hillclimb_scalefactor_search_lanes_impl                \
  QOAMAX_VARIANT_NAME(qoamax_encode_slice__hillclimb_scalefactor_search_lanes_impl)
#define qoamax_encode_slice__hillclimb_scalefactor_search_lanes                \
  QOAMAX_VARIANT_NAME(qoamax_encode_slice__hillclimb_scalefactor_search_lanes)
#define qoamax_encode_slice__ultrafast_lanes QOAMAX_VARIANT_NAME(qoamax_encode_slice__ultrafast_lanes)
#define qoamax_encode_slice__hillclimb_scalefactor_search_lanes_unclamped                \
  QOAMAX_VARIANT_NAME(qoamax_encode_slice__hillclimb_scalefactor_search_lanes_unclamped)

typedef int32_t qoamax_vi32 __attribute__((vector_size(QOAMAX_VARIANT_LANES * 4)));
typedef uint32_t qoamax_vu32 __attribute__((vector_size(QOAMAX_VARIANT_LANES * 4)));

//! The table columns of one lane window (see @ref qoamax_lanes_dqt).
typedef struct {
  qoamax_vi32 q0, q1, q2, d0, d1, d2, d3;
} qoamax_lanes_table;

//! The per-lane state of @ref qoamax_hillclimb_lanes_impl. Once inlined, the compiler keeps it in registers.
typedef struct {
  qoamax_vi32 h0, h1, h2, h3, w0, w1, w2, w3;
  // 64-bit errors, split into two halves. Both are biased by 2^31 (stored as value ^ INT32_MIN), so that signed
  // comparisons give their unsigned order: x86 has no unsigned vector comparisons before AVX-512.
  qoamax_vu32 error_low, error_high;
} qoamax_lanes_state;

//! Encodes sample i in every lane of @ref qoamax_hillclimb_lanes_impl.
//! Returns false once every lane's error reaches `bound` (checked only after some samples).
QOAMAX_VARIANT_TARGET
[[gnu::always_inline]]
static inline bool qoamax_hillclimb_lanes_step(qoamax_lanes_state *st, const qoamax_lanes_table *t, int i,
                                               int samples_count, int32_t sample_value, int pos, uint64_t bound,
                                               qoamax_scalefactor_results *restrict out, bool unclamped) {
  const qoamax_vi32 zero = {};
  const uint32_t bias    = 0x80000000;
  qoamax_vi32 sample     = zero + sample_value;

  // Unsigned to keep overflow defined; the result is the same as in the scalar code.
  qoamax_vu32 sum       = (qoamax_vu32)st->w0 * (qoamax_vu32)st->h0 + (qoamax_vu32)st->w1 * (qoamax_vu32)st->h1 +
                          (qoamax_vu32)st->w2 * (qoamax_vu32)st->h2 + (qoamax_vu32)st->w3 * (qoamax_vu32)st->h3;
  qoamax_vi32 predicted = (qoamax_vi32)sum >> 13;
  qoamax_vi32 residual  = sample - predicted;
  qoamax_vi32 negative  = residual <= 0;
  qoamax_vi32 abs       = (residual ^ (residual >> 31)) - (residual >> 31);
  // nested: above2 implies above1 implies above0
  qoamax_vi32 above0 = abs > t->q0, above1 = abs > t->q1, above2 = abs > t->q2;

  qoamax_vi32 magnitude   = t->d0 + (above0 & t->d1) + (above1 & t->d2) + (above2 & t->d3);
  qoamax_vi32 dequantized = (magnitude ^ negative) - negative;
  qoamax_vi32 quant       = (-(above0 + above1 + above2) << 1) | (negative & 1);
  memcpy(&out->quant[i][pos], &quant, sizeof quant);

  qoamax_vi32 reconstructed  = predicted + dequantized;
  qoamax_vi32 unclamped_diff = sample - reconstructed;
  qoamax_vi32 too_low        = reconstructed < INT16_MIN;
  reconstructed              = (reconstructed & ~too_low) | (INT16_MIN & too_low);
  qoamax_vi32 too_high       = reconstructed > INT16_MAX;
  reconstructed              = (reconstructed & ~too_high) | (INT16_MAX & too_high);

  qoamax_vu32 error_low = st->error_low, error_high = st->error_high;
  if (unclamped) {
    // The difference before clamping can exceed 16 bits (|diff| < 2^19), so its square is computed from 16-bit
    // halves: diff^2 = high^2 * 2^32 + 2 * high * low * 2^16 + low^2.
    qoamax_vu32 diff = (qoamax_vu32)((unclamped_diff ^ (unclamped_diff >> 31)) - (unclamped_diff >> 31));
    qoamax_vu32 low = diff & 0xFFFF, high = diff >> 16;
    qoamax_vu32 cross     = 2 * high * low; // < 2^20
    qoamax_vu32 new_error = error_low + low * low;
    error_high -= (qoamax_vu32)((qoamax_vi32)new_error < (qoamax_vi32)error_low);
    error_low = new_error;
    new_error = error_low + (cross << 16);
    error_high -= (qoamax_vu32)((qoamax_vi32)new_error < (qoamax_vi32)error_low);
    error_low = new_error;
    error_high += (cross >> 16) + high * high;
  } else {
    qoamax_vi32 diff      = sample - reconstructed;
    qoamax_vu32 new_error = error_low + (qoamax_vu32)diff * (qoamax_vu32)diff;
    error_high -= (qoamax_vu32)((qoamax_vi32)new_error < (qoamax_vi32)error_low); // carry (the comparison yields -1)
    error_low = new_error;
  }
  st->error_low = error_low, st->error_high = error_high;

  qoamax_vi32 delta = dequantized >> 4;
  qoamax_vi32 sign0 = st->h0 >> 31, sign1 = st->h1 >> 31, sign2 = st->h2 >> 31, sign3 = st->h3 >> 31;
  st->w0 += (delta ^ sign0) - sign0;
  st->w1 += (delta ^ sign1) - sign1;
  st->w2 += (delta ^ sign2) - sign2;
  st->w3 += (delta ^ sign3) - sign3;
  st->h0 = st->h1, st->h1 = st->h2, st->h2 = st->h3, st->h3 = reconstructed;

  // Many windows are dead after a few samples, later ones rarely die: check after the first four samples, then after
  // every second one.
  if ((i < 4 || i % 2 == 1) && i + 1 < samples_count) {
    int32_t bound_high = (uint32_t)(bound >> 32) ^ bias, bound_low = (uint32_t)bound ^ bias;
    qoamax_vi32 high = (qoamax_vi32)error_high, low = (qoamax_vi32)error_low;
    qoamax_vi32 alive = (high < bound_high) | ((high == bound_high) & (low < bound_low));
    bool any_alive    = false;
    for (int lane = 0; lane < QOAMAX_VARIANT_LANES; lane++) any_alive |= alive[lane] != 0;
    if (!any_alive) return false;
  }
  return true;
}

//! Runs @ref qoamax_encode_hillclimb_search (or its unclamped variant) for the QOAMAX_VARIANT_LANES scalefactors
//! starting at table column `column` (scalefactor column & 15), one per lane, and stores the results from position
//! `pos` on. Stops early once every lane's error reaches `bound`, returning false (the results are unspecified then).
QOAMAX_VARIANT_TARGET
[[gnu::always_inline]]
static inline bool qoamax_hillclimb_lanes_impl(const qoamax_lms *lms, int samples_count,
                                               const int16_t samples[static restrict samples_count], int column,
                                               int pos, uint64_t bound, qoamax_scalefactor_results *restrict out,
                                               bool unclamped) {
  qoamax_assume(samples_count >= 1 && samples_count <= 20);

  qoamax_lanes_table t;
  memcpy(&t.q0, &qoamax_lanes_qt[0][column], sizeof t.q0);
  memcpy(&t.q1, &qoamax_lanes_qt[1][column], sizeof t.q1);
  memcpy(&t.q2, &qoamax_lanes_qt[2][column], sizeof t.q2);
  memcpy(&t.d0, &qoamax_lanes_dqt[0][column], sizeof t.d0);
  memcpy(&t.d1, &qoamax_lanes_dqt[1][column], sizeof t.d1);
  memcpy(&t.d2, &qoamax_lanes_dqt[2][column], sizeof t.d2);
  memcpy(&t.d3, &qoamax_lanes_dqt[3][column], sizeof t.d3);

  const qoamax_vi32 zero = {};
  const uint32_t bias    = 0x80000000;
  qoamax_lanes_state st  = {
      .h0         = zero + lms->history[0],
      .h1         = zero + lms->history[1],
      .h2         = zero + lms->history[2],
      .h3         = zero + lms->history[3],
      .w0         = zero + lms->weights[0],
      .w1         = zero + lms->weights[1],
      .w2         = zero + lms->weights[2],
      .w3         = zero + lms->weights[3],
      .error_low  = (qoamax_vu32){} + bias,
      .error_high = (qoamax_vu32){} + bias,
  };

  if (samples_count == 20) {
    // The loop is bound by its vector multiplies, so unrolling it further only adds code (20x: about 60 KB more with
    // SSE2, no faster). Unrolled twice, the bound checks still have constant conditions.
    QOAMAX_PRAGMA(GCC unroll 2)
    for (int i = 0; i < 20; i++)
      if (!qoamax_hillclimb_lanes_step(&st, &t, i, 20, samples[i], pos, bound, out, unclamped)) return false;
  } else {
    QOAMAX_PRAGMA(GCC unroll 1)
    for (int i = 0; i < samples_count; i++)
      if (!qoamax_hillclimb_lanes_step(&st, &t, i, samples_count, samples[i], pos, bound, out, unclamped)) return false;
  }

  for (int lane = 0; lane < QOAMAX_VARIANT_LANES; lane++)
    out->errors[pos + lane] = (uint64_t)(st.error_high[lane] ^ bias) << 32 | (st.error_low[lane] ^ bias);
  memcpy(&out->history[0][pos], &st.h0, sizeof st.h0);
  memcpy(&out->history[1][pos], &st.h1, sizeof st.h1);
  memcpy(&out->history[2][pos], &st.h2, sizeof st.h2);
  memcpy(&out->history[3][pos], &st.h3, sizeof st.h3);
  memcpy(&out->weights[0][pos], &st.w0, sizeof st.w0);
  memcpy(&out->weights[1][pos], &st.w1, sizeof st.w1);
  memcpy(&out->weights[2][pos], &st.w2, sizeof st.w2);
  memcpy(&out->weights[3][pos], &st.w3, sizeof st.w3);
  return true;
}

//! Partial slices (the end of the stream) of @ref qoamax_hillclimb_lanes. A separate function, so that the rarely
//! used loop doesn't affect the code of the hot one (inlined, SUPERFAST was 4% slower).
QOAMAX_VARIANT_TARGET
[[gnu::noinline]]
static bool qoamax_hillclimb_lanes__partial(const qoamax_lms *lms, int samples_count,
                                            const int16_t samples[static restrict samples_count], int column, int pos,
                                            uint64_t bound, qoamax_scalefactor_results *restrict out) {
  qoamax_assume(samples_count < 20); // so that the unrolled loop isn't emitted again
  return qoamax_hillclimb_lanes_impl(lms, samples_count, samples, column, pos, bound, out, false);
}

//! See @ref qoamax_hillclimb_lanes__partial (unclamped variant).
QOAMAX_VARIANT_TARGET
[[gnu::noinline]]
static bool qoamax_hillclimb_lanes_unclamped__partial(const qoamax_lms *lms, int samples_count,
                                                      const int16_t samples[static restrict samples_count], int column,
                                                      int pos, uint64_t bound,
                                                      qoamax_scalefactor_results *restrict out) {
  qoamax_assume(samples_count < 20);
  return qoamax_hillclimb_lanes_impl(lms, samples_count, samples, column, pos, bound, out, true);
}

//! See @ref qoamax_hillclimb_lanes_impl. The common 20-sample case is inlined, partial slices call a separate function.
QOAMAX_VARIANT_TARGET
static bool qoamax_hillclimb_lanes(const qoamax_lms *lms, int samples_count,
                                   const int16_t samples[static restrict samples_count], int column, int pos,
                                   uint64_t bound, qoamax_scalefactor_results *restrict out) {
  if (samples_count == 20) return qoamax_hillclimb_lanes_impl(lms, 20, samples, column, pos, bound, out, false);
  return qoamax_hillclimb_lanes__partial(lms, samples_count, samples, column, pos, bound, out);
}

//! See @ref qoamax_hillclimb_lanes_impl (unclamped variant).
QOAMAX_VARIANT_TARGET
static bool qoamax_hillclimb_lanes_unclamped(const qoamax_lms *lms, int samples_count,
                                             const int16_t samples[static restrict samples_count], int column, int pos,
                                             uint64_t bound, qoamax_scalefactor_results *restrict out) {
  if (samples_count == 20) return qoamax_hillclimb_lanes_impl(lms, 20, samples, column, pos, bound, out, true);
  return qoamax_hillclimb_lanes_unclamped__partial(lms, samples_count, samples, column, pos, bound, out);
}

QOAMAX_VARIANT_TARGET
static uint64_t
qoamax_encode_slice__hillclimb_scalefactor_search_lanes(qoamax_lms *lms, int first_scalefactor, int lookahead,
                                                        const int16_t samples[static restrict lookahead],
                                                        uint64_t error_hint, uint64_t *out_slice, uint64_t *out_error);
QOAMAX_VARIANT_TARGET
static uint64_t qoamax_encode_slice__hillclimb_scalefactor_search_lanes_unclamped(
    qoamax_lms *lms, int first_scalefactor, int lookahead, const int16_t samples[static restrict lookahead],
    uint64_t error_hint, uint64_t *out_slice, uint64_t *out_error);

//! SIMD version of @ref qoamax_encode_slice__hillclimb_scalefactor_search_scalar_impl with identical results.
//! out_slice may be null when only the returned error is needed (out_error and lms are left unchanged then).
QOAMAX_VARIANT_TARGET
[[gnu::always_inline]]
static inline uint64_t qoamax_encode_slice__hillclimb_scalefactor_search_lanes_impl(
    qoamax_lms *lms, int first_scalefactor, int lookahead, const int16_t samples[static restrict lookahead],
    uint64_t error_hint, uint64_t *out_slice, uint64_t *out_error, bool unclamped) {
  int samples_count = QOAMAX_MIN(20, lookahead);
  qoamax_scalefactor_results results;

  // The lane windows are rotated so that the first one holds the scalefactors around first_scalefactor, which
  // usually include the best one. The other windows then hold only distant scalefactors, which exit early.
  // Results are indexed by position (scalefactor - base) & 15, window w holds positions w * QOAMAX_VARIANT_LANES and
  // up.
  int base = QOAMAX_MIN(QOAMAX_MAX(first_scalefactor - (QOAMAX_VARIANT_LANES / 2 - 1), 0), 16 - QOAMAX_VARIANT_LANES);
  int rank[16]; // by position
  for (int pos = 0; pos < 16; pos++) rank[pos] = qoamax_outward_rank((base + pos) & 15, first_scalefactor);

  int best            = -1; // position
  uint64_t best_error = error_hint;

  // Visit the windows outwards from the first one, as if they continued past both ends of the scalefactor range. A
  // window that wraps around is visited at the first of its two parts.
  unsigned visited = 0;
  for (int i = 0; visited != (1u << (16 / QOAMAX_VARIANT_LANES)) - 1; i++) {
    int offset = qoamax_outward(0, i) * QOAMAX_VARIANT_LANES;
    if (base + offset + QOAMAX_VARIANT_LANES <= 0 || base + offset >= 16) continue;
    int window = (offset / QOAMAX_VARIANT_LANES) & ((16 / QOAMAX_VARIANT_LANES) - 1);
    if (visited & (1u << window)) continue;
    visited |= 1u << window;

    int first = window * QOAMAX_VARIANT_LANES;

    // Lanes can only win with a lower error, or with an equal error and a better rank than the current best.
    bool may_win_tie = false;
    for (int pos = first; pos < first + QOAMAX_VARIANT_LANES; pos++) may_win_tie |= best >= 0 && rank[pos] < rank[best];
    uint64_t bound = best < 0 ? error_hint : best_error + may_win_tie;
    bool any_alive =
        unclamped ? qoamax_hillclimb_lanes_unclamped(lms, samples_count, samples, base + first, first, bound, &results)
                  : qoamax_hillclimb_lanes(lms, samples_count, samples, base + first, first, bound, &results);
    if (!any_alive) continue;

    // Visit the window's candidates by increasing error of their first slice (ties by rank), so the loop can stop
    // as soon as that error alone is too large.
    int order[QOAMAX_VARIANT_LANES];
    for (int j = 0; j < QOAMAX_VARIANT_LANES; j++) {
      int pos = first + j, k = j;
      for (; k > 0; k--) {
        int other = order[k - 1];
        if (results.errors[other] < results.errors[pos]) break;
        if (results.errors[other] == results.errors[pos] && rank[other] < rank[pos]) break;
        order[k] = other;
      }
      order[k] = pos;
    }

    for (int j = 0; j < QOAMAX_VARIANT_LANES; j++) {
      int pos        = order[j];
      uint64_t error = results.errors[pos];
      bool wins_tie  = best >= 0 && rank[pos] < rank[best];
      if (error > best_error) break;
      if (error == best_error && !wins_tie) continue;

      if (lookahead > 20) {
        qoamax_lms next_lms;
        for (int k = 0; k < 4; k++)
          next_lms.history[k] = results.history[k][pos], next_lms.weights[k] = results.weights[k][pos];

        int sf              = (base + pos) & 15;
        uint64_t next_hint  = best_error - error + wins_tie;
        uint64_t next_error = unclamped ? qoamax_encode_slice__hillclimb_scalefactor_search_lanes_unclamped(
                                              &next_lms, sf, lookahead - 20, &samples[20], next_hint, nullptr, nullptr)
                                        : qoamax_encode_slice__hillclimb_scalefactor_search_lanes(
                                              &next_lms, sf, lookahead - 20, &samples[20], next_hint, nullptr, nullptr);
        if (next_error >= next_hint) continue;
        error += next_error;
      }

      if (error < best_error || (error == best_error && wins_tie)) best = pos, best_error = error;
    }
  }

  if (best < 0) return error_hint;
  if (!out_slice) return best_error; // only the error is needed (lookahead)

  *out_slice = qoamax_scalefactor_results_get(&results, best, (base + best) & 15, samples_count, lms);
  *out_error = results.errors[best];
  return best_error;
}

//! See @ref qoamax_encode_slice__hillclimb_scalefactor_search_lanes_impl.
QOAMAX_VARIANT_TARGET
static uint64_t
qoamax_encode_slice__hillclimb_scalefactor_search_lanes(qoamax_lms *lms, int first_scalefactor, int lookahead,
                                                        const int16_t samples[static restrict lookahead],
                                                        uint64_t error_hint, uint64_t *out_slice, uint64_t *out_error) {
  return qoamax_encode_slice__hillclimb_scalefactor_search_lanes_impl(lms, first_scalefactor, lookahead, samples,
                                                                      error_hint, out_slice, out_error, false);
}

//! See @ref qoamax_encode_slice__hillclimb_scalefactor_search_lanes_impl.
QOAMAX_VARIANT_TARGET
static uint64_t qoamax_encode_slice__hillclimb_scalefactor_search_lanes_unclamped(
    qoamax_lms *lms, int first_scalefactor, int lookahead, const int16_t samples[static restrict lookahead],
    uint64_t error_hint, uint64_t *out_slice, uint64_t *out_error) {
  return qoamax_encode_slice__hillclimb_scalefactor_search_lanes_impl(lms, first_scalefactor, lookahead, samples,
                                                                      error_hint, out_slice, out_error, true);
}

#if QOAMAX_VARIANT_LANES >= 4
//! SIMD version of @ref qoamax_encode_slice__ultrafast_scalar with identical results: the scalefactors
//! prev_scalefactor - 1 and + 1 are lanes 0 and 2 of one lane window (the lanes in between are computed, but ignored).
QOAMAX_VARIANT_TARGET
static uint64_t qoamax_encode_slice__ultrafast_lanes(qoamax_lms *lms, int prev_scalefactor, int samples_count,
                                                     const int16_t samples[static restrict samples_count],
                                                     uint64_t *out_error) {
  qoamax_scalefactor_results results;
  int column = (prev_scalefactor - 1) & 15;
  qoamax_hillclimb_lanes(lms, samples_count, samples, column, 0, UINT64_MAX, &results);
  int best   = results.errors[2] < results.errors[0] ? 2 : 0;
  *out_error = results.errors[best];
  return qoamax_scalefactor_results_get(&results, best, (column + best) & 15, samples_count, lms);
}
#endif

// End of the variant (see above).
#undef qoamax_vi32
#undef qoamax_vu32
#undef qoamax_lanes_table
#undef qoamax_lanes_state
#undef qoamax_hillclimb_lanes_step
#undef qoamax_hillclimb_lanes_impl
#undef qoamax_hillclimb_lanes__partial
#undef qoamax_hillclimb_lanes_unclamped__partial
#undef qoamax_hillclimb_lanes
#undef qoamax_hillclimb_lanes_unclamped
#undef qoamax_encode_slice__hillclimb_scalefactor_search_lanes_impl
#undef qoamax_encode_slice__hillclimb_scalefactor_search_lanes
#undef qoamax_encode_slice__hillclimb_scalefactor_search_lanes_unclamped
#undef qoamax_encode_slice__ultrafast_lanes
#undef QOAMAX_VARIANT_LANES
#undef QOAMAX_VARIANT_SUFFIX
#undef QOAMAX_VARIANT_TARGET
