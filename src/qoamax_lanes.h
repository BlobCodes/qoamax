// The SIMD scalefactor search (one lane per scalefactor), ULTRAFAST (two lanes per stream) and the decoder (one lane
// per stream). Not a public header: qoamax.c includes this file once per SIMD variant, after defining
// - QOAMAX_VARIANT_LANES:  the number of lanes (2, 4, 8 or 16),
// - QOAMAX_VARIANT_SUFFIX: appended to every name defined here (empty for the compile target's default variant),
// - QOAMAX_VARIANT_TARGET: an attribute for every function, e.g. [[gnu::target("avx2")]] (or empty).
// Every variant produces identical results.

static_assert(QOAMAX_VARIANT_LANES >= 2 && QOAMAX_VARIANT_LANES <= 16 && 16 % QOAMAX_VARIANT_LANES == 0);

// The names defined here, renamed per variant (e.g. qoamax_hillclimb_lanes__x8), so that the code below reads as usual.
#define qoamax_vi32 QOAMAX_VARIANT_NAME(qoamax_vi32)
#define qoamax_vu32 QOAMAX_VARIANT_NAME(qoamax_vu32)
#define qoamax_vu64 QOAMAX_VARIANT_NAME(qoamax_vu64)
#define qoamax_lanes_swap_pairs QOAMAX_VARIANT_NAME(qoamax_lanes_swap_pairs)
#define qoamax_lanes_take_pair QOAMAX_VARIANT_NAME(qoamax_lanes_take_pair)
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
#define qoamax_encode_ultrafast_lanes QOAMAX_VARIANT_NAME(qoamax_encode_ultrafast_lanes)
#define qoamax_decode_lanes QOAMAX_VARIANT_NAME(qoamax_decode_lanes)
#define qoamax_simd QOAMAX_VARIANT_NAME(qoamax_simd)
#define qoamax_encode_slice__hillclimb_scalefactor_search_lanes_unclamped                \
  QOAMAX_VARIANT_NAME(qoamax_encode_slice__hillclimb_scalefactor_search_lanes_unclamped)

typedef int32_t qoamax_vi32 __attribute__((vector_size(QOAMAX_VARIANT_LANES * 4)));
typedef uint32_t qoamax_vu32 __attribute__((vector_size(QOAMAX_VARIANT_LANES * 4)));
typedef uint64_t qoamax_vu64 __attribute__((vector_size(QOAMAX_VARIANT_LANES * 4)));

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

//! Sets *other to v with the two lanes of every stream (2u and 2u + 1) swapped.
//! (Vectors are passed by pointer: gcc warns about vector return values, as their ABI depends on the target.)
QOAMAX_VARIANT_TARGET
[[gnu::always_inline]]
static inline void qoamax_lanes_swap_pairs(qoamax_vi32 *other, qoamax_vi32 v) {
  qoamax_vu64 pairs = (qoamax_vu64)v;
  *other            = (qoamax_vi32)(pairs << 32 | pairs >> 32);
}

//! Sets both lanes of a stream to the value of the other lane where `take` is set.
QOAMAX_VARIANT_TARGET
[[gnu::always_inline]]
static inline void qoamax_lanes_take_pair(qoamax_vi32 *v, qoamax_vi32 take) {
  qoamax_vi32 other;
  qoamax_lanes_swap_pairs(&other, *v);
  *v = (other & take) | (*v & ~take);
}

//! SIMD version of @ref qoamax_encode_ultrafast_scalar with identical results, for up to QOAMAX_VARIANT_LANES / 2
//! streams at once: stream u uses lanes 2u and 2u + 1, one per scalefactor candidate. Encodes slice_count full slices
//! (20 samples each) of every stream.
QOAMAX_VARIANT_TARGET
static void qoamax_encode_ultrafast_lanes(qoamax__stream *restrict streams, int count, int slice_count) {
  constexpr int lanes = QOAMAX_VARIANT_LANES;
  qoamax_assume(count >= 1 && count <= lanes / 2);
  qoamax_assume(slice_count >= 1);

  // The LMS states by lane (unused lanes are zero). Both lanes of a stream get the same value, which is stored as one
  // 64-bit value (the compilers store the int32 values one by one). They stay in vectors until the last slice.
  int32_t lms[8][lanes];
  int32_t odd_lanes[lanes];
  if (count < lanes / 2) memset(lms, 0, sizeof lms);
  for (int u = 0; u < count; u++) {
    for (int k = 0; k < 4; k++) {
      uint64_t history = (uint32_t)streams[u].state.lms.history[k] * UINT64_C(0x100000001);
      uint64_t weights = (uint32_t)streams[u].state.lms.weights[k] * UINT64_C(0x100000001);
      memcpy(&lms[k][2 * u], &history, sizeof history), memcpy(&lms[4 + k][2 * u], &weights, sizeof weights);
    }
  }
  for (int lane = 0; lane < lanes; lane++) odd_lanes[lane] = -(lane & 1);

  qoamax_vi32 h0, h1, h2, h3, w0, w1, w2, w3, odd;
  memcpy(&h0, lms[0], sizeof h0), memcpy(&h1, lms[1], sizeof h1), memcpy(&h2, lms[2], sizeof h2);
  memcpy(&h3, lms[3], sizeof h3), memcpy(&w0, lms[4], sizeof w0), memcpy(&w1, lms[5], sizeof w1);
  memcpy(&w2, lms[6], sizeof w2), memcpy(&w3, lms[7], sizeof w3), memcpy(&odd, odd_lanes, sizeof odd);

  // The error of the slices encoded so far (biased, see qoamax_lanes_state). Both lanes of a stream start every slice
  // with the same error, so comparing the sums compares the candidates.
  const qoamax_vi32 bias = (qoamax_vi32){} + INT32_MIN;
  qoamax_vi32 error_low = bias, error_high = bias, scalefactor = {};

  for (int s = 0; s < slice_count; s++) {
    // The samples by lane, like the LMS states.
    int32_t samples[20][lanes];
    if (count < lanes / 2) memset(samples, 0, sizeof samples);
    for (int u = 0; u < count; u++) {
      const int16_t *stream_samples = streams[u].samples;
      int stride                    = streams[u].sample_stride;
      for (int i = 0; i < 20; i++) {
        uint64_t pair = (uint32_t)stream_samples[i * stride] * UINT64_C(0x100000001);
        memcpy(&samples[i][2 * u], &pair, sizeof pair);
      }
    }

    // The estimate (see qoamax_ultrafast_estimate): independent predictions from the input samples.
    qoamax_vi32 x0 = h0, x1 = h1, x2 = h2, x3 = h3, max_residual = {};
    for (int i = 0; i < 20; i++) {
      qoamax_vi32 sample;
      memcpy(&sample, samples[i], sizeof sample);
      qoamax_vu32 sum      = (qoamax_vu32)w0 * (qoamax_vu32)x0 + (qoamax_vu32)w1 * (qoamax_vu32)x1 +
                             (qoamax_vu32)w2 * (qoamax_vu32)x2 + (qoamax_vu32)w3 * (qoamax_vu32)x3;
      qoamax_vi32 residual = sample - ((qoamax_vi32)sum >> 13);
      qoamax_vi32 abs      = (residual ^ (residual >> 31)) - (residual >> 31);
      qoamax_vi32 greater  = abs > max_residual;
      max_residual         = (abs & greater) | (max_residual & ~greater);
      x0 = x1, x1 = x2, x2 = x3, x3 = sample;
    }
    qoamax_vi32 estimate = {};
    for (int k = 0; k < 14; k++) estimate -= max_residual > qoamax_ultrafast_thresholds[k];

    // The table columns of each lane's scalefactor: stream u tries estimate and estimate + 1.
    int32_t columns[7][lanes];
    if (count < lanes / 2) memset(columns, 0, sizeof columns);
    for (int u = 0; u < count; u++) {
      int first = estimate[2 * u];
      for (int k = 0; k < 3; k++) memcpy(&columns[k][2 * u], &qoamax_lanes_qt[k][first], 2 * sizeof(int32_t));
      for (int k = 0; k < 4; k++) memcpy(&columns[3 + k][2 * u], &qoamax_lanes_dqt[k][first], 2 * sizeof(int32_t));
    }
    qoamax_lanes_table t;
    memcpy(&t.q0, columns[0], sizeof t.q0), memcpy(&t.q1, columns[1], sizeof t.q1);
    memcpy(&t.q2, columns[2], sizeof t.q2), memcpy(&t.d0, columns[3], sizeof t.d0);
    memcpy(&t.d1, columns[4], sizeof t.d1), memcpy(&t.d2, columns[5], sizeof t.d2);
    memcpy(&t.d3, columns[6], sizeof t.d3);

    // The hill-climb of every lane (like qoamax_hillclimb_lanes_step, without bound), collecting the residuals in two
    // halves of 10 samples (30 bits each).
    qoamax_vi32 quant_high = {}, quant_low = {};
    QOAMAX_PRAGMA(GCC unroll 2)
    for (int i = 0; i < 20; i++) {
      qoamax_vi32 sample;
      memcpy(&sample, samples[i], sizeof sample);
      qoamax_vu32 sum       = (qoamax_vu32)w0 * (qoamax_vu32)h0 + (qoamax_vu32)w1 * (qoamax_vu32)h1 +
                              (qoamax_vu32)w2 * (qoamax_vu32)h2 + (qoamax_vu32)w3 * (qoamax_vu32)h3;
      qoamax_vi32 predicted = (qoamax_vi32)sum >> 13;
      qoamax_vi32 residual  = sample - predicted;
      qoamax_vi32 negative  = residual <= 0;
      qoamax_vi32 abs       = (residual ^ (residual >> 31)) - (residual >> 31);
      qoamax_vi32 above0 = abs > t.q0, above1 = abs > t.q1, above2 = abs > t.q2;

      qoamax_vi32 magnitude   = t.d0 + (above0 & t.d1) + (above1 & t.d2) + (above2 & t.d3);
      qoamax_vi32 dequantized = (magnitude ^ negative) - negative;
      qoamax_vi32 quant       = (-(above0 + above1 + above2) << 1) | (negative & 1);
      if (i < 10)
        quant_high = quant_high << 3 | quant;
      else
        quant_low = quant_low << 3 | quant;

      qoamax_vi32 reconstructed = predicted + dequantized;
      qoamax_vi32 too_low       = reconstructed < INT16_MIN;
      reconstructed             = (reconstructed & ~too_low) | (INT16_MIN & too_low);
      qoamax_vi32 too_high      = reconstructed > INT16_MAX;
      reconstructed             = (reconstructed & ~too_high) | (INT16_MAX & too_high);

      qoamax_vi32 diff      = sample - reconstructed;
      qoamax_vi32 new_error = (qoamax_vi32)((qoamax_vu32)error_low + (qoamax_vu32)diff * (qoamax_vu32)diff);
      error_high -= new_error < error_low; // carry (the comparison yields -1)
      error_low = new_error;

      qoamax_vi32 delta = dequantized >> 4;
      qoamax_vi32 sign0 = h0 >> 31, sign1 = h1 >> 31, sign2 = h2 >> 31, sign3 = h3 >> 31;
      w0 += (delta ^ sign0) - sign0;
      w1 += (delta ^ sign1) - sign1;
      w2 += (delta ^ sign2) - sign2;
      w3 += (delta ^ sign3) - sign3;
      h0 = h1, h1 = h2, h2 = h3, h3 = reconstructed;
    }

    // Keep the better candidate of each stream (the lower scalefactor if both are equal) in both of its lanes.
    qoamax_vi32 other_high, other_low;
    qoamax_lanes_swap_pairs(&other_high, error_high), qoamax_lanes_swap_pairs(&other_low, error_low);
    qoamax_vi32 same_high = other_high == error_high;
    qoamax_vi32 take      = (other_high < error_high) | (same_high & (other_low < error_low)) |
                            (same_high & (other_low == error_low) & odd);
    scalefactor           = estimate - odd;
    qoamax_vi32 *state[]  = {&error_high, &error_low, &h0, &h1,         &h2,        &h3,         &w0,
                             &w1,         &w2,        &w3, &quant_high, &quant_low, &scalefactor};
    for (size_t k = 0; k < sizeof state / sizeof state[0]; k++) qoamax_lanes_take_pair(state[k], take);

    // The first 10 residuals are in quant_high, the others in quant_low.
    for (int u = 0; u < count; u++) {
      uint64_t slice =
          (uint64_t)scalefactor[2 * u] << 60 | (uint64_t)(uint32_t)quant_high[2 * u] << 30 | (uint32_t)quant_low[2 * u];
      qoamax__stream_put(&streams[u], slice, 20);
    }
  }

  for (int u = 0; u < count; u++) {
    qoamax_encoder_channel *state = &streams[u].state;
    int lane                      = 2 * u;
    state->error +=
        (uint64_t)((uint32_t)error_high[lane] ^ 0x80000000) << 32 | ((uint32_t)error_low[lane] ^ 0x80000000);
    state->prev_scalefactor = scalefactor[lane];
    state->lms.history[0] = h0[lane], state->lms.history[1] = h1[lane];
    state->lms.history[2] = h2[lane], state->lms.history[3] = h3[lane];
    state->lms.weights[0] = w0[lane], state->lms.weights[1] = w1[lane];
    state->lms.weights[2] = w2[lane], state->lms.weights[3] = w3[lane];
  }
}

//! Decodes slice_count full slices (20 samples each) of up to QOAMAX_VARIANT_LANES streams at once, one stream per
//! lane, with the same results as @ref qoamax_decode_slice.
QOAMAX_VARIANT_TARGET
static void qoamax_decode_lanes(qoamax__stream *restrict streams, int count, int slice_count) {
  constexpr int lanes = QOAMAX_VARIANT_LANES;
  qoamax_assume(count >= 1 && count <= lanes);

  // The LMS states by lane (unused lanes are zero). They stay in vectors until the last slice.
  int32_t lms[8][lanes];
  if (count < lanes) memset(lms, 0, sizeof lms);
  for (int l = 0; l < count; l++)
    for (int k = 0; k < 4; k++)
      lms[k][l] = streams[l].state.lms.history[k], lms[4 + k][l] = streams[l].state.lms.weights[k];
  qoamax_vi32 h0, h1, h2, h3, w0, w1, w2, w3;
  memcpy(&h0, lms[0], sizeof h0), memcpy(&h1, lms[1], sizeof h1), memcpy(&h2, lms[2], sizeof h2);
  memcpy(&h3, lms[3], sizeof h3), memcpy(&w0, lms[4], sizeof w0), memcpy(&w1, lms[5], sizeof w1);
  memcpy(&w2, lms[6], sizeof w2), memcpy(&w3, lms[7], sizeof w3);
  // The signs of the history (0 or -1), which select the direction of the weight updates.
  qoamax_vi32 s0 = h0 >> 31, s1 = h1 >> 31, s2 = h2 >> 31, s3 = h3 >> 31;

  for (int s = 0; s < slice_count; s++) {
    // Per lane: the slice's residuals in two halves of 30 bits (first residual at the top), and the dequantization
    // columns of its scalefactor (see qoamax_lanes_dqt).
    uint32_t slices[2][lanes];
    int32_t columns[4][lanes];
    if (count < lanes) memset(slices, 0, sizeof slices), memset(columns, 0, sizeof columns);
    for (int l = 0; l < count; l++) {
      uint64_t slice = qoamax__stream_get(&streams[l]);
      slices[0][l]   = (uint32_t)(slice >> 30) << 2;
      slices[1][l]   = (uint32_t)slice << 2;
      for (int k = 0; k < 4; k++) columns[k][l] = qoamax_lanes_dqt[k][slice >> 60];
    }
    qoamax_vu32 residuals;
    qoamax_vi32 d0, d1, d2, d3;
    memcpy(&residuals, slices[0], sizeof residuals);
    memcpy(&d0, columns[0], sizeof d0), memcpy(&d1, columns[1], sizeof d1);
    memcpy(&d2, columns[2], sizeof d2), memcpy(&d3, columns[3], sizeof d3);

    int32_t out[20][lanes];
    QOAMAX_PRAGMA(GCC unroll 5)
    for (int i = 0; i < 20; i++) {
      if (i == 10) memcpy(&residuals, slices[1], sizeof residuals);
      // The dequantized residual: {a, -a, b, -b, ...}[quant], with the magnitude d0 + d1 + ... (up to quant / 2).
      qoamax_vi32 quant    = (qoamax_vi32)(residuals >> 29);
      qoamax_vi32 negative = -(quant & 1);
      residuals <<= 3;
      qoamax_vi32 magnitude   = d0 + ((quant > 1) & d1) + ((quant > 3) & d2) + ((quant > 5) & d3);
      qoamax_vi32 dequantized = (magnitude ^ negative) - negative;

      // Unsigned to keep overflow defined; the result is the same as in the scalar code.
      qoamax_vu32 sum      = (qoamax_vu32)w0 * (qoamax_vu32)h0 + (qoamax_vu32)w1 * (qoamax_vu32)h1 +
                             (qoamax_vu32)w2 * (qoamax_vu32)h2 + (qoamax_vu32)w3 * (qoamax_vu32)h3;
      qoamax_vi32 sample   = ((qoamax_vi32)sum >> 13) + dequantized;
      qoamax_vi32 too_low  = sample < INT16_MIN;
      sample               = (sample & ~too_low) | (INT16_MIN & too_low);
      qoamax_vi32 too_high = sample > INT16_MAX;
      sample               = (sample & ~too_high) | (INT16_MAX & too_high);

      qoamax_vi32 delta = dequantized >> 4;
      w0 += (delta ^ s0) - s0;
      w1 += (delta ^ s1) - s1;
      w2 += (delta ^ s2) - s2;
      w3 += (delta ^ s3) - s3;
      h0 = h1, h1 = h2, h2 = h3, h3 = sample;
      s0 = s1, s1 = s2, s2 = s3, s3 = sample >> 31;
      memcpy(out[i], &sample, sizeof sample);
    }

    for (int l = 0; l < count; l++) {
      qoamax__stream *stream = &streams[l];
      for (int i = 0; i < 20; i++) stream->samples[i * stream->sample_stride] = (int16_t)out[i][l];
      qoamax__stream_next(stream, 20);
    }
  }

  for (int l = 0; l < count; l++) {
    qoamax_lms *state = &streams[l].state.lms;
    state->history[0] = h0[l], state->history[1] = h1[l], state->history[2] = h2[l], state->history[3] = h3[l];
    state->weights[0] = w0[l], state->weights[1] = w1[l], state->weights[2] = w2[l], state->weights[3] = w3[l];
  }
}

//! The functions of this variant, see @ref qoamax__simd_get.
static const qoamax__simd qoamax_simd = {
    .lanes            = QOAMAX_VARIANT_LANES,
    .search           = qoamax_encode_slice__hillclimb_scalefactor_search_lanes,
    .search_unclamped = qoamax_encode_slice__hillclimb_scalefactor_search_lanes_unclamped,
    .encode_ultrafast = qoamax_encode_ultrafast_lanes,
    .decode           = qoamax_decode_lanes,
};

// End of the variant (see above).
#undef qoamax_vi32
#undef qoamax_vu32
#undef qoamax_vu64
#undef qoamax_lanes_swap_pairs
#undef qoamax_lanes_take_pair
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
#undef qoamax_encode_ultrafast_lanes
#undef qoamax_decode_lanes
#undef qoamax_simd
#undef QOAMAX_VARIANT_LANES
#undef QOAMAX_VARIANT_SUFFIX
#undef QOAMAX_VARIANT_TARGET
