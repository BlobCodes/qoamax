// The beam search (FAST to PLACEBO), apart from its inline helpers. Not a public header: qoamax.c includes this file
// once for the compile target and, on x86 without AVX2, once more for AVX2 (chosen at runtime), after defining
// - QOAMAX_VARIANT_SUFFIX:  appended to every name defined here (empty for the compile target's default variant),
// - QOAMAX_VARIANT_TARGET:  an attribute for every function, e.g. [[gnu::target("avx2")]] (or empty),
// - QOAMAX_BEAM_BRANCHLESS: whether beams of width 1 and 2 also compute their candidates without branches (see
//   qoamax_encode_beam_search_impl).
// The code is the same scalar code in every variant. The compiler vectorizes the candidates' errors and the
// prediction, which needs 64-bit multiplies and comparisons (and 32-bit multiplies) that SSE2 doesn't have.
// Every variant produces identical results.

// The names defined here, renamed per variant (e.g. qoamax_encode_beam_search__avx2).
#define qoamax_encode_beam_search__wide QOAMAX_VARIANT_NAME(qoamax_encode_beam_search__wide)
#define qoamax_encode_beam_search QOAMAX_VARIANT_NAME(qoamax_encode_beam_search)
#define qoamax_encode_slice__beam_scalefactor_search QOAMAX_VARIANT_NAME(qoamax_encode_slice__beam_scalefactor_search)
#define qoamax_encode_slice__beam_scalefactor_search_wide                \
  QOAMAX_VARIANT_NAME(qoamax_encode_slice__beam_scalefactor_search_wide)
#define qoamax_encode_slice__beam_search QOAMAX_VARIANT_NAME(qoamax_encode_slice__beam_search)

//! See @ref qoamax_encode_beam_search_impl. Wide beams (and unusual widths) get their own function, so that their
//! large search state is only on the stack when it is used.
QOAMAX_VARIANT_TARGET
[[gnu::noinline]]
static int qoamax_encode_beam_search__wide(const qoamax_beam_node *restrict src, int src_count,
                                           qoamax_beam_node *restrict buffers, int width, bool first_slice,
                                           int samples_size, const int16_t samples[static restrict samples_size],
                                           int scalefactor, uint64_t error_hint,
                                           qoamax_beam_node *restrict *restrict out) {
  qoamax_beam_heap_entry heap[QOAMAX_MAX_BEAM_WIDTH];
  uint64_t error[QOAMAX_MAX_BEAM_WIDTH];
  uint16_t parent[QOAMAX_MAX_BEAM_WIDTH];
  uint8_t codepoint[QOAMAX_MAX_BEAM_WIDTH];
  int32_t predicted[QOAMAX_MAX_BEAM_WIDTH];
  qoamax_beam_slots slots = {heap, error, parent, codepoint};
  return qoamax_encode_beam_search_impl(src, src_count, buffers, width, first_slice, samples_size, samples, scalefactor,
                                        error_hint, out, slots, predicted, QOAMAX_BEAM_BRANCHLESS);
}

//! See @ref qoamax_encode_beam_search_impl. The small beam widths used by the encoder efforts (2, 4 and
//! QOAMAX_SMALL_BEAM_WIDTH) get their own copies. The caller chooses between this and the wide variant, which keeps
//! this function free of calls (otherwise gcc's code gets about 3% slower).
QOAMAX_VARIANT_TARGET
static int qoamax_encode_beam_search(const qoamax_beam_node *restrict src, int src_count,
                                     qoamax_beam_node *restrict buffers, int width, bool first_slice, int samples_size,
                                     const int16_t samples[static restrict samples_size], int scalefactor,
                                     uint64_t error_hint, qoamax_beam_node *restrict *restrict out) {
  qoamax_beam_heap_entry heap[QOAMAX_SMALL_BEAM_WIDTH];
  uint64_t error[QOAMAX_SMALL_BEAM_WIDTH];
  uint16_t parent[QOAMAX_SMALL_BEAM_WIDTH];
  uint8_t codepoint[QOAMAX_SMALL_BEAM_WIDTH];
  int32_t predicted[QOAMAX_SMALL_BEAM_WIDTH];
  qoamax_beam_slots slots = {heap, error, parent, codepoint};
  switch (width) {
  case 2:
    return qoamax_encode_beam_search_impl(src, src_count, buffers, 2, first_slice, samples_size, samples, scalefactor,
                                          error_hint, out, slots, predicted, QOAMAX_BEAM_BRANCHLESS);
  case 4:
    return qoamax_encode_beam_search_impl(src, src_count, buffers, 4, first_slice, samples_size, samples, scalefactor,
                                          error_hint, out, slots, predicted, QOAMAX_BEAM_BRANCHLESS);
  default:
    return qoamax_encode_beam_search_impl(src, src_count, buffers, QOAMAX_SMALL_BEAM_WIDTH, first_slice, samples_size,
                                          samples, scalefactor, error_hint, out, slots, predicted,
                                          QOAMAX_BEAM_BRANCHLESS);
  }
}

//! Searches the best scalefactor using a beam search for each one, recursively continuing the search for the
//! following slices of the lookahead from the resulting nodes. Only results better than error_hint are considered.
//! `buffers` holds 2 * beam_width nodes per slice of the lookahead, the depth-th slice uses the depth-th part.
QOAMAX_VARIANT_TARGET
static void qoamax_encode_slice__beam_scalefactor_search(const qoamax_beam_node *src, int src_count,
                                                         qoamax_beam_node *buffers, int beam_width, int depth,
                                                         int first_scalefactor, int lookahead,
                                                         const int16_t samples[static restrict lookahead],
                                                         uint64_t error_hint, qoamax_beam_node *out_node) {
  for (int rank = 0; rank < 31; rank++) {
    int scalefactor = qoamax_outward(first_scalefactor, rank);
    if ((unsigned)scalefactor >= 16) continue;

    qoamax_beam_node *nodes;
    bool small      = beam_width == 2 || beam_width == 4 || beam_width == QOAMAX_SMALL_BEAM_WIDTH;
    int nodes_count = (small ? qoamax_encode_beam_search : qoamax_encode_beam_search__wide)(
        src, src_count, &buffers[depth * 2 * beam_width], beam_width, depth == 0, QOAMAX_MIN(20, lookahead), samples,
        scalefactor, error_hint, &nodes);
    if (nodes_count == 0) continue;

    const qoamax_beam_node *best = &nodes[0];
    for (int i = 1; i < nodes_count; i++)
      if (nodes[i].error < best->error) best = &nodes[i];
    if (best->error >= error_hint) continue;

    if (lookahead <= 20) {
      error_hint = best->error;
      *out_node  = *best;
    } else {
      qoamax_beam_node next_node = {.error = -1};
      qoamax_encode_slice__beam_scalefactor_search(nodes, nodes_count, buffers, beam_width, depth + 1, scalefactor,
                                                   lookahead - 20, &samples[20], error_hint, &next_node);
      if (next_node.error < error_hint) {
        error_hint = next_node.error;
        *out_node  = next_node;
      }
    }
  }
}

//! See @ref qoamax_encode_slice__beam_scalefactor_search. The node buffers of wide beams (about 320 KB) are only on
//! the stack while this function runs.
QOAMAX_VARIANT_TARGET
[[gnu::noinline]]
static void qoamax_encode_slice__beam_scalefactor_search_wide(const qoamax_beam_node *root, int beam_width,
                                                              int first_scalefactor, int lookahead,
                                                              const int16_t samples[static restrict lookahead],
                                                              uint64_t error_hint, qoamax_beam_node *out_node) {
  qoamax_beam_node buffers[QOAMAX_MAX_LOOKAHEAD_SLICES * 2 * QOAMAX_MAX_BEAM_WIDTH];
  qoamax_encode_slice__beam_scalefactor_search(root, 1, buffers, beam_width, 0, first_scalefactor, lookahead, samples,
                                               error_hint, out_node);
}

//! Runs the beam search of @ref qoamax_encode_slice__beam from its root node, see
//! @ref qoamax_encode_slice__beam_scalefactor_search.
QOAMAX_VARIANT_TARGET
static void qoamax_encode_slice__beam_search(const qoamax_beam_node *root, int beam_width, int first_scalefactor,
                                             int lookahead, const int16_t samples[static restrict lookahead],
                                             uint64_t error_hint, qoamax_beam_node *out_node) {
  int slices = (lookahead + 19) / 20; // depth of the search
  if (slices * 2 * beam_width <= QOAMAX_SMALL_BEAM_NODES) {
    qoamax_beam_node buffers[QOAMAX_SMALL_BEAM_NODES];
    qoamax_encode_slice__beam_scalefactor_search(root, 1, buffers, beam_width, 0, first_scalefactor, lookahead, samples,
                                                 error_hint, out_node);
  } else {
    qoamax_encode_slice__beam_scalefactor_search_wide(root, beam_width, first_scalefactor, lookahead, samples,
                                                      error_hint, out_node);
  }
}

// End of the variant (see above).
#undef qoamax_encode_beam_search__wide
#undef qoamax_encode_beam_search
#undef qoamax_encode_slice__beam_scalefactor_search
#undef qoamax_encode_slice__beam_scalefactor_search_wide
#undef qoamax_encode_slice__beam_search
#undef QOAMAX_VARIANT_SUFFIX
#undef QOAMAX_VARIANT_TARGET
#undef QOAMAX_BEAM_BRANCHLESS
