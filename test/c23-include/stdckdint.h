// A minimal, portable <stdckdint.h> for compilers whose header needs GNU builtins (e.g. cproc with gcc's header).
// Only what qoamax.c uses. Not installed: only for the test build in the Makefile.
#ifndef QOAMAX_TEST_STDCKDINT_H
#define QOAMAX_TEST_STDCKDINT_H

#include <stdbool.h>
#include <stdint.h>

static inline bool qoamax_test_ckd_add_u64(uint64_t *result, uint64_t a, uint64_t b) {
  *result = a + b;
  return *result < a;
}

#define ckd_add(result, a, b) _Generic(*(result), uint64_t: qoamax_test_ckd_add_u64)((result), (a), (b))

#endif
