// A minimal, portable <stdbit.h> for compilers whose C library's header needs GNU builtins (e.g. cproc with glibc).
// Only what qoamax.c uses. Not installed: only for the test build in the Makefile.
#ifndef QOAMAX_TEST_STDBIT_H
#define QOAMAX_TEST_STDBIT_H

static inline unsigned int stdc_trailing_zeros_ui(unsigned int value) {
  unsigned int count = 0;
  for (; count < sizeof(value) * 8 && !(value & 1); count++) value >>= 1;
  return count;
}

#define stdc_trailing_zeros(value) _Generic((value), unsigned int: stdc_trailing_zeros_ui)(value)

#endif
