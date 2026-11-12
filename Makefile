CC := clang
# The C++ compiler of the same family (clang-19 -> clang++-19, gcc -> g++), for the C++ header test.
CXX := $(subst clang,clang++,$(subst gcc,g++,$(CC)))
AR ?= ar
BUILD_DIR := build

# Installation directories (make install PREFIX=/usr DESTDIR=...).
PREFIX ?= /usr/local
BINDIR ?= $(PREFIX)/bin
LIBDIR ?= $(PREFIX)/lib
INCLUDEDIR ?= $(PREFIX)/include
PKGCONFIGDIR ?= $(LIBDIR)/pkgconfig

# The version is defined in the header. Before 1.0, a minor version may change the ABI, so it is part of the soname.
VERSION := $(shell awk '/^.define QOAMAX_VERSION_(MAJOR|MINOR|PATCH) / { printf "%s%s", sep, $$3; sep = "." }' \
                       src/qoamax.h)
SOVERSION := $(basename $(VERSION))

CFLAGS := -std=c23 -Wno-unknown-attributes -Wno-unknown-pragmas
DEBUG_CFLAGS := $(CFLAGS) -O0 -g -fwrapv \
                -fsanitize=address,undefined \
                -fstack-protector-strong
RELEASE_CFLAGS := $(CFLAGS) -O3
# The tests run with sanitizers (without -fwrapv, so that UBSan reports signed overflows), stopping at the first error.
TEST_CFLAGS := $(CFLAGS) -O1 -g -Wall -Wextra -pthread \
               -fsanitize=address,undefined -fno-sanitize-recover=all \
               -fstack-protector-strong
TEST_CXXFLAGS := -std=c++11 -O1 -g -Wall -Wextra -pedantic-errors -Isrc \
                 -fsanitize=address,undefined -fno-sanitize-recover=all
# The library tests also run with these SIMD lane counts (besides the target's default), which must all produce the
# same output. Lane counts above the target's vector width work too, the compiler splits the vectors.
TEST_LANES := 1 2 4 8 16
TEST_LIBS := $(BUILD_DIR)/test_lib $(foreach lanes,$(TEST_LANES),$(BUILD_DIR)/test_lib_lanes$(lanes))

LIB_SOURCES := src/qoamax.h src/qoamax_plain.h src/qoamax.c src/qoamax_lanes.h src/qoamax_beam.h
SHARED_LIB := $(BUILD_DIR)/libqoamax.so.$(VERSION)

.PHONY: all debug release lib test test-build-systems install uninstall clean

all: debug release

debug: $(BUILD_DIR)/qoamax-debug

release: $(BUILD_DIR)/qoamax

lib: $(BUILD_DIR)/libqoamax.a $(SHARED_LIB)

test: $(TEST_LIBS) $(BUILD_DIR)/test_cpp $(BUILD_DIR)/wavgen $(BUILD_DIR)/qoamax-debug
	@for test in $(TEST_LIBS); do $$test || exit 1; done
	$(BUILD_DIR)/test_cpp
	@# The public header is C11 and C++11 without extensions (only qoamax.c needs C23), also without VLAs.
	for compiler in "$(CC) -x c -std=c11" "$(CC) -x c -std=c11 -D__STDC_NO_VLA__=1 -Werror=vla" \
	                "$(CXX) -x c++ -std=c++11"; do \
	  printf '#include "src/qoamax.h"\n' | $$compiler -pedantic-errors -Wall -Wextra -Werror -fsyntax-only - || exit 1; \
	done
	@# In C, the plain declarations (qoamax_plain.h) must be compatible with the library's definitions.
	$(CC) $(CFLAGS) -D__STDC_NO_VLA__=1 -fsyntax-only src/qoamax.c
	test/cli.sh $(BUILD_DIR)/qoamax-debug $(BUILD_DIR)/wavgen

# Builds and installs the library with the Makefile, CMake and Meson, and builds programs using it in each way.
test-build-systems:
	test/build_systems.sh $(BUILD_DIR)/build-systems

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/qoamax-debug: bin/qoamax.c $(LIB_SOURCES) | $(BUILD_DIR)
	$(CC) $(DEBUG_CFLAGS) bin/qoamax.c -o $@ -lm

$(BUILD_DIR)/qoamax: bin/qoamax.c $(LIB_SOURCES) | $(BUILD_DIR)
	$(CC) $(RELEASE_CFLAGS) bin/qoamax.c -o $@ -lm

$(BUILD_DIR)/libqoamax.o: $(LIB_SOURCES) | $(BUILD_DIR)
	$(CC) $(RELEASE_CFLAGS) -c src/qoamax.c -o $@

$(BUILD_DIR)/libqoamax.a: $(BUILD_DIR)/libqoamax.o
	rm -f $@
	$(AR) rcs $@ $<

$(BUILD_DIR)/libqoamax.pic.o: $(LIB_SOURCES) | $(BUILD_DIR)
	$(CC) $(RELEASE_CFLAGS) -fPIC -c src/qoamax.c -o $@

# libqoamax.so -> libqoamax.so.$(SOVERSION) -> libqoamax.so.$(VERSION)
$(SHARED_LIB): $(BUILD_DIR)/libqoamax.pic.o
	$(CC) -shared -Wl,-soname,libqoamax.so.$(SOVERSION) $(LDFLAGS) $< -o $@
	ln -sf libqoamax.so.$(VERSION) $(BUILD_DIR)/libqoamax.so.$(SOVERSION)
	ln -sf libqoamax.so.$(SOVERSION) $(BUILD_DIR)/libqoamax.so

install: release lib
	install -d $(DESTDIR)$(BINDIR) $(DESTDIR)$(LIBDIR) $(DESTDIR)$(INCLUDEDIR) $(DESTDIR)$(PKGCONFIGDIR)
	install -m 755 $(BUILD_DIR)/qoamax $(DESTDIR)$(BINDIR)/qoamax
	install -m 644 src/qoamax.h src/qoamax_plain.h $(DESTDIR)$(INCLUDEDIR)/
	install -m 644 $(BUILD_DIR)/libqoamax.a $(DESTDIR)$(LIBDIR)/
	install -m 755 $(SHARED_LIB) $(DESTDIR)$(LIBDIR)/
	ln -sf libqoamax.so.$(VERSION) $(DESTDIR)$(LIBDIR)/libqoamax.so.$(SOVERSION)
	ln -sf libqoamax.so.$(SOVERSION) $(DESTDIR)$(LIBDIR)/libqoamax.so
	@# Generated here, as it holds the installation directories.
	sed -e 's|@PREFIX@|$(PREFIX)|' -e 's|@LIBDIR@|$(LIBDIR)|' -e 's|@INCLUDEDIR@|$(INCLUDEDIR)|' \
	    -e 's|@VERSION@|$(VERSION)|' qoamax.pc.in > $(DESTDIR)$(PKGCONFIGDIR)/qoamax.pc

uninstall:
	rm -f $(DESTDIR)$(BINDIR)/qoamax $(DESTDIR)$(INCLUDEDIR)/qoamax.h $(DESTDIR)$(INCLUDEDIR)/qoamax_plain.h \
	      $(DESTDIR)$(LIBDIR)/libqoamax.a $(DESTDIR)$(LIBDIR)/libqoamax.so $(DESTDIR)$(LIBDIR)/libqoamax.so.$(SOVERSION) \
	      $(DESTDIR)$(LIBDIR)/libqoamax.so.$(VERSION) $(DESTDIR)$(PKGCONFIGDIR)/qoamax.pc

$(BUILD_DIR)/test_lib: test/test_lib.c test/signal.h $(LIB_SOURCES) | $(BUILD_DIR)
	$(CC) $(TEST_CFLAGS) test/test_lib.c -o $@ -lm

$(BUILD_DIR)/test_lib_lanes%: test/test_lib.c test/signal.h $(LIB_SOURCES) | $(BUILD_DIR)
	$(CC) $(TEST_CFLAGS) -DQOAMAX_LANES=$* test/test_lib.c -o $@ -lm

# The C++ test links the library compiled as C.
$(BUILD_DIR)/qoamax-test.o: $(LIB_SOURCES) | $(BUILD_DIR)
	$(CC) $(TEST_CFLAGS) -c src/qoamax.c -o $@

$(BUILD_DIR)/test_cpp: test/test_cpp.cpp test/signal.h $(BUILD_DIR)/qoamax-test.o
	$(CXX) $(TEST_CXXFLAGS) test/test_cpp.cpp $(BUILD_DIR)/qoamax-test.o -o $@

$(BUILD_DIR)/wavgen: test/wavgen.c test/signal.h | $(BUILD_DIR)
	$(CC) $(TEST_CFLAGS) test/wavgen.c -o $@

clean:
	rm -rf $(BUILD_DIR)
