CC := clang
BUILD_DIR := build

CFLAGS := -std=c23 -Wno-unknown-attributes -Wno-unknown-pragmas
DEBUG_CFLAGS := $(CFLAGS) -O0 -g -fwrapv \
                -fsanitize=address,undefined \
                -fstack-protector-strong
RELEASE_CFLAGS := $(CFLAGS) -O3
# The tests run with sanitizers (without -fwrapv, so that UBSan reports signed overflows), stopping at the first error.
TEST_CFLAGS := $(CFLAGS) -O1 -g -Wall -Wextra -pthread \
               -fsanitize=address,undefined -fno-sanitize-recover=all \
               -fstack-protector-strong
# The library tests also run with these SIMD lane counts (besides the target's default), which must all produce the
# same output. Lane counts above the target's vector width work too, the compiler splits the vectors.
TEST_LANES := 1 2 4 8 16
TEST_LIBS := $(BUILD_DIR)/test_lib $(foreach lanes,$(TEST_LANES),$(BUILD_DIR)/test_lib_lanes$(lanes))

.PHONY: all debug release test clean

all: debug release

debug: $(BUILD_DIR)/qoamax-debug

release: $(BUILD_DIR)/qoamax

test: $(TEST_LIBS) $(BUILD_DIR)/wavgen $(BUILD_DIR)/qoamax-debug
	@for test in $(TEST_LIBS); do $$test || exit 1; done
	test/cli.sh $(BUILD_DIR)/qoamax-debug $(BUILD_DIR)/wavgen

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/qoamax-debug: bin/qoamax.c src/qoamax.h src/qoamax.c src/qoamax_lanes.h src/qoamax_beam.h | $(BUILD_DIR)
	$(CC) $(DEBUG_CFLAGS) bin/qoamax.c -o $@ -lm

$(BUILD_DIR)/qoamax: bin/qoamax.c src/qoamax.h src/qoamax.c src/qoamax_lanes.h src/qoamax_beam.h | $(BUILD_DIR)
	$(CC) $(RELEASE_CFLAGS) bin/qoamax.c -o $@ -lm

$(BUILD_DIR)/test_lib: test/test_lib.c test/signal.h src/qoamax.h src/qoamax.c src/qoamax_lanes.h src/qoamax_beam.h | $(BUILD_DIR)
	$(CC) $(TEST_CFLAGS) test/test_lib.c -o $@ -lm

$(BUILD_DIR)/test_lib_lanes%: test/test_lib.c test/signal.h src/qoamax.h src/qoamax.c src/qoamax_lanes.h src/qoamax_beam.h | $(BUILD_DIR)
	$(CC) $(TEST_CFLAGS) -DQOAMAX_LANES=$* test/test_lib.c -o $@ -lm

$(BUILD_DIR)/wavgen: test/wavgen.c test/signal.h | $(BUILD_DIR)
	$(CC) $(TEST_CFLAGS) test/wavgen.c -o $@

clean:
	rm -rf $(BUILD_DIR)
