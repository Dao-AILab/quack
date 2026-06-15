# Build the cute-DSL CUDA 13.3 toolchain shim.
#
#   make           — builds libcute_dsl_shim.so in this directory
#   make verify    — builds and runs verify.py to confirm the shim is active
#   make clean     — removes build artifacts
#
# The .so has no Python dependency; it is loaded by quack/dsl/cute_dsl_shim.py
# via ctypes at quack-import time.

CC      ?= cc
CFLAGS  ?= -O2 -fPIC -fvisibility=hidden -Wall -Wextra -Wno-unused-parameter
LDFLAGS ?= -shared -ldl -lpthread

TARGET  := libcute_dsl_shim.so
SRC     := cute_dsl_shim.c

.PHONY: all clean verify

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)

# Build then run the verify script with QUACK_CUTE_DSL_SHIM=1 so the
# Python loader picks up THIS build (not a stale install).
verify: $(TARGET)
	QUACK_CUTE_DSL_SHIM=1 \
	QUACK_CUTE_DSL_SHIM_LIB=$(CURDIR)/$(TARGET) \
	python3 verify.py
