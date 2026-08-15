CC ?= cc
AR ?= ar
PYTHON ?= python3
CSTD ?= -std=c99
WARNINGS ?= -Wall -Wextra -Wpedantic -Werror -Wshadow -Wconversion -Wstrict-prototypes
OPT ?= -O3
CPPFLAGS ?= -Iinclude
CFLAGS ?= $(CSTD) $(WARNINGS) $(OPT) -fPIC
LDFLAGS ?=
LDLIBS ?= -lm

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
SHARED_EXT := dylib
SHARED_FLAGS := -dynamiclib -Wl,-install_name,@rpath/libperbacco.dylib
else
SHARED_EXT := so
SHARED_FLAGS := -shared -Wl,-soname,libperbacco.so
endif

BUILD_DIR := build
LIB_OBJ := $(BUILD_DIR)/perbacco.o
LIB_STATIC := $(BUILD_DIR)/libperbacco.a
LIB_SHARED := $(BUILD_DIR)/libperbacco.$(SHARED_EXT)
TEST_BIN := $(BUILD_DIR)/test_core

.PHONY: all clean test test-c test-python check sanitize

all: $(LIB_STATIC) $(LIB_SHARED)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(LIB_OBJ): src/perbacco.c include/perbacco.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DPB_BUILD_SHARED -c $< -o $@

$(LIB_STATIC): $(LIB_OBJ)
	$(AR) rcs $@ $^

$(LIB_SHARED): $(LIB_OBJ)
	$(CC) $(SHARED_FLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(TEST_BIN): tests/test_core.c $(LIB_STATIC) include/perbacco.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CSTD) $(WARNINGS) -O0 -g tests/test_core.c $(LIB_STATIC) $(LDFLAGS) $(LDLIBS) -o $@

test-c: $(TEST_BIN)
	$(TEST_BIN)

test-python: all
	PERBACCO_LIBRARY=$(abspath $(LIB_SHARED)) PYTHONPATH=python $(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v

test: test-c test-python

check: test

sanitize: clean
	$(MAKE) CFLAGS='$(CSTD) $(WARNINGS) -O1 -g -fPIC -fsanitize=address,undefined -fno-omit-frame-pointer' LDFLAGS='-fsanitize=address,undefined' test-c
	$(MAKE) clean

clean:
	rm -rf $(BUILD_DIR)
