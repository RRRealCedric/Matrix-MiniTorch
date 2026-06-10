CC = gcc

ifeq ($(OS),Windows_NT)
EXEEXT := .exe
RUN_PREFIX =
MKDIR_P = powershell -NoProfile -Command "New-Item -ItemType Directory -Force -Path '$(subst /,\,$(1))' | Out-Null"
RM_RF = powershell -NoProfile -Command "if (Test-Path '$(subst /,\,$(1))') { Remove-Item -LiteralPath '$(subst /,\,$(1))' -Recurse -Force }"
else
EXEEXT :=
RUN_PREFIX = ./
MKDIR_P = mkdir -p "$(1)"
RM_RF = rm -rf "$(1)"
endif

INCLUDE_DIR = include
SRC_DIR = src
TEST_DIR = tests
EXAMPLE_DIR = examples
BENCHMARK_DIR = benchmarks
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/obj
BIN_DIR = bin
ARTIFACTS_DIR = artifacts
ARTIFACTS_BENCHMARK_DIR = $(ARTIFACTS_DIR)/benchmark
DATA_DIR = data

USER_CPPFLAGS ?=
CPPFLAGS = -I$(INCLUDE_DIR) $(USER_CPPFLAGS)
CFLAGS = -std=c99 -Wall -Wextra -pedantic -O0 -g
BENCHMARK_CFLAGS = -std=c99 -Wall -Wextra -pedantic -O2 -g
SMALL_BENCHMARK_CFLAGS = $(BENCHMARK_CFLAGS) -DBENCHMARK_REPEAT=2 -DBENCHMARK_SIZE_1=16 -DBENCHMARK_SIZE_2=32 -DBENCHMARK_SIZE_3=64
LDFLAGS = -lm

MATRIX_SOURCES = \
	$(SRC_DIR)/matrix.c \
	$(SRC_DIR)/matrix_optimized.c \
	$(SRC_DIR)/backend.c \
	$(SRC_DIR)/matrix_lu.c \
	$(SRC_DIR)/matrix_solve.c

TENSOR_SOURCES = \
	$(SRC_DIR)/tensor.c \
	$(SRC_DIR)/tensor_ops.c \
	$(SRC_DIR)/digits_dataset.c

MATRIX_OBJS = \
	$(OBJ_DIR)/matrix.o \
	$(OBJ_DIR)/matrix_optimized.o \
	$(OBJ_DIR)/backend.o \
	$(OBJ_DIR)/matrix_lu.o \
	$(OBJ_DIR)/matrix_solve.o

TENSOR_OBJS = \
	$(OBJ_DIR)/tensor.o \
	$(OBJ_DIR)/tensor_ops.o \
	$(OBJ_DIR)/digits_dataset.o

TARGET = $(BIN_DIR)/matrix_demo$(EXEEXT)
TENSOR_TARGET = $(BIN_DIR)/tensor_demo$(EXEEXT)
MLP_TARGET = $(BIN_DIR)/mlp_demo$(EXEEXT)
DIGITS_TARGET = $(BIN_DIR)/digits_demo$(EXEEXT)
BENCHMARK_TARGET = $(BIN_DIR)/matrix_benchmark$(EXEEXT)
SMALL_BENCHMARK_TARGET = $(BIN_DIR)/matrix_benchmark_small$(EXEEXT)

.PHONY: all run run-tensor run-mlp run-digits benchmark benchmark-small benchmark-tuned \
	clean dirs artifacts-dirs matrix_demo tensor_demo mlp_demo digits_demo \
	matrix_benchmark matrix_benchmark_small

all: $(TARGET) $(TENSOR_TARGET) $(MLP_TARGET) $(DIGITS_TARGET)

matrix_demo: $(TARGET)

tensor_demo: $(TENSOR_TARGET)

mlp_demo: $(MLP_TARGET)

digits_demo: $(DIGITS_TARGET)

matrix_benchmark: $(BENCHMARK_TARGET)

matrix_benchmark_small: $(SMALL_BENCHMARK_TARGET)

dirs:
	$(call MKDIR_P,$(BUILD_DIR))
	$(call MKDIR_P,$(OBJ_DIR))
	$(call MKDIR_P,$(BIN_DIR))

artifacts-dirs:
	$(call MKDIR_P,$(ARTIFACTS_DIR))
	$(call MKDIR_P,$(ARTIFACTS_BENCHMARK_DIR))
	$(call MKDIR_P,$(DATA_DIR))

$(TARGET): $(OBJ_DIR)/main.o $(MATRIX_OBJS) | dirs
	$(CC) $(CFLAGS) -o $@ $(OBJ_DIR)/main.o $(MATRIX_OBJS) $(LDFLAGS)

$(TENSOR_TARGET): $(OBJ_DIR)/demo_tensor.o $(MATRIX_OBJS) $(TENSOR_OBJS) | dirs
	$(CC) $(CFLAGS) -o $@ $(OBJ_DIR)/demo_tensor.o $(MATRIX_OBJS) $(TENSOR_OBJS) $(LDFLAGS)

$(MLP_TARGET): $(OBJ_DIR)/demo_mlp.o $(MATRIX_OBJS) $(TENSOR_OBJS) | dirs
	$(CC) $(CFLAGS) -o $@ $(OBJ_DIR)/demo_mlp.o $(MATRIX_OBJS) $(TENSOR_OBJS) $(LDFLAGS)

$(BENCHMARK_TARGET): \
	$(OBJ_DIR)/benchmark_benchmark.o \
	$(OBJ_DIR)/matrix_benchmark.o \
	$(OBJ_DIR)/matrix_optimized_benchmark.o \
	$(OBJ_DIR)/tensor_benchmark.o \
	$(OBJ_DIR)/tensor_ops_benchmark.o \
	$(OBJ_DIR)/backend_benchmark.o | dirs
	$(CC) $(BENCHMARK_CFLAGS) -o $@ $^ $(LDFLAGS)

$(SMALL_BENCHMARK_TARGET): \
	$(OBJ_DIR)/benchmark_small.o \
	$(OBJ_DIR)/matrix_small.o \
	$(OBJ_DIR)/matrix_optimized_small.o \
	$(OBJ_DIR)/tensor_small.o \
	$(OBJ_DIR)/tensor_ops_small.o \
	$(OBJ_DIR)/backend_small.o | dirs
	$(CC) $(SMALL_BENCHMARK_CFLAGS) -o $@ $^ $(LDFLAGS)

$(DIGITS_TARGET): $(OBJ_DIR)/demo_digits.o $(MATRIX_OBJS) $(TENSOR_OBJS) | dirs
	$(CC) $(CFLAGS) -o $@ $(OBJ_DIR)/demo_digits.o $(MATRIX_OBJS) $(TENSOR_OBJS) $(LDFLAGS)

$(OBJ_DIR)/main.o: $(TEST_DIR)/main.c $(wildcard $(INCLUDE_DIR)/*.h) | dirs
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/demo_tensor.o: $(EXAMPLE_DIR)/demo_tensor.c $(wildcard $(INCLUDE_DIR)/*.h) | dirs
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/demo_mlp.o: $(EXAMPLE_DIR)/demo_mlp.c $(wildcard $(INCLUDE_DIR)/*.h) | dirs
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/demo_digits.o: $(EXAMPLE_DIR)/demo_digits.c $(wildcard $(INCLUDE_DIR)/*.h) | dirs
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/matrix.o: $(SRC_DIR)/matrix.c $(INCLUDE_DIR)/matrix.h | dirs
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/matrix_optimized.o: $(SRC_DIR)/matrix_optimized.c $(INCLUDE_DIR)/matrix_optimized.h $(INCLUDE_DIR)/matrix.h | dirs
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/matrix_lu.o: $(SRC_DIR)/matrix_lu.c $(INCLUDE_DIR)/matrix_lu.h $(INCLUDE_DIR)/matrix.h | dirs
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/matrix_solve.o: $(SRC_DIR)/matrix_solve.c $(INCLUDE_DIR)/matrix_solve.h $(INCLUDE_DIR)/matrix.h | dirs
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/tensor.o: $(SRC_DIR)/tensor.c $(INCLUDE_DIR)/tensor.h $(INCLUDE_DIR)/matrix.h | dirs
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/tensor_ops.o: $(SRC_DIR)/tensor_ops.c $(INCLUDE_DIR)/*.h | dirs
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/digits_dataset.o: $(SRC_DIR)/digits_dataset.c $(wildcard $(INCLUDE_DIR)/*.h) | dirs
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/backend.o: $(SRC_DIR)/backend.c $(INCLUDE_DIR)/backend.h | dirs
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/benchmark_benchmark.o: $(BENCHMARK_DIR)/benchmark.c $(wildcard $(INCLUDE_DIR)/*.h) | dirs
	$(CC) $(CPPFLAGS) $(BENCHMARK_CFLAGS) -c $< -o $@

$(OBJ_DIR)/matrix_benchmark.o: $(SRC_DIR)/matrix.c $(INCLUDE_DIR)/matrix.h | dirs
	$(CC) $(CPPFLAGS) $(BENCHMARK_CFLAGS) -c $< -o $@

$(OBJ_DIR)/matrix_optimized_benchmark.o: $(SRC_DIR)/matrix_optimized.c $(INCLUDE_DIR)/matrix_optimized.h $(INCLUDE_DIR)/matrix.h | dirs
	$(CC) $(CPPFLAGS) $(BENCHMARK_CFLAGS) -c $< -o $@

$(OBJ_DIR)/tensor_benchmark.o: $(SRC_DIR)/tensor.c $(INCLUDE_DIR)/tensor.h $(INCLUDE_DIR)/matrix.h | dirs
	$(CC) $(CPPFLAGS) $(BENCHMARK_CFLAGS) -c $< -o $@

$(OBJ_DIR)/tensor_ops_benchmark.o: $(SRC_DIR)/tensor_ops.c $(wildcard $(INCLUDE_DIR)/*.h) | dirs
	$(CC) $(CPPFLAGS) $(BENCHMARK_CFLAGS) -c $< -o $@

$(OBJ_DIR)/backend_benchmark.o: $(SRC_DIR)/backend.c $(INCLUDE_DIR)/backend.h | dirs
	$(CC) $(CPPFLAGS) $(BENCHMARK_CFLAGS) -c $< -o $@

$(OBJ_DIR)/benchmark_small.o: $(BENCHMARK_DIR)/benchmark.c $(wildcard $(INCLUDE_DIR)/*.h) | dirs
	$(CC) $(CPPFLAGS) $(SMALL_BENCHMARK_CFLAGS) -c $< -o $@

$(OBJ_DIR)/matrix_small.o: $(SRC_DIR)/matrix.c $(INCLUDE_DIR)/matrix.h | dirs
	$(CC) $(CPPFLAGS) $(SMALL_BENCHMARK_CFLAGS) -c $< -o $@

$(OBJ_DIR)/matrix_optimized_small.o: $(SRC_DIR)/matrix_optimized.c $(INCLUDE_DIR)/matrix_optimized.h $(INCLUDE_DIR)/matrix.h | dirs
	$(CC) $(CPPFLAGS) $(SMALL_BENCHMARK_CFLAGS) -c $< -o $@

$(OBJ_DIR)/tensor_small.o: $(SRC_DIR)/tensor.c $(INCLUDE_DIR)/tensor.h $(INCLUDE_DIR)/matrix.h | dirs
	$(CC) $(CPPFLAGS) $(SMALL_BENCHMARK_CFLAGS) -c $< -o $@

$(OBJ_DIR)/tensor_ops_small.o: $(SRC_DIR)/tensor_ops.c $(wildcard $(INCLUDE_DIR)/*.h) | dirs
	$(CC) $(CPPFLAGS) $(SMALL_BENCHMARK_CFLAGS) -c $< -o $@

$(OBJ_DIR)/backend_small.o: $(SRC_DIR)/backend.c $(INCLUDE_DIR)/backend.h | dirs
	$(CC) $(CPPFLAGS) $(SMALL_BENCHMARK_CFLAGS) -c $< -o $@

run: $(TARGET)
	$(RUN_PREFIX)$(TARGET)

run-tensor: $(TENSOR_TARGET)
	$(RUN_PREFIX)$(TENSOR_TARGET)

run-mlp: $(MLP_TARGET)
	$(RUN_PREFIX)$(MLP_TARGET)

run-digits: $(DIGITS_TARGET) | artifacts-dirs
	$(RUN_PREFIX)$(DIGITS_TARGET)

benchmark: $(BENCHMARK_TARGET)
	$(RUN_PREFIX)$(BENCHMARK_TARGET)

benchmark-small: $(SMALL_BENCHMARK_TARGET)
	$(RUN_PREFIX)$(SMALL_BENCHMARK_TARGET)

benchmark-tuned: $(BENCHMARK_TARGET) | artifacts-dirs
	$(RUN_PREFIX)$(BENCHMARK_TARGET) --sizes 128,256,512,1000 --repeat 5 --block-sizes 8,16,32,64 --csv $(ARTIFACTS_BENCHMARK_DIR)/benchmark_tuned.csv

clean:
	$(call RM_RF,$(BUILD_DIR))
	$(call RM_RF,$(BIN_DIR))
	$(call RM_RF,$(ARTIFACTS_DIR))
