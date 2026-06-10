CC = gcc

INCLUDE_DIR = include
SRC_DIR = src
TEST_DIR = tests
EXAMPLE_DIR = examples
BENCHMARK_DIR = benchmarks
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/obj
BIN_DIR = bin

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
	$(SRC_DIR)/tensor_ops.c

MATRIX_OBJS = \
	$(OBJ_DIR)/matrix.o \
	$(OBJ_DIR)/matrix_optimized.o \
	$(OBJ_DIR)/backend.o \
	$(OBJ_DIR)/matrix_lu.o \
	$(OBJ_DIR)/matrix_solve.o

TENSOR_OBJS = \
	$(OBJ_DIR)/tensor.o \
	$(OBJ_DIR)/tensor_ops.o

TARGET = $(BIN_DIR)/matrix_demo
TENSOR_TARGET = $(BIN_DIR)/tensor_demo
MLP_TARGET = $(BIN_DIR)/mlp_demo
BENCHMARK_TARGET = $(BIN_DIR)/matrix_benchmark
SMALL_BENCHMARK_TARGET = $(BIN_DIR)/matrix_benchmark_small

.PHONY: all run run-tensor run-mlp benchmark benchmark-small clean dirs \
	matrix_demo tensor_demo mlp_demo matrix_benchmark matrix_benchmark_small

all: $(TARGET) $(TENSOR_TARGET) $(MLP_TARGET)

matrix_demo: $(TARGET)

tensor_demo: $(TENSOR_TARGET)

mlp_demo: $(MLP_TARGET)

matrix_benchmark: $(BENCHMARK_TARGET)

matrix_benchmark_small: $(SMALL_BENCHMARK_TARGET)

dirs:
	mkdir -p $(OBJ_DIR) $(BIN_DIR)

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

$(OBJ_DIR)/main.o: $(TEST_DIR)/main.c $(INCLUDE_DIR)/*.h | dirs
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/demo_tensor.o: $(EXAMPLE_DIR)/demo_tensor.c $(INCLUDE_DIR)/*.h | dirs
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/demo_mlp.o: $(EXAMPLE_DIR)/demo_mlp.c $(INCLUDE_DIR)/*.h | dirs
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

$(OBJ_DIR)/backend.o: $(SRC_DIR)/backend.c $(INCLUDE_DIR)/backend.h | dirs
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/benchmark_benchmark.o: $(BENCHMARK_DIR)/benchmark.c $(INCLUDE_DIR)/*.h | dirs
	$(CC) $(CPPFLAGS) $(BENCHMARK_CFLAGS) -c $< -o $@

$(OBJ_DIR)/matrix_benchmark.o: $(SRC_DIR)/matrix.c $(INCLUDE_DIR)/matrix.h | dirs
	$(CC) $(CPPFLAGS) $(BENCHMARK_CFLAGS) -c $< -o $@

$(OBJ_DIR)/matrix_optimized_benchmark.o: $(SRC_DIR)/matrix_optimized.c $(INCLUDE_DIR)/matrix_optimized.h $(INCLUDE_DIR)/matrix.h | dirs
	$(CC) $(CPPFLAGS) $(BENCHMARK_CFLAGS) -c $< -o $@

$(OBJ_DIR)/tensor_benchmark.o: $(SRC_DIR)/tensor.c $(INCLUDE_DIR)/tensor.h $(INCLUDE_DIR)/matrix.h | dirs
	$(CC) $(CPPFLAGS) $(BENCHMARK_CFLAGS) -c $< -o $@

$(OBJ_DIR)/tensor_ops_benchmark.o: $(SRC_DIR)/tensor_ops.c $(INCLUDE_DIR)/*.h | dirs
	$(CC) $(CPPFLAGS) $(BENCHMARK_CFLAGS) -c $< -o $@

$(OBJ_DIR)/backend_benchmark.o: $(SRC_DIR)/backend.c $(INCLUDE_DIR)/backend.h | dirs
	$(CC) $(CPPFLAGS) $(BENCHMARK_CFLAGS) -c $< -o $@

$(OBJ_DIR)/benchmark_small.o: $(BENCHMARK_DIR)/benchmark.c $(INCLUDE_DIR)/*.h | dirs
	$(CC) $(CPPFLAGS) $(SMALL_BENCHMARK_CFLAGS) -c $< -o $@

$(OBJ_DIR)/matrix_small.o: $(SRC_DIR)/matrix.c $(INCLUDE_DIR)/matrix.h | dirs
	$(CC) $(CPPFLAGS) $(SMALL_BENCHMARK_CFLAGS) -c $< -o $@

$(OBJ_DIR)/matrix_optimized_small.o: $(SRC_DIR)/matrix_optimized.c $(INCLUDE_DIR)/matrix_optimized.h $(INCLUDE_DIR)/matrix.h | dirs
	$(CC) $(CPPFLAGS) $(SMALL_BENCHMARK_CFLAGS) -c $< -o $@

$(OBJ_DIR)/tensor_small.o: $(SRC_DIR)/tensor.c $(INCLUDE_DIR)/tensor.h $(INCLUDE_DIR)/matrix.h | dirs
	$(CC) $(CPPFLAGS) $(SMALL_BENCHMARK_CFLAGS) -c $< -o $@

$(OBJ_DIR)/tensor_ops_small.o: $(SRC_DIR)/tensor_ops.c $(INCLUDE_DIR)/*.h | dirs
	$(CC) $(CPPFLAGS) $(SMALL_BENCHMARK_CFLAGS) -c $< -o $@

$(OBJ_DIR)/backend_small.o: $(SRC_DIR)/backend.c $(INCLUDE_DIR)/backend.h | dirs
	$(CC) $(CPPFLAGS) $(SMALL_BENCHMARK_CFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

run-tensor: $(TENSOR_TARGET)
	./$(TENSOR_TARGET)

run-mlp: $(MLP_TARGET)
	./$(MLP_TARGET)

benchmark: $(BENCHMARK_TARGET)
	./$(BENCHMARK_TARGET)

benchmark-small: $(SMALL_BENCHMARK_TARGET)
	./$(SMALL_BENCHMARK_TARGET)

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)
