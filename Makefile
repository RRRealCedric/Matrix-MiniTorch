CC = gcc
CFLAGS = -std=c99 -Wall -Wextra -pedantic -O0 -g
BENCHMARK_CFLAGS = -std=c99 -Wall -Wextra -pedantic -O2 -g
SMALL_BENCHMARK_CFLAGS = $(BENCHMARK_CFLAGS) -DBENCHMARK_REPEAT=2 -DBENCHMARK_SIZE_1=16 -DBENCHMARK_SIZE_2=32 -DBENCHMARK_SIZE_3=64
LDFLAGS = -lm

MATRIX_OBJS = matrix.o matrix_optimized.o matrix_lu.o matrix_solve.o
TENSOR_OBJS = tensor.o tensor_ops.o backend.o

TARGET = matrix_demo
TENSOR_TARGET = tensor_demo
MLP_TARGET = mlp_demo
BENCHMARK_TARGET = matrix_benchmark
SMALL_BENCHMARK_TARGET = matrix_benchmark_small

.PHONY: all run run-tensor run-mlp benchmark benchmark-small clean

all: $(TARGET) $(TENSOR_TARGET) $(MLP_TARGET)

$(TARGET): main.o $(MATRIX_OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) main.o $(MATRIX_OBJS) $(LDFLAGS)

$(TENSOR_TARGET): demo_tensor.o $(MATRIX_OBJS) $(TENSOR_OBJS)
	$(CC) $(CFLAGS) -o $(TENSOR_TARGET) demo_tensor.o $(MATRIX_OBJS) $(TENSOR_OBJS) $(LDFLAGS)

$(MLP_TARGET): demo_mlp.o $(MATRIX_OBJS) $(TENSOR_OBJS)
	$(CC) $(CFLAGS) -o $(MLP_TARGET) demo_mlp.o $(MATRIX_OBJS) $(TENSOR_OBJS) $(LDFLAGS)

$(BENCHMARK_TARGET): benchmark_benchmark.o matrix_benchmark.o matrix_optimized_benchmark.o tensor_benchmark.o tensor_ops_benchmark.o backend_benchmark.o
	$(CC) $(BENCHMARK_CFLAGS) -o $(BENCHMARK_TARGET) benchmark_benchmark.o matrix_benchmark.o matrix_optimized_benchmark.o tensor_benchmark.o tensor_ops_benchmark.o backend_benchmark.o $(LDFLAGS)

$(SMALL_BENCHMARK_TARGET): benchmark_small.o matrix_small.o matrix_optimized_small.o tensor_small.o tensor_ops_small.o backend_small.o
	$(CC) $(SMALL_BENCHMARK_CFLAGS) -o $(SMALL_BENCHMARK_TARGET) benchmark_small.o matrix_small.o matrix_optimized_small.o tensor_small.o tensor_ops_small.o backend_small.o $(LDFLAGS)

main.o: main.c matrix.h matrix_optimized.h matrix_lu.h matrix_solve.h
	$(CC) $(CFLAGS) -c main.c

demo_tensor.o: demo_tensor.c tensor.h tensor_ops.h backend.h matrix.h
	$(CC) $(CFLAGS) -c demo_tensor.c

demo_mlp.o: demo_mlp.c tensor.h tensor_ops.h backend.h matrix.h
	$(CC) $(CFLAGS) -c demo_mlp.c

matrix.o: matrix.c matrix.h
	$(CC) $(CFLAGS) -c matrix.c

matrix_optimized.o: matrix_optimized.c matrix_optimized.h matrix.h
	$(CC) $(CFLAGS) -c matrix_optimized.c

matrix_lu.o: matrix_lu.c matrix_lu.h matrix.h
	$(CC) $(CFLAGS) -c matrix_lu.c

matrix_solve.o: matrix_solve.c matrix_solve.h matrix.h
	$(CC) $(CFLAGS) -c matrix_solve.c

tensor.o: tensor.c tensor.h matrix.h
	$(CC) $(CFLAGS) -c tensor.c

tensor_ops.o: tensor_ops.c tensor_ops.h tensor.h backend.h matrix.h matrix_optimized.h
	$(CC) $(CFLAGS) -c tensor_ops.c

backend.o: backend.c backend.h
	$(CC) $(CFLAGS) -c backend.c

benchmark_benchmark.o: benchmark.c matrix.h matrix_optimized.h tensor.h tensor_ops.h backend.h
	$(CC) $(BENCHMARK_CFLAGS) -c benchmark.c -o benchmark_benchmark.o

matrix_benchmark.o: matrix.c matrix.h
	$(CC) $(BENCHMARK_CFLAGS) -c matrix.c -o matrix_benchmark.o

matrix_optimized_benchmark.o: matrix_optimized.c matrix_optimized.h matrix.h
	$(CC) $(BENCHMARK_CFLAGS) -c matrix_optimized.c -o matrix_optimized_benchmark.o

tensor_benchmark.o: tensor.c tensor.h matrix.h
	$(CC) $(BENCHMARK_CFLAGS) -c tensor.c -o tensor_benchmark.o

tensor_ops_benchmark.o: tensor_ops.c tensor_ops.h tensor.h backend.h matrix.h matrix_optimized.h
	$(CC) $(BENCHMARK_CFLAGS) -c tensor_ops.c -o tensor_ops_benchmark.o

backend_benchmark.o: backend.c backend.h
	$(CC) $(BENCHMARK_CFLAGS) -c backend.c -o backend_benchmark.o

benchmark_small.o: benchmark.c matrix.h matrix_optimized.h tensor.h tensor_ops.h backend.h
	$(CC) $(SMALL_BENCHMARK_CFLAGS) -c benchmark.c -o benchmark_small.o

matrix_small.o: matrix.c matrix.h
	$(CC) $(SMALL_BENCHMARK_CFLAGS) -c matrix.c -o matrix_small.o

matrix_optimized_small.o: matrix_optimized.c matrix_optimized.h matrix.h
	$(CC) $(SMALL_BENCHMARK_CFLAGS) -c matrix_optimized.c -o matrix_optimized_small.o

tensor_small.o: tensor.c tensor.h matrix.h
	$(CC) $(SMALL_BENCHMARK_CFLAGS) -c tensor.c -o tensor_small.o

tensor_ops_small.o: tensor_ops.c tensor_ops.h tensor.h backend.h matrix.h matrix_optimized.h
	$(CC) $(SMALL_BENCHMARK_CFLAGS) -c tensor_ops.c -o tensor_ops_small.o

backend_small.o: backend.c backend.h
	$(CC) $(SMALL_BENCHMARK_CFLAGS) -c backend.c -o backend_small.o

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
	rm -f $(TARGET) $(TENSOR_TARGET) $(MLP_TARGET) $(BENCHMARK_TARGET) $(SMALL_BENCHMARK_TARGET) *.o
