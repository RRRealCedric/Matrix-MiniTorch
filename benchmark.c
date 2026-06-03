#include "backend.h"
#include "matrix.h"
#include "matrix_optimized.h"
#include "tensor.h"
#include "tensor_ops.h"

#include <stdio.h>
#include <time.h>

#ifndef BENCHMARK_REPEAT
#define BENCHMARK_REPEAT 5
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

#ifndef BENCHMARK_SIZE_1
#define BENCHMARK_SIZE_1 100
#endif

#ifndef BENCHMARK_SIZE_2
#define BENCHMARK_SIZE_2 1000
#endif

#ifndef BENCHMARK_SIZE_3
#define BENCHMARK_SIZE_3 2000
#endif

static double NowSeconds(void)
{
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

static int CreateBenchmarkMatrices(int n, Matrix *A, Matrix *B, Matrix *C, Matrix *AT)
{
    MatrixInit(A);
    MatrixInit(B);
    MatrixInit(C);
    MatrixInit(AT);

    if (MatrixCreate(A, n, n) != MATRIX_SUCCESS ||
        MatrixCreate(B, n, n) != MATRIX_SUCCESS ||
        MatrixCreate(C, n, n) != MATRIX_SUCCESS ||
        MatrixCreate(AT, n, n) != MATRIX_SUCCESS) {
        MatrixFree(A);
        MatrixFree(B);
        MatrixFree(C);
        MatrixFree(AT);
        return 0;
    }

    MatrixFillSequence(A, 1.0, 0.001);
    MatrixFillSequence(B, 2.0, 0.001);
    MatrixFillZero(C);
    MatrixFillZero(AT);
    return 1;
}

static double TimeAdd(int n)
{
    Matrix A, B, C, AT;
    double start;
    double elapsed;

    if (!CreateBenchmarkMatrices(n, &A, &B, &C, &AT)) {
        return -1.0;
    }
    start = NowSeconds();
    MatrixAdd(&A, &B, &C);
    elapsed = NowSeconds() - start;
    MatrixFree(&A);
    MatrixFree(&B);
    MatrixFree(&C);
    MatrixFree(&AT);
    return elapsed;
}

static double TimeScale(int n)
{
    Matrix A, B, C, AT;
    double start;
    double elapsed;

    if (!CreateBenchmarkMatrices(n, &A, &B, &C, &AT)) {
        return -1.0;
    }
    start = NowSeconds();
    MatrixScale(2.0, &A, &C);
    elapsed = NowSeconds() - start;
    MatrixFree(&A);
    MatrixFree(&B);
    MatrixFree(&C);
    MatrixFree(&AT);
    return elapsed;
}

static double TimeTranspose(int n)
{
    Matrix A, B, C, AT;
    double start;
    double elapsed;

    if (!CreateBenchmarkMatrices(n, &A, &B, &C, &AT)) {
        return -1.0;
    }
    start = NowSeconds();
    MatrixTranspose(&A, &AT);
    elapsed = NowSeconds() - start;
    MatrixFree(&A);
    MatrixFree(&B);
    MatrixFree(&C);
    MatrixFree(&AT);
    return elapsed;
}

static double TimeMultiplyNaive(int n)
{
    Matrix A, B, C, AT;
    double start;
    double elapsed;

    if (!CreateBenchmarkMatrices(n, &A, &B, &C, &AT)) {
        return -1.0;
    }
    start = NowSeconds();
    MatrixMultiply(&A, &B, &C);
    elapsed = NowSeconds() - start;
    MatrixFree(&A);
    MatrixFree(&B);
    MatrixFree(&C);
    MatrixFree(&AT);
    return elapsed;
}

static double TimeMultiplyIKJ(int n)
{
    Matrix A, B, C, AT;
    double start;
    double elapsed;

    if (!CreateBenchmarkMatrices(n, &A, &B, &C, &AT)) {
        return -1.0;
    }
    start = NowSeconds();
    MatrixMultiplyIKJ(&A, &B, &C);
    elapsed = NowSeconds() - start;
    MatrixFree(&A);
    MatrixFree(&B);
    MatrixFree(&C);
    MatrixFree(&AT);
    return elapsed;
}

static double TimeMultiplyBlocked(int n)
{
    Matrix A, B, C, AT;
    double start;
    double elapsed;

    if (!CreateBenchmarkMatrices(n, &A, &B, &C, &AT)) {
        return -1.0;
    }
    start = NowSeconds();
    MatrixMultiplyBlocked(&A, &B, &C, BLOCK_SIZE);
    elapsed = NowSeconds() - start;
    MatrixFree(&A);
    MatrixFree(&B);
    MatrixFree(&C);
    MatrixFree(&AT);
    return elapsed;
}

static double TimeMultiplyTransposeB(int n)
{
    Matrix A, B, C, AT;
    double start;
    double elapsed;

    if (!CreateBenchmarkMatrices(n, &A, &B, &C, &AT)) {
        return -1.0;
    }
    start = NowSeconds();
    MatrixMultiplyTransposeB(&A, &B, &C);
    elapsed = NowSeconds() - start;
    MatrixFree(&A);
    MatrixFree(&B);
    MatrixFree(&C);
    MatrixFree(&AT);
    return elapsed;
}

static double TimeMultiplyKahan(int n)
{
    Matrix A, B, C, AT;
    double start;
    double elapsed;

    if (!CreateBenchmarkMatrices(n, &A, &B, &C, &AT)) {
        return -1.0;
    }
    start = NowSeconds();
    MatrixMultiplyKahan(&A, &B, &C);
    elapsed = NowSeconds() - start;
    MatrixFree(&A);
    MatrixFree(&B);
    MatrixFree(&C);
    MatrixFree(&AT);
    return elapsed;
}

static double TimeTensorMatmulBlocked(int n)
{
    Tensor A, B, C;
    double start;
    double elapsed;

    TensorInit(&A);
    TensorInit(&B);
    TensorInit(&C);

    if (TensorCreate2D(&A, n, n) != MATRIX_SUCCESS ||
        TensorCreate2D(&B, n, n) != MATRIX_SUCCESS ||
        TensorCreate2D(&C, n, n) != MATRIX_SUCCESS) {
        TensorFree(&A);
        TensorFree(&B);
        TensorFree(&C);
        return -1.0;
    }

    TensorFillSequence(&A, 1.0, 0.001);
    TensorFillSequence(&B, 2.0, 0.001);
    TensorFillZero(&C);
    BackendSetType(BACKEND_BLOCKED);

    start = NowSeconds();
    TensorMatmul(&A, &B, &C);
    elapsed = NowSeconds() - start;

    TensorFree(&A);
    TensorFree(&B);
    TensorFree(&C);
    return elapsed;
}

static double AverageTime(double (*func)(int), int n)
{
    double total = 0.0;

    for (int r = 0; r < BENCHMARK_REPEAT; ++r) {
        double elapsed = func(n);
        if (elapsed < 0.0) {
            return -1.0;
        }
        total += elapsed;
    }
    return total / (double)BENCHMARK_REPEAT;
}

int main(void)
{
    int sizes[] = {BENCHMARK_SIZE_1, BENCHMARK_SIZE_2, BENCHMARK_SIZE_3};
    int count = (int)(sizeof(sizes) / sizeof(sizes[0]));

    printf("Matrix benchmark, repeat = %d, block size = %d\n",
           BENCHMARK_REPEAT, BLOCK_SIZE);
    printf("All times are average CPU seconds measured by clock().\n\n");

    printf("Basic operations\n");
    printf("%-12s %-14s %-14s %-14s\n", "size", "add", "scale", "transpose");
    for (int i = 0; i < count; ++i) {
        int n = sizes[i];
        printf("%-12d %-14.6f %-14.6f %-14.6f\n",
               n,
               AverageTime(TimeAdd, n),
               AverageTime(TimeScale, n),
               AverageTime(TimeTranspose, n));
    }

    printf("\nMatrix multiplication kernels\n");
    printf("%-12s %-14s %-14s %-14s %-14s %-14s %-18s\n",
           "size", "naive", "ikj", "blocked", "transposeB", "kahan", "tensor_blocked");
    for (int i = 0; i < count; ++i) {
        int n = sizes[i];
        printf("%-12d %-14.6f %-14.6f %-14.6f %-14.6f %-14.6f %-18.6f\n",
               n,
               AverageTime(TimeMultiplyNaive, n),
               AverageTime(TimeMultiplyIKJ, n),
               AverageTime(TimeMultiplyBlocked, n),
               AverageTime(TimeMultiplyTransposeB, n),
               AverageTime(TimeMultiplyKahan, n),
               AverageTime(TimeTensorMatmulBlocked, n));
    }

    return 0;
}
