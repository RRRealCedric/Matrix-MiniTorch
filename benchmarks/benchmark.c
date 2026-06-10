#include "backend.h"
#include "matrix.h"
#include "matrix_optimized.h"
#include "tensor.h"
#include "tensor_ops.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

#define MAX_SIZES 16
#define MAX_BLOCK_SIZES 16

typedef enum {
    KERNEL_NAIVE = 0,
    KERNEL_DEFAULT = 1,
    KERNEL_IKJ = 2,
    KERNEL_BLOCKED = 3,
    KERNEL_TRANSPOSE_B = 4,
    KERNEL_KAHAN = 5,
    KERNEL_AUTO = 6
} BenchmarkKernel;

typedef struct {
    int sizes[MAX_SIZES];
    int size_count;
    int block_sizes[MAX_BLOCK_SIZES];
    int block_size_count;
    int repeat;
    const char *csv_path;
} BenchmarkConfig;

typedef struct {
    double seconds;
    double speedup_vs_naive;
    double max_abs_error_vs_kahan;
    double rel_fro_error_vs_kahan;
} BenchmarkStats;

typedef struct {
    int block_size;
    BenchmarkStats stats;
} BlockedWinner;

static double NowSeconds(void)
{
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

static void InitBenchmarkConfig(BenchmarkConfig *config)
{
    config->sizes[0] = BENCHMARK_SIZE_1;
    config->sizes[1] = BENCHMARK_SIZE_2;
    config->sizes[2] = BENCHMARK_SIZE_3;
    config->size_count = 3;
    config->block_sizes[0] = BLOCK_SIZE;
    config->block_size_count = 1;
    config->repeat = BENCHMARK_REPEAT;
    config->csv_path = NULL;
}

static int ParseIntegerList(const char *text, int *values, int max_values, int *count_out)
{
    char *buffer;
    char *token;
    char *end_ptr;
    long value;
    int count = 0;

    buffer = (char *)malloc(strlen(text) + 1U);
    if (buffer == NULL) {
        return 0;
    }
    strcpy(buffer, text);

    token = strtok(buffer, ",");
    while (token != NULL) {
        if (count >= max_values) {
            free(buffer);
            return 0;
        }

        value = strtol(token, &end_ptr, 10);
        if (*token == '\0' || *end_ptr != '\0' || value <= 0) {
            free(buffer);
            return 0;
        }

        values[count++] = (int)value;
        token = strtok(NULL, ",");
    }

    free(buffer);
    if (count == 0) {
        return 0;
    }
    *count_out = count;
    return 1;
}

static int ParseArgs(int argc, char **argv, BenchmarkConfig *config)
{
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--sizes") == 0 && i + 1 < argc) {
            if (!ParseIntegerList(argv[++i], config->sizes, MAX_SIZES, &config->size_count)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) {
            int repeat = atoi(argv[++i]);
            if (repeat <= 0) {
                return 0;
            }
            config->repeat = repeat;
        } else if (strcmp(argv[i], "--block-sizes") == 0 && i + 1 < argc) {
            if (!ParseIntegerList(argv[++i], config->block_sizes, MAX_BLOCK_SIZES,
                                  &config->block_size_count)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            config->csv_path = argv[++i];
        } else {
            return 0;
        }
    }
    return 1;
}

static int CreateBenchmarkMatrices(int n, Matrix *A, Matrix *B, Matrix *C, Matrix *Ref)
{
    MatrixInit(A);
    MatrixInit(B);
    MatrixInit(C);
    MatrixInit(Ref);

    if (MatrixCreate(A, n, n) != MATRIX_SUCCESS ||
        MatrixCreate(B, n, n) != MATRIX_SUCCESS ||
        MatrixCreate(C, n, n) != MATRIX_SUCCESS ||
        MatrixCreate(Ref, n, n) != MATRIX_SUCCESS) {
        MatrixFree(A);
        MatrixFree(B);
        MatrixFree(C);
        MatrixFree(Ref);
        return 0;
    }

    MatrixFillSequence(A, 1.0, 0.001);
    MatrixFillSequence(B, 2.0, 0.001);
    MatrixFillZero(C);
    MatrixFillZero(Ref);
    return 1;
}

static double ComputeMaxAbsError(const Matrix *A, const Matrix *B)
{
    double max_abs_error = 0.0;
    size_t total = (size_t)A->row * (size_t)A->column;

    for (size_t i = 0; i < total; ++i) {
        double diff = fabs((double)A->data[i] - (double)B->data[i]);
        if (diff > max_abs_error) {
            max_abs_error = diff;
        }
    }
    return max_abs_error;
}

static double ComputeRelativeFroError(const Matrix *A, const Matrix *B)
{
    double diff_sum = 0.0;
    double ref_sum = 0.0;
    size_t total = (size_t)A->row * (size_t)A->column;

    for (size_t i = 0; i < total; ++i) {
        double diff = (double)A->data[i] - (double)B->data[i];
        double ref = (double)B->data[i];
        diff_sum += diff * diff;
        ref_sum += ref * ref;
    }

    if (ref_sum <= 0.0) {
        return 0.0;
    }
    return sqrt(diff_sum) / sqrt(ref_sum);
}

static MatrixError RunKernel(BenchmarkKernel kernel,
                             const Matrix *A,
                             const Matrix *B,
                             Matrix *C,
                             int block_size)
{
    switch (kernel) {
        case KERNEL_NAIVE:
            return MatrixMultiplyNaive(A, B, C);
        case KERNEL_DEFAULT:
            return MatrixMultiply(A, B, C);
        case KERNEL_IKJ:
            return MatrixMultiplyIKJ(A, B, C);
        case KERNEL_BLOCKED:
            return MatrixMultiplyBlocked(A, B, C, block_size);
        case KERNEL_TRANSPOSE_B:
            return MatrixMultiplyTransposeB(A, B, C);
        case KERNEL_KAHAN:
            return MatrixMultiplyKahan(A, B, C);
        case KERNEL_AUTO:
            BackendSetType(BACKEND_SIMD);
            return MatrixMultiplyAuto(A, B, C);
        default:
            return MATRIX_ERROR_INVALID_ARGUMENT;
    }
}

static const char *KernelName(BenchmarkKernel kernel)
{
    switch (kernel) {
        case KERNEL_NAIVE:
            return "naive";
        case KERNEL_DEFAULT:
            return "default";
        case KERNEL_IKJ:
            return "ikj";
        case KERNEL_BLOCKED:
            return "blocked";
        case KERNEL_TRANSPOSE_B:
            return "transposeB";
        case KERNEL_KAHAN:
            return "kahan";
        case KERNEL_AUTO:
            return "auto";
        default:
            return "unknown";
    }
}

static int MeasureKernel(int n,
                         BenchmarkKernel kernel,
                         int block_size,
                         int repeat,
                         const Matrix *reference,
                         double naive_seconds,
                         BenchmarkStats *stats)
{
    Matrix A, B, C, Ref;
    double total_seconds = 0.0;
    double max_abs_error = 0.0;
    double rel_fro_error = 0.0;

    if (!CreateBenchmarkMatrices(n, &A, &B, &C, &Ref)) {
        return 0;
    }

    (void)reference;
    for (int r = 0; r < repeat; ++r) {
        double start;
        MatrixFillZero(&C);
        start = NowSeconds();
        if (RunKernel(kernel, &A, &B, &C, block_size) != MATRIX_SUCCESS) {
            MatrixFree(&A);
            MatrixFree(&B);
            MatrixFree(&C);
            MatrixFree(&Ref);
            return 0;
        }
        total_seconds += NowSeconds() - start;
    }

    if (RunKernel(kernel, &A, &B, &C, block_size) != MATRIX_SUCCESS) {
        MatrixFree(&A);
        MatrixFree(&B);
        MatrixFree(&C);
        MatrixFree(&Ref);
        return 0;
    }

    max_abs_error = ComputeMaxAbsError(&C, reference);
    rel_fro_error = ComputeRelativeFroError(&C, reference);

    stats->seconds = total_seconds / (double)repeat;
    stats->speedup_vs_naive = (naive_seconds > 0.0) ? (naive_seconds / stats->seconds) : 0.0;
    stats->max_abs_error_vs_kahan = max_abs_error;
    stats->rel_fro_error_vs_kahan = rel_fro_error;

    MatrixFree(&A);
    MatrixFree(&B);
    MatrixFree(&C);
    MatrixFree(&Ref);
    return 1;
}

static int BuildReferenceResult(int n, Matrix *reference)
{
    Matrix A, B, Scratch;
    int success = 0;

    MatrixInit(&A);
    MatrixInit(&B);
    MatrixInit(&Scratch);
    MatrixInit(reference);

    if (MatrixCreate(&A, n, n) != MATRIX_SUCCESS ||
        MatrixCreate(&B, n, n) != MATRIX_SUCCESS ||
        MatrixCreate(reference, n, n) != MATRIX_SUCCESS ||
        MatrixCreate(&Scratch, n, n) != MATRIX_SUCCESS) {
        goto cleanup;
    }

    MatrixFillSequence(&A, 1.0, 0.001);
    MatrixFillSequence(&B, 2.0, 0.001);
    if (MatrixMultiplyKahan(&A, &B, reference) != MATRIX_SUCCESS) {
        goto cleanup;
    }

    success = 1;

cleanup:
    MatrixFree(&A);
    MatrixFree(&B);
    MatrixFree(&Scratch);
    if (!success) {
        MatrixFree(reference);
    }
    return success;
}

static void PrintCsvHeader(FILE *csv)
{
    fprintf(csv,
            "kernel,size,repeat,block_size,seconds,speedup_vs_naive,max_abs_error_vs_kahan,rel_fro_error_vs_kahan\n");
}

static void WriteCsvRow(FILE *csv,
                        const char *kernel,
                        int size,
                        int repeat,
                        int block_size,
                        const BenchmarkStats *stats)
{
    fprintf(csv, "%s,%d,%d,%d,%.9f,%.9f,%.12g,%.12g\n",
            kernel, size, repeat, block_size, stats->seconds,
            stats->speedup_vs_naive, stats->max_abs_error_vs_kahan,
            stats->rel_fro_error_vs_kahan);
}

static void PrintSummaryRow(const char *kernel,
                            int size,
                            int block_size,
                            const BenchmarkStats *stats)
{
    printf("%-12s %-8d %-10d %-12.6f %-12.3f %-16.6g %-16.6g\n",
           kernel, size, block_size, stats->seconds, stats->speedup_vs_naive,
           stats->max_abs_error_vs_kahan, stats->rel_fro_error_vs_kahan);
}

static void PrintAutoSummary(int size, const BlockedWinner *winner)
{
    const char *expected_backend = "ikj";

    if (size >= 640) {
        expected_backend = "blocked";
    } else if (size >= 192) {
        expected_backend = "transposeB";
    }

    printf("auto summary : size=%d expected_backend=%s tuned_block=%d tuned_seconds=%.6f\n",
           size, expected_backend, winner->block_size, winner->stats.seconds);
}

static void RunProfileDemo(int n)
{
    Matrix A, B, C, AT;

    if (!CreateBenchmarkMatrices(n, &A, &B, &C, &AT)) {
        printf("\nProfiling demo skipped: cannot allocate matrices.\n");
        return;
    }

    MatrixProfileEnable(1);
    MatrixProfileClear();
    BackendSetType(BACKEND_BLOCKED);
    MatrixMultiplyBlocked(&A, &B, &C, BLOCK_SIZE);
    BackendSetType(BACKEND_SIMD);
    MatrixMultiplyAuto(&A, &B, &C);
    MatrixTransposeBlocked(&A, &AT, BLOCK_SIZE);
    MatrixAxpy(2.0, &A, &B);
    BackendSetType(BACKEND_NAIVE);

    printf("\nProfiling records, demo size = %d\n", n);
    MatrixProfilePrint(stdout);
    MatrixProfileEnable(0);
    MatrixProfileClear();

    MatrixFree(&A);
    MatrixFree(&B);
    MatrixFree(&C);
    MatrixFree(&AT);
}

static int RunBenchmarkSuite(const BenchmarkConfig *config)
{
    FILE *csv = NULL;

    if (config->csv_path != NULL) {
        csv = fopen(config->csv_path, "w");
        if (csv == NULL) {
            printf("Cannot open CSV output: %s\n", config->csv_path);
            return 0;
        }
        PrintCsvHeader(csv);
    }

    printf("Matrix benchmark, repeat = %d\n", config->repeat);
    printf("%-12s %-8s %-10s %-12s %-12s %-16s %-16s\n",
           "kernel", "size", "block", "seconds", "speedup", "max_abs_error", "rel_fro_error");

    for (int i = 0; i < config->size_count; ++i) {
        int n = config->sizes[i];
        Matrix reference;
        BenchmarkStats naive_stats;
        BlockedWinner blocked_winner;

        blocked_winner.block_size = config->block_sizes[0];
        blocked_winner.stats.seconds = 0.0;
        blocked_winner.stats.speedup_vs_naive = 0.0;
        blocked_winner.stats.max_abs_error_vs_kahan = 0.0;
        blocked_winner.stats.rel_fro_error_vs_kahan = 0.0;

        if (!BuildReferenceResult(n, &reference)) {
            if (csv != NULL) {
                fclose(csv);
            }
            return 0;
        }

        if (!MeasureKernel(n, KERNEL_NAIVE, 0, config->repeat, &reference, 0.0, &naive_stats)) {
            MatrixFree(&reference);
            if (csv != NULL) {
                fclose(csv);
            }
            return 0;
        }
        naive_stats.speedup_vs_naive = 1.0;
        PrintSummaryRow("naive", n, 0, &naive_stats);
        if (csv != NULL) {
            WriteCsvRow(csv, "naive", n, config->repeat, 0, &naive_stats);
        }

        for (int kernel = KERNEL_DEFAULT; kernel <= KERNEL_AUTO; ++kernel) {
            if (kernel == KERNEL_BLOCKED) {
                for (int b = 0; b < config->block_size_count; ++b) {
                    BenchmarkStats stats;
                    int block_size = config->block_sizes[b];
                    if (!MeasureKernel(n, (BenchmarkKernel)kernel, block_size, config->repeat,
                                       &reference, naive_stats.seconds, &stats)) {
                        MatrixFree(&reference);
                        if (csv != NULL) {
                            fclose(csv);
                        }
                        return 0;
                    }
                    PrintSummaryRow(KernelName((BenchmarkKernel)kernel), n, block_size, &stats);
                    if (csv != NULL) {
                        WriteCsvRow(csv, KernelName((BenchmarkKernel)kernel), n,
                                    config->repeat, block_size, &stats);
                    }
                    if (b == 0 || stats.seconds < blocked_winner.stats.seconds) {
                        blocked_winner.block_size = block_size;
                        blocked_winner.stats = stats;
                    }
                }
            } else {
                BenchmarkStats stats;
                int block_size = (kernel == KERNEL_AUTO) ? BLOCK_SIZE : 0;
                if (!MeasureKernel(n, (BenchmarkKernel)kernel, block_size, config->repeat,
                                   &reference, naive_stats.seconds, &stats)) {
                    MatrixFree(&reference);
                    if (csv != NULL) {
                        fclose(csv);
                    }
                    return 0;
                }
                PrintSummaryRow(KernelName((BenchmarkKernel)kernel), n, block_size, &stats);
                if (csv != NULL) {
                    WriteCsvRow(csv, KernelName((BenchmarkKernel)kernel), n,
                                config->repeat, block_size, &stats);
                }
            }
        }

        PrintAutoSummary(n, &blocked_winner);
        MatrixFree(&reference);
    }

    if (csv != NULL) {
        fclose(csv);
    }
    RunProfileDemo(config->sizes[0]);
    return 1;
}

int main(int argc, char **argv)
{
    BenchmarkConfig config;

    InitBenchmarkConfig(&config);
    if (!ParseArgs(argc, argv, &config)) {
        printf("Usage: %s [--sizes a,b,c] [--repeat n] [--block-sizes a,b,c] [--csv path]\n",
               argv[0]);
        return 1;
    }

    if (!RunBenchmarkSuite(&config)) {
        return 1;
    }
    return 0;
}
