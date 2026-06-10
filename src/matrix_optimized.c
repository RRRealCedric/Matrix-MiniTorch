#include "matrix_optimized.h"

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define MATRIX_AUTO_SMALL_THRESHOLD 192
#define MATRIX_AUTO_MEDIUM_THRESHOLD 640
#define MATRIX_AUTO_SMALL_BLOCK_SIZE 16
#define MATRIX_AUTO_MEDIUM_BLOCK_SIZE 32
#define MATRIX_AUTO_LARGE_BLOCK_SIZE 64

static int profile_enabled = 0;
static MatrixProfileRecord profile_records[MATRIX_PROFILE_MAX_RECORDS];
static size_t profile_count = 0;

static double ProfileNowSeconds(void)
{
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

static void ProfileRecord(const char *kernel_name,
                          const char *requested_backend_name,
                          const char *backend_name,
                          int m,
                          int n,
                          int k,
                          int block_size,
                          double elapsed_seconds,
                          MatrixError error)
{
    MatrixProfileRecord *record;

    if (!profile_enabled || profile_count >= MATRIX_PROFILE_MAX_RECORDS) {
        return;
    }

    record = &profile_records[profile_count++];
    record->kernel_name = kernel_name;
    record->requested_backend_name = requested_backend_name;
    record->backend_name = backend_name;
    record->m = m;
    record->n = n;
    record->k = k;
    record->block_size = block_size;
    record->elapsed_seconds = elapsed_seconds;
    record->error = error;
}

static void MultiplyDims(const Matrix *A, const Matrix *B, int *m, int *n, int *k)
{
    *m = MatrixIsValid(A) ? A->row : 0;
    *n = MatrixIsValid(B) ? B->column : 0;
    *k = (MatrixIsValid(A) && MatrixIsValid(B) && A->column == B->row) ? A->column : 0;
}

static void UnaryDims(const Matrix *A, int *m, int *n)
{
    *m = MatrixIsValid(A) ? A->row : 0;
    *n = MatrixIsValid(A) ? A->column : 0;
}

void MatrixProfileEnable(int enabled)
{
    profile_enabled = enabled != 0;
}

int MatrixProfileIsEnabled(void)
{
    return profile_enabled;
}

void MatrixProfileClear(void)
{
    profile_count = 0;
}

const MatrixProfileRecord *MatrixProfileGetRecords(size_t *count)
{
    if (count != NULL) {
        *count = profile_count;
    }
    return profile_records;
}

void MatrixProfilePrint(FILE *out)
{
    if (out == NULL) {
        out = stdout;
    }

    fprintf(out, "%-20s %-14s %-14s %-8s %-8s %-8s %-10s %-14s %-24s\n",
            "kernel", "requested", "backend", "m", "n", "k", "block", "seconds", "status");
    for (size_t i = 0; i < profile_count; ++i) {
        const MatrixProfileRecord *record = &profile_records[i];
        fprintf(out, "%-20s %-14s %-14s %-8d %-8d %-8d %-10d %-14.6f %-24s\n",
                record->kernel_name,
                record->requested_backend_name,
                record->backend_name,
                record->m,
                record->n,
                record->k,
                record->block_size,
                record->elapsed_seconds,
                MatrixErrorMessage(record->error));
    }
}

static MatrixError CheckMultiplyShape(const Matrix *A, const Matrix *B, const Matrix *C)
{
    if (!MatrixIsValid(A) || !MatrixIsValid(B) || !MatrixIsValid(C)) {
        return MATRIX_ERROR_NULL_POINTER;
    }
    if (A->column != B->row || C->row != A->row || C->column != B->column) {
        return MATRIX_ERROR_SIZE_MISMATCH;
    }
    if (C->data == A->data || C->data == B->data) {
        return MATRIX_ERROR_ALIASING;
    }
    return MATRIX_SUCCESS;
}

static MatrixError CheckSameShapeNoAlias(const Matrix *A, const Matrix *B)
{
    if (!MatrixIsValid(A) || !MatrixIsValid(B)) {
        return MATRIX_ERROR_NULL_POINTER;
    }
    if (A->row != B->row || A->column != B->column) {
        return MATRIX_ERROR_SIZE_MISMATCH;
    }
    if (A->data == B->data) {
        return MATRIX_ERROR_ALIASING;
    }
    return MATRIX_SUCCESS;
}

static int MinInt(int a, int b)
{
    return a < b ? a : b;
}

static int MatrixAutoChooseBlockSize(int m, int n, int k)
{
    if (m < MATRIX_AUTO_SMALL_THRESHOLD &&
        n < MATRIX_AUTO_SMALL_THRESHOLD &&
        k < MATRIX_AUTO_SMALL_THRESHOLD) {
        return MATRIX_AUTO_SMALL_BLOCK_SIZE;
    }
    if (m < MATRIX_AUTO_MEDIUM_THRESHOLD &&
        n < MATRIX_AUTO_MEDIUM_THRESHOLD &&
        k < MATRIX_AUTO_MEDIUM_THRESHOLD) {
        return MATRIX_AUTO_MEDIUM_BLOCK_SIZE;
    }
    return MATRIX_AUTO_LARGE_BLOCK_SIZE;
}

static const char *MatrixAutoChooseBackendName(int m, int n, int k)
{
    if (m < MATRIX_AUTO_SMALL_THRESHOLD &&
        n < MATRIX_AUTO_SMALL_THRESHOLD &&
        k < MATRIX_AUTO_SMALL_THRESHOLD) {
        return "ikj";
    }
    if (m < MATRIX_AUTO_MEDIUM_THRESHOLD &&
        n < MATRIX_AUTO_MEDIUM_THRESHOLD &&
        k < MATRIX_AUTO_MEDIUM_THRESHOLD) {
        return "transposeB";
    }
    return "blocked";
}

MatrixError MatrixMultiplyIKJ(const Matrix *A, const Matrix *B, Matrix *C)
{
    int m;
    int n;
    int kdim;
    double start = ProfileNowSeconds();
    MatrixError error = CheckMultiplyShape(A, B, C);

    MultiplyDims(A, B, &m, &n, &kdim);
    if (error != MATRIX_SUCCESS) {
        ProfileRecord("matmul_ikj", BackendName(BACKEND_IKJ), BackendName(BACKEND_IKJ),
                      m, n, kdim, 0,
                      ProfileNowSeconds() - start, error);
        return error;
    }

    MatrixFillZero(C);
    for (int i = 0; i < A->row; ++i) {
        size_t a_row = (size_t)i * (size_t)A->column;
        size_t c_row = (size_t)i * (size_t)C->column;
        for (int k = 0; k < A->column; ++k) {
            REAL aik = A->data[a_row + (size_t)k];
            size_t b_row = (size_t)k * (size_t)B->column;
            int j = 0;
            for (; j + 3 < B->column; j += 4) {
                C->data[c_row + (size_t)j] += aik * B->data[b_row + (size_t)j];
                C->data[c_row + (size_t)(j + 1)] += aik * B->data[b_row + (size_t)(j + 1)];
                C->data[c_row + (size_t)(j + 2)] += aik * B->data[b_row + (size_t)(j + 2)];
                C->data[c_row + (size_t)(j + 3)] += aik * B->data[b_row + (size_t)(j + 3)];
            }
            for (; j < B->column; ++j) {
                C->data[c_row + (size_t)j] += aik * B->data[b_row + (size_t)j];
            }
        }
    }

    ProfileRecord("matmul_ikj", BackendName(BACKEND_IKJ), BackendName(BACKEND_IKJ),
                  m, n, kdim, 0,
                  ProfileNowSeconds() - start, MATRIX_SUCCESS);
    return MATRIX_SUCCESS;
}

MatrixError MatrixMultiplyBlocked(const Matrix *A, const Matrix *B, Matrix *C, int block_size)
{
    int m;
    int n;
    int kdim;
    double start = ProfileNowSeconds();
    MatrixError error = CheckMultiplyShape(A, B, C);

    MultiplyDims(A, B, &m, &n, &kdim);
    if (error != MATRIX_SUCCESS) {
        ProfileRecord("matmul_blocked", BackendName(BACKEND_BLOCKED), BackendName(BACKEND_BLOCKED),
                      m, n, kdim, block_size,
                      ProfileNowSeconds() - start, error);
        return error;
    }
    if (block_size <= 0) {
        ProfileRecord("matmul_blocked", BackendName(BACKEND_BLOCKED), BackendName(BACKEND_BLOCKED),
                      m, n, kdim, block_size,
                      ProfileNowSeconds() - start, MATRIX_ERROR_INVALID_ARGUMENT);
        return MATRIX_ERROR_INVALID_ARGUMENT;
    }

    MatrixFillZero(C);
    for (int ii = 0; ii < A->row; ii += block_size) {
        for (int kk = 0; kk < A->column; kk += block_size) {
            for (int jj = 0; jj < B->column; jj += block_size) {
                int i_end = MinInt(ii + block_size, A->row);
                int k_end = MinInt(kk + block_size, A->column);
                int j_end = MinInt(jj + block_size, B->column);

                for (int i = ii; i < i_end; ++i) {
                    size_t a_row = (size_t)i * (size_t)A->column;
                    size_t c_row = (size_t)i * (size_t)C->column;
                    for (int k = kk; k < k_end; ++k) {
                        REAL aik = A->data[a_row + (size_t)k];
                        size_t b_row = (size_t)k * (size_t)B->column;
                        int j = jj;
                        for (; j + 3 < j_end; j += 4) {
                            C->data[c_row + (size_t)j] += aik * B->data[b_row + (size_t)j];
                            C->data[c_row + (size_t)(j + 1)] += aik * B->data[b_row + (size_t)(j + 1)];
                            C->data[c_row + (size_t)(j + 2)] += aik * B->data[b_row + (size_t)(j + 2)];
                            C->data[c_row + (size_t)(j + 3)] += aik * B->data[b_row + (size_t)(j + 3)];
                        }
                        for (; j < j_end; ++j) {
                            C->data[c_row + (size_t)j] += aik * B->data[b_row + (size_t)j];
                        }
                    }
                }
            }
        }
    }

    ProfileRecord("matmul_blocked", BackendName(BACKEND_BLOCKED), BackendName(BACKEND_BLOCKED),
                  m, n, kdim, block_size,
                  ProfileNowSeconds() - start, MATRIX_SUCCESS);
    return MATRIX_SUCCESS;
}

MatrixError MatrixMultiplyTransposeB(const Matrix *A, const Matrix *B, Matrix *C)
{
    Matrix BT;
    int m;
    int n;
    int kdim;
    double start = ProfileNowSeconds();
    MatrixError error = CheckMultiplyShape(A, B, C);

    MultiplyDims(A, B, &m, &n, &kdim);
    if (error != MATRIX_SUCCESS) {
        ProfileRecord("matmul_transposeB", "transposeB", "transposeB", m, n, kdim, 0,
                      ProfileNowSeconds() - start, error);
        return error;
    }

    MatrixInit(&BT);
    error = MatrixCreate(&BT, B->column, B->row);
    if (error != MATRIX_SUCCESS) {
        ProfileRecord("matmul_transposeB", "transposeB", "transposeB", m, n, kdim, 0,
                      ProfileNowSeconds() - start, error);
        return error;
    }

    error = MatrixTranspose(B, &BT);
    if (error != MATRIX_SUCCESS) {
        MatrixFree(&BT);
        ProfileRecord("matmul_transposeB", "transposeB", "transposeB", m, n, kdim, 0,
                      ProfileNowSeconds() - start, error);
        return error;
    }

    for (int i = 0; i < A->row; ++i) {
        size_t a_row = (size_t)i * (size_t)A->column;
        size_t c_row = (size_t)i * (size_t)C->column;
        for (int j = 0; j < B->column; ++j) {
            const REAL *restrict a = A->data + a_row;
            const REAL *restrict bt = BT.data + (size_t)j * (size_t)BT.column;
            REAL sum = 0.0;
            for (int k = 0; k < A->column; ++k) {
                sum += a[k] * bt[k];
            }
            C->data[c_row + (size_t)j] = sum;
        }
    }

    MatrixFree(&BT);
    ProfileRecord("matmul_transposeB", "transposeB", "transposeB", m, n, kdim, 0,
                  ProfileNowSeconds() - start, MATRIX_SUCCESS);
    return MATRIX_SUCCESS;
}

MatrixError MatrixMultiplyKahan(const Matrix *A, const Matrix *B, Matrix *C)
{
    int m;
    int n;
    int kdim;
    double start = ProfileNowSeconds();
    MatrixError error = CheckMultiplyShape(A, B, C);

    MultiplyDims(A, B, &m, &n, &kdim);
    if (error != MATRIX_SUCCESS) {
        ProfileRecord("matmul_kahan", "kahan", "kahan", m, n, kdim, 0,
                      ProfileNowSeconds() - start, error);
        return error;
    }

    for (int i = 0; i < A->row; ++i) {
        size_t a_row = (size_t)i * (size_t)A->column;
        size_t c_row = (size_t)i * (size_t)C->column;
        for (int j = 0; j < B->column; ++j) {
            REAL sum = 0.0;
            REAL compensation = 0.0;
            for (int k = 0; k < A->column; ++k) {
                REAL product = A->data[a_row + (size_t)k] *
                               B->data[(size_t)k * (size_t)B->column + (size_t)j];
                REAL y = product - compensation;
                REAL t = sum + y;
                compensation = (t - sum) - y;
                sum = t;
            }
            C->data[c_row + (size_t)j] = sum;
        }
    }

    ProfileRecord("matmul_kahan", "kahan", "kahan", m, n, kdim, 0,
                  ProfileNowSeconds() - start, MATRIX_SUCCESS);
    return MATRIX_SUCCESS;
}

MatrixError MatrixMultiplyAuto(const Matrix *A, const Matrix *B, Matrix *C)
{
    int m;
    int n;
    int kdim;
    double start = ProfileNowSeconds();
    BackendType requested = BackendGetType();
    BackendType actual = BackendFallback(requested);
    MatrixError error;
    int block_size;
    const char *actual_backend_name;
    const char *requested_backend_name = BackendName(requested);

    MultiplyDims(A, B, &m, &n, &kdim);
    block_size = MatrixAutoChooseBlockSize(m, n, kdim);
    if (requested == BACKEND_NAIVE) {
        error = MatrixMultiplyNaive(A, B, C);
        actual = BACKEND_NAIVE;
        block_size = 0;
    } else if (requested == BACKEND_IKJ) {
        error = MatrixMultiplyIKJ(A, B, C);
        actual = BACKEND_IKJ;
        block_size = 0;
    } else if (requested == BACKEND_BLOCKED) {
        error = MatrixMultiplyBlocked(A, B, C, block_size);
        actual = BACKEND_BLOCKED;
    } else if (requested == BACKEND_SIMD || requested == BACKEND_THREADS || requested == BACKEND_BLAS) {
        actual_backend_name = MatrixAutoChooseBackendName(m, n, kdim);
        if (strcmp(actual_backend_name, "ikj") == 0) {
            error = MatrixMultiplyIKJ(A, B, C);
            actual = BACKEND_IKJ;
            block_size = 0;
        } else if (strcmp(actual_backend_name, "transposeB") == 0) {
            error = MatrixMultiplyTransposeB(A, B, C);
            ProfileRecord("matmul_auto", requested_backend_name, actual_backend_name,
                          m, n, kdim, 0, ProfileNowSeconds() - start, error);
            return error;
        } else {
            error = MatrixMultiplyBlocked(A, B, C, block_size);
            actual = BACKEND_BLOCKED;
        }
    } else {
        actual_backend_name = MatrixAutoChooseBackendName(m, n, kdim);
        if (strcmp(actual_backend_name, "ikj") == 0) {
            error = MatrixMultiplyIKJ(A, B, C);
            actual = BACKEND_IKJ;
            block_size = 0;
        } else if (strcmp(actual_backend_name, "transposeB") == 0) {
            error = MatrixMultiplyTransposeB(A, B, C);
            ProfileRecord("matmul_auto", requested_backend_name, actual_backend_name,
                          m, n, kdim, 0, ProfileNowSeconds() - start, error);
            return error;
        } else {
            error = MatrixMultiplyBlocked(A, B, C, block_size);
            actual = BACKEND_BLOCKED;
        }
    }

    ProfileRecord("matmul_auto", requested_backend_name, BackendName(actual), m, n, kdim,
                  block_size,
                  ProfileNowSeconds() - start, error);
    return error;
}

MatrixError MatrixTransposeBlocked(const Matrix *A, Matrix *AT, int block_size)
{
    int m;
    int n;
    double start = ProfileNowSeconds();

    UnaryDims(A, &m, &n);
    if (!MatrixIsValid(A) || !MatrixIsValid(AT)) {
        ProfileRecord("transpose_blocked", "transpose_blocked", "transpose_blocked", m, n, 0,
                      block_size,
                      ProfileNowSeconds() - start, MATRIX_ERROR_NULL_POINTER);
        return MATRIX_ERROR_NULL_POINTER;
    }
    if (AT->row != A->column || AT->column != A->row) {
        ProfileRecord("transpose_blocked", "transpose_blocked", "transpose_blocked", m, n, 0,
                      block_size,
                      ProfileNowSeconds() - start, MATRIX_ERROR_SIZE_MISMATCH);
        return MATRIX_ERROR_SIZE_MISMATCH;
    }
    if (AT->data == A->data) {
        ProfileRecord("transpose_blocked", "transpose_blocked", "transpose_blocked", m, n, 0,
                      block_size,
                      ProfileNowSeconds() - start, MATRIX_ERROR_ALIASING);
        return MATRIX_ERROR_ALIASING;
    }
    if (block_size <= 0) {
        ProfileRecord("transpose_blocked", "transpose_blocked", "transpose_blocked", m, n, 0,
                      block_size,
                      ProfileNowSeconds() - start, MATRIX_ERROR_INVALID_ARGUMENT);
        return MATRIX_ERROR_INVALID_ARGUMENT;
    }

    for (int ii = 0; ii < A->row; ii += block_size) {
        for (int jj = 0; jj < A->column; jj += block_size) {
            int i_end = MinInt(ii + block_size, A->row);
            int j_end = MinInt(jj + block_size, A->column);
            for (int i = ii; i < i_end; ++i) {
                size_t a_row = (size_t)i * (size_t)A->column;
                for (int j = jj; j < j_end; ++j) {
                    AT->data[(size_t)j * (size_t)AT->column + (size_t)i] =
                        A->data[a_row + (size_t)j];
                }
            }
        }
    }

    ProfileRecord("transpose_blocked", "transpose_blocked", "transpose_blocked", m, n, 0,
                  block_size,
                  ProfileNowSeconds() - start, MATRIX_SUCCESS);
    return MATRIX_SUCCESS;
}

MatrixError MatrixAddInPlace(Matrix *A, const Matrix *B)
{
    int m;
    int n;
    double start = ProfileNowSeconds();
    MatrixError error = CheckSameShapeNoAlias(A, B);

    UnaryDims(A, &m, &n);
    if (error != MATRIX_SUCCESS) {
        ProfileRecord("add_inplace", "add_inplace", "add_inplace", m, n, 0, 0,
                      ProfileNowSeconds() - start, error);
        return error;
    }

    size_t total = (size_t)A->row * (size_t)A->column;
    REAL *restrict a = A->data;
    const REAL *restrict b = B->data;
    for (size_t i = 0; i < total; ++i) {
        a[i] += b[i];
    }

    ProfileRecord("add_inplace", "add_inplace", "add_inplace", m, n, 0, 0,
                  ProfileNowSeconds() - start, MATRIX_SUCCESS);
    return MATRIX_SUCCESS;
}

MatrixError MatrixScaleInPlace(Matrix *A, REAL alpha)
{
    int m;
    int n;
    double start = ProfileNowSeconds();

    UnaryDims(A, &m, &n);
    if (!MatrixIsValid(A)) {
        ProfileRecord("scale_inplace", "scale_inplace", "scale_inplace", m, n, 0, 0,
                      ProfileNowSeconds() - start, MATRIX_ERROR_NULL_POINTER);
        return MATRIX_ERROR_NULL_POINTER;
    }

    size_t total = (size_t)A->row * (size_t)A->column;
    REAL *restrict a = A->data;
    for (size_t i = 0; i < total; ++i) {
        a[i] *= alpha;
    }

    ProfileRecord("scale_inplace", "scale_inplace", "scale_inplace", m, n, 0, 0,
                  ProfileNowSeconds() - start, MATRIX_SUCCESS);
    return MATRIX_SUCCESS;
}

MatrixError MatrixAxpy(REAL alpha, const Matrix *X, Matrix *Y)
{
    int m;
    int n;
    double start = ProfileNowSeconds();
    MatrixError error = CheckSameShapeNoAlias(X, Y);

    UnaryDims(X, &m, &n);
    if (error != MATRIX_SUCCESS) {
        ProfileRecord("axpy", "axpy", "axpy", m, n, 0, 0,
                      ProfileNowSeconds() - start, error);
        return error;
    }

    size_t total = (size_t)X->row * (size_t)X->column;
    const REAL *restrict x = X->data;
    REAL *restrict y = Y->data;
    for (size_t i = 0; i < total; ++i) {
        y[i] += alpha * x[i];
    }

    ProfileRecord("axpy", "axpy", "axpy", m, n, 0, 0,
                  ProfileNowSeconds() - start, MATRIX_SUCCESS);
    return MATRIX_SUCCESS;
}
