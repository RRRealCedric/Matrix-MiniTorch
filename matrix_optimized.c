#include "matrix_optimized.h"

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

static int MinInt(int a, int b)
{
    return a < b ? a : b;
}

MatrixError MatrixMultiplyIKJ(const Matrix *A, const Matrix *B, Matrix *C)
{
    MatrixError error = CheckMultiplyShape(A, B, C);

    if (error != MATRIX_SUCCESS) {
        return error;
    }

    MatrixFillZero(C);
    for (int i = 0; i < A->row; ++i) {
        size_t a_row = (size_t)i * (size_t)A->column;
        size_t c_row = (size_t)i * (size_t)C->column;
        for (int k = 0; k < A->column; ++k) {
            REAL aik = A->data[a_row + (size_t)k];
            size_t b_row = (size_t)k * (size_t)B->column;
            for (int j = 0; j < B->column; ++j) {
                C->data[c_row + (size_t)j] += aik * B->data[b_row + (size_t)j];
            }
        }
    }
    return MATRIX_SUCCESS;
}

MatrixError MatrixMultiplyBlocked(const Matrix *A, const Matrix *B, Matrix *C, int block_size)
{
    MatrixError error = CheckMultiplyShape(A, B, C);

    if (error != MATRIX_SUCCESS) {
        return error;
    }
    if (block_size <= 0) {
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
                        for (int j = jj; j < j_end; ++j) {
                            C->data[c_row + (size_t)j] += aik * B->data[b_row + (size_t)j];
                        }
                    }
                }
            }
        }
    }
    return MATRIX_SUCCESS;
}

MatrixError MatrixMultiplyTransposeB(const Matrix *A, const Matrix *B, Matrix *C)
{
    Matrix BT;
    MatrixError error = CheckMultiplyShape(A, B, C);

    if (error != MATRIX_SUCCESS) {
        return error;
    }

    MatrixInit(&BT);
    error = MatrixCreate(&BT, B->column, B->row);
    if (error != MATRIX_SUCCESS) {
        return error;
    }

    error = MatrixTranspose(B, &BT);
    if (error != MATRIX_SUCCESS) {
        MatrixFree(&BT);
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
    return MATRIX_SUCCESS;
}

MatrixError MatrixMultiplyKahan(const Matrix *A, const Matrix *B, Matrix *C)
{
    MatrixError error = CheckMultiplyShape(A, B, C);

    if (error != MATRIX_SUCCESS) {
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
    return MATRIX_SUCCESS;
}
