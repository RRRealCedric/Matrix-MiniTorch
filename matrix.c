#include "matrix.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void MatrixInit(Matrix *A)
{
    if (A == NULL) {
        return;
    }
    A->row = 0;
    A->column = 0;
    A->data = NULL;
}

int MatrixIsValid(const Matrix *A)
{
    return A != NULL && A->row > 0 && A->column > 0 && A->data != NULL;
}

int MatrixHasShape(const Matrix *A, int row, int column)
{
    return MatrixIsValid(A) && A->row == row && A->column == column;
}

int MatrixIndex(const Matrix *A, int i, int j)
{
    return i * A->column + j;
}

MatrixError MatrixCreate(Matrix *A, int row, int column)
{
    size_t nrow;
    size_t ncol;
    size_t total;

    if (A == NULL) {
        return MATRIX_ERROR_NULL_POINTER;
    }
    if (A->data != NULL) {
        return MATRIX_ERROR_ALREADY_ALLOCATED;
    }
    if (row <= 0 || column <= 0) {
        return MATRIX_ERROR_INVALID_SIZE;
    }

    nrow = (size_t)row;
    ncol = (size_t)column;
    if (nrow > SIZE_MAX / ncol || nrow * ncol > SIZE_MAX / sizeof(REAL)) {
        return MATRIX_ERROR_SIZE_OVERFLOW;
    }

    total = nrow * ncol;
    A->data = (REAL *)malloc(total * sizeof(REAL));
    if (A->data == NULL) {
        MatrixInit(A);
        return MATRIX_ERROR_ALLOC_FAILED;
    }

    A->row = row;
    A->column = column;
    return MATRIX_SUCCESS;
}

void MatrixFree(Matrix *A)
{
    if (A == NULL) {
        return;
    }
    free(A->data);
    MatrixInit(A);
}

MatrixError MatrixSet(Matrix *A, int i, int j, REAL value)
{
    if (!MatrixIsValid(A)) {
        return MATRIX_ERROR_NULL_POINTER;
    }
    if (i < 0 || i >= A->row || j < 0 || j >= A->column) {
        return MATRIX_ERROR_INDEX_OUT_OF_RANGE;
    }

    A->data[MatrixIndex(A, i, j)] = value;
    return MATRIX_SUCCESS;
}

MatrixError MatrixGet(const Matrix *A, int i, int j, REAL *value)
{
    if (!MatrixIsValid(A) || value == NULL) {
        return MATRIX_ERROR_NULL_POINTER;
    }
    if (i < 0 || i >= A->row || j < 0 || j >= A->column) {
        return MATRIX_ERROR_INDEX_OUT_OF_RANGE;
    }

    *value = A->data[MatrixIndex(A, i, j)];
    return MATRIX_SUCCESS;
}

MatrixError MatrixFillZero(Matrix *A)
{
    size_t total;

    if (!MatrixIsValid(A)) {
        return MATRIX_ERROR_NULL_POINTER;
    }

    total = (size_t)A->row * (size_t)A->column;
    for (size_t k = 0; k < total; ++k) {
        A->data[k] = 0.0;
    }
    return MATRIX_SUCCESS;
}

MatrixError MatrixFillSequence(Matrix *A, REAL start, REAL step)
{
    size_t total;

    if (!MatrixIsValid(A)) {
        return MATRIX_ERROR_NULL_POINTER;
    }

    total = (size_t)A->row * (size_t)A->column;
    for (size_t k = 0; k < total; ++k) {
        A->data[k] = start + step * (REAL)k;
    }
    return MATRIX_SUCCESS;
}

MatrixError MatrixFillIdentity(Matrix *A)
{
    if (!MatrixIsValid(A)) {
        return MATRIX_ERROR_NULL_POINTER;
    }
    if (A->row != A->column) {
        return MATRIX_ERROR_SIZE_MISMATCH;
    }

    MatrixFillZero(A);
    for (int i = 0; i < A->row; ++i) {
        A->data[(size_t)i * (size_t)A->column + (size_t)i] = 1.0;
    }
    return MATRIX_SUCCESS;
}

MatrixError MatrixFillRandom(Matrix *A, unsigned int seed, REAL min_value, REAL max_value)
{
    size_t total;

    if (!MatrixIsValid(A)) {
        return MATRIX_ERROR_NULL_POINTER;
    }
    if (min_value > max_value) {
        return MATRIX_ERROR_INVALID_ARGUMENT;
    }

    srand(seed);
    total = (size_t)A->row * (size_t)A->column;
    for (size_t k = 0; k < total; ++k) {
        REAL ratio = (REAL)rand() / (REAL)RAND_MAX;
        A->data[k] = min_value + ratio * (max_value - min_value);
    }
    return MATRIX_SUCCESS;
}

MatrixError MatrixCopy(const Matrix *src, Matrix *dst)
{
    size_t total;

    if (!MatrixIsValid(src) || !MatrixIsValid(dst)) {
        return MATRIX_ERROR_NULL_POINTER;
    }
    if (src->row != dst->row || src->column != dst->column) {
        return MATRIX_ERROR_SIZE_MISMATCH;
    }

    total = (size_t)src->row * (size_t)src->column;
    for (size_t k = 0; k < total; ++k) {
        dst->data[k] = src->data[k];
    }
    return MATRIX_SUCCESS;
}

void MatrixPrint(const Matrix *A, const char *name)
{
    if (name == NULL) {
        name = "Matrix";
    }
    if (!MatrixIsValid(A)) {
        printf("%s is an empty or invalid matrix.\n", name);
        return;
    }

    printf("%s = (%d x %d)\n", name, A->row, A->column);
    for (int i = 0; i < A->row; ++i) {
        printf("  ");
        for (int j = 0; j < A->column; ++j) {
            printf("%10.4f ", A->data[(size_t)i * (size_t)A->column + (size_t)j]);
        }
        printf("\n");
    }
}

const char *MatrixErrorMessage(MatrixError error)
{
    switch (error) {
        case MATRIX_SUCCESS:
            return "success";
        case MATRIX_ERROR_NULL_POINTER:
            return "null pointer or invalid matrix";
        case MATRIX_ERROR_INVALID_SIZE:
            return "invalid matrix size";
        case MATRIX_ERROR_SIZE_OVERFLOW:
            return "matrix size overflow";
        case MATRIX_ERROR_ALLOC_FAILED:
            return "memory allocation failed";
        case MATRIX_ERROR_INDEX_OUT_OF_RANGE:
            return "matrix index out of range";
        case MATRIX_ERROR_SIZE_MISMATCH:
            return "matrix size mismatch";
        case MATRIX_ERROR_ALREADY_ALLOCATED:
            return "matrix data already allocated";
        case MATRIX_ERROR_INVALID_ARGUMENT:
            return "invalid argument";
        case MATRIX_ERROR_ALIASING:
            return "output matrix aliases input matrix";
        case MATRIX_ERROR_NOT_SQUARE:
            return "matrix must be square";
        case MATRIX_ERROR_SINGULAR:
            return "matrix is singular or nearly singular";
        default:
            return "unknown matrix error";
    }
}

static int MatrixDataAliases(const Matrix *A, const Matrix *B)
{
    return A != NULL && B != NULL && A->data != NULL && A->data == B->data;
}

static MatrixError CheckSameShape(const Matrix *A, const Matrix *B, Matrix *C)
{
    if (!MatrixIsValid(A) || !MatrixIsValid(B) || !MatrixIsValid(C)) {
        return MATRIX_ERROR_NULL_POINTER;
    }
    if (A->row != B->row || A->column != B->column) {
        return MATRIX_ERROR_SIZE_MISMATCH;
    }
    if (C->row != A->row || C->column != A->column) {
        return MATRIX_ERROR_SIZE_MISMATCH;
    }
    if (MatrixDataAliases(C, A) || MatrixDataAliases(C, B)) {
        return MATRIX_ERROR_ALIASING;
    }
    return MATRIX_SUCCESS;
}

MatrixError MatrixAdd(const Matrix *A, const Matrix *B, Matrix *C)
{
    MatrixError error = CheckSameShape(A, B, C);
    size_t total;

    if (error != MATRIX_SUCCESS) {
        return error;
    }

    total = (size_t)A->row * (size_t)A->column;
    const REAL *restrict a = A->data;
    const REAL *restrict b = B->data;
    REAL *restrict c = C->data;
    for (size_t k = 0; k < total; ++k) {
        c[k] = a[k] + b[k];
    }
    return MATRIX_SUCCESS;
}

MatrixError MatrixSub(const Matrix *A, const Matrix *B, Matrix *C)
{
    MatrixError error = CheckSameShape(A, B, C);
    size_t total;

    if (error != MATRIX_SUCCESS) {
        return error;
    }

    total = (size_t)A->row * (size_t)A->column;
    const REAL *restrict a = A->data;
    const REAL *restrict b = B->data;
    REAL *restrict c = C->data;
    for (size_t k = 0; k < total; ++k) {
        c[k] = a[k] - b[k];
    }
    return MATRIX_SUCCESS;
}

MatrixError MatrixScale(REAL alpha, const Matrix *A, Matrix *B)
{
    size_t total;

    if (!MatrixIsValid(A) || !MatrixIsValid(B)) {
        return MATRIX_ERROR_NULL_POINTER;
    }
    if (A->row != B->row || A->column != B->column) {
        return MATRIX_ERROR_SIZE_MISMATCH;
    }
    if (MatrixDataAliases(B, A)) {
        return MATRIX_ERROR_ALIASING;
    }

    total = (size_t)A->row * (size_t)A->column;
    const REAL *restrict a = A->data;
    REAL *restrict b = B->data;
    for (size_t k = 0; k < total; ++k) {
        b[k] = alpha * a[k];
    }
    return MATRIX_SUCCESS;
}

MatrixError MatrixHadamard(const Matrix *A, const Matrix *B, Matrix *C)
{
    MatrixError error = CheckSameShape(A, B, C);
    size_t total;

    if (error != MATRIX_SUCCESS) {
        return error;
    }

    total = (size_t)A->row * (size_t)A->column;
    const REAL *restrict a = A->data;
    const REAL *restrict b = B->data;
    REAL *restrict c = C->data;
    for (size_t k = 0; k < total; ++k) {
        c[k] = a[k] * b[k];
    }
    return MATRIX_SUCCESS;
}

MatrixError MatrixTranspose(const Matrix *A, Matrix *AT)
{
    if (!MatrixIsValid(A) || !MatrixIsValid(AT)) {
        return MATRIX_ERROR_NULL_POINTER;
    }
    if (AT->row != A->column || AT->column != A->row) {
        return MATRIX_ERROR_SIZE_MISMATCH;
    }
    if (MatrixDataAliases(AT, A)) {
        return MATRIX_ERROR_ALIASING;
    }

    for (int i = 0; i < A->row; ++i) {
        size_t a_row = (size_t)i * (size_t)A->column;
        for (int j = 0; j < A->column; ++j) {
            size_t at_index = (size_t)j * (size_t)AT->column + (size_t)i;
            AT->data[at_index] = A->data[a_row + (size_t)j];
        }
    }
    return MATRIX_SUCCESS;
}

MatrixError MatrixNormFrobenius(const Matrix *A, REAL *norm_value)
{
    REAL sum = 0.0;
    size_t total;

    if (!MatrixIsValid(A) || norm_value == NULL) {
        return MATRIX_ERROR_NULL_POINTER;
    }

    total = (size_t)A->row * (size_t)A->column;
    for (size_t k = 0; k < total; ++k) {
        sum += A->data[k] * A->data[k];
    }
    *norm_value = sqrt(sum);
    return MATRIX_SUCCESS;
}

int MatrixAlmostEqual(const Matrix *A, const Matrix *B, REAL tolerance)
{
    size_t total;

    if (!MatrixIsValid(A) || !MatrixIsValid(B) || tolerance < 0.0) {
        return 0;
    }
    if (A->row != B->row || A->column != B->column) {
        return 0;
    }

    total = (size_t)A->row * (size_t)A->column;
    for (size_t k = 0; k < total; ++k) {
        if (fabs(A->data[k] - B->data[k]) > tolerance) {
            return 0;
        }
    }
    return 1;
}

MatrixError MatrixMultiply(const Matrix *A, const Matrix *B, Matrix *C)
{
    if (!MatrixIsValid(A) || !MatrixIsValid(B) || !MatrixIsValid(C)) {
        return MATRIX_ERROR_NULL_POINTER;
    }
    if (A->column != B->row || C->row != A->row || C->column != B->column) {
        return MATRIX_ERROR_SIZE_MISMATCH;
    }
    if (MatrixDataAliases(C, A) || MatrixDataAliases(C, B)) {
        return MATRIX_ERROR_ALIASING;
    }

    for (int i = 0; i < A->row; ++i) {
        size_t c_row = (size_t)i * (size_t)C->column;
        size_t a_row = (size_t)i * (size_t)A->column;
        for (int j = 0; j < B->column; ++j) {
            REAL sum = 0.0;
            for (int k = 0; k < A->column; ++k) {
                sum += A->data[a_row + (size_t)k] *
                       B->data[(size_t)k * (size_t)B->column + (size_t)j];
            }
            C->data[c_row + (size_t)j] = sum;
        }
    }
    return MATRIX_SUCCESS;
}
