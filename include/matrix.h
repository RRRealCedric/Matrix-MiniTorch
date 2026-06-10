#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>

typedef double REAL;

typedef enum {
    MATRIX_SUCCESS = 0,
    MATRIX_ERROR_NULL_POINTER = 1,
    MATRIX_ERROR_INVALID_SIZE = 2,
    MATRIX_ERROR_SIZE_OVERFLOW = 3,
    MATRIX_ERROR_ALLOC_FAILED = 4,
    MATRIX_ERROR_INDEX_OUT_OF_RANGE = 5,
    MATRIX_ERROR_SIZE_MISMATCH = 6,
    MATRIX_ERROR_ALREADY_ALLOCATED = 7,
    MATRIX_ERROR_INVALID_ARGUMENT = 8,
    MATRIX_ERROR_ALIASING = 9,
    MATRIX_ERROR_NOT_SQUARE = 10,
    MATRIX_ERROR_SINGULAR = 11
} MatrixError;

typedef struct {
    int row;
    int column;
    REAL *data;
} Matrix;

void MatrixInit(Matrix *A);
MatrixError MatrixCreate(Matrix *A, int row, int column);
void MatrixFree(Matrix *A);

int MatrixIsValid(const Matrix *A);
int MatrixHasShape(const Matrix *A, int row, int column);
int MatrixIndex(const Matrix *A, int i, int j);

MatrixError MatrixSet(Matrix *A, int i, int j, REAL value);
MatrixError MatrixGet(const Matrix *A, int i, int j, REAL *value);
MatrixError MatrixFillZero(Matrix *A);
MatrixError MatrixFillSequence(Matrix *A, REAL start, REAL step);
MatrixError MatrixFillIdentity(Matrix *A);
MatrixError MatrixFillRandom(Matrix *A, unsigned int seed, REAL min_value, REAL max_value);
MatrixError MatrixCopy(const Matrix *src, Matrix *dst);
void MatrixPrint(const Matrix *A, const char *name);
const char *MatrixErrorMessage(MatrixError error);

MatrixError MatrixAdd(const Matrix *A, const Matrix *B, Matrix *C);
MatrixError MatrixSub(const Matrix *A, const Matrix *B, Matrix *C);
MatrixError MatrixScale(REAL alpha, const Matrix *A, Matrix *B);
MatrixError MatrixHadamard(const Matrix *A, const Matrix *B, Matrix *C);
MatrixError MatrixTranspose(const Matrix *A, Matrix *AT);
MatrixError MatrixMultiplyNaive(const Matrix *A, const Matrix *B, Matrix *C);
MatrixError MatrixMultiply(const Matrix *A, const Matrix *B, Matrix *C);
MatrixError MatrixNormFrobenius(const Matrix *A, REAL *norm_value);
int MatrixAlmostEqual(const Matrix *A, const Matrix *B, REAL tolerance);

#endif
