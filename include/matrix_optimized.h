#ifndef MATRIX_OPTIMIZED_H
#define MATRIX_OPTIMIZED_H

#include "backend.h"
#include "matrix.h"

#include <stddef.h>
#include <stdio.h>

#ifndef MATRIX_PROFILE_MAX_RECORDS
#define MATRIX_PROFILE_MAX_RECORDS 256
#endif

typedef struct {
    const char *kernel_name;
    const char *backend_name;
    int m;
    int n;
    int k;
    double elapsed_seconds;
    MatrixError error;
} MatrixProfileRecord;

void MatrixProfileEnable(int enabled);
int MatrixProfileIsEnabled(void);
void MatrixProfileClear(void);
const MatrixProfileRecord *MatrixProfileGetRecords(size_t *count);
void MatrixProfilePrint(FILE *out);

MatrixError MatrixMultiplyIKJ(const Matrix *A, const Matrix *B, Matrix *C);
MatrixError MatrixMultiplyBlocked(const Matrix *A, const Matrix *B, Matrix *C, int block_size);
MatrixError MatrixMultiplyTransposeB(const Matrix *A, const Matrix *B, Matrix *C);
MatrixError MatrixMultiplyKahan(const Matrix *A, const Matrix *B, Matrix *C);
MatrixError MatrixMultiplyAuto(const Matrix *A, const Matrix *B, Matrix *C);

MatrixError MatrixTransposeBlocked(const Matrix *A, Matrix *AT, int block_size);
MatrixError MatrixAddInPlace(Matrix *A, const Matrix *B);
MatrixError MatrixScaleInPlace(Matrix *A, REAL alpha);
MatrixError MatrixAxpy(REAL alpha, const Matrix *X, Matrix *Y);

#endif
