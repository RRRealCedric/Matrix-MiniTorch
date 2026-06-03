#ifndef MATRIX_OPTIMIZED_H
#define MATRIX_OPTIMIZED_H

#include "matrix.h"

MatrixError MatrixMultiplyIKJ(const Matrix *A, const Matrix *B, Matrix *C);
MatrixError MatrixMultiplyBlocked(const Matrix *A, const Matrix *B, Matrix *C, int block_size);
MatrixError MatrixMultiplyTransposeB(const Matrix *A, const Matrix *B, Matrix *C);
MatrixError MatrixMultiplyKahan(const Matrix *A, const Matrix *B, Matrix *C);

#endif
