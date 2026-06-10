#ifndef MATRIX_SOLVE_H
#define MATRIX_SOLVE_H

#include "matrix.h"

MatrixError MatrixSwapRows(Matrix *A, int r1, int r2);
MatrixError GaussianSolveNoPivot(const Matrix *A, const Matrix *b, Matrix *x, REAL tol);
MatrixError GaussianSolvePartialPivot(const Matrix *A, const Matrix *b, Matrix *x, REAL tol);

#endif
