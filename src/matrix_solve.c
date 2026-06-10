#include "matrix_solve.h"

#include <math.h>

MatrixError MatrixSwapRows(Matrix *A, int r1, int r2)
{
    if (!MatrixIsValid(A)) {
        return MATRIX_ERROR_NULL_POINTER;
    }
    if (r1 < 0 || r1 >= A->row || r2 < 0 || r2 >= A->row) {
        return MATRIX_ERROR_INDEX_OUT_OF_RANGE;
    }
    if (r1 == r2) {
        return MATRIX_SUCCESS;
    }

    for (int j = 0; j < A->column; ++j) {
        int idx1 = MatrixIndex(A, r1, j);
        int idx2 = MatrixIndex(A, r2, j);
        REAL tmp = A->data[idx1];
        A->data[idx1] = A->data[idx2];
        A->data[idx2] = tmp;
    }
    return MATRIX_SUCCESS;
}

static MatrixError CheckLinearSystem(const Matrix *A, const Matrix *b, Matrix *x, REAL tol)
{
    if (!MatrixIsValid(A) || !MatrixIsValid(b) || !MatrixIsValid(x)) {
        return MATRIX_ERROR_NULL_POINTER;
    }
    if (tol < 0.0) {
        return MATRIX_ERROR_INVALID_ARGUMENT;
    }
    if (A->row != A->column) {
        return MATRIX_ERROR_NOT_SQUARE;
    }
    if (b->row != A->row || b->column != 1) {
        return MATRIX_ERROR_SIZE_MISMATCH;
    }
    if (x->row != A->column || x->column != 1) {
        return MATRIX_ERROR_SIZE_MISMATCH;
    }
    if (x->data == b->data) {
        return MATRIX_ERROR_ALIASING;
    }
    return MATRIX_SUCCESS;
}

MatrixError GaussianSolveNoPivot(const Matrix *A, const Matrix *b, Matrix *x, REAL tol)
{
    MatrixError error = CheckLinearSystem(A, b, x, tol);
    Matrix U, rhs;
    int n;

    if (error != MATRIX_SUCCESS) {
        return error;
    }

    n = A->row;
    MatrixInit(&U);
    MatrixInit(&rhs);

    error = MatrixCreate(&U, n, n);
    if (error != MATRIX_SUCCESS) {
        return error;
    }
    error = MatrixCreate(&rhs, n, 1);
    if (error != MATRIX_SUCCESS) {
        MatrixFree(&U);
        return error;
    }

    MatrixCopy(A, &U);
    MatrixCopy(b, &rhs);

    for (int k = 0; k < n - 1; ++k) {
        REAL pivot = U.data[MatrixIndex(&U, k, k)];
        if (fabs(pivot) < tol) {
            MatrixFree(&U);
            MatrixFree(&rhs);
            return MATRIX_ERROR_SINGULAR;
        }

        for (int i = k + 1; i < n; ++i) {
            REAL factor = U.data[MatrixIndex(&U, i, k)] / pivot;
            U.data[MatrixIndex(&U, i, k)] = 0.0;
            for (int j = k + 1; j < n; ++j) {
                U.data[MatrixIndex(&U, i, j)] -= factor * U.data[MatrixIndex(&U, k, j)];
            }
            rhs.data[i] -= factor * rhs.data[k];
        }
    }

    for (int i = n - 1; i >= 0; --i) {
        REAL sum = rhs.data[i];
        for (int j = i + 1; j < n; ++j) {
            sum -= U.data[MatrixIndex(&U, i, j)] * x->data[j];
        }

        REAL diag = U.data[MatrixIndex(&U, i, i)];
        if (fabs(diag) < tol) {
            MatrixFree(&U);
            MatrixFree(&rhs);
            return MATRIX_ERROR_SINGULAR;
        }
        x->data[i] = sum / diag;
    }

    MatrixFree(&U);
    MatrixFree(&rhs);
    return MATRIX_SUCCESS;
}

MatrixError GaussianSolvePartialPivot(const Matrix *A, const Matrix *b, Matrix *x, REAL tol)
{
    MatrixError error = CheckLinearSystem(A, b, x, tol);
    Matrix U, rhs;
    int n;

    if (error != MATRIX_SUCCESS) {
        return error;
    }

    n = A->row;
    MatrixInit(&U);
    MatrixInit(&rhs);

    error = MatrixCreate(&U, n, n);
    if (error != MATRIX_SUCCESS) {
        return error;
    }
    error = MatrixCreate(&rhs, n, 1);
    if (error != MATRIX_SUCCESS) {
        MatrixFree(&U);
        return error;
    }

    MatrixCopy(A, &U);
    MatrixCopy(b, &rhs);

    for (int k = 0; k < n - 1; ++k) {
        int pivot_row = k;
        REAL pivot_abs = fabs(U.data[MatrixIndex(&U, k, k)]);

        for (int i = k + 1; i < n; ++i) {
            REAL value_abs = fabs(U.data[MatrixIndex(&U, i, k)]);
            if (value_abs > pivot_abs) {
                pivot_abs = value_abs;
                pivot_row = i;
            }
        }

        if (pivot_abs < tol) {
            MatrixFree(&U);
            MatrixFree(&rhs);
            return MATRIX_ERROR_SINGULAR;
        }

        MatrixSwapRows(&U, k, pivot_row);
        MatrixSwapRows(&rhs, k, pivot_row);

        REAL pivot = U.data[MatrixIndex(&U, k, k)];
        for (int i = k + 1; i < n; ++i) {
            REAL factor = U.data[MatrixIndex(&U, i, k)] / pivot;
            U.data[MatrixIndex(&U, i, k)] = 0.0;
            for (int j = k + 1; j < n; ++j) {
                U.data[MatrixIndex(&U, i, j)] -= factor * U.data[MatrixIndex(&U, k, j)];
            }
            rhs.data[i] -= factor * rhs.data[k];
        }
    }

    for (int i = n - 1; i >= 0; --i) {
        REAL sum = rhs.data[i];
        for (int j = i + 1; j < n; ++j) {
            sum -= U.data[MatrixIndex(&U, i, j)] * x->data[j];
        }

        REAL diag = U.data[MatrixIndex(&U, i, i)];
        if (fabs(diag) < tol) {
            MatrixFree(&U);
            MatrixFree(&rhs);
            return MATRIX_ERROR_SINGULAR;
        }
        x->data[i] = sum / diag;
    }

    MatrixFree(&U);
    MatrixFree(&rhs);
    return MATRIX_SUCCESS;
}
