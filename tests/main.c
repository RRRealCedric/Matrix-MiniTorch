#include "matrix.h"
#include "matrix_lu.h"
#include "matrix_optimized.h"
#include "matrix_solve.h"

#include <math.h>
#include <stdio.h>

#define EPSILON 1e-9

static int CheckError(MatrixError error, const char *operation)
{
    printf("%-40s : %s\n", operation, MatrixErrorMessage(error));
    return error == MATRIX_SUCCESS;
}

static int NearlyEqual(REAL a, REAL b)
{
    return fabs(a - b) < EPSILON;
}

static int MatrixEqualsArray(const Matrix *A, const REAL *expected)
{
    int total;

    if (!MatrixIsValid(A) || expected == NULL) {
        return 0;
    }

    total = A->row * A->column;
    for (int k = 0; k < total; ++k) {
        if (!NearlyEqual(A->data[k], expected[k])) {
            return 0;
        }
    }
    return 1;
}

static void PrintPassFail(const char *name, int pass)
{
    printf("%-40s : %s\n", name, pass ? "PASS" : "FAIL");
}

static int Require(MatrixError error, const char *operation)
{
    if (error != MATRIX_SUCCESS) {
        printf("%s failed: %s\n", operation, MatrixErrorMessage(error));
        return 0;
    }
    return 1;
}

static int TestMatrixAdd(void)
{
    Matrix A, B, C;
    REAL expected[] = {11.0, 22.0, 33.0, 44.0};
    int pass;

    MatrixInit(&A);
    MatrixInit(&B);
    MatrixInit(&C);

    if (!Require(MatrixCreate(&A, 2, 2), "MatrixCreate(A)") ||
        !Require(MatrixCreate(&B, 2, 2), "MatrixCreate(B)") ||
        !Require(MatrixCreate(&C, 2, 2), "MatrixCreate(C)")) {
        MatrixFree(&A);
        MatrixFree(&B);
        MatrixFree(&C);
        return 0;
    }

    MatrixFillSequence(&A, 1.0, 1.0);
    MatrixFillSequence(&B, 10.0, 10.0);

    printf("\n[1] Matrix addition\n");
    MatrixPrint(&A, "A");
    MatrixPrint(&B, "B");
    CheckError(MatrixAdd(&A, &B, &C), "C = A + B");
    MatrixPrint(&C, "C");

    pass = MatrixEqualsArray(&C, expected);
    PrintPassFail("addition correctness", pass);

    MatrixFree(&A);
    MatrixFree(&B);
    MatrixFree(&C);
    return pass;
}

static int TestMatrixScale(void)
{
    Matrix A, B;
    REAL expected[] = {2.5, 5.0, 7.5, 10.0, 12.5, 15.0};
    int pass;

    MatrixInit(&A);
    MatrixInit(&B);

    if (!Require(MatrixCreate(&A, 2, 3), "MatrixCreate(A)") ||
        !Require(MatrixCreate(&B, 2, 3), "MatrixCreate(B)")) {
        MatrixFree(&A);
        MatrixFree(&B);
        return 0;
    }

    MatrixFillSequence(&A, 1.0, 1.0);

    printf("\n[2] Matrix scalar multiplication\n");
    MatrixPrint(&A, "A");
    CheckError(MatrixScale(2.5, &A, &B), "B = 2.5 * A");
    MatrixPrint(&B, "B");

    pass = MatrixEqualsArray(&B, expected);
    PrintPassFail("scale correctness", pass);

    MatrixFree(&A);
    MatrixFree(&B);
    return pass;
}

static int TestMatrixTranspose(void)
{
    Matrix A, AT;
    REAL expected[] = {1.0, 4.0, 2.0, 5.0, 3.0, 6.0};
    int pass;

    MatrixInit(&A);
    MatrixInit(&AT);

    if (!Require(MatrixCreate(&A, 2, 3), "MatrixCreate(A)") ||
        !Require(MatrixCreate(&AT, 3, 2), "MatrixCreate(AT)")) {
        MatrixFree(&A);
        MatrixFree(&AT);
        return 0;
    }

    MatrixFillSequence(&A, 1.0, 1.0);

    printf("\n[3] Matrix transpose\n");
    MatrixPrint(&A, "A");
    CheckError(MatrixTranspose(&A, &AT), "AT = transpose(A)");
    MatrixPrint(&AT, "AT");

    pass = MatrixEqualsArray(&AT, expected);
    PrintPassFail("transpose correctness", pass);

    MatrixFree(&A);
    MatrixFree(&AT);
    return pass;
}

static int TestMatrixMultiply(void)
{
    Matrix A, B, C;
    REAL expected[] = {22.0, 28.0, 49.0, 64.0};
    int pass;

    MatrixInit(&A);
    MatrixInit(&B);
    MatrixInit(&C);

    if (!Require(MatrixCreate(&A, 2, 3), "MatrixCreate(A)") ||
        !Require(MatrixCreate(&B, 3, 2), "MatrixCreate(B)") ||
        !Require(MatrixCreate(&C, 2, 2), "MatrixCreate(C)")) {
        MatrixFree(&A);
        MatrixFree(&B);
        MatrixFree(&C);
        return 0;
    }

    MatrixFillSequence(&A, 1.0, 1.0);
    MatrixFillSequence(&B, 1.0, 1.0);

    printf("\n[4] Matrix multiplication\n");
    MatrixPrint(&A, "A");
    MatrixPrint(&B, "B");
    CheckError(MatrixMultiply(&A, &B, &C), "C = A * B");
    MatrixPrint(&C, "C");

    pass = MatrixEqualsArray(&C, expected);
    PrintPassFail("multiplication correctness", pass);

    MatrixFree(&A);
    MatrixFree(&B);
    MatrixFree(&C);
    return pass;
}

static int TestExtraMatrixOps(void)
{
    Matrix A, B, C, I;
    REAL sub_expected[] = {-9.0, -18.0, -27.0, -36.0};
    REAL hadamard_expected[] = {10.0, 40.0, 90.0, 160.0};
    REAL norm_value = 0.0;
    int pass = 1;

    MatrixInit(&A);
    MatrixInit(&B);
    MatrixInit(&C);
    MatrixInit(&I);

    Require(MatrixCreate(&A, 2, 2), "MatrixCreate(A)");
    Require(MatrixCreate(&B, 2, 2), "MatrixCreate(B)");
    Require(MatrixCreate(&C, 2, 2), "MatrixCreate(C)");
    Require(MatrixCreate(&I, 3, 3), "MatrixCreate(I)");

    MatrixFillSequence(&A, 1.0, 1.0);
    MatrixFillSequence(&B, 10.0, 10.0);

    printf("\n[5] Extended matrix operations\n");

    MatrixSub(&A, &B, &C);
    pass = pass && MatrixEqualsArray(&C, sub_expected);
    PrintPassFail("subtraction correctness", pass);

    MatrixHadamard(&A, &B, &C);
    pass = pass && MatrixEqualsArray(&C, hadamard_expected);
    PrintPassFail("Hadamard correctness", pass);

    MatrixNormFrobenius(&A, &norm_value);
    pass = pass && NearlyEqual(norm_value, sqrt(30.0));
    PrintPassFail("Frobenius norm correctness", pass);

    MatrixFillIdentity(&I);
    pass = pass && NearlyEqual(I.data[0], 1.0) && NearlyEqual(I.data[4], 1.0) &&
           NearlyEqual(I.data[8], 1.0) && NearlyEqual(I.data[1], 0.0);
    PrintPassFail("identity fill correctness", pass);

    MatrixFree(&A);
    MatrixFree(&B);
    MatrixFree(&C);
    MatrixFree(&I);
    return pass;
}

static int TestOptimizedMultiply(void)
{
    Matrix A, B, C0, C1, C2, C3, C4, C5;
    int pass;

    MatrixInit(&A);
    MatrixInit(&B);
    MatrixInit(&C0);
    MatrixInit(&C1);
    MatrixInit(&C2);
    MatrixInit(&C3);
    MatrixInit(&C4);
    MatrixInit(&C5);

    Require(MatrixCreate(&A, 4, 4), "MatrixCreate(A)");
    Require(MatrixCreate(&B, 4, 4), "MatrixCreate(B)");
    Require(MatrixCreate(&C0, 4, 4), "MatrixCreate(C0)");
    Require(MatrixCreate(&C1, 4, 4), "MatrixCreate(C1)");
    Require(MatrixCreate(&C2, 4, 4), "MatrixCreate(C2)");
    Require(MatrixCreate(&C3, 4, 4), "MatrixCreate(C3)");
    Require(MatrixCreate(&C4, 4, 4), "MatrixCreate(C4)");
    Require(MatrixCreate(&C5, 4, 4), "MatrixCreate(C5)");

    MatrixFillSequence(&A, 1.0, 0.25);
    MatrixFillSequence(&B, 2.0, 0.5);

    printf("\n[6] Optimized multiplication kernels\n");
    MatrixMultiplyNaive(&A, &B, &C0);
    MatrixMultiply(&A, &B, &C1);
    MatrixMultiplyIKJ(&A, &B, &C1);
    MatrixMultiplyBlocked(&A, &B, &C2, 2);
    MatrixMultiplyTransposeB(&A, &B, &C3);
    MatrixMultiplyKahan(&A, &B, &C4);
    MatrixMultiplyAuto(&A, &B, &C5);

    pass = MatrixAlmostEqual(&C0, &C1, EPSILON) &&
           MatrixAlmostEqual(&C0, &C2, EPSILON) &&
           MatrixAlmostEqual(&C0, &C3, EPSILON) &&
           MatrixAlmostEqual(&C0, &C4, EPSILON) &&
           MatrixAlmostEqual(&C0, &C5, EPSILON);
    PrintPassFail("optimized kernels equal to naive baseline", pass);

    MatrixFree(&A);
    MatrixFree(&B);
    MatrixFree(&C0);
    MatrixFree(&C1);
    MatrixFree(&C2);
    MatrixFree(&C3);
    MatrixFree(&C4);
    MatrixFree(&C5);
    return pass;
}

static int TestOptimizedUtilities(void)
{
    Matrix A, B, AT0, AT1, ScaleTmp, AddTmp;
    size_t profile_count = 0;
    int pass = 1;

    MatrixInit(&A);
    MatrixInit(&B);
    MatrixInit(&AT0);
    MatrixInit(&AT1);
    MatrixInit(&ScaleTmp);
    MatrixInit(&AddTmp);

    Require(MatrixCreate(&A, 3, 4), "MatrixCreate(A)");
    Require(MatrixCreate(&B, 3, 4), "MatrixCreate(B)");
    Require(MatrixCreate(&AT0, 4, 3), "MatrixCreate(AT0)");
    Require(MatrixCreate(&AT1, 4, 3), "MatrixCreate(AT1)");
    Require(MatrixCreate(&ScaleTmp, 3, 4), "MatrixCreate(ScaleTmp)");
    Require(MatrixCreate(&AddTmp, 3, 4), "MatrixCreate(AddTmp)");

    MatrixFillSequence(&A, 1.0, 1.0);
    MatrixFillSequence(&B, 10.0, 2.0);

    printf("\n[7] Optimized utilities and profiling\n");

    MatrixTranspose(&A, &AT0);
    pass = pass && MatrixTransposeBlocked(&A, &AT1, 2) == MATRIX_SUCCESS;
    pass = pass && MatrixAlmostEqual(&AT0, &AT1, EPSILON);
    PrintPassFail("blocked transpose equals normal transpose", pass);

    MatrixScale(3.0, &A, &ScaleTmp);
    MatrixAdd(&B, &ScaleTmp, &AddTmp);
    pass = pass && MatrixAxpy(3.0, &A, &B) == MATRIX_SUCCESS;
    pass = pass && MatrixAlmostEqual(&B, &AddTmp, EPSILON);
    PrintPassFail("AXPY equals scale plus add", pass);

    pass = pass && MatrixScaleInPlace(&A, 0.5) == MATRIX_SUCCESS;
    pass = pass && MatrixAddInPlace(&A, &A) == MATRIX_ERROR_ALIASING;
    pass = pass && MatrixTransposeBlocked(&A, &AT1, 0) == MATRIX_ERROR_INVALID_ARGUMENT;
    PrintPassFail("optimized utility error handling", pass);

    MatrixProfileEnable(0);
    MatrixProfileClear();
    MatrixMultiplyIKJ(&A, &AT0, &ScaleTmp);
    MatrixProfileGetRecords(&profile_count);
    pass = pass && profile_count == 0;
    PrintPassFail("profiling disabled records nothing", pass);

    MatrixProfileEnable(1);
    MatrixProfileClear();
    MatrixMultiplyAuto(&A, &AT0, &ScaleTmp);
    MatrixProfileGetRecords(&profile_count);
    pass = pass && profile_count > 0;
    PrintPassFail("profiling records enabled kernels", pass);

    MatrixProfileClear();
    MatrixProfileGetRecords(&profile_count);
    pass = pass && profile_count == 0;
    PrintPassFail("profiling clear removes records", pass);
    MatrixProfileEnable(0);

    MatrixFree(&A);
    MatrixFree(&B);
    MatrixFree(&AT0);
    MatrixFree(&AT1);
    MatrixFree(&ScaleTmp);
    MatrixFree(&AddTmp);
    return pass;
}

static int TestErrorHandling(void)
{
    Matrix A, B, C, AT;
    REAL value = 0.0;
    int pass = 1;

    MatrixInit(&A);
    MatrixInit(&B);
    MatrixInit(&C);
    MatrixInit(&AT);

    Require(MatrixCreate(&A, 2, 3), "MatrixCreate(A)");
    Require(MatrixCreate(&B, 3, 2), "MatrixCreate(B)");
    Require(MatrixCreate(&C, 2, 2), "MatrixCreate(C)");
    Require(MatrixCreate(&AT, 2, 3), "MatrixCreate(AT)");

    printf("\n[8] Error handling\n");

    pass = pass && MatrixAdd(&A, &B, &C) == MATRIX_ERROR_SIZE_MISMATCH;
    PrintPassFail("addition size mismatch", pass);

    pass = pass && MatrixMultiply(&A, &A, &C) == MATRIX_ERROR_SIZE_MISMATCH;
    PrintPassFail("multiplication size mismatch", pass);

    pass = pass && MatrixTranspose(&A, &AT) == MATRIX_ERROR_SIZE_MISMATCH;
    PrintPassFail("transpose output size mismatch", pass);

    pass = pass && MatrixGet(&A, 9, 0, &value) == MATRIX_ERROR_INDEX_OUT_OF_RANGE;
    PrintPassFail("index out of range", pass);

    pass = pass && MatrixAdd(&A, &A, &A) == MATRIX_ERROR_ALIASING;
    PrintPassFail("output aliases input", pass);

    MatrixFree(&A);
    MatrixFree(&B);
    MatrixFree(&C);
    MatrixFree(&AT);
    return pass;
}

static int TestLUAndSolvers(void)
{
    Matrix A, L, U, LU, b, x_lu, x_gauss;
    REAL lu_expected[] = {1.0, 1.0, 2.0};
    REAL gauss_expected[] = {2.0, 3.0, -1.0};
    REAL det = 0.0;
    int pass = 1;

    MatrixInit(&A);
    MatrixInit(&L);
    MatrixInit(&U);
    MatrixInit(&LU);
    MatrixInit(&b);
    MatrixInit(&x_lu);
    MatrixInit(&x_gauss);

    Require(MatrixCreate(&A, 3, 3), "MatrixCreate(A)");
    Require(MatrixCreate(&L, 3, 3), "MatrixCreate(L)");
    Require(MatrixCreate(&U, 3, 3), "MatrixCreate(U)");
    Require(MatrixCreate(&LU, 3, 3), "MatrixCreate(LU)");
    Require(MatrixCreate(&b, 3, 1), "MatrixCreate(b)");
    Require(MatrixCreate(&x_lu, 3, 1), "MatrixCreate(x_lu)");
    Require(MatrixCreate(&x_gauss, 3, 1), "MatrixCreate(x_gauss)");

    REAL lu_a[] = {
        2.0, 1.0, 1.0,
        4.0, -6.0, 0.0,
        -2.0, 7.0, 2.0
    };
    REAL lu_b[] = {5.0, -2.0, 9.0};
    for (int i = 0; i < 9; ++i) {
        A.data[i] = lu_a[i];
    }
    for (int i = 0; i < 3; ++i) {
        b.data[i] = lu_b[i];
    }

    printf("\n[9] LU decomposition and linear solvers\n");
    pass = pass && LUDecomposeNoPivot(&A, &L, &U, 1e-12) == MATRIX_SUCCESS;
    pass = pass && MatrixMultiply(&L, &U, &LU) == MATRIX_SUCCESS;
    pass = pass && MatrixAlmostEqual(&A, &LU, EPSILON);
    PrintPassFail("LU reconstructs A", pass);

    pass = pass && LUSolve(&L, &U, &b, &x_lu, 1e-12) == MATRIX_SUCCESS;
    pass = pass && MatrixEqualsArray(&x_lu, lu_expected);
    PrintPassFail("LUSolve correctness", pass);

    pass = pass && LUDeterminant(&U, &det) == MATRIX_SUCCESS;
    pass = pass && NearlyEqual(det, -16.0);
    PrintPassFail("LU determinant correctness", pass);

    REAL gauss_a[] = {
        2.0, 1.0, -1.0,
        -3.0, -1.0, 2.0,
        -2.0, 1.0, 2.0
    };
    REAL gauss_b[] = {8.0, -11.0, -3.0};
    for (int i = 0; i < 9; ++i) {
        A.data[i] = gauss_a[i];
    }
    for (int i = 0; i < 3; ++i) {
        b.data[i] = gauss_b[i];
    }

    pass = pass && GaussianSolvePartialPivot(&A, &b, &x_gauss, 1e-12) == MATRIX_SUCCESS;
    pass = pass && MatrixEqualsArray(&x_gauss, gauss_expected);
    PrintPassFail("Gaussian partial pivot correctness", pass);

    MatrixFree(&A);
    MatrixFree(&L);
    MatrixFree(&U);
    MatrixFree(&LU);
    MatrixFree(&b);
    MatrixFree(&x_lu);
    MatrixFree(&x_gauss);
    return pass;
}

int main(void)
{
    int pass = 1;

    printf("Small C Matrix Library - correctness tests\n");
    printf("==========================================\n");

    pass = TestMatrixAdd() && pass;
    pass = TestMatrixScale() && pass;
    pass = TestMatrixTranspose() && pass;
    pass = TestMatrixMultiply() && pass;
    pass = TestExtraMatrixOps() && pass;
    pass = TestOptimizedMultiply() && pass;
    pass = TestOptimizedUtilities() && pass;
    pass = TestErrorHandling() && pass;
    pass = TestLUAndSolvers() && pass;

    printf("\nOverall result: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
