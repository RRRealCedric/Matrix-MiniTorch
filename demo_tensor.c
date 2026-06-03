#include "backend.h"
#include "tensor.h"
#include "tensor_ops.h"

#include <stdio.h>

static int Require(MatrixError error, const char *operation)
{
    if (error != MATRIX_SUCCESS) {
        printf("%s failed: %s\n", operation, MatrixErrorMessage(error));
        return 0;
    }
    return 1;
}

static void RunMatmulWithBackend(BackendType backend, const Tensor *A, const Tensor *B, Tensor *C)
{
    BackendSetType(backend);
    TensorFillZero(C);
    printf("\nTensorMatmul backend = %s\n", BackendName(BackendGetType()));
    Require(TensorMatmul(A, B, C), "TensorMatmul");
    TensorPrint(C, "C");
}

int main(void)
{
    Tensor A, B, C, D;

    TensorInit(&A);
    TensorInit(&B);
    TensorInit(&C);
    TensorInit(&D);

    if (!Require(TensorCreate2D(&A, 2, 3), "TensorCreate2D(A)") ||
        !Require(TensorCreate2D(&B, 3, 2), "TensorCreate2D(B)") ||
        !Require(TensorCreate2D(&C, 2, 2), "TensorCreate2D(C)") ||
        !Require(TensorCreate2D(&D, 2, 3), "TensorCreate2D(D)")) {
        TensorFree(&A);
        TensorFree(&B);
        TensorFree(&C);
        TensorFree(&D);
        return 1;
    }

    TensorFillSequence(&A, 1.0, 1.0);
    TensorFillSequence(&B, 1.0, 1.0);

    printf("Mini Tensor infra demo\n");
    printf("======================\n");
    TensorPrint(&A, "A");
    TensorPrint(&B, "B");

    Require(TensorScale(0.5, &A, &D), "D = 0.5 * A");
    TensorPrint(&D, "D = 0.5 * A");

    RunMatmulWithBackend(BACKEND_NAIVE, &A, &B, &C);
    RunMatmulWithBackend(BACKEND_IKJ, &A, &B, &C);
    RunMatmulWithBackend(BACKEND_BLOCKED, &A, &B, &C);

    TensorFree(&A);
    TensorFree(&B);
    TensorFree(&C);
    TensorFree(&D);
    return 0;
}
