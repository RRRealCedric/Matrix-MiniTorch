#include "backend.h"
#include "tensor.h"
#include "tensor_ops.h"

#include <stdio.h>

#define EPOCHS 2000
#define LEARNING_RATE 0.1

static int Require(MatrixError error, const char *operation)
{
    if (error != MATRIX_SUCCESS) {
        printf("%s failed: %s\n", operation, MatrixErrorMessage(error));
        return 0;
    }
    return 1;
}

static void SetXorData(Tensor *X, Tensor *Y)
{
    REAL x_data[] = {
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0
    };
    REAL y_data[] = {0.0, 1.0, 1.0, 0.0};

    for (int i = 0; i < X->size; ++i) {
        X->data[i] = x_data[i];
    }
    for (int i = 0; i < Y->size; ++i) {
        Y->data[i] = y_data[i];
    }
}

static void InitParameters(Tensor *W1, Tensor *b1, Tensor *W2, Tensor *b2)
{
    REAL w1_data[] = {
        0.80, -0.90,
       -1.10,  0.70
    };
    REAL b1_data[] = {0.05, -0.04};
    REAL w2_data[] = {0.70, 0.80};

    for (int i = 0; i < W1->size; ++i) {
        W1->data[i] = w1_data[i];
    }
    for (int i = 0; i < b1->size; ++i) {
        b1->data[i] = b1_data[i];
    }
    for (int i = 0; i < W2->size; ++i) {
        W2->data[i] = w2_data[i];
    }
    b2->data[0] = 0.10;
}

static MatrixError Forward(Tensor *X,
                           Tensor *Target,
                           Tensor *W1,
                           Tensor *b1,
                           Tensor *W2,
                           Tensor *b2,
                           Tensor *XW1,
                           Tensor *Z1,
                           Tensor *H,
                           Tensor *HW2,
                           Tensor *Pred,
                           Tensor *Loss)
{
    MatrixError error;

    error = TensorMatmulAuto(X, W1, XW1);
    if (error != MATRIX_SUCCESS) {
        return error;
    }
    error = TensorAddBiasAuto(XW1, b1, Z1);
    if (error != MATRIX_SUCCESS) {
        return error;
    }
    error = TensorReluAuto(Z1, H);
    if (error != MATRIX_SUCCESS) {
        return error;
    }
    error = TensorMatmulAuto(H, W2, HW2);
    if (error != MATRIX_SUCCESS) {
        return error;
    }
    error = TensorAddBiasAuto(HW2, b2, Pred);
    if (error != MATRIX_SUCCESS) {
        return error;
    }
    return TensorMSELossAuto(Pred, Target, Loss);
}

static void ZeroParameterGrads(Tensor *W1, Tensor *b1, Tensor *W2, Tensor *b2)
{
    TensorZeroGrad(W1);
    TensorZeroGrad(b1);
    TensorZeroGrad(W2);
    TensorZeroGrad(b2);
}

static void StepParameters(Tensor *W1, Tensor *b1, Tensor *W2, Tensor *b2)
{
    TensorSGDStep(W1, LEARNING_RATE);
    TensorSGDStep(b1, LEARNING_RATE);
    TensorSGDStep(W2, LEARNING_RATE);
    TensorSGDStep(b2, LEARNING_RATE);
}

int main(void)
{
    Tensor X, Target, W1, b1, W2, b2;
    Tensor XW1, Z1, H, HW2, Pred, Loss;

    TensorInit(&X);
    TensorInit(&Target);
    TensorInit(&W1);
    TensorInit(&b1);
    TensorInit(&W2);
    TensorInit(&b2);
    TensorInit(&XW1);
    TensorInit(&Z1);
    TensorInit(&H);
    TensorInit(&HW2);
    TensorInit(&Pred);
    TensorInit(&Loss);

    if (!Require(TensorCreate2D(&X, 4, 2), "TensorCreate2D(X)") ||
        !Require(TensorCreate2D(&Target, 4, 1), "TensorCreate2D(Target)") ||
        !Require(TensorCreate2D(&W1, 2, 2), "TensorCreate2D(W1)") ||
        !Require(TensorCreate1D(&b1, 2), "TensorCreate1D(b1)") ||
        !Require(TensorCreate2D(&W2, 2, 1), "TensorCreate2D(W2)") ||
        !Require(TensorCreate1D(&b2, 1), "TensorCreate1D(b2)") ||
        !Require(TensorCreate2D(&XW1, 4, 2), "TensorCreate2D(XW1)") ||
        !Require(TensorCreate2D(&Z1, 4, 2), "TensorCreate2D(Z1)") ||
        !Require(TensorCreate2D(&H, 4, 2), "TensorCreate2D(H)") ||
        !Require(TensorCreate2D(&HW2, 4, 1), "TensorCreate2D(HW2)") ||
        !Require(TensorCreate2D(&Pred, 4, 1), "TensorCreate2D(Pred)") ||
        !Require(TensorCreate1D(&Loss, 1), "TensorCreate1D(Loss)")) {
        return 1;
    }

    TensorSetRequiresGrad(&W1, 1);
    TensorSetRequiresGrad(&b1, 1);
    TensorSetRequiresGrad(&W2, 1);
    TensorSetRequiresGrad(&b2, 1);
    BackendSetType(BACKEND_IKJ);

    SetXorData(&X, &Target);
    InitParameters(&W1, &b1, &W2, &b2);

    printf("Mini autograd MLP demo: XOR\n");
    printf("===========================\n");
    printf("Network: X -> Linear(2,2) -> ReLU -> Linear(2,1) -> MSE\n");
    printf("Backend: %s\n\n", BackendName(BackendGetType()));

    for (int epoch = 0; epoch <= EPOCHS; ++epoch) {
        MatrixError error;

        ZeroParameterGrads(&W1, &b1, &W2, &b2);
        error = Forward(&X, &Target, &W1, &b1, &W2, &b2,
                        &XW1, &Z1, &H, &HW2, &Pred, &Loss);
        if (error != MATRIX_SUCCESS) {
            printf("Forward failed: %s\n", MatrixErrorMessage(error));
            return 1;
        }
        error = TensorBackward(&Loss);
        if (error != MATRIX_SUCCESS) {
            printf("Backward failed: %s\n", MatrixErrorMessage(error));
            return 1;
        }

        if (epoch % 400 == 0) {
            printf("epoch %4d  loss = %.8f\n", epoch, Loss.data[0]);
        }

        StepParameters(&W1, &b1, &W2, &b2);
    }

    Require(Forward(&X, &Target, &W1, &b1, &W2, &b2,
                    &XW1, &Z1, &H, &HW2, &Pred, &Loss),
            "final forward");

    printf("\nFinal prediction\n");
    TensorPrint(&Pred, "Pred");
    TensorPrint(&Target, "Target");
    printf("Final loss = %.8f\n", Loss.data[0]);

    TensorFree(&X);
    TensorFree(&Target);
    TensorFree(&W1);
    TensorFree(&b1);
    TensorFree(&W2);
    TensorFree(&b2);
    TensorFree(&XW1);
    TensorFree(&Z1);
    TensorFree(&H);
    TensorFree(&HW2);
    TensorFree(&Pred);
    TensorFree(&Loss);
    return 0;
}
