#include "backend.h"
#include "digits_dataset.h"
#include "tensor.h"
#include "tensor_ops.h"

#include <stdio.h>
#include <stdlib.h>

#define DIGITS_INPUT_DIM 784
#define DIGITS_HIDDEN_DIM 128
#define DIGITS_OUTPUT_DIM 10
#define DIGITS_DEFAULT_EPOCHS 200
#define DIGITS_LEARNING_RATE 0.05

static int Require(MatrixError error, const char *operation)
{
    if (error != MATRIX_SUCCESS) {
        printf("%s failed: %s\n", operation, MatrixErrorMessage(error));
        return 0;
    }
    return 1;
}

static void InitDigitsParameters(Tensor *W1, Tensor *b1, Tensor *W2, Tensor *b2)
{
    for (int i = 0; i < W1->size; ++i) {
        W1->data[i] = ((REAL)((i % 17) - 8)) / 128.0;
    }
    for (int i = 0; i < b1->size; ++i) {
        b1->data[i] = 0.0;
    }
    for (int i = 0; i < W2->size; ++i) {
        W2->data[i] = ((REAL)((i % 13) - 6)) / 64.0;
    }
    for (int i = 0; i < b2->size; ++i) {
        b2->data[i] = 0.0;
    }
}

static MatrixError ForwardDigits(Tensor *X,
                                 Tensor *Target,
                                 Tensor *W1,
                                 Tensor *b1,
                                 Tensor *W2,
                                 Tensor *b2,
                                 Tensor *HiddenLinear,
                                 Tensor *HiddenBias,
                                 Tensor *Hidden,
                                 Tensor *OutputLinear,
                                 Tensor *Pred,
                                 Tensor *Loss)
{
    MatrixError error;

    error = TensorMatmulAuto(X, W1, HiddenLinear);
    if (error != MATRIX_SUCCESS) {
        return error;
    }
    error = TensorAddBiasAuto(HiddenLinear, b1, HiddenBias);
    if (error != MATRIX_SUCCESS) {
        return error;
    }
    error = TensorReluAuto(HiddenBias, Hidden);
    if (error != MATRIX_SUCCESS) {
        return error;
    }
    error = TensorMatmulAuto(Hidden, W2, OutputLinear);
    if (error != MATRIX_SUCCESS) {
        return error;
    }
    error = TensorAddBiasAuto(OutputLinear, b2, Pred);
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
    TensorSGDStep(W1, DIGITS_LEARNING_RATE);
    TensorSGDStep(b1, DIGITS_LEARNING_RATE);
    TensorSGDStep(W2, DIGITS_LEARNING_RATE);
    TensorSGDStep(b2, DIGITS_LEARNING_RATE);
}

static int ArgMaxRow(const Tensor *T, int row)
{
    int best_index = 0;
    REAL best_value = T->data[row * T->stride[0]];

    for (int j = 1; j < T->shape[1]; ++j) {
        REAL value = T->data[row * T->stride[0] + j];
        if (value > best_value) {
            best_value = value;
            best_index = j;
        }
    }
    return best_index;
}

int main(int argc, char **argv)
{
    const char *train_path = "data/mnist_train_small.csv";
    const char *test_path = "data/mnist_test_small.csv";
    int epochs = DIGITS_DEFAULT_EPOCHS;
    int exit_code = 0;
    DigitsDataset train_set, test_set;
    Tensor W1, b1, W2, b2, HiddenLinear, HiddenBias, Hidden, OutputLinear, Pred, Loss;

    if (argc >= 2) {
        train_path = argv[1];
    }
    if (argc >= 3) {
        test_path = argv[2];
    }
    if (argc >= 4) {
        epochs = atoi(argv[3]);
        if (epochs <= 0) {
            epochs = DIGITS_DEFAULT_EPOCHS;
        }
    }

    DigitsDatasetInit(&train_set);
    DigitsDatasetInit(&test_set);
    TensorInit(&W1);
    TensorInit(&b1);
    TensorInit(&W2);
    TensorInit(&b2);
    TensorInit(&HiddenLinear);
    TensorInit(&HiddenBias);
    TensorInit(&Hidden);
    TensorInit(&OutputLinear);
    TensorInit(&Pred);
    TensorInit(&Loss);

    if (!Require(DigitsDatasetLoadCSV(train_path, &train_set), "DigitsDatasetLoadCSV(train)") ||
        !Require(DigitsDatasetLoadCSV(test_path, &test_set), "DigitsDatasetLoadCSV(test)")) {
        printf("Expected CSV files:\n  %s\n  %s\n", train_path, test_path);
        exit_code = 1;
        goto cleanup;
    }

    if (!Require(TensorCreate2D(&W1, DIGITS_INPUT_DIM, DIGITS_HIDDEN_DIM), "TensorCreate2D(W1)") ||
        !Require(TensorCreate1D(&b1, DIGITS_HIDDEN_DIM), "TensorCreate1D(b1)") ||
        !Require(TensorCreate2D(&W2, DIGITS_HIDDEN_DIM, DIGITS_OUTPUT_DIM), "TensorCreate2D(W2)") ||
        !Require(TensorCreate1D(&b2, DIGITS_OUTPUT_DIM), "TensorCreate1D(b2)") ||
        !Require(TensorCreate2D(&HiddenLinear, train_set.sample_count, DIGITS_HIDDEN_DIM), "TensorCreate2D(HiddenLinear)") ||
        !Require(TensorCreate2D(&HiddenBias, train_set.sample_count, DIGITS_HIDDEN_DIM), "TensorCreate2D(HiddenBias)") ||
        !Require(TensorCreate2D(&Hidden, train_set.sample_count, DIGITS_HIDDEN_DIM), "TensorCreate2D(Hidden)") ||
        !Require(TensorCreate2D(&OutputLinear, train_set.sample_count, DIGITS_OUTPUT_DIM), "TensorCreate2D(OutputLinear)") ||
        !Require(TensorCreate2D(&Pred, train_set.sample_count, DIGITS_OUTPUT_DIM), "TensorCreate2D(Pred)") ||
        !Require(TensorCreate1D(&Loss, 1), "TensorCreate1D(Loss)")) {
        goto cleanup;
    }

    TensorSetRequiresGrad(&W1, 1);
    TensorSetRequiresGrad(&b1, 1);
    TensorSetRequiresGrad(&W2, 1);
    TensorSetRequiresGrad(&b2, 1);
    BackendSetType(BACKEND_SIMD);

    InitDigitsParameters(&W1, &b1, &W2, &b2);

    printf("Digits MLP demo\n");
    printf("===============\n");
    printf("Train: %s\n", train_path);
    printf("Test:  %s\n", test_path);
    printf("Network: 784 -> 128 -> 10\n");
    printf("Loss: MSE + one-hot\n");
    printf("Epochs: %d\n\n", epochs);

    for (int epoch = 0; epoch <= epochs; ++epoch) {
        MatrixError error;

        ZeroParameterGrads(&W1, &b1, &W2, &b2);
        error = ForwardDigits(&train_set.inputs, &train_set.targets, &W1, &b1, &W2, &b2,
                              &HiddenLinear, &HiddenBias, &Hidden, &OutputLinear, &Pred, &Loss);
        if (error != MATRIX_SUCCESS) {
            printf("Forward failed: %s\n", MatrixErrorMessage(error));
            exit_code = 1;
            goto cleanup;
        }
        error = TensorBackward(&Loss);
        if (error != MATRIX_SUCCESS) {
            printf("Backward failed: %s\n", MatrixErrorMessage(error));
            exit_code = 1;
            goto cleanup;
        }

        if (epoch % 25 == 0) {
            printf("epoch %4d  loss = %.8f\n", epoch, Loss.data[0]);
        }

        StepParameters(&W1, &b1, &W2, &b2);
    }

    {
        Tensor TestHiddenLinear, TestHiddenBias, TestHidden, TestOutputLinear, TestPred, TestLoss;
        TensorInit(&TestHiddenLinear);
        TensorInit(&TestHiddenBias);
        TensorInit(&TestHidden);
        TensorInit(&TestOutputLinear);
        TensorInit(&TestPred);
        TensorInit(&TestLoss);

        if (Require(TensorCreate2D(&TestHiddenLinear, test_set.sample_count, DIGITS_HIDDEN_DIM), "TensorCreate2D(TestHiddenLinear)") &&
            Require(TensorCreate2D(&TestHiddenBias, test_set.sample_count, DIGITS_HIDDEN_DIM), "TensorCreate2D(TestHiddenBias)") &&
            Require(TensorCreate2D(&TestHidden, test_set.sample_count, DIGITS_HIDDEN_DIM), "TensorCreate2D(TestHidden)") &&
            Require(TensorCreate2D(&TestOutputLinear, test_set.sample_count, DIGITS_OUTPUT_DIM), "TensorCreate2D(TestOutputLinear)") &&
            Require(TensorCreate2D(&TestPred, test_set.sample_count, DIGITS_OUTPUT_DIM), "TensorCreate2D(TestPred)") &&
            Require(TensorCreate1D(&TestLoss, 1), "TensorCreate1D(TestLoss)") &&
            Require(ForwardDigits(&test_set.inputs, &test_set.targets, &W1, &b1, &W2, &b2,
                                  &TestHiddenLinear, &TestHiddenBias, &TestHidden, &TestOutputLinear, &TestPred, &TestLoss),
                    "final forward")) {
            int preview_count = test_set.sample_count < 10 ? test_set.sample_count : 10;
            printf("\nPreview predictions\n");
            for (int i = 0; i < preview_count; ++i) {
                printf("sample %d  pred = %d  target = %d\n",
                       i, ArgMaxRow(&TestPred, i), test_set.labels[i]);
            }
            printf("Final loss = %.8f\n", TestLoss.data[0]);
        } else {
            exit_code = 1;
        }

        TensorFree(&TestHiddenLinear);
        TensorFree(&TestHiddenBias);
        TensorFree(&TestHidden);
        TensorFree(&TestOutputLinear);
        TensorFree(&TestPred);
        TensorFree(&TestLoss);
    }

cleanup:
    DigitsDatasetFree(&train_set);
    DigitsDatasetFree(&test_set);
    TensorFree(&W1);
    TensorFree(&b1);
    TensorFree(&W2);
    TensorFree(&b2);
    TensorFree(&HiddenLinear);
    TensorFree(&HiddenBias);
    TensorFree(&Hidden);
    TensorFree(&OutputLinear);
    TensorFree(&Pred);
    TensorFree(&Loss);
    return exit_code;
}
