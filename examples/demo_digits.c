#include "backend.h"
#include "digits_dataset.h"
#include "tensor.h"
#include "tensor_ops.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DIGITS_INPUT_DIM 784
#define DIGITS_HIDDEN_DIM 128
#define DIGITS_OUTPUT_DIM 10
#define DIGITS_DEFAULT_EPOCHS 10
#define DIGITS_DEFAULT_SAVE_INTERVAL 10
#define DIGITS_LEARNING_RATE 0.05
#define DIGITS_MAX_PATH_LENGTH 1024

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

static void ZeroTensorGradIfPresent(Tensor *T)
{
    if (T != NULL) {
        (void)TensorZeroGrad(T);
    }
}

static void ZeroTrainingGrads(Tensor *W1,
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
    ZeroTensorGradIfPresent(W1);
    ZeroTensorGradIfPresent(b1);
    ZeroTensorGradIfPresent(W2);
    ZeroTensorGradIfPresent(b2);
    ZeroTensorGradIfPresent(HiddenLinear);
    ZeroTensorGradIfPresent(HiddenBias);
    ZeroTensorGradIfPresent(Hidden);
    ZeroTensorGradIfPresent(OutputLinear);
    ZeroTensorGradIfPresent(Pred);
    ZeroTensorGradIfPresent(Loss);
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

static double ComputeAccuracyPercent(const Tensor *Pred, const int *labels, int sample_count)
{
    int correct_count = 0;

    for (int i = 0; i < sample_count; ++i) {
        if (ArgMaxRow(Pred, i) == labels[i]) {
            ++correct_count;
        }
    }

    return 100.0 * (double)correct_count / (double)sample_count;
}

static MatrixError EvaluateDigits(Tensor *X,
                                  Tensor *Target,
                                  const int *labels,
                                  int sample_count,
                                  Tensor *W1,
                                  Tensor *b1,
                                  Tensor *W2,
                                  Tensor *b2,
                                  Tensor *HiddenLinear,
                                  Tensor *HiddenBias,
                                  Tensor *Hidden,
                                  Tensor *OutputLinear,
                                  Tensor *Pred,
                                  Tensor *Loss,
                                  double *loss_out,
                                  double *accuracy_out)
{
    MatrixError error = ForwardDigits(X, Target, W1, b1, W2, b2,
                                      HiddenLinear, HiddenBias, Hidden, OutputLinear, Pred, Loss);
    if (error != MATRIX_SUCCESS) {
        return error;
    }

    if (loss_out != NULL) {
        *loss_out = Loss->data[0];
    }
    if (accuracy_out != NULL) {
        *accuracy_out = ComputeAccuracyPercent(Pred, labels, sample_count);
    }
    return MATRIX_SUCCESS;
}

static MatrixError SaveTensorValues(FILE *file, const char *name, const Tensor *T)
{
    if (file == NULL || name == NULL || !TensorIsValid(T)) {
        return MATRIX_ERROR_NULL_POINTER;
    }

    fprintf(file, "%s %d %d\n", name, T->shape[0], T->ndim == 1 ? 1 : T->shape[1]);
    for (int i = 0; i < T->size; ++i) {
        fprintf(file, "%.17g\n", T->data[i]);
    }
    return MATRIX_SUCCESS;
}

static MatrixError LoadTensorValues(FILE *file, const char *expected_name, Tensor *T)
{
    char name_buffer[64];
    int row;
    int column;

    if (file == NULL || expected_name == NULL || !TensorIsValid(T)) {
        return MATRIX_ERROR_NULL_POINTER;
    }

    if (fscanf(file, "%63s %d %d", name_buffer, &row, &column) != 3) {
        return MATRIX_ERROR_INVALID_ARGUMENT;
    }
    if (strcmp(name_buffer, expected_name) != 0) {
        return MATRIX_ERROR_INVALID_ARGUMENT;
    }
    if ((T->ndim == 1 && (row != T->shape[0] || column != 1)) ||
        (T->ndim == 2 && (row != T->shape[0] || column != T->shape[1]))) {
        return MATRIX_ERROR_SIZE_MISMATCH;
    }

    for (int i = 0; i < T->size; ++i) {
        if (fscanf(file, "%lf", &T->data[i]) != 1) {
            return MATRIX_ERROR_INVALID_ARGUMENT;
        }
    }
    return MATRIX_SUCCESS;
}

static MatrixError SaveCheckpoint(const char *checkpoint_dir,
                                  int epoch,
                                  const Tensor *W1,
                                  const Tensor *b1,
                                  const Tensor *W2,
                                  const Tensor *b2)
{
    char checkpoint_path[DIGITS_MAX_PATH_LENGTH];
    FILE *file;

    if (checkpoint_dir == NULL) {
        return MATRIX_ERROR_NULL_POINTER;
    }

    snprintf(checkpoint_path, sizeof(checkpoint_path), "%s/checkpoint_epoch_%04d.txt", checkpoint_dir, epoch);
    file = fopen(checkpoint_path, "w");
    if (file == NULL) {
        return MATRIX_ERROR_INVALID_ARGUMENT;
    }

    fprintf(file, "epoch %d\n", epoch);
    if (SaveTensorValues(file, "W1", W1) != MATRIX_SUCCESS ||
        SaveTensorValues(file, "b1", b1) != MATRIX_SUCCESS ||
        SaveTensorValues(file, "W2", W2) != MATRIX_SUCCESS ||
        SaveTensorValues(file, "b2", b2) != MATRIX_SUCCESS) {
        fclose(file);
        return MATRIX_ERROR_INVALID_ARGUMENT;
    }

    fclose(file);
    return MATRIX_SUCCESS;
}

static MatrixError LoadCheckpoint(const char *checkpoint_path,
                                  Tensor *W1,
                                  Tensor *b1,
                                  Tensor *W2,
                                  Tensor *b2,
                                  int *loaded_epoch)
{
    FILE *file;
    char epoch_tag[32];
    int epoch_value;
    MatrixError error;

    if (checkpoint_path == NULL) {
        return MATRIX_ERROR_NULL_POINTER;
    }

    file = fopen(checkpoint_path, "r");
    if (file == NULL) {
        return MATRIX_ERROR_INVALID_ARGUMENT;
    }

    if (fscanf(file, "%31s %d", epoch_tag, &epoch_value) != 2 || strcmp(epoch_tag, "epoch") != 0) {
        fclose(file);
        return MATRIX_ERROR_INVALID_ARGUMENT;
    }

    error = LoadTensorValues(file, "W1", W1);
    if (error == MATRIX_SUCCESS) {
        error = LoadTensorValues(file, "b1", b1);
    }
    if (error == MATRIX_SUCCESS) {
        error = LoadTensorValues(file, "W2", W2);
    }
    if (error == MATRIX_SUCCESS) {
        error = LoadTensorValues(file, "b2", b2);
    }

    fclose(file);
    if (error != MATRIX_SUCCESS) {
        return error;
    }

    if (loaded_epoch != NULL) {
        *loaded_epoch = epoch_value;
    }
    return MATRIX_SUCCESS;
}

int main(int argc, char **argv)
{
    const char *train_path = "data/mnist_train.csv";
    const char *test_path = "data/mnist_test.csv";
    const char *metrics_path = "artifacts/digits/c_mnist_metrics.csv";
    const char *checkpoint_dir = "artifacts/digits/checkpoints";
    const char *load_checkpoint_path = NULL;
    int epochs = DIGITS_DEFAULT_EPOCHS;
    int save_interval = DIGITS_DEFAULT_SAVE_INTERVAL;
    int inference_only = 0;
    int exit_code = 0;
    int loaded_epoch = 0;
    FILE *metrics_file = NULL;
    DigitsDataset train_set, test_set;
    Tensor W1, b1, W2, b2, HiddenLinear, HiddenBias, Hidden, OutputLinear, Pred, Loss;
    Tensor TestHiddenLinear, TestHiddenBias, TestHidden, TestOutputLinear, TestPred, TestLoss;

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
    if (argc >= 5) {
        metrics_path = argv[4];
    }
    if (argc >= 6) {
        checkpoint_dir = argv[5];
    }
    if (argc >= 7) {
        save_interval = atoi(argv[6]);
        if (save_interval <= 0) {
            save_interval = DIGITS_DEFAULT_SAVE_INTERVAL;
        }
    }
    if (argc >= 8 && argv[7][0] != '\0') {
        load_checkpoint_path = argv[7];
    }
    if (argc >= 9) {
        inference_only = atoi(argv[8]) != 0;
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
    TensorInit(&TestHiddenLinear);
    TensorInit(&TestHiddenBias);
    TensorInit(&TestHidden);
    TensorInit(&TestOutputLinear);
    TensorInit(&TestPred);
    TensorInit(&TestLoss);

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
        !Require(TensorCreate1D(&Loss, 1), "TensorCreate1D(Loss)") ||
        !Require(TensorCreate2D(&TestHiddenLinear, test_set.sample_count, DIGITS_HIDDEN_DIM), "TensorCreate2D(TestHiddenLinear)") ||
        !Require(TensorCreate2D(&TestHiddenBias, test_set.sample_count, DIGITS_HIDDEN_DIM), "TensorCreate2D(TestHiddenBias)") ||
        !Require(TensorCreate2D(&TestHidden, test_set.sample_count, DIGITS_HIDDEN_DIM), "TensorCreate2D(TestHidden)") ||
        !Require(TensorCreate2D(&TestOutputLinear, test_set.sample_count, DIGITS_OUTPUT_DIM), "TensorCreate2D(TestOutputLinear)") ||
        !Require(TensorCreate2D(&TestPred, test_set.sample_count, DIGITS_OUTPUT_DIM), "TensorCreate2D(TestPred)") ||
        !Require(TensorCreate1D(&TestLoss, 1), "TensorCreate1D(TestLoss)")) {
        goto cleanup;
    }

    TensorSetRequiresGrad(&W1, 1);
    TensorSetRequiresGrad(&b1, 1);
    TensorSetRequiresGrad(&W2, 1);
    TensorSetRequiresGrad(&b2, 1);
    BackendSetType(BACKEND_SIMD);

    InitDigitsParameters(&W1, &b1, &W2, &b2);
    if (load_checkpoint_path != NULL) {
        if (!Require(LoadCheckpoint(load_checkpoint_path, &W1, &b1, &W2, &b2, &loaded_epoch), "LoadCheckpoint")) {
            exit_code = 1;
            goto cleanup;
        }
    }

    if (metrics_path != NULL) {
        metrics_file = fopen(metrics_path, "w");
        if (metrics_file == NULL) {
            printf("Failed to open metrics file: %s\n", metrics_path);
            exit_code = 1;
            goto cleanup;
        }
        fprintf(metrics_file, "epoch,train_loss,train_acc,test_loss,test_acc,epoch_seconds\n");
    }

    printf("Digits MLP demo\n");
    printf("===============\n");
    printf("Train: %s\n", train_path);
    printf("Test:  %s\n", test_path);
    printf("Network: 784 -> 128 -> 10\n");
    printf("Loss: MSE + one-hot\n");
    printf("Matmul kernel: auto\n");
    printf("Mode: %s\n", inference_only ? "inference-only" : "train");
    printf("Epochs: %d\n", epochs);
    printf("Save interval: %d\n", save_interval);
    printf("Checkpoint dir: %s\n", checkpoint_dir);
    if (load_checkpoint_path != NULL) {
        printf("Loaded checkpoint: %s (epoch %d)\n", load_checkpoint_path, loaded_epoch);
    }
    printf("\n");

    if (inference_only) {
        double train_loss;
        double train_acc;
        double test_loss;
        double test_acc;
        MatrixError error;

        if (load_checkpoint_path == NULL) {
            printf("Inference-only mode requires a checkpoint path.\n");
            exit_code = 1;
            goto cleanup;
        }

        error = EvaluateDigits(&train_set.inputs, &train_set.targets, train_set.labels, train_set.sample_count,
                               &W1, &b1, &W2, &b2,
                               &HiddenLinear, &HiddenBias, &Hidden, &OutputLinear, &Pred, &Loss,
                               &train_loss, &train_acc);
        if (error != MATRIX_SUCCESS) {
            printf("Train-set evaluation failed: %s\n", MatrixErrorMessage(error));
            exit_code = 1;
            goto cleanup;
        }

        error = EvaluateDigits(&test_set.inputs, &test_set.targets, test_set.labels, test_set.sample_count,
                               &W1, &b1, &W2, &b2,
                               &TestHiddenLinear, &TestHiddenBias, &TestHidden, &TestOutputLinear, &TestPred, &TestLoss,
                               &test_loss, &test_acc);
        if (error != MATRIX_SUCCESS) {
            printf("Test-set evaluation failed: %s\n", MatrixErrorMessage(error));
            exit_code = 1;
            goto cleanup;
        }

        printf("checkpoint %4d  train_loss = %.8f  train_acc = %.2f%%  test_loss = %.8f  test_acc = %.2f%%\n",
               loaded_epoch, train_loss, train_acc, test_loss, test_acc);
        if (metrics_file != NULL) {
            fprintf(metrics_file, "%d,%.10f,%.6f,%.10f,%.6f,%.6f\n",
                    loaded_epoch, train_loss, train_acc, test_loss, test_acc, 0.0);
            fflush(metrics_file);
        }
    } else {
        for (int epoch = 1; epoch <= epochs; ++epoch) {
        MatrixError error;
        int absolute_epoch = loaded_epoch + epoch;
        double train_loss;
        double train_acc;
        double test_loss;
        double test_acc;
        double epoch_seconds;
        int should_save;
        clock_t start_clock = clock();

        ZeroTrainingGrads(&W1, &b1, &W2, &b2,
                          &HiddenLinear, &HiddenBias, &Hidden, &OutputLinear, &Pred, &Loss);
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

        train_loss = Loss.data[0];
        train_acc = ComputeAccuracyPercent(&Pred, train_set.labels, train_set.sample_count);

        StepParameters(&W1, &b1, &W2, &b2);

        error = EvaluateDigits(&test_set.inputs, &test_set.targets, test_set.labels, test_set.sample_count,
                               &W1, &b1, &W2, &b2,
                               &TestHiddenLinear, &TestHiddenBias, &TestHidden, &TestOutputLinear, &TestPred, &TestLoss,
                               &test_loss, &test_acc);
        if (error != MATRIX_SUCCESS) {
            printf("Test evaluation failed: %s\n", MatrixErrorMessage(error));
            exit_code = 1;
            goto cleanup;
        }

        epoch_seconds = (double)(clock() - start_clock) / (double)CLOCKS_PER_SEC;
        printf("epoch %4d  train_loss = %.8f  train_acc = %.2f%%  test_loss = %.8f  test_acc = %.2f%%  seconds = %.3f\n",
               absolute_epoch, train_loss, train_acc, test_loss, test_acc, epoch_seconds);

        if (metrics_file != NULL) {
            fprintf(metrics_file, "%d,%.10f,%.6f,%.10f,%.6f,%.6f\n",
                    absolute_epoch, train_loss, train_acc, test_loss, test_acc, epoch_seconds);
            fflush(metrics_file);
        }

        should_save = ((absolute_epoch % save_interval) == 0) || (epoch == epochs);
        if (should_save) {
            error = SaveCheckpoint(checkpoint_dir, absolute_epoch, &W1, &b1, &W2, &b2);
            if (error != MATRIX_SUCCESS) {
                printf("SaveCheckpoint failed at epoch %d: %s\n", absolute_epoch, MatrixErrorMessage(error));
                exit_code = 1;
                goto cleanup;
            }
        }
    }
    }

    {
        int preview_count = test_set.sample_count < 10 ? test_set.sample_count : 10;
        printf("\nPreview predictions\n");
        for (int i = 0; i < preview_count; ++i) {
            int sample_index = (i * test_set.sample_count) / preview_count;
            printf("sample %d  pred = %d  target = %d\n",
                   sample_index, ArgMaxRow(&TestPred, sample_index), test_set.labels[sample_index]);
        }
        printf("Final test accuracy = %.2f%%\n",
               ComputeAccuracyPercent(&TestPred, test_set.labels, test_set.sample_count));
        printf("Final test loss = %.8f\n", TestLoss.data[0]);
        if (metrics_path != NULL) {
            printf("Metrics CSV: %s\n", metrics_path);
        }
        printf("Checkpoint directory: %s\n", checkpoint_dir);
    }

cleanup:
    if (metrics_file != NULL) {
        fclose(metrics_file);
    }
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
    TensorFree(&TestHiddenLinear);
    TensorFree(&TestHiddenBias);
    TensorFree(&TestHidden);
    TensorFree(&TestOutputLinear);
    TensorFree(&TestPred);
    TensorFree(&TestLoss);
    return exit_code;
}
