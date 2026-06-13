#include "tensor.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void TensorInit(Tensor *T)
{
    if (T == NULL) {
        return;
    }
    T->ndim = 0;
    T->size = 0;
    T->data = NULL;
    T->grad = NULL;
    T->requires_grad = 0;
    T->op = TENSOR_OP_NONE;
    T->left = NULL;
    T->right = NULL;
    for (int i = 0; i < TENSOR_MAX_DIMS; ++i) {
        T->shape[i] = 0;
        T->stride[i] = 0;
    }
}

int TensorIsValid(const Tensor *T)
{
    return T != NULL && T->ndim > 0 && T->size > 0 && T->data != NULL;
}

static MatrixError TensorComputeSizeAndStride(Tensor *T, int ndim, const int *shape)
{
    size_t total = 1;

    if (T == NULL || shape == NULL) {
        return MATRIX_ERROR_NULL_POINTER;
    }
    if (ndim <= 0 || ndim > 2) {
        return MATRIX_ERROR_INVALID_SIZE;
    }

    for (int i = 0; i < ndim; ++i) {
        if (shape[i] <= 0) {
            return MATRIX_ERROR_INVALID_SIZE;
        }
        if (total > SIZE_MAX / (size_t)shape[i]) {
            return MATRIX_ERROR_SIZE_OVERFLOW;
        }
        total *= (size_t)shape[i];
    }
    if (total > (size_t)INT32_MAX || total > SIZE_MAX / sizeof(REAL)) {
        return MATRIX_ERROR_SIZE_OVERFLOW;
    }

    T->ndim = ndim;
    T->size = (int)total;
    for (int i = 0; i < ndim; ++i) {
        T->shape[i] = shape[i];
    }

    T->stride[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) {
        T->stride[i] = T->stride[i + 1] * T->shape[i + 1];
    }
    return MATRIX_SUCCESS;
}

MatrixError TensorCreate(Tensor *T, int ndim, const int *shape)
{
    MatrixError error;

    if (T == NULL) {
        return MATRIX_ERROR_NULL_POINTER;
    }
    if (T->data != NULL || T->grad != NULL) {
        return MATRIX_ERROR_ALREADY_ALLOCATED;
    }

    TensorInit(T);
    error = TensorComputeSizeAndStride(T, ndim, shape);
    if (error != MATRIX_SUCCESS) {
        TensorInit(T);
        return error;
    }

    T->data = (REAL *)malloc((size_t)T->size * sizeof(REAL));
    if (T->data == NULL) {
        TensorInit(T);
        return MATRIX_ERROR_ALLOC_FAILED;
    }
    return MATRIX_SUCCESS;
}

MatrixError TensorCreate1D(Tensor *T, int length)
{
    int shape[] = {length};
    return TensorCreate(T, 1, shape);
}

MatrixError TensorCreate2D(Tensor *T, int row, int column)
{
    int shape[] = {row, column};
    return TensorCreate(T, 2, shape);
}

void TensorFree(Tensor *T)
{
    if (T == NULL) {
        return;
    }
    free(T->data);
    free(T->grad);
    TensorInit(T);
}

void TensorSetRequiresGrad(Tensor *T, int requires_grad)
{
    if (T == NULL) {
        return;
    }
    T->requires_grad = requires_grad ? 1 : 0;
    if (T->requires_grad) {
        (void)TensorEnsureGrad(T);
    }
}

MatrixError TensorEnsureGrad(Tensor *T)
{
    if (!TensorIsValid(T)) {
        return MATRIX_ERROR_NULL_POINTER;
    }
    if (T->grad != NULL) {
        return MATRIX_SUCCESS;
    }

    T->grad = (REAL *)calloc((size_t)T->size, sizeof(REAL));
    if (T->grad == NULL) {
        return MATRIX_ERROR_ALLOC_FAILED;
    }
    return MATRIX_SUCCESS;
}

MatrixError TensorZeroGrad(Tensor *T)
{
    if (!TensorIsValid(T)) {
        return MATRIX_ERROR_NULL_POINTER;
    }
    if (T->grad == NULL) {
        return MATRIX_SUCCESS;
    }
    for (int i = 0; i < T->size; ++i) {
        T->grad[i] = 0.0;
    }
    return MATRIX_SUCCESS;
}

MatrixError TensorSGDStep(Tensor *T, REAL learning_rate)
{
    if (!TensorIsValid(T)) {
        return MATRIX_ERROR_NULL_POINTER;
    }
    if (learning_rate < 0.0) {
        return MATRIX_ERROR_INVALID_ARGUMENT;
    }
    if (!T->requires_grad) {
        return MATRIX_SUCCESS;
    }
    if (T->grad == NULL) {
        return MATRIX_ERROR_NULL_POINTER;
    }

    for (int i = 0; i < T->size; ++i) {
        T->data[i] -= learning_rate * T->grad[i];
    }
    return MATRIX_SUCCESS;
}

MatrixError TensorFillZero(Tensor *T)
{
    if (!TensorIsValid(T)) {
        return MATRIX_ERROR_NULL_POINTER;
    }
    for (int i = 0; i < T->size; ++i) {
        T->data[i] = 0.0;
    }
    return MATRIX_SUCCESS;
}

MatrixError TensorFillSequence(Tensor *T, REAL start, REAL step)
{
    if (!TensorIsValid(T)) {
        return MATRIX_ERROR_NULL_POINTER;
    }
    for (int i = 0; i < T->size; ++i) {
        T->data[i] = start + step * (REAL)i;
    }
    return MATRIX_SUCCESS;
}

MatrixError TensorSet2D(Tensor *T, int i, int j, REAL value)
{
    int index;

    if (!TensorIsValid(T)) {
        return MATRIX_ERROR_NULL_POINTER;
    }
    if (T->ndim != 2) {
        return MATRIX_ERROR_INVALID_SIZE;
    }
    if (i < 0 || i >= T->shape[0] || j < 0 || j >= T->shape[1]) {
        return MATRIX_ERROR_INDEX_OUT_OF_RANGE;
    }

    index = i * T->stride[0] + j * T->stride[1];
    T->data[index] = value;
    return MATRIX_SUCCESS;
}

MatrixError TensorGet2D(const Tensor *T, int i, int j, REAL *value)
{
    int index;

    if (!TensorIsValid(T) || value == NULL) {
        return MATRIX_ERROR_NULL_POINTER;
    }
    if (T->ndim != 2) {
        return MATRIX_ERROR_INVALID_SIZE;
    }
    if (i < 0 || i >= T->shape[0] || j < 0 || j >= T->shape[1]) {
        return MATRIX_ERROR_INDEX_OUT_OF_RANGE;
    }

    index = i * T->stride[0] + j * T->stride[1];
    *value = T->data[index];
    return MATRIX_SUCCESS;
}

void TensorPrint(const Tensor *T, const char *name)
{
    if (name == NULL) {
        name = "Tensor";
    }
    if (!TensorIsValid(T)) {
        printf("%s is an empty or invalid tensor.\n", name);
        return;
    }

    if (T->ndim == 1) {
        printf("%s = [%d]\n  ", name, T->shape[0]);
        for (int i = 0; i < T->shape[0]; ++i) {
            printf("%10.4f ", T->data[i]);
        }
        printf("\n");
        return;
    }

    if (T->ndim == 2) {
        printf("%s = [%d x %d]\n", name, T->shape[0], T->shape[1]);
        for (int i = 0; i < T->shape[0]; ++i) {
            printf("  ");
            for (int j = 0; j < T->shape[1]; ++j) {
                printf("%10.4f ", T->data[i * T->stride[0] + j * T->stride[1]]);
            }
            printf("\n");
        }
        return;
    }

    printf("%s has unsupported ndim = %d for printing.\n", name, T->ndim);
}

void TensorPrintGrad(const Tensor *T, const char *name)
{
    if (name == NULL) {
        name = "Tensor.grad";
    }
    if (!TensorIsValid(T) || T->grad == NULL) {
        printf("%s is an empty or invalid tensor gradient.\n", name);
        return;
    }

    if (T->ndim == 1) {
        printf("%s = [%d]\n  ", name, T->shape[0]);
        for (int i = 0; i < T->shape[0]; ++i) {
            printf("%10.4f ", T->grad[i]);
        }
        printf("\n");
        return;
    }

    if (T->ndim == 2) {
        printf("%s = [%d x %d]\n", name, T->shape[0], T->shape[1]);
        for (int i = 0; i < T->shape[0]; ++i) {
            printf("  ");
            for (int j = 0; j < T->shape[1]; ++j) {
                printf("%10.4f ", T->grad[i * T->stride[0] + j * T->stride[1]]);
            }
            printf("\n");
        }
        return;
    }

    printf("%s has unsupported ndim = %d for printing.\n", name, T->ndim);
}
