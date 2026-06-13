#include "tensor_ops.h"

#include "matrix_optimized.h"

#define TENSOR_BLOCK_SIZE 32

static MatrixError CheckSameShape(const Tensor *A, const Tensor *B, const Tensor *C)
{
    if (!TensorIsValid(A) || !TensorIsValid(B) || !TensorIsValid(C)) {
        return MATRIX_ERROR_NULL_POINTER;
    }
    if (A->ndim != B->ndim || A->ndim != C->ndim) {
        return MATRIX_ERROR_SIZE_MISMATCH;
    }
    for (int i = 0; i < A->ndim; ++i) {
        if (A->shape[i] != B->shape[i] || A->shape[i] != C->shape[i]) {
            return MATRIX_ERROR_SIZE_MISMATCH;
        }
    }
    return MATRIX_SUCCESS;
}

static Matrix MatrixViewFromTensor2D(const Tensor *T)
{
    Matrix view;

    view.row = T->shape[0];
    view.column = T->shape[1];
    view.data = T->data;
    return view;
}

static Matrix MatrixGradViewFromTensor2D(Tensor *T)
{
    Matrix view;

    view.row = T->shape[0];
    view.column = T->shape[1];
    view.data = T->grad;
    return view;
}

MatrixError TensorAdd(const Tensor *A, const Tensor *B, Tensor *C)
{
    MatrixError error = CheckSameShape(A, B, C);

    if (error != MATRIX_SUCCESS) {
        return error;
    }

    for (int i = 0; i < A->size; ++i) {
        C->data[i] = A->data[i] + B->data[i];
    }
    return MATRIX_SUCCESS;
}

MatrixError TensorScale(REAL alpha, const Tensor *A, Tensor *B)
{
    if (!TensorIsValid(A) || !TensorIsValid(B)) {
        return MATRIX_ERROR_NULL_POINTER;
    }
    if (A->ndim != B->ndim || A->size != B->size) {
        return MATRIX_ERROR_SIZE_MISMATCH;
    }
    for (int i = 0; i < A->ndim; ++i) {
        if (A->shape[i] != B->shape[i]) {
            return MATRIX_ERROR_SIZE_MISMATCH;
        }
    }

    for (int i = 0; i < A->size; ++i) {
        B->data[i] = alpha * A->data[i];
    }
    return MATRIX_SUCCESS;
}

MatrixError TensorMatmul(const Tensor *A, const Tensor *B, Tensor *C)
{
    Matrix A_view;
    Matrix B_view;
    Matrix C_view;
    BackendType backend;

    if (!TensorIsValid(A) || !TensorIsValid(B) || !TensorIsValid(C)) {
        return MATRIX_ERROR_NULL_POINTER;
    }
    if (A->ndim != 2 || B->ndim != 2 || C->ndim != 2) {
        return MATRIX_ERROR_INVALID_SIZE;
    }
    if (A->shape[1] != B->shape[0] ||
        C->shape[0] != A->shape[0] ||
        C->shape[1] != B->shape[1]) {
        return MATRIX_ERROR_SIZE_MISMATCH;
    }

    A_view = MatrixViewFromTensor2D(A);
    B_view = MatrixViewFromTensor2D(B);
    C_view = MatrixViewFromTensor2D(C);
    backend = BackendGetType();

    (void)backend;
    return MatrixMultiplyAuto(&A_view, &B_view, &C_view);
}

static void TensorAttachOp(Tensor *Out, TensorOp op, Tensor *Left, Tensor *Right)
{
    Out->op = op;
    Out->left = Left;
    Out->right = Right;
    Out->requires_grad = ((Left != NULL && Left->requires_grad) ||
                          (Right != NULL && Right->requires_grad));
    if (Out->requires_grad) {
        (void)TensorEnsureGrad(Out);
    }
    TensorZeroGrad(Out);
}

MatrixError TensorAddAuto(Tensor *A, Tensor *B, Tensor *C)
{
    MatrixError error = TensorAdd(A, B, C);

    if (error != MATRIX_SUCCESS) {
        return error;
    }
    TensorAttachOp(C, TENSOR_OP_ADD, A, B);
    return MATRIX_SUCCESS;
}

MatrixError TensorAddBiasAuto(Tensor *A, Tensor *Bias, Tensor *C)
{
    int out_dim;

    if (!TensorIsValid(A) || !TensorIsValid(Bias) || !TensorIsValid(C)) {
        return MATRIX_ERROR_NULL_POINTER;
    }
    if (A->ndim != 2 || C->ndim != 2) {
        return MATRIX_ERROR_INVALID_SIZE;
    }
    if (A->shape[0] != C->shape[0] || A->shape[1] != C->shape[1]) {
        return MATRIX_ERROR_SIZE_MISMATCH;
    }

    out_dim = A->shape[1];
    if (!((Bias->ndim == 1 && Bias->shape[0] == out_dim) ||
          (Bias->ndim == 2 && Bias->shape[0] == 1 && Bias->shape[1] == out_dim))) {
        return MATRIX_ERROR_SIZE_MISMATCH;
    }

    for (int i = 0; i < A->shape[0]; ++i) {
        int row = i * A->stride[0];
        int c_row = i * C->stride[0];
        for (int j = 0; j < out_dim; ++j) {
            C->data[c_row + j] = A->data[row + j] + Bias->data[j];
        }
    }

    TensorAttachOp(C, TENSOR_OP_ADD_BIAS, A, Bias);
    return MATRIX_SUCCESS;
}

MatrixError TensorMatmulAuto(Tensor *A, Tensor *B, Tensor *C)
{
    MatrixError error = TensorMatmul(A, B, C);

    if (error != MATRIX_SUCCESS) {
        return error;
    }
    TensorAttachOp(C, TENSOR_OP_MATMUL, A, B);
    return MATRIX_SUCCESS;
}

MatrixError TensorReluAuto(Tensor *A, Tensor *Y)
{
    if (!TensorIsValid(A) || !TensorIsValid(Y)) {
        return MATRIX_ERROR_NULL_POINTER;
    }
    if (A->ndim != Y->ndim || A->size != Y->size) {
        return MATRIX_ERROR_SIZE_MISMATCH;
    }
    for (int i = 0; i < A->ndim; ++i) {
        if (A->shape[i] != Y->shape[i]) {
            return MATRIX_ERROR_SIZE_MISMATCH;
        }
    }

    for (int i = 0; i < A->size; ++i) {
        Y->data[i] = A->data[i] > 0.0 ? A->data[i] : 0.0;
    }

    TensorAttachOp(Y, TENSOR_OP_RELU, A, NULL);
    return MATRIX_SUCCESS;
}

MatrixError TensorMSELossAuto(Tensor *Prediction, Tensor *Target, Tensor *Loss)
{
    REAL sum = 0.0;

    if (!TensorIsValid(Prediction) || !TensorIsValid(Target) || !TensorIsValid(Loss)) {
        return MATRIX_ERROR_NULL_POINTER;
    }
    if (Loss->size != 1) {
        return MATRIX_ERROR_INVALID_SIZE;
    }
    if (Prediction->ndim != Target->ndim || Prediction->size != Target->size) {
        return MATRIX_ERROR_SIZE_MISMATCH;
    }
    for (int i = 0; i < Prediction->ndim; ++i) {
        if (Prediction->shape[i] != Target->shape[i]) {
            return MATRIX_ERROR_SIZE_MISMATCH;
        }
    }

    for (int i = 0; i < Prediction->size; ++i) {
        REAL diff = Prediction->data[i] - Target->data[i];
        sum += diff * diff;
    }
    Loss->data[0] = sum / (REAL)Prediction->size;

    TensorAttachOp(Loss, TENSOR_OP_MSE, Prediction, Target);
    return MATRIX_SUCCESS;
}

static void BackwardAdd(Tensor *T)
{
    Tensor *A = T->left;
    Tensor *B = T->right;

    for (int i = 0; i < T->size; ++i) {
        if (A->requires_grad) {
            A->grad[i] += T->grad[i];
        }
        if (B->requires_grad) {
            B->grad[i] += T->grad[i];
        }
    }
}

static void BackwardAddBias(Tensor *T)
{
    Tensor *A = T->left;
    Tensor *Bias = T->right;
    int batch = T->shape[0];
    int out_dim = T->shape[1];

    for (int i = 0; i < batch; ++i) {
        int row = i * T->stride[0];
        for (int j = 0; j < out_dim; ++j) {
            REAL grad = T->grad[row + j];
            if (A->requires_grad) {
                A->grad[row + j] += grad;
            }
            if (Bias->requires_grad) {
                Bias->grad[j] += grad;
            }
        }
    }
}

static void BackwardRelu(Tensor *T)
{
    Tensor *A = T->left;

    for (int i = 0; i < T->size; ++i) {
        if (A->requires_grad) {
            A->grad[i] += A->data[i] > 0.0 ? T->grad[i] : 0.0;
        }
    }
}

static void BackwardMSE(Tensor *T)
{
    Tensor *Prediction = T->left;
    Tensor *Target = T->right;
    REAL scale = 2.0 * T->grad[0] / (REAL)Prediction->size;

    for (int i = 0; i < Prediction->size; ++i) {
        REAL grad = scale * (Prediction->data[i] - Target->data[i]);
        if (Prediction->requires_grad) {
            Prediction->grad[i] += grad;
        }
        if (Target->requires_grad) {
            Target->grad[i] -= grad;
        }
    }
}

static MatrixError BackwardMatmul(Tensor *T)
{
    Tensor *A = T->left;
    Tensor *B = T->right;
    Matrix A_view = MatrixViewFromTensor2D(A);
    Matrix B_view = MatrixViewFromTensor2D(B);
    Matrix T_grad_view = MatrixGradViewFromTensor2D(T);
    Matrix A_grad_view = MatrixGradViewFromTensor2D(A);
    Matrix B_grad_view = MatrixGradViewFromTensor2D(B);
    Matrix AT;
    Matrix BT;
    Matrix temp;
    MatrixError error = MATRIX_SUCCESS;

    MatrixInit(&AT);
    MatrixInit(&BT);
    MatrixInit(&temp);

    if (A->requires_grad) {
        error = MatrixCreate(&BT, B->shape[1], B->shape[0]);
        if (error != MATRIX_SUCCESS) {
            goto cleanup;
        }
        error = MatrixTranspose(&B_view, &BT);
        if (error != MATRIX_SUCCESS) {
            goto cleanup;
        }
        error = MatrixCreate(&temp, A->shape[0], A->shape[1]);
        if (error != MATRIX_SUCCESS) {
            goto cleanup;
        }
        error = MatrixMultiplyAuto(&T_grad_view, &BT, &temp);
        if (error != MATRIX_SUCCESS) {
            goto cleanup;
        }
        error = MatrixAddInPlace(&A_grad_view, &temp);
        if (error != MATRIX_SUCCESS) {
            goto cleanup;
        }
        MatrixFree(&BT);
        MatrixInit(&BT);
        MatrixFree(&temp);
        MatrixInit(&temp);
    }

    if (B->requires_grad) {
        error = MatrixCreate(&AT, A->shape[1], A->shape[0]);
        if (error != MATRIX_SUCCESS) {
            goto cleanup;
        }
        error = MatrixTranspose(&A_view, &AT);
        if (error != MATRIX_SUCCESS) {
            goto cleanup;
        }
        error = MatrixCreate(&temp, B->shape[0], B->shape[1]);
        if (error != MATRIX_SUCCESS) {
            goto cleanup;
        }
        error = MatrixMultiplyAuto(&AT, &T_grad_view, &temp);
        if (error != MATRIX_SUCCESS) {
            goto cleanup;
        }
        error = MatrixAddInPlace(&B_grad_view, &temp);
        if (error != MATRIX_SUCCESS) {
            goto cleanup;
        }
    }

cleanup:
    MatrixFree(&AT);
    MatrixFree(&BT);
    MatrixFree(&temp);
    return error;
}

static MatrixError TensorBackwardInternal(Tensor *T)
{
    if (!TensorIsValid(T)) {
        return MATRIX_ERROR_NULL_POINTER;
    }
    if (T->op == TENSOR_OP_NONE) {
        return MATRIX_SUCCESS;
    }

    if (T->op == TENSOR_OP_ADD) {
        BackwardAdd(T);
    } else if (T->op == TENSOR_OP_ADD_BIAS) {
        BackwardAddBias(T);
    } else if (T->op == TENSOR_OP_MATMUL) {
        MatrixError error = BackwardMatmul(T);
        if (error != MATRIX_SUCCESS) {
            return error;
        }
    } else if (T->op == TENSOR_OP_RELU) {
        BackwardRelu(T);
    } else if (T->op == TENSOR_OP_MSE) {
        BackwardMSE(T);
    } else {
        return MATRIX_ERROR_INVALID_ARGUMENT;
    }

    if (T->left != NULL && T->left->requires_grad) {
        MatrixError error = TensorBackwardInternal(T->left);
        if (error != MATRIX_SUCCESS) {
            return error;
        }
    }
    if (T->right != NULL && T->right->requires_grad) {
        MatrixError error = TensorBackwardInternal(T->right);
        if (error != MATRIX_SUCCESS) {
            return error;
        }
    }
    return MATRIX_SUCCESS;
}

MatrixError TensorBackward(Tensor *Loss)
{
    if (!TensorIsValid(Loss)) {
        return MATRIX_ERROR_NULL_POINTER;
    }
    if (Loss->size != 1) {
        return MATRIX_ERROR_INVALID_SIZE;
    }

    Loss->grad[0] = 1.0;
    return TensorBackwardInternal(Loss);
}
