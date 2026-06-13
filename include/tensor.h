#ifndef TENSOR_H
#define TENSOR_H

#include "matrix.h"

#define TENSOR_MAX_DIMS 4

typedef enum {
    TENSOR_OP_NONE = 0,
    TENSOR_OP_ADD = 1,
    TENSOR_OP_ADD_BIAS = 2,
    TENSOR_OP_MATMUL = 3,
    TENSOR_OP_RELU = 4,
    TENSOR_OP_MSE = 5
} TensorOp;

typedef struct Tensor {
    int ndim;
    int shape[TENSOR_MAX_DIMS];
    int stride[TENSOR_MAX_DIMS];
    int size;
    REAL *data;
    REAL *grad;
    int requires_grad;
    TensorOp op;
    struct Tensor *left;
    struct Tensor *right;
} Tensor;

void TensorInit(Tensor *T);
MatrixError TensorCreate(Tensor *T, int ndim, const int *shape);
MatrixError TensorCreate1D(Tensor *T, int length);
MatrixError TensorCreate2D(Tensor *T, int row, int column);
void TensorFree(Tensor *T);

int TensorIsValid(const Tensor *T);
void TensorSetRequiresGrad(Tensor *T, int requires_grad);
MatrixError TensorEnsureGrad(Tensor *T);
MatrixError TensorZeroGrad(Tensor *T);
MatrixError TensorSGDStep(Tensor *T, REAL learning_rate);
MatrixError TensorFillZero(Tensor *T);
MatrixError TensorFillSequence(Tensor *T, REAL start, REAL step);
MatrixError TensorSet2D(Tensor *T, int i, int j, REAL value);
MatrixError TensorGet2D(const Tensor *T, int i, int j, REAL *value);
void TensorPrint(const Tensor *T, const char *name);
void TensorPrintGrad(const Tensor *T, const char *name);

#endif
