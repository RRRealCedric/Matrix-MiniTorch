#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include "backend.h"
#include "tensor.h"

MatrixError TensorAdd(const Tensor *A, const Tensor *B, Tensor *C);
MatrixError TensorScale(REAL alpha, const Tensor *A, Tensor *B);
MatrixError TensorMatmul(const Tensor *A, const Tensor *B, Tensor *C);

MatrixError TensorAddAuto(Tensor *A, Tensor *B, Tensor *C);
MatrixError TensorAddBiasAuto(Tensor *A, Tensor *Bias, Tensor *C);
MatrixError TensorMatmulAuto(Tensor *A, Tensor *B, Tensor *C);
MatrixError TensorReluAuto(Tensor *A, Tensor *Y);
MatrixError TensorMSELossAuto(Tensor *Prediction, Tensor *Target, Tensor *Loss);
MatrixError TensorBackward(Tensor *Loss);

#endif
