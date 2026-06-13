# Matrix-MiniTorch

`Matrix-MiniTorch` 是一个基于 `C99` 的教学型数值计算项目。它从一个小型矩阵库出发，逐步扩展出：

- 基础矩阵运算
- 矩阵乘法优化与 benchmark
- LU 分解与线性方程组求解
- 简化版 `Tensor` 与 `autograd`
- 一个基于 MNIST CSV 的 `MLP(784 -> 128 -> 10)` 训练 / 推理 demo


## 1. Project Overview

项目大致可以分成四层：

1. 基础层：`Matrix`
   负责矩阵数据结构、内存管理、基础运算和错误处理。

2. 优化层：`Optimized Kernels`
   负责矩阵乘法不同 kernel 的实现、自动调度、profiling 和 benchmark。

3. 抽象层：`Tensor / Autograd`
   在矩阵运算之上增加更高层的张量抽象与计算图，支持一个极简版本的自动求导。

4. 演示层：`MLP / Digits Demo`
   用 MNIST 数据跑通一个小型多层感知机，展示训练、评估、checkpoint 保存与加载推理。

## 2. Repository Structure

```text
Matrix-MiniTorch/
├─ include/               # 头文件
├─ src/                   # 源码实现
├─ tests/                 # 基础正确性测试入口
├─ examples/              # demo 程序
├─ benchmarks/            # benchmark 程序
├─ tools/                 # 数据转换与实验脚本
├─ data/                  # 本地数据目录
├─ artifacts/             # 实验输出目录
├─ build/                 # 编译中间文件
├─ bin/                   # 可执行文件
├─ Makefile               # 统一构建入口
└─ README.md
```

## 3. Key Files

### 3.1 Matrix and Solvers

- [include/matrix.h](Matrix-MiniTorch/include/matrix.h)
  基础矩阵类型、错误码、公开接口。

- [src/matrix.c](Matrix-MiniTorch/src/matrix.c)
  基础矩阵运算实现，如加减乘转置、范数、填充等。

- [include/matrix_lu.h](Matrix-MiniTorch/include/matrix_lu.h)
  LU 分解相关接口。

- [src/matrix_lu.c](Matrix-MiniTorch/src/matrix_lu.c)
  LU 分解、前代回代、行列式计算等。

- [include/matrix_solve.h](Matrix-MiniTorch/include/matrix_solve.h)
  高斯消元求解接口。

- [src/matrix_solve.c](Matrix-MiniTorch/src/matrix_solve.c)
  无主元与部分主元高斯消元实现。

### 3.2 Optimized Matrix Kernels

- [include/matrix_optimized.h](Matrix-MiniTorch/include/matrix_optimized.h)
  优化矩阵乘法、blocked transpose、AXPY、profiling 接口。

- [src/matrix_optimized.c](Matrix-MiniTorch/src/matrix_optimized.c)
  `naive / ikj / blocked / transposeB / kahan / auto` 等 kernel 的实现。

- [include/backend.h](Matrix-MiniTorch/include/backend.h)
  backend 类型定义。

- [src/backend.c](Matrix-MiniTorch/src/backend.c)
  backend 名称映射、可用性判断与 fallback。

### 3.3 Tensor and Autograd

- [include/tensor.h](Matrix-MiniTorch/include/tensor.h)
  简化版 `Tensor` 结构定义。

- [src/tensor.c](Matrix-MiniTorch/src/tensor.c)
  `Tensor` 的创建、释放、梯度清零、SGD 更新等。

- [include/tensor_ops.h](Matrix-MiniTorch/include/tensor_ops.h)
  Tensor 运算与自动求导接口。

- [src/tensor_ops.c](Matrix-MiniTorch/src/tensor_ops.c)
  `matmul / add / bias add / ReLU / MSE` 及反向传播实现。

### 3.4 Digits Demo and Dataset

- [include/digits_dataset.h](Matrix-MiniTorch/include/digits_dataset.h)
  MNIST CSV 数据集读取接口。

- [src/digits_dataset.c](Matrix-MiniTorch/src/digits_dataset.c)
  CSV 解析、像素归一化、标签 one-hot 转换。

- [examples/demo_digits.c](Matrix-MiniTorch/examples/demo_digits.c)
  纯 C 版 MNIST MLP 训练、评估、checkpoint 保存 / 加载、推理模式。

- [tools/convert_hf_mnist_to_csv.py](Matrix-MiniTorch/tools/convert_hf_mnist_to_csv.py)
  Hugging Face parquet 格式 MNIST 转 CSV。

- [tools/run_c_mnist_experiment.py](Matrix-MiniTorch/tools/run_c_mnist_experiment.py)
  纯 C MLP 的外层实验脚本：构建、训练、出图、管理 checkpoint 参数。

## 4. Matrix Library Usage Guide


### 4.1 Core Matrix Type

基础矩阵类型定义在 [include/matrix.h](/c:/C_langue/Matrix project/pytorch project/Matrix-MiniTorch/include/matrix.h)：

```c
typedef double REAL;

typedef struct {
    int row;
    int column;
    REAL *data;
} Matrix;
```

- 使用连续一维数组存储二维矩阵
- 默认按行主序存储
- 便于实现缓存友好的循环顺序优化
- 便于和高层 `Tensor` 抽象衔接

### 4.2 Basic Lifecycle

使用矩阵库时，最基本的流程是：

1. `MatrixInit`
   初始化一个空矩阵句柄

2. `MatrixCreate`
   按行列数申请内存

3. `MatrixFill...` 或 `MatrixSet`
   写入数据

4. 调用运算函数

5. `MatrixFree`
   释放内存

一个最小示例：

```c
#include "matrix.h"

int main(void)
{
    Matrix A, B, C;

    MatrixInit(&A);
    MatrixInit(&B);
    MatrixInit(&C);

    MatrixCreate(&A, 2, 2);
    MatrixCreate(&B, 2, 2);
    MatrixCreate(&C, 2, 2);

    MatrixFillSequence(&A, 1.0, 1.0);
    MatrixFillIdentity(&B);
    MatrixMultiply(&A, &B, &C);

    MatrixPrint(&C, "C");

    MatrixFree(&A);
    MatrixFree(&B);
    MatrixFree(&C);
    return 0;
}
```

### 4.3 Common Matrix APIs

基础矩阵接口主要包括：

- `MatrixCreate`
- `MatrixFree`
- `MatrixSet`
- `MatrixGet`
- `MatrixFillZero`
- `MatrixFillSequence`
- `MatrixFillIdentity`
- `MatrixFillRandom`
- `MatrixCopy`
- `MatrixAdd`
- `MatrixSub`
- `MatrixScale`
- `MatrixHadamard`
- `MatrixTranspose`
- `MatrixMultiply`
- `MatrixMultiplyNaive`
- `MatrixNormFrobenius`
- `MatrixAlmostEqual`

这些函数的实现主要位于 [src/matrix.c](Matrix-MiniTorch/src/matrix.c)。

### 4.4 Error Handling Style

本项目不依赖异常机制，而是统一返回 `MatrixError`。

这样设计保持纯 C 语言，便于跨平台，方便在测试和 benchmark 中显式检查错误，调用链清晰。

常见错误类型包括：

- 空指针
- 尺寸非法
- 尺寸不匹配
- 越界访问
- 内存申请失败
- 输出与输入发生别名冲突
- 非方阵或奇异矩阵

## 5. Matrix Features

项目目前已实现的核心矩阵能力包括：

- `MatrixAdd`
- `MatrixSub`
- `MatrixScale`
- `MatrixHadamard`
- `MatrixTranspose`
- `MatrixMultiplyNaive`
- `MatrixMultiply`
- `MatrixNormFrobenius`
- `MatrixAlmostEqual`
- `MatrixCopy`
- `MatrixFillZero`
- `MatrixFillSequence`
- `MatrixFillIdentity`
- `MatrixFillRandom`

这些能力主要在 [src/matrix.c](Matrix-MiniTorch/src/matrix.c) 中实现。

## 6. Optimized Matrix Multiplication

当前项目的矩阵乘法优化重点放在不同 kernel 的性能比较与自动调度：

- `naive`
- `ikj`
- `blocked`
- `transposeB`
- `kahan`
- `auto`

### 6.1 Multiple Kernels Explanation

虽然这些写法在数学上都在计算 `C = A * B`，但它们的内存访问方式不同，所以缓存命中率、循环开销、数值稳定性都会不同。

### 6.2 Current Auto Strategy

`MatrixMultiplyAuto` 会根据矩阵规模选择不同策略：

- 小矩阵优先 `ikj`
- 中矩阵优先 `transposeB`
- 大矩阵优先 `blocked`

这部分逻辑位于 [src/matrix_optimized.c](Matrix-MiniTorch/src/matrix_optimized.c)。

### 6.3 Profiling and Benchmark

benchmark 输出会记录：

- `kernel`
- `size`
- `repeat`
- `block_size`
- `seconds`
- `speedup_vs_naive`
- `max_abs_error_vs_kahan`
- `rel_fro_error_vs_kahan`

完整 benchmark 入口：

- [benchmarks/benchmark.c](Matrix-MiniTorch/benchmarks/benchmark.c)

## 7. Core Algorithm Map

### 7.1 Basic Matrix Arithmetic

基础矩阵运算在：

- [src/matrix.c](Matrix-MiniTorch/src/matrix.c)

主要负责：

- 矩阵创建与释放
- 矩阵打印
- 元素访问
- 加减法
- 标量乘法
- Hadamard 乘法
- 转置
- 默认矩阵乘法

### 7.2 Optimized Matrix Multiplication

矩阵乘法优化相关实现集中在：

- [src/matrix_optimized.c](Matrix-MiniTorch/src/matrix_optimized.c)

其中最重要的 kernel 包括：

- `MatrixMultiplyIKJ`
  调整循环顺序，提高访存局部性

- `MatrixMultiplyBlocked`
  分块乘法，提高缓存利用率

- `MatrixMultiplyTransposeB`
  先转置 `B`，把原本不连续的访问改成更连续的点积访问

- `MatrixMultiplyKahan`
  通过补偿求和提高数值稳定性

- `MatrixMultiplyAuto`
  根据矩阵规模自动选择更合适的实现

### 7.3 Profiling and Benchmark

profiling 接口在：

- [include/matrix_optimized.h](Matrix-MiniTorch/include/matrix_optimized.h)

benchmark 主程序在：

- [benchmarks/benchmark.c](Matrix-MiniTorch/benchmarks/benchmark.c)

有三个值得关注的角度：
1. 不同 kernel 的运行时间
2. 相对 `naive` 的加速比
3. 相对 `Kahan` 的误差

### 7.4 LU and Linear Solvers

LU 和线性方程组求解实现位于：

- [src/matrix_lu.c](Matrix-MiniTorch/src/matrix_lu.c)
- [src/matrix_solve.c](Matrix-MiniTorch/src/matrix_solve.c)

这部分说明项目不仅能做“矩阵乘法优化”，还覆盖了基础数值线性代数的常见任务。

## 8. Tensor / Autograd Overview

项目中的 `Tensor` 是一个极简实现，只支持当前 demo 需要的 1D / 2D 场景。

支持的关键能力：

- `TensorMatmulAuto`
- `TensorAddAuto`
- `TensorAddBiasAuto`
- `TensorReluAuto`
- `TensorMSELossAuto`
- `TensorBackward`
- `TensorSGDStep`

### 8.1 Current MLP Backward Optimization

对于 `matmul` 反向传播已经做了进一步优化：

- 原先的手写三重循环计算 `dA` 和 `dB` 改为矩阵公式
  - `dA = dY @ B^T`
  - `dB = A^T @ dY`
- 再复用 `MatrixMultiplyAuto`

这使得反向传播也能通过 `ikj / blocked / auto` 取得性能优化。


## 10. Tests

主测试入口：

```powershell
make run
```

覆盖内容包括：

- 基础矩阵运算
- 优化 kernel 正确性
- profiling
- LU / Gaussian solve
- auto 调度策略

## 11. Build and Run

### 11.1 Build Everything

```powershell
make
```

默认构建：

- `bin/matrix_demo`
- `bin/tensor_demo`
- `bin/mlp_demo`
- `bin/digits_demo`

### 11.2 Matrix Tests

```powershell
make run
```

### 11.3 Tensor Demo

```powershell
make run-tensor
```

### 11.4 XOR MLP Demo

```powershell
make run-mlp
```

### 11.5 Digits Demo

```powershell
make run-digits
```

说明：

- `make run-digits` 只是直接运行默认参数
- 更推荐用 `tools/run_c_mnist_experiment.py`

### 11.6 Benchmark

```powershell
make benchmark
```

快速版本：

```powershell
make benchmark-small
```

汇报版本：

```powershell
make benchmark-tuned
```

### 11.7 Clean

```powershell
make clean
```

会删除：

- `build/`
- `bin/`
- `artifacts/`

## 12. MNIST MLP Demo

为了检验我们自建的 MiniTorch 的实用性，项目实现了一个基于 MNIST CSV 数据的多层感知机 demo。

### 12.1 Model Structure

当前 digits demo 使用的是一个纯 C 的多层感知机：

```text
Input(784)
-> Linear(784, 128)
-> ReLU
-> Linear(128, 10)
-> MSE + one-hot
```

其中：

- 输入维度 `784` 对应 `28 x 28` 灰度图像
- 隐藏层维度固定为 `128`
- 输出维度 `10` 对应数字 `0-9`
- 激活函数是 `ReLU`
- 损失函数保持为 `MSE + one-hot`
- 训练方式是 `full-batch`

### 12.2 Important Design Choices

当前 MLP demo 的几个重要特点：

- 当前仍然使用 `MSE + one-hot`，而不是 `softmax + cross-entropy`。因此预期训练速度较慢、收敛时间较长。

### 12.3 Data Format

digits demo 默认读取：

- `data/mnist_train.csv`
- `data/mnist_test.csv`

CSV 格式要求：

- 每行一个样本
- 第 1 列是标签 `0-9`
- 后 784 列是像素值 `0-255`

程序读取后会自动完成：

- 像素归一化到 `[0, 1]`
- 标签转成 10 维 one-hot

### 12.4 From Hugging Face MNIST to CSV

本项目当前的数据准备流程是：

1. 下载 Hugging Face 的 `ylecun/mnist`
2. 将 parquet 文件放到 `mnist_hf/`
3. 用转换脚本生成本地 CSV

如果 `mnist_hf/` 目录已经就位，可以运行：

```powershell
python tools/convert_hf_mnist_to_csv.py
```

这会生成：

- `data/mnist_train.csv`
- `data/mnist_test.csv`

### 12.5 Build the Digits Demo

编译 digits demo：

```powershell
make digits_demo
```

或者：

```powershell
make
```

### 12.6 Training Mode

最推荐使用外层脚本：

```powershell
python tools/run_c_mnist_experiment.py --epochs 10 --save_intervals 10 --make-program make
```

含义：

- `--epochs`
  本次训练轮数

- `--save_intervals`
  每隔多少个 epoch 保存一次 checkpoint

- 最后一轮一定保存

如果想直接调用底层二进制，也可以：

```powershell
.\bin\digits_demo.exe data\mnist_train.csv data\mnist_test.csv 10 artifacts\digits\c_mnist_metrics.csv artifacts\digits\checkpoints 10
```

参数顺序是：

1. train CSV 路径
2. test CSV 路径
3. epochs
4. metrics CSV 路径
5. checkpoint 目录
6. save interval
7. checkpoint 加载路径，可为空字符串
8. inference-only 标志，`0` 表示训练，`1` 表示只推理

### 12.7 Checkpoint Saving Rules

checkpoint 保存规则如下：

- 每隔 `save_interval` 个 epoch 保存一次
- 最后一轮一定保存一次

例如：

```powershell
python tools/run_c_mnist_experiment.py --epochs 50 --save_intervals 10 --make-program make
```

会保存：

- `checkpoint_epoch_0010.txt`
- `checkpoint_epoch_0020.txt`
- `checkpoint_epoch_0030.txt`
- `checkpoint_epoch_0040.txt`
- `checkpoint_epoch_0050.txt`

checkpoint 默认保存目录：

- `artifacts/digits/checkpoints/`

### 12.8 Resume Training from Checkpoint

如果想从已有 checkpoint 继续训练：

```powershell
python tools/run_c_mnist_experiment.py --epochs 20 --save_intervals 10 --load-checkpoint artifacts/digits/checkpoints/checkpoint_epoch_0050.txt --make-program make
```

含义：

- 先加载 `epoch 50` 的参数
- 再继续训练 `20` 个 epoch
- 新日志和新 checkpoint 的 epoch 编号会接着往后记

### 12.9 Inference-Only Mode

如果只想加载 checkpoint 做评估，而不继续训练：

```powershell
python tools/run_c_mnist_experiment.py --inference-only --load-checkpoint artifacts/digits/checkpoints/checkpoint_epoch_0050.txt --make-program make
```

推理模式行为：

- 必须提供 `--load-checkpoint`
- 不训练
- 不再更新参数
- 不再新增 checkpoint
- 会输出当前 checkpoint 在 train/test 上的 loss 和 accuracy

### 12.10 Output Files

默认情况下，MLP demo 的输出都放在 `artifacts/digits/` 下面。

默认输出包括：

- metrics CSV
  `artifacts/digits/c_mnist_metrics.csv`

- loss 曲线
  `artifacts/digits/c_mnist_loss.png`

- accuracy 曲线
  `artifacts/digits/c_mnist_accuracy.png`

- checkpoint 目录
  `artifacts/digits/checkpoints/`

checkpoint 文件名示例：

- `artifacts/digits/checkpoints/checkpoint_epoch_0010.txt`
- `artifacts/digits/checkpoints/checkpoint_epoch_0020.txt`

如果在脚本里传了自定义参数，则会按你传入的路径保存。

### 12.11 Metrics CSV Logs

每个 epoch 都会记录：

- `epoch`
- `train_loss`
- `train_acc`
- `test_loss`
- `test_acc`
- `epoch_seconds`

可以直接生成：

- loss 曲线
- accuracy 曲线
- 每轮耗时图

### 12.12 Current Limitations

当前 MLP demo 的限制包括：

- 仅支持一个固定结构：`784 -> 128 -> 10`
- 当前损失函数仍是 `MSE + one-hot`
- full-batch 训练速度较慢
- 没有 mini-batch、没有 Adam、没有 softmax cross-entropy 等
- 目标仅是演示与实验，不是追求最优精度

