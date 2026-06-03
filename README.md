# 小型 C 语言矩阵计算库 Matrix

本目录是最终提交的完整工程文件包。工程以基础矩阵计算库为核心，完成课程要求的四项矩阵运算：矩阵加法、矩阵标量乘法、矩阵转置和矩阵乘法。在此基础上，工程进一步加入了矩阵乘法优化版本、简化 Tensor 抽象、后端分发机制和一个 mini autograd / MLP 演示，用于展示程序的可扩展性。

本文件第六部分展示了编译运行的命令，可以直接跳转阅读。

## 一、工程文件结构

| 文件 | 作用 |
| --- | --- |
| `matrix.h` | 基础矩阵库头文件，定义 `Matrix`、`MatrixError` 和基础矩阵函数接口。 |
| `matrix.c` | 基础矩阵库实现文件，实现矩阵创建、释放、访问、填充、打印和基础矩阵运算。 |
| `matrix_optimized.h` | 优化矩阵乘法接口声明。 |
| `matrix_optimized.c` | 优化矩阵乘法实现，包括 IKJ 循环顺序、分块乘法、转置 B 乘法和 Kahan 补偿求和乘法。 |
| `matrix_lu.h` | LU 分解与三角方程求解接口声明。 |
| `matrix_lu.c` | 实现无主元 LU 分解、前代、回代、LU 求解和行列式计算。 |
| `matrix_solve.h` | 高斯消元线性方程组求解接口声明。 |
| `matrix_solve.c` | 实现无主元高斯消元和带部分主元高斯消元。 |
| `tensor.h` | 简化 Tensor 数据结构定义，包含 shape、stride、data、grad 和 autograd 元数据。 |
| `tensor.c` | Tensor 创建、释放、填充、打印、梯度清零和 SGD 参数更新等基础功能。 |
| `tensor_ops.h` | Tensor 运算接口声明，包括普通 Tensor 运算和 autograd 运算。 |
| `tensor_ops.c` | Tensor 运算、后端分发、自动求导反向传播实现。 |
| `backend.h` | 后端类型定义与后端选择接口。 |
| `backend.c` | 后端选择逻辑实现，可在 naive、IKJ、blocked 三种矩阵乘法后端之间切换。 |
| `main.c` | 基础矩阵库正确性测试主程序。 |
| `demo_tensor.c` | mini PyTorch 风格 Tensor/Backend 演示程序。 |
| `demo_mlp.c` | mini autograd 多层感知机训练演示程序。 |
| `benchmark.c` | 性能测试程序，输出基础运算和不同矩阵乘法实现的平均运行时间。 |
| `Makefile` | 工程编译、运行、性能测试和清理命令。 |
| `README.md` | 工程说明文档。 |

## 二、基础矩阵库说明

### 1. 数据结构

矩阵对象使用 `Matrix` 结构体表示：

```c
typedef double REAL;

typedef struct {
    int row;
    int column;
    REAL *data;
} Matrix;
```

矩阵元素按行主序存储在一维连续数组中。第 `i` 行第 `j` 列元素对应的一维下标为：

```c
i * column + j
```

这种设计便于统一进行内存申请和释放，也有利于按行访问时利用缓存局部性。

### 2. 课程要求的四项基础功能

基础矩阵库实现了以下四项必需功能，均为独立函数，不写在 `main.c` 中：

- `MatrixAdd`：矩阵加法。
- `MatrixScale`：矩阵标量乘法。
- `MatrixTranspose`：矩阵转置。
- `MatrixMultiply`：矩阵乘法。

这些函数会检查输入矩阵是否有效、输出矩阵维度是否正确，以及矩阵维度是否满足运算要求。

### 3. 扩展矩阵函数

为了提高矩阵库完整性，工程还实现了以下辅助函数：

- `MatrixSub`：矩阵减法。
- `MatrixHadamard`：矩阵逐元素乘法。
- `MatrixNormFrobenius`：Frobenius 范数。
- `MatrixFillIdentity`：填充单位矩阵。
- `MatrixFillRandom`：填充随机数。
- `MatrixAlmostEqual`：矩阵近似相等判断，用于浮点结果验证。
- `MatrixCopy`：复制同尺寸矩阵内容。
- `MatrixHasShape`：判断矩阵是否为指定行列数。

此外，工程已整合 LU 分解和线性方程组求解功能：

- `LUDecomposeNoPivot`：无主元 LU 分解，将方阵 A 分解为 L 和 U。
- `ForwardSubstitution`：求解下三角方程 Ly=b。
- `BackSubstitution`：求解上三角方程 Ux=y。
- `LUSolve`：基于已经分解得到的 L、U 求解 Ax=b。
- `LUDeterminant`：根据 U 的对角线元素计算 det(A)。
- `GaussianSolveNoPivot`：无主元高斯消元求解 Ax=b。
- `GaussianSolvePartialPivot`：带部分主元选择的高斯消元求解 Ax=b。

其中，带部分主元的高斯消元会在每一列中选择绝对值最大的候选主元行并交换到当前行，通常比无主元版本更稳定。

### 4. 错误处理

矩阵库统一使用 `MatrixError` 返回函数执行状态，常见错误包括：

- 空指针或无效矩阵。
- 非法矩阵尺寸。
- 矩阵尺寸溢出。
- 内存申请失败。
- 下标越界。
- 矩阵维度不匹配。
- 输出矩阵与输入矩阵共享同一块 `data` 内存。
- 非方阵参与 LU 分解或线性方程组求解。
- 主元过小或矩阵接近奇异。

其中，输出矩阵与输入矩阵共享同一块 `data` 的情况会返回 `MATRIX_ERROR_ALIASING`。例如矩阵乘法中，如果 `C` 和 `A` 是同一个矩阵，计算过程中会一边读取输入、一边覆盖结果，可能导致后续计算错误。因此本工程要求写入型运算的输出矩阵使用独立数据缓冲区。

### 5. 基础库局部优化

基础矩阵库在保持代码可读性的同时做了几项低风险优化：

- 元素总数使用 `size_t`，避免大矩阵元素数量超过 `int` 范围。
- 热点循环中减少 `MatrixIndex()` 的重复调用，改用局部行偏移。
- 逐元素运算中使用局部 `restrict` 指针，帮助编译器优化内存访问。
- 对输出矩阵与输入矩阵的 alias 情况进行显式检查，避免隐藏错误。

## 三、矩阵乘法优化版本

矩阵乘法是本工程中计算量最大的操作。基础版本 `MatrixMultiply` 使用清晰的三重循环实现，便于理解算法逻辑。

为了比较不同实现方式的效率和精度，工程额外提供多个优化版本：

- `MatrixMultiplyIKJ`：将循环顺序调整为 `i-k-j`，使对 `B` 和 `C` 的访问更符合行主序连续存储特点。
- `MatrixMultiplyBlocked`：使用分块计算，提高缓存局部性。
- `MatrixMultiplyTransposeB`：先将 `B` 转置为 `BT`，再将 `A` 的行与 `BT` 的行做连续点积，减少内层循环中的跨行访问。
- `MatrixMultiplyKahan`：在每个点积中使用 Kahan 补偿求和，减少大量浮点加法带来的舍入误差，主要用于精度对比。

这些版本的数学结果应保持一致，但运行时间和数值误差可能不同。一般来说，IKJ、分块和转置 B 版本偏向速度优化，Kahan 版本偏向精度优化。可通过 `make benchmark` 进行性能比较。

## 四、mini Tensor / Backend 设计

在基础矩阵库之上，工程实现了一个简化的 Tensor 层，用于展示类似 PyTorch / NumPy 的分层思想。

整体结构可以理解为：

```text
用户程序
  -> Tensor API
  -> Backend 分发
  -> Matrix kernel
  -> 一维连续内存
```

### 1. Tensor 数据结构

`Tensor` 支持 1D 和 2D 连续存储，包含：

- `ndim`：维度数量。
- `shape`：每一维长度。
- `stride`：每一维步长。
- `size`：元素总数。
- `data`：数据数组。
- `grad`：梯度数组。
- `requires_grad`：是否需要梯度。
- `op`、`left`、`right`：用于记录简化计算图。

### 2. 后端分发

后端类型定义如下：

- `BACKEND_NAIVE`：调用基础矩阵乘法。
- `BACKEND_IKJ`：调用 IKJ 优化矩阵乘法。
- `BACKEND_BLOCKED`：调用分块矩阵乘法。

用户可通过：

```c
BackendSetType(BACKEND_IKJ);
```

选择后端。随后调用 `TensorMatmul` 或 `TensorMatmulAuto` 时，会根据当前后端自动选择底层矩阵乘法 kernel。

## 五、mini Autograd 与 MLP 演示

工程实现了一个简化版自动求导系统，用于展示 PyTorch 风格的核心思想。该系统不追求完整深度学习框架能力，而是支持一个小型多层感知机所需的基本算子。

### 1. 支持的 autograd 算子

- `TensorAddAuto`：带计算图记录的 Tensor 加法。
- `TensorAddBiasAuto`：对二维 Tensor 按列加 bias。
- `TensorMatmulAuto`：带计算图记录的矩阵乘法。
- `TensorReluAuto`：ReLU 激活函数。
- `TensorMSELossAuto`：均方误差损失。
- `TensorBackward`：从标量损失开始反向传播。

### 2. 支持的训练工具

- `TensorZeroGrad`：清空梯度。
- `TensorSGDStep`：使用 SGD 更新参数。
- `TensorPrintGrad`：打印 Tensor 梯度。

### 3. MLP 示例

`demo_mlp.c` 中实现了一个简单 XOR 训练示例：

```text
X -> Linear(2,2) -> ReLU -> Linear(2,1) -> MSE
```

程序会执行：

1. 前向传播。
2. 计算 MSE 损失。
3. 调用 `TensorBackward` 自动反向传播。
4. 使用 `TensorSGDStep` 更新参数。
5. 输出训练过程中的 loss 和最终预测结果。

该演示说明本项目不仅能完成矩阵运算，还能在矩阵库基础上搭建一个非常简化的神经网络训练基础设施。

## 六、编译与运行方法

请在终端进入本目录：

```sh
cd Matrix
```

### 1. 编译全部默认目标

```sh
make
```

该命令会编译：

- `matrix_demo`
- `tensor_demo`
- `mlp_demo`

### 2. 运行基础矩阵库测试

```sh
make run
```

该命令运行 `main.c` 中的正确性测试，测试顺序符合课程要求：

1. 矩阵加法。
2. 矩阵标量乘法。
3. 矩阵转置。
4. 矩阵乘法。

此外，还会测试扩展矩阵函数、优化乘法 kernel 和错误处理逻辑。
现在的基础测试还包括 LU 分解和线性方程组求解：

- 检查 `L * U` 是否能重构原矩阵 A。
- 检查 `LUSolve` 是否得到预期解向量。
- 检查 `LUDeterminant` 是否得到预期行列式。
- 检查 `GaussianSolvePartialPivot` 是否得到预期解向量。

### 3. 运行 Tensor/Backend 演示

```sh
make run-tensor
```

该命令展示同一个 `TensorMatmul` 高层接口在不同后端下的运行结果：

- naive backend
- IKJ backend
- blocked backend

### 4. 运行 mini MLP 训练演示

```sh
make run-mlp
```

该命令训练一个用于 XOR 数据的小型 MLP。运行结果中会输出训练过程中的 loss，并打印最终预测值和目标值。

### 5. 运行完整性能测试

```sh
make benchmark
```

性能测试包含矩阵规模：

- `100 x 100`
- `1000 x 1000`
- `2000 x 2000`

每种规模重复测试 5 次，输出平均 CPU 时间，计时方法为 C 标准库中的 `clock()`。

注意：`2000 x 2000` 的朴素矩阵乘法计算量较大，运行时间可能较长，这是正常现象。

### 6. 运行快速性能测试

```sh
make benchmark-small
```

该命令使用较小矩阵规模，仅用于快速确认 benchmark 流程是否能正常运行，不建议作为正式报告数据。

### 7. 清理编译产物

```sh
make clean
```

该命令会删除可执行文件和 `.o` 目标文件，使目录恢复为只包含源码和说明文档的状态。

## 七、建议的报告写法

报告中可以将本工程分为三层介绍：

1. 基础层：小型 C 语言矩阵计算库，完成四项基础矩阵运算。
2. 优化层：比较朴素乘法、IKJ 循环顺序和分块乘法的性能。
3. 扩展层：参考 PyTorch 的 Tensor、Backend 和 Autograd 思想，实现一个简化 MLP 训练示例。

基础要求部分重点说明 `matrix.h` / `matrix.c` 中的矩阵结构、错误处理、四项运算和正确性测试。

提高内容部分可以说明：

- 不同矩阵乘法 kernel 的设计差异。
- 不同矩阵规模下的平均运行时间。
- Tensor/Backend 分层如何体现可扩展性。
- mini autograd 如何支持前向传播、反向传播和 SGD 参数更新。

## 八、注意事项

- 本项目使用 C99 标准编译。
- 本项目不依赖第三方库，只使用 C 标准库和数学库 `-lm`。
- 大矩阵性能测试可能耗时较长，建议在报告取数时单独运行 `make benchmark`。
- Tensor/autograd 部分是教学演示性质的简化实现，不等同于完整 PyTorch，但体现了 Tensor、后端分发、计算图和反向传播的基本思想。
