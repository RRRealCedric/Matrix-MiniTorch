# Matrix-MiniTorch

`Matrix-MiniTorch` 是一个基于 **C99** 实现的教学型数值计算项目。它以“小型矩阵计算库”为基础，逐步向上扩展出：

- 基础矩阵运算与错误处理
- 矩阵乘法优化 kernel 与 profiling
- LU 分解和线性方程组求解
- 简化版 `Tensor` 抽象与后端分发
- 极简 `autograd` 与 MLP 训练示例
- 面向实验汇报的 benchmark 流程

这个项目的核心目标不是复刻完整深度学习框架，而是用一个结构清晰、可完全读懂的 C 工程，展示从 **线性代数基础** 到 **性能优化** 再到 **简化神经网络训练** 的完整技术链路。

## 1. 项目定位与整体思路

从功能层次上看，本项目可以分成四层：

1. **基础层：Matrix**
   负责矩阵数据结构、内存管理、基础运算、错误处理。
2. **优化层：Optimized Kernels**
   重点研究矩阵乘法的不同实现方式，以及不同实现之间的性能与精度差异。
3. **抽象层：Tensor / Backend**
   在矩阵库之上加入更高层的数据抽象和后端分发机制，模拟 PyTorch / NumPy 的部分设计思想。
4. **演示层：Autograd / MLP / Digits Demo**
   展示如何在矩阵运算和 Tensor 抽象基础上，搭建一个非常简化的神经网络训练流程。

如果把这个项目看作一个“小型课程工程”，那么它的主线是：

`Matrix -> Optimized Matrix Kernels -> Tensor -> Autograd -> Mini MLP Demo`

## 2. 工程结构

项目目录结构如下：

```text
Matrix-MiniTorch/
├── include/                 # 公开头文件
├── src/                     # 具体实现
├── tests/                   # 正确性测试入口
├── examples/                # 示例程序
├── benchmarks/              # benchmark 程序
├── data/                    # 本地数据文件目录
├── artifacts/               # benchmark 输出目录，由 make 自动创建
├── build/                   # 编译中间文件，由 make 自动创建
├── bin/                     # 可执行文件，由 make 自动创建
├── Makefile                 # 统一构建入口
└── README.md
```

### 2.1 关键文件说明

#### 基础矩阵与求解器

- `include/matrix.h`
  对外公开的基础矩阵接口，定义 `Matrix`、`MatrixError` 以及基础矩阵函数。
- `src/matrix.c`
  基础矩阵功能实现，包括矩阵创建、释放、访问、填充、打印、加减、转置、乘法等。
- `include/matrix_lu.h`
  LU 分解与三角方程求解接口。
- `src/matrix_lu.c`
  无主元 LU 分解、前代、回代、行列式计算等实现。
- `include/matrix_solve.h`
  高斯消元线性方程组求解接口。
- `src/matrix_solve.c`
  无主元高斯消元和部分主元高斯消元实现。

#### 优化 kernel 与 profiling

- `include/matrix_optimized.h`
  优化矩阵乘法、blocked transpose、AXPY、profiling 等接口。
- `src/matrix_optimized.c`
  所有优化 kernel 的核心实现，包括 `IKJ`、`blocked`、`transposeB`、`Kahan`、`auto` 调度以及 profiling。
- `include/backend.h`
  后端枚举与后端查询接口。
- `src/backend.c`
  后端可用性判断、回退逻辑、名字映射。

#### Tensor 与 autograd

- `include/tensor.h`
  简化版 `Tensor` 数据结构与基础张量操作接口。
- `src/tensor.c`
  `Tensor` 的创建、释放、填充、打印、梯度清零、SGD 更新等。
- `include/tensor_ops.h`
  Tensor 运算与 autograd 运算接口。
- `src/tensor_ops.c`
  Tensor 加法、matmul、ReLU、MSE、反向传播等实现。

#### 数据集与 demo

- `include/digits_dataset.h`
  digits CSV 数据集加载接口。
- `src/digits_dataset.c`
  CSV 读取、像素归一化、标签 one-hot 编码等实现。
- `examples/demo_tensor.c`
  Tensor 与 backend 分发演示。
- `examples/demo_mlp.c`
  XOR 小型 MLP 训练演示。
- `examples/demo_digits.c`
  基于本地 CSV 的 `MLP(784->128->10)` digits 分类 demo。

#### 测试与实验

- `tests/main.c`
  项目主测试入口，覆盖基础矩阵功能、优化 kernel、profiling、LU、Gaussian solve，以及当前的 auto 调度策略。
- `benchmarks/benchmark.c`
  benchmark 主程序，用于比较不同矩阵乘法 kernel 的速度、加速比和数值误差。

## 3. 核心数据结构与模块设计

### 3.1 Matrix

基础矩阵类型定义在 `include/matrix.h` 中：

```c
typedef double REAL;

typedef struct {
    int row;
    int column;
    REAL *data;
} Matrix;
```

设计要点：

- 使用连续一维数组存储矩阵元素
- 默认按 **行主序** 存储
- 便于统一内存申请、释放和遍历
- 便于后续做缓存友好的循环优化

对矩阵元素 `(i, j)`，其线性下标是：

```text
i * column + j
```

### 3.2 MatrixError

项目中大部分写入型和计算型函数都通过 `MatrixError` 返回执行状态，而不是依赖全局错误码或异常机制。常见错误包括：

- 空指针或无效矩阵
- 非法尺寸
- 尺寸不匹配
- 内存申请失败
- 越界访问
- 输出与输入 `data` 指针别名冲突
- 非方阵参与 LU/求解
- 主元过小或矩阵接近奇异

这种设计的意义是：

- 每个函数的失败方式明确
- 测试代码容易覆盖
- 对课程工程和汇报来说，可读性和可解释性都很好

### 3.3 Tensor

`Tensor` 是在 `Matrix` 基础上的更高层抽象。它并不是完整框架的张量，而是一个 **支持 1D / 2D、连续存储、带梯度数组的简化张量结构**。

它包含：

- `ndim`
- `shape[]`
- `stride[]`
- `size`
- `data`
- `grad`
- `requires_grad`
- `op / left / right`

最后三个字段用于记录简化版计算图，供 `TensorBackward()` 执行反向传播。

## 4. 基础功能说明

### 4.1 矩阵基础运算

在 `src/matrix.c` 中，实现了以下基础运算：

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

其中：

- `MatrixMultiplyNaive` 是最直接的三重循环基线实现
- `MatrixMultiply` 是默认基础乘法实现，已经使用了更 cache-friendly 的循环顺序

### 4.2 LU 分解与线性方程求解

在 `src/matrix_lu.c` 和 `src/matrix_solve.c` 中，实现了：

- `LUDecomposeNoPivot`
- `ForwardSubstitution`
- `BackSubstitution`
- `LUSolve`
- `LUDeterminant`
- `GaussianSolveNoPivot`
- `GaussianSolvePartialPivot`

这部分功能说明项目不仅是“算矩阵乘法”，还覆盖了课程中常见的线性代数求解任务。

## 5. 优化 kernel 与 auto 调度

### 5.1 为什么要单独做优化层

矩阵乘法通常是整个项目中计算量最大、最值得优化的部分。不同写法虽然数学结果相同，但运行效率可能差很多。因此项目把优化逻辑集中到 `src/matrix_optimized.c`，单独比较不同 kernel：

- `MatrixMultiplyNaive`
- `MatrixMultiplyIKJ`
- `MatrixMultiplyBlocked`
- `MatrixMultiplyTransposeB`
- `MatrixMultiplyKahan`
- `MatrixMultiplyAuto`

### 5.2 各 kernel 的作用

#### `naive`

最原始的三重循环实现，用作性能基线。

#### `IKJ`

通过调整循环顺序，使得对 `B` 和 `C` 的访问更符合行主序缓存局部性。  
这是最经典、最容易解释的一步优化。

#### `blocked`

将矩阵划分为小块，以块为单位计算，减少大矩阵乘法中的缓存失效。  
这是性能优化中非常重要的一种技术。

#### `transposeB`

先对 `B` 转置，再把原来的跨行访问改成连续点积访问。  
它的优点是逻辑清晰、思路直观，适合作为“中等规模矩阵的候选优化策略”。

#### `Kahan`

采用 Kahan 补偿求和，不是为了更快，而是为了在大量浮点加法中获得更稳定的数值结果。  
在 benchmark 中，它被用作精度参考。

### 5.3 auto 调度策略

`MatrixMultiplyAuto` 的目的不是再发明一种新算法，而是根据矩阵规模在已有 kernel 中选择更合适的实现。

当前策略分三段：

- 小矩阵：优先 `IKJ`
- 中矩阵：优先 `transposeB`
- 大矩阵：优先 `blocked`

同时，`blocked` 不再固定一个 block size，而是按规模选择合适的块大小。当前实现中使用了按规模映射的 block size 策略，而不是全尺寸统一固定值。

这样设计的原因是：

- 小矩阵太小，复杂调度和转置预处理的额外开销不一定值得
- 中等规模下，`transposeB` 的连续访问模式常有优势
- 大矩阵时，缓存局部性的重要性更高，`blocked` 更稳定

### 5.4 profiling

优化层支持轻量 profiling。每条记录包含：

- `kernel_name`
- `requested_backend_name`
- `backend_name`
- `m / n / k`
- `block_size`
- `elapsed_seconds`
- `error`

它的意义是：

- 不只是“算出了结果”，还能说明“实际用了哪个策略”
- 便于解释 `auto` 的行为
- 便于把实验结果纳入汇报

## 6. Tensor、Backend 与 Autograd

### 6.1 Backend 分发

项目中定义了多个后端类型：

- `BACKEND_NAIVE`
- `BACKEND_IKJ`
- `BACKEND_BLOCKED`
- `BACKEND_SIMD`
- `BACKEND_THREADS`
- `BACKEND_BLAS`

其中真正实现的核心仍然是纯 C 版本；`SIMD / THREADS / BLAS` 当前更多是扩展入口与回退框架，用于体现工程可扩展性。

### 6.2 Tensor 运算

当前已实现的主要 Tensor 运算包括：

- `TensorAdd`
- `TensorScale`
- `TensorMatmul`
- `TensorAddAuto`
- `TensorAddBiasAuto`
- `TensorMatmulAuto`
- `TensorReluAuto`
- `TensorMSELossAuto`
- `TensorBackward`

这部分的作用是把底层矩阵运算包装成更接近神经网络编程的接口。

### 6.3 XOR MLP 示例

`examples/demo_mlp.c` 实现了一个简单的 XOR 训练流程：

```text
X -> Linear(2,2) -> ReLU -> Linear(2,1) -> MSE
```

这个示例说明项目已经具备最基础的：

- 前向传播
- 损失计算
- 反向传播
- SGD 参数更新

## 7. Digits CSV Demo

为了把项目从 XOR 演示进一步拓展到一个更像真实分类任务的例子，项目增加了 digits demo：

```text
MLP(784 -> 128 -> 10)
```

它采用：

- 输入维度 `784`
- 隐藏层维度 `128`
- 输出维度 `10`
- 激活函数 `ReLU`
- 损失函数 `MSE + one-hot`
- full-batch 训练

### 7.1 数据格式

digits demo 默认从以下路径读取本地 CSV：

- `data/mnist_train_small.csv`
- `data/mnist_test_small.csv`

格式要求：

- 每行一个样本
- 第 1 列是标签 `0-9`
- 后 784 列是像素值 `0-255`

程序会在加载时完成：

- CSV 解析
- 像素归一化到 `[0, 1]`
- 标签转为 10 维 one-hot 向量

### 7.2 使用说明

运行命令：

```sh
make run-digits
```

如果需要替换数据文件，直接把本地合法 CSV 放到 `data/` 中，并使用上述默认文件名，或者自行修改命令行参数调用对应可执行程序。

说明：

- `data/` 目录中建议放置用户自己的合法本地数据文件
- 该 demo 是扩展实验入口，不是当前主线性能实验的核心部分

## 8. benchmark 设计与指标含义

### 8.1 benchmark 目标

benchmark 的目的不是只输出一个“谁更快”的结论，而是同时比较：

- 运行时间
- 相对基线加速比
- 相对高精度参考结果的误差

### 8.2 当前比较的 kernel

benchmark 会比较：

- `naive`
- `default`
- `ikj`
- `blocked`
- `transposeB`
- `kahan`
- `auto`

### 8.3 输出字段

CSV 和控制台摘要中包含：

- `kernel`
- `size`
- `repeat`
- `block_size`
- `seconds`
- `speedup_vs_naive`
- `max_abs_error_vs_kahan`
- `rel_fro_error_vs_kahan`

含义如下：

- `seconds`：平均 CPU 时间
- `speedup_vs_naive`：相对 `naive` 的加速比
- `max_abs_error_vs_kahan`：与 `Kahan` 结果相比的最大绝对误差
- `rel_fro_error_vs_kahan`：与 `Kahan` 结果相比的相对 Frobenius 误差

### 8.4 为什么以 Kahan 作为精度参考

`Kahan` 版本不是为了性能，而是为了降低浮点累加过程中的数值误差。因此在 benchmark 中：

- `naive` 是速度基线
- `kahan` 是精度参考

这种设计使得实验更完整：我们不仅比较快慢，也比较“快了之后数值结果是否还稳定”。

## 9. 测试覆盖说明

主测试入口是：

- `tests/main.c`

当前测试覆盖包括：

- 基础矩阵加减乘转置
- 范数、Hadamard、单位矩阵填充
- 优化 kernel 与基线结果一致性
- blocked transpose、AXPY、in-place 工具
- profiling 启用/关闭/清空/字段正确性
- 错误处理与 aliasing 检查
- LU 分解与线性方程求解
- auto 调度策略在小/中/大矩阵上的行为验证

运行：

```sh
make run
```

## 10. 编译与运行

项目统一使用 `make` 作为入口。

### 10.1 构建全部目标

```sh
make
```

默认会构建：

- `bin/matrix_demo`
- `bin/tensor_demo`
- `bin/mlp_demo`
- `bin/digits_demo`

在 Windows 下可执行文件名通常会带 `.exe` 后缀。

### 10.2 基础测试

```sh
make run
```

### 10.3 Tensor 演示

```sh
make run-tensor
```

### 10.4 XOR MLP 演示

```sh
make run-mlp
```

### 10.5 Digits MLP 演示

```sh
make run-digits
```

### 10.6 完整 benchmark

```sh
make benchmark
```

### 10.7 快速 benchmark

```sh
make benchmark-small
```

用于快速检查 benchmark 流程与输出格式。

### 10.8 报告用 benchmark

```sh
make benchmark-tuned
```

该目标固定使用：

- 矩阵规模：`128,256,512,1000`
- 重复次数：`5`
- block size 扫描：`8,16,32,64`
- CSV 输出：`artifacts/benchmark/benchmark_tuned.csv`

说明：

- 该目标可能运行较长时间，尤其是包含 `kahan` 和 `1000x1000` 时
- 适合用于最终汇报取数

### 10.9 清理

```sh
make clean
```

该命令会删除：

- `build/`
- `bin/`
- `artifacts/`

## 11. 建议的阅读顺序

如果第一次接触这个项目，建议按下面顺序阅读：

1. `include/matrix.h` / `src/matrix.c`
   先理解基础矩阵结构和基础运算。
2. `include/matrix_optimized.h` / `src/matrix_optimized.c`
   再看优化 kernel 和 auto 调度。
3. `tests/main.c`
   通过测试了解项目到底验证了哪些行为。
4. `benchmarks/benchmark.c`
   理解实验设计、指标和性能对比方法。
5. `include/tensor.h` / `src/tensor.c` / `src/tensor_ops.c`
   理解 Tensor 和 autograd 的简化实现。
6. `examples/demo_mlp.c`
   看最小可训练网络如何搭建。
7. `examples/demo_digits.c`
   看如何接入更接近真实任务的数据集输入。

## 12. 项目总结

`Matrix-MiniTorch` 的价值不在于“功能很多”本身，而在于它把一个典型的计算系统拆成了清晰的层次，并且每一层都能被读懂、被测试、被 benchmark：

- 底层有完整矩阵库
- 中层有可比较的优化 kernel
- 上层有 Tensor 和 autograd
- 最终有可运行的 demo 和可导出的实验结果

因此，它既可以被看作一个小型线性代数库，也可以被看作一个“从数值计算到简化神经网络”的教学型系统工程。
