将 digits demo 所需的本地 CSV 数据放在这里。

默认文件名：

- `mnist_train_small.csv`
- `mnist_test_small.csv`

格式要求：

- 每行一个样本
- 第 1 列是标签 `0-9`
- 后 784 列是像素值 `0-255`

示例结构：

```text
label,pixel0,pixel1,...,pixel783
5,0,0,12,...,0
0,0,0,0,...,0
```

当前仓库默认不提交真实数据文件，只约定读取格式与路径。
