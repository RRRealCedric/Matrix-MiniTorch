`data/` is the local dataset directory used by `digits_demo`.

Expected default filenames:

- `mnist_train.csv`
- `mnist_test.csv`

Expected CSV format:

- one sample per line
- first column: label `0-9`
- next 784 columns: grayscale pixels `0-255`

Example:

```text
label,pixel0,pixel1,...,pixel783
5,0,0,12,...,0
0,0,0,0,...,0
```

Recommended workflow for this repository:

1. Keep the Hugging Face download inside `mnist_hf/`
2. Convert parquet files to CSV locally
3. Run `make run-digits`

Notes:

- Full MNIST CSV files are generated artifacts and can be large.
- The repository ignores `data/mnist_train.csv` and `data/mnist_test.csv` by default.
- The source `mnist_hf/` parquet files are much more suitable for versioning than full CSV exports.
