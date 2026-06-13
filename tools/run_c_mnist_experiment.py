import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and run the C MNIST experiment, then plot metrics.")
    parser.add_argument("--train", default="data/mnist_train.csv", help="Training CSV path relative to repo root.")
    parser.add_argument("--test", default="data/mnist_test.csv", help="Test CSV path relative to repo root.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--metrics", default="artifacts/digits/c_mnist_metrics.csv", help="Metrics CSV output path.")
    parser.add_argument("--loss-plot", default="artifacts/digits/c_mnist_loss.png", help="Loss plot path.")
    parser.add_argument("--acc-plot", default="artifacts/digits/c_mnist_accuracy.png", help="Accuracy plot path.")
    parser.add_argument("--checkpoint-dir", default="artifacts/digits/checkpoints", help="Checkpoint output directory.")
    parser.add_argument("--save-interval", "--save_intervals", dest="save_interval", type=int, default=10,
                        help="Save checkpoint every N epochs; final epoch is always saved.")
    parser.add_argument("--load-checkpoint", default="", help="Optional checkpoint file to load before training.")
    parser.add_argument("--inference-only", action="store_true",
                        help="Load a checkpoint and run evaluation only, without training.")
    parser.add_argument("--make-program", default="make", help="Make executable available on PATH.")
    parser.add_argument("--skip-build", action="store_true", help="Run the existing binary without rebuilding.")
    return parser.parse_args()


def repo_path(relative_path: str) -> Path:
    return ROOT / relative_path


def read_metrics(path: Path) -> dict[str, list[float]]:
    result = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "epoch_seconds": [],
    }

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for key in result:
                result[key].append(float(row[key]))

    return result


def plot_curve(x_values: list[float],
               y_a: list[float],
               y_b: list[float],
               y_label: str,
               legend_a: str,
               legend_b: str,
               title: str,
               output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_a, marker="o", label=legend_a)
    plt.plot(x_values, y_b, marker="s", label=legend_b)
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> int:
    args = parse_args()
    train_path = repo_path(args.train)
    test_path = repo_path(args.test)
    metrics_path = repo_path(args.metrics)
    loss_plot_path = repo_path(args.loss_plot)
    acc_plot_path = repo_path(args.acc_plot)
    checkpoint_dir = repo_path(args.checkpoint_dir)
    load_checkpoint_path = repo_path(args.load_checkpoint) if args.load_checkpoint else ""

    if not train_path.exists() or not test_path.exists():
        print("MNIST CSV files are missing.")
        print(f"Expected train: {train_path}")
        print(f"Expected test:  {test_path}")
        return 1
    if args.inference_only and not args.load_checkpoint:
        print("--inference-only requires --load-checkpoint.")
        return 1

    make_program = shutil.which(args.make_program)
    if not args.skip_build and make_program is None:
        print(f"Could not find make executable: {args.make_program}")
        print("Pass --make-program explicitly or put make on PATH.")
        return 1

    binary_name = "digits_demo.exe" if sys.platform.startswith("win") else "digits_demo"
    binary_path = ROOT / "bin" / binary_name

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    loss_plot_path.parent.mkdir(parents=True, exist_ok=True)
    acc_plot_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_build:
        subprocess.run([make_program, "digits_demo"], cwd=ROOT, check=True)

    if not binary_path.exists():
        print(f"Missing binary: {binary_path}")
        return 1

    run_cmd = [
        str(binary_path),
        str(train_path),
        str(test_path),
        str(args.epochs),
        str(metrics_path),
        str(checkpoint_dir),
        str(args.save_interval),
        str(load_checkpoint_path),
        "1" if args.inference_only else "0",
    ]
    subprocess.run(run_cmd, cwd=ROOT, check=True)

    metrics = read_metrics(metrics_path)
    plot_curve(metrics["epoch"], metrics["train_loss"], metrics["test_loss"],
               "Loss", "Train Loss", "Test Loss", "C MiniTorch MNIST Loss", loss_plot_path)
    plot_curve(metrics["epoch"], metrics["train_acc"], metrics["test_acc"],
               "Accuracy (%)", "Train Accuracy", "Test Accuracy", "C MiniTorch MNIST Accuracy", acc_plot_path)

    print(f"Saved metrics CSV: {metrics_path}")
    print(f"Saved loss plot:   {loss_plot_path}")
    print(f"Saved acc plot:    {acc_plot_path}")
    print(f"Final test accuracy: {metrics['test_acc'][-1]:.2f}%")
    print(f"Total epoch time:    {sum(metrics['epoch_seconds']):.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
