import csv
import io
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "mnist_hf" / "mnist"
OUTPUT_DIR = ROOT / "data"
TRAIN_PARQUET = INPUT_DIR / "train-00000-of-00001.parquet"
TEST_PARQUET = INPUT_DIR / "test-00000-of-00001.parquet"
TRAIN_CSV = OUTPUT_DIR / "mnist_train.csv"
TEST_CSV = OUTPUT_DIR / "mnist_test.csv"
INPUT_DIM = 28 * 28


def png_bytes_to_pixels(png_bytes: bytes) -> list[int]:
    with Image.open(io.BytesIO(png_bytes)) as image:
        grayscale = image.convert("L")
        return list(grayscale.tobytes())


def write_csv_from_parquet(parquet_path: Path, csv_path: Path) -> None:
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label"] + [f"pixel{i}" for i in range(INPUT_DIM)])

        for index, row in enumerate(rows, start=1):
            pixels = png_bytes_to_pixels(row["image"]["bytes"])
            if len(pixels) != INPUT_DIM:
                raise ValueError(
                    f"unexpected pixel count in {parquet_path.name} row {index}: {len(pixels)}"
                )
            writer.writerow([int(row["label"])] + pixels)

            if index % 5000 == 0 or index == len(rows):
                print(f"{parquet_path.name}: {index}/{len(rows)}")


def main() -> None:
    if not TRAIN_PARQUET.exists():
        raise FileNotFoundError(f"missing {TRAIN_PARQUET}")
    if not TEST_PARQUET.exists():
        raise FileNotFoundError(f"missing {TEST_PARQUET}")

    print("Converting Hugging Face MNIST parquet to CSV...")
    write_csv_from_parquet(TRAIN_PARQUET, TRAIN_CSV)
    write_csv_from_parquet(TEST_PARQUET, TEST_CSV)
    print("Done.")
    print(f"Train CSV: {TRAIN_CSV}")
    print(f"Test CSV:  {TEST_CSV}")


if __name__ == "__main__":
    main()
