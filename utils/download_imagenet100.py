"""Download and prepare ImageNet-100 from Kaggle for CNN-JEPA pre-training.

Steps performed:
  1. Download the dataset via the Kaggle API (requires ~/.kaggle/kaggle.json).
  2. Unzip the archives.
  3. Merge the 4 Kaggle train splits (train.X1 … train.X4) into a single
     `train/` directory using symlinks (no extra disk space needed).
  4. Rename the validation folder from `val.X` to `val/`.

Resulting structure expected by torchvision.datasets.ImageFolder:
  <output_dir>/
    train/
      n01440764/  ...
    val/
      n01440764/  ...

Usage (from the repo root):
  pip install kaggle          # one-time
  # Place your kaggle.json at ~/.kaggle/kaggle.json  (chmod 600)
  python utils/download_imagenet100.py --output_dir /data/imagenet100
"""

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: str):
    print(f"$ {cmd}")
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        print(f"ERROR: command failed with exit code {ret.returncode}")
        sys.exit(ret.returncode)


def check_kaggle():
    result = subprocess.run("kaggle --version", shell=True, capture_output=True)
    if result.returncode != 0:
        print("kaggle CLI not found. Installing …")
        run("pip install kaggle")

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print(
            "\nERROR: Kaggle credentials not found.\n"
            "Please:\n"
            "  1. Go to https://www.kaggle.com/settings  →  'Create new token'\n"
            "  2. Save the downloaded kaggle.json to ~/.kaggle/kaggle.json\n"
            "  3. Run: chmod 600 ~/.kaggle/kaggle.json\n"
            "Then re-run this script."
        )
        sys.exit(1)


def download_dataset(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "imagenet100.zip"

    if zip_path.exists():
        print(f"Archive already exists at {zip_path}, skipping download.")
    else:
        print("Downloading ImageNet-100 from Kaggle …")
        run(f"kaggle datasets download -d ambityga/imagenet100 -p {output_dir}")

    # The downloaded file might be named differently; find the zip.
    zips = list(output_dir.glob("*.zip"))
    if not zips:
        print("ERROR: No zip file found after download.")
        sys.exit(1)

    print("Extracting archive …")
    for z in zips:
        with zipfile.ZipFile(z, "r") as zf:
            zf.extractall(output_dir)
        z.unlink()  # remove zip after extraction
    print("Extraction complete.")


def merge_train_splits(output_dir: Path):
    """Merge train.X1 … train.X4 into a single train/ directory via symlinks."""
    train_dir = output_dir / "train"
    train_dir.mkdir(exist_ok=True)

    merged = 0
    for split_idx in range(1, 5):
        split_dir = output_dir / f"train.X{split_idx}"
        if not split_dir.exists():
            continue
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            link = train_dir / class_dir.name
            if not link.exists():
                link.symlink_to(class_dir.resolve())
                merged += 1

    print(f"Merged {merged} class folders into {train_dir}")


def prepare_val(output_dir: Path):
    """Rename val.X → val/."""
    val_x = output_dir / "val.X"
    val_dir = output_dir / "val"

    if val_dir.exists():
        print(f"val/ already exists at {val_dir}")
        return

    if val_x.exists():
        val_x.rename(val_dir)
        print(f"Renamed {val_x} → {val_dir}")
    else:
        print("WARNING: neither val.X nor val/ found. Check the dataset structure.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download and prepare ImageNet-100.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/imagenet100",
        help="Directory where ImageNet-100 will be stored (default: /data/imagenet100).",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    check_kaggle()
    download_dataset(output_dir)
    merge_train_splits(output_dir)
    prepare_val(output_dir)

    print(
        f"\nDone! ImageNet-100 is ready at {output_dir}\n"
        f"  train : {output_dir / 'train'}\n"
        f"  val   : {output_dir / 'val'}\n\n"
        "Now run CNN-JEPA pre-training:\n"
        "  PYTHONPATH=. python pretrain/train_ijepacnn.py "
        "--config-name ijepacnn_imagenet100_folder "
        f"data.train_root={output_dir}/train data.val_root={output_dir}/val"
    )


if __name__ == "__main__":
    main()
