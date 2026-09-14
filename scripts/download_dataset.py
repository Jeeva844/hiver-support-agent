"""
Download the twcs.csv dataset from Kaggle.

Prerequisites:
  1. Install kaggle: pip install kaggle
  2. Place kaggle.json at ~/.kaggle/kaggle.json
     (Get it from: kaggle.com > Account > Create New API Token)
"""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

RAW_DIR = Path("data/raw")
ZIP_PATH = RAW_DIR / "customer-support-on-twitter.zip"
CSV_PATH = RAW_DIR / "twcs.csv"
KAGGLE_DATASET = "thoughtvector/customer-support-on-twitter"


def main() -> None:
    if CSV_PATH.exists():
        print(f"Dataset already exists at {CSV_PATH}. Skipping download.")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset '{KAGGLE_DATASET}'...")
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(RAW_DIR)],
            capture_output=True, text=True, check=True,
        )
        print(result.stdout)
    except FileNotFoundError:
        print("ERROR: kaggle CLI not found. Install with: pip install kaggle")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: kaggle download failed:\n{e.stderr}")
        print("Make sure ~/.kaggle/kaggle.json exists and has valid credentials.")
        sys.exit(1)

    # Extract zip
    if ZIP_PATH.exists():
        print(f"Extracting {ZIP_PATH}...")
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(RAW_DIR)
        # Some zips contain the CSV directly, others in a subdirectory
        if not CSV_PATH.exists():
            # Search for twcs.csv anywhere in data/raw
            found = list(RAW_DIR.rglob("twcs.csv"))
            if found:
                shutil.move(str(found[0]), str(CSV_PATH))
                # Clean up subdirectories created by zip
                for f in found[0].parent.iterdir():
                    if f != CSV_PATH:
                        if f.is_file():
                            f.unlink()
                        elif f.is_dir():
                            shutil.rmtree(f)
        ZIP_PATH.unlink(missing_ok=True)

    if CSV_PATH.exists():
        size_mb = CSV_PATH.stat().st_size / (1024 * 1024)
        print(f"Dataset ready at {CSV_PATH} ({size_mb:.1f} MB)")
    else:
        print("ERROR: twcs.csv not found after extraction.")
        print("Contents of data/raw/:")
        for p in RAW_DIR.iterdir():
            print(f"  {p.name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
