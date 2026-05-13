"""
Download sheep breed datasets from Roboflow and Kaggle.

Usage:
    python src/download_data.py --source roboflow --api-key YOUR_KEY
    python src/download_data.py --source kaggle
    python src/download_data.py --source all --api-key YOUR_KEY

Prerequisites:
    Roboflow:  pip install roboflow  (pass --api-key or set ROBOFLOW_API_KEY env var)
    Kaggle:    pip install kaggle
               Place ~/.kaggle/kaggle.json with your credentials, OR set
               KAGGLE_USERNAME and KAGGLE_KEY environment variables.
"""

import argparse
import os
import shutil
import zipfile
from pathlib import Path


RAW_DIR = Path("data/raw")

# Roboflow dataset details
RF_WORKSPACE  = "biomedical-engineering-mansoura-university"
RF_PROJECT    = "sheep-breeds-4ytic"
RF_VERSION    = 1

# Kaggle competition slug
KAGGLE_COMPETITION = "sheep-classification-challenge-2025"


def download_roboflow(api_key: str):
    try:
        from roboflow import Roboflow
    except ImportError:
        raise SystemExit("Install roboflow:  pip install roboflow")

    print("Downloading Roboflow dataset...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(RF_WORKSPACE).project(RF_PROJECT)
    dataset = project.version(RF_VERSION).download(
        "folder",                    # download as class-named folders
        location=str(RAW_DIR / "roboflow"),
        overwrite=False,
    )
    print(f"Roboflow dataset saved to {dataset.location}")
    return Path(dataset.location)


def download_kaggle():
    try:
        import kaggle  # noqa: F401
    except ImportError:
        raise SystemExit("Install kaggle:  pip install kaggle")

    dest = RAW_DIR / "kaggle"
    dest.mkdir(parents=True, exist_ok=True)

    print("Downloading Kaggle competition data...")
    os.system(
        f"kaggle competitions download -c {KAGGLE_COMPETITION} -p {dest}"
    )

    # Unzip all downloaded zips
    for zf in dest.glob("*.zip"):
        print(f"Extracting {zf.name}...")
        with zipfile.ZipFile(zf, "r") as z:
            z.extractall(dest)
        zf.unlink()

    print(f"Kaggle data saved to {dest}")
    return dest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", choices=["roboflow", "kaggle", "all"], default="all"
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ROBOFLOW_API_KEY", ""),
        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)",
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.source in ("roboflow", "all"):
        if not args.api_key:
            print(
                "WARNING: No Roboflow API key provided. "
                "Skipping Roboflow download.\n"
                "Provide --api-key or set ROBOFLOW_API_KEY."
            )
        else:
            download_roboflow(args.api_key)

    if args.source in ("kaggle", "all"):
        download_kaggle()

    print("\nDownload complete. Run src/preprocess.py next.")


if __name__ == "__main__":
    main()
