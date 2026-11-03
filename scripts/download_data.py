"""
Download the Construction Site Safety Image Dataset from Kaggle.

Auth is resolved automatically by the kaggle library in this order:
  1. KAGGLE_API_TOKEN env var  (KGAT_ bearer token)
  2. KAGGLE_USERNAME + KAGGLE_KEY env vars
  3. ~/.kaggle/kaggle.json
"""

import subprocess
import sys
from pathlib import Path

DATASET = "snehilsanyal/construction-site-safety-image-dataset-roboflow"
DEST = Path(__file__).parent.parent / "data" / "raw"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "kaggle",
        "datasets", "download",
        "-d", DATASET,
        "-p", str(DEST),
        "--unzip",
    ]
    print(f"Downloading {DATASET} -> {DEST}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print("Download failed. Check your Kaggle credentials.")
        sys.exit(result.returncode)
    print("Done.")


if __name__ == "__main__":
    main()
