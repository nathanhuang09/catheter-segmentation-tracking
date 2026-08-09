"""
Extract a small subset from video_action_understanding.zip.

Usage (after downloading the zip to data/raw/):
    python scripts/extract_video_subset.py

Keeps training.csv, validation.csv, and frames for the first N videos found in the CSV.
"""
import shutil
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_ZIP = ROOT / "data" / "raw" / "video_action_understanding.zip"
OUT = ROOT / "data" / "action"
NUM_VIDEOS = 3


def main():
    if not RAW_ZIP.exists():
        print(f"Download first:\n  huggingface-cli download airvlab/CathAction video_action_understanding.zip --repo-type dataset --local-dir ./data/raw")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(RAW_ZIP, "r") as zf:
        names = zf.namelist()

        for csv_name in ("training.csv", "validation.csv"):
            matches = [n for n in names if n.endswith(csv_name)]
            if not matches:
                print(f"Warning: {csv_name} not found in zip")
                continue
            target = OUT / csv_name
            with zf.open(matches[0]) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            print(f"Extracted {target}")

        train_csv = OUT / "training.csv"
        if not train_csv.exists():
            print("No training.csv — cannot pick video subset.")
            return

        df = pd.read_csv(train_csv)
        video_ids = df["video_id"].astype(str).unique()[:NUM_VIDEOS]
        print(f"Extracting frames for: {list(video_ids)}")

        frames_out = OUT / "video_frames"
        frames_out.mkdir(parents=True, exist_ok=True)

        for vid in video_ids:
            prefix_candidates = [
                f"video_frames/{vid}/",
                f"{vid}/",
                f"video_frames/{vid}",
            ]
            extracted = 0
            for name in names:
                if not any(name.startswith(p) for p in prefix_candidates):
                    continue
                if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                rel = Path(name).name
                dest = frames_out / vid / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(name) as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted += 1
            print(f"  {vid}: {extracted} frames")

    print("\nDone. You can delete the zip to free space:")
    print(f"  del {RAW_ZIP}")


if __name__ == "__main__":
    main()
