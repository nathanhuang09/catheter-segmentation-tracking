"""Show frame clips with ground-truth action labels (recognition task)."""
import ast
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ACTION_NAMES, FRAMES_DIR, OUTPUT, TRAIN_CSV

NUM_CLIPS = 3
FRAMES_PER_CLIP = 8


def parse_action_classes(raw) -> list[int]:
    if isinstance(raw, list):
        return [int(x) for x in raw]
    if isinstance(raw, str):
        return [int(x) for x in ast.literal_eval(raw)]
    return [int(raw)]


def frame_path(video_id: str, frame_idx: int) -> Path:
    # Common layouts: video_frames/video_1/000430.jpg or frame_430.jpg
    vid_dir = FRAMES_DIR / video_id
    for pattern in (f"{frame_idx:06d}.jpg", f"{frame_idx:06d}.png", f"frame_{frame_idx}.jpg"):
        p = vid_dir / pattern
        if p.exists():
            return p
    matches = list(vid_dir.glob(f"*{frame_idx}*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No frame {frame_idx} in {vid_dir}")


def load_clip(video_id: str, start: int, stop: int, n: int = FRAMES_PER_CLIP):
    indices = [int(x) for x in np.linspace(start, stop, n, dtype=int)]
    frames = []
    for idx in indices:
        p = frame_path(video_id, idx)
        img = cv2.imread(str(p))
        if img is None:
            continue
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return frames


def main():
    if not TRAIN_CSV.exists():
        print(f"Missing {TRAIN_CSV}")
        print("Extract training.csv from video_action_understanding.zip into data/action/")
        return

    df = pd.read_csv(TRAIN_CSV)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(NUM_CLIPS, 1, figsize=(12, 3.5 * NUM_CLIPS))
    if NUM_CLIPS == 1:
        axes = [axes]

    shown = 0
    for _, row in df.iterrows():
        if shown >= NUM_CLIPS:
            break
        video_id = str(row["video_id"])
        if not (FRAMES_DIR / video_id).is_dir():
            continue
        try:
            frames = load_clip(video_id, int(row["start_frame"]), int(row["stop_frame"]))
        except FileNotFoundError:
            continue
        if len(frames) < 2:
            continue

        action_ids = parse_action_classes(row["all_action_classes"])
        labels = ", ".join(ACTION_NAMES.get(a, str(a)) for a in action_ids)

        ax = axes[shown]
        ax.imshow(_grid(frames))
        ax.set_title(f"Recognition — {video_id} | Action: {labels}", fontsize=11)
        ax.axis("off")
        shown += 1

    if shown == 0:
        print("No clips loaded. Check data/action/video_frames/ and training.csv.")
        return

    fig.suptitle("CathAction — Action Recognition (ground truth)", fontsize=14)
    fig.tight_layout()
    out_path = OUTPUT / "recognition_demo.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def _grid(frames):
    h, w = frames[0].shape[:2]
    row = np.concatenate(
        [cv2.resize(f, (w, h)) for f in frames], axis=1
    )
    return row


if __name__ == "__main__":
    main()
