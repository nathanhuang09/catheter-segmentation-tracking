"""Show frames BEFORE an action starts — anticipation task (ground-truth next action)."""
import ast
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    ACTION_NAMES,
    ANTICIPATION_SEC_BEFORE,
    FPS,
    FRAMES_DIR,
    OUTPUT,
    TRAIN_CSV,
)

NUM_CLIPS = 3
FRAMES_PER_CLIP = 8


def parse_action_classes(raw) -> list[int]:
    if isinstance(raw, list):
        return [int(x) for x in raw]
    if isinstance(raw, str):
        return [int(x) for x in ast.literal_eval(raw)]
    return [int(raw)]


def frame_path(video_id: str, frame_idx: int) -> Path:
    vid_dir = FRAMES_DIR / video_id
    for pattern in (f"{frame_idx:06d}.jpg", f"{frame_idx:06d}.png", f"frame_{frame_idx}.jpg"):
        p = vid_dir / pattern
        if p.exists():
            return p
    matches = list(vid_dir.glob(f"*{frame_idx}*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No frame {frame_idx} in {vid_dir}")


def anticipation_window(start_frame: int) -> tuple[int, int]:
    offset = int(ANTICIPATION_SEC_BEFORE * FPS)
    stop = max(0, start_frame - 1)
    start = max(0, stop - offset)
    return start, stop


def load_clip(video_id: str, start: int, stop: int, n: int = FRAMES_PER_CLIP):
    if stop <= start:
        return []
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

        start_f = int(row["start_frame"])
        win_start, win_stop = anticipation_window(start_f)
        try:
            frames = load_clip(video_id, win_start, win_stop)
        except FileNotFoundError:
            continue
        if len(frames) < 2:
            continue

        action_ids = parse_action_classes(row["all_action_classes"])
        next_action = ", ".join(ACTION_NAMES.get(a, str(a)) for a in action_ids)

        ax = axes[shown]
        row_img = np.concatenate(frames, axis=1)
        ax.imshow(row_img)
        ax.set_title(
            f"Anticipation — frames {win_start}-{win_stop} (before action at {start_f})\n"
            f"Next action (ground truth): {next_action}",
            fontsize=10,
            color="darkgreen",
        )
        ax.axis("off")
        shown += 1

    if shown == 0:
        print("No anticipation clips loaded. Check video_frames/ paths.")
        return

    fig.suptitle(
        f"CathAction — Action Anticipation ({ANTICIPATION_SEC_BEFORE}s before onset)",
        fontsize=14,
    )
    fig.tight_layout()
    out_path = OUTPUT / "anticipation_demo.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
