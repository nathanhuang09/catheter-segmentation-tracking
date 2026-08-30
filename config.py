"""Paths for CathAction demo — edit these if your folders differ."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUT = ROOT / "outputs"

SEGMENTATION_DIR = DATA / "segmentation"
PHANTOM_SEGMENTATION_DIR = SEGMENTATION_DIR / "phantom"
ACTION_DIR = DATA / "action"
FRAMES_DIR = ACTION_DIR / "video_frames"
TRAIN_CSV = ACTION_DIR / "training.csv"
VAL_CSV = ACTION_DIR / "validation.csv"

ACTION_NAMES = {
    0: "advance catheter",
    1: "retract catheter",
    2: "advance guidewire",
    3: "retract guidewire",
    4: "rotate",
}

# Anticipation: use frames this many seconds BEFORE action starts
ANTICIPATION_SEC_BEFORE = 1.0
FPS = 24
