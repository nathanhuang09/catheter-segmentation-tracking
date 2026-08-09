"""Overlay catheter/guidewire segmentation masks on X-ray images."""
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OUTPUT, SEGMENTATION_DIR

MAX_SAMPLES = 6


def find_image_label_pairs(root: Path):
    for split in ("train", "test"):
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        if not img_dir.is_dir():
            continue
        for img_path in sorted(img_dir.glob("*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lbl_path = lbl_dir / f"{img_path.stem}.png"
            if not lbl_path.exists():
                lbl_path = lbl_dir / f"{img_path.stem}.jpg"
            if lbl_path.exists():
                yield img_path, lbl_path


def overlay_mask(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask_bin = (mask > 0).astype(np.uint8)
    color = np.zeros_like(image_bgr)
    color[:, :, 1] = 180  # green overlay
    out = image_bgr.copy()
    out[mask_bin == 1] = cv2.addWeighted(
        image_bgr, 0.45, color, 0.55, 0
    )[mask_bin == 1]
    return out


def main():
    pairs = list(find_image_label_pairs(SEGMENTATION_DIR))
    if not pairs:
        print(f"No image/label pairs found under {SEGMENTATION_DIR}")
        print("Extract segmentation_human_train.zip into data/segmentation/")
        return

    OUTPUT.mkdir(parents=True, exist_ok=True)
    n = min(MAX_SAMPLES, len(pairs))
    fig, axes = plt.subplots(n, 2, figsize=(10, 3 * n))
    if n == 1:
        axes = np.array([axes])

    for i, (img_path, lbl_path) in enumerate(pairs[:n]):
        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(lbl_path), cv2.IMREAD_GRAYSCALE)
        overlay = overlay_mask(img, mask)
        axes[i, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title("Original X-ray")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[i, 1].set_title("Segmentation overlay")
        axes[i, 1].axis("off")

    fig.suptitle("CathAction — Catheter & Guidewire Segmentation", fontsize=14)
    fig.tight_layout()
    out_path = OUTPUT / "segmentation_demo.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
