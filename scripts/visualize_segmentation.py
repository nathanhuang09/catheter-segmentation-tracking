"""Overlay catheter/guidewire segmentation masks on X-ray images."""
import argparse
from pathlib import Path
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OUTPUT, PHANTOM_SEGMENTATION_DIR, SEGMENTATION_DIR

MAX_SAMPLES = 10


def find_image_label_pairs(root: Path):
    for split in ("train", "test"):
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        if not img_dir.is_dir():
            continue
        for img_path in sorted(img_dir.glob("*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            candidates = (
                lbl_dir / f"{img_path.stem}.npy",
                lbl_dir / f"{img_path.stem}.png",
                lbl_dir / f"{img_path.stem}_mask.png",
                lbl_dir / f"{img_path.stem}.jpg",
                lbl_dir / f"{img_path.stem}_mask.jpg",
            )
            lbl_path = next((path for path in candidates if path.exists()), None)
            if lbl_path is not None:
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


def load_mask(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("human", "phantom"), default="human")
    args = parser.parse_args()
    root = PHANTOM_SEGMENTATION_DIR if args.dataset == "phantom" else SEGMENTATION_DIR
    pairs = list(find_image_label_pairs(root))
    if not pairs:
        print(f"No image/label pairs found under {root}")
        return

    OUTPUT.mkdir(parents=True, exist_ok=True)
    n = min(MAX_SAMPLES, len(pairs))
    fig, axes = plt.subplots(n, 2, figsize=(10, 3 * n))
    if n == 1:
        axes = np.array([axes])

    for i, (img_path, lbl_path) in enumerate(pairs[:n]):
        img = cv2.imread(str(img_path))
        mask = load_mask(lbl_path)
        if img is None or mask is None:
            raise RuntimeError(f"Could not read {img_path} or {lbl_path}")
        if img.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Image/mask size mismatch: {img_path.name} {img.shape[:2]} vs "
                f"{lbl_path.name} {mask.shape[:2]}"
            )
        overlay = overlay_mask(img, mask)
        axes[i, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title("Original X-ray")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[i, 1].set_title("Segmentation overlay")
        axes[i, 1].axis("off")

    fig.suptitle(f"CathAction — {args.dataset.title()} Segmentation", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    filename = "segmentation_demo.png" if args.dataset == "human" else "segmentation_phantom_demo.png"
    out_path = OUTPUT / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
