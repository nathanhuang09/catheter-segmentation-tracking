"""Create a compact diagram of the two three-frame temporal U-Net paths."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from human_experiment_utils import ThreeFrameDataset, read_names


def centroid(mask):
    ys, xs = np.where(mask > 0)
    return None if not len(xs) else np.array([xs.mean(), ys.mean()])


def load_mask(dataset, path):
    mask = cv2.imread(str(dataset.label_dir / f"{path.stem}_mask.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None: raise FileNotFoundError(f"Missing mask for {path.name}")
    return cv2.resize((mask > 0).astype(np.uint8), (dataset.image_size, dataset.image_size),
                      interpolation=cv2.INTER_NEAREST)


def choose_sample(dataset, filename, min_motion, max_motion, min_pixels):
    if filename:
        for index, sample in enumerate(dataset.samples):
            if sample[-1].name == filename: return index, None
        raise ValueError(f"{filename} is not an eligible current frame in the selected manifest")
    choices = []
    for index, paths in enumerate(dataset.samples):
        first, current = load_mask(dataset, paths[0]), load_mask(dataset, paths[-1])
        if first.sum() < min_pixels or current.sum() < min_pixels: continue
        first_center, current_center = centroid(first), centroid(current)
        motion = float(np.linalg.norm(current_center - first_center))
        if min_motion <= motion <= max_motion:
            choices.append((motion, index))
    if not choices:
        raise ValueError("No eligible triplet met the requested motion and mask-size bounds")
    motion, index = max(choices)
    return index, motion


def motion_crop(paths, dataset, padding=30):
    union = np.zeros((dataset.image_size, dataset.image_size), dtype=bool)
    for path in paths: union |= load_mask(dataset, path) > 0
    ys, xs = np.where(union)
    if not len(xs): return (0, dataset.image_size, 0, dataset.image_size)
    center_x, center_y = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    side = int(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) + 2 * padding)
    side = min(max(side, dataset.image_size // 3), dataset.image_size)
    x0 = int(round(center_x - side / 2)); y0 = int(round(center_y - side / 2))
    x0 = min(max(x0, 0), dataset.image_size - side); y0 = min(max(y0, 0), dataset.image_size - side)
    return y0, y0 + side, x0, x0 + side


def box(axis, xy, width, height, text, color, fontsize=10):
    patch = FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.015,rounding_size=0.02",
                           linewidth=1.4, edgecolor=color, facecolor="white")
    axis.add_patch(patch)
    axis.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center",
              fontsize=fontsize, color="#172033")
    return patch


def arrow(axis, start, end, color="#52606d"):
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                  linewidth=1.4, color=color, shrinkA=2, shrinkB=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests-from", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--filename", help="Optional current-frame filename t")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--min-motion", type=float, default=5.0)
    parser.add_argument("--max-motion", type=float, default=30.0)
    parser.add_argument("--min-mask-pixels", type=int, default=50)
    parser.add_argument("--crop-padding", type=int, default=30,
                        help="Pixels around the union of the three ground-truth masks")
    parser.add_argument("--full-frame", action="store_true", help="Disable the shared square motion crop")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    names = read_names(args.manifests_from / f"{args.split}_files.csv")
    dataset = ThreeFrameDataset(names, args.image_size)
    index, selection_motion = choose_sample(dataset, args.filename, args.min_motion,
                                             args.max_motion, args.min_mask_pixels)
    frames, _ = dataset[index]; paths = dataset.samples[index]
    crop = ((0, args.image_size, 0, args.image_size) if args.full_frame else
            motion_crop(paths, dataset, args.crop_padding))
    first_center = centroid(load_mask(dataset, paths[0])); current_center = centroid(load_mask(dataset, paths[-1]))
    measured_motion = None if first_center is None or current_center is None else float(np.linalg.norm(current_center - first_center))
    output = args.output_dir or args.manifests_from / "slide_figures" / "temporal_unet"
    output.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 6.6), facecolor="white")
    positions = ((.08, .60, .24, .36), (.38, .60, .24, .36), (.68, .60, .24, .36))
    labels = (r"Frame $t-4$", r"Frame $t-2$", r"Current frame $t$")
    for frame, position, label in zip(frames, positions, labels):
        axis = fig.add_axes(position)
        y0, y1, x0, x1 = crop
        axis.imshow(frame.numpy()[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=1)
        axis.set_title(label, fontsize=14, pad=7); axis.set_xticks([]); axis.set_yticks([])
        for spine in axis.spines.values(): spine.set_edgecolor("#8898aa"); spine.set_linewidth(1)
    overlay = fig.add_axes((0, 0, 1, 1)); overlay.set_xlim(0, 1); overlay.set_ylim(0, 1); overlay.axis("off")
    # Time arrows above the two model paths.
    arrow(overlay, (.32, .78), (.38, .78)); arrow(overlay, (.62, .78), (.68, .78))
    overlay.text(.50, .565, "Same three causal frames", ha="center", fontsize=11, color="#52606d")
    arrow(overlay, (.50, .58), (.50, .52))
    # Lightweight channel-stacked path.
    box(overlay, (.04, .36), .19, .11, "Stack frames\nas 3 channels", "#3478b8")
    box(overlay, (.30, .36), .15, .11, "Standard\nU-Net", "#3478b8")
    box(overlay, (.51, .36), .16, .11, r"Mask at $t$", "#3478b8")
    arrow(overlay, (.23, .415), (.30, .415), "#3478b8"); arrow(overlay, (.45, .415), (.51, .415), "#3478b8")
    overlay.text(.04, .49, "Channel-stacked temporal U-Net", fontsize=12, weight="bold", color="#245b8a")
    # Shared-encoder, motion-sensitive path.
    box(overlay, (.04, .10), .19, .12, "Shared encoder\nfor each frame", "#7a4eb2")
    box(overlay, (.29, .10), .24, .12, "Multi-scale fusion\n+ feature differences", "#7a4eb2")
    box(overlay, (.59, .10), .14, .12, "U-Net\ndecoder", "#7a4eb2")
    box(overlay, (.79, .10), .16, .12, r"Mask at $t$", "#7a4eb2")
    arrow(overlay, (.23, .16), (.29, .16), "#7a4eb2"); arrow(overlay, (.53, .16), (.59, .16), "#7a4eb2")
    arrow(overlay, (.73, .16), (.79, .16), "#7a4eb2")
    overlay.text(.04, .245, "Shared-encoder temporal U-Net", fontsize=12, weight="bold", color="#603a91")
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"temporal_unet_paths.{suffix}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    metadata = {"split": args.split, "filenames": [path.name for path in paths],
                "timestamps": ["t-4", "t-2", "t"], "image_size": args.image_size,
                "ground_truth_centroid_displacement_pixels": measured_motion,
                "selection": ("user-selected current frame" if args.filename else
                              "largest bounded ground-truth centroid motion satisfying stated criteria"),
                "selection_bounds": {"min_motion": args.min_motion, "max_motion": args.max_motion,
                                     "min_mask_pixels": args.min_mask_pixels},
                "shared_crop_yxyx": list(crop), "full_frame": args.full_frame,
                "note": "Ground truth was used only to choose a visibly moving triplet; masks are not model inputs."}
    with (output / "temporal_unet_paths.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print("Sequential frames:"); [print(f"  {label}: {path.name}") for label, path in zip(metadata["timestamps"], paths)]
    print(f"Ground-truth centroid displacement t-4 to t: {measured_motion:.2f} pixels")
    print(f"Saved temporal diagram under {output}")


if __name__ == "__main__":
    main()
