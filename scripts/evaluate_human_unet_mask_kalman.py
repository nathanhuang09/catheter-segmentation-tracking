"""Evaluate motion-aligned per-pixel Kalman filtering of U-Net masks.

The filter treats each foreground probability as a scalar state. The previous
probability/uncertainty fields are motion-compensated with optical flow before
the current U-Net prediction is incorporated as a noisy measurement.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from human_experiment_utils import ManifestDataset, frame_number, read_names, sequence_key
from train_phantom_unet import UNet


def metrics(prediction, target, epsilon=1e-6):
    prediction = prediction.astype(bool); target = target.astype(bool)
    intersection = np.logical_and(prediction, target).sum()
    predicted = prediction.sum(); actual = target.sum()
    union = np.logical_or(prediction, target).sum()
    return {
        "dice": float((2 * intersection + epsilon) / (predicted + actual + epsilon)),
        "iou": float((intersection + epsilon) / (union + epsilon)),
        "precision": float((intersection + epsilon) / (predicted + epsilon)),
        "recall": float((intersection + epsilon) / (actual + epsilon)),
    }


def warp_to_current(previous, previous_gray, current_gray):
    """Backward-warp a previous field into current-frame coordinates."""
    # For every current pixel, estimate where it came from in the prior frame.
    flow = cv2.calcOpticalFlowFarneback(
        current_gray, previous_gray, None, pyr_scale=.5, levels=3,
        winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    height, width = current_gray.shape
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32),
                                 np.arange(height, dtype=np.float32))
    return cv2.remap(previous.astype(np.float32), grid_x + flow[..., 0],
                     grid_y + flow[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def keep_components(mask, minimum_area):
    if minimum_area <= 0: return mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    kept = np.zeros_like(mask, dtype=bool)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= minimum_area: kept |= labels == label
    return kept


def summarize(rows, prefix):
    result = {}
    for metric in ("dice", "iou", "precision", "recall"):
        values = np.asarray([row[f"{prefix}_{metric}"] for row in rows])
        result[metric] = {"mean": float(values.mean()), "median": float(np.median(values)),
                          "std": float(values.std())}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val",
                        help="Tune on val; use test only after settings are frozen")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--process-noise", type=float, default=.03,
                        help="Uncertainty added between frames (Q)")
    parser.add_argument("--measurement-noise", type=float, default=.12,
                        help="Uncertainty assigned to each U-Net probability (R)")
    parser.add_argument("--initial-variance", type=float, default=.12)
    parser.add_argument("--min-component-area", type=int, default=0,
                        help="Optional connected-component cleanup after thresholding")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.process_noise <= 0 or args.measurement_noise <= 0 or args.initial_variance <= 0:
        raise ValueError("Kalman noise and variance values must be positive")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved = checkpoint.get("arguments", {})
    if checkpoint.get("model_kind", saved.get("model", "unet")) not in (None, "unet"):
        raise ValueError("This experiment requires a single-frame U-Net checkpoint")
    image_size = int(saved.get("image_size", 256)); root = args.checkpoint.resolve().parent
    dataset = ManifestDataset(read_names(root / f"{args.split}_files.csv"), image_size)
    loader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(3, 1, 16).to(device); model.load_state_dict(checkpoint["model_state"]); model.eval()

    frames = {}
    offset = 0
    with torch.no_grad():
        for images, targets in loader:
            probabilities = torch.sigmoid(model(images.to(device))).cpu().numpy()[:, 0]
            images_np = images.numpy(); targets_np = targets.numpy()[:, 0]
            for index in range(len(images)):
                name = dataset.images[offset + index].name
                gray = np.clip(images_np[index].mean(axis=0) * 255, 0, 255).astype(np.uint8)
                frames[name] = (gray, probabilities[index], targets_np[index] > .5)
            offset += len(images)

    groups = {}
    for path in dataset.images: groups.setdefault(sequence_key(path), []).append(path.name)
    rows = []; examples = []
    for group, names in groups.items():
        names.sort(key=frame_number)
        state = variance = previous_gray = None
        for name in names:
            gray, measurement, target = frames[name]
            if state is None:
                state = measurement.copy(); variance = np.full_like(state, args.initial_variance)
            else:
                state = warp_to_current(state, previous_gray, gray)
                variance = warp_to_current(variance, previous_gray, gray) + args.process_noise
                gain = variance / (variance + args.measurement_noise)
                state = np.clip(state + gain * (measurement - state), 0, 1)
                variance = (1 - gain) * variance
            raw_mask = keep_components(measurement >= args.threshold, args.min_component_area)
            filtered_mask = keep_components(state >= args.threshold, args.min_component_area)
            raw = metrics(raw_mask, target); filtered = metrics(filtered_mask, target)
            row = {"filename": name, "sequence": group, "frame": frame_number(name)}
            row.update({f"raw_{key}": value for key, value in raw.items()})
            row.update({f"kalman_{key}": value for key, value in filtered.items()})
            rows.append(row)
            if len(examples) < 12 and (len(rows) == 1 or filtered["dice"] - raw["dice"] != 0):
                examples.append((gray, target, measurement, state.copy(), raw_mask, filtered_mask,
                                 raw["dice"], filtered["dice"]))
            previous_gray = gray

    output = args.output_dir or root / f"mask_kalman_{args.split}"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "segmentation_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    summary = {"split": args.split, "frames": len(rows), "sequences": len(groups),
               "raw": summarize(rows, "raw"), "kalman": summarize(rows, "kalman"),
               "delta_mean_dice": float(np.mean([r["kalman_dice"] - r["raw_dice"] for r in rows])),
               "parameters": {"threshold": args.threshold, "process_noise": args.process_noise,
                              "measurement_noise": args.measurement_noise,
                              "initial_variance": args.initial_variance,
                              "min_component_area": args.min_component_area},
               "checkpoint": str(args.checkpoint), "checkpoint_epoch": checkpoint.get("epoch")}
    with (output / "segmentation_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    if examples:
        fig, axes = plt.subplots(len(examples), 6, figsize=(18, 3 * len(examples)), squeeze=False)
        titles = ("Image", "Ground truth", "Raw probability", "Filtered probability", "Raw mask", "Filtered mask")
        for row, example in enumerate(examples):
            image, target, raw_p, filtered_p, raw_m, filtered_m, raw_dice, filtered_dice = example
            values = (image, target, raw_p, filtered_p, raw_m, filtered_m)
            for axis, value, title in zip(axes[row], values, titles):
                axis.imshow(value, cmap="gray" if title != "Raw probability" and title != "Filtered probability" else "magma",
                            vmin=0, vmax=255 if title == "Image" else 1)
                axis.set_title(title); axis.axis("off")
            axes[row, 4].set_title(f"Raw Dice={raw_dice:.3f}")
            axes[row, 5].set_title(f"Kalman Dice={filtered_dice:.3f}")
        fig.tight_layout(); fig.savefig(output / "mask_comparison.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Evaluated {len(rows):,} frames from {len(groups)} sequences on {device}")
    print(f"Raw Dice:    {summary['raw']['dice']['mean']:.4f}")
    print(f"Kalman Dice: {summary['kalman']['dice']['mean']:.4f}")
    print(f"Mean delta:  {summary['delta_mean_dice']:+.4f}")
    print(f"Saved under {output}")


if __name__ == "__main__":
    main()
