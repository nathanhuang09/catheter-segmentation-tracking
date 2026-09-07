"""Physics-informed temporal filtering of a U-Net catheter centerline."""
import argparse
import csv
import json
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_human_unet_kalman import TipKalman, skeletonize
from human_experiment_utils import ManifestDataset, frame_number, read_names, sequence_key
from train_phantom_unet import UNet


def ordered_centerline(mask, points=24):
    skeleton = skeletonize(mask)
    pixels = [tuple(point) for point in np.column_stack(np.where(skeleton > 0))]  # y, x
    if len(pixels) < 2:
        return None
    pixel_set = set(pixels)
    neighbors = {point: [(point[0] + dy, point[1] + dx)
                         for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                         if (dy or dx) and (point[0] + dy, point[1] + dx) in pixel_set]
                 for point in pixels}
    endpoints = [point for point in pixels if len(neighbors[point]) == 1]
    base = min(endpoints or pixels, key=lambda point: (point[0], point[1]))
    queue = deque([base]); parent = {base: None}; distance = {base: 0.0}
    while queue:
        current = queue.popleft()
        for nxt in neighbors[current]:
            step = float(np.hypot(nxt[0] - current[0], nxt[1] - current[1]))
            candidate = distance[current] + step
            if nxt not in distance or candidate < distance[nxt]:
                distance[nxt] = candidate; parent[nxt] = current; queue.append(nxt)
    candidates = [point for point in endpoints if point in distance and point != base] or list(distance)
    tip = max(candidates, key=distance.get)
    path = []
    while tip is not None:
        path.append(tip); tip = parent[tip]
    path = np.asarray(path[::-1], dtype=np.float64)[:, ::-1]  # x, y
    segments = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(segments)]
    if cumulative[-1] == 0:
        return None
    samples = np.linspace(0, cumulative[-1], points)
    return np.column_stack((np.interp(samples, cumulative, path[:, 0]),
                            np.interp(samples, cumulative, path[:, 1])))


def centerline_distance(first, second):
    if first is None or second is None:
        return None
    distances = np.linalg.norm(first[:, None, :] - second[None, :, :], axis=2)
    return float((distances.min(axis=1).mean() + distances.min(axis=0).mean()) / 2)


def dice(prediction, target):
    intersection = np.logical_and(prediction, target).sum()
    return float((2 * intersection + 1) / (prediction.sum() + target.sum() + 1))


def line_mask(centerline, shape, thickness):
    result = np.zeros(shape, np.uint8)
    if centerline is not None:
        cv2.polylines(result, [np.rint(centerline).astype(np.int32)], False, 1,
                      max(1, int(round(thickness))), cv2.LINE_AA)
    return result > 0


def enforce_segment_lengths(points, segment_lengths):
    """Project an ordered centerline onto an approximately inextensible chain."""
    constrained = points.copy()
    for index, desired in enumerate(segment_lengths, start=1):
        direction = constrained[index] - constrained[index - 1]
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            constrained[index] = constrained[index - 1] + direction * (desired / norm)
    return constrained


def mean(values):
    values = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(values)) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--control-points", type=int, default=24)
    parser.add_argument("--process-noise", type=float, default=1.0)
    parser.add_argument("--measurement-noise", type=float, default=9.0)
    parser.add_argument("--gate", type=float, default=40.0)
    parser.add_argument("--max-speed", type=float, default=12.0)
    parser.add_argument("--length-adaptation", type=float, default=.05,
                        help="Slow update rate for the catheter segment-length constraint")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved = checkpoint.get("arguments", {}); root = args.checkpoint.resolve().parent
    image_size = int(saved.get("image_size", 256))
    dataset = ManifestDataset(read_names(root / f"{args.split}_files.csv"), image_size)
    loader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(3, 1, 16).to(device); model.load_state_dict(checkpoint["model_state"]); model.eval()
    observations = {}; offset = 0
    with torch.no_grad():
        for images, targets in loader:
            probabilities = torch.sigmoid(model(images.to(device))).cpu().numpy()[:, 0]
            for index in range(len(images)):
                name = dataset.images[offset + index].name
                raw_mask = probabilities[index] >= args.threshold
                target_mask = targets[index, 0].numpy() > .5
                observations[name] = (raw_mask, target_mask,
                                      ordered_centerline(raw_mask, args.control_points),
                                      ordered_centerline(target_mask, args.control_points))
            offset += len(images)
    groups = {}
    for path in dataset.images: groups.setdefault(sequence_key(path), []).append(path.name)
    rows = []
    for sequence, names in groups.items():
        filters = [TipKalman(args.process_noise, args.measurement_noise, args.gate, args.max_speed)
                   for _ in range(args.control_points)]
        previous_frame = None; previous_measured = None; missing_run = 0; segment_lengths = None
        for name in sorted(names, key=frame_number):
            frame = frame_number(name); raw_mask, target_mask, measured, truth = observations[name]
            dt = 1 if previous_frame is None else frame - previous_frame
            # Skeleton endpoint identity can flip between frames. Align the ordered
            # measurement to the previous frame before giving it to the motion model.
            if measured is not None and previous_measured is not None:
                forward = np.linalg.norm(measured - previous_measured, axis=1).mean()
                reverse = np.linalg.norm(measured[::-1] - previous_measured, axis=1).mean()
                if reverse < forward:
                    measured = measured[::-1].copy()
            if measured is None:
                missing_run += 1
            else:
                missing_run = 0; previous_measured = measured.copy()
            # Do not extrapolate indefinitely through a lost segmentation.
            if missing_run > 5:
                filters = [TipKalman(args.process_noise, args.measurement_noise, args.gate, args.max_speed)
                           for _ in range(args.control_points)]
                segment_lengths = None
            filtered_points = []; accepted_count = 0
            for index, kalman in enumerate(filters):
                point, accepted = kalman.update(None if measured is None else measured[index], dt)
                accepted_count += int(accepted)
                filtered_points.append(point)
            # A topology change can move many resampled points beyond their gates.
            # Reacquire the observed curve instead of allowing a stale state to diverge.
            if measured is not None and accepted_count < args.control_points // 2:
                filters = [TipKalman(args.process_noise, args.measurement_noise, args.gate, args.max_speed)
                           for _ in range(args.control_points)]
                filtered_points = [kalman.update(measured[index], dt)[0]
                                   for index, kalman in enumerate(filters)]
                segment_lengths = None
            filtered = None if any(point is None for point in filtered_points) else np.asarray(filtered_points)
            if filtered is not None and len(filtered) > 2:
                # Low bending-energy prior followed by an inextensible-chain
                # projection. Lengths adapt slowly to insertion/retraction.
                filtered[1:-1] = .25 * filtered[:-2] + .5 * filtered[1:-1] + .25 * filtered[2:]
                observed = np.linalg.norm(np.diff(measured if measured is not None else filtered, axis=0), axis=1)
                segment_lengths = (observed if segment_lengths is None else
                                   (1 - args.length_adaptation) * segment_lengths +
                                   args.length_adaptation * observed)
                filtered = enforce_segment_lengths(filtered, segment_lengths)
                filtered[:, 0] = np.clip(filtered[:, 0], 0, raw_mask.shape[1] - 1)
                filtered[:, 1] = np.clip(filtered[:, 1], 0, raw_mask.shape[0] - 1)
            raw_length = 0 if measured is None else np.linalg.norm(np.diff(measured, axis=0), axis=1).sum()
            thickness = 1 if raw_length <= 0 else np.clip(raw_mask.sum() / raw_length, 1, 15)
            filtered_mask = line_mask(filtered, raw_mask.shape, thickness)
            rows.append({"filename": name, "sequence": sequence, "frame": frame,
                         "raw_dice": dice(raw_mask, target_mask),
                         "centerline_kalman_dice": dice(filtered_mask, target_mask),
                         "raw_centerline_error_pixels": centerline_distance(measured, truth),
                         "kalman_centerline_error_pixels": centerline_distance(filtered, truth),
                         "raw_tip_error_pixels": None if measured is None or truth is None else float(np.linalg.norm(measured[-1] - truth[-1])),
                         "kalman_tip_error_pixels": None if filtered is None or truth is None else float(np.linalg.norm(filtered[-1] - truth[-1])),
                         "raw_pixels": int(raw_mask.sum()), "filtered_pixels": int(filtered_mask.sum()),
                         "target_pixels": int(target_mask.sum())})
            previous_frame = frame
    output = args.output_dir or root / f"centerline_kalman_{args.split}"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "centerline_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    summary = {"split": args.split, "frames": len(rows), "sequences": len(groups),
               "raw_mean_dice": mean([row["raw_dice"] for row in rows]),
               "centerline_kalman_mean_dice": mean([row["centerline_kalman_dice"] for row in rows]),
               "raw_mean_centerline_error_pixels": mean([row["raw_centerline_error_pixels"] for row in rows]),
               "kalman_mean_centerline_error_pixels": mean([row["kalman_centerline_error_pixels"] for row in rows]),
               "raw_mean_tip_error_pixels": mean([row["raw_tip_error_pixels"] for row in rows]),
               "kalman_mean_tip_error_pixels": mean([row["kalman_tip_error_pixels"] for row in rows]),
               "parameters": vars(args),
               "note": "Mask Dice is secondary: a constant-width mask is reconstructed from the filtered centerline."}
    summary["parameters"]["checkpoint"] = str(args.checkpoint)
    summary["parameters"]["output_dir"] = None if args.output_dir is None else str(args.output_dir)
    with (output / "centerline_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(json.dumps(summary, indent=2)); print(f"Saved under {output}")


if __name__ == "__main__":
    main()
