"""Compare raw U-Net tip measurements with physics-constrained Kalman tracking."""
import argparse
import csv
import json
import math
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


def largest_component(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1: return np.zeros_like(mask, dtype=np.uint8)
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == label).astype(np.uint8)


def skeletonize(mask):
    """Morphological skeletonization using OpenCV only."""
    image = largest_component(mask); skeleton = np.zeros_like(image)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(image):
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(image, opened))
        image = cv2.erode(image, element)
    return skeleton


def mask_tip(mask):
    """Derive a reproducible endpoint: farthest skeleton endpoint from the topmost endpoint."""
    skeleton = skeletonize(mask)
    if not skeleton.any(): return None
    neighbors = cv2.filter2D(skeleton, cv2.CV_16S, np.ones((3, 3), np.int16))
    ys, xs = np.where((skeleton > 0) & (neighbors == 2))  # self + one neighbor
    if len(xs) < 2:
        ys, xs = np.where(skeleton > 0)
        if not len(xs): return None
    points = np.column_stack((xs, ys)).astype(np.float64)
    base = points[np.lexsort((points[:, 0], points[:, 1]))[0]]
    return points[np.argmax(np.linalg.norm(points - base, axis=1))]


class TipKalman:
    def __init__(self, process_noise=1.0, measurement_noise=9.0, gate=40.0, max_speed=12.0):
        self.q, self.r, self.gate, self.max_speed = process_noise, measurement_noise, gate, max_speed
        self.x = None; self.P = None

    def update(self, measurement, dt=1.0):
        if self.x is None:
            if measurement is None: return None, False
            self.x = np.array([measurement[0], measurement[1], 0., 0.])
            self.P = np.diag([self.r, self.r, 25., 25.]); return self.x[:2].copy(), True
        dt = max(float(dt), 1.0)
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1.]])
        G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        self.x = F @ self.x; self.P = F @ self.P @ F.T + self.q * (G @ G.T)
        accepted = False
        if measurement is not None:
            z = np.asarray(measurement, dtype=float); innovation = z - self.x[:2]
            speed = np.linalg.norm(innovation) / dt
            if np.linalg.norm(innovation) <= self.gate and speed <= self.max_speed:
                H = np.array([[1., 0, 0, 0], [0, 1., 0, 0]])
                S = H @ self.P @ H.T + self.r * np.eye(2)
                K = self.P @ H.T @ np.linalg.inv(S)
                self.x = self.x + K @ (z - H @ self.x); self.P = (np.eye(4) - K @ H) @ self.P
                accepted = True
        speed_now = np.linalg.norm(self.x[2:])
        if speed_now > self.max_speed: self.x[2:] *= self.max_speed / speed_now
        return self.x[:2].copy(), accepted


def mean(values):
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val",
                        help="Tune on val; report test only after parameters are frozen")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--process-noise", type=float, default=1.0)
    parser.add_argument("--measurement-noise", type=float, default=9.0)
    parser.add_argument("--gate", type=float, default=40.0)
    parser.add_argument("--max-speed", type=float, default=12.0, help="Maximum pixels per source frame")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved = checkpoint.get("arguments", {})
    if saved.get("dataset", "human") != "human": raise ValueError("A human U-Net checkpoint is required")
    image_size = int(saved.get("image_size", 256)); root = args.checkpoint.resolve().parent
    dataset = ManifestDataset(read_names(root / f"{args.split}_files.csv"), image_size)
    loader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(3, 1, 16).to(device); model.load_state_dict(checkpoint["model_state"]); model.eval()
    measurements = {}
    offset = 0
    with torch.no_grad():
        for images, targets in loader:
            probabilities = torch.sigmoid(model(images.to(device))).cpu().numpy()[:, 0]
            for index in range(len(images)):
                name = dataset.images[offset + index].name
                measurements[name] = (mask_tip(probabilities[index] >= args.threshold),
                                      mask_tip(targets[index, 0].numpy() > .5))
            offset += len(images)
    groups = {}
    for path in dataset.images: groups.setdefault(sequence_key(path), []).append(path.name)
    rows = []
    for group, names in groups.items():
        names.sort(key=frame_number); kalman = TipKalman(args.process_noise, args.measurement_noise,
                                                         args.gate, args.max_speed)
        previous_frame = None; previous_raw = None; previous_filtered = None
        for name in names:
            frame = frame_number(name); raw, truth = measurements[name]
            filtered, accepted = kalman.update(raw, 1 if previous_frame is None else frame - previous_frame)
            raw_error = None if raw is None or truth is None else float(np.linalg.norm(raw - truth))
            filtered_error = None if filtered is None or truth is None else float(np.linalg.norm(filtered - truth))
            raw_jitter = None if raw is None or previous_raw is None else float(np.linalg.norm(raw - previous_raw))
            filtered_jitter = None if filtered is None or previous_filtered is None else float(np.linalg.norm(filtered - previous_filtered))
            rows.append({"filename": name, "sequence": group, "frame": frame,
                "ground_truth_x": None if truth is None else truth[0], "ground_truth_y": None if truth is None else truth[1],
                "raw_x": None if raw is None else raw[0], "raw_y": None if raw is None else raw[1],
                "filtered_x": None if filtered is None else filtered[0], "filtered_y": None if filtered is None else filtered[1],
                "measurement_accepted": accepted, "raw_error_pixels": raw_error,
                "filtered_error_pixels": filtered_error, "raw_jitter_pixels": raw_jitter,
                "filtered_jitter_pixels": filtered_jitter})
            previous_frame = frame
            if raw is not None: previous_raw = raw
            if filtered is not None: previous_filtered = filtered
    output = args.output_dir or root / f"kalman_{args.split}"; output.mkdir(parents=True, exist_ok=True)
    with (output / "tip_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    summary = {"split": args.split, "frames": len(rows), "sequences": len(groups),
        "raw_mean_error_pixels": mean([r["raw_error_pixels"] for r in rows]),
        "kalman_mean_error_pixels": mean([r["filtered_error_pixels"] for r in rows]),
        "raw_mean_jitter_pixels": mean([r["raw_jitter_pixels"] for r in rows]),
        "kalman_mean_jitter_pixels": mean([r["filtered_jitter_pixels"] for r in rows]),
        "accepted_measurements": sum(bool(r["measurement_accepted"]) for r in rows),
        "parameters": {"threshold": args.threshold, "process_noise": args.process_noise,
                       "measurement_noise": args.measurement_noise, "gate": args.gate,
                       "max_speed": args.max_speed}}
    with (output / "tip_summary.json").open("w", encoding="utf-8") as file: json.dump(summary, file, indent=2)
    raw_errors = [r["raw_error_pixels"] for r in rows if r["raw_error_pixels"] is not None]
    kalman_errors = [r["filtered_error_pixels"] for r in rows if r["filtered_error_pixels"] is not None]
    plt.figure(figsize=(7, 4)); plt.hist(raw_errors, bins=40, alpha=.55, label="Raw U-Net")
    plt.hist(kalman_errors, bins=40, alpha=.55, label="Kalman"); plt.xlabel("Tip error (pixels)")
    plt.ylabel("Frames"); plt.legend(); plt.tight_layout(); plt.savefig(output / "tip_error_histogram.png", dpi=150); plt.close()
    print(json.dumps(summary, indent=2)); print(f"Saved under {output}")


if __name__ == "__main__":
    main()
