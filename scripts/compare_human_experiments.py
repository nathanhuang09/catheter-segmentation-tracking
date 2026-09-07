"""Create paired, sequence-aware comparisons of human segmentation experiments."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from human_experiment_utils import sequence_key


METRICS = ("dice", "iou", "precision", "recall")


def locate(path: Path, filename: str, subdirectory: str):
    candidates = (path, path / filename, path / subdirectory / filename)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find {filename} from {path}")


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def clustered_delta_ci(first, second, names, seed=42, repetitions=5000):
    """Bootstrap paired mean difference by resampling whole recordings."""
    groups = {}
    for index, name in enumerate(names):
        groups.setdefault(sequence_key(name), []).append(index)
    group_indices = list(groups.values())
    rng = np.random.default_rng(seed); differences = np.asarray(second) - np.asarray(first)
    samples = np.empty(repetitions)
    for iteration in range(repetitions):
        chosen = rng.integers(0, len(group_indices), len(group_indices))
        indices = np.concatenate([group_indices[index] for index in chosen])
        samples[iteration] = differences[indices].mean()
    return float(np.quantile(samples, .025)), float(np.quantile(samples, .975))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet", type=Path, required=True)
    parser.add_argument("--segformer", type=Path, required=True)
    parser.add_argument("--three-frame", type=Path, required=True)
    parser.add_argument("--temporal-unet", type=Path, required=True)
    parser.add_argument("--kalman", type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    inputs = {"U-Net": args.unet, "SegFormer-B0": args.segformer,
              "3-frame stacked U-Net": args.three_frame,
              "Shared-encoder temporal U-Net": args.temporal_unet}
    model_rows = {}
    for label, root in inputs.items():
        metric_path = locate(root, "test_metrics.csv", "test_evaluation_3frame_matched")
        rows = read_rows(metric_path)
        if len({row["filename"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate filenames in {metric_path}")
        model_rows[label] = {row["filename"]: row for row in rows}
    reference_names = list(model_rows["U-Net"])
    reference_set = set(reference_names)
    for label, rows in model_rows.items():
        if set(rows) != reference_set:
            missing = sorted(reference_set - set(rows))[:3]
            extra = sorted(set(rows) - reference_set)[:3]
            raise ValueError(f"{label} uses a different cohort; missing={missing}, extra={extra}")
    architecture = []
    for label, rows in model_rows.items():
        record = {"model": label, "frames": len(reference_names),
                  "sequences": len({sequence_key(name) for name in reference_names})}
        for metric in METRICS:
            values = np.asarray([float(rows[name][metric]) for name in reference_names])
            record[f"mean_{metric}"] = float(values.mean())
            record[f"median_{metric}"] = float(np.median(values))
        architecture.append(record)
    baseline = model_rows["U-Net"]
    paired = []
    for label, rows in model_rows.items():
        if label == "U-Net": continue
        for metric in METRICS:
            base_values = [float(baseline[name][metric]) for name in reference_names]
            model_values = [float(rows[name][metric]) for name in reference_names]
            low, high = clustered_delta_ci(base_values, model_values, reference_names,
                                           args.seed, args.bootstrap_repetitions)
            paired.append({"comparison": f"{label} minus U-Net", "metric": metric,
                           "mean_paired_delta": float(np.mean(np.asarray(model_values) - base_values)),
                           "cluster_bootstrap_ci_low": low, "cluster_bootstrap_ci_high": high})
    physics = None
    if args.kalman:
        kalman_path = locate(args.kalman, "centerline_metrics.csv", "centerline_kalman_test_3frame_matched")
        kalman = {row["filename"]: row for row in read_rows(kalman_path)}
        if set(kalman) != reference_set:
            raise ValueError("Kalman results do not use the exact architecture-comparison cohort")
        physics = {"frames": len(reference_names),
                   "sequences": len({sequence_key(name) for name in reference_names})}
        comparisons = (("dice", "raw_dice", "centerline_kalman_dice", False),
                       ("centerline_error_pixels", "raw_centerline_error_pixels", "kalman_centerline_error_pixels", True),
                       ("tip_error_pixels", "raw_tip_error_pixels", "kalman_tip_error_pixels", True))
        for label, raw_key, filtered_key, lower_is_better in comparisons:
            valid = [(name, float(row[raw_key]), float(row[filtered_key])) for name, row in kalman.items()
                     if row.get(raw_key) not in (None, "") and row.get(filtered_key) not in (None, "")]
            names = [item[0] for item in valid]; raw = [item[1] for item in valid]; filtered = [item[2] for item in valid]
            low, high = clustered_delta_ci(raw, filtered, names, args.seed, args.bootstrap_repetitions)
            physics[label] = {"valid_frames": len(valid), "raw_mean": float(np.mean(raw)),
                              "kalman_mean": float(np.mean(filtered)),
                              "kalman_minus_raw": float(np.mean(np.asarray(filtered) - raw)),
                              "cluster_bootstrap_ci_low": low, "cluster_bootstrap_ci_high": high,
                              "lower_is_better": lower_is_better}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "architecture_comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=architecture[0].keys()); writer.writeheader(); writer.writerows(architecture)
    with (args.output_dir / "paired_differences_vs_unet.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=paired[0].keys()); writer.writeheader(); writer.writerows(paired)
    result = {"cohort": {"frames": len(reference_names),
                         "sequences": len({sequence_key(name) for name in reference_names}),
                         "exact_filename_match": True},
              "architecture_results": architecture, "paired_differences_vs_unet": paired,
              "physics_results": physics,
              "bootstrap": {"unit": "recording sequence", "repetitions": args.bootstrap_repetitions,
                            "confidence_interval": .95, "seed": args.seed}}
    with (args.output_dir / "comparison_summary.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    print(f"Verified an identical cohort of {len(reference_names):,} frames across all architectures")
    for row in architecture:
        print(f"{row['model']}: Dice={row['mean_dice']:.4f}, IoU={row['mean_iou']:.4f}")
    if physics:
        print(f"Kalman centerline error: {physics['centerline_error_pixels']['raw_mean']:.3f} -> "
              f"{physics['centerline_error_pixels']['kalman_mean']:.3f} pixels")
    print(f"Saved comparison under {args.output_dir}")


if __name__ == "__main__":
    main()
