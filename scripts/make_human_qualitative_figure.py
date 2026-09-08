"""Create an honest qualitative figure from a held-out human U-Net result."""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from human_experiment_utils import ManifestDataset, ThreeFrameDataset, read_names
from human_models import build_model


def read_metrics(path):
    with path.open(newline="", encoding="utf-8") as file:
        return {row["filename"]: row for row in csv.DictReader(file)}


def predict(model, dataset, name, device, threshold):
    by_name = {path.name: index for index, path in enumerate(dataset.images)}
    image, target = dataset[by_name[name]]
    with torch.no_grad():
        probability = torch.sigmoid(model(image.unsqueeze(0).to(device)))[0, 0].cpu().numpy()
    return image.permute(1, 2, 0).numpy(), target[0].numpy() > .5, probability >= threshold


def color_overlay(image, target, prediction):
    overlay = image.copy()
    true_positive = target & prediction
    false_positive = ~target & prediction
    false_negative = target & ~prediction
    for mask, color in ((true_positive, np.array([0., 1., 0.])),
                        (false_positive, np.array([1., 0., 0.])),
                        (false_negative, np.array([0., .35, 1.]))):
        overlay[mask] = .25 * overlay[mask] + .75 * color
    return overlay


def draw_row(axes, image, target, prediction, filename, dice, order, titles=True,
             style="masks", show_row_label=True):
    if style == "overlay-errors":
        prediction_overlay = image.copy()
        prediction_overlay[prediction] = .25 * prediction_overlay[prediction] + .75 * np.array([0., 1., 0.])
        panels = {
            "input": (image, "Input fluoroscopy", None),
            "prediction": (prediction_overlay, "U-Net prediction (green)", None),
            "ground_truth": (color_overlay(image, target, prediction),
                             "Errors: TP green, FP red, FN blue", None),
        }
    else:
        panels = {
            "input": (image, "Input fluoroscopy", None),
            "prediction": (prediction, "U-Net prediction", "gray"),
            "ground_truth": (target, "Ground truth (evaluation only)", "gray"),
        }
    letters = "ABC"
    for index, (axis, key) in enumerate(zip(axes, order)):
        data, title, cmap = panels[key]
        axis.imshow(data, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None,
                    interpolation="nearest" if cmap else "antialiased")
        if titles:
            axis.set_title(f"({letters[index]}) {title}", fontsize=11, pad=8)
        axis.set_xticks([]); axis.set_yticks([])
        for spine in axis.spines.values(): spine.set_visible(False)
    if show_row_label:
        axes[0].set_ylabel(f"{filename}\nDice={dice:.3f}", fontsize=8, rotation=90, labelpad=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metrics", type=Path,
                        help="Matched test_metrics.csv; defaults beside the checkpoint")
    parser.add_argument("--filename", help="Chosen matched-cohort filename; omit to create candidate preview")
    parser.add_argument("--min-dice", type=float, default=.80)
    parser.add_argument("--max-dice", type=float, default=.90)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--expected-count", type=int, default=580)
    parser.add_argument("--order", choices=("input-prediction-ground_truth", "input-ground_truth-prediction"),
                        default="input-prediction-ground_truth")
    parser.add_argument("--style", choices=("masks", "overlay-errors"), default="masks",
                        help="Use original-backed color overlays instead of black/white mask panels")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved = checkpoint.get("arguments", {}); kind = checkpoint.get("model_kind", saved.get("model", "unet"))
    if kind != "unet":
        raise ValueError(f"This figure requires the augmented single-frame U-Net, found {kind}")
    image_size = int(saved.get("image_size", 256))
    if image_size != 256:
        raise ValueError(f"Reported figure requires 256x256 preprocessing, checkpoint uses {image_size}")
    root = args.checkpoint.resolve().parent
    metric_path = args.metrics or root / "test_evaluation_3frame_matched" / "test_metrics.csv"
    metrics = read_metrics(metric_path)
    manifest_names = read_names(root / "test_files.csv")
    matched_names = [path.name for path in ThreeFrameDataset(manifest_names, image_size).images]
    if len(matched_names) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} matched frames, found {len(matched_names)}")
    if set(metrics) != set(matched_names):
        raise ValueError("Matched metrics filenames do not exactly match the reconstructed eligible cohort")
    dataset = ManifestDataset(matched_names, image_size)
    candidates = [name for name in matched_names
                  if args.min_dice <= float(metrics[name]["dice"]) <= args.max_dice
                  and int(metrics[name]["target_pixels"]) > 0]
    if not candidates:
        raise ValueError(f"No nonempty targets have Dice in [{args.min_dice}, {args.max_dice}]")
    # Evenly sample the score-ordered interval instead of quietly cherry-picking
    # only its highest-scoring frames.
    candidates.sort(key=lambda name: float(metrics[name]["dice"]))
    positions = np.linspace(0, len(candidates) - 1, min(args.candidates, len(candidates))).round().astype(int)
    preview_names = [candidates[position] for position in positions]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model("unet").to(device); model.load_state_dict(checkpoint["model_state"]); model.eval()
    output = args.output_dir or root / "qualitative_figure"
    output.mkdir(parents=True, exist_ok=True)
    order = tuple(args.order.split("-"))
    if args.filename:
        if args.filename not in matched_names:
            raise ValueError(f"{args.filename} is not in the matched {len(matched_names)}-frame cohort")
        row = metrics[args.filename]; score = float(row["dice"])
        if int(row["target_pixels"]) <= 0 or not args.min_dice <= score <= args.max_dice:
            raise ValueError(f"Chosen frame is not a nonempty-target Dice {args.min_dice:.2f}-{args.max_dice:.2f} candidate")
        image, target, prediction = predict(model, dataset, args.filename, device, args.threshold)
        fig, axes = plt.subplots(1, 3, figsize=(9.3, 3.15), gridspec_kw={"wspace": .055})
        draw_row(axes, image, target, prediction, args.filename, score, order,
                 style=args.style, show_row_label=False)
        fig.subplots_adjust(left=.01, right=.995, top=.88, bottom=.02)
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"selected_successful_example.{suffix}", dpi=300,
                        bbox_inches="tight", facecolor="white")
        plt.close(fig)
        metadata = {"description": "Selected successful example; not representative of the entire dataset.",
                    "filename": args.filename, "dice": score, "target_pixels": int(row["target_pixels"]),
                    "predicted_pixels": int(row["predicted_pixels"]), "threshold": args.threshold,
                    "image_size": image_size, "matched_cohort_frames": len(matched_names),
                    "checkpoint": str(args.checkpoint), "checkpoint_epoch": checkpoint.get("epoch"),
                    "panel_order": order, "style": args.style,
                    "overlay_colors": {"true_positive": "green", "false_positive": "red",
                                       "false_negative": "blue"} if args.style == "overlay-errors" else None,
                    "annotations_modified": False, "prediction_postprocessed": False}
        with (output / "selected_successful_example.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)
        print(f"Confirmed {args.filename} belongs to the matched {len(matched_names)}-frame test cohort")
        print(f"Saved selected successful example (Dice={score:.4f}) under {output}")
    else:
        fig, axes = plt.subplots(len(preview_names), 3, figsize=(9.3, 2.8 * len(preview_names)),
                                 squeeze=False, gridspec_kw={"wspace": .055, "hspace": .20})
        for row_index, name in enumerate(preview_names):
            image, target, prediction = predict(model, dataset, name, device, args.threshold)
            draw_row(axes[row_index], image, target, prediction, name, float(metrics[name]["dice"]),
                     order, titles=row_index == 0, style=args.style)
        fig.subplots_adjust(left=.12, right=.995, top=.96, bottom=.01)
        fig.savefig(output / "candidate_preview.png", dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        with (output / "candidate_examples.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file); writer.writerow(("filename", "dice", "target_pixels", "predicted_pixels"))
            for name in preview_names:
                writer.writerow((name, metrics[name]["dice"], metrics[name]["target_pixels"],
                                 metrics[name]["predicted_pixels"]))
        print(f"Confirmed matched cohort: {len(matched_names)} frames")
        print(f"Found {len(candidates)} nonempty candidates with Dice {args.min_dice:.2f}-{args.max_dice:.2f}")
        for name in preview_names: print(f"{name}: Dice={float(metrics[name]['dice']):.4f}")
        print(f"Preview saved to {output / 'candidate_preview.png'}")


if __name__ == "__main__":
    main()
