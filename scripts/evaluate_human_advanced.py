"""Evaluate SegFormer or three-frame U-Net on held-out human sequences."""
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_phantom_unet import binary_metrics, summarize
from human_experiment_utils import ManifestDataset, ThreeFrameDataset, read_names
from human_models import build_model


@torch.no_grad()
def scored_examples(model, dataset, rows, kind, device, threshold):
    """Load worst/median/best non-empty test examples ranked by Dice."""
    candidates = [index for index, row in enumerate(rows) if row["target_pixels"] > 0]
    if not candidates:
        candidates = list(range(len(rows)))
    ordered = sorted(candidates, key=lambda index: rows[index]["dice"])
    selected = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    examples = []
    for label, index in zip(("Worst", "Median", "Best"), selected):
        image, target = dataset[index]
        probability = torch.sigmoid(model(image.unsqueeze(0).to(device)))[0, 0].cpu().numpy()
        shown = image[-1].numpy() if kind == "unet3" else image.permute(1, 2, 0).numpy()
        examples.append((label, rows[index]["filename"], rows[index]["dice"],
                         shown, target[0].numpy(), probability, probability >= threshold))
    return examples


def plot_scored_examples(examples, path):
    fig, axes = plt.subplots(len(examples), 5, figsize=(15, 3 * len(examples)), squeeze=False)
    for row, (label, filename, dice, image, target, probability, prediction) in enumerate(examples):
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        overlay = image.copy()
        overlay[prediction] = .45 * overlay[prediction] + .55 * np.array([0, 1, 0])
        items = ((image, f"{label}: {filename}\nDice={dice:.3f}", None),
                 (target, "Ground truth", "gray"), (probability, "Probability", "magma"),
                 (prediction, "Prediction", "gray"), (overlay, "Prediction overlay", None))
        for axis, (data, title, cmap) in zip(axes[row], items):
            axis.imshow(data, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
            axis.set_title(title); axis.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def main(required_model=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--manifests-from", type=Path,
                        help="Fallback directory containing test_files.csv")
    parser.add_argument("--eligible-3frame-only", action="store_true",
                        help="Evaluate only frames with complete t-4,t-2,t context for matched comparisons")
    parser.add_argument("--save-probabilities", action="store_true",
                        help="Save float16 probability maps and filenames for reproducible analysis")
    parser.add_argument("--qc-csv", type=Path,
                        help="Prediction-blind annotation_qc.csv; writes a secondary unflagged sensitivity result")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved = checkpoint["arguments"]; kind = checkpoint.get("model_kind", saved.get("model")) or "unet"
    if required_model and kind != required_model:
        raise ValueError(f"Expected a {required_model} checkpoint, found {kind}")
    image_size = int(saved["image_size"]); model_name = saved.get("model_name", "nvidia/mit-b0")
    local_manifest = args.checkpoint.resolve().parent / "test_files.csv"
    saved_manifest_root = saved.get("manifests_from")
    fallback_root = args.manifests_from or (Path(saved_manifest_root) if saved_manifest_root else None)
    if local_manifest.exists():
        manifest = local_manifest
    elif fallback_root and (fallback_root / "test_files.csv").exists():
        manifest = fallback_root / "test_files.csv"
        print(f"Using fallback test manifest: {manifest}")
    else:
        raise FileNotFoundError(
            f"No test manifest at {local_manifest}. Pass --manifests-from pointing "
            "to the U-Net baseline experiment containing test_files.csv."
        )
    names = read_names(manifest)
    if args.eligible_3frame_only:
        eligible = ThreeFrameDataset(names, image_size)
        names = [path.name for path in eligible.images]
    dataset = ThreeFrameDataset(names, image_size) if kind == "unet3" else ManifestDataset(names, image_size)
    loader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(kind, model_name).to(device); model.load_state_dict(checkpoint["model_state"]); model.eval()
    rows = []; offset = 0; saved_probabilities = []
    with torch.no_grad():
        for images, targets in loader:
            probabilities = torch.sigmoid(model(images.to(device))).cpu(); predictions = probabilities >= args.threshold
            if args.save_probabilities:
                saved_probabilities.append(probabilities[:, 0].numpy().astype(np.float16))
            metrics = binary_metrics(predictions, targets)
            for index in range(len(images)):
                row = {"filename": dataset.images[offset + index].name}
                for key in ("dice", "iou", "precision", "recall"):
                    row[key] = float(metrics[key][index])
                row["predicted_pixels"] = int(metrics["predicted_pixels"][index])
                row["target_pixels"] = int(metrics["target_pixels"][index]); rows.append(row)
            offset += len(images)
    default_name = "test_evaluation_3frame_matched" if args.eligible_3frame_only else "test_evaluation"
    output = args.output_dir or args.checkpoint.resolve().parent / default_name; output.mkdir(parents=True, exist_ok=True)
    with (output / "test_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    summary = summarize(rows); summary.update({"model": kind, "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint["epoch"], "threshold": args.threshold, "image_size": image_size,
        "device": str(device), "split": "human/held-out-sequences",
        "eligible_3frame_only": args.eligible_3frame_only, "manifest": str(manifest)})
    with (output / "test_summary.json").open("w", encoding="utf-8") as file: json.dump(summary, file, indent=2)
    if args.qc_csv:
        with args.qc_csv.open(newline="", encoding="utf-8") as file:
            flagged = {row["filename"] for row in csv.DictReader(file) if row.get("flags")}
        retained = [row for row in rows if row["filename"] not in flagged]
        if not retained:
            raise ValueError("QC criteria excluded every evaluated frame")
        sensitivity = summarize(retained)
        sensitivity.update({"primary_result": False, "retained_frames": len(retained),
                            "excluded_flagged_frames": len(rows) - len(retained),
                            "qc_csv": str(args.qc_csv),
                            "note": "Secondary sensitivity analysis; primary result is the untouched test set."})
        with (output / "test_summary_qc_sensitivity.json").open("w", encoding="utf-8") as file:
            json.dump(sensitivity, file, indent=2)
    examples = scored_examples(model, dataset, rows, kind, device, args.threshold)
    plot_scored_examples(examples, output / "test_predictions.png")
    if args.save_probabilities:
        np.savez_compressed(output / "test_probabilities.npz",
                            filenames=np.asarray([row["filename"] for row in rows]),
                            probabilities=np.concatenate(saved_probabilities))
    print(f"Evaluated {len(dataset):,} {kind} test frames on {device}")
    print(f"Dice={summary['dice']['mean']:.4f} IoU={summary['iou']['mean']:.4f}")
    print(f"Saved under {output}")


if __name__ == "__main__":
    main()
