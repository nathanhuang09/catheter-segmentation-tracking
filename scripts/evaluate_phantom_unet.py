"""Evaluate a trained binary U-Net once on the official phantom test split."""
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
from train_phantom_unet import PhantomDataset, UNet


def binary_metrics(prediction, target, epsilon=1e-6):
    prediction = prediction.float()
    target = target.float()
    intersection = (prediction * target).sum((1, 2, 3))
    predicted = prediction.sum((1, 2, 3))
    actual = target.sum((1, 2, 3))
    union = predicted + actual - intersection
    return {
        "dice": (2 * intersection + epsilon) / (predicted + actual + epsilon),
        "iou": (intersection + epsilon) / (union + epsilon),
        "precision": (intersection + epsilon) / (predicted + epsilon),
        "recall": (intersection + epsilon) / (actual + epsilon),
        "predicted_pixels": predicted,
        "target_pixels": actual,
    }


def summarize(rows):
    summary = {"num_images": len(rows)}
    for metric in ("dice", "iou", "precision", "recall"):
        values = np.array([row[metric] for row in rows], dtype=np.float64)
        summary[metric] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "std": float(values.std()),
            "p10": float(np.percentile(values, 10)),
            "p90": float(np.percentile(values, 90)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


@torch.no_grad()
def collect_examples(model, dataset, device, rows):
    ordered = sorted(range(len(rows)), key=lambda index: rows[index]["dice"])
    selected = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    examples = []
    for label, index in zip(("Worst", "Median", "Best"), selected):
        image, target = dataset[index]
        probability = torch.sigmoid(model(image.unsqueeze(0).to(device)))[0, 0].cpu().numpy()
        prediction = probability >= 0.5
        examples.append((label, image.permute(1, 2, 0).numpy(), target[0].numpy(),
                         probability, prediction, rows[index]["dice"]))
    return examples


def plot_examples(examples, path):
    fig, axes = plt.subplots(len(examples), 5, figsize=(15, 3 * len(examples)), squeeze=False)
    for row, (label, image, target, probability, prediction, dice) in enumerate(examples):
        overlay = image.copy()
        overlay[prediction] = 0.45 * overlay[prediction] + 0.55 * np.array([0, 1, 0])
        items = (
            (image, f"{label} image\nDice={dice:.3f}", None),
            (target, "Ground truth", "gray"),
            (probability, "Probability", "magma"),
            (prediction, "Prediction >= 0.5", "gray"),
            (overlay, "Prediction overlay", None),
        )
        for axis, (data, title, cmap) in zip(axes[row], items):
            axis.imshow(data, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
            axis.set_title(title)
            axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    image_size = int(checkpoint.get("arguments", {}).get("image_size", 256))
    dataset = PhantomDataset("test", image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = UNet(3, 1, 16).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    rows = []
    offset = 0
    with torch.no_grad():
        for images, targets in loader:
            probabilities = torch.sigmoid(model(images.to(device))).cpu()
            predictions = probabilities >= args.threshold
            metrics = binary_metrics(predictions, targets)
            for batch_index in range(len(images)):
                rows.append({
                    "filename": dataset.images[offset + batch_index].name,
                    "dice": float(metrics["dice"][batch_index]),
                    "iou": float(metrics["iou"][batch_index]),
                    "precision": float(metrics["precision"][batch_index]),
                    "recall": float(metrics["recall"][batch_index]),
                    "predicted_pixels": int(metrics["predicted_pixels"][batch_index]),
                    "target_pixels": int(metrics["target_pixels"][batch_index]),
                })
            offset += len(images)

    output_dir = args.output_dir or args.checkpoint.resolve().parent / "test_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "test_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    summary.update({
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "image_size": image_size,
        "threshold": args.threshold,
        "device": str(device),
        "split": "phantom/test",
    })
    with (output_dir / "test_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    plot_examples(collect_examples(model, dataset, device, rows), output_dir / "test_predictions.png")

    print(f"Evaluated {len(dataset):,} phantom test images on {device}")
    print(f"Dice: {summary['dice']['mean']:.4f} mean, {summary['dice']['median']:.4f} median")
    print(f"IoU:  {summary['iou']['mean']:.4f} mean, {summary['iou']['median']:.4f} median")
    print(f"Saved results under {output_dir}")


if __name__ == "__main__":
    main()
