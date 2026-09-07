"""Prediction-blind audit of human segmentation masks and temporal consistency."""
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from human_experiment_utils import ManifestDataset, frame_number, read_names, sequence_key


def mask_stats(mask):
    binary = (mask > 0).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    area = int(binary.sum())
    if area == 0:
        return area, 0, None, None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return area, count - 1, centroids[largest], int(stats[largest, cv2.CC_STAT_AREA])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests-from", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--tiny-pixels", type=int, default=20,
                        help="Flag masks at or below this area after resizing")
    parser.add_argument("--area-ratio", type=float, default=4.0)
    parser.add_argument("--centroid-jump", type=float, default=35.0)
    parser.add_argument("--max-examples", type=int, default=30)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    names = read_names(args.manifests_from / f"{args.split}_files.csv")
    dataset = ManifestDataset(names, args.image_size)
    by_name = {path.name: index for index, path in enumerate(dataset.images)}
    groups = {}
    for name in names:
        groups.setdefault(sequence_key(name), []).append(name)
    rows = []
    for sequence, sequence_names in sorted(groups.items()):
        previous_area = None; previous_centroid = None; previous_frame = None
        for name in sorted(sequence_names, key=frame_number):
            _, target = dataset[by_name[name]]
            mask = target[0].numpy()
            area, components, centroid, largest = mask_stats(mask)
            frame = frame_number(name)
            ratio = None if previous_area in (None, 0) or area == 0 else max(area / previous_area, previous_area / area)
            jump = None if centroid is None or previous_centroid is None else float(np.linalg.norm(centroid - previous_centroid))
            flags = []
            if area == 0: flags.append("empty")
            elif area <= args.tiny_pixels: flags.append("tiny")
            if ratio is not None and ratio >= args.area_ratio: flags.append("abrupt_area_change")
            if jump is not None and jump >= args.centroid_jump: flags.append("centroid_jump")
            rows.append({"filename": name, "sequence": sequence, "frame": frame,
                         "mask_pixels": area, "mask_fraction": area / (args.image_size ** 2),
                         "components": components, "largest_component_pixels": largest,
                         "area_change_ratio": ratio, "centroid_jump_pixels": jump,
                         "frame_gap": None if previous_frame is None else frame - previous_frame,
                         "flags": ";".join(flags)})
            previous_area, previous_centroid, previous_frame = area, centroid, frame
    output = args.output_dir or args.manifests_from / f"annotation_audit_{args.split}"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "annotation_qc.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    flagged = [row for row in rows if row["flags"]]
    summary = {"split": args.split, "frames": len(rows), "sequences": len(groups),
               "flagged_frames": len(flagged), "flagged_fraction": len(flagged) / len(rows),
               "criteria": {"tiny_pixels": args.tiny_pixels, "area_ratio": args.area_ratio,
                            "centroid_jump": args.centroid_jump},
               "note": "Prediction-blind screening flags are not proof of annotation error."}
    with (output / "annotation_qc_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    shown_rows = sorted(flagged, key=lambda row: (row["mask_pixels"], row["filename"]))[:args.max_examples]
    if shown_rows:
        fig, axes = plt.subplots(len(shown_rows), 3, figsize=(11, 3 * len(shown_rows)), squeeze=False)
        for axes_row, row in zip(axes, shown_rows):
            image, target = dataset[by_name[row["filename"]]]
            image = image.permute(1, 2, 0).numpy(); mask = target[0].numpy() > .5
            overlay = image.copy(); overlay[mask] = .4 * overlay[mask] + .6 * np.array([1, 0, 0])
            for axis, data, title, cmap in ((axes_row[0], image, row["filename"], None),
                                             (axes_row[1], mask, f"GT: {row['mask_pixels']} px", "gray"),
                                             (axes_row[2], overlay, row["flags"], None)):
                axis.imshow(data, cmap=cmap); axis.set_title(title); axis.axis("off")
        fig.tight_layout(); fig.savefig(output / "annotation_qc_montage.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(json.dumps(summary, indent=2)); print(f"Saved under {output}")


if __name__ == "__main__":
    main()
