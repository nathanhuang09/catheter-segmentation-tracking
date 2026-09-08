"""Create a vertical slide figure showing actual paired training augmentation."""
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
from human_experiment_utils import ManifestDataset, read_names, sequence_key


def rotate_pair(image, mask, angle):
    height, width = mask.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    rotated_image = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT_101)
    rotated_mask = cv2.warpAffine(mask.astype(np.uint8), matrix, (width, height),
                                  flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=0).astype(bool)
    return rotated_image, rotated_mask


def appearance_augmentation(image, seed):
    rng = np.random.default_rng(seed)
    augmented = np.clip(image * 1.10 + .04, 0, 1)
    augmented = np.clip(augmented + rng.normal(0, .012, augmented.shape), 0, 1)
    return cv2.GaussianBlur(augmented.astype(np.float32), (3, 3), 0)


def visibility_score(image, mask):
    """Favor a coherent annotation that contrasts with its immediate surroundings."""
    area = int(mask.sum())
    if area < 50: return -np.inf
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(float)
    dilated = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    ring = dilated & ~mask
    if not ring.any(): return -np.inf
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    largest_ratio = 1.0 if count <= 1 else float(stats[1:, cv2.CC_STAT_AREA].max() / area)
    local_contrast = abs(float(gray[mask].mean() - gray[ring].mean()))
    return local_contrast * np.sqrt(min(area, 1500)) * largest_ratio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests-from", type=Path, required=True,
                        help="Experiment containing train_files.csv")
    parser.add_argument("--filename", help="Optional filename from the training manifest")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--angle", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview-candidates", type=int, default=0,
                        help="Create a contact sheet of ranked examples, then choose with --filename")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    names = read_names(args.manifests_from / "train_files.csv")
    dataset = ManifestDataset(names, args.image_size)
    output = args.output_dir or args.manifests_from / "slide_figures" / "augmentation"
    output.mkdir(parents=True, exist_ok=True)
    ranked = None
    if not args.filename or args.preview_candidates:
        scored = []
        for candidate in range(len(dataset)):
            candidate_image, candidate_mask = dataset[candidate]
            score = visibility_score(candidate_image.permute(1, 2, 0).numpy(),
                                     candidate_mask[0].numpy() > .5)
            scored.append((score, candidate))
        ranked = sorted((item for item in scored if np.isfinite(item[0])), reverse=True)
    if args.preview_candidates:
        selected = []; used_sequences = set()
        for item in ranked:
            key = sequence_key(dataset.images[item[1]])
            if key in used_sequences: continue
            selected.append(item); used_sequences.add(key)
            if len(selected) >= args.preview_candidates: break
        fig, axes = plt.subplots(len(selected), 4, figsize=(12, 2.8 * len(selected)),
                                 squeeze=False, gridspec_kw={"hspace": .18, "wspace": .04})
        preview_rows = []
        for row_index, (score, candidate) in enumerate(selected):
            image_tensor, mask_tensor = dataset[candidate]
            image = image_tensor.permute(1, 2, 0).numpy(); mask = mask_tensor[0].numpy() > .5
            appearance = appearance_augmentation(image, args.seed)
            rotated_image, rotated_mask = rotate_pair(image, mask, args.angle)
            items = ((image, "Original", None), (appearance, "Appearance", None),
                     (rotated_image, f"Rotation ({args.angle:g} deg)", None),
                     (rotated_mask, "Rotated mask", "gray"))
            for column, (data, title, cmap) in enumerate(items):
                axis = axes[row_index, column]
                axis.imshow(data, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None,
                            interpolation="nearest" if cmap else "antialiased")
                if row_index == 0: axis.set_title(title, fontsize=11, pad=6)
                axis.set_xticks([]); axis.set_yticks([])
                for spine in axis.spines.values(): spine.set_visible(False)
            axes[row_index, 0].set_ylabel(dataset.images[candidate].name, fontsize=7, labelpad=6)
            preview_rows.append({"filename": dataset.images[candidate].name,
                                 "visibility_score": score, "mask_pixels": int(mask.sum())})
        fig.subplots_adjust(left=.17, right=.995, top=.97, bottom=.01)
        fig.savefig(output / "augmentation_candidate_preview.png", dpi=200,
                    bbox_inches="tight", facecolor="white"); plt.close(fig)
        with (output / "augmentation_candidates.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=preview_rows[0].keys())
            writer.writeheader(); writer.writerows(preview_rows)
        print(f"Saved {len(selected)} ranked augmentation candidates under {output}")
        for row in preview_rows: print(f"{row['filename']}: mask={row['mask_pixels']} px")
        return
    if args.filename:
        if args.filename not in names:
            raise ValueError(f"{args.filename} is not in the training manifest")
        index = next(i for i, path in enumerate(dataset.images) if path.name == args.filename)
    else:
        index = ranked[0][1]
    image_tensor, mask_tensor = dataset[index]
    image = image_tensor.permute(1, 2, 0).numpy(); mask = mask_tensor[0].numpy() > .5
    appearance = appearance_augmentation(image, args.seed)
    rotated_image, rotated_mask = rotate_pair(image, mask, args.angle)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 7.1), facecolor="white",
                             gridspec_kw={"hspace": .14, "wspace": .06})
    panels = ((axes[0, 0], image, "Original training image", None),
              (axes[0, 1], appearance, "Intensity, noise, and blur", None),
              (axes[1, 0], rotated_image, f"Rotation ({args.angle:g}°)", None),
              (axes[1, 1], rotated_mask, "Corresponding rotated mask", "gray"))
    for axis, data, title, cmap in panels:
        axis.imshow(data, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None,
                    interpolation="nearest" if cmap else "antialiased")
        axis.set_title(title, fontsize=12, pad=7); axis.set_xticks([]); axis.set_yticks([])
        for spine in axis.spines.values(): spine.set_visible(False)
    fig.subplots_adjust(left=.01, right=.99, top=.95, bottom=.01)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"training_augmentation_2x2.{suffix}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    metadata = {"filename": dataset.images[index].name, "split": "train",
                "image_size": args.image_size, "rotation_degrees": args.angle,
                "appearance_example": {"contrast": 1.10, "brightness": .04,
                                       "gaussian_noise_std": .012, "blur_kernel": 3},
                "seed": args.seed, "mask_interpolation": "nearest",
                "note": "Deterministic examples within the training augmentation policy; no synthetic anatomy."}
    with (output / "training_augmentation_2x2.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print(f"Confirmed training image: {dataset.images[index].name}")
    print(f"Saved 2x2 augmentation figure under {output}")


if __name__ == "__main__":
    main()
