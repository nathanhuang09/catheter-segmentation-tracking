"""Create a vertical slide figure showing actual paired training augmentation."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from human_experiment_utils import ManifestDataset, read_names


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests-from", type=Path, required=True,
                        help="Experiment containing train_files.csv")
    parser.add_argument("--filename", help="Optional filename from the training manifest")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--angle", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    names = read_names(args.manifests_from / "train_files.csv")
    dataset = ManifestDataset(names, args.image_size)
    if args.filename:
        if args.filename not in names:
            raise ValueError(f"{args.filename} is not in the training manifest")
        index = next(i for i, path in enumerate(dataset.images) if path.name == args.filename)
    else:
        # Choose a clear, nonempty annotation near the upper quartile of mask area.
        areas = [int(dataset[i][1].sum()) for i in range(len(dataset))]
        target_area = float(np.quantile([area for area in areas if area > 0], .75))
        index = min((i for i, area in enumerate(areas) if area > 0),
                    key=lambda i: abs(areas[i] - target_area))
    image_tensor, mask_tensor = dataset[index]
    image = image_tensor.permute(1, 2, 0).numpy(); mask = mask_tensor[0].numpy() > .5
    appearance = appearance_augmentation(image, args.seed)
    rotated_image, rotated_mask = rotate_pair(image, mask, args.angle)
    output = args.output_dir or args.manifests_from / "slide_figures" / "augmentation"
    output.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(4.5, 9.3), facecolor="white")
    grid = fig.add_gridspec(3, 2, height_ratios=(1, 1, 1), hspace=.20, wspace=.06)
    top = fig.add_subplot(grid[0, :]); middle = fig.add_subplot(grid[1, :])
    bottom_image = fig.add_subplot(grid[2, 0]); bottom_mask = fig.add_subplot(grid[2, 1])
    panels = ((top, image, "Original training image", None),
              (middle, appearance, "Intensity, noise, and blur", None),
              (bottom_image, rotated_image, f"Rotation ({args.angle:g}°)", None),
              (bottom_mask, rotated_mask, "Transformed mask", "gray"))
    for axis, data, title, cmap in panels:
        axis.imshow(data, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None,
                    interpolation="nearest" if cmap else "antialiased")
        axis.set_title(title, fontsize=12, pad=7); axis.set_xticks([]); axis.set_yticks([])
        for spine in axis.spines.values(): spine.set_visible(False)
    fig.subplots_adjust(left=.02, right=.98, top=.97, bottom=.01)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"training_augmentation_vertical.{suffix}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    metadata = {"filename": dataset.images[index].name, "split": "train",
                "image_size": args.image_size, "rotation_degrees": args.angle,
                "appearance_example": {"contrast": 1.10, "brightness": .04,
                                       "gaussian_noise_std": .012, "blur_kernel": 3},
                "seed": args.seed, "mask_interpolation": "nearest",
                "note": "Deterministic examples within the training augmentation policy; no synthetic anatomy."}
    with (output / "training_augmentation_vertical.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print(f"Confirmed training image: {dataset.images[index].name}")
    print(f"Saved vertical augmentation figure under {output}")


if __name__ == "__main__":
    main()
