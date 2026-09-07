"""Shared datasets and utilities for human segmentation experiments."""
import csv
import random
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from train_phantom_unet import SegmentationDataset, make_human_subsets, write_manifest


def read_names(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as file:
        return [row["filename"] for row in csv.DictReader(file)]


def select_names(dataset, names: list[str]) -> None:
    by_name = {path.name: path for path in dataset.images}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise FileNotFoundError(f"Manifest files are missing: {missing[:3]}")
    dataset.images = [by_name[name] for name in names]


def prepare_manifests(source, output_dir: Path, max_train: int, max_val: int,
                      max_test: int, seed: int, manifests_from: Path | None = None):
    """Create splits or copy the exact filenames from another experiment."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if manifests_from:
        result = {split: read_names(manifests_from / f"{split}_files.csv")
                  for split in ("train", "val", "test")}
        available = {path.name for path in source.images}
        for split, names in result.items():
            missing = [name for name in names if name not in available]
            if missing:
                raise FileNotFoundError(f"Missing {split} manifest files: {missing[:3]}")
        split_groups = {split: {sequence_key(name) for name in names} for split, names in result.items()}
        overlaps = {
            "train/val": split_groups["train"] & split_groups["val"],
            "train/test": split_groups["train"] & split_groups["test"],
            "val/test": split_groups["val"] & split_groups["test"],
        }
        bad = {pair: sorted(groups) for pair, groups in overlaps.items() if groups}
        if bad:
            raise ValueError(
                f"Sequence leakage in {manifests_from}: {bad}. "
                "Regenerate the baseline manifests with the corrected splitter."
            )
            with (output_dir / f"{split}_files.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file); writer.writerow(["filename"])
                writer.writerows([[name] for name in names])
        return result
    indices = make_human_subsets(source, max_train, max_val, max_test, seed)
    for split, split_indices in indices.items():
        write_manifest(output_dir / f"{split}_files.csv", source, split_indices)
    return {split: [source.images[index].name for index in split_indices]
            for split, split_indices in indices.items()}


class ManifestDataset(Dataset):
    def __init__(self, names: list[str], image_size: int):
        self.source = SegmentationDataset("human", "train", image_size)
        select_names(self.source, names)
        self.images = self.source.images

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        return self.source[index]


def sequence_key(path_or_name) -> str:
    return re.sub(r"-\d+(?:_\d+)?$", "", Path(path_or_name).stem)


def frame_number(path_or_name) -> int:
    match = re.search(r"-(\d+)(?:_\d+)?$", Path(path_or_name).stem)
    if not match:
        raise ValueError(f"No frame number in {path_or_name}")
    return int(match.group(1))


class ThreeFrameDataset(Dataset):
    """Causal grayscale channels at t-4, t-2, and t; target is the mask at t."""
    def __init__(self, names: list[str], image_size: int, drop_incomplete: bool = True):
        source = SegmentationDataset("human", "train", image_size)
        all_by_sequence = {}
        for path in source.images:
            all_by_sequence.setdefault(sequence_key(path), {})[frame_number(path)] = path
        selected = {name for name in names}
        self.samples = []
        for path in source.images:
            if path.name not in selected:
                continue
            frames = all_by_sequence[sequence_key(path)]
            t = frame_number(path)
            if drop_incomplete and (t - 4 not in frames or t - 2 not in frames):
                continue
            earliest = frames[min(frames)]
            self.samples.append((frames.get(t - 4, earliest), frames.get(t - 2, earliest), path))
        self.images = [sample[-1] for sample in self.samples]
        self.image_size = image_size
        self.label_dir = source.label_dir

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        paths = self.samples[index]
        channels = []
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not read {path}")
            image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
            channels.append(image.astype(np.float32) / 255.0)
        label_path = self.label_dir / f"{paths[-1].stem}_mask.png"
        mask = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read {label_path}")
        mask = cv2.resize((mask > 0).astype(np.uint8), (self.image_size, self.image_size),
                          interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy(np.stack(channels)), torch.from_numpy(mask.astype(np.float32))[None]


def dice_bce_loss(logits, targets, bce_weight=0.0, smooth=1.0):
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum((1, 2, 3))
    denominator = probabilities.sum((1, 2, 3)) + targets.sum((1, 2, 3))
    dice = (1.0 - (2.0 * intersection + smooth) / (denominator + smooth)).mean()
    if bce_weight <= 0:
        return dice
    return dice + bce_weight * F.binary_cross_entropy_with_logits(logits, targets)


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
