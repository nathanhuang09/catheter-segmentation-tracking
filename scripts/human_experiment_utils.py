"""Shared datasets and utilities for human segmentation experiments."""
import csv
import json
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


def recording_prefix(path_or_name) -> str:
    return Path(path_or_name).stem.split("_", 1)[0]


def make_prefix_stratified_subsets(dataset, max_train, max_val, max_test, seed):
    """Keep sequences intact while approximately balancing recording prefixes."""
    groups = {}
    for index, image in enumerate(dataset.images):
        groups.setdefault(sequence_key(image), []).append(index)
    prefix_groups = {}
    for key, indices in groups.items():
        prefix_groups.setdefault(recording_prefix(dataset.images[indices[0]]), []).append(key)
    total_frames = len(dataset)
    selected = {"train": [], "val": [], "test": []}
    rng = random.Random(seed)
    for prefix in sorted(prefix_groups):
        keys = prefix_groups[prefix][:]
        rng.shuffle(keys)
        prefix_frames = sum(len(groups[key]) for key in keys)
        targets = {
            "test": round(max_test * prefix_frames / total_frames),
            "val": round(max_val * prefix_frames / total_frames),
            "train": round(max_train * prefix_frames / total_frames),
        }
        for split in ("test", "val", "train"):
            start_count = len(selected[split])
            while keys and len(selected[split]) - start_count < targets[split]:
                remaining = targets[split] - (len(selected[split]) - start_count)
                # Choose the recording that lands closest to the requested frame count.
                # The preceding shuffle provides deterministic tie-breaking.
                best_position = min(range(len(keys)), key=lambda pos: abs(len(groups[keys[pos]]) - remaining))
                selected[split].extend(groups[keys.pop(best_position)])
    if not all(selected.values()):
        raise ValueError("Not enough sequences for non-empty stratified splits")
    return selected


def split_summary(names_by_split):
    summary = {}
    for split, names in names_by_split.items():
        prefixes = {}
        for name in names:
            prefix = recording_prefix(name)
            prefixes[prefix] = prefixes.get(prefix, 0) + 1
        summary[split] = {
            "frames": len(names),
            "sequences": len({sequence_key(name) for name in names}),
            "prefix_frames": prefixes,
            "prefix_percent": {key: round(100 * value / len(names), 2)
                               for key, value in prefixes.items()},
        }
    return summary


def prepare_manifests(source, output_dir: Path, max_train: int, max_val: int,
                      max_test: int, seed: int, manifests_from: Path | None = None,
                      split_strategy: str = "prefix_stratified"):
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
        for split, names in result.items():
            with (output_dir / f"{split}_files.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file); writer.writerow(["filename"])
                writer.writerows([[name] for name in names])
        with (output_dir / "split_summary.json").open("w", encoding="utf-8") as file:
            json.dump({"strategy": "reused", "source": str(manifests_from),
                       "splits": split_summary(result)}, file, indent=2)
        return result
    indices = (
        make_prefix_stratified_subsets(source, max_train, max_val, max_test, seed)
        if split_strategy == "prefix_stratified"
        else make_human_subsets(source, max_train, max_val, max_test, seed)
    )
    for split, split_indices in indices.items():
        write_manifest(output_dir / f"{split}_files.csv", source, split_indices)
    result = {split: [source.images[index].name for index in split_indices]
              for split, split_indices in indices.items()}
    with (output_dir / "split_summary.json").open("w", encoding="utf-8") as file:
        json.dump({"strategy": split_strategy, "seed": seed, "splits": split_summary(result)}, file, indent=2)
    return result


def augment_segmentation_pair(image: torch.Tensor, mask: torch.Tensor):
    """Moderate paired augmentation; geometry is identical for image and mask."""
    image_np = image.numpy().transpose(1, 2, 0)
    mask_np = mask[0].numpy()
    height, width = mask_np.shape
    if random.random() < 0.5:
        angle = random.uniform(-10.0, 10.0)
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        image_np = cv2.warpAffine(image_np, matrix, (width, height), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT_101)
        mask_np = cv2.warpAffine(mask_np, matrix, (width, height), flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if image_np.ndim == 2:
            image_np = image_np[..., None]
    if random.random() < 0.5:
        image_np = np.ascontiguousarray(image_np[:, ::-1])
        mask_np = np.ascontiguousarray(mask_np[:, ::-1])
    if random.random() < 0.5:
        contrast = random.uniform(0.85, 1.15)
        brightness = random.uniform(-0.08, 0.08)
        image_np = np.clip(image_np * contrast + brightness, 0, 1)
    if random.random() < 0.25:
        gamma = random.uniform(0.85, 1.15)
        image_np = np.clip(image_np, 0, 1) ** gamma
    if random.random() < 0.2:
        image_np = np.clip(image_np + np.random.normal(0, random.uniform(0.005, 0.025), image_np.shape), 0, 1)
    if random.random() < 0.15:
        image_np = cv2.GaussianBlur(image_np, (3, 3), 0)
        if image_np.ndim == 2:
            image_np = image_np[..., None]
    if random.random() < 0.15:
        for _ in range(random.randint(1, 3)):
            box_w = random.randint(max(2, width // 50), max(3, width // 12))
            box_h = random.randint(max(2, height // 50), max(3, height // 12))
            x, y = random.randrange(width - box_w + 1), random.randrange(height - box_h + 1)
            image_np[y:y + box_h, x:x + box_w] = float(image_np.mean())
    image_out = torch.from_numpy(np.ascontiguousarray(image_np.transpose(2, 0, 1))).float()
    mask_out = torch.from_numpy(np.ascontiguousarray(mask_np))[None].float()
    return image_out, mask_out


class ManifestDataset(Dataset):
    def __init__(self, names: list[str], image_size: int, augment: bool = False):
        self.source = SegmentationDataset("human", "train", image_size)
        select_names(self.source, names)
        self.images = self.source.images
        self.augment = augment

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        image, mask = self.source[index]
        return augment_segmentation_pair(image, mask) if self.augment else (image, mask)


def sequence_key(path_or_name) -> str:
    return re.sub(r"-\d+(?:_\d+)?$", "", Path(path_or_name).stem)


def frame_number(path_or_name) -> int:
    match = re.search(r"-(\d+)(?:_\d+)?$", Path(path_or_name).stem)
    if not match:
        raise ValueError(f"No frame number in {path_or_name}")
    return int(match.group(1))


class ThreeFrameDataset(Dataset):
    """Causal grayscale channels at t-4, t-2, and t; target is the mask at t."""
    def __init__(self, names: list[str], image_size: int, drop_incomplete: bool = True,
                 augment: bool = False):
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
        self.augment = augment

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
        image_tensor = torch.from_numpy(np.stack(channels))
        mask_tensor = torch.from_numpy(mask.astype(np.float32))[None]
        return augment_segmentation_pair(image_tensor, mask_tensor) if self.augment else (image_tensor, mask_tensor)


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
