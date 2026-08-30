"""Train a small binary U-Net on CathAction phantom segmentation data."""
import argparse
import csv
import random
import sys
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OUTPUT, PHANTOM_SEGMENTATION_DIR


class PhantomDataset(Dataset):
    def __init__(self, split: str, image_size: int = 256):
        self.image_dir = PHANTOM_SEGMENTATION_DIR / split / "images"
        self.label_dir = PHANTOM_SEGMENTATION_DIR / split / "labels"
        self.image_size = image_size
        self.images = sorted(self.image_dir.glob("*.png"))
        if not self.images:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        label_path = self.label_dir / f"{image_path.stem}.npy"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = np.load(label_path, allow_pickle=False)
        if image is None or image.shape[:2] != mask.shape[:2]:
            raise ValueError(f"Invalid image/mask pair: {image_path}, {label_path}")
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(
            (mask > 0).astype(np.uint8),
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST,
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return (
            torch.from_numpy(image.transpose(2, 0, 1)),
            torch.from_numpy(mask.astype(np.float32)).unsqueeze(0),
        )


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    """U-Net with 3 input channels, 1 output channel, and 16 base channels."""
    def __init__(self, in_channels=3, out_channels=1, base_channels=16):
        super().__init__()
        b = base_channels
        self.enc1, self.enc2 = DoubleConv(in_channels, b), DoubleConv(b, b * 2)
        self.enc3, self.enc4 = DoubleConv(b * 2, b * 4), DoubleConv(b * 4, b * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(b * 8, b * 16)
        self.up4, self.dec4 = nn.ConvTranspose2d(b * 16, b * 8, 2, 2), DoubleConv(b * 16, b * 8)
        self.up3, self.dec3 = nn.ConvTranspose2d(b * 8, b * 4, 2, 2), DoubleConv(b * 8, b * 4)
        self.up2, self.dec2 = nn.ConvTranspose2d(b * 4, b * 2, 2, 2), DoubleConv(b * 4, b * 2)
        self.up1, self.dec1 = nn.ConvTranspose2d(b * 2, b, 2, 2), DoubleConv(b * 2, b)
        self.output = nn.Conv2d(b, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        x = self.bottleneck(self.pool(e4))
        x = self.dec4(torch.cat((self.up4(x), e4), 1))
        x = self.dec3(torch.cat((self.up3(x), e3), 1))
        x = self.dec2(torch.cat((self.up2(x), e2), 1))
        x = self.dec1(torch.cat((self.up1(x), e1), 1))
        return self.output(x)


def dice_loss(logits, targets, smooth=1.0):
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum((1, 2, 3))
    denominator = probabilities.sum((1, 2, 3)) + targets.sum((1, 2, 3))
    return (1.0 - (2.0 * intersection + smooth) / (denominator + smooth)).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    losses = [dice_loss(model(x.to(device)), y.to(device)).item() for x, y in loader]
    return float(np.mean(losses))


def make_subsets(dataset, max_train, max_val, seed):
    indices = np.random.default_rng(seed).permutation(len(dataset)).tolist()
    val_count = min(max_val, len(indices) - 1)
    train_count = min(max_train, len(indices) - val_count)
    val_indices = indices[:val_count]
    train_indices = indices[val_count:val_count + train_count]
    return Subset(dataset, train_indices), Subset(dataset, val_indices), train_indices, val_indices


def write_manifest(path, dataset, indices):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["filename"])
        writer.writerows([[dataset.images[index].name] for index in indices])


def read_manifest(path, dataset):
    """Restore exact subset indices from a previous experiment manifest."""
    index_by_name = {image.name: index for index, image in enumerate(dataset.images)}
    with path.open(newline="", encoding="utf-8") as file:
        names = [row["filename"] for row in csv.DictReader(file)]
    missing = [name for name in names if name not in index_by_name]
    if missing:
        raise FileNotFoundError(f"Manifest files are missing from the dataset: {missing[:3]}")
    return [index_by_name[name] for name in names]


def read_history(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return [
            {"epoch": int(row["epoch"]),
             "train_dice_loss": float(row["train_dice_loss"]),
             "val_dice_loss": float(row["val_dice_loss"]),
             "seconds": float(row["seconds"])}
            for row in csv.DictReader(file)
        ]


def save_checkpoint(path, model, optimizer, epoch, val_loss, args):
    torch.save(
        {"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
         "val_dice_loss": val_loss, "arguments": vars(args)},
        path,
    )


def plot_history(history, output_dir):
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [row["train_dice_loss"] for row in history], label="Train")
    plt.plot(epochs, [row["val_dice_loss"] for row in history], label="Validation")
    plt.xlabel("Epoch"); plt.ylabel("Dice loss"); plt.title("Phantom U-Net training")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close()


@torch.no_grad()
def plot_predictions(model, loader, device, output_dir, count=4):
    model.eval()
    images, masks = next(iter(loader))
    probabilities = torch.sigmoid(model(images.to(device))).cpu()
    count = min(count, len(images))
    fig, axes = plt.subplots(count, 4, figsize=(12, 3 * count), squeeze=False)
    for row in range(count):
        image = images[row].permute(1, 2, 0).numpy()
        target = masks[row, 0].numpy()
        probability = probabilities[row, 0].numpy()
        prediction = probability >= 0.5
        for ax, data, title, cmap in zip(
            axes[row], (image, target, probability, prediction),
            ("Image", "Ground truth", "Probability", "Prediction >= 0.5"),
            (None, "gray", "magma", "gray"),
        ):
            ax.imshow(data, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
            ax.set_title(title); ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "predictions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity", action="store_true", help="Use 32 train/16 validation images for 1 epoch")
    parser.add_argument("--max-train", type=int, default=500)
    parser.add_argument("--max-val", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment", default="phantom_unet_subset")
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Parent directory for experiment outputs (for example, mounted Google Drive)",
    )
    parser.add_argument("--resume", type=Path, help="Resume from an experiment checkpoint")
    args = parser.parse_args()
    if args.sanity:
        args.max_train, args.max_val, args.epochs = 32, 16, 1
        args.experiment = "phantom_unet_sanity"
    checkpoint = None
    if args.resume:
        if not args.resume.exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        saved_args = checkpoint["arguments"]
        # Restore all data/model settings. --epochs may extend the original run.
        requested_epochs = args.epochs if "--epochs" in sys.argv else saved_args["epochs"]
        for name in ("max_train", "max_val", "batch_size", "image_size", "seed", "experiment"):
            setattr(args, name, saved_args[name])
        args.epochs = requested_epochs

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    source = PhantomDataset("train", args.image_size)
    if checkpoint:
        output_dir = args.resume.resolve().parent
        train_indices = read_manifest(output_dir / "train_files.csv", source)
        val_indices = read_manifest(output_dir / "val_files.csv", source)
        train_data, val_data = Subset(source, train_indices), Subset(source, val_indices)
    else:
        train_data, val_data, train_indices, val_indices = make_subsets(
            source, args.max_train, args.max_val, args.seed
        )
        output_root = args.output_root.expanduser() if args.output_root else OUTPUT
        output_dir = output_root / args.experiment
        output_dir.mkdir(parents=True, exist_ok=True)
        write_manifest(output_dir / "train_files.csv", source, train_indices)
        write_manifest(output_dir / "val_files.csv", source, val_indices)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, args.batch_size, shuffle=False, num_workers=0)
    model = UNet(3, 1, 16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    history_path = output_dir / "history.csv"
    history = read_history(history_path) if checkpoint else []
    start_epoch = 1
    if checkpoint:
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming {args.resume} after epoch {checkpoint['epoch']}")
    best_val = min((row["val_dice_loss"] for row in history), default=float("inf"))
    print(
        f"device={device}, train={len(train_data)}, val={len(val_data)}, "
        f"epochs={start_epoch}-{args.epochs}"
    )
    if start_epoch > args.epochs:
        print(f"Checkpoint already reached epoch {checkpoint['epoch']}; use --epochs with a larger total.")
        return

    for epoch in range(start_epoch, args.epochs + 1):
        started = time.perf_counter()
        model.train(); losses = []
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = dice_loss(model(images), masks)
            loss.backward(); optimizer.step(); losses.append(loss.item())
        train_loss = float(np.mean(losses))
        val_loss = evaluate(model, val_loader, device)
        seconds = time.perf_counter() - started
        row = {"epoch": epoch, "train_dice_loss": train_loss,
               "val_dice_loss": val_loss, "seconds": seconds}
        history.append(row)
        with history_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=row.keys())
            writer.writeheader(); writer.writerows(history)
        save_checkpoint(output_dir / "last_model.pt", model, optimizer, epoch, val_loss, args)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(output_dir / "best_model.pt", model, optimizer, epoch, val_loss, args)
        plot_history(history, output_dir)
        print(f"epoch={epoch:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} seconds={seconds:.1f}")

    best = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    plot_predictions(model, val_loader, device, output_dir)
    print(f"Saved experiment artifacts under {output_dir}")


if __name__ == "__main__":
    main()
