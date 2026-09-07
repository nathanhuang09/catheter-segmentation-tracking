"""Train SegFormer or causal three-frame U-Net on identical human splits."""
import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OUTPUT
from human_experiment_utils import (
    ManifestDataset, ThreeFrameDataset, dice_bce_loss, prepare_manifests,
    read_names, seed_everything,
)
from human_models import build_model
from train_phantom_unet import SegmentationDataset


def make_dataset(kind, names, image_size, augment=False):
    return (ThreeFrameDataset(names, image_size, augment=augment) if kind in {"unet3", "temporal_unet"}
            else ManifestDataset(names, image_size, augment=augment))


@torch.no_grad()
def validation_loss(model, loader, device, bce_weight):
    model.eval(); losses = []
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        losses.append(dice_bce_loss(model(images), masks, bce_weight).item())
    return float(np.mean(losses))


def write_history(path, history):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader(); writer.writerows(history)


def plot_history(history, path):
    plt.figure(figsize=(7, 4))
    plt.plot([r["epoch"] for r in history], [r["train_loss"] for r in history], label="Train")
    plt.plot([r["epoch"] for r in history], [r["val_loss"] for r in history], label="Validation")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()


@torch.no_grad()
def plot_predictions(model, loader, device, path, model_kind):
    model.eval(); images, masks = next(iter(loader)); probabilities = torch.sigmoid(model(images.to(device))).cpu()
    count = min(4, len(images)); fig, axes = plt.subplots(count, 4, figsize=(12, 3 * count), squeeze=False)
    for row in range(count):
        shown = images[row, -1].numpy() if model_kind in {"unet3", "temporal_unet"} else images[row].permute(1, 2, 0).numpy()
        if shown.ndim == 2:
            shown = np.repeat(shown[..., None], 3, axis=2)
        items = ((shown, "Current image", None), (masks[row, 0], "Ground truth", "gray"),
                 (probabilities[row, 0], "Probability", "magma"),
                 (probabilities[row, 0] >= .5, "Prediction", "gray"))
        for axis, (data, title, cmap) in zip(axes[row], items):
            axis.imshow(data, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
            axis.set_title(title); axis.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def parse_args(default_model=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("unet", "segformer", "unet3", "temporal_unet"),
                        default=default_model, required=default_model is None)
    parser.add_argument("--model-name", default="nvidia/mit-b0")
    parser.add_argument("--max-train", type=int, default=3500)
    parser.add_argument("--max-val", type=int, default=600)
    parser.add_argument("--max-test", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--bce-weight", type=float, default=0.0,
                        help="0 keeps Dice-only parity with the U-Net baseline")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--manifests-from", type=Path,
                        help="Reuse train/val/test CSVs from the U-Net experiment")
    parser.add_argument("--split-strategy", choices=("prefix_stratified", "sequence_random"),
                        default="prefix_stratified")
    parser.add_argument("--augment", action="store_true",
                        help="Apply moderate paired geometry/intensity augmentation to training only")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--sanity", action="store_true")
    return parser.parse_args()


def main(default_model=None):
    args = parse_args(default_model)
    if args.experiment is None:
        defaults = {"unet": "human_unet_baseline", "segformer": "human_segformer_b0",
                    "unet3": "human_unet_3frame", "temporal_unet": "human_temporal_unet"}
        args.experiment = defaults[args.model]
    if args.sanity:
        args.max_train, args.max_val, args.max_test, args.epochs = 64, 32, 32, 1
        args.experiment += "_sanity"
    seed_everything(args.seed)
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        saved = checkpoint["arguments"]
        requested_epochs = args.epochs if "--epochs" in sys.argv else saved["epochs"]
        for key, value in saved.items():
            if hasattr(args, key) and key != "epochs": setattr(args, key, value)
        args.epochs = requested_epochs
        output_dir = args.resume.resolve().parent
    else:
        output_dir = (args.output_root.expanduser() if args.output_root else OUTPUT) / args.experiment
    source = SegmentationDataset("human", "train", args.image_size)
    if checkpoint:
        manifests = {split: read_names(output_dir / f"{split}_files.csv") for split in ("train", "val", "test")}
    else:
        manifests = prepare_manifests(source, output_dir, args.max_train, args.max_val,
                                      args.max_test, args.seed, args.manifests_from,
                                      args.split_strategy)
    train_data = make_dataset(args.model, manifests["train"], args.image_size, augment=args.augment)
    val_data = make_dataset(args.model, manifests["val"], args.image_size)
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_data, args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, args.model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    start_epoch = 1
    best_val = float("inf"); stale = 0
    if checkpoint:
        model.load_state_dict(checkpoint["model_state"]); optimizer.load_state_dict(checkpoint["optimizer_state"])
        if checkpoint.get("scheduler_state"): scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint["epoch"] + 1; best_val = checkpoint.get("best_val_loss", checkpoint["val_loss"])
        history_path = output_dir / "history.csv"
        if history_path.exists():
            with history_path.open(newline="", encoding="utf-8") as file:
                history = [{"epoch": int(r["epoch"]), "train_loss": float(r["train_loss"]),
                            "val_loss": float(r["val_loss"]), "lr": float(r["lr"]),
                            "seconds": float(r["seconds"])} for r in csv.DictReader(file)]
    print(f"model={args.model} device={device} parameters={sum(p.numel() for p in model.parameters()):,} "
          f"train={len(train_data)} val={len(val_data)} epochs={start_epoch}-{args.epochs}")
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.perf_counter(); model.train(); losses = []
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device); optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = dice_bce_loss(model(images), masks, args.bce_weight)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); losses.append(loss.item())
        train_loss = float(np.mean(losses)); val_loss = validation_loss(model, val_loader, device, args.bce_weight)
        lr = optimizer.param_groups[0]["lr"]; scheduler.step()
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "lr": lr, "seconds": time.perf_counter() - started})
        write_history(output_dir / "history.csv", history); plot_history(history, output_dir / "loss_curve.png")
        improved = val_loss < best_val - args.min_delta
        if improved: best_val, stale = val_loss, 0
        else: stale += 1
        state = {"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                 "scheduler_state": scheduler.state_dict(), "val_loss": val_loss, "best_val_loss": best_val,
                 "model_kind": args.model, "arguments": vars(args)}
        torch.save(state, output_dir / "last_model.pt")
        if improved: torch.save(state, output_dir / "best_model.pt")
        print(f"epoch={epoch:03d} train={train_loss:.4f} val={val_loss:.4f} stale={stale}/{args.patience}")
        if stale >= args.patience:
            print("Early stopping: validation loss stopped improving."); break
    best = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    plot_predictions(model, val_loader, device, output_dir / "predictions.png", args.model)
    print(f"Saved under {output_dir}")


if __name__ == "__main__":
    main()
