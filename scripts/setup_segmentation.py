"""Download and extract CathAction human segmentation subset (~143 MB)."""
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "segmentation_human_train.zip"
OUT = ROOT / "data" / "segmentation"


def find_split_dirs(extracted: Path) -> dict[str, tuple[Path, Path]]:
    """Locate image/mask directories and map them to normalized splits."""
    for img_dir in extracted.rglob("train/images"):
        if img_dir.is_dir() and (img_dir.parent / "labels").is_dir():
            root = img_dir.parent.parent
            splits = {}
            for split in ("train", "test"):
                images = root / split / "images"
                labels = root / split / "labels"
                if images.is_dir() and labels.is_dir():
                    splits[split] = (images, labels)
            return splits

    # The Hugging Face human-training archive currently uses
    # human_dataset_train/img and human_dataset_train/mask.
    for img_dir in extracted.rglob("img"):
        mask_dir = img_dir.parent / "mask"
        if img_dir.is_dir() and mask_dir.is_dir():
            return {"train": (img_dir, mask_dir)}

    return {}


def main():
    if not RAW.exists():
        print("Zip not found. Run this first in Cursor terminal:\n")
        print(
            "  huggingface-cli download airvlab/CathAction segmentation_human_train.zip "
            "--repo-type dataset --local-dir ./data/raw"
        )
        return

    tmp = ROOT / "data" / "_seg_extract_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    print(f"Extracting {RAW} ...")
    with zipfile.ZipFile(RAW, "r") as zf:
        zf.extractall(tmp)

    splits = find_split_dirs(tmp)
    if not splits:
        print("Could not find image and mask directories in zip.")
        print("Contents sample:", list(tmp.rglob("*"))[:15])
        return

    OUT.mkdir(parents=True)
    for split, (images, labels) in splits.items():
        split_out = OUT / split
        if split_out.exists():
            shutil.rmtree(split_out)
        shutil.copytree(images, OUT / split / "images")
        shutil.copytree(labels, OUT / split / "labels")
        print(f"Copied {split} images and labels -> {OUT / split}")

    shutil.rmtree(tmp)
    n_images = len(list((OUT / "train" / "images").glob("*")))
    print(f"\nReady: {n_images} training images under {OUT}")
    print("Run:  python scripts/visualize_segmentation.py")


if __name__ == "__main__":
    main()
