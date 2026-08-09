"""Download and extract CathAction human segmentation subset (~143 MB)."""
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "segmentation_human_train.zip"
OUT = ROOT / "data" / "segmentation"


def find_train_root(extracted: Path) -> Path | None:
    """Locate train/images under whatever nested folder the zip uses."""
    for img_dir in extracted.rglob("train/images"):
        if img_dir.is_dir() and (img_dir.parent / "labels").is_dir():
            return img_dir.parent.parent
    return None


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

    train_root = find_train_root(tmp)
    if train_root is None:
        print("Could not find train/images + train/labels in zip.")
        print("Contents sample:", list(tmp.rglob("*"))[:15])
        return

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for name in ("train", "test"):
        src = train_root / name
        if src.exists():
            shutil.copytree(src, OUT / name)
            print(f"Copied {name}/ -> {OUT / name}")

    shutil.rmtree(tmp)
    n_images = len(list((OUT / "train" / "images").glob("*")))
    print(f"\nReady: {n_images} training images under {OUT}")
    print("Run:  python scripts/visualize_segmentation.py")


if __name__ == "__main__":
    main()
