"""Extract only CathAction phantom segmentation data from the combined archive."""
import shutil
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "segmentation_animal_phantom.zip"
OUT = ROOT / "data" / "segmentation" / "phantom"


def destination(member: str) -> Path | None:
    """Map a phantom ZIP member to the normalized local layout."""
    parts = PurePosixPath(member).parts
    if len(parts) != 4 or parts[0] != "segmentation":
        return None
    source_split, source_kind, filename = parts[1:]
    split_map = {"phantom_train": "train", "phantom_test": "test"}
    kind_map = {"images": "images", "masks": "labels"}
    if source_split not in split_map or source_kind not in kind_map:
        return None
    return OUT / split_map[source_split] / kind_map[source_kind] / filename


def main():
    if not RAW.exists():
        print(f"Missing {RAW}")
        return
    with zipfile.ZipFile(RAW) as zf:
        members = [(info, destination(info.filename)) for info in zf.infolist()]
        members = [(info, dest) for info, dest in members if dest is not None]
        if not members:
            print("No phantom_train or phantom_test files found in the archive.")
            return
        # Only this phantom subtree is replaced. Existing human data is untouched.
        if OUT.exists():
            shutil.rmtree(OUT)
        print(f"Extracting {len(members):,} phantom files to {OUT} ...")
        for index, (info, dest) in enumerate(members, start=1):
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as source, dest.open("wb") as target:
                shutil.copyfileobj(source, target)
            if index % 5000 == 0:
                print(f"  extracted {index:,}/{len(members):,}")
    for split in ("train", "test"):
        n_images = len(list((OUT / split / "images").glob("*")))
        n_labels = len(list((OUT / split / "labels").glob("*")))
        print(f"{split}: {n_images:,} images, {n_labels:,} labels")
    print("Run: python scripts/visualize_segmentation.py --dataset phantom")


if __name__ == "__main__":
    main()
