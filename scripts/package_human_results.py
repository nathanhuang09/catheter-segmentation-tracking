"""Package human experiment results into a compact, shareable ZIP archive."""
import argparse
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_SUFFIXES = {".csv", ".json", ".png", ".txt", ".md"}
CODE_FILES = (
    "README.md", "requirements.txt", "config.py",
    "scripts/human_experiment_utils.py", "scripts/human_models.py",
    "scripts/train_human_advanced.py", "scripts/evaluate_human_advanced.py",
    "scripts/train_human_unet.py", "scripts/evaluate_human_unet.py",
    "scripts/train_human_segformer.py", "scripts/evaluate_human_segformer.py",
    "scripts/train_human_3frame_unet.py", "scripts/evaluate_human_3frame_unet.py",
    "scripts/train_human_temporal_unet.py", "scripts/evaluate_human_temporal_unet.py",
    "scripts/evaluate_human_unet_centerline_kalman.py",
    "scripts/audit_human_annotations.py", "scripts/compare_human_experiments.py",
)


def git_revision(project_root):
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-root", type=Path, required=True)
    parser.add_argument("--experiment", action="append",
                        help="Experiment folder name; repeat as needed. Default: every result folder")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-probabilities", action="store_true",
                        help="Include potentially large .npz probability maps")
    parser.add_argument("--include-checkpoints", action="store_true",
                        help="Include very large .pt model checkpoints")
    parser.add_argument("--no-code", action="store_true")
    args = parser.parse_args()
    root = args.experiments_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Experiments root does not exist: {root}")
    selected = ([root / name for name in args.experiment] if args.experiment else
                sorted(path for path in root.iterdir() if path.is_dir()))
    missing = [str(path) for path in selected if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Experiment directories are missing: {missing}")
    project_root = Path(__file__).resolve().parents[1]
    files = []
    for directory in selected:
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            allowed = path.suffix.lower() in DEFAULT_SUFFIXES
            allowed |= args.include_probabilities and path.suffix.lower() == ".npz"
            allowed |= args.include_checkpoints and path.suffix.lower() == ".pt"
            if allowed:
                files.append((path, Path("experiments") / directory.name / path.relative_to(directory)))
    if not files:
        raise FileNotFoundError("No result files were found in the selected experiment directories")
    if not args.no_code:
        for relative in CODE_FILES:
            path = project_root / relative
            if path.is_file():
                files.append((path, Path("project_code") / relative))
    total_bytes = sum(path.stat().st_size for path, _ in files)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(project_root),
        "experiments_root": str(root),
        "experiment_folders": [path.name for path in selected],
        "files": len(files),
        "uncompressed_bytes": total_bytes,
        "includes_probabilities": args.include_probabilities,
        "includes_checkpoints": args.include_checkpoints,
        "notes": [
            "The primary architecture comparison uses test_evaluation_3frame_matched.",
            "Kalman tracking metrics are distinct from full-mask segmentation metrics.",
            "Adjacent frames are correlated; use sequence-clustered confidence intervals.",
        ],
    }
    args.output = args.output.expanduser()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("bundle_manifest.json", json.dumps(manifest, indent=2))
        seen = set()
        for source, destination in files:
            destination_text = destination.as_posix()
            if destination_text in seen:
                continue
            archive.write(source, destination_text); seen.add(destination_text)
    print(f"Packaged {len(seen):,} files from {len(selected):,} experiment folders")
    print(f"Uncompressed selection: {total_bytes / (1024 ** 2):.1f} MB")
    print(f"ZIP size: {args.output.stat().st_size / (1024 ** 2):.1f} MB")
    print(f"Saved: {args.output}")
    if not args.include_probabilities:
        print("Excluded probability .npz files (use --include-probabilities if needed).")
    if not args.include_checkpoints:
        print("Excluded model .pt checkpoints (recommended for analysis sharing).")


if __name__ == "__main__":
    main()
