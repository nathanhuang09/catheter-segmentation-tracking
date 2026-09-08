# Catheter Segmentation and Tracking

Visualize **segmentation**, **recognition**, and **anticipation** on a small CathAction subset.

## 1. Open this folder in Cursor

1. Open **Cursor**
2. **File → Open Folder…**
3. Choose the cloned `catheter-segmentation-tracking` folder.

You should see this README and a `scripts/` folder in the left sidebar (Explorer).

## 2. How to make / edit code files in Cursor

| Goal | How |
|------|-----|
| **New file** | In the left Explorer, right-click a folder → **New File** → type e.g. `scripts/my_script.py` |
| **Edit a file** | Click the file in Explorer; it opens in the main editor |
| **Save** | `Ctrl + S` |
| **Run Python** | Open terminal (**View → Terminal** or `` Ctrl + ` ``), then `python scripts/visualize_segmentation.py` |
| **Ask AI to write code** | Open Chat (`Ctrl + L`), describe what you want, or say "create a script that …" |

You do **not** need to create files by hand for this project — the starter scripts are already here.

## 3. One-time setup (terminal in Cursor)

```powershell
cd path\to\catheter-segmentation-tracking
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install huggingface_hub
huggingface-cli login
```

## 4. Download a ~5 GB subset from Hugging Face

The dataset repo has a few large zips. For this demo:

| File | Size | Get it? |
|------|------|---------|
| `segmentation_human_train.zip` | ~143 MB | **Yes** — segmentation task |
| `video_action_understanding.zip` | ~42 GB | Download once, extract **2–3 videos only**, then **delete the zip** |

### Segmentation (small)

```powershell
huggingface-cli download airvlab/CathAction segmentation_human_train.zip --repo-type dataset --local-dir ./data/raw
```

Then run `python scripts/setup_human_segmentation.py`. It extracts the archive
into `data/segmentation/human/train/`.

### Action recognition & anticipation (subset)

The video zip is one big file — Hugging Face does not let you download part of it. Practical approach:

1. Download the full zip (uses ~42 GB **temporarily** on disk).
2. Extract only a few `video_*` folders + the CSV files.
3. Delete `video_action_understanding.zip` so you keep ~2–4 GB of frames.

```powershell
huggingface-cli download airvlab/CathAction video_action_understanding.zip --repo-type dataset --local-dir ./data/raw
```

After download, run the helper (once we know your extracted layout):

```powershell
python scripts/extract_video_subset.py
```

Or ask in Cursor chat: *"Help me extract 3 videos from video_action_understanding.zip into data/action/"*

**Skip** `segmentation_animal_phantom.zip` (~10 GB) and `collision_detection.zip` (~4 GB) unless your professor asks for them.

## 5. Run the visualizations

```powershell
.venv\Scripts\activate
python scripts/visualize_segmentation.py
python scripts/visualize_recognition.py
python scripts/visualize_anticipation.py
```

Outputs go to `outputs/` as PNG images you can put in your report.

## Segmentation datasets and U-Net baselines

The human and phantom workflows share the same binary U-Net implementation.
Dataset-specific entry points make experiments harder to mix up while retaining
the older phantom filenames for existing Colab notebooks:

| Dataset | Setup | Train | Evaluate |
|---|---|---|---|
| Human | `setup_human_segmentation.py` | `train_human_unet.py` | `evaluate_human_unet.py` |
| Phantom | `setup_phantom_segmentation.py` | `train_phantom_unet.py` | `evaluate_phantom_unet.py` |

The human archive contains 5,283 matched 512x512 image/mask pairs. It has no
official test directory, so the trainer creates train, validation, and test
filename manifests without copying images. Entire frame sequences are assigned
to one split to prevent neighboring frames from leaking between splits.

Run a cheap human pipeline check:

```powershell
python scripts/setup_human_segmentation.py
python scripts/train_human_unet.py --sanity
```

Then train a small baseline (whole-sequence grouping may make the exact counts
slightly larger than the requested targets):

```powershell
python scripts/train_human_unet.py --max-train 3500 --max-val 600 --max-test 600 --epochs 30 --batch-size 16 --image-size 256 --experiment human_unet_baseline
python scripts/evaluate_human_unet.py --checkpoint outputs/human_unet_baseline/best_model.pt
```

In Colab, save checkpoints directly to mounted Drive:

```bash
!python scripts/train_human_unet.py \
  --max-train 3500 --max-val 600 --max-test 600 \
  --epochs 30 --batch-size 16 --image-size 256 \
  --experiment human_unet_baseline \
  --output-root "/content/drive/MyDrive/CathAction/experiments"
```

### Phantom segmentation

The combined animal/phantom archive is kept in `data/raw/`, but the setup script
extracts only its official phantom train and test splits. Existing human data is
left untouched.

```powershell
python scripts/setup_phantom_segmentation.py
python scripts/visualize_segmentation.py --dataset phantom
python scripts/train_phantom_unet.py --sanity
```

Phantom `.npy` masks contain background `0` and instrument labels `1` and
sometimes `2`. The current binary pipeline intentionally maps every value above
zero to foreground. The sanity mode uses 32 training and 16 validation images
from the official training split for one epoch; it verifies the pipeline but is
not a meaningful experiment.

Subsets are stored as filename manifests rather than copied image folders. The
official phantom test split stays untouched during development. A useful first
experiment is:

```powershell
python scripts/train_phantom_unet.py --max-train 500 --max-val 100 --epochs 15 --batch-size 4 --image-size 256
```

Each experiment saves `history.csv`, `loss_curve.png`, train/validation filename
manifests, best/latest checkpoints, and `predictions.png` under `outputs/`.

To save experiment artifacts directly to a mounted location such as Google
Drive, set an output root; no symbolic link is needed:

```powershell
python scripts/train_phantom_unet.py --output-root "/content/drive/MyDrive/CathAction/experiments" --experiment phantom_unet_1000_gpu
```

Stop with `Ctrl+C`; the previous completed epoch remains safe. Resume to a total
target epoch count with:

```powershell
python scripts/train_phantom_unet.py --resume outputs/phantom_unet_subset/last_model.pt --epochs 15
```

After model and hyperparameter choices are fixed, evaluate the best checkpoint
once on the official phantom test split:

```powershell
python scripts/evaluate_phantom_unet.py --checkpoint outputs/phantom_unet_subset/best_model.pt
```

## Human model comparison

The default splitter keeps recordings intact and balances the JFQ/TQY/WJZ
prefixes approximately across splits. Preserve the resulting CSV manifests for
every comparison. First run a controlled augmentation ablation on one split:

```bash
# Split reference: no augmentation
python scripts/train_human_unet.py \
  --epochs 100 --batch-size 16 --image-size 256 \
  --experiment human_unet_stratified_noaug_v4 \
  --output-root "/content/drive/MyDrive/CathAction/experiments"

# Same filenames and settings, moderate training-only augmentation
python scripts/train_human_unet.py \
  --manifests-from "/content/drive/MyDrive/CathAction/experiments/human_unet_stratified_noaug_v4" \
  --augment --epochs 100 --batch-size 16 --image-size 256 \
  --experiment human_unet_stratified_aug_v4 \
  --output-root "/content/drive/MyDrive/CathAction/experiments"

# Pretrained SegFormer-B0, same split and augmentation policy
python scripts/train_human_segformer.py \
  --manifests-from "/content/drive/MyDrive/CathAction/experiments/human_unet_stratified_noaug_v4" \
  --augment --epochs 100 --batch-size 8 --image-size 256 \
  --experiment human_segformer_stratified_aug_v4 \
  --output-root "/content/drive/MyDrive/CathAction/experiments"

# Causal grayscale channels: t-4, t-2, t; target mask: t
python scripts/train_human_3frame_unet.py \
  --manifests-from "/content/drive/MyDrive/CathAction/experiments/human_unet_stratified_noaug_v4" \
  --augment --epochs 150 --batch-size 16 --image-size 256 \
  --experiment human_unet_3frame_stratified_aug_v4 \
  --output-root "/content/drive/MyDrive/CathAction/experiments"

# Shared encoder per frame with learned temporal fusion at every U-Net scale
python scripts/train_human_temporal_unet.py \
  --manifests-from "/content/drive/MyDrive/CathAction/experiments/human_unet_stratified_noaug_v4" \
  --augment --epochs 150 --batch-size 8 --image-size 256 \
  --experiment human_temporal_unet_stratified_aug_v4 \
  --output-root "/content/drive/MyDrive/CathAction/experiments"
```

Both trainers use AdamW, Dice loss by default, mixed precision on CUDA, cosine
learning-rate decay, early stopping, resumable checkpoints, and Drive-safe
outputs. Use `--bce-weight 0.5` to experiment with Dice+BCE, but keep the loss
identical when making the primary architecture comparison.

```bash
python scripts/evaluate_human_unet.py \
  --checkpoint "/content/drive/MyDrive/CathAction/experiments/human_unet_stratified_aug_v4/best_model.pt" \
  --eligible-3frame-only --save-probabilities
python scripts/evaluate_human_segformer.py \
  --checkpoint "/content/drive/MyDrive/CathAction/experiments/human_segformer_stratified_aug_v4/best_model.pt" \
  --eligible-3frame-only --save-probabilities
python scripts/evaluate_human_3frame_unet.py \
  --checkpoint "/content/drive/MyDrive/CathAction/experiments/human_unet_3frame_stratified_aug_v4/best_model.pt" \
  --eligible-3frame-only --save-probabilities
python scripts/evaluate_human_temporal_unet.py \
  --checkpoint "/content/drive/MyDrive/CathAction/experiments/human_temporal_unet_stratified_aug_v4/best_model.pt" \
  --eligible-3frame-only --save-probabilities
```

The channel-stacked model mixes time in its first convolution. The shared-
encoder model instead applies identical encoder weights to each grayscale frame
and fuses ordered features plus current-versus-past feature differences at all
five resolutions. Both use the same
`t-4,t-2,t` samples, targets, loss, augmentation, and split, isolating the
effect of the temporal architecture.

`audit_human_annotations.py` performs prediction-blind annotation screening.
Flags are candidates for manual review, not automatic exclusions; retain the
untouched test score as the primary result.

```bash
python scripts/audit_human_annotations.py \
  --manifests-from "/content/drive/MyDrive/CathAction/experiments/human_unet_stratified_noaug_v4" \
  --split test
```

After reviewing and freezing the QC rules, pass its `annotation_qc.csv` to an
evaluator with `--qc-csv`. This adds a secondary unflagged sensitivity summary
while still retaining and reporting the full test set as the primary result.

The recommended physics-informed temporal experiment filters an ordered
centerline with a constant-velocity state model, maximum-speed gating, spatial
smoothness, and a slowly adapting inextensible-chain constraint. It reports
centerline and tip errors as primary tracking outcomes plus a secondary Dice
score for a mask reconstructed from that centerline. Tune only on validation,
freeze the parameters, then run test once.

```bash
python scripts/evaluate_human_unet_centerline_kalman.py \
  --checkpoint "/content/drive/MyDrive/CathAction/experiments/human_unet_stratified_aug_v4/best_model.pt" \
  --split val
python scripts/evaluate_human_unet_centerline_kalman.py \
  --checkpoint "/content/drive/MyDrive/CathAction/experiments/human_unet_stratified_aug_v4/best_model.pt" \
  --split test
```

The old per-pixel mask Kalman script remains for documenting the negative
ablation, but is no longer the recommended physics model: independent pixel
states do not represent a moving thin catheter.

After all matched evaluations finish, generate one verified comparison with
paired differences and 95% sequence-clustered bootstrap confidence intervals:

```bash
python scripts/compare_human_experiments.py \
  --unet "/content/drive/MyDrive/CathAction/experiments/human_unet_stratified_aug_v4" \
  --segformer "/content/drive/MyDrive/CathAction/experiments/human_segformer_stratified_aug_v4" \
  --three-frame "/content/drive/MyDrive/CathAction/experiments/human_unet_3frame_stratified_aug_v4" \
  --temporal-unet "/content/drive/MyDrive/CathAction/experiments/human_temporal_unet_stratified_aug_v4" \
  --kalman "/content/drive/MyDrive/CathAction/experiments/human_unet_stratified_aug_v4" \
  --output-dir "/content/drive/MyDrive/CathAction/experiments/human_model_comparison_v4"
```

The script refuses to compare architectures unless their filename sets match
exactly. Its primary architecture table contains Dice, IoU, precision, and
recall. Kalman centerline/tip errors are reported separately because those are
tracking outcomes rather than full-mask segmentation metrics.

Create a compact ZIP for analysis or sharing after the evaluations finish:

```bash
python scripts/package_human_results.py \
  --experiments-root "/content/drive/MyDrive/CathAction/experiments" \
  --output "/content/drive/MyDrive/CathAction/human_results_bundle.zip"
```

By default it includes CSV/JSON results, manifests, histories, plots, annotation
audits, comparison outputs, and relevant source code from every experiment
folder. It excludes `.pt` checkpoints and `.npz` probability maps because those
can make the archive unnecessarily large. Select folders with repeated
`--experiment NAME`, or explicitly opt into large files with
`--include-probabilities` or `--include-checkpoints`.

Create a publication figure from an actual successful held-out U-Net example in
two stages. First generate a candidate preview, then rerun with one listed
filename. The script verifies the exact matched cohort, checkpoint resolution,
threshold, target non-emptiness, and requested Dice interval.

```bash
python scripts/make_human_qualitative_figure.py \
  --checkpoint "/content/drive/MyDrive/CathAction/experiments/human_unet_stratified_aug_v4/best_model.pt"

python scripts/make_human_qualitative_figure.py \
  --checkpoint "/content/drive/MyDrive/CathAction/experiments/human_unet_stratified_aug_v4/best_model.pt" \
  --filename "PASTE_ONE_FILENAME_FROM_candidate_examples.csv" \
  --min-dice 0.70 --max-dice 0.90 --style overlay-errors
```

The default panel order is input image, U-Net prediction, and ground truth
(evaluation only). It exports a 300-DPI PNG, vector PDF, and JSON provenance
record without changing the annotation, prediction, or 0.5 threshold.
With `--style overlay-errors`, all panels retain the original fluoroscopy
background: prediction is green in the middle panel, while the final comparison
uses green for true positives, red for false positives, and blue for false
negatives. The selected figure has only panel titles; filename and Dice remain
in its JSON provenance and should be stated in the caption.

Generate slide-ready augmentation and temporal-method visuals from actual
training-manifest frames:

```bash
python scripts/make_augmentation_figure.py \
  --manifests-from "/content/drive/MyDrive/CathAction/experiments/human_unet_stratified_noaug_v4"

python scripts/make_temporal_unet_diagram.py \
  --manifests-from "/content/drive/MyDrive/CathAction/experiments/human_unet_stratified_noaug_v4" \
  --split train
```

The augmentation figure is a compact 2x2 grid and shows nearest-neighbor paired
mask rotation. Its automatic selection favors a coherent annotated device with
strong local contrast. The temporal visual contains only actual `t-4,t-2,t`
grayscale inputs with green ground-truth overlays, automatically selects bounded
visible motion, and applies one identical square trajectory crop to all frames.
Use `--filename` to select an eligible frame manually or `--full-frame` to
disable the crop.

## 6. Folder layout (target)

```
Cath/
├── data/
│   ├── segmentation/
│   │   ├── human/        # train/images, train/labels
│   │   └── phantom/      # train and official test
│   └── action/           # training.csv, validation.csv, video_frames/video_*/
├── scripts/
├── outputs/
└── requirements.txt
```

## Action class labels

| ID | Action |
|----|--------|
| 0 | advance catheter |
| 1 | retract catheter |
| 2 | advance guidewire |
| 3 | retract guidewire |
| 4 | rotate |
