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

Then unzip `data/raw/segmentation_human_train.zip` into `data/segmentation/`.

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

## Phantom segmentation and U-Net sanity check

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

## 6. Folder layout (target)

```
Cath/
├── data/
│   ├── segmentation/     # train/images, train/labels
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
