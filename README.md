# CathAction Demo (Cursor)

Visualize **segmentation**, **recognition**, and **anticipation** on a small CathAction subset.

## 1. Open this folder in Cursor

1. Open **Cursor**
2. **File → Open Folder…**
3. Choose `C:\Users\natha\Cath`

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
cd C:\Users\natha\Cath
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
