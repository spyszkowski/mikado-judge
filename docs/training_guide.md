# Training Guide — YOLO-OBB on Google Colab

## Prerequisites

- A Google account with Google Drive (15 GB free, or paid)
- A prepared dataset (see annotation and preparation workflow below)
- Access to [Google Colab](https://colab.research.google.com) (free T4 GPU)

---

## Full Workflow

```
Record videos → extract_frames.py → Annotate in CVAT → cvat_to_yolo_obb.py
    → prepare_dataset.py → Upload to Drive → train.ipynb → Download best.pt
    → run_inference.py / demo.py
```

---

## Step 1: Collect Training Data

Record short videos (30–60 seconds each) of a Mikado stick pile from above.
Vary:
- Lighting conditions
- Camera angle (slightly)
- Number of sticks on the table
- Different stick configurations

Aim for at least **50 diverse images** to start. More is always better.

## Step 2: Extract Frames

```bash
python scripts/extract_frames.py \
    --input recordings/ \
    --output frames/ \
    --fps 1 \
    --ssim-threshold 0.98
```

This extracts ~1 frame per second, skipping near-duplicates.

## Step 3: Annotate in CVAT

See [annotation_guide.md](annotation_guide.md) for detailed instructions.

Export in **YOLO OBB 1.1** format.

## Step 4: Convert Annotations

```bash
python scripts/cvat_to_yolo_obb.py \
    --input cvat_export/ \
    --output labels/ \
    --classes configs/classes.yaml
```

## Step 5: Prepare Dataset

```bash
python scripts/prepare_dataset.py \
    --frames frames/ \
    --labels labels/ \
    --output datasets/mikado/ \
    --val-ratio 0.2
```

## Step 6: Upload to Google Drive

Upload the entire `datasets/mikado/` directory to your Google Drive:

```
MyDrive/
└── mikado-judge/
    ├── datasets/
    │   └── mikado/
    │       ├── images/
    │       │   ├── train/
    │       │   └── val/
    │       └── labels/
    │           ├── train/
    │           └── val/
    └── weights/    ← trained weights saved here
```

## Step 7: Train on Colab

1. Open `notebooks/train.ipynb` in Google Colab.
2. Set **Runtime → Change runtime type → T4 GPU**.
3. Update `DRIVE_ROOT`, `DATASET_DIR`, and `REPO_URL` in the config cell.
4. Run all cells in order.

Training ~100 epochs on 50–100 images takes about 20–40 minutes on T4.

### Recommended hyperparameters (small dataset)

| Parameter  | Value  | Reason |
|------------|--------|--------|
| `epochs`   | 100    | With early stopping (patience=20), often converges earlier |
| `imgsz`    | 640    | Balance of detail vs speed; use 1024 if GPU allows |
| `batch`    | 16     | T4 handles 16 at 640px comfortably |
| `degrees`  | 180    | Sticks have no preferred orientation |
| `mosaic`   | 0.5    | Helps model see more stick configurations per image |
| `hsv_s/v`  | 0.3    | Lighting variation compensation |

## Step 8: Download Best Weights

After training, weights are saved to Drive at:
`MyDrive/mikado-judge/weights/mikado_v1_<timestamp>_best.pt`

Download them and place locally at `weights/best.pt` (path excluded from git).

## Step 9: Run Inference

```bash
python scripts/run_inference.py \
    --model weights/best.pt \
    --input frames/test/ \
    --show
```

## Step 10: Iterate

Check the validation visualisations and annotation guide to identify:
- Missed sticks (add more training examples at that camera angle)
- Wrong class predictions (check tip colour contrast, add examples)
- Poor angle predictions (check OBB tightness in annotations)

Use `semi_auto_label.py` to accelerate the next annotation round:

```bash
python scripts/semi_auto_label.py \
    --model weights/best.pt \
    --input frames/new_unlabelled/ \
    --output labels/pre_annotated/ \
    --confidence 0.25
```

---

## Troubleshooting

### "CUDA out of memory"
Reduce `--batch` to 8 or `--imgsz` to 480.

### Very low mAP (< 0.3) after 100 epochs
- Check annotation quality — are OBBs tight and correctly labelled?
- Add more diverse training data (different lighting, angles).
- Try `yolo11m-obb.pt` (medium model) instead of small.

### Alignment failures at inference
The homography requires texture in the background (table surface).
If the table is plain white/black, add a patterned cloth underneath.

### Judge gives many false faults
Increase `centroid_displacement_px` in `configs/judge.yaml`.
Run `evaluate_judge.py` on labelled before/after pairs to find
the optimal threshold for your setup.
