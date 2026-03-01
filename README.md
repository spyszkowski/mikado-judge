# Mikado Judge

Automated fault detection for the [Mikado](https://en.wikipedia.org/wiki/Mikado_(game))
(pick-up sticks) board game using computer vision.

A camera pointed at the playing field detects all sticks via YOLO-OBB, tracks them
across frames, and determines whether a **fault** occurred — i.e., whether any
non-target stick moved during a player's turn.

## Quickstart

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Run demo (image pair mode)
python scripts/demo.py \
    --model weights/best.pt \
    --before before.jpg \
    --after  after.jpg

# 3. Run demo (live camera)
python scripts/demo.py --model weights/best.pt --camera 0
```

## Pipeline

```
Camera → extract_frames.py → Annotate in CVAT → cvat_to_yolo_obb.py
       → prepare_dataset.py → train.ipynb (Colab) → run_inference.py / demo.py
```

## Project Structure

```
configs/          YAML configuration (classes, judge thresholds, dataset paths)
src/mikado/       Core library (detect, align, track, judge, game, visualize, utils)
scripts/          CLI scripts for data preparation, training, and inference
notebooks/        Google Colab training and evaluation notebooks
tests/            pytest unit tests
docs/             Annotation and training guides
```

## Key Scripts

| Script | Purpose |
|--------|---------|
| `extract_frames.py` | Extract key frames from recorded game videos |
| `cvat_to_yolo_obb.py` | Convert CVAT annotations to YOLO-OBB format |
| `prepare_dataset.py` | Build train/val dataset from frames + labels |
| `split_dataset.py` | Re-split an existing dataset by game session |
| `semi_auto_label.py` | Generate pre-annotations for CVAT using a trained model |
| `run_inference.py` | Run detection on images or video |
| `evaluate_judge.py` | Evaluate judge accuracy on labelled turn pairs |
| `demo.py` | Interactive demo (image pair or live camera) |

## Running Tests

```bash
pytest
```

## Configuration

All thresholds and parameters live in `configs/judge.yaml`. Key settings:

- `movement.centroid_displacement_px` — pixel threshold to flag a moved stick (default: 8)
- `movement.angle_change_deg` — angle threshold in degrees (default: 1.5)
- `matching.max_centroid_distance_px` — maximum distance for stick matching (default: 50)

## Documentation

- [Annotation Guide](docs/annotation_guide.md) — how to label sticks in CVAT
- [Training Guide](docs/training_guide.md) — full workflow from data collection to trained model
