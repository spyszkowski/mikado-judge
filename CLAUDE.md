# Mikado Judge

## What This Project Is

A computer vision system that acts as an automated judge for the
Mikado (pick-up sticks) board game. In Mikado, players take turns
removing sticks from a tangled pile. The rule is: you may only move
the stick you are trying to pick up. If any OTHER stick moves during
your turn, that is a "fault" and your turn ends.

The system uses a camera (mobile phone) pointed at the playing field
to detect all sticks, track them across frames, and determine whether
a fault occurred — i.e., whether any non-target stick moved during a
player's turn.

## Technical Architecture

```
Camera (phone) → Capture frames → YOLO-OBB detection → Stick tracking → Fault judgment
                                       ↓
                              Per-stick: position, angle, class
                                       ↓
                         Compare before/after → Did non-target sticks move?
```

### Core Pipeline (Phase 1 — implement this first)

1. **Frame capture**: Capture a "before" frame and an "after" frame
   for each player turn (manual trigger — user taps a button).

2. **Stick detection**: Run YOLO-OBB on each frame to get oriented
   bounding boxes for every stick. Each detection includes:
   - 4 corner points of the rotated bounding box
   - Class label (stick color/type)
   - Confidence score

3. **Frame alignment**: Before comparing frames, align them using
   homography estimation (ORB features + RANSAC). This corrects for
   small camera movements between the before/after shots.

4. **Stick matching**: Match sticks between frames using a combination
   of centroid distance, angle similarity, and class label. Use the
   Hungarian algorithm (scipy.optimize.linear_sum_assignment) for
   optimal matching.

5. **Movement detection**: For each matched stick pair, compute:
   - Centroid displacement (pixels)
   - Angle change (degrees)
   - IoU between the two OBBs
   Flag a stick as "moved" if displacement > threshold OR
   angle change > threshold.

6. **Fault judgment**: If any stick OTHER than the one being removed
   was flagged as "moved", it's a fault. The removed stick is
   identified as the one with the largest displacement or the one
   that disappeared (present in "before" but not matched in "after").

### Enhanced Pipeline (Phase 2 — implement after Phase 1 works)

7. **Hand masking**: Use MediaPipe Hands to detect and mask the
   player's hand before running stick detection. This prevents
   the hand from occluding sticks or being confused with sticks.

8. **Real-time mode**: Instead of before/after snapshots, process
   video continuously. Track sticks frame-by-frame using their
   OBB centroids and angles. Detect movement in real time.

9. **Scoring**: Track which sticks have been removed and compute
   scores based on stick types (different colors = different points).

## Domain Knowledge: Mikado Sticks

A standard Mikado set contains 41 sticks:
- 1 Mikado stick (spiral/special pattern) — 20 points
- 5 Mandarin sticks (blue tips) — 10 points each
- 5 Japanese sticks (red tips) — 5 points each
- 15 Chinese sticks (yellow tips) — 2 points each
- 15 Bonzai sticks (green tips) — 3 points each

Note: Stick colors and point values vary by set. The classes should
be configurable in configs/classes.yaml. The user's set may use
different colors — what matters is that sticks are distinguishable
by the color of their tips.

The sticks are thin wooden dowels, roughly 17cm long and 3mm diameter.
When piled up, there is heavy overlap and occlusion.

## Key Technical Challenges

### Challenge 1: Thin overlapping objects
Sticks are very thin (just a few pixels wide at typical camera
distances) and heavily overlap. The OBB detector must learn to
distinguish individual sticks even when they cross over each other.
- Use high resolution input (imgsz=640 minimum, 1024 if GPU allows)
- Annotate occluded sticks at their full estimated extent
- Use mosaic augmentation to increase density variation

### Challenge 2: Precise movement detection
Small movements (1-2mm) can constitute a fault. The system needs
sub-pixel level precision for movement detection.
- Frame alignment via homography is critical
- Compare stick angles (even 0.5 degree change matters)
- Use centroid + angle + IoU as a combined movement metric
- Movement threshold should be configurable and tunable

### Challenge 3: Hand occlusion
The player's hand enters the frame to pick up a stick, potentially
occluding other sticks and causing false positives.
- Phase 1: Use before/after frames (hand not in frame)
- Phase 2: MediaPipe hand segmentation to mask hand region

### Challenge 4: Camera instability
Mobile phone on a tripod still has micro-movements, and the user
may nudge the setup between shots.
- ORB + RANSAC homography alignment between frames
- Reject alignment if too few features match (camera was moved
  too much — ask user to recalibrate)

### Challenge 5: Stick identity across frames
After a stick is removed, the pile shifts slightly. Need to match
which stick in frame B corresponds to which stick in frame A.
- Hungarian algorithm on a cost matrix combining:
  - Centroid distance (weight: 0.4)
  - Angle difference (weight: 0.3)
  - Class match (weight: 0.3)
- Unmatched sticks in frame A = removed sticks
- Unmatched sticks in frame B = likely false positives or
  previously occluded sticks now visible

### Challenge 6: Limited training data and hardware
Training happens on Google Colab (free T4 GPU). Dataset may be small
initially (50-100 images).
- Always use transfer learning from pretrained OBB weights
- Start with yolo26s-obb.pt (or yolo11s-obb.pt as fallback)
- Use aggressive augmentation: degrees=180, mosaic, HSV shifts
- Semi-automated labeling: train rough model, predict on unlabeled
  frames, import into CVAT for correction

### Challenge 7: Annotation efficiency
Annotating rotated bounding boxes is extremely slow for thin objects.
We use a 2-point polyline annotation method: annotate each stick as
a line from tip to tip, then convert to OBB with configurable
thickness. This reduces annotation time by ~80%. The tradeoff is
that box width is approximate (uniform per class), but stick width
variation is minimal and does not significantly impact detection.

### Annotation Workflow
Sticks are annotated as 2-point polylines in CVAT (not rotated
bounding boxes). This is dramatically faster — two clicks per stick
instead of draw + rotate + resize. A conversion script
(`scripts/lines_to_obb.py`) generates YOLO-OBB format labels by
adding a configurable thickness around each line.

Pipeline:
```
Annotate polylines in CVAT (2 points per stick: tip to tip)
  → Export as "CVAT for images 1.1" XML
  → lines_to_obb.py converts to YOLO-OBB .txt labels
  → visualize_obb.py to verify the conversion looks correct
  → prepare_dataset.py to organize train/val splits
```

Thickness per stick class is configured in `configs/classes.yaml`
under `thickness_px`. Use `scripts/visualize_obb.py` to verify
the conversion looks correct before training.

## Tech Stack

- **Python 3.10+**
- **Ultralytics YOLO** (v26 preferred, v11 as fallback) — OBB task
- **OpenCV** — frame processing, homography, visualization
- **NumPy / SciPy** — geometry, Hungarian algorithm
- **MediaPipe** — hand segmentation (Phase 2)
- **CVAT** — annotation tool (external, not part of codebase)
- **Google Colab** — training (via notebooks/)
- **Google Drive** — dataset and model weight storage

## Project Structure

```
mikado-judge/
├── CLAUDE.md                       # This file
├── README.md                       # Project overview + quickstart
├── .gitignore                      # Exclude data, weights, venv
├── pyproject.toml                  # Dependencies
│
├── configs/
│   ├── classes.yaml                # Stick class definitions (editable)
│   ├── dataset_colab.yaml          # Dataset paths for Colab training
│   ├── dataset_local.yaml          # Dataset paths for local inference
│   └── judge.yaml                  # Judgment thresholds and parameters
│
├── src/mikado/
│   ├── __init__.py
│   ├── detect.py                   # YOLO-OBB inference wrapper
│   ├── align.py                    # Frame alignment (homography)
│   ├── track.py                    # Stick matching across frames
│   ├── judge.py                    # Fault detection logic
│   ├── hand_mask.py                # MediaPipe hand masking
│   ├── game.py                     # Game state management
│   ├── visualize.py                # Debug overlays and result display
│   └── utils.py                    # OBB geometry helpers
│
├── scripts/
│   ├── extract_frames.py           # Extract key frames from videos
│   ├── prepare_dataset.py          # Build YOLO dataset from annotations
│   ├── cvat_to_yolo_obb.py         # Convert CVAT export to YOLO-OBB txt
│   ├── split_dataset.py            # Train/val split by game session
│   ├── run_inference.py            # Run detection on image/video
│   ├── evaluate_judge.py           # Test judge accuracy on labeled turns
│   ├── semi_auto_label.py          # Generate pre-annotations for CVAT
│   └── demo.py                     # Interactive demo (webcam/video)
│
├── notebooks/
│   ├── train.ipynb                 # Colab training notebook
│   ├── evaluate.ipynb              # Model evaluation
│   └── visualize_detections.ipynb  # Debug visualization
│
├── tests/
│   ├── conftest.py                 # Shared fixtures
│   ├── test_align.py               # Frame alignment tests
│   ├── test_track.py               # Stick matching tests
│   ├── test_judge.py               # Fault detection tests
│   └── test_utils.py               # Geometry helper tests
│
└── docs/
    ├── annotation_guide.md         # CVAT annotation instructions
    └── training_guide.md           # Colab training walkthrough
```

## File Details

### configs/judge.yaml
```yaml
# Movement detection thresholds — TUNE THESE
movement:
  centroid_displacement_px: 8    # pixels — flag if stick moved more than this
  angle_change_deg: 1.5          # degrees — flag if stick rotated more than this
  iou_threshold: 0.85            # flag if IoU between before/after OBB drops below this

# Stick matching parameters
matching:
  max_centroid_distance_px: 50   # don't match sticks further apart than this
  centroid_weight: 0.4           # weight in cost matrix
  angle_weight: 0.3
  class_weight: 0.3

# Frame alignment
alignment:
  min_features: 20               # minimum ORB features for reliable alignment
  ransac_threshold: 5.0          # RANSAC reprojection threshold
  warn_if_features_below: 50    # warn user about camera instability

# Detection
detection:
  confidence_threshold: 0.3      # YOLO confidence threshold
  iou_nms_threshold: 0.5         # NMS IoU threshold (for YOLO11, not needed for YOLO26)
  input_size: 640                # inference image size
```

### configs/classes.yaml
```yaml
# Define your Mikado stick classes here.
# Adjust to match your physical set.
names:
  0: mikado
  1: blue
  2: red
  3: yellow
  4: green

# Point values per class
points:
  mikado: 20
  blue: 10
  red: 5
  yellow: 2
  green: 3

# Thickness in pixels for OBB generation from polyline annotations.
# Calibrated for 1280x720 resolution at ~50cm camera distance.
thickness_px:
  mikado: 10
  blue: 7
  red: 7
  yellow: 7
  green: 7
  default: 7
```

### src/mikado/utils.py — Key geometry functions needed:
- `obb_to_corners(cx, cy, w, h, angle)` → 4 corner points
- `obb_centroid(obb)` → (cx, cy)
- `obb_angle(obb)` → angle in degrees
- `obb_iou(obb1, obb2)` → intersection over union for rotated boxes
- `angle_diff(a1, a2)` → smallest angle between two angles (handles wrapping)
- `normalize_angle(angle)` → normalize to [0, 180) since sticks are symmetric
- `line_to_obb_corners(x1, y1, x2, y2, thickness)` → 4 OBB corners from polyline + thickness

### src/mikado/detect.py — Wraps YOLO inference:
- Load model once, run on frames
- Return list of Stick objects with: corners, centroid, angle, class_id, confidence
- Filter by confidence threshold
- Handle both YOLO11 and YOLO26 output formats

### src/mikado/align.py — Frame alignment:
- Extract ORB features from both frames
- Match features using BFMatcher
- Estimate homography with RANSAC
- Warp the "after" frame to align with "before" frame
- Return alignment quality score (number of inliers)
- Important: align BEFORE running detection, or transform detection
  coordinates after detection — choose one approach consistently

### src/mikado/track.py — Stick matching between frames:
- Build cost matrix between before-sticks and after-sticks
- Cost = weighted combination of centroid distance, angle diff, class mismatch
- Solve with Hungarian algorithm
- Return: matched pairs, unmatched_before (removed), unmatched_after (new)

### src/mikado/judge.py — The core judgment:
- Takes before_sticks and after_sticks (already matched)
- For each matched pair, compute displacement and angle change
- Identify the "target stick" (the one being removed — largest displacement
  or present in before but not after)
- If any OTHER matched stick exceeds movement thresholds → FAULT
- Return: JudgmentResult with fault (bool), moved_sticks (list),
  target_stick, details (per-stick movements)

### src/mikado/game.py — Game state:
- Track players (2+), current turn, scores
- Manage game flow: wait for turn → capture before → wait → capture after → judge
- Keep history of all turns and their results
- Handle scoring when a stick is successfully removed

### scripts/extract_frames.py
- Input: directory of video files
- For each video, extract 3 frames: first (before), middle (during), last (after)
- Also extract 1 frame per second as additional training candidates
- Skip near-duplicate frames (SSIM > 0.98)
- Save with naming convention: {video_name}_f{frame_num}.png
- CLI with argparse

### scripts/cvat_to_yolo_obb.py
- Input: CVAT export directory (XML or YOLO OBB 1.1 format)
- Convert to YOLO-OBB format if needed:
  `class_id x1 y1 x2 y2 x3 y3 x4 y4` (normalized 0-1)
- Validate: check that all coordinates are within [0,1], all class IDs valid
- Report statistics: images annotated, sticks per image, class distribution

### scripts/prepare_dataset.py
- Input: frames directory + labels directory
- Organize into YOLO dataset structure (train/val split)
- Keep frames from the same video session together (don't leak)
- Generate dataset.yaml with correct paths
- Report dataset statistics

### scripts/demo.py
- Interactive demo for testing the full pipeline
- Mode 1: Load two images (before/after), run judge, show result
- Mode 2: Live camera — press 'b' for before, 'a' for after, auto-judge
- Visualize: draw OBBs on both frames, highlight moved sticks in red,
  show displacement arrows, display FAULT/OK verdict

### notebooks/train.ipynb
- Mount Google Drive
- pip install ultralytics
- git clone the repo
- Load dataset from Drive
- Train with recommended hyperparameters:
  - model: yolo26s-obb.pt (fallback: yolo11s-obb.pt)
  - epochs: 100
  - imgsz: 640
  - batch: 16
  - patience: 20
  - degrees: 180
  - mosaic: 0.5
  - hsv_s: 0.3, hsv_v: 0.3
  - cache: "ram"
- Save weights to Drive
- Show validation metrics and sample predictions
- Include resume cell for interrupted training

## Code Style and Conventions

- Use type hints everywhere
- Dataclasses for Stick, JudgmentResult, GameState
- All thresholds and parameters come from configs/judge.yaml — no magic numbers
- Logging with Python's logging module, not print()
- All scripts use argparse with --help
- Tests use pytest with fixtures for sample OBB data

## What NOT to Do

- Don't hardcode any file paths — always use config files or CLI args
- Don't include any .pt model files or training data in the repo
- Don't build a GUI framework — keep it CLI and OpenCV windows for now
- Don't implement real-time video tracking in Phase 1 — just before/after
- Don't try to detect "which stick the player is touching" in Phase 1 —
  just detect which stick was removed (disappeared between frames)

## Definition of Done (Phase 1)

The project is "done" for Phase 1 when:
1. extract_frames.py can extract key frames from recorded videos
2. cvat_to_yolo_obb.py can convert CVAT annotations to YOLO format
3. prepare_dataset.py can build a training-ready dataset
4. train.ipynb can train a YOLO-OBB model on Colab
5. run_inference.py can run the trained model on a new image/video
6. The full pipeline works: given before + after images and a trained
   model, the system correctly identifies whether a fault occurred
7. demo.py provides a visual demonstration of the system
8. All core modules have basic unit tests
9. lines_to_obb.py correctly converts CVAT polyline annotations to YOLO-OBB labels
10. visualize_obb.py shows the generated OBBs overlaid on images for verification