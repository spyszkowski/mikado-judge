# CVAT Annotation Guide for Mikado Sticks

## Overview

Sticks are annotated as **2-point polylines** in CVAT — one click at each
tip of the stick. This is much faster than drawing rotated bounding boxes
(~80% less time per stick). A conversion script generates YOLO-OBB labels
automatically by adding a configurable thickness around each line.

> **Note**: The old rotated bounding box method is deprecated. Use polylines.

---

## Setup

1. Log in to [cvat.ai](https://cvat.ai) (or your self-hosted instance).
2. Create a new project:
   - **Label format**: CVAT for images 1.1 (the default XML format — preserves polylines)
   - Add labels: `mikado`, `blue`, `red`, `yellow`, `green`
3. Create a task and upload your extracted frames (output of `extract_frames.py`).
4. Set the annotation tool to **Polyline**.

---

## Class Definitions

| Label    | Description                          | Points | Thickness |
|----------|--------------------------------------|--------|-----------|
| `mikado` | The special stick (spiral/decorated) | 20     | 10 px     |
| `blue`   | Mandarin sticks (blue tips)          | 10     | 7 px      |
| `red`    | Japanese sticks (red tips)           | 5      | 7 px      |
| `yellow` | Chinese sticks (yellow tips)         | 2      | 7 px      |
| `green`  | Bonzai sticks (green tips)           | 3      | 7 px      |

> Thickness values are calibrated for 1280×720 resolution at ~50cm camera distance.
> Adjust `thickness_px` in `configs/classes.yaml` if your setup differs.
> Adjust labels to match your physical set.

---

## Annotation Rules

### 1. Draw a polyline from tip to tip
Click once at one end of the stick, click again at the other end.
That's it — the conversion script handles the rest.

### 2. Place the point precisely at the visible tip
Even if one end is occluded, try to estimate where the tip would be
and click there. The conversion works best when the full length is annotated.

### 3. Use the correct label
Identify sticks by the **colour of their tips**.
The Mikado stick is identifiable by its decorative spiral/stripe pattern.

### 4. Annotate every visible stick
Even if a stick is mostly hidden under others, annotate as much of it as you can see.
A stick that's 60% visible is still worth annotating.

### 5. Skip truly unidentifiable sticks
If you cannot determine the class at all, skip rather than guessing.

---

## Handling Occlusion

- **Visible portion**: Annotate tip to tip including the hidden part you can infer.
  Sticks are straight — extrapolate from the visible ends.
- **Crossed sticks**: Annotate both sticks independently. Their polylines will overlap.
- **Dense pile**: Do your best. It's fine to skip the innermost sticks if truly
  impossible to separate.

---

## Annotation Workflow

1. Open the task in CVAT.
2. Select the **Polyline** tool (`P` shortcut).
3. Click the first tip of a stick.
4. Click the second tip — the polyline is complete (2 points only).
5. Select the correct class label.
6. Repeat for all sticks in the image.
7. Press **Save** when done with the image.

**Keyboard shortcuts:**
- `N` — start a new shape
- `F` / `D` — next / previous frame
- `Esc` — cancel current shape
- `Enter` — finish current shape

---

## Exporting and Converting

After annotating, export from CVAT:
**Actions → Export dataset → CVAT for images 1.1**

Then convert to YOLO-OBB format:

```bash
python scripts/lines_to_obb.py \
    --cvat-xml annotations.xml \
    --output labels/ \
    --classes configs/classes.yaml
```

Verify the conversion visually:

```bash
python scripts/visualize_obb.py \
    --images frames/ \
    --labels labels/ \
    --output visualized/ \
    --interactive
```

This opens an OpenCV window showing each image with the generated OBBs and
centre-lines drawn. Press any key to advance, `q` to quit.

---

## Building the Dataset

```bash
# Option A: standard (labels already converted)
python scripts/prepare_dataset.py \
    --frames frames/ \
    --labels labels/ \
    --output datasets/mikado/

# Option B: one-step (convert + build in one command)
python scripts/prepare_dataset.py \
    --cvat-xml annotations.xml \
    --from-polylines \
    --output datasets/mikado/
# --frames defaults to frames/ if omitted
```

---

## Quality Checks

Before training, spot-check the visualised OBBs:

- OBBs should be tight around each stick
- OBBs should be oriented along the stick axis
- Width should look right (~7–10px equivalent at your resolution)
- No OBBs in clearly empty areas of the image

If the boxes look too wide or narrow, adjust `thickness_px` in
`configs/classes.yaml` and re-run `lines_to_obb.py`.

---

## Semi-Automated Labelling (after first model)

After training a rough model, speed up subsequent annotation rounds:

```bash
python scripts/semi_auto_label.py \
    --model weights/best.pt \
    --input frames/new_unlabelled/ \
    --output labels/pre_annotated/ \
    --confidence 0.25
```

Import the pre-annotations into CVAT and correct only the errors.
