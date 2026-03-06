"""Tile high-resolution images into overlapping crops for YOLO training.

Problem: 4624x3472 images downscaled to imgsz=1280 make sticks only ~8px wide.
Solution: Tile into 1280x1280 crops at native resolution -> sticks stay ~28px wide.

Usage:
    python scripts/tile_dataset.py \
        --images data/dataset/images/train \
        --labels data/dataset/labels/train \
        --output data/dataset_tiled \
        --crop-size 1280 \
        --overlap 0.3 \
        --min-stick-length 0.3 \
        --val-images data/dataset/images/val \
        --val-labels data/dataset/labels/val

    Then point your dataset.yaml at data/dataset_tiled/ for training.

Each 4624x3472 image produces ~20 tiles (5x4 grid with 30% overlap).
68 images -> ~1360 tiles. Sticks are 3.5x wider = dramatically easier to detect.

Annotations are clipped to each tile. Sticks that are mostly outside a tile
(less than min-stick-length fraction visible) are dropped to avoid teaching
the model to detect tiny stick fragments.

Rollback: git reset --hard v2-before-tiling
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)


def _parse_obb_line(line: str) -> Optional[tuple[int, np.ndarray]]:
    """Parse a YOLO-OBB label line into (class_id, corners_norm[4,2])."""
    parts = line.strip().split()
    if len(parts) != 9:
        return None
    class_id = int(parts[0])
    coords = np.array([float(x) for x in parts[1:]], dtype=np.float64).reshape(4, 2)
    return class_id, coords


def _clip_obb_to_tile(
    class_id: int,
    corners_px: np.ndarray,
    tile_x: int,
    tile_y: int,
    crop_w: int,
    crop_h: int,
    min_length_frac: float,
) -> Optional[str]:
    """Clip an OBB annotation to a tile region.

    Transforms corners to tile-local coords, clips the stick centerline,
    rebuilds the OBB, and returns a YOLO-OBB label line or None.
    """
    local = corners_px.copy()
    local[:, 0] -= tile_x
    local[:, 1] -= tile_y

    # Identify short and long edges
    edges = []
    for i in range(4):
        j = (i + 1) % 4
        d = np.linalg.norm(local[j] - local[i])
        edges.append((d, i, j))
    edges.sort(key=lambda e: e[0])

    # Short edges midpoints = stick endpoints
    mid1 = (local[edges[0][1]] + local[edges[0][2]]) / 2
    mid2 = (local[edges[1][1]] + local[edges[1][2]]) / 2
    full_length = np.linalg.norm(mid2 - mid1)

    # Parametric line clipping: P = mid1 + t*(mid2 - mid1), t in [0,1]
    d = mid2 - mid1
    t_min, t_max = 0.0, 1.0

    for axis in (0, 1):
        lo, hi = 0.0, float(crop_w if axis == 0 else crop_h)
        if abs(d[axis]) < 1e-9:
            if mid1[axis] < lo or mid1[axis] > hi:
                return None
        else:
            t1 = (lo - mid1[axis]) / d[axis]
            t2 = (hi - mid1[axis]) / d[axis]
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)

    if t_min >= t_max:
        return None

    visible_frac = t_max - t_min
    if visible_frac < min_length_frac:
        return None

    # Clipped centerline endpoints
    p1 = mid1 + t_min * d
    p2 = mid1 + t_max * d

    # Rebuild OBB from clipped centerline + original half-thickness
    half_thick = (edges[0][0] + edges[1][0]) / 4
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1:
        return None

    px = -dy / length * half_thick
    py = dx / length * half_thick

    clipped = np.array([
        [p1[0] - px, p1[1] + py],
        [p1[0] + px, p1[1] - py],
        [p2[0] + px, p2[1] - py],
        [p2[0] - px, p2[1] + py],
    ])

    # Normalize to [0,1] and clamp
    clipped[:, 0] = np.clip(clipped[:, 0] / crop_w, 0.0, 1.0)
    clipped[:, 1] = np.clip(clipped[:, 1] / crop_h, 0.0, 1.0)

    # Reject degenerate
    ce = []
    for i in range(4):
        j = (i + 1) % 4
        ce.append(np.linalg.norm(clipped[j] - clipped[i]))
    ce.sort()
    if ce[0] < 1e-6 or ce[2] < 0.01:
        return None

    coords_str = " ".join(f"{clipped[i, 0]:.6f} {clipped[i, 1]:.6f}" for i in range(4))
    return f"{class_id} {coords_str}"


def tile_image(
    img_path: Path,
    label_path: Path,
    output_images: Path,
    output_labels: Path,
    crop_size: int,
    overlap: float,
    min_length_frac: float,
) -> tuple[int, int]:
    """Tile one image and its labels. Returns (n_tiles, n_annotations)."""
    img = cv2.imread(str(img_path))
    if img is None:
        logger.warning("Cannot read image: %s", img_path)
        return 0, 0

    h, w = img.shape[:2]

    # Parse labels
    annotations = []
    if label_path.exists():
        for line in label_path.read_text().strip().splitlines():
            parsed = _parse_obb_line(line)
            if parsed:
                class_id, corners_norm = parsed
                corners_px = corners_norm.copy()
                corners_px[:, 0] *= w
                corners_px[:, 1] *= h
                annotations.append((class_id, corners_px))

    step = int(crop_size * (1 - overlap))

    # Build tile grid covering the full image
    x_starts = list(range(0, max(1, w - crop_size + 1), step))
    if not x_starts or x_starts[-1] + crop_size < w:
        x_starts.append(max(0, w - crop_size))
    y_starts = list(range(0, max(1, h - crop_size + 1), step))
    if not y_starts or y_starts[-1] + crop_size < h:
        y_starts.append(max(0, h - crop_size))

    x_starts = sorted(set(x_starts))
    y_starts = sorted(set(y_starts))

    n_tiles = 0
    n_anns = 0

    for yi, ty in enumerate(y_starts):
        for xi, tx in enumerate(x_starts):
            crop = img[ty:ty + crop_size, tx:tx + crop_size]
            ch, cw = crop.shape[:2]

            tile_labels = []
            for class_id, corners_px in annotations:
                label_line = _clip_obb_to_tile(
                    class_id, corners_px, tx, ty, cw, ch, min_length_frac
                )
                if label_line:
                    tile_labels.append(label_line)

            tile_name = f"{img_path.stem}_t{yi}x{xi}"
            cv2.imwrite(
                str(output_images / f"{tile_name}.jpg"),
                crop,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )
            label_file = output_labels / f"{tile_name}.txt"
            label_file.write_text("\n".join(tile_labels) + "\n" if tile_labels else "")

            n_tiles += 1
            n_anns += len(tile_labels)

    return n_tiles, n_anns


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tile high-res images into overlapping crops for YOLO training"
    )
    parser.add_argument("--images", required=True, help="Input images directory")
    parser.add_argument("--labels", required=True, help="Input labels directory")
    parser.add_argument("--output", required=True, help="Output dataset directory")
    parser.add_argument("--crop-size", type=int, default=1280)
    parser.add_argument("--overlap", type=float, default=0.3)
    parser.add_argument("--min-stick-length", type=float, default=0.3,
                        help="Min visible fraction to keep annotation (default: 0.3)")
    parser.add_argument("--val-images", default=None)
    parser.add_argument("--val-labels", default=None)
    parser.add_argument("--classes", default="configs/classes.yaml")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    for split_name, img_dir, lbl_dir in [
        ("train", Path(args.images), Path(args.labels)),
        ("val",
         Path(args.val_images) if args.val_images else None,
         Path(args.val_labels) if args.val_labels else None),
    ]:
        if img_dir is None or not img_dir.exists():
            if split_name == "val" and img_dir is not None:
                logger.warning("Val directory not found: %s", img_dir)
            continue

        out_imgs = Path(args.output) / "images" / split_name
        out_lbls = Path(args.output) / "labels" / split_name
        out_imgs.mkdir(parents=True, exist_ok=True)
        out_lbls.mkdir(parents=True, exist_ok=True)

        image_files = sorted(
            p for p in img_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )

        total_tiles = 0
        total_anns = 0
        for img_path in image_files:
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            nt, na = tile_image(
                img_path, lbl_path, out_imgs, out_lbls,
                args.crop_size, args.overlap, args.min_stick_length,
            )
            total_tiles += nt
            total_anns += na

        print(f"{split_name}: {len(image_files)} images -> {total_tiles} tiles, "
              f"{total_anns} annotations")

    # Generate dataset.yaml
    classes_path = Path(args.classes)
    if classes_path.exists():
        with classes_path.open() as f:
            classes_cfg = yaml.safe_load(f)
        names = classes_cfg.get("names", {})
    else:
        names = {0: "mikado", 1: "blue", 2: "red", 3: "yellow", 4: "green"}

    dataset_yaml = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "nc": len(names),
        "names": names,
    }
    yaml_path = Path(args.output) / "dataset.yaml"
    with yaml_path.open("w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)
    print(f"\nDataset YAML: {yaml_path}")
    print(f"To train: update DATASET_DIR in train.ipynb to point to {args.output}")


if __name__ == "__main__":
    main()
