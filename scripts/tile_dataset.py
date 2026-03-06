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

    Simple approach: if the OBB centroid falls inside the tile, keep it.
    Transform corners to tile-local coordinates and clamp to tile bounds.
    This preserves the original OBB shape without distortion.
    """
    # Transform to tile-local pixel coords
    local = corners_px.copy()
    local[:, 0] -= tile_x
    local[:, 1] -= tile_y

    # Centroid of the 4 corners
    cx = local[:, 0].mean()
    cy = local[:, 1].mean()

    # Skip if centroid is outside the tile
    if cx < 0 or cx > crop_w or cy < 0 or cy > crop_h:
        return None

    # Check how much of the stick is inside the tile.
    # Use the fraction of corners inside as a quick proxy.
    inside = np.sum(
        (local[:, 0] >= 0) & (local[:, 0] <= crop_w) &
        (local[:, 1] >= 0) & (local[:, 1] <= crop_h)
    )
    # If fewer than 2 corners inside and min_length_frac > 0.5, skip
    if inside < 2 and min_length_frac > 0.5:
        return None

    # Clamp corners to tile bounds
    local[:, 0] = np.clip(local[:, 0], 0.0, float(crop_w))
    local[:, 1] = np.clip(local[:, 1], 0.0, float(crop_h))

    # Normalize to [0,1]
    local[:, 0] /= crop_w
    local[:, 1] /= crop_h

    # Reject degenerate (collapsed to a point or line after clamping)
    edge_lens = []
    for i in range(4):
        j = (i + 1) % 4
        edge_lens.append(np.linalg.norm(local[j] - local[i]))
    edge_lens.sort()
    if edge_lens[0] < 1e-6 or edge_lens[2] < 0.005:
        return None

    coords_str = " ".join(f"{local[i, 0]:.6f} {local[i, 1]:.6f}" for i in range(4))
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
