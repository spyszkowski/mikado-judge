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

Annotations are clipped to each tile using Sutherland-Hodgman polygon clipping.
The clipped region is then projected onto the original OBB's axes to reconstruct
an OBB that preserves the original stick width and angle. Only the length changes.

Rollback: git reset --hard v2-before-tiling
"""

from __future__ import annotations

import argparse
import logging
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


def _sutherland_hodgman(subject: np.ndarray, clip_rect: tuple[float, float, float, float]) -> np.ndarray:
    """Clip a polygon against a rectangle using Sutherland-Hodgman algorithm.

    Args:
        subject: polygon vertices (N, 2)
        clip_rect: (x_min, y_min, x_max, y_max)

    Returns:
        Clipped polygon vertices (M, 2). May be empty if fully outside.
    """
    x_min, y_min, x_max, y_max = clip_rect

    def _clip_edge(poly: list, inside, intersect):
        if not poly:
            return []
        output = []
        prev = poly[-1]
        prev_inside = inside(prev)
        for curr in poly:
            curr_inside = inside(curr)
            if curr_inside:
                if not prev_inside:
                    output.append(intersect(prev, curr))
                output.append(curr)
            elif prev_inside:
                output.append(intersect(prev, curr))
            prev = curr
            prev_inside = curr_inside
        return output

    def _lerp(a, b, t):
        return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))

    poly = [(p[0], p[1]) for p in subject]

    poly = _clip_edge(
        poly,
        lambda p: p[0] >= x_min,
        lambda a, b: _lerp(a, b, (x_min - a[0]) / (b[0] - a[0]) if b[0] != a[0] else 0.0),
    )
    poly = _clip_edge(
        poly,
        lambda p: p[0] <= x_max,
        lambda a, b: _lerp(a, b, (x_max - a[0]) / (b[0] - a[0]) if b[0] != a[0] else 0.0),
    )
    poly = _clip_edge(
        poly,
        lambda p: p[1] >= y_min,
        lambda a, b: _lerp(a, b, (y_min - a[1]) / (b[1] - a[1]) if b[1] != a[1] else 0.0),
    )
    poly = _clip_edge(
        poly,
        lambda p: p[1] <= y_max,
        lambda a, b: _lerp(a, b, (y_max - a[1]) / (b[1] - a[1]) if b[1] != a[1] else 0.0),
    )

    if not poly:
        return np.empty((0, 2), dtype=np.float64)
    return np.array(poly, dtype=np.float64)


def _obb_axes(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Extract long-axis direction, perpendicular direction, and width from OBB corners.

    Returns (long_dir, perp_dir, width) where:
    - long_dir: unit vector along the stick's length
    - perp_dir: unit vector perpendicular to the stick
    - width: the stick's width in pixels (average of the two short edges)
    """
    e01 = np.linalg.norm(corners[1] - corners[0])
    e12 = np.linalg.norm(corners[2] - corners[1])

    if e01 < e12:
        # 0-1 and 2-3 are short edges (width)
        mid1 = (corners[0] + corners[1]) / 2
        mid2 = (corners[2] + corners[3]) / 2
        width = (e01 + np.linalg.norm(corners[3] - corners[2])) / 2
    else:
        # 1-2 and 3-0 are short edges (width)
        mid1 = (corners[1] + corners[2]) / 2
        mid2 = (corners[3] + corners[0]) / 2
        width = (e12 + np.linalg.norm(corners[0] - corners[3])) / 2

    long_vec = mid2 - mid1
    length = np.linalg.norm(long_vec)
    if length < 1e-6:
        return np.array([1.0, 0.0]), np.array([0.0, 1.0]), width

    long_dir = long_vec / length
    perp_dir = np.array([-long_dir[1], long_dir[0]])
    return long_dir, perp_dir, width


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

    Uses Sutherland-Hodgman polygon clipping, then reconstructs the OBB by
    projecting the clipped region onto the original OBB's axes. This preserves
    the original stick width and angle — only the length is shortened.
    """
    # Transform to tile-local pixel coords
    local = corners_px.copy()
    local[:, 0] -= tile_x
    local[:, 1] -= tile_y

    # Quick reject: centroid far outside tile
    cx = local[:, 0].mean()
    cy = local[:, 1].mean()
    if cx < -crop_w * 0.1 or cx > crop_w * 1.1 or cy < -crop_h * 0.1 or cy > crop_h * 1.1:
        return None

    # Get original OBB axes and width before clipping
    long_dir, perp_dir, orig_width = _obb_axes(local)

    # Compute original length for min_length_frac check
    long_projs = np.array([np.dot(local[i], long_dir) for i in range(4)])
    orig_length = long_projs.max() - long_projs.min()

    # Clip the OBB polygon against the tile rectangle
    clipped = _sutherland_hodgman(local, (0.0, 0.0, float(crop_w), float(crop_h)))
    if len(clipped) < 3:
        return None

    # Project clipped polygon onto the original OBB's long axis
    clip_long_projs = np.array([np.dot(p, long_dir) for p in clipped])
    clip_perp_projs = np.array([np.dot(p, perp_dir) for p in clipped])

    t_min = clip_long_projs.min()
    t_max = clip_long_projs.max()
    clipped_length = t_max - t_min

    # Check minimum length fraction
    if orig_length > 0 and (clipped_length / orig_length) < min_length_frac:
        return None

    # Reject degenerate (very short after clipping)
    if clipped_length < 2.0:
        return None

    # Reconstruct OBB: use original width, clipped length, original orientation
    center_t = (t_min + t_max) / 2
    center_p = (clip_perp_projs.min() + clip_perp_projs.max()) / 2

    center = center_t * long_dir + center_p * perp_dir
    half_len = clipped_length / 2
    half_wid = orig_width / 2

    # Build the 4 corners, then nudge the center inward so all corners
    # stay within [0, crop_w] x [0, crop_h].  This avoids both:
    #  - clamping (which collapses width)
    #  - coords outside [0,1] (which some YOLO loaders reject)
    # The positional shift is at most half_wid (~14px), negligible for training.
    def _build_box(c):
        return np.array([
            c - half_len * long_dir - half_wid * perp_dir,
            c - half_len * long_dir + half_wid * perp_dir,
            c + half_len * long_dir + half_wid * perp_dir,
            c + half_len * long_dir - half_wid * perp_dir,
        ])

    box = _build_box(center)

    # Compute how far each edge overflows and shift center to fix
    x_min_overflow = -box[:, 0].min()          # positive if box goes left of 0
    x_max_overflow = box[:, 0].max() - crop_w  # positive if box goes right of crop_w
    y_min_overflow = -box[:, 1].min()
    y_max_overflow = box[:, 1].max() - crop_h

    shift = np.zeros(2)
    if x_min_overflow > 0:
        shift[0] += x_min_overflow
    if x_max_overflow > 0:
        shift[0] -= x_max_overflow
    if y_min_overflow > 0:
        shift[1] += y_min_overflow
    if y_max_overflow > 0:
        shift[1] -= y_max_overflow

    if shift[0] != 0 or shift[1] != 0:
        box = _build_box(center + shift)

    # Final clamp for floating-point safety (shift should have fixed it)
    box[:, 0] = np.clip(box[:, 0], 0.0, float(crop_w))
    box[:, 1] = np.clip(box[:, 1], 0.0, float(crop_h))

    # Normalize to [0,1]
    box[:, 0] /= crop_w
    box[:, 1] /= crop_h

    coords_str = " ".join(f"{box[i, 0]:.6f} {box[i, 1]:.6f}" for i in range(4))
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
