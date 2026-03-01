"""Convert CVAT polyline annotations to YOLO-OBB label files.

Sticks are annotated in CVAT as 2-point polylines (tip to tip).
This script converts them to oriented bounding boxes by adding a
configurable thickness around each line, then writes YOLO-OBB .txt files.

Input format:  CVAT for images 1.1 (XML)
Output format: YOLO-OBB  — one .txt per image, lines:
                   class_id x1 y1 x2 y2 x3 y3 x4 y4  (normalised 0-1)

Usage:
    python scripts/lines_to_obb.py \\
        --cvat-xml path/to/annotations.xml \\
        --output labels/ \\
        --classes configs/classes.yaml \\
        --default-thickness 7
"""

from __future__ import annotations

import argparse
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

from mikado.utils import line_to_obb_corners

logger = logging.getLogger(__name__)


def _load_classes(classes_yaml: Path) -> tuple[dict[str, int], dict[str, int]]:
    """Return (class_name → id, class_name → thickness_px) from classes.yaml."""
    with classes_yaml.open() as f:
        data = yaml.safe_load(f)

    names: dict[int, str] = data.get("names", {})
    class_map: dict[str, int] = {v: int(k) for k, v in names.items()}

    thickness_cfg: dict[str, int] = {
        str(k): int(v) for k, v in data.get("thickness_px", {}).items()
    }
    return class_map, thickness_cfg


def _parse_points(points_str: str) -> list[tuple[float, float]] | None:
    """Parse a CVAT points string 'x1,y1;x2,y2;...' into a list of (x, y) tuples."""
    try:
        return [tuple(map(float, p.split(","))) for p in points_str.strip().split(";")]
    except (ValueError, AttributeError):
        return None


def _normalise_and_clamp(value: float, size: float) -> float:
    """Normalise a pixel coordinate to [0, 1]."""
    return max(0.0, min(1.0, value / size))


def convert_xml(
    xml_path: Path,
    class_map: dict[str, int],
    thickness_cfg: dict[str, int],
    default_thickness: int,
) -> tuple[dict[str, list[str]], int]:
    """Parse CVAT XML and convert polyline annotations to YOLO-OBB label lines.

    Args:
        xml_path: Path to CVAT XML export file.
        class_map: Mapping from class name to class id.
        thickness_cfg: Per-class thickness overrides loaded from classes.yaml.
        default_thickness: Fallback thickness when class has no entry in thickness_cfg.

    Returns:
        Tuple of (label_data, n_skipped) where label_data maps image filename
        stem → list of YOLO-OBB label lines.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    label_data: dict[str, list[str]] = {}
    unknown_labels: set[str] = set()
    n_skipped = 0

    for image_elem in root.findall(".//image"):
        img_name = image_elem.get("name", "")
        width = float(image_elem.get("width", 1))
        height = float(image_elem.get("height", 1))
        lines: list[str] = []

        for poly in image_elem.findall("polyline"):
            label = poly.get("label", "")
            points_str = poly.get("points", "")

            if label not in class_map:
                unknown_labels.add(label)
                n_skipped += 1
                continue

            pts = _parse_points(points_str)
            if pts is None or len(pts) != 2:
                logger.warning(
                    "Skipping polyline in '%s' (label=%s): expected 2 points, got %s",
                    img_name, label, len(pts) if pts else "unparseable",
                )
                n_skipped += 1
                continue

            (x1, y1), (x2, y2) = pts
            class_id = class_map[label]
            thickness = int(thickness_cfg.get(label, thickness_cfg.get("default", default_thickness)))

            try:
                corners = line_to_obb_corners(x1, y1, x2, y2, thickness)
            except ValueError as exc:
                logger.warning("Skipping degenerate polyline in '%s' (label=%s): %s", img_name, label, exc)
                n_skipped += 1
                continue

            # Normalise and clamp each coordinate
            coords: list[str] = []
            for cx, cy in corners:
                coords.append(f"{_normalise_and_clamp(cx, width):.6f}")
                coords.append(f"{_normalise_and_clamp(cy, height):.6f}")

            lines.append(f"{class_id} {' '.join(coords)}")

        if lines:
            stem = Path(img_name).stem
            label_data[stem] = lines

    if unknown_labels:
        logger.warning("Unknown class labels (not in classes.yaml): %s", sorted(unknown_labels))

    return label_data, n_skipped


def write_labels(label_data: dict[str, list[str]], output_dir: Path) -> None:
    """Write label data to .txt files in output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for stem, lines in label_data.items():
        out_path = output_dir / f"{stem}.txt"
        out_path.write_text("\n".join(lines) + "\n")


def print_statistics(
    label_data: dict[str, list[str]],
    class_map: dict[str, int],
    n_skipped: int,
) -> None:
    id_to_name = {v: k for k, v in class_map.items()}
    class_counts: dict[int, int] = {}
    total = 0
    for lines in label_data.values():
        for line in lines:
            cid = int(line.split()[0])
            class_counts[cid] = class_counts.get(cid, 0) + 1
            total += 1

    print(f"\nConverted {len(label_data)} images, {total} sticks total")
    for cid in sorted(class_counts):
        name = id_to_name.get(cid, str(cid))
        print(f"  {name}: {class_counts[cid]}")
    if n_skipped:
        print(f"  Skipped: {n_skipped} annotations (wrong point count or unknown label)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert CVAT polyline annotations to YOLO-OBB label files"
    )
    parser.add_argument("--cvat-xml", required=True, help="Path to CVAT XML export file")
    parser.add_argument("--output", required=True, help="Output directory for YOLO .txt label files")
    parser.add_argument("--classes", default="configs/classes.yaml", help="Path to classes.yaml")
    parser.add_argument(
        "--default-thickness",
        type=int,
        default=7,
        help="Default stick thickness in pixels (overrides classes.yaml default, default: 7)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    classes_yaml = Path(args.classes)
    if not classes_yaml.exists():
        logger.error("classes.yaml not found: %s", classes_yaml)
        return

    xml_path = Path(args.cvat_xml)
    if not xml_path.exists():
        logger.error("CVAT XML not found: %s", xml_path)
        return

    class_map, thickness_cfg = _load_classes(classes_yaml)
    logger.info("Loaded %d classes, thickness config: %s", len(class_map), thickness_cfg)

    label_data, n_skipped = convert_xml(xml_path, class_map, thickness_cfg, args.default_thickness)
    write_labels(label_data, Path(args.output))
    print_statistics(label_data, class_map, n_skipped)
    print(f"\nLabels written → {args.output}")


if __name__ == "__main__":
    main()
