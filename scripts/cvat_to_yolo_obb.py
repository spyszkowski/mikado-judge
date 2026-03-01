"""Convert CVAT OBB annotations to YOLO-OBB format.

Supports CVAT XML export (task format) and CVAT YOLO OBB 1.1 format.
Outputs one .txt file per image with lines of the form:
    class_id x1 y1 x2 y2 x3 y3 x4 y4  (normalised 0-1)

Usage:
    python scripts/cvat_to_yolo_obb.py --input cvat_export/ --output labels/ --classes configs/classes.yaml
"""

from __future__ import annotations

import argparse
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def _load_class_names(classes_yaml: Path) -> dict[str, int]:
    """Load class name → id mapping from classes.yaml."""
    with classes_yaml.open() as f:
        data = yaml.safe_load(f)
    names: dict[int, str] = data.get("names", {})
    return {v: int(k) for k, v in names.items()}


def _normalise(value: float, size: float) -> float:
    return max(0.0, min(1.0, value / size))


def _parse_cvat_xml(xml_path: Path, class_map: dict[str, int]) -> dict[str, list[str]]:
    """Parse a CVAT XML export and return {image_name: [label_lines]}."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    results: dict[str, list[str]] = {}
    unknown_classes: set[str] = set()

    for image_elem in root.findall(".//image"):
        name = image_elem.get("name", "")
        width = float(image_elem.get("width", 1))
        height = float(image_elem.get("height", 1))
        lines: list[str] = []

        for poly in image_elem.findall("polyline") + image_elem.findall("polygon"):
            label = poly.get("label", "")
            if label not in class_map:
                unknown_classes.add(label)
                continue
            class_id = class_map[label]
            points_str = poly.get("points", "")
            pts = [tuple(map(float, p.split(","))) for p in points_str.split(";")]
            if len(pts) != 4:
                logger.warning("Skipping non-4-point polygon in %s (label=%s)", name, label)
                continue
            coords = []
            for x, y in pts:
                coords.append(f"{_normalise(x, width):.6f}")
                coords.append(f"{_normalise(y, height):.6f}")
            lines.append(f"{class_id} {' '.join(coords)}")

        if lines:
            results[name] = lines

    if unknown_classes:
        logger.warning("Unknown class labels (not in classes.yaml): %s", unknown_classes)

    return results


def _parse_yolo_obb_dir(labels_dir: Path, class_map: dict[str, int]) -> dict[str, list[str]]:
    """Re-validate an existing YOLO-OBB labels directory.

    Checks that all coordinates are in [0, 1] and all class IDs are known.
    Returns {stem: [valid_lines]}.
    """
    valid_ids = set(class_map.values())
    results: dict[str, list[str]] = {}
    for txt in sorted(labels_dir.glob("*.txt")):
        valid_lines: list[str] = []
        for lineno, line in enumerate(txt.read_text().splitlines(), 1):
            parts = line.strip().split()
            if len(parts) != 9:
                logger.warning("%s:%d — expected 9 values, got %d", txt.name, lineno, len(parts))
                continue
            class_id = int(parts[0])
            if class_id not in valid_ids:
                logger.warning("%s:%d — unknown class_id %d", txt.name, lineno, class_id)
                continue
            coords = [float(v) for v in parts[1:]]
            if any(v < 0.0 or v > 1.0 for v in coords):
                logger.warning("%s:%d — coordinates out of [0,1] range", txt.name, lineno)
                continue
            valid_lines.append(line.strip())
        if valid_lines:
            results[txt.stem] = valid_lines
    return results


def _write_labels(label_data: dict[str, list[str]], output_dir: Path) -> int:
    """Write label data to .txt files. Returns number of files written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for image_name, lines in label_data.items():
        stem = Path(image_name).stem
        out_path = output_dir / f"{stem}.txt"
        out_path.write_text("\n".join(lines) + "\n")
        count += 1
    return count


def _print_statistics(label_data: dict[str, list[str]], class_map: dict[str, int]) -> None:
    id_to_name = {v: k for k, v in class_map.items()}
    class_counts: dict[int, int] = {}
    total_sticks = 0
    for lines in label_data.values():
        for line in lines:
            class_id = int(line.split()[0])
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
            total_sticks += 1

    print(f"\nStatistics:")
    print(f"  Images with annotations: {len(label_data)}")
    print(f"  Total stick annotations: {total_sticks}")
    if label_data:
        avg = total_sticks / len(label_data)
        print(f"  Average sticks per image: {avg:.1f}")
    print("  Class distribution:")
    for class_id in sorted(class_counts):
        name = id_to_name.get(class_id, str(class_id))
        print(f"    {name} (id={class_id}): {class_counts[class_id]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert CVAT annotations to YOLO-OBB format")
    parser.add_argument("--input", required=True,
                        help="CVAT export directory or XML file, or existing YOLO labels dir")
    parser.add_argument("--output", required=True, help="Output directory for YOLO .txt label files")
    parser.add_argument("--classes", default="configs/classes.yaml", help="Path to classes.yaml")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate existing YOLO labels, don't convert")
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

    class_map = _load_class_names(classes_yaml)
    logger.info("Loaded %d classes: %s", len(class_map), list(class_map))

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if args.validate_only:
        label_data = _parse_yolo_obb_dir(input_path, class_map)
        print(f"Validated {len(label_data)} label files in {input_path}")
        _print_statistics(label_data, class_map)
        return

    # Auto-detect input format
    if input_path.is_file() and input_path.suffix.lower() == ".xml":
        logger.info("Parsing CVAT XML: %s", input_path)
        label_data = _parse_cvat_xml(input_path, class_map)
    elif input_path.is_dir():
        # Look for XML files inside
        xml_files = list(input_path.glob("*.xml"))
        if xml_files:
            logger.info("Parsing CVAT XML: %s", xml_files[0])
            label_data = _parse_cvat_xml(xml_files[0], class_map)
        else:
            # Assume it's an existing YOLO labels dir — validate and copy
            logger.info("Treating input as YOLO-OBB labels dir, validating...")
            label_data = _parse_yolo_obb_dir(input_path, class_map)
    else:
        logger.error("Input must be a CVAT XML file or a directory")
        return

    n = _write_labels(label_data, output_dir)
    _print_statistics(label_data, class_map)
    print(f"\nWrote {n} label files → {output_dir}")


if __name__ == "__main__":
    main()
