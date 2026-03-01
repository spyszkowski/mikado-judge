"""Generate pre-annotations for CVAT using a partially-trained model.

Runs YOLO-OBB inference on unlabelled frames and exports predictions in
YOLO-OBB txt format, ready to be imported into CVAT for human correction.

Usage:
    python scripts/semi_auto_label.py \\
        --model weights/best.pt \\
        --input frames/unlabelled/ \\
        --output labels/pre_annotated/ \\
        --confidence 0.25
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import yaml

from mikado.detect import Detector
from mikado.utils import obb_centroid

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def _corners_to_yolo_obb(corners: np.ndarray, img_w: int, img_h: int) -> str:
    """Convert (4,2) corner array to a YOLO-OBB label line (normalised)."""
    coords = []
    for x, y in corners:
        coords.append(f"{x / img_w:.6f}")
        coords.append(f"{y / img_h:.6f}")
    return " ".join(coords)


def predict_and_save(
    detector: Detector,
    image_path: Path,
    output_dir: Path,
) -> int:
    """Run inference on one image and write a pre-annotation .txt file.

    Returns the number of sticks written.
    """
    frame = cv2.imread(str(image_path))
    if frame is None:
        logger.error("Cannot read: %s", image_path)
        return 0

    h, w = frame.shape[:2]
    sticks = detector.detect(frame)

    lines: list[str] = []
    for stick in sticks:
        coords = _corners_to_yolo_obb(stick.corners, w, h)
        lines.append(f"{stick.class_id} {coords}")

    out_path = output_dir / (image_path.stem + ".txt")
    out_path.write_text("\n".join(lines) + "\n" if lines else "")
    return len(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CVAT pre-annotations with a partially-trained model")
    parser.add_argument("--model", required=True, help="Path to YOLO .pt weights file")
    parser.add_argument("--input", required=True, help="Directory of unlabelled images")
    parser.add_argument("--output", required=True, help="Output directory for pre-annotation .txt files")
    parser.add_argument("--config", default="configs/judge.yaml", help="Path to judge.yaml")
    parser.add_argument("--confidence", type=float, default=0.25,
                        help="Lower confidence threshold to include uncertain predictions (default: 0.25)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Override confidence to the user-specified value
    config.setdefault("detection", {})["confidence_threshold"] = args.confidence

    detector = Detector.from_config(args.model, config)

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = [p for p in sorted(input_dir.rglob("*")) if p.suffix.lower() in _IMAGE_EXTENSIONS]
    if not images:
        logger.error("No images found in %s", input_dir)
        return

    total_sticks = 0
    for img_path in images:
        n = predict_and_save(detector, img_path, output_dir)
        total_sticks += n
        logger.debug("%s → %d sticks", img_path.name, n)

    print(f"Pre-annotated {len(images)} images, {total_sticks} sticks total → {output_dir}")
    print("Import the .txt files into CVAT for human review and correction.")


if __name__ == "__main__":
    main()
