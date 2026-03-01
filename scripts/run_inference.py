"""Run YOLO-OBB inference on images or video and visualise detections.

Usage:
    python scripts/run_inference.py --model weights/best.pt --input image.jpg
    python scripts/run_inference.py --model weights/best.pt --input frames/ --show
    python scripts/run_inference.py --model weights/best.pt --input video.mp4 --output out.mp4
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import yaml

from mikado.detect import Detector
from mikado.visualize import draw_sticks

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_on_image(
    detector: Detector,
    image_path: Path,
    show: bool,
    output_dir: Path | None,
) -> None:
    frame = cv2.imread(str(image_path))
    if frame is None:
        logger.error("Cannot read image: %s", image_path)
        return

    sticks = detector.detect(frame)
    vis = draw_sticks(frame, sticks)
    logger.info("%s: detected %d sticks", image_path.name, len(sticks))

    if output_dir:
        out_path = output_dir / image_path.name
        cv2.imwrite(str(out_path), vis)

    if show:
        cv2.imshow(image_path.name, vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_on_video(
    detector: Detector,
    video_path: Path,
    show: bool,
    output_path: Path | None,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        sticks = detector.detect(frame)
        vis = draw_sticks(frame, sticks)

        if writer:
            writer.write(vis)
        if show:
            cv2.imshow("Mikado inference", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO-OBB inference on images or video")
    parser.add_argument("--model", required=True, help="Path to YOLO .pt weights file")
    parser.add_argument("--input", required=True, help="Image file, image directory, or video file")
    parser.add_argument("--output", default=None, help="Output file or directory for results")
    parser.add_argument("--config", default="configs/judge.yaml", help="Path to judge.yaml")
    parser.add_argument("--show", action="store_true", help="Display results in a window")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    config = _load_config(args.config)
    detector = Detector.from_config(args.model, config)

    input_path = Path(args.input)

    if input_path.is_dir():
        output_dir = Path(args.output) if args.output else None
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        images = [p for p in sorted(input_path.rglob("*")) if p.suffix.lower() in _IMAGE_EXTENSIONS]
        logger.info("Running inference on %d images", len(images))
        for img_path in images:
            run_on_image(detector, img_path, args.show, output_dir)

    elif input_path.suffix.lower() in _IMAGE_EXTENSIONS:
        output_dir = Path(args.output) if args.output else None
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        run_on_image(detector, input_path, args.show, output_dir)

    elif input_path.suffix.lower() in _VIDEO_EXTENSIONS:
        output_path = Path(args.output) if args.output else None
        run_on_video(detector, input_path, args.show, output_path)

    else:
        logger.error("Unsupported input type: %s", input_path)


if __name__ == "__main__":
    main()
