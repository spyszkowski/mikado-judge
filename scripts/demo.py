"""Interactive demo of the full Mikado Judge pipeline.

Mode 1 (image pair): Load before/after images, run the judge, show result.
Mode 2 (live camera): Press 'b' to capture before, 'a' for after, auto-judge.
Press 'q' to quit.

Usage:
    # Image pair mode
    python scripts/demo.py --model weights/best.pt --before before.jpg --after after.jpg

    # Live camera mode
    python scripts/demo.py --model weights/best.pt --camera 0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from mikado.align import FrameAligner
from mikado.detect import Detector
from mikado.judge import Judge
from mikado.track import StickMatcher
from mikado.visualize import draw_judgment, draw_sticks

logger = logging.getLogger(__name__)


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_class_names(classes_yaml: str) -> dict[int, str]:
    with open(classes_yaml) as f:
        data = yaml.safe_load(f)
    names: dict[int, str] = data.get("names", {})
    return {int(k): v for k, v in names.items()}


def run_pipeline(
    before_frame: np.ndarray,
    after_frame: np.ndarray,
    detector: Detector,
    aligner: FrameAligner,
    matcher: StickMatcher,
    judge: Judge,
) -> tuple[bool, np.ndarray]:
    """Run the full detection + alignment + judgment pipeline.

    Returns (fault, visualisation_image).
    """
    alignment = aligner.align(before_frame, after_frame)
    if not alignment.success:
        logger.warning("Alignment failed or unreliable (inliers=%d)", alignment.n_inliers)

    after_aligned = alignment.aligned_frame

    before_sticks = detector.detect(before_frame)
    after_sticks = detector.detect(after_aligned)
    logger.info("Detected: %d before, %d after", len(before_sticks), len(after_sticks))

    match_result = matcher.match(before_sticks, after_sticks)
    judgment = judge.judge(match_result)

    vis = draw_judgment(before_frame, after_aligned, judgment)
    return judgment.fault, vis


def demo_image_pair(args: argparse.Namespace, config: dict) -> None:
    """Run demo on a before/after image pair."""
    detector = Detector.from_config(args.model, config)
    aligner = FrameAligner.from_config(config)
    matcher = StickMatcher.from_config(config)
    judge = Judge.from_config(config)

    before_frame = cv2.imread(args.before)
    after_frame = cv2.imread(args.after)

    if before_frame is None or after_frame is None:
        logger.error("Cannot load images: %s / %s", args.before, args.after)
        sys.exit(1)

    fault, vis = run_pipeline(before_frame, after_frame, detector, aligner, matcher, judge)

    verdict = "FAULT" if fault else "OK"
    print(f"\nVerdict: {verdict}")

    if args.output:
        cv2.imwrite(args.output, vis)
        print(f"Saved visualisation → {args.output}")

    cv2.imshow("Mikado Judge — " + verdict, vis)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def demo_live_camera(args: argparse.Namespace, config: dict) -> None:
    """Run interactive demo with live camera feed."""
    detector = Detector.from_config(args.model, config)
    aligner = FrameAligner.from_config(config)
    matcher = StickMatcher.from_config(config)
    judge = Judge.from_config(config)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        logger.error("Cannot open camera %d", args.camera)
        sys.exit(1)

    before_frame: np.ndarray | None = None
    result_vis: np.ndarray | None = None

    print("\nLive camera mode:")
    print("  Press 'b' to capture BEFORE frame")
    print("  Press 'a' to capture AFTER frame and run judgment")
    print("  Press 'q' to quit\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        display = frame.copy()

        if before_frame is not None:
            cv2.putText(display, "BEFORE captured — press 'a' for AFTER", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(display, "Press 'b' to capture BEFORE frame", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

        cv2.imshow("Mikado Judge — Live", display)

        if result_vis is not None:
            cv2.imshow("Mikado Judge — Result", result_vis)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("b"):
            before_frame = frame.copy()
            result_vis = None
            print("Before frame captured.")

        elif key == ord("a"):
            if before_frame is None:
                print("Capture before frame first (press 'b')")
                continue
            after_frame = frame.copy()
            print("Running judgment...")
            fault, result_vis = run_pipeline(
                before_frame, after_frame, detector, aligner, matcher, judge
            )
            verdict = "FAULT" if fault else "OK"
            print(f"Verdict: {verdict}\n")
            before_frame = None

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive Mikado Judge demo")
    parser.add_argument("--model", required=True, help="Path to YOLO .pt weights file")
    parser.add_argument("--config", default="configs/judge.yaml")
    parser.add_argument("--classes", default="configs/classes.yaml")

    # Image pair mode
    parser.add_argument("--before", help="Before-frame image path")
    parser.add_argument("--after", help="After-frame image path")
    parser.add_argument("--output", default=None, help="Save visualisation to this path")

    # Live camera mode
    parser.add_argument("--camera", type=int, default=None, help="Camera device index")

    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    config = _load_config(args.config)

    if args.camera is not None:
        demo_live_camera(args, config)
    elif args.before and args.after:
        demo_image_pair(args, config)
    else:
        parser.error("Provide either --camera or both --before and --after")


if __name__ == "__main__":
    main()
