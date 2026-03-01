"""Evaluate judge accuracy on a set of labelled before/after image pairs.

The ground truth CSV has columns:
    before_image, after_image, expected_fault (0 or 1)

Usage:
    python scripts/evaluate_judge.py \\
        --model weights/best.pt \\
        --ground-truth data/ground_truth.csv \\
        --config configs/judge.yaml
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import yaml

from mikado.align import FrameAligner
from mikado.detect import Detector
from mikado.judge import Judge
from mikado.track import StickMatcher

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    before_image: str
    after_image: str
    expected_fault: bool
    predicted_fault: bool

    @property
    def correct(self) -> bool:
        return self.expected_fault == self.predicted_fault


def evaluate(
    ground_truth_csv: Path,
    detector: Detector,
    aligner: FrameAligner,
    matcher: StickMatcher,
    judge: Judge,
) -> list[EvalResult]:
    results: list[EvalResult] = []

    with ground_truth_csv.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        before_path = Path(row["before_image"])
        after_path = Path(row["after_image"])
        expected = bool(int(row["expected_fault"]))

        before_frame = cv2.imread(str(before_path))
        after_frame = cv2.imread(str(after_path))

        if before_frame is None or after_frame is None:
            logger.warning("Cannot load images: %s / %s", before_path, after_path)
            continue

        alignment = aligner.align(before_frame, after_frame)
        after_aligned = alignment.aligned_frame

        before_sticks = detector.detect(before_frame)
        after_sticks = detector.detect(after_aligned)

        match_result = matcher.match(before_sticks, after_sticks)
        judgment = judge.judge(match_result)

        result = EvalResult(
            before_image=str(before_path),
            after_image=str(after_path),
            expected_fault=expected,
            predicted_fault=judgment.fault,
        )
        results.append(result)

        status = "✓" if result.correct else "✗"
        logger.info(
            "%s %s|%s  expected=%s predicted=%s",
            status, before_path.name, after_path.name,
            expected, judgment.fault,
        )

    return results


def print_metrics(results: list[EvalResult]) -> None:
    if not results:
        print("No results to evaluate.")
        return

    total = len(results)
    correct = sum(r.correct for r in results)
    tp = sum(r.expected_fault and r.predicted_fault for r in results)
    tn = sum(not r.expected_fault and not r.predicted_fault for r in results)
    fp = sum(not r.expected_fault and r.predicted_fault for r in results)
    fn = sum(r.expected_fault and not r.predicted_fault for r in results)

    accuracy = correct / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print(f"\nEvaluation results ({total} turns):")
    print(f"  Accuracy:  {accuracy:.1%} ({correct}/{total})")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall:    {recall:.1%}")
    print(f"  F1:        {f1:.1%}")
    print(f"  TP={tp}, TN={tn}, FP={fp}, FN={fn}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate judge accuracy on labelled before/after pairs")
    parser.add_argument("--model", required=True, help="Path to YOLO .pt weights file")
    parser.add_argument("--ground-truth", required=True, help="CSV with before_image, after_image, expected_fault")
    parser.add_argument("--config", default="configs/judge.yaml", help="Path to judge.yaml")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    with open(args.config) as f:
        config = yaml.safe_load(f)

    detector = Detector.from_config(args.model, config)
    aligner = FrameAligner.from_config(config)
    matcher = StickMatcher.from_config(config)
    judge = Judge.from_config(config)

    results = evaluate(Path(args.ground_truth), detector, aligner, matcher, judge)
    print_metrics(results)


if __name__ == "__main__":
    main()
