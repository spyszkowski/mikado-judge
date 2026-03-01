"""Extract key frames from recorded game videos for annotation.

Usage:
    python scripts/extract_frames.py --input recordings/ --output frames/
    python scripts/extract_frames.py --input recordings/ --output frames/ --fps 1 --ssim-threshold 0.98
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def _ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    """Compute a fast approximate SSIM between two grayscale images."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    mu_a, mu_b = a.mean(), b.mean()
    sigma_a = a.std()
    sigma_b = b.std()
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = 6.5025, 58.5225  # (0.01*255)^2 and (0.03*255)^2
    numerator = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a**2 + mu_b**2 + c1) * (sigma_a**2 + sigma_b**2 + c2)
    return float(numerator / denominator) if denominator > 0 else 1.0


def extract_frames(
    video_path: Path,
    output_dir: Path,
    fps_sample: float = 1.0,
    ssim_threshold: float = 0.98,
) -> list[Path]:
    """Extract key frames from a single video.

    Always extracts the first, middle, and last frame. Also extracts one
    frame per `fps_sample` seconds. Skips near-duplicate frames (SSIM above
    `ssim_threshold`).

    Args:
        video_path: Path to the video file.
        output_dir: Directory where extracted frames are saved.
        fps_sample: Sampling rate in frames per second.
        ssim_threshold: Skip frame if SSIM to previous saved frame exceeds this.

    Returns:
        List of saved frame paths.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Frames to unconditionally capture: first, middle, last
    must_capture = {0, total_frames // 2, max(0, total_frames - 1)}

    # Sampling interval in frames
    sample_every = max(1, int(video_fps / fps_sample))

    stem = video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    last_gray: np.ndarray | None = None

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        should_capture = frame_idx in must_capture or frame_idx % sample_every == 0

        if should_capture:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_small = cv2.resize(gray, (160, 120))

            if last_gray is not None and frame_idx not in must_capture:
                sim = _ssim_gray(last_gray, gray_small)
                if sim > ssim_threshold:
                    frame_idx += 1
                    continue

            out_path = output_dir / f"{stem}_f{frame_idx:06d}.png"
            cv2.imwrite(str(out_path), frame)
            saved.append(out_path)
            last_gray = gray_small
            logger.debug("Saved %s", out_path.name)

        frame_idx += 1

    cap.release()
    logger.info("Extracted %d frames from %s", len(saved), video_path.name)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract key frames from Mikado game videos")
    parser.add_argument("--input", required=True, help="Directory containing video files")
    parser.add_argument("--output", required=True, help="Directory to save extracted frames")
    parser.add_argument("--fps", type=float, default=1.0, help="Sampling rate in frames/second (default: 1)")
    parser.add_argument("--ssim-threshold", type=float, default=0.98,
                        help="Skip frame if SSIM to previous is above this (default: 0.98)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    videos = [p for p in sorted(input_dir.rglob("*")) if p.suffix.lower() in _VIDEO_EXTENSIONS]
    if not videos:
        logger.error("No video files found in %s", input_dir)
        return

    total_saved = 0
    for video_path in videos:
        saved = extract_frames(video_path, output_dir, fps_sample=args.fps, ssim_threshold=args.ssim_threshold)
        total_saved += len(saved)

    print(f"Done. Extracted {total_saved} frames from {len(videos)} video(s) → {output_dir}")


if __name__ == "__main__":
    main()
