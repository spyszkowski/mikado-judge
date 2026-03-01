"""Re-split an existing YOLO dataset by game session without data leakage.

Useful when you need to redistribute an already-built dataset.

Usage:
    python scripts/split_dataset.py --dataset datasets/mikado/ --val-ratio 0.2
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def _session_key(stem: str) -> str:
    if "_f" in stem:
        return stem.rsplit("_f", 1)[0]
    return stem


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-split a YOLO dataset by game session")
    parser.add_argument("--dataset", required=True, help="Root dataset directory (with images/ and labels/)")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    dataset_dir = Path(args.dataset)

    # Collect all image/label pairs across all existing splits
    pairs: list[tuple[Path, Path]] = []
    for split in ("train", "val"):
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.rglob("*")):
            if img.suffix.lower() not in _IMAGE_EXTENSIONS:
                continue
            lbl = lbl_dir / (img.stem + ".txt")
            if lbl.exists():
                pairs.append((img, lbl))

    logger.info("Collected %d pairs from existing dataset", len(pairs))

    sessions: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for pair in pairs:
        sessions[_session_key(pair[0].stem)].append(pair)

    session_keys = sorted(sessions)
    rng = random.Random(args.seed)
    rng.shuffle(session_keys)
    n_val = max(1, round(len(session_keys) * args.val_ratio))
    val_keys = set(session_keys[:n_val])

    # Wipe and rebuild
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            d = dataset_dir / sub / split
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)

    for key in session_keys:
        split = "val" if key in val_keys else "train"
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split
        for img, lbl in sessions[key]:
            shutil.copy2(img, img_dir / img.name)
            shutil.copy2(lbl, lbl_dir / lbl.name)

    train_count = sum(len(sessions[k]) for k in session_keys if k not in val_keys)
    val_count = sum(len(sessions[k]) for k in val_keys)
    print(f"Split: {train_count} train / {val_count} val ({len(val_keys)} val sessions)")


if __name__ == "__main__":
    main()
