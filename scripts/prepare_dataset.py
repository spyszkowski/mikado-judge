"""Build a YOLO-OBB dataset from extracted frames and converted labels.

Groups frames by their source video session (the filename prefix before '_f')
to ensure all frames from the same game session end up in the same split.
This prevents data leakage between train and val.

Usage — standard (pre-converted labels):
    python scripts/prepare_dataset.py \\
        --frames frames/ --labels labels/ --output datasets/mikado/

Usage — from CVAT polyline XML (auto-converts on the fly):
    python scripts/prepare_dataset.py \\
        --cvat-xml annotations.xml --from-polylines --output datasets/mikado/
    # --frames defaults to frames/ when omitted with --from-polylines
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def _session_key(stem: str) -> str:
    """Extract the session (video) name from a frame filename.

    Expected naming: {video_name}_f{frame_num:06d}
    Falls back to the full stem if pattern doesn't match.
    """
    if "_f" in stem:
        return stem.rsplit("_f", 1)[0]
    return stem


def collect_pairs(frames_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    """Find (image, label) pairs where both files exist."""
    pairs: list[tuple[Path, Path]] = []
    for img_path in sorted(frames_dir.rglob("*")):
        if img_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        label_path = labels_dir / (img_path.stem + ".txt")
        if label_path.exists():
            pairs.append((img_path, label_path))
        else:
            logger.debug("No label for %s — skipping", img_path.name)
    return pairs


def split_by_session(
    pairs: list[tuple[Path, Path]],
    val_ratio: float,
    seed: int,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    """Split pairs into train/val keeping sessions together."""
    sessions: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for pair in pairs:
        key = _session_key(pair[0].stem)
        sessions[key].append(pair)

    session_keys = sorted(sessions)
    rng = random.Random(seed)
    rng.shuffle(session_keys)

    n_val_sessions = max(1, round(len(session_keys) * val_ratio))
    val_keys = set(session_keys[:n_val_sessions])

    train_pairs: list[tuple[Path, Path]] = []
    val_pairs: list[tuple[Path, Path]] = []
    for key in session_keys:
        if key in val_keys:
            val_pairs.extend(sessions[key])
        else:
            train_pairs.extend(sessions[key])

    return train_pairs, val_pairs


def write_split(pairs: list[tuple[Path, Path]], images_dir: Path, labels_dir: Path) -> None:
    """Copy image/label pairs into the YOLO dataset directory structure."""
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for img_src, lbl_src in pairs:
        shutil.copy2(img_src, images_dir / img_src.name)
        shutil.copy2(lbl_src, labels_dir / lbl_src.name)


def write_dataset_yaml(output_dir: Path, classes_yaml: Path | None) -> None:
    """Write the dataset.yaml for YOLO training."""
    names = {0: "mikado", 1: "blue", 2: "red", 3: "yellow", 4: "green"}
    if classes_yaml and classes_yaml.exists():
        with classes_yaml.open() as f:
            data = yaml.safe_load(f)
        names = data.get("names", names)

    dataset_yaml = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "nc": len(names),
        "names": names,
    }

    out = output_dir / "dataset.yaml"
    with out.open("w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, sort_keys=False)
    logger.info("Wrote dataset.yaml → %s", out)


def print_stats(train: list, val: list) -> None:
    print(f"\nDataset statistics:")
    print(f"  Train: {len(train)} images")
    print(f"  Val:   {len(val)} images")
    print(f"  Total: {len(train) + len(val)} images")
    if train:
        train_sessions = {_session_key(p[0].stem) for p in train}
        val_sessions = {_session_key(p[0].stem) for p in val}
        print(f"  Train sessions: {len(train_sessions)}")
        print(f"  Val sessions:   {len(val_sessions)}")


def _run_polyline_conversion(
    cvat_xml: Path,
    classes_yaml: Path,
    labels_dir: Path,
) -> None:
    """Convert a CVAT polyline XML to YOLO-OBB labels in labels_dir.

    Delegates to the lines_to_obb conversion functions so that
    prepare_dataset.py --from-polylines works as a one-step pipeline.
    """
    # Import here to keep the default path free of this dependency
    from scripts.lines_to_obb import _load_classes, convert_xml, write_labels, print_statistics

    class_map, thickness_cfg = _load_classes(classes_yaml)
    label_data, n_skipped = convert_xml(cvat_xml, class_map, thickness_cfg, default_thickness=7)
    write_labels(label_data, labels_dir)
    print_statistics(label_data, class_map, n_skipped)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare YOLO-OBB dataset from frames and labels")

    # Standard path
    parser.add_argument("--frames", default=None,
                        help="Directory containing extracted frame images (default: frames/ with --from-polylines)")
    parser.add_argument("--labels", default=None,
                        help="Directory containing YOLO .txt label files")

    # Polyline conversion path
    parser.add_argument("--cvat-xml", default=None,
                        help="CVAT XML file with polyline annotations (use with --from-polylines)")
    parser.add_argument("--from-polylines", action="store_true",
                        help="Convert CVAT polyline XML to YOLO-OBB labels before building dataset")

    parser.add_argument("--output", required=True, help="Output dataset directory")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Fraction of sessions for validation (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--classes", default="configs/classes.yaml", help="Path to classes.yaml")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    output_dir = Path(args.output)
    classes_yaml = Path(args.classes)

    # Resolve labels_dir: may be a temp dir when converting from polylines
    _tmp_dir = None

    if args.from_polylines:
        if not args.cvat_xml:
            parser.error("--cvat-xml is required when using --from-polylines")

        cvat_xml = Path(args.cvat_xml)
        if not cvat_xml.exists():
            logger.error("CVAT XML not found: %s", cvat_xml)
            return

        # Use --labels as output dir for converted labels, or a temp dir
        if args.labels:
            labels_dir = Path(args.labels)
        else:
            _tmp_dir = tempfile.mkdtemp(prefix="mikado_labels_")
            labels_dir = Path(_tmp_dir)
            logger.info("Converting polylines to YOLO-OBB labels (temp dir: %s)", labels_dir)

        _run_polyline_conversion(cvat_xml, classes_yaml, labels_dir)
        frames_dir = Path(args.frames) if args.frames else Path("frames")

    else:
        # Standard path: both --frames and --labels required
        if not args.frames or not args.labels:
            parser.error("--frames and --labels are required (or use --cvat-xml --from-polylines)")
        frames_dir = Path(args.frames)
        labels_dir = Path(args.labels)

    pairs = collect_pairs(frames_dir, labels_dir)
    if not pairs:
        logger.error("No image/label pairs found. Check --frames and --labels paths.")
        if _tmp_dir:
            shutil.rmtree(_tmp_dir, ignore_errors=True)
        return

    logger.info("Found %d image/label pairs", len(pairs))
    train_pairs, val_pairs = split_by_session(pairs, args.val_ratio, args.seed)
    print_stats(train_pairs, val_pairs)

    write_split(train_pairs, output_dir / "images/train", output_dir / "labels/train")
    write_split(val_pairs, output_dir / "images/val", output_dir / "labels/val")
    write_dataset_yaml(output_dir, classes_yaml)

    if _tmp_dir:
        shutil.rmtree(_tmp_dir, ignore_errors=True)

    print(f"\nDataset ready at: {output_dir}")


if __name__ == "__main__":
    main()
