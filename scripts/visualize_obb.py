"""Visual QA tool: draw generated OBBs on images to verify polyline-to-OBB conversion.

Usage:
    # Save annotated copies (no window)
    python scripts/visualize_obb.py --images frames/ --labels data/labels --output visualized/

    # Interactive viewer (zoom/pan with keyboard)
    python scripts/visualize_obb.py --images frames/ --labels data/labels --interactive

Interactive controls:
    +  /  =       Zoom in
    -             Zoom out
    w/a/s/d       Pan (up/left/down/right)
    r             Reset view
    n / Space     Next image
    p             Previous image
    q             Quit
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

_CLASS_COLOURS = [
    (0, 215, 255),   # 0 mikado — gold
    (200, 100, 0),   # 1 blue
    (0, 0, 220),     # 2 red
    (0, 230, 230),   # 3 yellow
    (0, 180, 0),     # 4 green
]
_FALLBACK_COLOUR = (180, 180, 180)


def _load_class_names(classes_yaml: Path) -> dict[int, str]:
    with classes_yaml.open() as f:
        data = yaml.safe_load(f)
    return {int(k): str(v) for k, v in data.get("names", {}).items()}


def _colour_for(class_id: int) -> tuple[int, int, int]:
    if 0 <= class_id < len(_CLASS_COLOURS):
        return _CLASS_COLOURS[class_id]
    return _FALLBACK_COLOUR


def _parse_label_file(label_path: Path) -> list[tuple[int, np.ndarray]]:
    entries: list[tuple[int, np.ndarray]] = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 9:
            continue
        class_id = int(parts[0])
        coords = np.array(list(map(float, parts[1:])), dtype=np.float32).reshape(4, 2)
        entries.append((class_id, coords))
    return entries


def draw_labels_on_image(
    frame: np.ndarray,
    entries: list[tuple[int, np.ndarray]],
    class_names: dict[int, str],
    show_centreline: bool = True,
    thickness: int = 2,
) -> np.ndarray:
    h, w = frame.shape[:2]
    out = frame.copy()
    for class_id, corners_norm in entries:
        colour = _colour_for(class_id)
        corners_px = (corners_norm * np.array([w, h], dtype=np.float32)).astype(np.int32)
        cv2.polylines(out, [corners_px.reshape((-1, 1, 2))], isClosed=True,
                      color=colour, thickness=thickness)
        if show_centreline:
            mid_start = ((corners_px[0] + corners_px[3]) / 2).astype(np.int32)
            mid_end = ((corners_px[1] + corners_px[2]) / 2).astype(np.int32)
            cv2.line(out, tuple(mid_start), tuple(mid_end), colour, 1, cv2.LINE_AA)
        name = class_names.get(class_id, str(class_id))
        cx = int(corners_px[:, 0].mean())
        cy = int(corners_px[:, 1].mean()) - 6
        cv2.putText(out, name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, colour, 1, cv2.LINE_AA)
    return out


def _render(annotated: np.ndarray, zoom: float, ox: int, oy: int,
            win_w: int, win_h: int) -> np.ndarray:
    """Crop a region from the annotated image and scale it, preserving aspect ratio."""
    ih, iw = annotated.shape[:2]

    # How many image pixels fit in the window at this zoom level
    region_w = max(1, int(win_w / zoom))
    region_h = max(1, int(win_h / zoom))

    # Clamp so region stays inside image
    ox = max(0, min(ox, max(0, iw - region_w)))
    oy = max(0, min(oy, max(0, ih - region_h)))
    region_w = min(region_w, iw - ox)
    region_h = min(region_h, ih - oy)

    patch = annotated[oy:oy + region_h, ox:ox + region_w]

    # Scale the patch to display pixels (preserve exact aspect ratio)
    disp_w = int(region_w * zoom)
    disp_h = int(region_h * zoom)
    disp_w = max(1, min(disp_w, win_w))
    disp_h = max(1, min(disp_h, win_h))

    scaled = cv2.resize(patch, (disp_w, disp_h),
                        interpolation=cv2.INTER_AREA if zoom < 1 else cv2.INTER_LINEAR)

    # Place scaled patch on a black canvas of exactly win_w x win_h
    canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
    canvas[:disp_h, :disp_w] = scaled

    info = (f"zoom {zoom:.2f}x  pos ({ox},{oy})  "
            f"+/-=zoom  wasd=pan  r=reset  p=prev  n/spc=next  q=quit")
    cv2.putText(canvas, info, (8, 20), cv2.FONT_HERSHEY_PLAIN, 1.0,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, info, (8, 20), cv2.FONT_HERSHEY_PLAIN, 1.0,
                (255, 255, 200), 1, cv2.LINE_AA)
    return canvas


def _show_interactive(annotated: np.ndarray, title: str,
                      win_w: int, win_h: int) -> str:
    """Keyboard-only interactive viewer. Returns 'next', 'prev', or 'quit'."""
    ih, iw = annotated.shape[:2]
    zoom = win_w / iw          # fit to window width initially
    ox, oy = 0, 0
    pan_step = max(1, int(50 / zoom))  # pan 50 display-px worth of image pixels

    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)

    while True:
        frame = _render(annotated, zoom, ox, oy, win_w, win_h)
        cv2.imshow(title, frame)
        # Use full 32-bit keycode — arrow keys on Windows need the high bits
        key = cv2.waitKey(100)

        # Detect window closed via X button
        if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            return "quit"

        if key == -1:
            continue

        key_low = key & 0xFF
        key_full = key & 0xFFFF

        region_w = max(1, int(win_w / zoom))
        region_h = max(1, int(win_h / zoom))

        if key_low == ord("q"):
            cv2.destroyWindow(title)
            return "quit"
        elif key_low == ord("p"):
            cv2.destroyWindow(title)
            return "prev"
        elif key_low in (ord("n"), ord(" "), 13):
            cv2.destroyWindow(title)
            return "next"
        elif key_low in (ord("+"), ord("=")):
            zoom = min(zoom * 1.3, 20.0)
        elif key_low == ord("-"):
            zoom = max(zoom / 1.3, 0.05)
        elif key_low == ord("r"):
            zoom = win_w / iw
            ox, oy = 0, 0
        elif key_low == ord("a") or key_full == 0x25:   # left
            ox = max(0, ox - pan_step)
        elif key_low == ord("d") or key_full == 0x27:   # right
            ox = min(max(0, iw - region_w), ox + pan_step)
        elif key_low == ord("w") or key_full == 0x26:   # up
            oy = max(0, oy - pan_step)
        elif key_low == ord("s") or key_full == 0x28:   # down
            oy = min(max(0, ih - region_h), oy + pan_step)
        else:
            cv2.destroyWindow(title)
            return "next"

        pan_step = max(1, int(50 / zoom))


def process_directory(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path | None,
    class_names: dict[int, str],
    interactive: bool,
    show_centreline: bool,
    win_w: int = 1280,
    win_h: int = 800,
) -> None:
    images = sorted(p for p in images_dir.rglob("*")
                    if p.suffix.lower() in _IMAGE_EXTENSIONS)
    if not images:
        logger.error("No images found in %s", images_dir)
        return

    pairs = [(p, labels_dir / (p.stem + ".txt")) for p in images]
    pairs = [(p, lp) for p, lp in pairs if lp.exists()]
    if not pairs:
        logger.error("No matching label files found in %s", labels_dir)
        return

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    idx = 0
    while idx < len(pairs):
        img_path, label_path = pairs[idx]
        frame = cv2.imread(str(img_path))
        if frame is None:
            logger.warning("Cannot read image: %s", img_path)
            idx += 1
            continue

        entries = _parse_label_file(label_path)
        vis = draw_labels_on_image(frame, entries, class_names,
                                   show_centreline=show_centreline)

        if output_dir:
            cv2.imwrite(str(output_dir / img_path.name), vis)

        if interactive:
            title = f"[{idx + 1}/{len(pairs)}] {img_path.name} ({len(entries)} sticks)"
            action = _show_interactive(vis, title, win_w, win_h)
            if action == "quit":
                break
            elif action == "prev":
                idx = max(0, idx - 1)
            else:
                idx += 1
        else:
            idx += 1

    cv2.destroyAllWindows()
    print(f"Visualised {len(pairs)} images" + (f" → {output_dir}" if output_dir else ""))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw YOLO-OBB labels on images for visual verification"
    )
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--classes", default="configs/classes.yaml")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--no-centreline", action="store_true")
    parser.add_argument("--win-width", type=int, default=1280)
    parser.add_argument("--win-height", type=int, default=800)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not args.output and not args.interactive:
        parser.error("Specify --output, --interactive, or both")

    classes_yaml = Path(args.classes)
    class_names = _load_class_names(classes_yaml) if classes_yaml.exists() else {}

    process_directory(
        images_dir=Path(args.images),
        labels_dir=Path(args.labels),
        output_dir=Path(args.output) if args.output else None,
        class_names=class_names,
        interactive=args.interactive,
        show_centreline=not args.no_centreline,
        win_w=args.win_width,
        win_h=args.win_height,
    )


if __name__ == "__main__":
    main()
