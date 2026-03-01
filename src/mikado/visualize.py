"""Debug overlays and result display for Mikado Judge."""

from __future__ import annotations

import cv2
import numpy as np

from mikado.detect import Stick
from mikado.judge import JudgmentResult, StickMovement
from mikado.utils import obb_centroid

# Colour palette (BGR)
_COLOUR_OK = (0, 200, 0)        # Green — stick did not move
_COLOUR_MOVED = (0, 0, 220)     # Red — stick moved (fault)
_COLOUR_TARGET = (0, 180, 255)  # Orange — target stick
_COLOUR_NEW = (200, 200, 0)     # Cyan — new / previously occluded stick
_COLOUR_REMOVED = (200, 0, 200) # Magenta — removed stick
_COLOUR_ARROW = (30, 30, 240)   # Dark red — displacement arrow
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_sticks(
    frame: np.ndarray,
    sticks: list[Stick],
    colour: tuple[int, int, int] = _COLOUR_OK,
    thickness: int = 2,
    show_labels: bool = True,
) -> np.ndarray:
    """Draw oriented bounding boxes for a list of sticks on the frame.

    Args:
        frame: BGR image to draw on (will be copied).
        sticks: Sticks to draw.
        colour: BGR colour for the boxes.
        thickness: Line thickness.
        show_labels: Whether to draw class name and confidence.

    Returns:
        New frame with drawings applied.
    """
    out = frame.copy()
    for stick in sticks:
        pts = stick.corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [pts], isClosed=True, color=colour, thickness=thickness)

        if show_labels:
            cx, cy = obb_centroid(stick.corners)
            label = f"{stick.class_name} {stick.confidence:.2f}"
            cv2.putText(out, label, (int(cx), int(cy) - 6), _FONT, 0.4, colour, 1, cv2.LINE_AA)

    return out


def draw_judgment(
    frame_before: np.ndarray,
    frame_after: np.ndarray,
    result: JudgmentResult,
) -> np.ndarray:
    """Create a side-by-side visualisation of a judgment result.

    - Green OBBs: sticks that did not move
    - Red OBBs: sticks that moved (fault cause)
    - Orange OBBs: the target stick
    - Arrows on before-frame: displacement vectors

    Returns:
        Combined side-by-side image (before | after).
    """
    before_vis = frame_before.copy()
    after_vis = frame_after.copy()

    # Build lookup sets
    target_before = result.target_stick
    moved_before_sticks = {mv.before for mv in result.moved_sticks}

    for mv in result.all_movements:
        is_target = mv.before is target_before
        is_moved = mv.before in moved_before_sticks

        colour = _COLOUR_TARGET if is_target else (_COLOUR_MOVED if is_moved else _COLOUR_OK)

        # Draw OBBs on before and after
        _draw_obb(before_vis, mv.before.corners, colour)
        _draw_obb(after_vis, mv.after.corners, colour)

        # Draw displacement arrow on before frame
        if is_moved or is_target:
            cx_b, cy_b = obb_centroid(mv.before.corners)
            cx_a, cy_a = obb_centroid(mv.after.corners)
            cv2.arrowedLine(
                before_vis,
                (int(cx_b), int(cy_b)),
                (int(cx_a), int(cy_a)),
                _COLOUR_ARROW,
                thickness=2,
                tipLength=0.3,
            )

    # Draw removed sticks
    for stick in result.removed_sticks:
        _draw_obb(before_vis, stick.corners, _COLOUR_REMOVED)

    # Draw new sticks
    for stick in result.new_sticks:
        _draw_obb(after_vis, stick.corners, _COLOUR_NEW)

    # Verdict overlay
    verdict_text = "FAULT" if result.fault else "OK"
    verdict_colour = _COLOUR_MOVED if result.fault else _COLOUR_OK
    for vis in (before_vis, after_vis):
        cv2.putText(vis, verdict_text, (20, 50), _FONT, 1.8, verdict_colour, 3, cv2.LINE_AA)

    # Resize both to same height, then concatenate
    h = max(before_vis.shape[0], after_vis.shape[0])
    before_vis = _pad_to_height(before_vis, h)
    after_vis = _pad_to_height(after_vis, h)

    return np.concatenate([before_vis, after_vis], axis=1)


def draw_movement_details(
    frame: np.ndarray,
    movements: list[StickMovement],
) -> np.ndarray:
    """Overlay per-stick movement metrics as text on the frame."""
    out = frame.copy()
    for i, mv in enumerate(movements):
        cx, cy = obb_centroid(mv.before.corners)
        text = f"d={mv.centroid_displacement_px:.1f}px a={mv.angle_change_deg:.1f}° iou={mv.iou:.2f}"
        y = int(cy) + 15 * (i % 3)
        cv2.putText(out, text, (int(cx) + 5, y), _FONT, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _draw_obb(frame: np.ndarray, corners: np.ndarray, colour: tuple[int, int, int]) -> None:
    pts = corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=colour, thickness=2)


def _pad_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h >= target_h:
        return img
    pad = np.zeros((target_h - h, w, 3), dtype=img.dtype)
    return np.concatenate([img, pad], axis=0)
