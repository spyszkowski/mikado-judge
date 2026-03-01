"""Shared test helper functions (not fixtures — importable directly)."""

from __future__ import annotations

import numpy as np

from mikado.detect import Stick
from mikado.utils import obb_to_corners


def make_stick(
    cx: float = 100.0,
    cy: float = 100.0,
    w: float = 80.0,
    h: float = 6.0,
    angle_deg: float = 0.0,
    class_id: int = 0,
    confidence: float = 0.9,
    class_name: str = "mikado",
) -> Stick:
    """Create a Stick at the given position and orientation."""
    corners = obb_to_corners(cx, cy, w, h, angle_deg)
    return Stick(corners=corners, class_id=class_id, confidence=confidence, class_name=class_name)
