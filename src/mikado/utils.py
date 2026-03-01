"""OBB geometry helpers for Mikado Judge."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
# An OBB is represented as an (N, 2) array of 4 corner points in pixel space.
Corners = np.ndarray  # shape (4, 2), dtype float32


def obb_to_corners(cx: float, cy: float, w: float, h: float, angle_deg: float) -> Corners:
    """Convert centre + size + angle to 4 corner points.

    Args:
        cx: Centre x in pixels.
        cy: Centre y in pixels.
        w: Width (long axis) in pixels.
        h: Height (short axis) in pixels.
        angle_deg: Rotation angle in degrees (counter-clockwise from x-axis).

    Returns:
        Array of shape (4, 2) with the four corner points ordered:
        top-left, top-right, bottom-right, bottom-left (before rotation).
    """
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    hw = w / 2.0
    hh = h / 2.0

    # Local corners (unrotated): TL, TR, BR, BL
    local = np.array([
        [-hw, -hh],
        [ hw, -hh],
        [ hw,  hh],
        [-hw,  hh],
    ], dtype=np.float32)

    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    corners = local @ rotation.T + np.array([cx, cy], dtype=np.float32)
    return corners


def obb_centroid(corners: Corners) -> tuple[float, float]:
    """Return the centroid of an OBB given its 4 corner points."""
    c = corners.mean(axis=0)
    return float(c[0]), float(c[1])


def obb_angle(corners: Corners) -> float:
    """Estimate the dominant axis angle of an OBB from its corner points.

    Uses the edge from corner 0 to corner 1 (the long axis after standard
    ordering). Returns angle in degrees in [0, 180).
    """
    edge = corners[1] - corners[0]
    angle = math.degrees(math.atan2(float(edge[1]), float(edge[0])))
    return normalize_angle(angle)


def normalize_angle(angle: float) -> float:
    """Normalise an angle to [0, 180) since sticks have 180° symmetry."""
    angle = angle % 180.0
    if angle < 0.0:
        angle += 180.0
    return angle


def angle_diff(a1: float, a2: float) -> float:
    """Smallest angular difference between two angles (handles 180° wrap).

    Both angles are expected to be in [0, 180) (normalised stick angles).
    Returns a value in [0, 90].
    """
    diff = abs(normalize_angle(a1) - normalize_angle(a2))
    return min(diff, 180.0 - diff)


def _polygon_area(poly: np.ndarray) -> float:
    """Compute the signed area of a polygon using the shoelace formula."""
    n = len(poly)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += float(poly[i, 0]) * float(poly[j, 1])
        area -= float(poly[j, 0]) * float(poly[i, 1])
    return abs(area) / 2.0


def _clip_polygon_by_halfplane(poly: list[np.ndarray], a: np.ndarray, b: np.ndarray) -> list[np.ndarray]:
    """Sutherland-Hodgman clip of polygon by the half-plane left of edge a→b."""
    if not poly:
        return poly

    def inside(p: np.ndarray) -> bool:
        return float((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) >= 0.0

    def intersection(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        r = b - a
        s = q - p
        denom = float(r[0] * s[1] - r[1] * s[0])
        if abs(denom) < 1e-10:
            return p.copy()
        t = float((p[0] - a[0]) * s[1] - (p[1] - a[1]) * s[0]) / denom
        return a + t * r

    output: list[np.ndarray] = []
    for i, current in enumerate(poly):
        previous = poly[i - 1]
        if inside(current):
            if not inside(previous):
                output.append(intersection(previous, current))
            output.append(current)
        elif inside(previous):
            output.append(intersection(previous, current))
    return output


def obb_iou(corners1: Corners, corners2: Corners) -> float:
    """Compute IoU between two oriented bounding boxes (rotated rectangles).

    Uses the Sutherland-Hodgman polygon clipping algorithm to find the
    intersection polygon, then computes areas.

    Returns:
        IoU in [0, 1].
    """
    poly1 = [corners1[i] for i in range(4)]
    poly2 = [corners2[i] for i in range(4)]

    # Clip poly1 against each edge of poly2
    clipped = poly1
    n = len(poly2)
    for i in range(n):
        clipped = _clip_polygon_by_halfplane(clipped, poly2[i], poly2[(i + 1) % n])
        if not clipped:
            return 0.0

    if len(clipped) < 3:
        return 0.0

    intersection_area = _polygon_area(np.array(clipped))
    area1 = _polygon_area(corners1)
    area2 = _polygon_area(corners2)
    union_area = area1 + area2 - intersection_area

    if union_area <= 0.0:
        return 0.0
    return float(intersection_area / union_area)


def centroid_distance(corners1: Corners, corners2: Corners) -> float:
    """Euclidean distance between the centroids of two OBBs."""
    c1 = np.array(obb_centroid(corners1))
    c2 = np.array(obb_centroid(corners2))
    return float(np.linalg.norm(c2 - c1))


def line_to_obb_corners(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    thickness: float,
) -> Corners:
    """Generate 4 OBB corners from a stick centre-line (tip-to-tip) and thickness.

    Computes a perpendicular offset of ``thickness / 2`` on each side of the line
    to form a tight oriented rectangle around it.

    Corner ordering:
        0: left of start  (x1 - dx, y1 + dy)
        1: right of start (x1 + dx, y1 - dy)
        2: right of end   (x2 + dx, y2 - dy)
        3: left of end    (x2 - dx, y2 + dy)

    where ``theta = atan2(y2-y1, x2-x1)``,
    ``dx = (thickness/2) * sin(theta)``,
    ``dy = (thickness/2) * cos(theta)``.

    Args:
        x1, y1: First endpoint in pixels.
        x2, y2: Second endpoint in pixels.
        thickness: OBB width in pixels (stick diameter).

    Returns:
        Array of shape (4, 2), dtype float32, in pixel space.

    Raises:
        ValueError: If the two endpoints are identical (degenerate line).
    """
    length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if length < 1.0:
        raise ValueError(
            f"Degenerate line: endpoints are identical or less than 1 pixel apart "
            f"({x1:.1f},{y1:.1f}) → ({x2:.1f},{y2:.1f})"
        )

    theta = math.atan2(y2 - y1, x2 - x1)
    half_t = thickness / 2.0
    dx = half_t * math.sin(theta)
    dy = half_t * math.cos(theta)

    corners = np.array([
        [x1 - dx, y1 + dy],  # 0: left of start
        [x1 + dx, y1 - dy],  # 1: right of start
        [x2 + dx, y2 - dy],  # 2: right of end
        [x2 - dx, y2 + dy],  # 3: left of end
    ], dtype=np.float32)
    return corners
