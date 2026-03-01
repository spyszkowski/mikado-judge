"""Tests for OBB geometry helpers in mikado.utils."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mikado.utils import (
    angle_diff,
    centroid_distance,
    line_to_obb_corners,
    normalize_angle,
    obb_angle,
    obb_centroid,
    obb_iou,
    obb_to_corners,
)


class TestObbToCorners:
    def test_horizontal_box_centred_at_origin(self):
        corners = obb_to_corners(0, 0, 10, 4, 0)
        assert corners.shape == (4, 2)
        # Width along x: should span -5 to 5
        xs = corners[:, 0]
        ys = corners[:, 1]
        assert pytest.approx(xs.min(), abs=1e-5) == -5.0
        assert pytest.approx(xs.max(), abs=1e-5) == 5.0
        assert pytest.approx(ys.min(), abs=1e-5) == -2.0
        assert pytest.approx(ys.max(), abs=1e-5) == 2.0

    def test_centroid_preserved(self):
        cx, cy = 123.0, 456.0
        corners = obb_to_corners(cx, cy, 20, 6, 30)
        computed_cx, computed_cy = obb_centroid(corners)
        assert pytest.approx(computed_cx, abs=1e-4) == cx
        assert pytest.approx(computed_cy, abs=1e-4) == cy

    def test_rotated_90_degrees(self):
        corners = obb_to_corners(0, 0, 10, 4, 90)
        xs = corners[:, 0]
        ys = corners[:, 1]
        # After 90° rotation: width (10) goes along y, height (4) along x
        assert pytest.approx(abs(xs.max() - xs.min()), abs=1e-4) == 4.0
        assert pytest.approx(abs(ys.max() - ys.min()), abs=1e-4) == 10.0


class TestObbCentroid:
    def test_returns_correct_centroid(self):
        corners = np.array([[0, 0], [10, 0], [10, 4], [0, 4]], dtype=np.float32)
        cx, cy = obb_centroid(corners)
        assert pytest.approx(cx, abs=1e-5) == 5.0
        assert pytest.approx(cy, abs=1e-5) == 2.0

    def test_arbitrary_position(self):
        corners = obb_to_corners(200, 300, 60, 8, 45)
        cx, cy = obb_centroid(corners)
        assert pytest.approx(cx, abs=1e-4) == 200.0
        assert pytest.approx(cy, abs=1e-4) == 300.0


class TestNormalizeAngle:
    @pytest.mark.parametrize("angle,expected", [
        (0.0, 0.0),
        (90.0, 90.0),
        (180.0, 0.0),
        (270.0, 90.0),
        (-90.0, 90.0),
        (45.0, 45.0),
        (181.0, 1.0),
    ])
    def test_values(self, angle, expected):
        assert pytest.approx(normalize_angle(angle), abs=1e-5) == expected


class TestAngleDiff:
    def test_same_angle(self):
        assert pytest.approx(angle_diff(45.0, 45.0), abs=1e-5) == 0.0

    def test_wrap_around(self):
        # 1° and 179° differ by 2° (wrapping)
        assert pytest.approx(angle_diff(1.0, 179.0), abs=1e-4) == 2.0

    def test_orthogonal(self):
        assert pytest.approx(angle_diff(0.0, 90.0), abs=1e-5) == 90.0

    def test_symmetry(self):
        assert pytest.approx(angle_diff(10.0, 50.0), abs=1e-5) == angle_diff(50.0, 10.0)


class TestObbIou:
    def test_identical_boxes(self):
        corners = obb_to_corners(100, 100, 60, 8, 0)
        iou = obb_iou(corners, corners)
        assert pytest.approx(iou, abs=1e-4) == 1.0

    def test_non_overlapping(self):
        c1 = obb_to_corners(0, 0, 10, 4, 0)
        c2 = obb_to_corners(200, 200, 10, 4, 0)
        iou = obb_iou(c1, c2)
        assert pytest.approx(iou, abs=1e-6) == 0.0

    def test_partial_overlap(self):
        c1 = obb_to_corners(0, 0, 10, 4, 0)
        c2 = obb_to_corners(5, 0, 10, 4, 0)   # shifted right by half width
        iou = obb_iou(c1, c2)
        assert 0.0 < iou < 1.0

    def test_symmetry(self):
        c1 = obb_to_corners(100, 100, 60, 8, 15)
        c2 = obb_to_corners(110, 105, 60, 8, 20)
        assert pytest.approx(obb_iou(c1, c2), abs=1e-6) == obb_iou(c2, c1)

    def test_identical_rotated(self):
        corners = obb_to_corners(100, 100, 60, 8, 45)
        iou = obb_iou(corners, corners)
        assert pytest.approx(iou, abs=1e-4) == 1.0


class TestCentroidDistance:
    def test_zero_distance(self):
        c = obb_to_corners(100, 100, 40, 8, 0)
        assert pytest.approx(centroid_distance(c, c), abs=1e-5) == 0.0

    def test_known_distance(self):
        c1 = obb_to_corners(0, 0, 10, 4, 0)
        c2 = obb_to_corners(3, 4, 10, 4, 0)
        assert pytest.approx(centroid_distance(c1, c2), abs=1e-4) == 5.0


class TestLineToObbCorners:
    def test_returns_shape_4x2(self):
        corners = line_to_obb_corners(10, 100, 90, 100, thickness=6)
        assert corners.shape == (4, 2)

    def test_returns_float32(self):
        corners = line_to_obb_corners(0, 0, 100, 0, thickness=8)
        assert corners.dtype == np.float32

    def test_horizontal_line_y_span(self):
        """Horizontal line: corners should span y ± thickness/2."""
        corners = line_to_obb_corners(10, 100, 90, 100, thickness=6)
        ys = corners[:, 1]
        assert pytest.approx(ys.min(), abs=1e-4) == 97.0   # 100 - 3
        assert pytest.approx(ys.max(), abs=1e-4) == 103.0  # 100 + 3

    def test_horizontal_line_x_span(self):
        """Horizontal line: x range matches the line endpoints."""
        corners = line_to_obb_corners(10, 100, 90, 100, thickness=6)
        xs = corners[:, 0]
        assert pytest.approx(xs.min(), abs=1e-4) == 10.0
        assert pytest.approx(xs.max(), abs=1e-4) == 90.0

    def test_vertical_line_x_span(self):
        """Vertical line: corners should span x ± thickness/2."""
        corners = line_to_obb_corners(100, 10, 100, 90, thickness=6)
        xs = corners[:, 0]
        assert pytest.approx(xs.min(), abs=1e-4) == 97.0
        assert pytest.approx(xs.max(), abs=1e-4) == 103.0

    def test_vertical_line_y_span(self):
        """Vertical line: y range matches the line endpoints."""
        corners = line_to_obb_corners(100, 10, 100, 90, thickness=6)
        ys = corners[:, 1]
        assert pytest.approx(ys.min(), abs=1e-4) == 10.0
        assert pytest.approx(ys.max(), abs=1e-4) == 90.0

    def test_centroid_is_line_midpoint(self):
        """The centroid of the OBB should equal the midpoint of the line."""
        corners = line_to_obb_corners(20, 30, 80, 70, thickness=8)
        cx, cy = obb_centroid(corners)
        assert pytest.approx(cx, abs=1e-4) == 50.0   # (20+80)/2
        assert pytest.approx(cy, abs=1e-4) == 50.0   # (30+70)/2

    def test_diagonal_line_corner_count(self):
        """A 45° diagonal line must also produce exactly 4 corners."""
        corners = line_to_obb_corners(0, 0, 100, 100, thickness=8)
        assert corners.shape == (4, 2)

    def test_degenerate_line_raises(self):
        """Identical endpoints (zero-length line) must raise ValueError."""
        with pytest.raises(ValueError):
            line_to_obb_corners(50, 50, 50, 50, thickness=8)

    def test_near_zero_length_raises(self):
        """A line shorter than 1 pixel must raise ValueError."""
        with pytest.raises(ValueError):
            line_to_obb_corners(0, 0, 0.5, 0, thickness=8)
