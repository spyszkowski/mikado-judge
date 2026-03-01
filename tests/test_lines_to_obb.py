"""Tests for the polyline-to-OBB conversion pipeline (scripts/lines_to_obb.py)."""

from __future__ import annotations

import math
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from scripts.lines_to_obb import (
    _load_classes,
    _normalise_and_clamp,
    _parse_points,
    convert_xml,
    write_labels,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_xml(annotations: list[dict]) -> str:
    """Build a minimal CVAT XML string from a list of annotation dicts.

    Each dict should have: image_name, width, height, polylines
    where polylines is a list of {label, points} dicts.
    """
    lines = ['<?xml version="1.0" encoding="utf-8"?>', "<annotations>"]
    for ann in annotations:
        lines.append(
            f'  <image name="{ann["image_name"]}" '
            f'width="{ann["width"]}" height="{ann["height"]}">'
        )
        for pl in ann.get("polylines", []):
            lines.append(f'    <polyline label="{pl["label"]}" points="{pl["points"]}"/>')
        lines.append("  </image>")
    lines.append("</annotations>")
    return "\n".join(lines)


def _xml_to_path(xml_str: str, tmp_path: Path) -> Path:
    p = tmp_path / "annotations.xml"
    p.write_text(xml_str)
    return p


_CLASS_MAP = {"mikado": 0, "blue": 1, "red": 2, "yellow": 3, "green": 4}
_THICKNESS_CFG = {
    "mikado": 10, "blue": 7, "red": 7, "yellow": 7, "green": 7, "default": 7
}


# ---------------------------------------------------------------------------
# _parse_points
# ---------------------------------------------------------------------------

class TestParsePoints:
    def test_valid_two_point_string(self):
        pts = _parse_points("10.0,20.5;90.0,80.5")
        assert pts == [(10.0, 20.5), (90.0, 80.5)]

    def test_valid_four_point_string(self):
        pts = _parse_points("0,0;100,0;100,10;0,10")
        assert len(pts) == 4

    def test_returns_none_on_empty(self):
        assert _parse_points("") is None or _parse_points("") == [("",)]

    def test_integer_coordinates(self):
        pts = _parse_points("0,0;640,480")
        assert pts == [(0.0, 0.0), (640.0, 480.0)]


# ---------------------------------------------------------------------------
# _normalise_and_clamp
# ---------------------------------------------------------------------------

class TestNormaliseAndClamp:
    def test_midpoint_normalised(self):
        assert pytest.approx(_normalise_and_clamp(320, 640), abs=1e-6) == 0.5

    def test_zero_normalised(self):
        assert pytest.approx(_normalise_and_clamp(0, 640), abs=1e-6) == 0.0

    def test_full_size_normalised(self):
        assert pytest.approx(_normalise_and_clamp(640, 640), abs=1e-6) == 1.0

    def test_negative_clamped_to_zero(self):
        assert pytest.approx(_normalise_and_clamp(-10, 640), abs=1e-6) == 0.0

    def test_over_size_clamped_to_one(self):
        assert pytest.approx(_normalise_and_clamp(700, 640), abs=1e-6) == 1.0


# ---------------------------------------------------------------------------
# convert_xml — main conversion
# ---------------------------------------------------------------------------

class TestConvertXml:
    def test_horizontal_line_produces_label(self, tmp_path):
        xml = _make_xml([{
            "image_name": "frame_000001.png",
            "width": 640, "height": 480,
            "polylines": [{"label": "blue", "points": "100,200;500,200"}],
        }])
        label_data, n_skipped = convert_xml(
            _xml_to_path(xml, tmp_path), _CLASS_MAP, _THICKNESS_CFG, 7
        )
        assert "frame_000001" in label_data
        assert len(label_data["frame_000001"]) == 1
        assert n_skipped == 0

    def test_label_line_starts_with_class_id(self, tmp_path):
        xml = _make_xml([{
            "image_name": "img.png", "width": 640, "height": 480,
            "polylines": [{"label": "red", "points": "10,10;200,10"}],
        }])
        label_data, _ = convert_xml(
            _xml_to_path(xml, tmp_path), _CLASS_MAP, _THICKNESS_CFG, 7
        )
        line = label_data["img"][0]
        assert line.startswith("2 ")  # red == class_id 2

    def test_output_has_nine_values_per_line(self, tmp_path):
        xml = _make_xml([{
            "image_name": "img.png", "width": 640, "height": 480,
            "polylines": [{"label": "green", "points": "50,100;550,100"}],
        }])
        label_data, _ = convert_xml(
            _xml_to_path(xml, tmp_path), _CLASS_MAP, _THICKNESS_CFG, 7
        )
        parts = label_data["img"][0].split()
        assert len(parts) == 9  # class_id + 8 coordinates

    def test_coordinates_normalised_to_0_1(self, tmp_path):
        xml = _make_xml([{
            "image_name": "img.png", "width": 640, "height": 480,
            "polylines": [{"label": "blue", "points": "0,0;640,480"}],
        }])
        label_data, _ = convert_xml(
            _xml_to_path(xml, tmp_path), _CLASS_MAP, _THICKNESS_CFG, 7
        )
        parts = label_data["img"][0].split()
        coords = [float(v) for v in parts[1:]]
        assert all(0.0 <= c <= 1.0 for c in coords), f"Out-of-range coord: {coords}"

    def test_coordinates_clamped_when_out_of_bounds(self, tmp_path):
        """A polyline that slightly exceeds image bounds should clamp to [0, 1]."""
        xml = _make_xml([{
            "image_name": "img.png", "width": 100, "height": 100,
            # Endpoints near edges — with thickness the perpendicular corners
            # may exceed the image boundary
            "polylines": [{"label": "blue", "points": "2,50;98,50"}],
        }])
        label_data, _ = convert_xml(
            _xml_to_path(xml, tmp_path), _CLASS_MAP, _THICKNESS_CFG, 7
        )
        parts = label_data["img"][0].split()
        coords = [float(v) for v in parts[1:]]
        assert all(0.0 <= c <= 1.0 for c in coords)

    def test_wrong_point_count_skipped(self, tmp_path):
        """Polylines with != 2 points should be skipped with a warning."""
        xml = _make_xml([{
            "image_name": "img.png", "width": 640, "height": 480,
            "polylines": [
                {"label": "blue", "points": "10,10;200,10;300,50"},  # 3 points — skip
                {"label": "red",  "points": "50,50;500,50"},          # 2 points — keep
            ],
        }])
        label_data, n_skipped = convert_xml(
            _xml_to_path(xml, tmp_path), _CLASS_MAP, _THICKNESS_CFG, 7
        )
        assert n_skipped == 1
        assert len(label_data["img"]) == 1  # only the red stick

    def test_unknown_label_skipped(self, tmp_path):
        xml = _make_xml([{
            "image_name": "img.png", "width": 640, "height": 480,
            "polylines": [{"label": "purple", "points": "10,10;200,10"}],
        }])
        label_data, n_skipped = convert_xml(
            _xml_to_path(xml, tmp_path), _CLASS_MAP, _THICKNESS_CFG, 7
        )
        assert n_skipped == 1
        assert "img" not in label_data

    def test_degenerate_line_skipped(self, tmp_path):
        """A polyline where both endpoints are the same should be skipped."""
        xml = _make_xml([{
            "image_name": "img.png", "width": 640, "height": 480,
            "polylines": [{"label": "blue", "points": "100,100;100,100"}],
        }])
        label_data, n_skipped = convert_xml(
            _xml_to_path(xml, tmp_path), _CLASS_MAP, _THICKNESS_CFG, 7
        )
        assert n_skipped == 1
        assert "img" not in label_data

    def test_multiple_sticks_per_image(self, tmp_path):
        xml = _make_xml([{
            "image_name": "img.png", "width": 640, "height": 480,
            "polylines": [
                {"label": "blue",   "points": "10,100;200,100"},
                {"label": "red",    "points": "50,200;400,200"},
                {"label": "yellow", "points": "300,50;300,400"},
            ],
        }])
        label_data, n_skipped = convert_xml(
            _xml_to_path(xml, tmp_path), _CLASS_MAP, _THICKNESS_CFG, 7
        )
        assert len(label_data["img"]) == 3
        assert n_skipped == 0

    def test_image_with_no_polylines_excluded(self, tmp_path):
        """Images with no valid annotations should not appear in output."""
        xml = _make_xml([{
            "image_name": "empty.png", "width": 640, "height": 480,
            "polylines": [],
        }])
        label_data, _ = convert_xml(
            _xml_to_path(xml, tmp_path), _CLASS_MAP, _THICKNESS_CFG, 7
        )
        assert "empty" not in label_data

    def test_per_class_thickness_used(self, tmp_path):
        """mikado thickness (10) should produce wider OBB than blue (7)."""
        xml = _make_xml([{
            "image_name": "img.png", "width": 1000, "height": 1000,
            "polylines": [
                {"label": "mikado", "points": "100,500;900,500"},
                {"label": "blue",   "points": "100,300;900,300"},
            ],
        }])
        label_data, _ = convert_xml(
            _xml_to_path(xml, tmp_path), _CLASS_MAP, _THICKNESS_CFG, 7
        )
        lines = label_data["img"]
        # Both are horizontal lines at different y: check y-spread proportional to thickness
        def y_spread(line: str) -> float:
            parts = line.split()
            ys = [float(parts[i]) for i in range(2, 9, 2)]  # y coords at positions 2,4,6,8
            return max(ys) - min(ys)

        mikado_line = next(l for l in lines if l.startswith("0 "))
        blue_line   = next(l for l in lines if l.startswith("1 "))
        # mikado thickness=10, blue=7 → at 1000px height:
        # mikado y-spread = 10/1000 = 0.01, blue = 7/1000 = 0.007
        assert y_spread(mikado_line) > y_spread(blue_line)

    def test_default_thickness_override(self, tmp_path):
        """--default-thickness should be used when a class has no entry."""
        thin_cfg = {"default": 4}  # no per-class entries
        xml = _make_xml([{
            "image_name": "img.png", "width": 1000, "height": 1000,
            "polylines": [{"label": "blue", "points": "100,500;900,500"}],
        }])
        label_data, _ = convert_xml(
            _xml_to_path(xml, tmp_path), _CLASS_MAP, thin_cfg, default_thickness=4
        )
        parts = label_data["img"][0].split()
        ys = [float(parts[i]) for i in range(2, 9, 2)]
        y_spread = max(ys) - min(ys)
        # 4px / 1000px height = 0.004
        assert pytest.approx(y_spread, abs=1e-4) == 0.004


# ---------------------------------------------------------------------------
# write_labels
# ---------------------------------------------------------------------------

class TestWriteLabels:
    def test_creates_txt_files(self, tmp_path):
        label_data = {
            "frame_000001": ["0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8"],
            "frame_000002": ["1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"],
        }
        write_labels(label_data, tmp_path)
        assert (tmp_path / "frame_000001.txt").exists()
        assert (tmp_path / "frame_000002.txt").exists()

    def test_file_content_matches(self, tmp_path):
        label_data = {"img": ["2 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8"]}
        write_labels(label_data, tmp_path)
        content = (tmp_path / "img.txt").read_text().strip()
        assert content == "2 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8"
