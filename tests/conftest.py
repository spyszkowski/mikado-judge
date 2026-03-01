"""Shared pytest fixtures for Mikado Judge tests."""

from __future__ import annotations

import numpy as np
import pytest

from mikado.detect import Stick

from tests.helpers import make_stick


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def horizontal_stick() -> Stick:
    """A horizontal stick centred at (100, 100)."""
    return make_stick(cx=100.0, cy=100.0, w=80.0, h=6.0, angle_deg=0.0)


@pytest.fixture
def vertical_stick() -> Stick:
    """A vertical stick centred at (200, 200)."""
    return make_stick(cx=200.0, cy=200.0, w=80.0, h=6.0, angle_deg=90.0)


@pytest.fixture
def diagonal_stick() -> Stick:
    """A 45° diagonal stick centred at (150, 150)."""
    return make_stick(cx=150.0, cy=150.0, w=80.0, h=6.0, angle_deg=45.0)


@pytest.fixture
def before_sticks() -> list[Stick]:
    """A set of three sticks in the 'before' frame."""
    return [
        make_stick(cx=100.0, cy=100.0, angle_deg=0.0, class_id=0, class_name="mikado"),
        make_stick(cx=200.0, cy=150.0, angle_deg=45.0, class_id=1, class_name="blue"),
        make_stick(cx=300.0, cy=200.0, angle_deg=90.0, class_id=2, class_name="red"),
    ]


@pytest.fixture
def after_sticks_no_movement(before_sticks: list[Stick]) -> list[Stick]:
    """After sticks identical to before — no movement."""
    return [
        make_stick(cx=s.centroid[0], cy=s.centroid[1],
                   angle_deg=s.angle, class_id=s.class_id, class_name=s.class_name)
        for s in before_sticks
    ]


@pytest.fixture
def after_sticks_target_removed(before_sticks: list[Stick]) -> list[Stick]:
    """After sticks with the first stick (mikado) removed, others unchanged."""
    return [
        make_stick(cx=s.centroid[0], cy=s.centroid[1],
                   angle_deg=s.angle, class_id=s.class_id, class_name=s.class_name)
        for s in before_sticks[1:]
    ]


@pytest.fixture
def after_sticks_fault(before_sticks: list[Stick]) -> list[Stick]:
    """After sticks: target (first) removed, second stick moved significantly."""
    # Second stick (blue) moved by 20 pixels — above default threshold of 8px
    moved_cx = before_sticks[1].centroid[0] + 20.0
    moved_cy = before_sticks[1].centroid[1]
    return [
        make_stick(cx=moved_cx, cy=moved_cy,
                   angle_deg=before_sticks[1].angle, class_id=1, class_name="blue"),
        make_stick(cx=before_sticks[2].centroid[0], cy=before_sticks[2].centroid[1],
                   angle_deg=before_sticks[2].angle, class_id=2, class_name="red"),
    ]


@pytest.fixture
def blank_frame() -> np.ndarray:
    """A blank 640×480 BGR image."""
    return np.zeros((480, 640, 3), dtype=np.uint8)
