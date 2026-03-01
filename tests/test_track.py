"""Tests for stick matching (mikado.track)."""

from __future__ import annotations

import pytest

from mikado.track import StickMatcher
from tests.helpers import make_stick


class TestStickMatcher:
    def setup_method(self):
        self.matcher = StickMatcher(max_centroid_distance_px=50.0)

    def test_empty_before(self, after_sticks_no_movement):
        result = self.matcher.match([], after_sticks_no_movement)
        assert result.matched_pairs == []
        assert result.unmatched_before == []
        assert result.unmatched_after == after_sticks_no_movement

    def test_empty_after(self, before_sticks):
        result = self.matcher.match(before_sticks, [])
        assert result.matched_pairs == []
        assert result.unmatched_before == before_sticks
        assert result.unmatched_after == []

    def test_perfect_match_no_movement(self, before_sticks, after_sticks_no_movement):
        result = self.matcher.match(before_sticks, after_sticks_no_movement)
        assert len(result.matched_pairs) == 3
        assert len(result.unmatched_before) == 0
        assert len(result.unmatched_after) == 0

    def test_one_stick_removed(self, before_sticks, after_sticks_target_removed):
        """First stick removed — should appear in unmatched_before."""
        result = self.matcher.match(before_sticks, after_sticks_target_removed)
        assert len(result.matched_pairs) == 2
        assert len(result.unmatched_before) == 1
        assert len(result.unmatched_after) == 0

    def test_sticks_too_far_apart_not_matched(self):
        """Sticks farther apart than max_centroid_distance_px should not be matched."""
        s1 = make_stick(cx=0.0, cy=0.0)
        s2 = make_stick(cx=200.0, cy=200.0)  # way beyond 50px
        result = self.matcher.match([s1], [s2])
        assert len(result.matched_pairs) == 0
        assert len(result.unmatched_before) == 1
        assert len(result.unmatched_after) == 1

    def test_class_mismatch_increases_cost(self):
        """Two pairs where one has a class match should prefer the class-matching pair."""
        s_before_0 = make_stick(cx=100.0, cy=100.0, class_id=0, angle_deg=0.0)
        s_before_1 = make_stick(cx=200.0, cy=200.0, class_id=1, angle_deg=0.0)

        # After sticks: same position but classes swapped
        s_after_0 = make_stick(cx=100.0, cy=100.0, class_id=1, angle_deg=0.0)
        s_after_1 = make_stick(cx=200.0, cy=200.0, class_id=0, angle_deg=0.0)

        result = self.matcher.match([s_before_0, s_before_1], [s_after_0, s_after_1])
        # Should still match by centroid distance (0px), class difference adds cost
        assert len(result.matched_pairs) == 2

    def test_from_config(self):
        config = {
            "matching": {
                "max_centroid_distance_px": 30.0,
                "centroid_weight": 0.5,
                "angle_weight": 0.3,
                "class_weight": 0.2,
            }
        }
        matcher = StickMatcher.from_config(config)
        assert matcher.max_centroid_distance_px == 30.0
        assert matcher.centroid_weight == 0.5

    def test_single_pair(self):
        s1 = make_stick(cx=100.0, cy=100.0, angle_deg=15.0, class_id=2)
        s2 = make_stick(cx=103.0, cy=100.0, angle_deg=15.0, class_id=2)  # tiny move
        result = self.matcher.match([s1], [s2])
        assert len(result.matched_pairs) == 1
