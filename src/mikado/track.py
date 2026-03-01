"""Stick matching between frames using the Hungarian algorithm."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from mikado.detect import Stick
from mikado.utils import centroid_distance, angle_diff

logger = logging.getLogger(__name__)

_INF_COST = 1e9


@dataclass
class MatchResult:
    """Result of matching sticks between two frames."""

    matched_pairs: list[tuple[Stick, Stick]]   # (before_stick, after_stick)
    unmatched_before: list[Stick]              # sticks removed (in before, not in after)
    unmatched_after: list[Stick]               # new / previously occluded sticks


class StickMatcher:
    """Matches sticks between a before-frame and after-frame using Hungarian algorithm.

    The cost matrix combines:
    - Centroid distance (penalised if > max_centroid_distance_px → infinite cost)
    - Angle difference (normalised to [0, 1] over 90°)
    - Class mismatch (0 if same class, 1 if different)

    Args:
        max_centroid_distance_px: Maximum centroid distance to allow a match.
        centroid_weight: Weight of centroid distance in cost.
        angle_weight: Weight of angle difference in cost.
        class_weight: Weight of class mismatch in cost.
    """

    def __init__(
        self,
        max_centroid_distance_px: float = 50.0,
        centroid_weight: float = 0.4,
        angle_weight: float = 0.3,
        class_weight: float = 0.3,
    ) -> None:
        self.max_centroid_distance_px = max_centroid_distance_px
        self.centroid_weight = centroid_weight
        self.angle_weight = angle_weight
        self.class_weight = class_weight

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "StickMatcher":
        """Create a StickMatcher from a judge.yaml matching config block."""
        m = config.get("matching", {})
        return cls(
            max_centroid_distance_px=m.get("max_centroid_distance_px", 50.0),
            centroid_weight=m.get("centroid_weight", 0.4),
            angle_weight=m.get("angle_weight", 0.3),
            class_weight=m.get("class_weight", 0.3),
        )

    def match(self, before_sticks: list[Stick], after_sticks: list[Stick]) -> MatchResult:
        """Match sticks between the before and after frames.

        Args:
            before_sticks: Sticks detected in the before frame.
            after_sticks: Sticks detected in the after frame.

        Returns:
            MatchResult with matched pairs and unmatched sticks.
        """
        if not before_sticks or not after_sticks:
            return MatchResult(
                matched_pairs=[],
                unmatched_before=list(before_sticks),
                unmatched_after=list(after_sticks),
            )

        cost_matrix = self._build_cost_matrix(before_sticks, after_sticks)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_pairs: list[tuple[Stick, Stick]] = []
        matched_before_idx: set[int] = set()
        matched_after_idx: set[int] = set()

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < _INF_COST / 2:
                matched_pairs.append((before_sticks[r], after_sticks[c]))
                matched_before_idx.add(r)
                matched_after_idx.add(c)

        unmatched_before = [s for i, s in enumerate(before_sticks) if i not in matched_before_idx]
        unmatched_after = [s for i, s in enumerate(after_sticks) if i not in matched_after_idx]

        logger.debug(
            "Matching: %d pairs, %d removed, %d new",
            len(matched_pairs), len(unmatched_before), len(unmatched_after),
        )
        return MatchResult(
            matched_pairs=matched_pairs,
            unmatched_before=unmatched_before,
            unmatched_after=unmatched_after,
        )

    def _build_cost_matrix(
        self, before_sticks: list[Stick], after_sticks: list[Stick]
    ) -> np.ndarray:
        """Build the N×M cost matrix for the Hungarian algorithm."""
        n = len(before_sticks)
        m = len(after_sticks)
        cost = np.full((n, m), _INF_COST, dtype=np.float64)

        for i, b in enumerate(before_sticks):
            for j, a in enumerate(after_sticks):
                dist = centroid_distance(b.corners, a.corners)
                if dist > self.max_centroid_distance_px:
                    continue  # leave as INF — impossible match

                # Normalise to [0, 1]
                dist_norm = dist / self.max_centroid_distance_px
                adiff_norm = angle_diff(b.angle, a.angle) / 90.0
                class_penalty = 0.0 if b.class_id == a.class_id else 1.0

                cost[i, j] = (
                    self.centroid_weight * dist_norm
                    + self.angle_weight * adiff_norm
                    + self.class_weight * class_penalty
                )

        return cost
