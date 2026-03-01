"""Fault detection logic for Mikado Judge."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from mikado.detect import Stick
from mikado.track import MatchResult
from mikado.utils import centroid_distance, angle_diff, obb_iou

logger = logging.getLogger(__name__)


@dataclass
class StickMovement:
    """Movement metrics for a single matched stick pair."""

    before: Stick
    after: Stick
    centroid_displacement_px: float
    angle_change_deg: float
    iou: float
    moved: bool   # True if any threshold was exceeded


@dataclass
class JudgmentResult:
    """Result of judging a single player turn."""

    fault: bool
    target_stick: Stick | None            # The stick the player was removing
    moved_sticks: list[StickMovement]     # Non-target sticks that moved
    all_movements: list[StickMovement]    # Movements for ALL matched sticks
    removed_sticks: list[Stick]           # Sticks in before but not matched in after
    new_sticks: list[Stick]               # Sticks in after but not in before


class Judge:
    """Determines whether a fault occurred during a player's turn.

    A "fault" occurs when any stick OTHER than the target stick moves
    beyond the configured thresholds.

    The target stick is identified as the matched stick with the largest
    total movement, or (if nothing is matched) the first unmatched before-stick.

    Args:
        centroid_displacement_px: Movement threshold in pixels.
        angle_change_deg: Rotation threshold in degrees.
        iou_threshold: Minimum IoU between before/after OBB (below this → moved).
    """

    def __init__(
        self,
        centroid_displacement_px: float = 8.0,
        angle_change_deg: float = 1.5,
        iou_threshold: float = 0.85,
    ) -> None:
        self.centroid_displacement_px = centroid_displacement_px
        self.angle_change_deg = angle_change_deg
        self.iou_threshold = iou_threshold

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Judge":
        """Create a Judge from a judge.yaml movement config block."""
        mv = config.get("movement", {})
        return cls(
            centroid_displacement_px=mv.get("centroid_displacement_px", 8.0),
            angle_change_deg=mv.get("angle_change_deg", 1.5),
            iou_threshold=mv.get("iou_threshold", 0.85),
        )

    def judge(self, match_result: MatchResult) -> JudgmentResult:
        """Produce a fault judgment from a MatchResult.

        Args:
            match_result: Output of StickMatcher.match().

        Returns:
            JudgmentResult indicating whether a fault occurred.
        """
        all_movements = [
            self._compute_movement(before, after)
            for before, after in match_result.matched_pairs
        ]

        # Identify the target stick: the matched stick with the largest
        # composite movement score. If none matched, it's the first removed stick.
        target_stick: Stick | None = None
        target_movement: StickMovement | None = None

        if all_movements:
            target_movement = max(
                all_movements,
                key=lambda mv: mv.centroid_displacement_px + mv.angle_change_deg,
            )
            target_stick = target_movement.before
        elif match_result.unmatched_before:
            target_stick = match_result.unmatched_before[0]

        # Collect movements for non-target sticks that exceeded thresholds
        moved_sticks: list[StickMovement] = []
        for mv in all_movements:
            if mv.before is target_stick:
                continue
            if mv.moved:
                moved_sticks.append(mv)

        fault = len(moved_sticks) > 0

        if fault:
            logger.info("FAULT: %d non-target stick(s) moved", len(moved_sticks))
        else:
            logger.info("OK: No non-target sticks moved")

        return JudgmentResult(
            fault=fault,
            target_stick=target_stick,
            moved_sticks=moved_sticks,
            all_movements=all_movements,
            removed_sticks=match_result.unmatched_before,
            new_sticks=match_result.unmatched_after,
        )

    def _compute_movement(self, before: Stick, after: Stick) -> StickMovement:
        """Compute movement metrics between a matched stick pair."""
        displacement = centroid_distance(before.corners, after.corners)
        adiff = angle_diff(before.angle, after.angle)
        iou = obb_iou(before.corners, after.corners)

        moved = (
            displacement > self.centroid_displacement_px
            or adiff > self.angle_change_deg
            or iou < self.iou_threshold
        )

        return StickMovement(
            before=before,
            after=after,
            centroid_displacement_px=displacement,
            angle_change_deg=adiff,
            iou=iou,
            moved=moved,
        )
