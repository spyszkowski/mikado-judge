"""Frame alignment via ORB features and homography estimation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AlignmentResult:
    """Result of aligning two frames."""

    aligned_frame: np.ndarray   # The 'after' frame warped to match 'before'
    homography: np.ndarray      # 3×3 homography matrix (or identity on failure)
    n_inliers: int              # Number of RANSAC inliers
    success: bool               # Whether alignment was considered reliable


class FrameAligner:
    """Aligns a pair of frames using ORB feature matching and RANSAC homography.

    Corrects for small camera movements between the before and after shots.

    Args:
        min_features: Minimum number of inliers for reliable alignment.
        ransac_threshold: RANSAC reprojection error threshold in pixels.
        warn_if_features_below: Log a warning when inlier count is below this.
        max_features: Maximum ORB keypoints to detect per frame.
    """

    def __init__(
        self,
        min_features: int = 20,
        ransac_threshold: float = 5.0,
        warn_if_features_below: int = 50,
        max_features: int = 5000,
    ) -> None:
        self.min_features = min_features
        self.ransac_threshold = ransac_threshold
        self.warn_if_features_below = warn_if_features_below
        self._orb = cv2.ORB_create(nfeatures=max_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "FrameAligner":
        """Create a FrameAligner from a judge.yaml alignment config block."""
        alignment = config.get("alignment", {})
        return cls(
            min_features=alignment.get("min_features", 20),
            ransac_threshold=alignment.get("ransac_threshold", 5.0),
            warn_if_features_below=alignment.get("warn_if_features_below", 50),
        )

    def align(self, frame_before: np.ndarray, frame_after: np.ndarray) -> AlignmentResult:
        """Align `frame_after` to `frame_before` using ORB + RANSAC homography.

        Args:
            frame_before: Reference frame (before player's turn).
            frame_after: Frame to warp (after player's turn).

        Returns:
            AlignmentResult with the warped frame and alignment quality info.
        """
        gray_before = self._to_gray(frame_before)
        gray_after = self._to_gray(frame_after)

        kp_before, desc_before = self._orb.detectAndCompute(gray_before, None)
        kp_after, desc_after = self._orb.detectAndCompute(gray_after, None)

        if desc_before is None or desc_after is None or len(kp_before) < 4 or len(kp_after) < 4:
            logger.warning("Too few keypoints for alignment (before=%d, after=%d)", len(kp_before), len(kp_after))
            return self._identity_result(frame_after)

        matches = self._matcher.match(desc_before, desc_after)
        matches = sorted(matches, key=lambda m: m.distance)

        if len(matches) < 4:
            logger.warning("Too few feature matches (%d) for homography", len(matches))
            return self._identity_result(frame_after)

        pts_before = np.float32([kp_before[m.queryIdx].pt for m in matches])
        pts_after = np.float32([kp_after[m.trainIdx].pt for m in matches])

        H, mask = cv2.findHomography(
            pts_after,
            pts_before,
            cv2.RANSAC,
            self.ransac_threshold,
        )

        if H is None:
            logger.warning("Homography estimation failed")
            return self._identity_result(frame_after)

        n_inliers = int(mask.sum()) if mask is not None else 0

        if n_inliers < self.warn_if_features_below:
            logger.warning(
                "Only %d homography inliers — camera may have moved significantly",
                n_inliers,
            )

        if n_inliers < self.min_features:
            logger.error(
                "Alignment unreliable: %d inliers < minimum %d. "
                "Consider recalibrating the camera.",
                n_inliers, self.min_features,
            )
            return AlignmentResult(
                aligned_frame=frame_after,
                homography=H,
                n_inliers=n_inliers,
                success=False,
            )

        h, w = frame_before.shape[:2]
        aligned = cv2.warpPerspective(frame_after, H, (w, h))

        logger.debug("Alignment succeeded with %d inliers", n_inliers)
        return AlignmentResult(
            aligned_frame=aligned,
            homography=H,
            n_inliers=n_inliers,
            success=True,
        )

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    @staticmethod
    def _identity_result(frame: np.ndarray) -> AlignmentResult:
        return AlignmentResult(
            aligned_frame=frame,
            homography=np.eye(3, dtype=np.float64),
            n_inliers=0,
            success=False,
        )
