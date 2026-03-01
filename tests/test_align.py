"""Tests for frame alignment (mikado.align)."""

from __future__ import annotations

import numpy as np
import pytest

from mikado.align import FrameAligner


def _make_textured_frame(h: int = 480, w: int = 640, seed: int = 0) -> np.ndarray:
    """Create a synthetic frame with enough texture for ORB matching."""
    rng = np.random.default_rng(seed)
    frame = rng.integers(50, 200, (h, w, 3), dtype=np.uint8)
    # Add some rectangles to give ORB more features
    for _ in range(20):
        x = int(rng.integers(0, w - 50))
        y = int(rng.integers(0, h - 50))
        colour = tuple(int(c) for c in rng.integers(0, 256, 3))
        frame[y:y+40, x:x+40] = colour
    return frame


class TestFrameAligner:
    def test_identity_on_identical_frames(self):
        """Aligning a frame with itself should produce a near-identity homography."""
        frame = _make_textured_frame()
        aligner = FrameAligner(min_features=4, warn_if_features_below=4)
        result = aligner.align(frame, frame)
        assert result.success
        assert result.n_inliers > 0
        # H should be close to identity
        np.testing.assert_allclose(result.homography, np.eye(3), atol=0.1)

    def test_small_translation(self):
        """Aligning a slightly shifted frame should succeed with high inlier count."""
        import cv2
        frame_before = _make_textured_frame()
        # Translate by 5 pixels
        M = np.float32([[1, 0, 5], [0, 1, 3]])
        frame_after = cv2.warpAffine(frame_before, M, (640, 480))
        aligner = FrameAligner(min_features=4, warn_if_features_below=10)
        result = aligner.align(frame_before, frame_after)
        assert result.success
        assert result.n_inliers >= 4

    def test_blank_frames_fail_gracefully(self):
        """Blank frames have no features — alignment should return success=False, not crash."""
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        aligner = FrameAligner(min_features=4)
        result = aligner.align(blank, blank)
        assert not result.success
        # Should return the original after-frame unchanged
        np.testing.assert_array_equal(result.aligned_frame, blank)

    def test_from_config(self):
        config = {
            "alignment": {
                "min_features": 15,
                "ransac_threshold": 3.0,
                "warn_if_features_below": 40,
            }
        }
        aligner = FrameAligner.from_config(config)
        assert aligner.min_features == 15
        assert aligner.ransac_threshold == 3.0
        assert aligner.warn_if_features_below == 40

    def test_aligned_frame_shape_matches_before(self):
        """Aligned frame should have the same shape as the before frame."""
        frame_before = _make_textured_frame(480, 640)
        frame_after = _make_textured_frame(480, 640, seed=1)
        aligner = FrameAligner(min_features=4)
        result = aligner.align(frame_before, frame_after)
        assert result.aligned_frame.shape == frame_before.shape
