"""MediaPipe hand segmentation for masking the player's hand (Phase 2)."""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class HandMasker:
    """Detects and masks the player's hand using MediaPipe Hands.

    Phase 2 feature. When a hand is detected in the frame, a convex hull
    mask is computed and applied to the frame before stick detection runs.
    This prevents the hand from occluding sticks or being confused with them.

    Args:
        min_detection_confidence: Minimum confidence for hand detection.
        expand_px: Number of pixels to expand the hand mask outward (morphological dilation).
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        expand_px: int = 20,
    ) -> None:
        self.min_detection_confidence = min_detection_confidence
        self.expand_px = expand_px
        self._hands = None

    def _ensure_loaded(self) -> None:
        if self._hands is not None:
            return
        try:
            import mediapipe as mp
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=self.min_detection_confidence,
            )
        except ImportError as exc:
            raise ImportError("mediapipe is required for hand masking: pip install mediapipe") from exc

    def mask_hand(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Detect and mask the hand in the frame.

        Args:
            frame: BGR image (H, W, 3).

        Returns:
            Tuple of (masked_frame, mask) where mask is a uint8 binary image
            (255 = hand region) or None if no hand was detected.
        """
        self._ensure_loaded()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)

        if not result.multi_hand_landmarks:
            return frame.copy(), None

        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        for hand_landmarks in result.multi_hand_landmarks:
            points = np.array(
                [[int(lm.x * w), int(lm.y * h)] for lm in hand_landmarks.landmark],
                dtype=np.int32,
            )
            hull = cv2.convexHull(points)
            cv2.fillConvexPoly(mask, hull, 255)

        if self.expand_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.expand_px * 2 + 1, self.expand_px * 2 + 1),
            )
            mask = cv2.dilate(mask, kernel)

        masked_frame = frame.copy()
        masked_frame[mask == 255] = 0   # black out hand region

        logger.debug("Hand detected and masked")
        return masked_frame, mask
