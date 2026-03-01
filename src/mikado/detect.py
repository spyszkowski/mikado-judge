"""YOLO-OBB inference wrapper for Mikado stick detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from mikado.utils import Corners, obb_centroid, obb_angle

logger = logging.getLogger(__name__)


@dataclass
class Stick:
    """A single detected stick with oriented bounding box information."""

    corners: Corners        # (4, 2) float32 array of corner points
    class_id: int
    confidence: float
    class_name: str = ""

    @property
    def centroid(self) -> tuple[float, float]:
        return obb_centroid(self.corners)

    @property
    def angle(self) -> float:
        return obb_angle(self.corners)

    def __repr__(self) -> str:
        cx, cy = self.centroid
        return (
            f"Stick(class={self.class_name!r}, conf={self.confidence:.2f}, "
            f"centroid=({cx:.1f}, {cy:.1f}), angle={self.angle:.1f}°)"
        )


class Detector:
    """Wraps Ultralytics YOLO-OBB model for stick detection.

    Supports both YOLO11 and YOLO26 output formats.

    Args:
        model_path: Path to the .pt weights file.
        confidence_threshold: Minimum confidence to include a detection.
        iou_nms_threshold: NMS IoU threshold (used by YOLO11, ignored by YOLO26).
        input_size: Inference image size (pixels).
        class_names: Optional mapping from class_id to name.
    """

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.3,
        iou_nms_threshold: float = 0.5,
        input_size: int = 640,
        class_names: dict[int, str] | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("ultralytics is required: pip install ultralytics") from exc

        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_nms_threshold = iou_nms_threshold
        self.input_size = input_size
        self.class_names: dict[int, str] = class_names or {}

        logger.info("Loading YOLO-OBB model from %s", self.model_path)
        self._model = YOLO(str(self.model_path))
        logger.info("Model loaded successfully")

    @classmethod
    def from_config(cls, model_path: str | Path, config: dict[str, Any]) -> "Detector":
        """Create a Detector from a judge.yaml detection config block."""
        detection = config.get("detection", {})
        return cls(
            model_path=model_path,
            confidence_threshold=detection.get("confidence_threshold", 0.3),
            iou_nms_threshold=detection.get("iou_nms_threshold", 0.5),
            input_size=detection.get("input_size", 640),
        )

    def detect(self, frame: np.ndarray) -> list[Stick]:
        """Run detection on a single frame.

        Args:
            frame: BGR image as a numpy array (H, W, 3).

        Returns:
            List of Stick objects, filtered by confidence threshold.
        """
        results = self._model.predict(
            frame,
            imgsz=self.input_size,
            conf=self.confidence_threshold,
            iou=self.iou_nms_threshold,
            verbose=False,
        )

        sticks: list[Stick] = []
        for result in results:
            if result.obb is None:
                continue
            sticks.extend(self._parse_obb_result(result))

        logger.debug("Detected %d sticks", len(sticks))
        return sticks

    def _parse_obb_result(self, result: Any) -> list[Stick]:
        """Parse an Ultralytics OBB result object into Stick instances."""
        obb = result.obb
        sticks: list[Stick] = []

        # obb.xyxyxyxy: (N, 4, 2) tensor of corner points in pixel coords
        # obb.cls: (N,) class indices
        # obb.conf: (N,) confidence scores
        if obb.xyxyxyxy is None or len(obb.xyxyxyxy) == 0:
            return sticks

        corners_batch = obb.xyxyxyxy.cpu().numpy()   # (N, 4, 2)
        classes = obb.cls.cpu().numpy().astype(int)  # (N,)
        confs = obb.conf.cpu().numpy()               # (N,)

        # Use model names if class_names not provided
        model_names: dict[int, str] = self.class_names
        if not model_names and hasattr(result, "names"):
            model_names = result.names or {}

        for corners, class_id, conf in zip(corners_batch, classes, confs):
            stick = Stick(
                corners=corners.astype(np.float32),
                class_id=int(class_id),
                confidence=float(conf),
                class_name=model_names.get(int(class_id), str(class_id)),
            )
            sticks.append(stick)

        return sticks
