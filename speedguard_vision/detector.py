"""YOLOv8 vehicle detector integration."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import cv2
import numpy as np

from speedguard_vision.types import Detection

COCO_VEHICLE_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class YOLOVehicleDetector:
    """Vehicle detector powered by Ultralytics YOLOv8."""

    def __init__(
        self,
        weights: Union[str, Path] = "yolov8s.pt",
        confidence: float = 0.5,
        iou: float = 0.45,
        classes: Sequence[int] = (2, 3, 5, 7),
        device: Optional[str] = None,
    ) -> None:
        """Load a YOLO model and configure vehicle-class filtering."""
        self.weights = str(weights)
        self.confidence = confidence
        self.iou = iou
        self.classes = list(classes)
        self.device = device
        self.model = self._load_model()

    def _load_model(self):
        """Import Ultralytics lazily and load the configured model."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError("Ultralytics is not installed. Run `pip install ultralytics`.") from exc
        return YOLO(self.weights)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run vehicle detection on a frame and return filtered detections."""
        results = self.model.predict(
            frame,
            conf=self.confidence,
            iou=self.iou,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )
        detections: List[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].detach().cpu().numpy().astype(float)
                class_id = int(box.cls[0].detach().cpu().item())
                conf = float(box.conf[0].detach().cpu().item())
                detections.append(
                    Detection(
                        bbox=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                        confidence=conf,
                        class_id=class_id,
                        class_name=COCO_VEHICLE_NAMES.get(class_id, "vehicle"),
                    )
                )
        return detections


class ClassicalMotionDetector:
    """Lightweight fallback detector for smoke tests when YOLO is unavailable."""

    def __init__(self, min_area: int = 500) -> None:
        """Create a background-subtraction based detector."""
        self.min_area = min_area
        self.subtracter = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=32, detectShadows=True)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect moving blobs and emit generic vehicle detections."""
        mask = self.subtracter.apply(frame)
        mask = cv2.medianBlur(mask, 5)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: List[Detection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            detections.append(Detection(bbox=(x, y, x + w, y + h), confidence=0.35, class_id=2, class_name="vehicle"))
        return detections


def build_detector(config: dict, allow_fallback: bool = True):
    """Build a YOLO detector and optionally fall back to classical motion detection."""
    try:
        return YOLOVehicleDetector(
            weights=config["model"]["weights"],
            confidence=float(config["model"]["confidence"]),
            iou=float(config["model"]["iou"]),
            classes=config["model"]["classes"],
        )
    except RuntimeError:
        if not allow_fallback:
            raise
        return ClassicalMotionDetector()


def detections_to_array(detections: Iterable[Detection]) -> List[List[float]]:
    """Convert detection dataclasses to `[x1, y1, x2, y2, conf, class_id]` rows."""
    rows: List[List[float]] = []
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        rows.append([x1, y1, x2, y2, det.confidence, float(det.class_id)])
    return rows
