"""License-plate recognition utilities for violating vehicles."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np


class LicensePlateRecognizer:
    """Detect and OCR a license plate from a vehicle crop."""

    def __init__(self, enabled: bool = True, languages: Optional[List[str]] = None, cascade_path: Union[str, Path] = "") -> None:
        """Initialize EasyOCR lazily and optionally load a plate cascade."""
        self.enabled = enabled
        self.languages = languages or ["en"]
        self.cascade_path = Path(cascade_path) if cascade_path else None
        self.reader = None
        self.cascade = None
        if self.cascade_path and self.cascade_path.exists():
            self.cascade = cv2.CascadeClassifier(str(self.cascade_path))

    def recognize(self, vehicle_crop: np.ndarray) -> str:
        """Return OCR text for a vehicle crop or `UNKNOWN` when unreadable."""
        if not self.enabled or vehicle_crop is None or vehicle_crop.size == 0:
            return "UNKNOWN"
        plate = self._detect_plate_region(vehicle_crop)
        text = self._ocr(plate)
        return text or "UNKNOWN"

    def _detect_plate_region(self, crop: np.ndarray) -> np.ndarray:
        """Locate a likely plate sub-region using cascade or lower-body heuristic."""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if self.cascade is not None:
            plates = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(25, 8))
            if len(plates) > 0:
                x, y, w, h = max(plates, key=lambda b: b[2] * b[3])
                return crop[y : y + h, x : x + w]
        h, w = crop.shape[:2]
        y1, y2 = int(h * 0.55), int(h * 0.95)
        x1, x2 = int(w * 0.15), int(w * 0.85)
        return crop[y1:y2, x1:x2]

    def _ocr(self, plate_crop: np.ndarray) -> str:
        """Run EasyOCR on a plate crop and normalize its text."""
        try:
            import easyocr
            if self.reader is None:
                self.reader = easyocr.Reader(self.languages, gpu=False)
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 7, 75, 75)
            result = self.reader.readtext(gray, detail=0, paragraph=False)
            cleaned = ["".join(ch for ch in item.upper() if ch.isalnum()) for item in result]
            cleaned = [item for item in cleaned if item]
            return " ".join(cleaned[:2]) if cleaned else "UNKNOWN"
        except Exception:
            return "UNKNOWN"


def build_lpr(config: dict) -> LicensePlateRecognizer:
    """Build a configured license-plate recognizer."""
    lpr_cfg = config.get("lpr", {})
    return LicensePlateRecognizer(
        enabled=bool(lpr_cfg.get("enabled", True)),
        languages=list(lpr_cfg.get("languages", ["en"])),
        cascade_path=lpr_cfg.get("cascade_path", ""),
    )
