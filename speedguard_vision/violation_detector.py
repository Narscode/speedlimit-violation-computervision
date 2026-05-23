"""Violation event detection, crop export, and CSV logging."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Union

import cv2
import numpy as np

from speedguard_vision.lpr import LicensePlateRecognizer
from speedguard_vision.speed_estimator import SpeedEvent
from speedguard_vision.types import Track, Violation


class ViolationDetector:
    """Compare speed estimates against a limit and persist violation evidence."""

    def __init__(
        self,
        speed_limit: float = 60.0,
        csv_path: Union[str, Path] = "outputs/violations.csv",
        crops_dir: Union[str, Path] = "outputs/violations",
        lpr: Optional[LicensePlateRecognizer] = None,
    ) -> None:
        """Create a detector with CSV and crop output destinations."""
        self.speed_limit = float(speed_limit)
        self.csv_path = Path(csv_path)
        self.crops_dir = Path(crops_dir)
        self.lpr = lpr
        self.violations: List[Violation] = []
        self._logged_track_ids: Set[int] = set()
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_csv_header()

    def process(
        self,
        tracks: Iterable[Track],
        speed_events: Iterable[SpeedEvent],
        frame: np.ndarray,
        frame_number: int,
    ) -> List[Violation]:
        """Create violation records for tracks whose measured speed exceeds the limit."""
        tracks_by_id: Dict[int, Track] = {track.track_id: track for track in tracks}
        new_violations: List[Violation] = []
        for event in speed_events:
            if event.speed_kmh <= self.speed_limit or event.track_id in self._logged_track_ids:
                continue
            track = tracks_by_id.get(event.track_id)
            if track is None:
                continue
            crop = self._crop_track(frame, track)
            crop_path = self._save_crop(crop, event.track_id, frame_number)
            plate_text = self.lpr.recognize(crop) if self.lpr else "UNKNOWN"
            violation = Violation(
                vehicle_id=event.track_id,
                speed_kmh=event.speed_kmh,
                timestamp=event.timestamp,
                frame_number=frame_number,
                bbox=track.bbox,
                class_name=track.class_name,
                plate_text=plate_text,
                crop_path=crop_path,
            )
            self.violations.append(violation)
            self._logged_track_ids.add(event.track_id)
            self._append_csv(violation)
            new_violations.append(violation)
        return new_violations

    def _ensure_csv_header(self) -> None:
        """Create the violations CSV with a header when it does not exist."""
        if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
            return
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["vehicle_id", "speed_kmh", "timestamp", "frame_number", "class_name", "plate_text", "bbox", "crop_path"])

    def _crop_track(self, frame: np.ndarray, track: Track) -> np.ndarray:
        """Crop a vehicle bounding box from the current frame."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in track.bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return frame[y1:y2, x1:x2].copy()

    def _save_crop(self, crop: np.ndarray, track_id: int, frame_number: int) -> Path:
        """Save a cropped vehicle image for audit evidence."""
        path = self.crops_dir / f"vehicle_{track_id}_frame_{frame_number}.jpg"
        if crop.size > 0:
            cv2.imwrite(str(path), crop)
        return path

    def _append_csv(self, violation: Violation) -> None:
        """Append one violation record to the CSV log."""
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    violation.vehicle_id,
                    f"{violation.speed_kmh:.2f}",
                    f"{violation.timestamp:.3f}",
                    violation.frame_number,
                    violation.class_name,
                    violation.plate_text,
                    list(map(float, violation.bbox)),
                    str(violation.crop_path or ""),
                ]
            )


def build_violation_detector(config: dict, lpr: Optional[LicensePlateRecognizer] = None) -> ViolationDetector:
    """Build the configured violation detector."""
    return ViolationDetector(
        speed_limit=float(config["speed"]["speed_limit"]),
        csv_path=config["output"]["violations_csv"],
        crops_dir=config["output"]["crops_dir"],
        lpr=lpr,
    )
