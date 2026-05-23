"""Shared data structures used across the SpeedGuard Vision pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

BBox = Tuple[float, float, float, float]
Point = Tuple[float, float]


@dataclass
class Detection:
    """Represents one detector output in image coordinates."""

    bbox: BBox
    confidence: float
    class_id: int
    class_name: str = "vehicle"

    def to_deepsort(self) -> Tuple[Tuple[float, float, float, float], float, str]:
        """Convert the detection to the `(ltwh, confidence, class)` Deep SORT format."""
        x1, y1, x2, y2 = self.bbox
        return (x1, y1, x2 - x1, y2 - y1), self.confidence, self.class_name


@dataclass
class Track:
    """Represents a tracked vehicle with a persistent identity."""

    track_id: int
    bbox: BBox
    class_id: int
    class_name: str = "vehicle"
    confidence: float = 1.0
    speed_kmh: Optional[float] = None
    is_confirmed: bool = True

    @property
    def centroid(self) -> Point:
        """Return the center point of the track bounding box."""
        x1, y1, x2, y2 = self.bbox
        return (float((x1 + x2) / 2), float((y1 + y2) / 2))

    @property
    def bottom_center(self) -> Point:
        """Return the road-contact point of the vehicle bounding box."""
        x1, _, x2, y2 = self.bbox
        return (float((x1 + x2) / 2), float(y2))


@dataclass
class Violation:
    """Represents a speed-limit violation event."""

    vehicle_id: int
    speed_kmh: float
    timestamp: float
    frame_number: int
    bbox: BBox
    class_name: str
    plate_text: str = "UNKNOWN"
    crop_path: Optional[Path] = None
    metadata: dict = field(default_factory=dict)
