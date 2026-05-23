"""Visualization and video annotation utilities."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

from speedguard_vision.types import Track


def speed_color(speed: Optional[float], speed_limit: float) -> Tuple[int, int, int]:
    """Return green, orange, or red depending on speed relative to the limit."""
    if speed is None:
        return (170, 170, 170)
    if speed > speed_limit:
        return (0, 0, 255)
    if speed > speed_limit * 0.85:
        return (0, 165, 255)
    return (0, 200, 0)


class FrameAnnotator:
    """Draw detections, tracks, speed lines, and dashboard overlays."""

    def __init__(self, speed_limit: float = 60.0, line_a_y: int = 256, line_b_y: int = 384) -> None:
        """Create an annotator with speed limit and virtual line positions."""
        self.speed_limit = speed_limit
        self.line_a_y = line_a_y
        self.line_b_y = line_b_y

    def annotate(
        self,
        frame: np.ndarray,
        tracks: Sequence[Track],
        violating_ids: Iterable[int] = (),
        frame_number: int = 0,
        fps: float = 0.0,
    ) -> np.ndarray:
        """Return an annotated copy of the input frame."""
        annotated = frame.copy()
        violating_ids = set(violating_ids)
        self._draw_lines(annotated)
        for track in tracks:
            self._draw_track(annotated, track, track.track_id in violating_ids, frame_number)
        self._draw_speedometer(annotated, tracks, fps)
        return annotated

    def _draw_lines(self, frame: np.ndarray) -> None:
        """Draw virtual speed measurement lines."""
        h, w = frame.shape[:2]
        for y, label, color in ((self.line_a_y, "Line A", (255, 220, 0)), (self.line_b_y, "Line B", (255, 120, 0))):
            y = max(0, min(h - 1, int(y)))
            cv2.line(frame, (0, y), (w, y), color, 2)
            cv2.putText(frame, label, (12, max(22, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def _draw_track(self, frame: np.ndarray, track: Track, is_violation: bool, frame_number: int) -> None:
        """Draw a single tracked vehicle."""
        x1, y1, x2, y2 = [int(round(v)) for v in track.bbox]
        color = speed_color(track.speed_kmh, self.speed_limit)
        if is_violation and frame_number % 10 < 5:
            color = (0, 0, 255)
        thickness = 4 if is_violation else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        speed_text = "--" if track.speed_kmh is None else f"{track.speed_kmh:.1f} km/h"
        label = f"ID {track.track_id} {track.class_name} {speed_text}"
        self._label(frame, label, x1, max(0, y1 - 8), color)

    def _label(self, frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
        """Draw a readable text label with a filled background."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        y = max(th + baseline + 4, y)
        cv2.rectangle(frame, (x, y - th - baseline - 6), (x + tw + 8, y + baseline), color, -1)
        cv2.putText(frame, text, (x + 4, y - 4), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    def _draw_speedometer(self, frame: np.ndarray, tracks: Sequence[Track], fps: float) -> None:
        """Draw a compact speedometer-style status panel."""
        speeds = [t.speed_kmh for t in tracks if t.speed_kmh is not None]
        max_speed = max(speeds) if speeds else 0.0
        h, w = frame.shape[:2]
        cx, cy = w - 85, 82
        radius = 48
        cv2.circle(frame, (cx, cy), radius, (30, 30, 30), -1)
        cv2.circle(frame, (cx, cy), radius, (230, 230, 230), 2)
        ratio = min(max_speed / max(self.speed_limit * 1.5, 1), 1.0)
        angle = np.deg2rad(210 + ratio * 240)
        end = (int(cx + np.cos(angle) * (radius - 12)), int(cy + np.sin(angle) * (radius - 12)))
        cv2.line(frame, (cx, cy), end, (0, 0, 255) if max_speed > self.speed_limit else (0, 220, 0), 3)
        cv2.putText(frame, f"{max_speed:.0f}", (cx - 20, cy + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS {fps:.1f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, f"Limit {self.speed_limit:.0f} km/h", (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
