"""Line-crossing based speed estimation."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from speedguard_vision.calibration import CalibrationConfig, project_point
from speedguard_vision.types import Track


@dataclass
class SpeedEvent:
    """Represents one completed speed measurement."""

    track_id: int
    speed_kmh: float
    timestamp: float
    elapsed_seconds: float


@dataclass
class _TrackTiming:
    """Internal timing state for virtual line crossings."""

    previous_y: Optional[float] = None
    line_a_time: Optional[float] = None
    line_b_time: Optional[float] = None
    samples: Deque[float] = field(default_factory=lambda: deque(maxlen=5))


class SpeedEstimator:
    """Estimate vehicle speed from crossings between two virtual lines."""

    def __init__(
        self,
        frame_height: int = 640,
        line_a_y: float = 0.4,
        line_b_y: float = 0.6,
        real_world_distance: float = 10.0,
        smoothing_window: int = 5,
        calibration: Optional[CalibrationConfig] = None,
    ) -> None:
        """Create a speed estimator with configurable lines and calibration."""
        self.frame_height = frame_height
        self.line_a_y_fraction = line_a_y
        self.line_b_y_fraction = line_b_y
        self.line_a_y = int(frame_height * line_a_y)
        self.line_b_y = int(frame_height * line_b_y)
        self.real_world_distance = float(real_world_distance)
        self.smoothing_window = smoothing_window
        self.calibration = calibration
        self._states: Dict[int, _TrackTiming] = defaultdict(lambda: _TrackTiming(samples=deque(maxlen=smoothing_window)))
        self.latest_speeds: Dict[int, float] = {}
        self.events: List[SpeedEvent] = []

    def update(self, tracks: List[Track], timestamp: float) -> List[SpeedEvent]:
        """Update crossing state for all tracks and return newly completed speed events."""
        new_events: List[SpeedEvent] = []
        for track in tracks:
            y = self._road_y(track)
            state = self._states[track.track_id]
            if state.previous_y is not None:
                self._register_crossing(state, state.previous_y, y, timestamp)
                event = self._maybe_compute_speed(track.track_id, state, timestamp)
                if event:
                    track.speed_kmh = event.speed_kmh
                    self.latest_speeds[track.track_id] = event.speed_kmh
                    self.events.append(event)
                    new_events.append(event)
            if track.track_id in self.latest_speeds:
                track.speed_kmh = self.latest_speeds[track.track_id]
            state.previous_y = y
        return new_events

    def _road_y(self, track: Track) -> float:
        """Return the projected road-plane y coordinate used for line crossing."""
        point = track.bottom_center
        if self.calibration is not None:
            return project_point(point, self.calibration.matrix)[1]
        return point[1]

    def _register_crossing(self, state: _TrackTiming, prev_y: float, current_y: float, timestamp: float) -> None:
        """Record virtual line crossing timestamps for a track."""
        for line_name, line_y in (("a", self.line_a_y), ("b", self.line_b_y)):
            crossed_down = prev_y < line_y <= current_y
            crossed_up = prev_y > line_y >= current_y
            if crossed_down or crossed_up:
                if line_name == "a" and state.line_a_time is None:
                    state.line_a_time = timestamp
                if line_name == "b" and state.line_b_time is None:
                    state.line_b_time = timestamp

    def _maybe_compute_speed(self, track_id: int, state: _TrackTiming, timestamp: float) -> Optional[SpeedEvent]:
        """Compute smoothed speed after both lines have been crossed."""
        if state.line_a_time is None or state.line_b_time is None:
            return None
        elapsed = abs(state.line_b_time - state.line_a_time)
        if elapsed <= 1e-6:
            return None
        raw_speed = (self.real_world_distance / elapsed) * 3.6
        state.line_a_time = None
        state.line_b_time = None
        if raw_speed < 0 or raw_speed > 300:
            return None
        state.samples.append(raw_speed)
        smoothed = float(np.mean(state.samples))
        return SpeedEvent(track_id=track_id, speed_kmh=smoothed, timestamp=timestamp, elapsed_seconds=elapsed)

    def line_coordinates(self, width: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Return `(Line A y endpoints, Line B y endpoints)` for drawing."""
        return (0, self.line_a_y), (width, self.line_b_y)


def build_speed_estimator(config: dict, frame_height: int = 640, calibration: Optional[CalibrationConfig] = None) -> SpeedEstimator:
    """Build a configured speed estimator."""
    speed_cfg = config["speed"]
    return SpeedEstimator(
        frame_height=frame_height,
        line_a_y=float(speed_cfg["line_a_y"]),
        line_b_y=float(speed_cfg["line_b_y"]),
        real_world_distance=float(speed_cfg["real_world_distance"]),
        smoothing_window=int(speed_cfg["smoothing_window"]),
        calibration=calibration,
    )
