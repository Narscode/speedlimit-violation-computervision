"""Deep SORT and fallback multi-object tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from speedguard_vision.types import Detection, Track


def iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Compute intersection-over-union for two `[x1, y1, x2, y2]` boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


@dataclass
class _SimpleState:
    """Internal state for the fallback tracker."""

    bbox: Tuple[float, float, float, float]
    class_id: int
    class_name: str
    age: int = 0
    hits: int = 1
    confidence: float = 1.0


class DeepSortVehicleTracker:
    """Vehicle tracker using Deep SORT with an IoU fallback for constrained installs."""

    def __init__(self, max_age: int = 30, min_hits: int = 3, max_cosine_distance: float = 0.4) -> None:
        """Configure Deep SORT association and track confirmation thresholds."""
        self.max_age = max_age
        self.min_hits = min_hits
        self.max_cosine_distance = max_cosine_distance
        self.backend = self._load_backend()
        self._simple_tracks: Dict[int, _SimpleState] = {}
        self._next_id = 1

    def _load_backend(self):
        """Load deep-sort-realtime if installed, otherwise return `None`."""
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
        except ImportError:
            return None
        return DeepSort(max_age=self.max_age, n_init=self.min_hits, max_cosine_distance=self.max_cosine_distance)

    def update(self, detections: Sequence[Detection], frame: np.ndarray) -> List[Track]:
        """Update tracks with current-frame detections and return active tracks."""
        if self.backend is not None:
            return self._update_deepsort(detections, frame)
        return self._update_simple(detections)

    def _update_deepsort(self, detections: Sequence[Detection], frame: np.ndarray) -> List[Track]:
        """Update tracks using the deep-sort-realtime package."""
        ds_detections = [det.to_deepsort() for det in detections]
        raw_tracks = self.backend.update_tracks(ds_detections, frame=frame)
        tracks: List[Track] = []
        class_by_name = {det.class_name: det.class_id for det in detections}
        for raw in raw_tracks:
            if not raw.is_confirmed() or raw.time_since_update > 1:
                continue
            x1, y1, x2, y2 = raw.to_ltrb()
            class_name = raw.det_class or "vehicle"
            tracks.append(
                Track(
                    track_id=int(raw.track_id),
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    class_id=int(class_by_name.get(class_name, 2)),
                    class_name=class_name,
                    is_confirmed=True,
                )
            )
        return tracks

    def _update_simple(self, detections: Sequence[Detection]) -> List[Track]:
        """Update tracks using greedy IoU matching as a deterministic fallback."""
        for state in self._simple_tracks.values():
            state.age += 1
        unmatched_detections = set(range(len(detections)))
        for track_id, state in list(self._simple_tracks.items()):
            best_idx: Optional[int] = None
            best_iou = 0.0
            for idx in unmatched_detections:
                score = iou(state.bbox, detections[idx].bbox)
                if score > best_iou:
                    best_iou, best_idx = score, idx
            if best_idx is not None and best_iou >= 0.2:
                det = detections[best_idx]
                state.bbox = det.bbox
                state.class_id = det.class_id
                state.class_name = det.class_name
                state.confidence = det.confidence
                state.age = 0
                state.hits += 1
                unmatched_detections.remove(best_idx)
        for idx in unmatched_detections:
            det = detections[idx]
            self._simple_tracks[self._next_id] = _SimpleState(det.bbox, det.class_id, det.class_name, confidence=det.confidence)
            self._next_id += 1
        self._simple_tracks = {tid: s for tid, s in self._simple_tracks.items() if s.age <= self.max_age}
        tracks: List[Track] = []
        for track_id, state in self._simple_tracks.items():
            if state.hits >= self.min_hits or self.min_hits <= 1:
                tracks.append(Track(track_id=track_id, bbox=state.bbox, class_id=state.class_id, class_name=state.class_name, confidence=state.confidence))
        return tracks


def build_tracker(config: dict) -> DeepSortVehicleTracker:
    """Build the configured vehicle tracker."""
    return DeepSortVehicleTracker(
        max_age=int(config["tracking"]["max_age"]),
        min_hits=int(config["tracking"]["min_hits"]),
        max_cosine_distance=float(config["tracking"]["max_cosine_distance"]),
    )
