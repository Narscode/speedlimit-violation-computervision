"""Perspective calibration helpers for road-plane speed estimation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

Point = Tuple[float, float]


@dataclass
class CalibrationConfig:
    """Stores homography and scale data for image-to-road projection."""

    image_points: List[Point]
    world_points: List[Point]
    homography: List[List[float]]
    meters_per_pixel: Optional[float] = None
    output_size: Tuple[int, int] = (640, 640)

    @property
    def matrix(self) -> np.ndarray:
        """Return the homography as a `numpy.ndarray`."""
        return np.array(self.homography, dtype=np.float32)


def compute_homography(image_points: Sequence[Point], world_points: Sequence[Point]) -> np.ndarray:
    """Compute a perspective transform from four image points to four world-plane points."""
    if len(image_points) != 4 or len(world_points) != 4:
        raise ValueError("Exactly four image points and four world points are required.")
    return cv2.getPerspectiveTransform(np.float32(image_points), np.float32(world_points))


def compute_meters_per_pixel(world_points: Sequence[Point], known_distance_m: float) -> float:
    """Estimate meters per pixel from the first two warped reference points."""
    p1 = np.array(world_points[0], dtype=float)
    p2 = np.array(world_points[1], dtype=float)
    pixel_distance = float(np.linalg.norm(p2 - p1))
    if pixel_distance <= 0:
        raise ValueError("Reference points must not overlap.")
    return float(known_distance_m / pixel_distance)


def save_calibration(config: CalibrationConfig, path: Union[str, Path]) -> None:
    """Save calibration data as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)


def load_calibration(path: Union[str, Path]) -> CalibrationConfig:
    """Load calibration data from JSON."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return CalibrationConfig(**data)


def warp_birds_eye(frame: np.ndarray, config: CalibrationConfig) -> np.ndarray:
    """Warp a frame to a bird's-eye road-plane view."""
    return cv2.warpPerspective(frame, config.matrix, tuple(config.output_size))


def project_point(point: Point, homography: np.ndarray) -> Point:
    """Project a single image point through a homography matrix."""
    pts = np.array([[[point[0], point[1]]]], dtype=np.float32)
    projected = cv2.perspectiveTransform(pts, homography)[0, 0]
    return float(projected[0]), float(projected[1])


def visualize_calibration(frame: np.ndarray, config: CalibrationConfig, output_path: Union[str, Path]) -> Path:
    """Save a side-by-side source and bird's-eye calibration visualization."""
    warped = warp_birds_eye(frame, config)
    source = frame.copy()
    pts = np.array(config.image_points, dtype=np.int32)
    cv2.polylines(source, [pts], True, (0, 255, 255), 2)
    combined = np.hstack([source, warped])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), combined)
    return output_path


class ReferencePointSelector:
    """Interactive OpenCV selector for four road reference points."""

    def __init__(self, window_name: str = "Select four road points") -> None:
        """Create a selector with an OpenCV window label."""
        self.window_name = window_name
        self.points: List[Point] = []

    def select(self, frame: np.ndarray) -> List[Point]:
        """Open an interactive window and return four user-clicked points."""
        clone = frame.copy()
        self.points = []

        def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
            """Collect left-click points for calibration."""
            if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
                self.points.append((float(x), float(y)))
                cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(clone, str(len(self.points)), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow(self.window_name, clone)

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, on_mouse)
        cv2.imshow(self.window_name, clone)
        while len(self.points) < 4:
            if cv2.waitKey(20) & 0xFF == 27:
                break
        cv2.destroyWindow(self.window_name)
        if len(self.points) != 4:
            raise RuntimeError("Calibration cancelled before four points were selected.")
        return self.points


def build_calibration_from_clicks(
    frame: np.ndarray,
    known_width_m: float,
    known_length_m: float,
    output_size: Tuple[int, int] = (640, 640),
) -> CalibrationConfig:
    """Collect four points interactively and build a rectangular bird's-eye calibration."""
    selector = ReferencePointSelector()
    image_points = selector.select(frame)
    world_points = [(0.0, 0.0), (known_width_m * 20, 0.0), (known_width_m * 20, known_length_m * 20), (0.0, known_length_m * 20)]
    homography = compute_homography(image_points, world_points)
    meters_per_pixel = compute_meters_per_pixel(world_points, known_width_m)
    return CalibrationConfig(image_points=image_points, world_points=world_points, homography=homography.tolist(), meters_per_pixel=meters_per_pixel, output_size=output_size)
