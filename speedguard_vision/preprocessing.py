"""Video and frame-folder preprocessing utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

Frame = np.ndarray


@dataclass
class FramePacket:
    """Container for one normalized frame and its timing metadata."""

    frame: Frame
    frame_number: int
    timestamp: float
    original_shape: Tuple[int, int, int]
    source_path: Optional[Path] = None


def sorted_frame_paths(folder: Union[str, Path]) -> List[Path]:
    """Return image files from a folder in natural-ish sequential order."""
    folder = Path(folder)
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in suffixes],
        key=lambda p: (len(p.stem), p.stem, p.suffix),
    )


def resize_frame(frame: Frame, size: Tuple[int, int] = (640, 640)) -> Frame:
    """Resize a frame to the requested `(width, height)` resolution."""
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def gaussian_blur(frame: Frame, kernel_size: int = 3) -> Frame:
    """Apply Gaussian blur for lightweight sensor-noise reduction."""
    if kernel_size <= 1:
        return frame
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)


def apply_roi_mask(frame: Frame, roi_points: Optional[Sequence[Sequence[float]]]) -> Frame:
    """Mask a frame to a road polygon while preserving frame dimensions."""
    if not roi_points:
        return frame
    points = np.array(roi_points, dtype=np.int32)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    return cv2.bitwise_and(frame, frame, mask=mask)


def preprocess_frame(
    frame: Frame,
    resize: Tuple[int, int] = (640, 640),
    roi_points: Optional[Sequence[Sequence[float]]] = None,
    blur_kernel: int = 3,
) -> Frame:
    """Normalize one frame for inference using resize, blur, and optional ROI masking."""
    frame = resize_frame(frame, resize)
    frame = gaussian_blur(frame, blur_kernel)
    frame = apply_roi_mask(frame, roi_points)
    return frame


def extract_frames_from_video(
    video_path: Union[str, Path],
    output_dir: Union[str, Path],
    target_fps: float = 25.0,
    resize: Tuple[int, int] = (640, 640),
    max_frames: Optional[int] = None,
) -> List[Path]:
    """Extract frames from a video at a normalized FPS into `output_dir`."""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    source_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    stride = max(1, int(round(source_fps / target_fps)))
    saved_paths: List[Path] = []
    frame_idx = 0
    saved_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            frame = resize_frame(frame, resize)
            path = output_dir / f"frame_{saved_idx:06d}.jpg"
            cv2.imwrite(str(path), frame)
            saved_paths.append(path)
            saved_idx += 1
            if max_frames and saved_idx >= max_frames:
                break
        frame_idx += 1
    cap.release()
    return saved_paths


class FrameSource:
    """Unified reader for MP4/AVI videos and folders of sequential image frames."""

    def __init__(
        self,
        source: Union[str, Path],
        fps: float = 25.0,
        resize: Tuple[int, int] = (640, 640),
        roi_points: Optional[Sequence[Sequence[float]]] = None,
        blur_kernel: int = 3,
        demo_mode: bool = False,
        demo_frames: int = 100,
    ) -> None:
        """Initialize a source reader with preprocessing and timing settings."""
        self.source = Path(source)
        self.target_fps = float(fps)
        self.resize = resize
        self.roi_points = roi_points
        self.blur_kernel = blur_kernel
        self.demo_mode = demo_mode
        self.demo_frames = demo_frames
        self._frame_paths = sorted_frame_paths(self.source) if self.source.is_dir() else []
        self._video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}

    def __iter__(self) -> Iterator[FramePacket]:
        """Yield normalized frames as `FramePacket` objects."""
        if self.source.is_dir():
            yield from self._iter_folder()
        elif self.source.suffix.lower() in self._video_extensions:
            yield from self._iter_video()
        else:
            raise ValueError(f"Unsupported input source: {self.source}")

    def __len__(self) -> int:
        """Return the expected number of frames when it can be estimated cheaply."""
        if self.source.is_dir():
            count = len(self._frame_paths)
        else:
            cap = cv2.VideoCapture(str(self.source))
            source_fps = cap.get(cv2.CAP_PROP_FPS) or self.target_fps
            raw_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
            count = math.ceil(raw_count / max(1, int(round(source_fps / self.target_fps))))
        return min(count, self.demo_frames) if self.demo_mode else count

    def _iter_folder(self) -> Iterator[FramePacket]:
        """Yield preprocessed frames from an image folder."""
        paths = self._frame_paths[: self.demo_frames] if self.demo_mode else self._frame_paths
        for idx, path in enumerate(paths):
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            original_shape = frame.shape
            frame = preprocess_frame(frame, self.resize, self.roi_points, self.blur_kernel)
            yield FramePacket(frame=frame, frame_number=idx, timestamp=idx / self.target_fps, original_shape=original_shape, source_path=path)

    def _iter_video(self) -> Iterator[FramePacket]:
        """Yield preprocessed frames from a video file with FPS normalization."""
        cap = cv2.VideoCapture(str(self.source))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {self.source}")
        source_fps = cap.get(cv2.CAP_PROP_FPS) or self.target_fps
        stride = max(1, int(round(source_fps / self.target_fps)))
        frame_idx = 0
        emitted_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % stride == 0:
                original_shape = frame.shape
                timestamp = frame_idx / source_fps
                frame = preprocess_frame(frame, self.resize, self.roi_points, self.blur_kernel)
                yield FramePacket(frame=frame, frame_number=emitted_idx, timestamp=timestamp, original_shape=original_shape)
                emitted_idx += 1
                if self.demo_mode and emitted_idx >= self.demo_frames:
                    break
            frame_idx += 1
        cap.release()
