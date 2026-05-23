# Python
import unittest
import numpy as np
import tempfile
import shutil
import zipfile  # Fix for missing import
from pathlib import Path
from speedguard_vision.calibration import (
    compute_homography,
    compute_meters_per_pixel,
    save_calibration,
    load_calibration,
    CalibrationConfig,
)
from speedguard_vision.dataset_utils import extract_archive, discover_video_files
from speedguard_vision.preprocessing import resize_frame, gaussian_blur, apply_roi_mask
from speedguard_vision.speed_estimator import SpeedEstimator
from speedguard_vision.violation_detector import ViolationDetector
from speedguard_vision.types import Track, Violation

class TestCalibration(unittest.TestCase):
    def test_compute_homography(self):
        image_points = [(0, 0), (1, 0), (1, 1), (0, 1)]
        world_points = [(0, 0), (2, 0), (2, 2), (0, 2)]
        homography = compute_homography(image_points, world_points)
        self.assertEqual(homography.shape, (3, 3))

    def test_compute_meters_per_pixel(self):
        world_points = [(0, 0), (10, 0)]
        meters_per_pixel = compute_meters_per_pixel(world_points, 10)
        self.assertAlmostEqual(meters_per_pixel, 1.0)

    def test_save_and_load_calibration(self):
        config = CalibrationConfig(
            image_points=[(0, 0), (1, 0), (1, 1), (0, 1)],
            world_points=[(0, 0), (2, 0), (2, 2), (0, 2)],
            homography=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.json"
            save_calibration(config, path)
            loaded_config = load_calibration(path)
            self.assertEqual(config.image_points, [tuple(point) for point in loaded_config.image_points])

class TestDatasetUtils(unittest.TestCase):
    def test_extract_archive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "test.zip"
            output_dir = Path(tmpdir) / "output"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("test.txt", "content")
            extracted_path = extract_archive(archive_path, output_dir)
            self.assertTrue((extracted_path / "test.txt").exists())

    def test_discover_video_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "video.mp4"
            video_path.touch()
            videos = discover_video_files(tmpdir)
            self.assertIn(video_path, videos)

class TestPreprocessing(unittest.TestCase):
    def test_resize_frame(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        resized = resize_frame(frame, (50, 50))
        self.assertEqual(resized.shape, (50, 50, 3))

    def test_gaussian_blur(self):
        frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        blurred = gaussian_blur(frame, 3)
        self.assertEqual(blurred.shape, frame.shape)

    def test_apply_roi_mask(self):
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 255
        roi_points = [[10, 10], [90, 10], [90, 90], [10, 90]]
        masked = apply_roi_mask(frame, roi_points)
        self.assertEqual(masked[0, 0, 0], 0)  # Outside ROI

class TestSpeedEstimator(unittest.TestCase):
    def test_speed_estimator(self):
        tracks = [Track(track_id=1, class_id=0, bbox=(0, 0, 10, 10))]  # Fix for missing class_id
        estimator = SpeedEstimator(frame_height=100, real_world_distance=10.0)
        events = estimator.update(tracks, timestamp=1.0)
        self.assertEqual(len(events), 0)

class TestViolationDetector(unittest.TestCase):
    def test_violation_detector(self):
        detector = ViolationDetector(speed_limit=50.0)
        tracks = [Track(track_id=1, class_id=0, bbox=(0, 0, 10, 10))]  # Fix for missing class_id
        speed_events = []
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        violations = detector.process(tracks, speed_events, frame, frame_number=1)
        self.assertEqual(len(violations), 0)

if __name__ == "__main__":
    unittest.main()
