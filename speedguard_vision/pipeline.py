"""Main real-time speed-limit violation detection pipeline."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import cv2
from tqdm import tqdm

from speedguard_vision.analytics import AnalyticsReporter, summarize_speeds
from speedguard_vision.annotator import FrameAnnotator
from speedguard_vision.calibration import load_calibration
from speedguard_vision.config import apply_cli_overrides, ensure_output_dirs, load_config
from speedguard_vision.detector import build_detector
from speedguard_vision.evaluation import detection_metrics, empty_metric_report, parse_ua_detrac_xml
from speedguard_vision.lpr import build_lpr
from speedguard_vision.preprocessing import FrameSource
from speedguard_vision.speed_estimator import build_speed_estimator
from speedguard_vision.tracker import build_tracker
from speedguard_vision.types import Track
from speedguard_vision.violation_detector import build_violation_detector


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the SpeedGuard pipeline."""
    parser = argparse.ArgumentParser(description="SpeedGuard Vision real-time speed violation detector")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML/JSON config file")
    parser.add_argument("--input", help="Input video file or folder of sequential frames")
    parser.add_argument("--speed_limit", type=float, help="Speed limit in km/h")
    parser.add_argument("--fps", type=float, help="Normalized processing FPS")
    parser.add_argument("--calibration", help="Path to calibration JSON file")
    parser.add_argument("--output", help="Annotated output video path")
    parser.add_argument("--weights", help="YOLOv8 weights path/name")
    parser.add_argument("--demo", action="store_true", help="Process only demo_frames for quick testing")
    parser.add_argument("--full", action="store_true", help="Process all frames")
    parser.add_argument("--live", action="store_true", help="Display live annotated frames")
    parser.add_argument("--no_live", action="store_true", help="Disable live display")
    return parser.parse_args()


class SpeedGuardPipeline:
    """End-to-end detector, tracker, speed estimator, violation logger, and reporter."""

    def __init__(self, config: dict) -> None:
        """Initialize all modules from a normalized configuration dictionary."""
        self.config = config
        ensure_output_dirs(config)
        calibration_path = config["calibration"].get("config_path")
        self.calibration = load_calibration(calibration_path) if calibration_path else None
        resize = tuple(config["input"].get("resize", [640, 640]))
        self.frame_source = FrameSource(
            config["input"]["source"],
            fps=float(config["speed"]["fps"]),
            resize=resize,
            roi_points=config["input"].get("roi_points", []),
            demo_mode=bool(config["input"].get("demo_mode", False)),
            demo_frames=int(config["input"].get("demo_frames", 100)),
        )
        self.detector = build_detector(config)
        self.tracker = build_tracker(config)
        self.speed_estimator = build_speed_estimator(config, frame_height=int(resize[1]), calibration=self.calibration)
        self.lpr = build_lpr(config)
        self.violation_detector = build_violation_detector(config, self.lpr)
        self.annotator = FrameAnnotator(
            speed_limit=float(config["speed"]["speed_limit"]),
            line_a_y=self.speed_estimator.line_a_y,
            line_b_y=self.speed_estimator.line_b_y,
        )
        self.class_counts: Counter = Counter()
        self.detection_boxes: Dict[int, List[tuple]] = defaultdict(list)
        self.last_tracks: List[Track] = []

    def run(self) -> dict:
        """Process all configured frames and return a run summary."""
        output_video = Path(self.config["output"]["video_path"])
        writer = None
        total = len(self.frame_source)
        start_time = time.perf_counter()
        recent_violating_ids = set()
        progress = tqdm(total=total if total > 0 else None, desc="SpeedGuard", unit="frame")
        for packet in self.frame_source:
            loop_start = time.perf_counter()
            detections = self.detector.detect(packet.frame)
            self.detection_boxes[packet.frame_number] = [det.bbox for det in detections]
            tracks = self.tracker.update(detections, packet.frame)
            speed_events = self.speed_estimator.update(tracks, packet.timestamp)
            violations = self.violation_detector.process(tracks, speed_events, packet.frame, packet.frame_number)
            recent_violating_ids.update(v.vehicle_id for v in violations)
            for track in tracks:
                self.class_counts[track.class_name] += 1
            elapsed = max(time.perf_counter() - loop_start, 1e-6)
            instantaneous_fps = 1.0 / elapsed
            annotated = self.annotator.annotate(packet.frame, tracks, recent_violating_ids, packet.frame_number, instantaneous_fps)
            if self.config["output"].get("save_video", True):
                writer = self._ensure_writer(writer, output_video, annotated)
                writer.write(annotated)
            if self.config["output"].get("show_live", False):
                cv2.imshow("SpeedGuard Vision", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            progress.set_postfix({"fps": f"{instantaneous_fps:.1f}", "tracks": len(tracks), "violations": len(self.violation_detector.violations)})
            progress.update(1)
            self.last_tracks = tracks
        progress.close()
        if writer is not None:
            writer.release()
        if self.config["output"].get("show_live", False):
            cv2.destroyAllWindows()
        analytics = AnalyticsReporter(self.config["output"]["analytics_dir"]).generate(
            self.speed_estimator.events,
            self.violation_detector.violations,
            dict(self.class_counts),
        )
        metrics = self._evaluate()
        summary = {
            "processed_frames": progress.n,
            "runtime_seconds": round(time.perf_counter() - start_time, 3),
            "average_fps": round(progress.n / max(time.perf_counter() - start_time, 1e-6), 2),
            "violations": len(self.violation_detector.violations),
            "speed_stats": summarize_speeds(self.speed_estimator.events),
            "metrics": metrics,
            "outputs": {key: str(path) for key, path in analytics.items()},
            "video": str(output_video),
            "csv": self.config["output"]["violations_csv"],
        }
        print(json.dumps(summary, indent=2))
        return summary

    def _ensure_writer(self, writer, output_path: Path, frame):
        """Create a video writer lazily after the first frame is available."""
        if writer is not None:
            return writer
        output_path.parent.mkdir(parents=True, exist_ok=True)
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = float(self.config["speed"]["fps"])
        return cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    def _evaluate(self) -> dict:
        """Compute metrics when ground truth is configured, otherwise return descriptive placeholders."""
        report = empty_metric_report()
        gt_path = self.config.get("evaluation", {}).get("ground_truth", "")
        if gt_path and Path(gt_path).exists() and Path(gt_path).suffix.lower() == ".xml":
            gt = parse_ua_detrac_xml(gt_path)
            report["detection"] = detection_metrics(self.detection_boxes, gt)
        return report


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    pipeline = SpeedGuardPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
