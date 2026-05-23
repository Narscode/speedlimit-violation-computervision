"""Analytics and reporting for SpeedGuard Vision outputs."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from speedguard_vision.speed_estimator import SpeedEvent
from speedguard_vision.types import Track, Violation


def read_violations(csv_path: Union[str, Path]) -> List[dict]:
    """Read violation rows from a CSV file."""
    path = Path(csv_path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class AnalyticsReporter:
    """Generate plots and a combined PDF report for a completed run."""

    def __init__(self, output_dir: Union[str, Path] = "outputs/analytics") -> None:
        """Create an analytics reporter with an output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        speed_events: Iterable[SpeedEvent],
        violations: Iterable[Violation],
        class_counts: Dict[str, int],
    ) -> Dict[str, Path]:
        """Generate histogram, frequency, class breakdown, and PDF report."""
        speed_events = list(speed_events)
        violations = list(violations)
        paths = {
            "speed_histogram": self.speed_distribution(speed_events),
            "violation_frequency": self.violation_frequency(violations),
            "class_breakdown": self.class_breakdown(class_counts),
        }
        paths["pdf_report"] = self.combined_pdf(paths.values())
        return paths

    def speed_distribution(self, speed_events: List[SpeedEvent]) -> Path:
        """Save a speed distribution histogram."""
        speeds = [event.speed_kmh for event in speed_events]
        path = self.output_dir / "speed_distribution.png"
        plt.figure(figsize=(9, 5))
        if speeds:
            plt.hist(speeds, bins=min(20, max(5, len(speeds))), color="#2563eb", edgecolor="white")
            plt.axvline(np.mean(speeds), color="#ef4444", linestyle="--", label=f"Mean {np.mean(speeds):.1f} km/h")
            plt.legend()
        else:
            plt.text(0.5, 0.5, "No completed speed events", ha="center", va="center")
        plt.title("Vehicle Speed Distribution")
        plt.xlabel("Speed (km/h)")
        plt.ylabel("Vehicle count")
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        return path

    def violation_frequency(self, violations: List[Violation]) -> Path:
        """Save a violation frequency-over-time plot."""
        path = self.output_dir / "violation_frequency.png"
        plt.figure(figsize=(9, 5))
        if violations:
            timestamps = [v.timestamp for v in violations]
            bins = min(20, max(5, len(timestamps)))
            plt.hist(timestamps, bins=bins, color="#dc2626", edgecolor="white")
        else:
            plt.text(0.5, 0.5, "No violations detected", ha="center", va="center")
        plt.title("Violation Frequency Over Time")
        plt.xlabel("Timestamp (seconds)")
        plt.ylabel("Violations")
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        return path

    def class_breakdown(self, class_counts: Dict[str, int]) -> Path:
        """Save a vehicle class breakdown pie chart."""
        path = self.output_dir / "vehicle_class_breakdown.png"
        plt.figure(figsize=(7, 6))
        if class_counts:
            labels = list(class_counts.keys())
            values = list(class_counts.values())
            plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
        else:
            plt.text(0.5, 0.5, "No tracked vehicles", ha="center", va="center")
        plt.title("Tracked Vehicle Class Breakdown")
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        return path

    def combined_pdf(self, image_paths: Iterable[Path]) -> Path:
        """Combine generated PNG plots into a single PDF report."""
        pdf_path = self.output_dir / "analytics_report.pdf"
        image_paths = list(image_paths)
        with PdfPages(pdf_path) as pdf:
            for image_path in image_paths:
                image = plt.imread(image_path)
                plt.figure(figsize=(11, 8.5))
                plt.imshow(image)
                plt.axis("off")
                plt.tight_layout()
                pdf.savefig()
                plt.close()
        return pdf_path


def summarize_speeds(speed_events: Iterable[SpeedEvent]) -> Dict[str, float]:
    """Return descriptive statistics for measured speed events."""
    speeds = np.array([event.speed_kmh for event in speed_events], dtype=float)
    if speeds.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    return {
        "count": float(speeds.size),
        "mean": float(np.mean(speeds)),
        "median": float(np.median(speeds)),
        "min": float(np.min(speeds)),
        "max": float(np.max(speeds)),
        "std": float(np.std(speeds)),
    }


def class_counter_from_tracks(tracks: Iterable[Track]) -> Counter:
    """Count track classes from an iterable of tracks."""
    return Counter(track.class_name for track in tracks)
