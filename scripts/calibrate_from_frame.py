"""Create a perspective calibration JSON from an input video or frame folder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from speedguard_vision.calibration import build_calibration_from_clicks, save_calibration, visualize_calibration
from speedguard_vision.preprocessing import FrameSource


def parse_args() -> argparse.Namespace:
    """Parse calibration arguments."""
    parser = argparse.ArgumentParser(description="Interactively calibrate road perspective")
    parser.add_argument("--input", required=True, help="Video file or frame folder")
    parser.add_argument("--output", default="outputs/calibration.json", help="Calibration JSON output")
    parser.add_argument("--width_m", type=float, default=3.5, help="Known lane/reference width in meters")
    parser.add_argument("--length_m", type=float, default=10.0, help="Known road/reference length in meters")
    return parser.parse_args()


def main() -> None:
    """Open the first frame, collect four points, and save calibration artifacts."""
    args = parse_args()
    first_packet = next(iter(FrameSource(args.input, demo_mode=True, demo_frames=1)))
    config = build_calibration_from_clicks(first_packet.frame, args.width_m, args.length_m)
    save_calibration(config, args.output)
    preview = Path(args.output).with_suffix(".preview.jpg")
    visualize_calibration(first_packet.frame, config, preview)
    print(f"Calibration saved to {args.output}")
    print(f"Preview saved to {preview}")


if __name__ == "__main__":
    main()
