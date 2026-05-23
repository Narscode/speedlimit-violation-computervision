"""Prepare local or Kaggle traffic datasets for SpeedGuard Vision."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from speedguard_vision.dataset_utils import copy_sample_video, discover_video_files, download_ua_detrac_with_kagglehub, extract_archive


def parse_args() -> argparse.Namespace:
    """Parse dataset-preparation arguments."""
    parser = argparse.ArgumentParser(description="Prepare traffic data for SpeedGuard Vision")
    parser.add_argument("--archive", help="Path to a local zip archive, e.g. /Users/.../archive (7).zip")
    parser.add_argument("--kagglehub", action="store_true", help="Download UA-DETRAC with kagglehub")
    parser.add_argument("--dataset", default="bratjay/ua-detrac-orig", help="KaggleHub dataset slug")
    parser.add_argument("--output", default="data/raw", help="Output directory for extracted data")
    parser.add_argument("--sample", default="data/sample_video.avi", help="Optional copied sample video path")
    return parser.parse_args()


def main() -> None:
    """Prepare a dataset and print discovered video files."""
    args = parse_args()
    if args.archive:
        root = extract_archive(args.archive, args.output)
    elif args.kagglehub:
        root = download_ua_detrac_with_kagglehub(args.dataset)
    else:
        raise SystemExit("Provide --archive or --kagglehub.")

    videos = discover_video_files(root)
    print(f"Dataset root: {root}")
    print(f"Discovered videos: {len(videos)}")
    for video in videos[:10]:
        print(f"  {video}")
    if videos:
        sample = copy_sample_video(root, args.sample)
        print(f"Sample video copied to: {sample}")


if __name__ == "__main__":
    main()
