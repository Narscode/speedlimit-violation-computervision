"""Dataset preparation helpers for local traffic archives and Kaggle downloads."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import List, Union


def extract_archive(archive_path: Union[str, Path], output_dir: Union[str, Path] = "data/raw") -> Path:
    """Extract a zip archive into `output_dir` and return the extraction path."""
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / archive_path.stem.replace(" ", "_").replace("(", "").replace(")", "")
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(target)
    return target


def discover_video_files(root: Union[str, Path]) -> List[Path]:
    """Find video files recursively under a dataset root."""
    root = Path(root)
    extensions = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
    return sorted([path for path in root.rglob("*") if path.suffix.lower() in extensions])


def copy_sample_video(root: Union[str, Path], output_path: Union[str, Path] = "data/sample_video.avi") -> Path:
    """Copy the first discovered video to a stable sample path for demos."""
    videos = discover_video_files(root)
    if not videos:
        raise FileNotFoundError(f"No video files found under {root}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(videos[0], output_path)
    return output_path


def download_ua_detrac_with_kagglehub(dataset: str = "bratjay/ua-detrac-orig") -> Path:
    """Download UA-DETRAC through kagglehub and return the local dataset path."""
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("Install kagglehub first: `pip install kagglehub`.") from exc
    return Path(kagglehub.dataset_download(dataset))
