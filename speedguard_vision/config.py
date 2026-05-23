"""Configuration loading and normalization for SpeedGuard Vision."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Union

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only in minimal environments
    yaml = None


DEFAULT_CONFIG: Dict[str, Any] = {
    "model": {
        "weights": "yolov8s.pt",
        "confidence": 0.5,
        "iou": 0.45,
        "classes": [2, 3, 5, 7],
    },
    "tracking": {
        "max_age": 30,
        "min_hits": 3,
        "max_cosine_distance": 0.4,
    },
    "speed": {
        "fps": 25,
        "line_a_y": 0.4,
        "line_b_y": 0.6,
        "real_world_distance": 10.0,
        "speed_limit": 60,
        "smoothing_window": 5,
    },
    "input": {
        "source": "data/MVI_40701/",
        "roi_points": [],
        "resize": [640, 640],
        "demo_mode": True,
        "demo_frames": 100,
    },
    "output": {
        "save_video": True,
        "save_csv": True,
        "show_live": False,
        "directory": "outputs",
        "video_path": "outputs/annotated_output.mp4",
        "violations_csv": "outputs/violations.csv",
        "analytics_dir": "outputs/analytics",
        "crops_dir": "outputs/violations",
    },
    "calibration": {
        "enabled": False,
        "config_path": "",
        "meters_per_pixel": None,
    },
    "lpr": {
        "enabled": True,
        "languages": ["en"],
        "cascade_path": "",
    },
    "evaluation": {
        "ground_truth": "",
    },
}


def deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge `patch` into `base` and return the updated dictionary."""
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Union[str, Path] = "config.yaml") -> Dict[str, Any]:
    """Load YAML or JSON configuration, falling back to default values."""
    config = deepcopy(DEFAULT_CONFIG)
    path = Path(path)
    if not path.exists():
        return config
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            loaded = json.load(handle)
        elif yaml is not None:
            loaded = yaml.safe_load(handle) or {}
        else:
            raise RuntimeError("PyYAML is required to read YAML configuration files.")
    return deep_update(config, loaded)


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Apply command-line arguments to the nested configuration dictionary."""
    if args.input:
        config["input"]["source"] = args.input
    if args.speed_limit is not None:
        config["speed"]["speed_limit"] = args.speed_limit
    if args.fps is not None:
        config["speed"]["fps"] = args.fps
    if args.calibration:
        config["calibration"]["enabled"] = True
        config["calibration"]["config_path"] = args.calibration
    if args.output:
        config["output"]["video_path"] = args.output
    if args.no_live:
        config["output"]["show_live"] = False
    if args.live:
        config["output"]["show_live"] = True
    if args.demo:
        config["input"]["demo_mode"] = True
    if args.full:
        config["input"]["demo_mode"] = False
    if args.weights:
        config["model"]["weights"] = args.weights
    return config


def ensure_output_dirs(config: Dict[str, Any]) -> None:
    """Create all configured output directories if they do not already exist."""
    for key in ("directory", "analytics_dir", "crops_dir"):
        Path(config["output"][key]).mkdir(parents=True, exist_ok=True)
