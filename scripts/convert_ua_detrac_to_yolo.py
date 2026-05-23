"""Convert UA-DETRAC XML annotations and frames into YOLO detection format."""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

CLASS_MAP = {
    "car": 0,
    "van": 0,
    "others": 0,
    "motorcycle": 1,
    "bus": 2,
    "truck": 3,
}
CLASS_NAMES = ["car", "motorcycle", "bus", "truck"]


def parse_args() -> argparse.Namespace:
    """Parse conversion arguments."""
    parser = argparse.ArgumentParser(description="Convert UA-DETRAC XML to YOLO format")
    parser.add_argument("--frames", required=True, help="Folder containing sequence images")
    parser.add_argument("--xml", required=True, help="UA-DETRAC XML annotation file")
    parser.add_argument("--output", default="data/ua_detrac_yolo", help="YOLO dataset output directory")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random split seed")
    return parser.parse_args()


def find_frame_image(frames_dir: Path, frame_num: int) -> Path:
    """Find a frame image by common UA-DETRAC naming conventions."""
    candidates = [
        frames_dir / f"img{frame_num:05d}.jpg",
        frames_dir / f"img{frame_num:06d}.jpg",
        frames_dir / f"{frame_num:05d}.jpg",
        frames_dir / f"{frame_num:06d}.jpg",
        frames_dir / f"frame_{frame_num:06d}.jpg",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(frames_dir.glob(f"*{frame_num:05d}*.jpg"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No image found for frame {frame_num} in {frames_dir}")


def image_size(path: Path) -> Tuple[int, int]:
    """Read image dimensions with OpenCV."""
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    h, w = image.shape[:2]
    return w, h


def parse_annotations(xml_path: Path) -> Dict[int, List[Tuple[int, float, float, float, float]]]:
    """Parse UA-DETRAC XML into frame-indexed YOLO rows before normalization."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    rows: Dict[int, List[Tuple[int, float, float, float, float]]] = {}
    for frame in root.iter("frame"):
        frame_num = int(frame.attrib.get("num", "0"))
        frame_rows = []
        for target in frame.iter("target"):
            box = target.find("box")
            attribute = target.find("attribute")
            if box is None:
                continue
            vehicle_type = "car"
            if attribute is not None:
                vehicle_type = attribute.attrib.get("vehicle_type", "car").lower()
            class_id = CLASS_MAP.get(vehicle_type, 0)
            left = float(box.attrib.get("left", 0))
            top = float(box.attrib.get("top", 0))
            width = float(box.attrib.get("width", 0))
            height = float(box.attrib.get("height", 0))
            frame_rows.append((class_id, left, top, width, height))
        rows[frame_num] = frame_rows
    return rows


def write_data_yaml(output_dir: Path) -> None:
    """Write YOLO `data.yaml` for Ultralytics training."""
    data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {idx: name for idx, name in enumerate(CLASS_NAMES)},
    }
    if yaml is not None:
        with (output_dir / "data.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, sort_keys=False)
    else:
        lines = [f"path: {data['path']}", "train: images/train", "val: images/val", "names:"]
        lines.extend(f"  {idx}: {name}" for idx, name in enumerate(CLASS_NAMES))
        (output_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def convert(frames_dir: Path, xml_path: Path, output_dir: Path, val_split: float, seed: int) -> None:
    """Convert one UA-DETRAC sequence into YOLO images and labels."""
    annotations = parse_annotations(xml_path)
    frame_nums = sorted(annotations.keys())
    random.Random(seed).shuffle(frame_nums)
    val_count = int(len(frame_nums) * val_split)
    val_set = set(frame_nums[:val_count])
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    for frame_num in frame_nums:
        source = find_frame_image(frames_dir, frame_num)
        width, height = image_size(source)
        split = "val" if frame_num in val_set else "train"
        dest_image = output_dir / "images" / split / source.name
        dest_label = output_dir / "labels" / split / f"{source.stem}.txt"
        shutil.copy2(source, dest_image)
        label_lines = []
        for class_id, left, top, box_w, box_h in annotations[frame_num]:
            cx = (left + box_w / 2) / width
            cy = (top + box_h / 2) / height
            norm_w = box_w / width
            norm_h = box_h / height
            label_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {norm_w:.6f} {norm_h:.6f}")
        dest_label.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
    write_data_yaml(output_dir)
    print(f"Converted {len(frame_nums)} frames into {output_dir}")
    print(f"Train: {len(frame_nums) - val_count}, Val: {val_count}")
    print(f"YOLO data file: {output_dir / 'data.yaml'}")


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    convert(Path(args.frames), Path(args.xml), Path(args.output), args.val_split, args.seed)


if __name__ == "__main__":
    main()
