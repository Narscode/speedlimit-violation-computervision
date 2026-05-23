"""Evaluation metrics for detection, tracking, speed, and violations."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np

from speedguard_vision.tracker import iou

BBox = Tuple[float, float, float, float]


def parse_ua_detrac_xml(xml_path: Union[str, Path]) -> Dict[int, List[BBox]]:
    """Parse a UA-DETRAC XML annotation file into frame-indexed bounding boxes."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    frames: Dict[int, List[BBox]] = defaultdict(list)
    for frame in root.iter("frame"):
        frame_num = int(frame.attrib.get("num", "0"))
        for target in frame.iter("target"):
            box = target.find("box")
            if box is None:
                continue
            left = float(box.attrib.get("left", 0))
            top = float(box.attrib.get("top", 0))
            width = float(box.attrib.get("width", 0))
            height = float(box.attrib.get("height", 0))
            frames[frame_num].append((left, top, left + width, top + height))
    return dict(frames)


def detection_metrics(
    predictions: Dict[int, List[BBox]],
    ground_truth: Dict[int, List[BBox]],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute detection precision, recall, and a practical AP@0.5 approximation."""
    tp = fp = fn = 0
    for frame_id, gt_boxes in ground_truth.items():
        pred_boxes = predictions.get(frame_id, [])
        matched = set()
        for pred in pred_boxes:
            best_idx, best_iou = -1, 0.0
            for idx, gt in enumerate(gt_boxes):
                if idx in matched:
                    continue
                score = iou(pred, gt)
                if score > best_iou:
                    best_idx, best_iou = idx, score
            if best_idx >= 0 and best_iou >= iou_threshold:
                tp += 1
                matched.add(best_idx)
            else:
                fp += 1
        fn += len(gt_boxes) - len(matched)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    ap50 = precision * recall
    return {"precision": precision, "recall": recall, "mAP@0.5": ap50, "tp": float(tp), "fp": float(fp), "fn": float(fn)}


def tracking_metrics(id_switches: int = 0, misses: int = 0, false_positives: int = 0, ground_truth_count: int = 0) -> Dict[str, float]:
    """Compute MOTA and ID-switch count from aggregate tracking errors."""
    mota = 1.0 - (misses + false_positives + id_switches) / ground_truth_count if ground_truth_count else 0.0
    return {"MOTA": float(mota), "ID_switches": float(id_switches)}


def speed_metrics(predicted: Sequence[float], actual: Sequence[float]) -> Dict[str, float]:
    """Compute MAE and RMSE for speed estimation."""
    if not predicted or not actual:
        return {"MAE": 0.0, "RMSE": 0.0}
    length = min(len(predicted), len(actual))
    pred = np.array(predicted[:length], dtype=float)
    truth = np.array(actual[:length], dtype=float)
    diff = pred - truth
    return {"MAE": float(np.mean(np.abs(diff))), "RMSE": float(np.sqrt(np.mean(diff**2)))}


def violation_metrics(predicted: Sequence[bool], actual: Sequence[bool]) -> Dict[str, float]:
    """Compute violation classification accuracy and false-positive rate."""
    if not predicted or not actual:
        return {"accuracy": 0.0, "false_positive_rate": 0.0}
    length = min(len(predicted), len(actual))
    pred = np.array(predicted[:length], dtype=bool)
    truth = np.array(actual[:length], dtype=bool)
    accuracy = float(np.mean(pred == truth))
    fp = float(np.sum((pred == True) & (truth == False)))
    tn = float(np.sum((pred == False) & (truth == False)))
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"accuracy": accuracy, "false_positive_rate": float(fpr)}


def empty_metric_report() -> Dict[str, Dict[str, float]]:
    """Return a complete metric report with zeroed values for missing ground truth."""
    return {
        "detection": {"precision": 0.0, "recall": 0.0, "mAP@0.5": 0.0},
        "tracking": {"MOTA": 0.0, "ID_switches": 0.0},
        "speed": {"MAE": 0.0, "RMSE": 0.0},
        "violation": {"accuracy": 0.0, "false_positive_rate": 0.0},
    }
