"""Train or fine-tune a YOLOv8 vehicle detector for SpeedGuard Vision."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Parse YOLO training arguments."""
    parser = argparse.ArgumentParser(description="Train YOLOv8 for SpeedGuard vehicle detection")
    parser.add_argument("--data", required=True, help="YOLO data.yaml path")
    parser.add_argument("--weights", default="yolov8s.pt", help="Base YOLO weights")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--project", default="outputs/training", help="Ultralytics project directory")
    parser.add_argument("--name", default="speedguard_yolov8s", help="Run name")
    parser.add_argument("--device", default=None, help="CUDA device id, mps, or cpu")
    return parser.parse_args()


def main() -> None:
    """Run Ultralytics YOLO training."""
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install Ultralytics first: `pip install ultralytics`.") from exc
    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
    )


if __name__ == "__main__":
    main()
