# SpeedGuard Vision

SpeedGuard Vision is a production-oriented computer vision pipeline for real-time speed-limit violation detection. It detects vehicles with YOLOv8, tracks them with Deep SORT, estimates speed from calibrated virtual line crossings, logs violations, optionally runs license plate OCR, and generates analytics plots plus a PDF report.

## Features

- Accepts a video file (`.mp4`, `.avi`, `.mov`, `.mkv`) or a folder of sequential image frames.
- YOLOv8 vehicle detection for COCO vehicle classes: car, motorcycle, bus, truck.
- Deep SORT tracking with appearance embeddings when `deep-sort-realtime` is installed.
- Deterministic IoU tracker fallback so smoke tests still run in minimal environments.
- Perspective calibration helpers with homography save/load and bird's-eye validation.
- Speed estimation between Line A and Line B with moving-average smoothing and outlier rejection.
- Violation logging to CSV with vehicle crop evidence and optional EasyOCR LPR.
- Annotated output video, live display option, progress bar, FPS counter, and analytics report.
- Notebook workflow for dataset EDA and YOLO training experiments.

## Project Structure

```text
speedguard_vision/
├── pipeline.py
├── preprocessing.py
├── calibration.py
├── detector.py
├── tracker.py
├── speed_estimator.py
├── violation_detector.py
├── lpr.py
├── annotator.py
├── analytics.py
├── evaluation.py
├── dataset_utils.py
└── types.py
notebooks/
└── EDA_Training_SpeedGuard.ipynb
scripts/
├── prepare_dataset.py
├── calibrate_from_frame.py
├── convert_ua_detrac_to_yolo.py
└── train_yolo.py
config.yaml
requirements.txt
README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch CUDA support depends on your machine and CUDA version. If GPU wheels are needed, install PyTorch from the official selector before running `pip install -r requirements.txt`.

## Prepare Datasets

For the provided archive:

```bash
python scripts/prepare_dataset.py --archive "/Users/nareswari/Downloads/archive (7).zip"
```

This extracts the archive into `data/raw/`, discovers traffic videos, and copies one sample clip to `data/sample_video.avi`.

For UA-DETRAC with KaggleHub:

```bash
export KAGGLE_API_TOKEN="your_token_here"
python scripts/prepare_dataset.py --kagglehub --dataset bratjay/ua-detrac-orig
```

Do not commit Kaggle tokens to the repo. Keep them in your shell environment.

## Run A Quick Demo

```bash
python -m speedguard_vision.pipeline \
  --input data/sample_video.avi \
  --speed_limit 60 \
  --fps 25 \
  --demo \
  --output outputs/annotated_output.mp4
```

Run the full video:

```bash
python -m speedguard_vision.pipeline \
  --input data/sample_video.avi \
  --speed_limit 60 \
  --fps 25 \
  --full
```

Run on a folder of sequential frames:

```bash
python -m speedguard_vision.pipeline --input data/MVI_40701/ --full
```

Outputs are written to:

- `outputs/annotated_output.mp4`
- `outputs/violations.csv`
- `outputs/violations/` vehicle crops
- `outputs/analytics/*.png`
- `outputs/analytics/analytics_report.pdf`

## Calibration

Speed accuracy depends on calibration. The simple default assumes Line A and Line B are `speed.real_world_distance` meters apart. For better accuracy, create a perspective calibration:

```bash
python scripts/calibrate_from_frame.py \
  --input data/sample_video.avi \
  --output outputs/calibration.json \
  --width_m 3.5 \
  --length_m 10.0
```

Click four road points in clockwise order. A JSON calibration and preview image are saved in `outputs/`.

Then run:

```bash
python -m speedguard_vision.pipeline \
  --input data/sample_video.avi \
  --calibration outputs/calibration.json \
  --full
```

## Notebook Workflow

Open:

```bash
jupyter notebook notebooks/EDA_Training_SpeedGuard.ipynb
```

The notebook includes:

- archive extraction,
- frame EDA,
- sample frame visualization,
- optional YOLOv8 fine-tuning scaffold,
- pipeline execution from Python,
- violation CSV inspection.

For supervised YOLO training, convert UA-DETRAC annotations to YOLO format and point the notebook training cell to `data/ua_detrac_yolo/data.yaml`.

Terminal conversion and training example:

```bash
python scripts/convert_ua_detrac_to_yolo.py \
  --frames data/ua_detrac/MVI_40701 \
  --xml data/ua_detrac/DETRAC-Train-Annotations-XML/MVI_40701.xml \
  --output data/ua_detrac_yolo

python scripts/train_yolo.py \
  --data data/ua_detrac_yolo/data.yaml \
  --weights yolov8s.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 16
```

## Metrics

The pipeline prints a JSON summary with:

- detection precision, recall, mAP@0.5 when UA-DETRAC XML ground truth is configured,
- tracking MOTA and ID-switch placeholders unless tracking ground truth is supplied,
- speed MAE/RMSE placeholders unless speed ground truth is supplied,
- violation accuracy/FPR placeholders unless violation labels are supplied,
- descriptive speed distribution statistics when ground truth is unavailable.

Configure UA-DETRAC XML ground truth in `config.yaml`:

```yaml
evaluation:
  ground_truth: "data/annotations/MVI_40701.xml"
```

## Notes On Accuracy

The largest real-world error source is camera calibration, not YOLO. For a credible report, include:

- a screenshot of Line A and Line B,
- the measured real-world line distance,
- calibration preview,
- FPS used for timestamping,
- speed distribution and violation CSV,
- examples of safe, near-limit, and violating vehicle annotations.
