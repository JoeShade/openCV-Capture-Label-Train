# System Design

## Purpose
This repository provides a small desktop workflow for collecting images, annotating defects in YOLO format, launching YOLOv8 training, and running live model inference against a camera feed.

## Scope
- Live camera preview and capture in `main.py`
- Mandatory post-capture labeling, null marking, and capture discard behavior in `main.py`
- Class-name and class-color management through flat files
- Dataset browsing/edit/delete flows in `main.py`
- Temporary YOLO dataset preparation and training-process launch in `train_model.py`
- Live model inference, telemetry overlay, and screenshot capture in `run_inference.py`

## Non-goals
- Headless services or APIs
- Multi-user coordination
- Database-backed storage
- Automated pre-labeling or proposal models
- Production-grade inspection throughput or hardened deployment behavior

## Current architecture
The repo is a script-oriented desktop application rather than a packaged Python library. Each top-level script owns a user-facing workflow:

- `main.py` is the primary integration script and contains the largest amount of behavior: capture, annotation, inspect/edit flows, camera settings, and supporting persistence helpers.
- `train_model.py` is a companion GUI that prepares a disposable `.yolo_training_cache/` tree and launches the Ultralytics CLI in a subprocess.
- `run_inference.py` is a companion GUI that loads a YOLO model, reads camera frames on a timer, and renders detections plus local telemetry.

The tools are coupled through on-disk conventions instead of imports between scripts. Shared concepts such as classes, colors, capture folders, and splash/icon handling are duplicated in small amounts across scripts rather than centralized into a common package.

## Module/file responsibilities
- `main.py`: capture window, OpenCV annotation loop, YOLO label read/write helpers, class/color editing, inspect-mode navigation, and camera property dialog
- `train_model.py`: classes-file reader, GPU detection, training-cache builder, Ultralytics CLI resolution, and training log/progress UI
- `run_inference.py`: Torch preload, delayed OpenCV import, model loading, inference overlay rendering, telemetry collection, and inference screenshot capture
- `classes.txt`: ordered class list used across labeling, training, and inference
- `class_colors.json`: optional RGB palette aligned with class indices
- `config.json`: preview-timer and annotation-loop timing overrides
- `captures/`: image store for captured images plus `null/` and `inference/` subfolders

## Core flows
### Flow 1
Capture and label:

1. `CameraWindow` opens a selected camera and shows the live feed.
2. A capture writes a timestamped image under `captures/`.
3. `annotate_image()` opens an OpenCV labeling window inside the running Qt application.
4. Saving writes an adjacent YOLO label file, null-marking moves the image to `captures/null/`, and cancel removes the captured image so the main capture flow does not intentionally leave new unlabeled files behind.

### Flow 2
Training launch:

1. The training window accepts a dataset folder plus `classes.txt`.
2. `prepare_dataset()` filters to images that already have adjacent `.txt` labels.
3. The method rebuilds `.yolo_training_cache/`, distributes labeled items into train/validation folders with a deterministic shuffle, and writes `data.yaml`.
4. A `QProcess` starts `yolo detect train ...` and streams merged logs back into the GUI while parsing simple `current/total` tokens for progress and ETA.

### Flow 3
Live inference:

1. The inference window preloads Torch before importing OpenCV to reduce Windows DLL/OpenMP conflicts.
2. The user chooses a camera and a `.pt` model.
3. A timer reads frames, runs YOLO inference when a model is loaded, draws detections, and overlays telemetry.
4. Optional screenshots are written to `captures/inference/`.

## State/data/contracts
- YOLO labels are stored next to images using `<image-stem>.txt`.
- Each valid label line is `class_id x_center y_center width height` with normalized coordinates.
- `classes.txt` is one class per non-empty line.
- `class_colors.json` is a JSON array of RGB triples; invalid or missing data falls back to built-in defaults.
- `config.json` is optional and only overrides `timer_interval_ms` and `annotate_wait_key_ms` within bounded ranges.
- `.yolo_training_cache/` is disposable output and is recreated by the training tool.

## Error and warning model
- File-based helper functions usually fall back to defaults instead of raising user-visible exceptions.
- GUI problems are surfaced through status labels, message boxes, and the training log view.
- Camera and GPU support are best-effort and depend on local drivers/runtime availability.
- The inference app fails early on missing Torch/Ultralytics imports and shows a blocking error dialog.
- The training app logs a clear message if the `yolo` CLI cannot be resolved or started.

## External dependencies
- Windows desktop environment with camera access
- Python runtime compatible with the current scripts; the README currently recommends Python 3.12
- `opencv-python`
- `PyQt5`
- `ultralytics`
- `torch`
- `psutil`
- optional `pynvml`
- optional `nvidia-smi`
- optional `pyinstaller`

## Testing boundaries
- Unit tests should cover deterministic helpers such as config parsing, class/color loading, label read/write helpers, and ETA formatting.
- Implementation tests should cover dataset-cache creation and warning behavior without requiring real cameras or GPUs.
- Hygiene tests should enforce required repo files, deviations-file rules, ignore rules, and footer policy.

## Repository invariants
- Captures created through `main.py` should end as labeled data, a null capture under `captures/null/`, or be deleted when labeling is cancelled.
- The training workflow only uses images that already have adjacent label files.
- Shared state between tools lives on disk, not in a service or database.
- The three top-level scripts are the current integration boundaries; cross-tool coupling happens through files and conventions rather than shared modules.
