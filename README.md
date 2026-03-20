# OpenCV Capture, Label, Train

![Platform: Windows](https://img.shields.io/badge/platform-Windows%20first-0078D4)
![Python: 3.12](https://img.shields.io/badge/python-3.12-3776AB)
![Tests: pytest](https://img.shields.io/badge/tests-pytest-0A9EDC)

## Introduction
This repository is a Windows-first desktop toolkit for a small YOLO object-detection workflow. It provides three companion scripts for capture and mandatory labeling, temporary YOLO dataset preparation and training launch, and live inference with telemetry overlays.

## Disclaimer
The tools are intended for local desktop use, controlled imaging setups, and experiment-scale model work. Camera support, GPU detection, and runtime stability depend on the installed Windows drivers and Python packages, and the repo is not positioned as a production inspection system.

## Set-up
Use Python 3.12 and keep dependencies inside a dedicated `.venv312` environment so the GUI tools and native packages stay aligned.

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\activate
pip install --upgrade pip
pip install --only-binary=:all: "numpy<2.3.0" "opencv-python<4.13" pyqt5 ultralytics torch psutil pynvml
```

The repo stores shared state on disk in `captures/`, `classes.txt`, `class_colors.json`, and `config.json`. Keep those files alongside the three top-level scripts when running locally.

## Usage
Run the primary capture and labeling app:

```powershell
.\.venv312\Scripts\python.exe .\main.py --camera 0
```

Launch the training companion to build a temporary `.yolo_training_cache/` dataset and start the Ultralytics CLI:

```powershell
.\.venv312\Scripts\python.exe .\train_model.py
```

Launch the inference viewer for live model checks and screenshot capture:

```powershell
.\.venv312\Scripts\python.exe .\run_inference.py
```

## Contributing
See [CONTRIBUTING.md](./CONTRIBUTING.md) for local workflow, validation expectations, and review readiness.

## Supporting docs
- [CONTRIBUTING.md](./CONTRIBUTING.md)
- [AGENTS.md](./AGENTS.md)
- [systemDesign.md](./systemDesign.md)
- [docs/architecture-notes.md](./docs/architecture-notes.md)
- [docs/deviations.md](./docs/deviations.md)
