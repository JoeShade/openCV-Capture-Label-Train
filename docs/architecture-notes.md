# Architecture Notes

## Current boundaries worth preserving
- `main.py` owns capture-time mutation of raw capture files, annotation, and dataset inspection/editing.
- `train_model.py` owns temporary training-cache generation and subprocess orchestration; it should read the dataset rather than mutate the raw capture set.
- `run_inference.py` owns model loading, runtime inference, telemetry, and inference screenshot capture.
- The filesystem is the integration seam between the three tools.

## Practical implementation notes
- `annotate_image()` runs an OpenCV interaction loop inside the Qt application lifecycle. Treat it as an integration-heavy path and avoid casual rewrites.
- `run_inference.py` intentionally preloads Torch before importing OpenCV and also adjusts the Windows DLL search path. That order is part of the current operational behavior.
- `train_model.py` deletes and rebuilds `.yolo_training_cache/` on each preparation pass.
- Camera property support is driver-dependent; the camera settings dialog probes support before enabling controls.
- Small helper duplication across top-level scripts is part of the current design. Do not consolidate it unless the change is deliberate and test-backed.

## Testability notes
- Test pure helpers directly with temporary files and deterministic inputs.
- Use temporary directories for label round-trips and training-cache preparation.
- For Qt-driven tests, create one offscreen `QApplication` and avoid real GUI interaction where possible.
- Mock message boxes, GPU discovery, and external executables rather than requiring local hardware in CI-style runs.
