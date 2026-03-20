# AGENTS.md

## Repository at a glance
This repository is a Windows-first desktop toolkit for a small YOLO object-detection workflow. The current implementation is organized as three top-level entry-point scripts:

- `main.py`: capture, mandatory labeling, dataset browsing/editing, class/color settings, and camera settings
- `train_model.py`: temporary YOLO dataset preparation plus Ultralytics CLI orchestration
- `run_inference.py`: live inference viewer with telemetry and screenshot capture

Shared state lives on disk in `captures/`, `classes.txt`, `class_colors.json`, and `config.json`.

## Current operating model
This repository is maintained using:

- design authority in `systemDesign.md`
- implementation notes in `docs/architecture-notes.md`
- temporary design/code mismatch tracking in `docs/deviations.md`
- test-backed validation
- repository-hygiene enforcement

## Working method
- Read `systemDesign.md` before changing structure or behavior.
- Preserve existing behavior unless a change is explicitly intended.
- Keep changes small and aligned with the current script-based design.
- Update tests and docs in the same change when behavior or repo rules move.
- Prefer harnesses, mocks, and temporary directories before refactoring production GUI code for testability.

## Comments and explainers
- Comment non-obvious constraints such as OpenCV/Qt interaction, Windows DLL ordering, and driver-specific camera behavior.
- Prefer repo-level documentation for architectural guidance over scattered explanatory comments.

## Testing expectations
- Use `pytest` for regression and hygiene checks.
- Cover pure helpers directly.
- Test orchestration through lightweight seams such as temporary directories, subprocess argument construction, and mocked dialogs.
- Use an offscreen Qt application for automated tests.
- Run the full suite from the repo root before finalizing work:
  `.\.venv312\Scripts\python.exe -m pytest`

## Documentation expectations
- Update `README.md` when onboarding, setup, usage, or supporting-doc navigation changes.
- Keep `README.md` ordered as: Title, badges, introduction, disclaimer, set-up, usage, contributing, supporting docs.
- Update `CONTRIBUTING.md` when contributor workflow or validation expectations change.
- `systemDesign.md` must describe the implementation that exists today.
- `docs/architecture-notes.md` should stay short and practical.
- `docs/deviations.md` should track only live mismatches between design and code.
- Do not document aspirational package boundaries or services that are not present in the repo.

## Validation and self-review
- Re-check docs against the current scripts and file layout.
- Confirm no accidental behavior drift was introduced.
- Run the full test suite.
- Keep the diff tight; do not mix unrelated cleanup into repo-governance changes.

## Repository hygiene
- Required governance files must exist.
- `README.md` and `CONTRIBUTING.md` should remain present and aligned with the current workflow.
- Test and cache artifacts should stay ignored.
- `docs/deviations.md` should either be explicitly empty or use the standard deviation template.
- Repo conventions should be enforced mechanically where practical instead of relying on memory.

## Source footer policy
- Canonical footer text: `source-code-footer.txt`
- Applies to: checked-in `.py` files in the repository root
- Does not apply to: `tests/`, docs, icons, captures, JSON/TXT data files, virtualenv content, generated build outputs, or lockfiles
- Implementation form: append the canonical footer text inside a trailing standalone triple-quoted string literal so Python syntax stays valid while the footer text itself remains unchanged
- Rule: use the canonical footer text exactly inside that wrapper and apply it consistently to every covered production source file
