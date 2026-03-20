# Contributing

## Getting started
Use Python 3.12 and work from the repository root. The expected local environment is `.venv312`.

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\activate
pip install --upgrade pip
pip install --only-binary=:all: "numpy<2.3.0" "opencv-python<4.13" pyqt5 ultralytics torch psutil pynvml
```

Read [systemDesign.md](./systemDesign.md) before changing structure or behavior. Use [docs/architecture-notes.md](./docs/architecture-notes.md) for the short practical constraints and [docs/deviations.md](./docs/deviations.md) for any live design/code mismatches.

## How to propose changes
Work in focused changes that match the current script-based design unless a broader structural change is deliberate and test-backed. Explain whether your change preserves behavior or intentionally changes it, and keep unrelated cleanup out of the same diff.

## Expectations for changes
- Keep changes aligned with `systemDesign.md`.
- Preserve existing behavior unless a change is explicitly intended.
- Update tests for behavior changes and regression risks.
- Update docs in the same change when setup, usage, architecture notes, or repo rules move.
- Prefer temporary directories, mocks, subprocess seams, and the offscreen Qt test app before refactoring GUI-heavy production code for testability.
- Keep the canonical footer policy intact for checked-in root `.py` files.

## Validation checklist
- [ ] `.\.venv312\Scripts\python.exe -m pytest` passes from the repo root
- [ ] Docs are updated where relevant
- [ ] `README.md` remains ordered as Title, badges, introduction, disclaimer, set-up, usage, contributing, supporting docs
- [ ] `docs/deviations.md` is updated only if a live temporary mismatch exists
- [ ] Repository hygiene checks still pass

## Supporting design and architecture docs
- [systemDesign.md](./systemDesign.md)
- [AGENTS.md](./AGENTS.md)
- [docs/architecture-notes.md](./docs/architecture-notes.md)
- [docs/deviations.md](./docs/deviations.md)
