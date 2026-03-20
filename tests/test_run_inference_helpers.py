import importlib
import sys
import types


def load_run_inference():
    sys.modules.pop("run_inference", None)
    sys.modules["torch"] = types.ModuleType("torch")
    return importlib.import_module("run_inference")


def test_load_class_colors_falls_back_to_defaults_when_missing(tmp_path):
    run_inference = load_run_inference()
    colors = run_inference.load_class_colors(tmp_path / "missing.json")
    assert colors == list(run_inference.DEFAULT_CLASS_COLORS)


def test_load_class_colors_ignores_invalid_entries(tmp_path):
    run_inference = load_run_inference()
    path = tmp_path / "class_colors.json"
    path.write_text("[[1, 2, 3], [4, 5]]", encoding="utf-8")
    assert run_inference.load_class_colors(path) == [(1, 2, 3)]


def test_class_color_wraps_palette():
    run_inference = load_run_inference()
    palette = [(1, 2, 3), (4, 5, 6)]
    assert run_inference.class_color(0, palette) == (1, 2, 3)
    assert run_inference.class_color(3, palette) == (4, 5, 6)
