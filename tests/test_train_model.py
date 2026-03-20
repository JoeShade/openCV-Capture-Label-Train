from pathlib import Path

import train_model


def test_human_eta_formats_minutes_and_hours():
    assert train_model.human_eta(0) == "--:--"
    assert train_model.human_eta(61) == "01m 01s"
    assert train_model.human_eta(3661) == "1h 01m"


def test_read_classes_filters_blank_lines(tmp_path):
    path = tmp_path / "classes.txt"
    path.write_text("dent\n\nscratch\n", encoding="utf-8")
    assert train_model.read_classes(path) == ["dent", "scratch"]


def test_prepare_dataset_builds_cache_from_labeled_images(tmp_path, monkeypatch, qt_app):
    monkeypatch.setattr(train_model, "detect_gpus", lambda: [])

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    for idx in range(5):
        image = dataset_dir / f"capture_{idx}.jpg"
        image.write_bytes(b"image")
        image.with_suffix(".txt").write_text("0 0.5 0.5 0.2 0.2", encoding="utf-8")
    (dataset_dir / "unlabeled.jpg").write_bytes(b"image")

    classes_path = tmp_path / "classes.txt"
    classes_path.write_text("dent\nscratch\n", encoding="utf-8")

    window = train_model.TrainingWindow()
    try:
        data_yaml = window.prepare_dataset(dataset_dir, classes_path)
    finally:
        window.close()

    assert data_yaml == dataset_dir / ".yolo_training_cache" / "data.yaml"
    assert data_yaml is not None

    cache_dir = data_yaml.parent
    train_images = sorted((cache_dir / "images" / "train").iterdir())
    val_images = sorted((cache_dir / "images" / "val").iterdir())
    train_labels = sorted((cache_dir / "labels" / "train").iterdir())
    val_labels = sorted((cache_dir / "labels" / "val").iterdir())

    assert len(train_images) == 4
    assert len(val_images) == 1
    assert len(train_labels) == 4
    assert len(val_labels) == 1
    assert all(path.suffix == ".jpg" for path in train_images + val_images)
    assert all(path.suffix == ".txt" for path in train_labels + val_labels)
    assert "unlabeled.jpg" not in {path.name for path in train_images + val_images}

    yaml_text = data_yaml.read_text(encoding="utf-8")
    assert f"path: {cache_dir}" in yaml_text
    assert "train: images/train" in yaml_text
    assert "val: images/val" in yaml_text
    assert "0: dent" in yaml_text
    assert "1: scratch" in yaml_text


def test_prepare_dataset_warns_when_no_labeled_images(tmp_path, monkeypatch, qt_app):
    monkeypatch.setattr(train_model, "detect_gpus", lambda: [])
    warnings = []
    monkeypatch.setattr(
        train_model.QtWidgets.QMessageBox,
        "warning",
        lambda *args: warnings.append(args),
    )

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "capture.jpg").write_bytes(b"image")
    classes_path = tmp_path / "classes.txt"
    classes_path.write_text("dent\n", encoding="utf-8")

    window = train_model.TrainingWindow()
    try:
        result = window.prepare_dataset(dataset_dir, classes_path)
    finally:
        window.close()

    assert result is None
    assert warnings
    assert warnings[0][1] == "No data"
