import json

import main


def test_load_config_returns_defaults_for_missing_file(tmp_path):
    config = main.load_config(tmp_path / "missing.json")
    assert config == main.DEFAULT_CONFIG


def test_load_config_applies_valid_int_overrides_only(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "timer_interval_ms": 12,
                "annotate_wait_key_ms": True,
                "ignored": 99,
                "too_large": 9999,
            }
        ),
        encoding="utf-8",
    )

    config = main.load_config(path)

    assert config["timer_interval_ms"] == 12
    assert config["annotate_wait_key_ms"] == main.DEFAULT_ANNOTATE_WAIT_KEY_MS
    assert set(config) == set(main.DEFAULT_CONFIG)


def test_load_classes_falls_back_to_defect_for_missing_or_empty_file(tmp_path):
    assert main.load_classes(tmp_path / "missing.txt") == ["defect"]

    empty = tmp_path / "classes.txt"
    empty.write_text("\n\n", encoding="utf-8")
    assert main.load_classes(empty) == ["defect"]


def test_save_classes_round_trip(tmp_path):
    path = tmp_path / "classes.txt"
    main.save_classes(["dent", "scratch"], path)
    assert main.load_classes(path) == ["dent", "scratch"]


def test_class_color_wraps_and_defaults():
    assert main.class_color(4, []) == (0, 255, 0)
    assert main.class_color(3, [(1, 2, 3), (4, 5, 6)]) == (4, 5, 6)


def test_save_and_load_class_colors_round_trip(tmp_path):
    path = tmp_path / "class_colors.json"
    colors = [(10, 20, 30), (40, 50, 60)]
    main.save_class_colors(colors, path)
    assert main.load_class_colors(path) == colors


def test_load_class_colors_falls_back_on_invalid_json(tmp_path):
    path = tmp_path / "class_colors.json"
    path.write_text("{not-json}", encoding="utf-8")
    assert main.load_class_colors(path) == list(main.DEFAULT_CLASS_COLORS)


def test_save_and_load_labels_yolo_round_trip(tmp_path):
    image_path = tmp_path / "capture.jpg"
    boxes = [
        [10, 20, 110, 70, 2],
        [150, 10, 199, 50, 1],
    ]
    image_shape = (100, 200, 3)

    label_path = main.save_labels_yolo(image_path, boxes, image_shape)

    loaded = main.load_labels_yolo(label_path, image_shape)
    assert loaded == boxes


def test_save_labels_yolo_clips_and_drops_invalid_boxes(tmp_path):
    image_path = tmp_path / "capture.jpg"
    boxes = [
        [-10, 10, 20, 30, 0],
        [50, 50, 50, 80, 1],
        [70, 90, 90, 91, 2],
    ]

    label_path = main.save_labels_yolo(image_path, boxes, (100, 100, 3))

    lines = label_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_load_labels_yolo_ignores_malformed_lines(tmp_path):
    label_path = tmp_path / "capture.txt"
    label_path.write_text(
        "\n".join(
            [
                "bad line",
                "0 0.5 0.5 0.2 0.2",
                "1 0.1 0.1 0.0 0.1",
            ]
        ),
        encoding="utf-8",
    )

    loaded = main.load_labels_yolo(label_path, (100, 100, 3))
    assert loaded == [[40, 40, 60, 60, 0]]
