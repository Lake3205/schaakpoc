import yaml

from schaakgui.data import dataset_yaml, read_yolo_labels, scan_folder
from schaakgui.yaml_sync import (
    apply_mapping,
    collect_tags,
    suggest_mapping,
    unified_names,
    yaml_label_dirs,
)


def _two_datasets(tmp_path):
    from conftest import make_yolo_split

    tmp_path = tmp_path / "yolo"

    # Dataset A: 0 = W-R, 1 = black-king (data.yaml in the dataset root, shared by train + valid).
    a_train = make_yolo_split(tmp_path / "a" / "train", names=("W-R", "black-king"))
    a_valid = make_yolo_split(tmp_path / "a" / "valid", names=("W-R", "black-king"))
    # Dataset B: 0 = Black_King, 1 = White-Rook (data.yaml inside the split folder).
    b_train = make_yolo_split(tmp_path / "b" / "train", names=("Black_King", "White-Rook"),
                              yaml_in_parent=False)
    return a_train, a_valid, b_train


def test_dataset_yaml_detection(tmp_path, datasets):
    a_train, a_valid, b_train = _two_datasets(tmp_path)
    tmp_path = tmp_path / "yolo"
    assert dataset_yaml(a_train) == (tmp_path / "a" / "data.yaml").resolve()
    assert dataset_yaml(a_train / "images") == dataset_yaml(a_valid)
    assert dataset_yaml(b_train) == (b_train / "data.yaml").resolve()
    assert dataset_yaml(datasets["a"]) is None  # ImageFolder without yaml
    assert dataset_yaml(tmp_path / "missing") is None
    assert yaml_label_dirs(tmp_path / "a" / "data.yaml") == [a_train / "labels", a_valid / "labels"]


def test_collect_and_suggest_tags(tmp_path):
    _two_datasets(tmp_path)
    tmp_path = tmp_path / "yolo"
    paths = [tmp_path / "a" / "data.yaml", tmp_path / "b" / "train" / "data.yaml"]
    tags = collect_tags(paths + paths)  # duplicates are ignored
    assert list(tags) == ["W-R", "black-king", "Black_King", "White-Rook"]
    assert tags["W-R"] == [paths[0].resolve()]
    mapping = suggest_mapping(tags)
    assert mapping["Black_King"] == "black-king" and mapping["W-R"] == "W-R"
    assert unified_names(paths, mapping) == ["W-R", "black-king", "White-Rook"]


def test_apply_mapping_rewrites_yaml_and_labels(tmp_path):
    a_train, a_valid, b_train = _two_datasets(tmp_path)
    paths = [a_train.parent / "data.yaml", b_train / "data.yaml"]
    (b_train / "labels" / "board0.txt").write_text(
        "1 0.25 0.5 0.5 1.0\n\n0 0.75 0.5 0.5 1.0\n2 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    mapping = {"W-R": "white-rook", "White-Rook": "white-rook", "Black_King": "black-king"}
    report = apply_mapping(paths, mapping)

    assert report["names"] == ["white-rook", "black-king", "class_2"]
    for path in paths:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert data["names"] == report["names"] and data["nc"] == 3
    assert yaml.safe_load(paths[0].read_text(encoding="utf-8"))["train"] == "train/images"
    # Dataset A already used the unified order: untouched. Dataset B: 0 <-> 1 swapped.
    assert [c for c, _ in read_yolo_labels(a_valid / "labels" / "board0.txt")] == [0, 1]
    assert [c for c, _ in read_yolo_labels(b_train / "labels" / "board1.txt")] == [1, 0]
    assert (b_train / "labels" / "board0.txt").read_text(encoding="utf-8") == \
        "0 0.25 0.5 0.5 1.0\n\n1 0.75 0.5 0.5 1.0\n2 0.5 0.5 0.2 0.2\n"
    assert report["label_files"] == 3 and report["boxes"] == 6
    # Both datasets now produce the same class names for the same pieces.
    assert set(scan_folder(a_train)) == {"white-rook", "black-king"}
    assert set(scan_folder(b_train)) == {"white-rook", "black-king", "class_2"}
    assert len(scan_folder(b_train)["white-rook"]) == 3
