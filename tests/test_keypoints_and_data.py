import pytest

from schaakgui.data import read_yolo_labels, scan_yolo_folder, yolo_kpt_shape
from schaakgui.keypoints import (BOARD_FLIP_IDX, BOARD_KPT_SHAPE, board_label_line, read_board_label,
                                 write_board_label, write_board_yaml)

CORNERS = ((100.0, 50.0), (300.0, 60.0), (320.0, 250.0), (90.0, 240.0))


def test_board_label_round_trip(tmp_path):
    line = board_label_line(CORNERS, 400, 300)
    values = line.split()
    assert values[0] == "0" and len(values) == 5 + 4 * 3
    write_board_label(tmp_path / "a.txt", CORNERS, 400, 300)
    corners = read_board_label(tmp_path / "a.txt", 400, 300)
    for got, want in zip(corners, CORNERS):
        assert got == pytest.approx(want, abs=0.01)


def test_board_yaml_and_kpt_shape(tmp_path):
    for split in ("train", "valid"):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "labels").mkdir(parents=True)
    path = write_board_yaml(tmp_path)
    text = path.read_text()
    assert "kpt_shape" in text and "flip_idx" in text and "\ntest:" not in text
    assert BOARD_KPT_SHAPE == [4, 3] and BOARD_FLIP_IDX == [1, 0, 3, 2]
    assert yolo_kpt_shape(tmp_path / "train") == [4, 3]


def test_read_yolo_labels_pose_lines_are_boxes(tmp_path):
    from PIL import Image

    (tmp_path / "train" / "images").mkdir(parents=True)
    Image.new("RGB", (400, 300)).save(tmp_path / "train" / "images" / "a.jpg")
    write_board_label(tmp_path / "train" / "labels" / "a.txt", CORNERS, 400, 300)
    write_board_yaml(tmp_path, splits=("train",))
    label = tmp_path / "train" / "labels" / "a.txt"
    boxes = read_yolo_labels(label, kpt_shape=[4, 3])
    assert len(boxes) == 1
    cls, box = boxes[0][0], boxes[0][1]
    assert cls == 0
    xc, yc, w, h = (float(v) for v in label.read_text().split()[1:5])
    assert box == pytest.approx((xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2), abs=1e-6)
    found = scan_yolo_folder(tmp_path / "train" / "images", tmp_path / "train" / "labels")
    assert {name: len(items) for name, items in found.items()} == {"board": 1}
    assert found["board"][0][1] == pytest.approx(box, abs=1e-6)
