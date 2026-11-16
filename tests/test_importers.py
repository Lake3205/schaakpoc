import json

import pytest
from PIL import Image

from schaakgui.data import yolo_class_names, yolo_kpt_shape
from schaakgui.importers import chessred_test_cases, import_chessred2k
from schaakgui.keypoints import read_board_label
from schaakgui.position import PIECE_CLASSES


def make_chessred(root):
    (root / "images" / "0").mkdir(parents=True)
    images = []
    for i in range(4):
        Image.new("RGB", (200, 100), (90, 90, 90)).save(root / "images" / "0" / f"G000_IMG00{i}.jpg")
        images.append({"id": i, "file_name": f"G000_IMG00{i}.jpg", "path": f"images/0/G000_IMG00{i}.jpg",
                       "width": 200, "height": 100, "game_id": 0})
    pieces = []
    for i in range(3):  # image 3 has no boxes (not in the 2K subset)
        pieces.append({"id": 2 * i, "image_id": i, "category_id": 5, "chessboard_position": "e1",
                       "bbox": [20, 10, 30, 40]})
        pieces.append({"id": 2 * i + 1, "image_id": i, "category_id": 11, "chessboard_position": "e8",
                       "bbox": [150, 40, 40, 70]})  # clipped at the bottom
    corners = [{"image_id": i, "corners": {"bottom_left": [10, 90], "top_left": [20, 10],
                                           "top_right": [180, 12], "bottom_right": [190, 95]}}
               for i in range(4)]
    data = {
        "info": {}, "images": images,
        "categories": [{"id": k, "name": n} for k, n in enumerate(
            ["white-pawn", "white-rook", "white-knight", "white-bishop", "white-queen", "white-king",
             "black-pawn", "black-rook", "black-knight", "black-bishop", "black-queen", "black-king",
             "empty"])],
        "annotations": {"pieces": pieces, "corners": corners},
        "splits": {"train": {"image_ids": [0, 1, 2, 3]},
                   "chessred2k": {"train": {"image_ids": [0]}, "val": {"image_ids": [1]},
                                  "test": {"image_ids": [2]}}},
    }
    path = root / "annotations.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_import_chessred2k(tmp_path):
    ann = make_chessred(tmp_path / "chessred")
    result = import_chessred2k(ann, tmp_path / "chessred", tmp_path / "out")
    assert result["counts"] == {"train": 1, "valid": 1, "test": 1}
    assert result["skipped"]["no_boxes_or_corners"] == 1
    pieces, board = tmp_path / "out" / "pieces", tmp_path / "out" / "board"
    assert yolo_class_names(pieces) == dict(enumerate(PIECE_CLASSES))
    assert yolo_kpt_shape(board / "train") == [4, 3]
    for split in ("train", "valid", "test"):
        assert len(list((pieces / split / "images").iterdir())) == 1
        assert len(list((board / split / "images").iterdir())) == 1
    label = next((pieces / "test" / "labels").iterdir()).read_text().split("\n")
    king = label[0].split()
    assert PIECE_CLASSES[int(king[0])] == "white-king"
    assert [float(v) for v in king[1:]] == pytest.approx([0.175, 0.3, 0.15, 0.4])
    black = label[1].split()
    assert PIECE_CLASSES[int(black[0])] == "black-king"
    assert float(black[4]) == pytest.approx(0.6)  # clipped to the image
    corners = read_board_label(next((board / "train" / "labels").iterdir()), 200, 100)
    assert [v for p in corners for v in p] == pytest.approx([20, 10, 180, 12, 190, 95, 10, 90], abs=0.05)
    assert set(result["folders"]["pieces"]) == {"train", "valid", "test"}


def test_chessred_test_cases(tmp_path):
    ann = make_chessred(tmp_path / "chessred")
    cases = chessred_test_cases(ann, tmp_path / "chessred")
    assert len(cases) == 1
    assert cases[0]["fen"] == "4k3/8/8/8/8/8/8/4K3"
    assert cases[0]["image"].endswith("G000_IMG002.jpg")


def test_import_rejects_other_json(tmp_path):
    path = tmp_path / "x.json"
    path.write_text("[]")
    with pytest.raises(ValueError):
        import_chessred2k(path, tmp_path, tmp_path / "out")
