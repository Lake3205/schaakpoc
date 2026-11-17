import json

import pytest
from PIL import Image

from schaakgui.board import BoardFinder
from schaakgui.data import yolo_class_names
from schaakgui.fen import GameInfo, placement_fen
from schaakgui.keypoints import read_board_label
from schaakgui.scan_session import ScanSession
from scan_helpers import (PHOTO_CORNERS, POSITION, FakeTester, NoFinder, board_photo, draw_pieces,
                          piece_boxes)

EXPECTED = placement_fen(POSITION)  # "4k3/p7/2n5/8/8/8/P6P/3QK3"


def make_session(tmp_path, boxes=None, finder=None, a1_corner=3):
    boxes = piece_boxes(a1_corner=a1_corner) if boxes is None else boxes
    photo = draw_pieces(board_photo(a1_corner=a1_corner), boxes)
    path = tmp_path / "garden.png"
    photo.save(path)
    tester = FakeTester(boxes)
    session = ScanSession(tester, finder or NoFinder())
    session.load_photo(path)
    return session, tester


def test_full_pipeline_expected_fen(tmp_path):
    session, tester = make_session(tmp_path)
    session.set_corners(list(reversed(PHOTO_CORNERS)))  # any order
    session.detect_pieces(0.25)
    assert session.suggest_orientation() == 3
    assert session.fen().split()[0] == EXPECTED
    assert session.problems() == []
    assert tester.calls == 1


@pytest.mark.parametrize("a1", range(4))
def test_orientation_is_found_for_every_board_rotation(tmp_path, a1):
    session, _ = make_session(tmp_path, a1_corner=a1)
    session.set_corners(PHOTO_CORNERS)
    session.detect_pieces()
    assert session.suggest_orientation() == a1
    assert session.fen().split()[0] == EXPECTED


def test_auto_corners_with_opencv(tmp_path):
    # empty board: the classic checkerboard finder must see the inner corners
    session = ScanSession(FakeTester([]), BoardFinder(None))
    session.set_image(board_photo())
    proposal = session.auto_corners()
    assert proposal is not None and session.corner_method == "opencv"
    for (x, y), (tx, ty) in zip(session.corners, PHOTO_CORNERS):
        assert abs(x - tx) < 8 and abs(y - ty) < 8


def test_corner_drag_and_rotation_do_not_rerun_detection(tmp_path):
    session, tester = make_session(tmp_path)
    session.set_corners(PHOTO_CORNERS)
    session.detect_pieces()
    fen = session.fen()
    session.rotate()
    assert session.a1_corner == 0 and session.fen() != fen
    for _ in range(3):
        session.rotate()
    assert session.fen() == fen
    # dragging a corner far away moves every piece to other squares, without re-detecting
    session.move_corner(3, (80.0, 400.0))
    assert session.fen() != fen
    session.move_corner(3, PHOTO_CORNERS[3])
    assert session.fen() == fen
    session.set_lift(0.08)
    assert tester.calls == 1


def test_square_edits_survive_rotation_and_replacement(tmp_path):
    session, tester = make_session(tmp_path)
    session.set_corners(PHOTO_CORNERS)
    session.detect_pieces()
    session.set_square("c6", "b")  # knight → bishop
    session.set_square("a2", None)  # remove the pawn
    session.set_square("d4", "Q")  # add a missed piece
    squares = session.squares()
    assert squares["c6"] == "b" and "a2" not in squares and squares["d4"] == "Q"
    session.rotate()
    session.rotate()  # a1 at the top-right now: the same physical cells are edited
    assert session.squares()["f3"] == "b" and session.squares()["e5"] == "Q"
    session.set_a1_corner(3)
    session.set_conf(0.3)
    session.move_corner(0, PHOTO_CORNERS[0])
    assert session.squares()["c6"] == "b" and "a2" not in session.squares()
    session.clear_edits()
    assert session.fen().split()[0] == EXPECTED
    assert tester.calls == 1


def test_conf_threshold_filters_without_detecting(tmp_path):
    boxes = piece_boxes()
    boxes[0]["conf"] = 0.2  # e1 king
    session, tester = make_session(tmp_path, boxes)
    session.set_corners(PHOTO_CORNERS)
    session.detect_pieces(0.25)
    assert "e1" not in session.squares()
    session.set_conf(0.1)
    assert session.squares()["e1"] == "K"
    assert tester.calls == 1


def test_box_edits(tmp_path):
    session, _ = make_session(tmp_path)
    session.set_corners(PHOTO_CORNERS)
    session.detect_pieces()
    n = len(session.detections)
    index = session.add_box((560, 380, 600, 330), "R")
    assert index == n and session.detections[index]["label"] == "white-rook"
    added = [p for p in session.placements if p.index == index][0]
    assert added.cell is not None
    session.relabel(0, "white-queen")
    assert session.detections[0]["conf"] == 1.0
    session.remove_detection(index)
    assert len(session.detections) == n


def test_warnings_and_info(tmp_path):
    session, _ = make_session(tmp_path)
    session.set_corners(PHOTO_CORNERS)
    session.detect_pieces()
    assert session.warnings() == []
    session.set_square("e8", None)
    assert any("no king" in w.lower() for w in session.warnings())
    session.set_info(GameInfo(side="b", castling="-", fullmove=12))
    assert session.fen().endswith(" b - - 0 12")


def test_exports(tmp_path):
    session, _ = make_session(tmp_path)
    session.set_corners(PHOTO_CORNERS)
    session.detect_pieces()
    written = session.export_all(tmp_path / "out", debug=True)
    out = tmp_path / "out"
    assert (out / "fen.txt").read_text().strip() == session.fen()
    assert Image.open(out / "annotated.png").size == session.image.size
    assert Image.open(out / "diagram.png").size == (640, 640)
    data = json.loads((out / "scan.json").read_text())
    assert data["fen"] == session.fen() and data["a1_corner"] == "BL"
    names = sorted(p.name for p in out.glob("[1-5]_*.png"))
    assert names == ["1_corners.png", "2_straight.png", "3_pieces.png", "4_feet.png", "5_diagram.png"]
    assert len(written["debug"]) == 5


def test_save_training_sample(tmp_path):
    session, _ = make_session(tmp_path)
    session.set_corners(PHOTO_CORNERS)
    session.detect_pieces()
    session.set_square("c6", "b")  # relabels the knight's box
    session.set_square("a2", None)  # drops the pawn's box
    session.set_square("d4", "Q")  # no box → note
    result = session.save_training_sample(tmp_path / "samples", "valid")
    root = tmp_path / "samples"
    names = yolo_class_names(root / "pieces")
    lines = [line.split() for line in open(result["pieces_label"]).read().splitlines()]
    labels = sorted(names[int(line[0])] for line in lines)
    assert labels == sorted(["white-king", "white-queen", "white-pawn", "black-king", "black-pawn",
                             "black-bishop"])
    assert any("d4" in note for note in result["notes"])
    corners = read_board_label(result["board_label"], *session.image.size)
    assert corners[0] == pytest.approx(PHOTO_CORNERS[0], abs=0.1)
    assert "kpt_shape" in (root / "board" / "data.yaml").read_text()
    assert "val: valid/images" in (root / "pieces" / "data.yaml").read_text()
    with pytest.raises(ValueError):
        session.save_training_sample(root, "nope")


def test_detect_requires_detector(tmp_path):
    session = ScanSession(FakeTester([]), NoFinder())
    session.tester.detector = None
    session.set_image(board_photo())
    with pytest.raises(ValueError):
        session.detect_pieces()


def test_scan_cli(tmp_path, monkeypatch, capsys):
    import scan
    import schaakgui.detector as detector_module

    boxes = piece_boxes()
    draw_pieces(board_photo(), boxes).save(tmp_path / "photo.png")
    monkeypatch.setattr(detector_module, "ImageTester", lambda weights=None: FakeTester(boxes))
    corners = ",".join(f"{v:g}" for p in PHOTO_CORNERS for v in p)
    assert scan.main([str(tmp_path / "photo.png"), "--pieces", "fake.pt", "--corners", corners,
                      "--side", "b", "--out", str(tmp_path / "out"), "--debug"]) == 0
    out = capsys.readouterr().out
    assert f"FEN: {EXPECTED} b - - 0 1" in out
    assert (tmp_path / "out" / "fen.txt").is_file() and (tmp_path / "out" / "5_diagram.png").is_file()
    with pytest.raises(SystemExit):
        scan.main(["--corners", "1,2,3"])
