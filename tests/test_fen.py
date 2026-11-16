import subprocess
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
chess = pytest.importorskip("chess")

from schaakfen import (CalibrationError, Detection, FenSession, GameTracker, base_point,  # noqa: E402
                       calibrate, render_annotated)
from schaakfen.calibration import BOARD_SIZE, square_center  # noqa: E402

NAMES = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
         chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king"}
HEIGHTS = {chess.PAWN: 30, chess.KNIGHT: 40, chess.BISHOP: 45, chess.ROOK: 35,
           chess.QUEEN: 55, chess.KING: 60}


def camera(rotation: int) -> np.ndarray:
    """Board -> image homography: an oblique camera with the board rotated clockwise."""
    corners = np.float32([(0, 0), (800, 0), (800, 800), (0, 800)])
    image = np.float32([(250, 120), (750, 120), (900, 600), (100, 600)])  # trapezoid (tilted)
    image = np.roll(image, -(rotation // 90), axis=0)
    return cv2.getPerspectiveTransform(corners, image)


def scene(board: chess.Board, G: np.ndarray, names=None) -> list[Detection]:
    """Detections whose base points lie on the square centers of ``board`` seen through ``G``."""
    detections = []
    for square, piece in board.piece_map().items():
        (x, y), = cv2.perspectiveTransform(np.float32([[square_center(square)]]), G)[0:1, 0]
        h = HEIGHTS[piece.piece_type]
        y2 = y + 0.15 * h
        color = "white" if piece.color == chess.WHITE else "black"
        name = f"{color}-{NAMES[piece.piece_type]}"
        detections.append(Detection(names(name) if names else name, (x - 12, y2 - h, x + 12, y2), 0.9))
    return detections


def test_base_point_is_near_bottom_center():
    assert base_point((10, 100, 30, 200)) == (20, 185)
    assert Detection("White-Knight", (0, 0, 1, 1)).piece == chess.Piece(chess.KNIGHT, chess.WHITE)
    assert Detection("board", (0, 0, 1, 1)).piece is None


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_calibration_recovers_squares_and_orientation(rotation):
    G = camera(rotation)
    board = chess.Board()
    detections = scene(board, G, names=lambda n: n.title())  # dataset style "White-King"
    calibration = calibrate(detections)
    assert calibration.orientation == rotation
    assert np.allclose(calibration.H_inv @ calibration.H / (calibration.H_inv @ calibration.H)[2, 2],
                       np.eye(3), atol=1e-6)
    tracker = GameTracker(calibration)
    assert tracker.observe(detections) == board.piece_map()
    expected = cv2.perspectiveTransform(np.float32([[(0, 0), (BOARD_SIZE, 0)]]), G)[0]
    assert np.allclose(calibration.outer_corners()[:2], expected, atol=1.0)


def test_calibration_ignores_outliers_and_missing_pieces():
    G = camera(0)
    detections = scene(chess.Board(), G)
    detections = [d for d in detections if d.name != "white-pawn" or d.box[0] > 300][:-2]
    detections.append(Detection("black-queen", (500, 500, 520, 540)))  # false detection
    tracker = GameTracker(calibrate(detections))
    observed = tracker.observe(scene(chess.Board(), G))
    assert observed == chess.Board().piece_map()


def test_calibration_fails_without_pieces():
    with pytest.raises(CalibrationError):
        calibrate([Detection("white-king", (0, 0, 10, 10))])


def test_tracker_follows_legal_moves_and_fen():
    G = camera(90)
    tracker = GameTracker()
    truth = chess.Board()
    assert tracker.process(scene(truth, G)).fen == chess.STARTING_FEN
    for san in ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "O-O", "d6"]:
        truth.push_san(san)
        detections = scene(truth, G)
        result = tracker.process(detections)
        assert result.san == san
        assert result.fen == truth.fen()
        assert all(s is not None for s in result.squares)
    # A repeated frame (no move made) keeps the state.
    assert tracker.process(scene(truth, G)).move is None
    assert tracker.board.fen() == truth.fen()


def test_tracker_tolerates_type_confusion_and_missed_piece():
    G = camera(0)
    tracker = GameTracker()
    tracker.process(scene(chess.Board(), G))
    truth = chess.Board()
    truth.push_san("Nf3")
    detections = scene(truth, G)
    knight = next(d for d in detections if d.name == "white-knight" and d.box[0] > 500)
    detections.remove(knight)
    detections.append(Detection("white-bishop", knight.box, 0.6))  # misclassified
    detections = [d for d in detections if d.name != "black-rook"]  # occluded rooks
    assert tracker.process(detections).san == "Nf3"


def test_session_exports_images_and_fen_log(tmp_path):
    G = camera(0)
    image = np.full((720, 1000, 3), 90, dtype=np.uint8)
    session = FenSession(tmp_path / "game")
    truth = chess.Board()
    frames = [session.process(image, scene(truth, G))]
    truth.push_san("d4")
    frames.append(session.process(image, scene(truth, G)))
    assert [f.image_path.name for f in frames] == ["move_0.jpg", "move_1.jpg"]
    assert all(f.image_path.exists() for f in frames)
    log = (tmp_path / "game" / "moves.fen").read_text(encoding="utf-8").splitlines()
    assert log == [chess.STARTING_FEN, truth.fen()]
    exported = cv2.imread(str(frames[1].image_path))
    assert exported.shape[1] == 1000 and exported.shape[0] > 720  # banner added on top


def test_render_draws_border_boxes_and_banner():
    G = camera(0)
    image = np.zeros((720, 1000, 3), dtype=np.uint8)
    detections = scene(chess.Board(), G)
    calibration = calibrate(detections)
    squares = GameTracker(calibration).assign_squares(detections)
    out = render_annotated(image, detections, squares, calibration, chess.STARTING_FEN, 0)
    banner = out.shape[0] - image.shape[0]
    assert banner > 0
    assert out[:banner].max() == 255  # white FEN text on black banner
    assert out[banner:].any()
    assert image.max() == 0  # the input image is not modified


def test_fen_cli_help():
    root = Path(__file__).resolve().parents[1]
    out = subprocess.run([sys.executable, str(root / "fen.py"), "--help"],
                         capture_output=True, text=True, check=True).stdout
    assert "--source" in out and "moves.fen" in out


def test_tracker_resyncs_after_hidden_moved_piece_and_missed_frame():
    G = camera(0)
    tracker = GameTracker()
    truth = chess.Board()
    tracker.process(scene(truth, G))
    truth.push_san("e4")
    hidden = [d for d in scene(truth, G) if d.name != "white-pawn" or d.box[1] > 400 or d.box[3] < 300]
    first = tracker.process(hidden)  # the moved pawn on e4 is not detected
    assert first.move is not None and tracker.board.turn == chess.BLACK
    truth.push_san("e5")
    second = tracker.process(scene(truth, G))
    assert tracker.board.fen() == truth.fen() and second.mismatches == 0
    for san in ["Nf3", "Nc6"]:  # the frame after Nf3 is never captured
        truth.push_san(san)
    third = tracker.process(scene(truth, G))
    assert third.san == "Nf3 Nc6" and tracker.board.fen() == truth.fen()
