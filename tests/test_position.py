import pytest

from schaakgui.board import BOARD_PX, board_homography, to_image
from schaakgui.position import (BoardState, Placement, cell_of, corner_square_is_dark, foot_point,
                                piece_symbol, place_pieces, square_name, suggest_a1_corner)
from schaakgui.synthetic import checkerboard
from scan_helpers import PHOTO_CORNERS

SQUARES = [f + r for f in "abcdefgh" for r in "12345678"]


def test_piece_symbol_is_tolerant():
    assert piece_symbol("white-king") == "K"
    assert piece_symbol("Black_Knight") == "n"
    assert piece_symbol("WHITE PAWN") == "P"
    assert piece_symbol("blackqueen") == "q"
    assert piece_symbol("Black") is None


def test_foot_point_handles_tall_pieces():
    # A tall piece leaning into the square behind: the box top is far away, the foot stays low.
    assert foot_point((100, 0, 140, 200), lift=0.08) == pytest.approx((120, 184))
    assert foot_point((0, 0, 10, 10), lift=0) == (5, 10)


@pytest.mark.parametrize("a1", range(4))
def test_square_name_and_cell_of_are_inverse(a1):
    names = {square_name((c, r), a1) for c in range(8) for r in range(8)}
    assert names == set(SQUARES)
    for square in SQUARES:
        assert square_name(cell_of(square, a1), a1) == square


def test_square_name_corners():
    assert square_name((0, 7), 3) == "a1" and square_name((7, 0), 3) == "h8"
    assert square_name((0, 0), 0) == "a1"
    assert square_name((7, 0), 1) == "a1"
    assert square_name((7, 7), 2) == "a1"


def _placement(i, label, cell, conf=0.9, edge=0.5):
    return Placement(i, label, piece_symbol(label), conf, (0, 0, 1, 1), (0, 0), cell, edge,
                     ((cell[0] + 0.5) * 100, (cell[1] + 0.5) * 100) if cell else (0, 0))


def test_place_pieces_uses_homography():
    h = board_homography(PHOTO_CORNERS)
    fx, fy = to_image(h, (450, 750))  # centre of cell (4, 7) = e1 with a1 bottom-left
    placements = place_pieces([{"box": (fx - 10, fy - 50, fx + 10, fy), "label": "white-king",
                                "conf": 0.8}], PHOTO_CORNERS, lift=0)
    assert placements[0].cell == (4, 7)
    assert placements[0].symbol == "K"
    assert placements[0].edge == pytest.approx(0.5, abs=0.02)


def test_board_state_conflicts_uncertain_off_board():
    placements = [
        _placement(0, "white-king", (4, 7), 0.9),
        _placement(1, "white-queen", (4, 7), 0.6),  # same square → conflict, best one wins
        _placement(2, "black-pawn", (0, 1), 0.3),  # low confidence → uncertain
        _placement(3, "black-rook", (7, 0), 0.9, edge=0.05),  # near a grid line → uncertain
        _placement(4, "black-king", None),  # off-board
    ]
    state = BoardState.from_placements(placements, min_conf=0.5, min_edge=0.15)
    assert state.cells[(4, 7)] == "K"
    assert set(state.conflicts) == {(4, 7)}
    assert state.uncertain >= {(0, 1), (7, 0)}
    assert [p.index for p in state.off_board] == [4]
    squares = state.squares(3)
    assert squares["e1"] == "K" and squares["a7"] == "p" and squares["h8"] == "r"
    state.set_cell((4, 7), None)
    assert "e1" not in state.squares(3)
    assert (4, 7) in state.edited and (4, 7) not in state.conflicts


@pytest.mark.parametrize("a1", range(4))
def test_suggest_a1_corner_all_rotations(a1):
    board = checkerboard(BOARD_PX // 8).rotate(-90 * ((a1 - 3) % 4))
    assert corner_square_is_dark(board, a1)
    placements = []
    for i, square in enumerate([f + r for f in "abcdefgh" for r in "12"]):
        placements.append(_placement(i, "white-pawn", cell_of(square, a1)))
    for i, square in enumerate([f + r for f in "abcdefgh" for r in "78"]):
        placements.append(_placement(100 + i, "black-pawn", cell_of(square, a1)))
    assert suggest_a1_corner(board, placements) == a1
    # only black pieces: a1 is the dark corner away from them
    assert suggest_a1_corner(board, [p for p in placements if p.symbol == "p"]) == a1
