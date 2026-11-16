import itertools

import numpy as np
import pytest

from schaakgui.board import (BOARD_PX, BoardFinder, board_homography, cell_at, edge_distance,
                             find_corners_opencv, grid_lines, inner_to_outer, order_corners,
                             to_board, to_image, warp_board)
from scan_helpers import PHOTO_CORNERS, board_photo

TRUTH = ((100.0, 50.0), (300.0, 60.0), (320.0, 250.0), (90.0, 240.0))


@pytest.mark.parametrize("perm", list(itertools.permutations(range(4))))
def test_order_corners_any_order(perm):
    assert order_corners([TRUTH[i] for i in perm]) == TRUTH


def test_order_corners_rotated_board():
    diamond = [(200.0, 10.0), (390.0, 200.0), (200.0, 390.0), (10.0, 200.0)]
    ordered = order_corners(list(reversed(diamond)))
    # clockwise, starting from the corner nearest (0, 0)
    assert len(set(ordered)) == 4
    xs = [p[0] for p in ordered]
    ys = [p[1] for p in ordered]
    area = sum(xs[i] * ys[(i + 1) % 4] - xs[(i + 1) % 4] * ys[i] for i in range(4))
    assert area > 0  # clockwise in image coordinates (y down)


def test_homography_round_trip():
    h = board_homography(PHOTO_CORNERS)
    for corner, target in zip(PHOTO_CORNERS, [(0, 0), (BOARD_PX, 0), (BOARD_PX, BOARD_PX), (0, BOARD_PX)]):
        assert to_board(h, corner) == pytest.approx(target, abs=1e-6)
    for point in [(400.0, 300.0), (123.4, 567.8), (700.0, 600.0)]:
        assert to_image(h, to_board(h, point)) == pytest.approx(point, abs=1e-6)


def test_cell_at_and_edge_distance():
    assert cell_at((50, 50)) == (0, 0)
    assert cell_at((799, 799)) == (7, 7)
    assert cell_at((450, 150)) == (4, 1)
    assert cell_at((-1, 50)) is None
    assert cell_at((50, 800)) is None
    assert edge_distance((50, 50)) == pytest.approx(0.5)
    assert edge_distance((105, 150)) == pytest.approx(0.05)
    assert edge_distance((450, 150)) == pytest.approx(0.5)


def test_grid_lines_and_warp():
    lines = grid_lines(PHOTO_CORNERS)
    assert len(lines) == 18
    warped = warp_board(board_photo(), PHOTO_CORNERS)
    assert warped.size == (BOARD_PX, BOARD_PX)
    gray = np.asarray(warped.convert("L"), dtype=float)
    # a1 (bottom-left) is dark, b1 light
    assert gray[750, 50] < 100 < gray[750, 150]


def test_inner_to_outer_extrapolates_one_square():
    inner = np.array([[(x * 10.0 + 10, y * 10.0 + 10) for x in range(1, 8)] for y in range(1, 8)])
    outer = inner_to_outer(inner.reshape(-1, 2))
    assert np.allclose(outer, [(10, 10), (90, 10), (90, 90), (10, 90)])


def test_find_corners_opencv_on_rendered_board():
    proposal = find_corners_opencv(board_photo())
    assert proposal is not None and proposal.method == "opencv"
    for found, truth in zip(proposal.corners, PHOTO_CORNERS):
        assert np.hypot(found[0] - truth[0], found[1] - truth[1]) < 8


def test_board_finder_falls_back_to_opencv_and_none():
    from PIL import Image

    finder = BoardFinder(None)
    assert finder.propose(board_photo()).method == "opencv"
    assert finder.propose(Image.new("RGB", (300, 200), (60, 130, 60))) is None
