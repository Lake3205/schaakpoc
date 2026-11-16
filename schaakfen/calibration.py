"""Stage 2: frame-0 homography calibration, locked for the rest of the game.

The canonical board space is ``800x800`` pixels with white at the bottom: square ``a8`` is the
top-left 100x100 cell and ``h1`` the bottom-right one.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import chess
import cv2
import numpy as np

from .detection import Detection

BOARD_SIZE = 800
SQUARE_SIZE = BOARD_SIZE // 8
OUTER_CORNERS = np.array([(0, 0), (BOARD_SIZE, 0), (BOARD_SIZE, BOARD_SIZE), (0, BOARD_SIZE)],
                         dtype=np.float32)
# Back-rank files of each piece type in the starting setup (a=0 ... h=7).
BACK_RANK_FILES = {chess.ROOK: (0, 7), chess.KNIGHT: (1, 6), chess.BISHOP: (2, 5),
                   chess.QUEEN: (3,), chess.KING: (4,)}


class CalibrationError(RuntimeError):
    """Frame 0 does not show enough of the starting position to compute the homography."""


def square_center(square: chess.Square) -> tuple[float, float]:
    """Center of ``square`` in canonical board space."""
    return ((chess.square_file(square) + 0.5) * SQUARE_SIZE,
            (7 - chess.square_rank(square) + 0.5) * SQUARE_SIZE)


def _transform(points, matrix: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return cv2.perspectiveTransform(pts, matrix).reshape(-1, 2)


@dataclass(frozen=True)
class BoardCalibration:
    """Frozen perspective transform of one game.

    ``H`` maps image pixels to canonical board space, ``H_inv`` maps back, and ``orientation``
    (0, 90, 180 or 270) is the clockwise rotation of the board in the image: 0 means white
    at the bottom of the image, 90 white on the left.
    """

    H: np.ndarray
    H_inv: np.ndarray
    orientation: int

    @classmethod
    def from_homography(cls, H: np.ndarray) -> "BoardCalibration":
        H = np.asarray(H, dtype=np.float64)
        H_inv = np.linalg.inv(H)
        bottom, top = _transform([(BOARD_SIZE / 2, BOARD_SIZE), (BOARD_SIZE / 2, 0)], H_inv)
        dx, dy = top - bottom  # image direction from white's side to black's side
        angle = math.degrees(math.atan2(dx, -dy))
        orientation = int(round(angle / 90.0)) * 90 % 360
        return cls(H, H_inv, orientation)

    def to_board(self, points) -> np.ndarray:
        """Image pixel points -> canonical board points, shape ``(N, 2)``."""
        return _transform(points, self.H)

    def to_image(self, points) -> np.ndarray:
        """Canonical board points -> image pixel points, shape ``(N, 2)``."""
        return _transform(points, self.H_inv)

    def outer_corners(self) -> np.ndarray:
        """The 4 outer board corners in image pixels (for drawing the border)."""
        return self.to_image(OUTER_CORNERS)

    def square_at(self, point: tuple[float, float]) -> chess.Square | None:
        """The square under an image point, or ``None`` if it lies off the board."""
        x, y = self.to_board([point])[0]
        if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
            return None
        return chess.square(int(x // SQUARE_SIZE), 7 - int(y // SQUARE_SIZE))


def _initial_matches(detections: list[Detection]) -> list[tuple[Detection, chess.Square]]:
    """Match frame-0 detections to starting squares without knowing the board geometry.

    The direction from the white pieces to the black pieces gives the rank axis; turning it
    90 degrees clockwise gives the a->h file axis (camera images are never mirrored). Pieces of
    each starting rank are sorted along that axis to find their files.
    """
    pieces = [d for d in detections if d.piece is not None]
    white = [d for d in pieces if d.piece.color == chess.WHITE]
    black = [d for d in pieces if d.piece.color == chess.BLACK]
    if not white or not black:
        return []
    up = (np.mean([d.base_point for d in black], axis=0)
          - np.mean([d.base_point for d in white], axis=0))
    file_axis = np.array([-up[1], up[0]])

    def by_file(group: list[Detection]) -> list[Detection]:
        return sorted(group, key=lambda d: float(np.dot(d.base_point, file_axis)))

    matches = []
    for color, color_pieces in ((chess.WHITE, white), (chess.BLACK, black)):
        back_rank = 0 if color == chess.WHITE else 7
        pawn_rank = 1 if color == chess.WHITE else 6
        pawns = [d for d in color_pieces if d.piece.piece_type == chess.PAWN]
        officers = [d for d in color_pieces if d.piece.piece_type != chess.PAWN]
        if len(pawns) == 8:
            matches += [(d, chess.square(f, pawn_rank)) for f, d in enumerate(by_file(pawns))]
        if len(officers) == 8:
            matches += [(d, chess.square(f, back_rank)) for f, d in enumerate(by_file(officers))]
            continue
        by_type = defaultdict(list)
        for d in officers:
            by_type[d.piece.piece_type].append(d)
        for piece_type, files in BACK_RANK_FILES.items():
            group = by_type.get(piece_type, [])
            if len(group) == len(files):
                matches += [(d, chess.square(f, back_rank)) for f, d in zip(files, by_file(group))]
    return matches


def _refined_matches(detections: list[Detection], H: np.ndarray) -> list[tuple[Detection, chess.Square]]:
    """Match each detection to the nearest free starting square of its own piece."""
    start = chess.Board()
    pieces = [d for d in detections if d.piece is not None]
    projected = _transform([d.base_point for d in pieces], H)
    candidates = []
    for i, (d, (x, y)) in enumerate(zip(pieces, projected)):
        for square in start.pieces(d.piece.piece_type, d.piece.color):
            cx, cy = square_center(square)
            distance = math.hypot(x - cx, y - cy)
            if distance < SQUARE_SIZE:
                candidates.append((distance, i, square))
    used_detections, used_squares, matches = set(), set(), []
    for _, i, square in sorted(candidates):
        if i not in used_detections and square not in used_squares:
            used_detections.add(i)
            used_squares.add(square)
            matches.append((pieces[i], square))
    return matches


def _homography(matches: list[tuple[Detection, chess.Square]], threshold: float):
    if len(matches) < 4:
        return None, 0
    src = np.array([d.base_point for d, _ in matches], dtype=np.float32)
    dst = np.array([square_center(s) for _, s in matches], dtype=np.float32)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, threshold)
    if H is None or abs(np.linalg.det(H)) < 1e-12:
        return None, 0
    return H, int(mask.sum())


def calibrate(detections: list[Detection], ransac_threshold: float = SQUARE_SIZE / 2,
              refine_iterations: int = 3, min_inliers: int = 8) -> BoardCalibration:
    """Compute the frozen board calibration from frame 0 (starting position ``rnbqkbnr/...``).

    Detected base points are matched to their canonical starting squares and the homography
    is estimated with ``cv2.findHomography(..., cv2.RANSAC)`` so outliers and occluded pieces
    are ignored. The estimate is refined by re-matching all detections to the nearest
    starting square of their own piece.
    """
    H, inliers = _homography(_initial_matches(detections), ransac_threshold)
    if H is None:
        raise CalibrationError("Not enough starting-position pieces detected to locate the board")
    for _ in range(refine_iterations):
        new_H, new_inliers = _homography(_refined_matches(detections, H), ransac_threshold)
        if new_H is None or new_inliers < inliers:
            break
        H, inliers = new_H, new_inliers
    if inliers < min_inliers:
        raise CalibrationError(f"Only {inliers} pieces match the starting position "
                               f"(need {min_inliers})")
    return BoardCalibration.from_homography(H)
