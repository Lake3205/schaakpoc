"""Stage 3: temporal board-state tracking by matching legal moves to the detections."""
from __future__ import annotations

from dataclasses import dataclass, field

import chess

from .calibration import BoardCalibration, calibrate
from .detection import Detection


@dataclass
class FrameResult:
    """State after one processed frame (frame 0 is the starting position)."""

    frame: int
    fen: str
    move: chess.Move | None = None
    san: str | None = None
    squares: list[chess.Square | None] = field(default_factory=list)
    mismatches: float = 0.0


def position_mismatch(expected: dict[chess.Square, chess.Piece],
                      observed: dict[chess.Square, chess.Piece]) -> float:
    """How badly a board position explains the observed pieces (0 = perfect).

    Every square with a missing, extra or wrongly colored piece costs 1; a piece of the
    right color but the wrong type costs 0.5.
    """
    cost = 0.0
    for square in set(expected) | set(observed):
        want, seen = expected.get(square), observed.get(square)
        if want == seen:
            continue
        if want is not None and seen is not None and want.color == seen.color:
            cost += 0.5
        else:
            cost += 1.0
    return cost


class GameTracker:
    """Keeps a ``chess.Board`` across frames, captured once between every move.

    The first processed frame must show the starting position; it locks the calibration.
    Every later frame is explained by the legal move whose resulting position best matches
    the detections projected onto the 8x8 grid.
    """

    def __init__(self, calibration: BoardCalibration | None = None,
                 board: chess.Board | None = None) -> None:
        self.calibration = calibration
        self.board = board if board is not None else chess.Board()
        self.frame = -1

    def assign_squares(self, detections: list[Detection]) -> list[chess.Square | None]:
        """The square under each detection's base point (``None`` when off the board)."""
        if self.calibration is None:
            return [None] * len(detections)
        return [self.calibration.square_at(d.base_point) for d in detections]

    def observe(self, detections: list[Detection],
                squares: list[chess.Square | None] | None = None) -> dict[chess.Square, chess.Piece]:
        """Observed piece per square; the most confident detection wins on shared squares."""
        if squares is None:
            squares = self.assign_squares(detections)
        best: dict[chess.Square, Detection] = {}
        for d, square in zip(detections, squares):
            if square is None or d.piece is None:
                continue
            if square not in best or d.conf > best[square].conf:
                best[square] = d
        return {square: d.piece for square, d in best.items()}

    def best_move(self, observed: dict[chess.Square, chess.Piece]) -> tuple[chess.Move | None, float]:
        """The legal move that best explains ``observed`` (``None`` if no move fits better
        than the current position) and its mismatch cost."""
        best_move = None
        best_cost = position_mismatch(self.board.piece_map(), observed)
        for move in self.board.legal_moves:
            self.board.push(move)
            cost = position_mismatch(self.board.piece_map(), observed)
            self.board.pop()
            if cost < best_cost:
                best_move, best_cost = move, cost
        return best_move, best_cost

    def process(self, detections: list[Detection]) -> FrameResult:
        """Update the game with the detections of the next captured frame."""
        if self.calibration is None:
            self.calibration = calibrate(detections)
        self.frame += 1
        squares = self.assign_squares(detections)
        observed = self.observe(detections, squares)
        if self.frame == 0:
            return FrameResult(0, self.board.fen(), squares=squares,
                               mismatches=position_mismatch(self.board.piece_map(), observed))
        move, cost = self.best_move(observed)
        san = None
        if move is not None:
            san = self.board.san(move)
            self.board.push(move)
        return FrameResult(self.frame, self.board.fen(), move, san, squares, cost)
