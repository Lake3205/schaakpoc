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
    corrected: bool = False  # the previous frame's move was replaced


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

    def _search(self, observed: dict[chess.Square, chess.Piece],
                depth: int) -> tuple[list[chess.Move], float]:
        """Best line of exactly ``depth`` legal moves from the current board (ties: first found)."""
        best_line: list[chess.Move] = []
        best_cost = float("inf")
        for move in list(self.board.legal_moves):
            self.board.push(move)
            if depth == 1:
                line, cost = [], position_mismatch(self.board.piece_map(), observed)
            else:
                line, cost = self._search(observed, depth - 1)
            self.board.pop()
            if cost < best_cost:
                best_line, best_cost = [move, *line], cost
                if cost == 0:
                    break
        return best_line, best_cost

    def best_move(self, observed: dict[chess.Square, chess.Piece]) -> tuple[chess.Move | None, float]:
        """The legal move that best explains ``observed`` and its mismatch cost; ``None`` when
        the current position already matches exactly (the same position captured twice)."""
        current = position_mismatch(self.board.piece_map(), observed)
        if current == 0:
            return None, 0.0
        line, cost = self._search(observed, 1)
        return (line[0], cost) if line else (None, current)

    def process(self, detections: list[Detection]) -> FrameResult:
        """Update the game with the detections of the next captured frame.

        A frame is captured after every move, so the best legal move is played unless the
        position did not change at all. When that move still leaves mismatches, two-move lines
        are tried too: from the current board (a frame was missed) and replacing the previous
        move (it was misread, e.g. because the moved piece was hidden).
        """
        if self.calibration is None:
            self.calibration = calibrate(detections)
        self.frame += 1
        squares = self.assign_squares(detections)
        observed = self.observe(detections, squares)
        if self.frame == 0:
            return FrameResult(0, self.board.fen(), squares=squares,
                               mismatches=position_mismatch(self.board.piece_map(), observed))
        move, cost = self.best_move(observed)
        line, corrected = ([move] if move is not None else []), False
        if cost > 0 and self.board.move_stack and self.frame > 1:
            previous = self.board.pop()
            redo_line, redo_cost = self._search(observed, 2)
            self.board.push(previous)
            if redo_cost < cost and redo_line[0] != previous:
                line, cost, corrected = redo_line, redo_cost, True
        if cost > 0:
            two_line, two_cost = self._search(observed, 2)
            if two_cost < cost:
                line, cost, corrected = two_line, two_cost, False
        if corrected:
            self.board.pop()
        sans = []
        for m in line:
            sans.append(self.board.san(m))
            self.board.push(m)
        return FrameResult(self.frame, self.board.fen(), line[-1] if line else None,
                           " ".join(sans) or None, squares, cost, corrected)
