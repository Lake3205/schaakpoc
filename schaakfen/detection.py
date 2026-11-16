"""Stage 1: detected piece boxes and their ground-plane base points."""
from __future__ import annotations

from dataclasses import dataclass

import chess

# Fraction of the box height above the bottom edge where the piece touches the board.
BASE_OFFSET = 0.15

PIECE_NAMES: dict[str, chess.Piece] = {
    f"{color_name}-{type_name}": chess.Piece(piece_type, color)
    for color_name, color in (("white", chess.WHITE), ("black", chess.BLACK))
    for type_name, piece_type in (("pawn", chess.PAWN), ("knight", chess.KNIGHT),
                                  ("bishop", chess.BISHOP), ("rook", chess.ROOK),
                                  ("queen", chess.QUEEN), ("king", chess.KING))
}


def normalize_name(name: str) -> str:
    """``"White-Knight"`` / ``"white_knight"`` / ``"white knight"`` -> ``"white-knight"``."""
    return name.strip().lower().replace("_", "-").replace(" ", "-")


def base_point(box: tuple[float, float, float, float]) -> tuple[float, float]:
    """Footprint of a piece near the bottom-center of its ``(x1, y1, x2, y2)`` box.

    The box center is not used: on steep camera angles tall pieces (kings, queens)
    would project onto the square behind them.
    """
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, y2 - BASE_OFFSET * (y2 - y1)


@dataclass(frozen=True)
class Detection:
    """One detected piece: class name, ``(x1, y1, x2, y2)`` box in image pixels and confidence."""

    name: str
    box: tuple[float, float, float, float]
    conf: float = 1.0

    @property
    def base_point(self) -> tuple[float, float]:
        return base_point(self.box)

    @property
    def piece(self) -> chess.Piece | None:
        """The chess piece of this class, or ``None`` for unknown class names."""
        return PIECE_NAMES.get(normalize_name(self.name))


def detections_from_result(result) -> list[Detection]:
    """Convert one Ultralytics YOLO result into :class:`Detection` objects."""
    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
        name = result.names[int(box.cls)]
        detections.append(Detection(normalize_name(name), (x1, y1, x2, y2), float(box.conf)))
    return detections
