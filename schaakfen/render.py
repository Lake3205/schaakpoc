"""Stage 4a: annotated per-move images (board border, labeled pieces, FEN banner)."""
from __future__ import annotations

import chess
import cv2
import numpy as np

from .calibration import BoardCalibration
from .detection import Detection

FONT = cv2.FONT_HERSHEY_SIMPLEX
BORDER_COLOR = (0, 255, 0)
WHITE_PIECE_COLOR = (0, 165, 255)
BLACK_PIECE_COLOR = (255, 255, 0)
UNKNOWN_COLOR = (180, 180, 180)
BASE_POINT_COLOR = (0, 0, 255)


def _piece_color(detection: Detection) -> tuple[int, int, int]:
    piece = detection.piece
    if piece is None:
        return UNKNOWN_COLOR
    return WHITE_PIECE_COLOR if piece.color == chess.WHITE else BLACK_PIECE_COLOR


def _fit_scale(text: str, width: int, thickness: int, max_scale: float) -> float:
    """Largest font scale (up to ``max_scale``) at which ``text`` fits in ``width`` pixels."""
    text_width = cv2.getTextSize(text, FONT, 1.0, thickness)[0][0]
    return max(0.3, min(max_scale, width / max(text_width, 1)))


def _banner(width: int, lines: list[str]) -> np.ndarray:
    """High-contrast status bar (white text on black) with one row per line."""
    max_scale = max(0.5, width / 1200)
    scales = [_fit_scale(line, width - 20, 1, max_scale) for line in lines]
    heights = [cv2.getTextSize(line, FONT, s, 1)[0][1] for line, s in zip(lines, scales)]
    pad = 10
    bar = np.zeros((sum(heights) + pad * (len(lines) + 1), width, 3), dtype=np.uint8)
    y = pad
    for line, scale, height in zip(lines, scales, heights):
        y += height
        thickness = 2 if scale >= 0.8 else 1
        cv2.putText(bar, line, (10, y), FONT, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += pad
    return bar


def render_annotated(image: np.ndarray, detections: list[Detection],
                     squares: list[chess.Square | None], calibration: BoardCalibration | None,
                     fen: str, move_number: int, move_text: str | None = None) -> np.ndarray:
    """Draw the board border, every detected piece (``white-knight @ f3``) with its base point,
    and a banner with the move number and full FEN above the image."""
    out = image.copy()
    h, w = out.shape[:2]
    thick = max(2, round(max(h, w) / 300))
    if calibration is not None:
        corners = calibration.outer_corners().round().astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [corners], isClosed=True, color=BORDER_COLOR, thickness=thick * 2,
                      lineType=cv2.LINE_AA)
    font_scale = max(0.4, max(h, w) / 1600)
    for detection, square in zip(detections, squares):
        color = _piece_color(detection)
        x1, y1, x2, y2 = (int(round(v)) for v in detection.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        label = f"{detection.name} @ {chess.square_name(square) if square is not None else '?'}"
        (tw, th), baseline = cv2.getTextSize(label, FONT, font_scale, 1)
        top = max(0, y1 - th - baseline - 4)
        cv2.rectangle(out, (x1, top), (x1 + tw + 4, top + th + baseline + 4), color, cv2.FILLED)
        cv2.putText(out, label, (x1 + 2, top + th + 2), FONT, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
        bx, by = (int(round(v)) for v in detection.base_point)
        cv2.circle(out, (bx, by), thick + 2, BASE_POINT_COLOR, cv2.FILLED)
    title = f"Move {move_number}" + (f": {move_text}" if move_text else "")
    return np.vstack([_banner(w, [title, f"FEN: {fen}"]), out])
