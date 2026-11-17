"""Pieces → squares (scan step 4, Qt independent): foot points, square names, orientation, board state.

Squares are stored as image-grid cells ``(col, row)`` of the straightened board (row 0 = top).
Chess names are only computed at the end for a chosen a1 corner (0 = TL, 1 = TR, 2 = BR, 3 = BL),
so rotating the board never re-runs detection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image

from .board import BOARD_PX, Cell, Corners, Point, board_homography, cell_at, edge_distance, to_board

PIECE_SYMBOLS = {
    "white-king": "K", "white-queen": "Q", "white-rook": "R",
    "white-bishop": "B", "white-knight": "N", "white-pawn": "P",
    "black-king": "k", "black-queen": "q", "black-rook": "r",
    "black-bishop": "b", "black-knight": "n", "black-pawn": "p",
}
SYMBOL_NAMES = {symbol: name for name, symbol in PIECE_SYMBOLS.items()}
# Class order of the piece datasets (same as chess.yaml / the synchronized rookeye dataset).
PIECE_CLASSES = sorted(PIECE_SYMBOLS)
FILES = "abcdefgh"
# Straight-board pixel position of each corner (TL, TR, BR, BL).
_CORNER_UNITS = ((0, 0), (1, 0), (1, 1), (0, 1))


def _norm(label: str) -> str:
    return str(label).strip().lower().replace("_", "-").replace(" ", "-")


_COMPACT = {name.replace("-", ""): symbol for name, symbol in PIECE_SYMBOLS.items()}


def piece_symbol(label: str) -> Optional[str]:
    """FEN symbol of a class name, tolerant to case and separators (``White_Knight`` → ``N``)."""
    norm = _norm(label)
    if norm in PIECE_SYMBOLS:
        return PIECE_SYMBOLS[norm]
    return _COMPACT.get(norm.replace("-", ""))


def foot_point(box, lift: float = 0.08) -> Point:
    """Bottom centre of a box, lifted by ``lift`` x box height (where the piece stands)."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, y2 - lift * (y2 - y1))


def _file_rank(cell: Cell, a1_corner: int) -> tuple[int, int]:
    col, row = cell
    if a1_corner == 3:  # a1 bottom-left: White at the bottom
        return col, 7 - row
    if a1_corner == 1:  # a1 top-right: White at the top
        return 7 - col, row
    if a1_corner == 0:  # a1 top-left: White on the left
        return row, col
    if a1_corner == 2:  # a1 bottom-right: White on the right
        return 7 - row, 7 - col
    raise ValueError(f"a1_corner must be 0..3, got {a1_corner}")


def square_name(cell: Cell, a1_corner: int) -> str:
    """Chess name (``e4``) of a grid cell for a chosen a1 corner."""
    file, rank = _file_rank(cell, a1_corner)
    return f"{FILES[file]}{rank + 1}"


def cell_of(square: str, a1_corner: int) -> Cell:
    """Grid cell of a chess square (inverse of :func:`square_name`)."""
    file, rank = FILES.index(square[0].lower()), int(square[1]) - 1
    if not (0 <= rank < 8):
        raise ValueError(f"Invalid square {square!r}")
    if a1_corner == 3:
        return file, 7 - rank
    if a1_corner == 1:
        return 7 - file, rank
    if a1_corner == 0:
        return rank, file
    if a1_corner == 2:
        return 7 - rank, 7 - file
    raise ValueError(f"a1_corner must be 0..3, got {a1_corner}")


@dataclass
class Placement:
    """One detection with its foot point, straight-board point, cell (``None`` = off-board)."""

    index: int
    label: str
    symbol: Optional[str]
    conf: float
    box: tuple[float, float, float, float]
    foot: Point
    cell: Optional[Cell]
    edge: float
    board_point: Point = (0.0, 0.0)


def place_pieces(detections: list[dict], corners: Corners, lift: float = 0.08,
                 size: int = BOARD_PX) -> list[Placement]:
    """Foot point → homography → cell for every detection (``box``, ``label``, ``conf``)."""
    h = board_homography(corners, size)
    placements = []
    for i, d in enumerate(detections):
        box = tuple(float(v) for v in d["box"])
        foot = foot_point(box, lift)
        point = to_board(h, foot)
        cell = cell_at(point, size)
        placements.append(Placement(
            index=int(d.get("index", i)), label=str(d["label"]), symbol=piece_symbol(d["label"]),
            conf=float(d.get("conf", 1.0)), box=box, foot=foot, cell=cell,
            edge=edge_distance(point, size) if cell is not None else 0.0, board_point=point))
    return placements


def _square_brightness(gray: np.ndarray, cell: Cell) -> float:
    size = gray.shape[0]
    square = size / 8
    col, row = cell
    m = square * 0.2  # ignore the lines and border bleeding
    patch = gray[int(row * square + m):int((row + 1) * square - m),
                 int(col * square + m):int((col + 1) * square - m)]
    return float(np.median(patch)) if patch.size else 0.0


def _gray(warped) -> np.ndarray:
    image = warped if isinstance(warped, Image.Image) else Image.fromarray(np.asarray(warped))
    return np.asarray(image.convert("L"), dtype=np.float32)


def _dark_parity(gray: np.ndarray, occupied: set[Cell] = frozenset()) -> int:
    """0 if the squares with even ``col + row`` (the TL/BR corners) are the dark ones, else 1.

    Compares the median brightness of both checker colours over the free squares (pieces standing
    on a corner square would fool a single-square test)."""
    free = [(c, r) for c in range(8) for r in range(8) if (c, r) not in occupied]
    if len(free) < 8:
        free = [(c, r) for c in range(8) for r in range(8)]
    even = [_square_brightness(gray, cell) for cell in free if sum(cell) % 2 == 0]
    odd = [_square_brightness(gray, cell) for cell in free if sum(cell) % 2 == 1]
    if not even or not odd:
        return 0
    return 0 if float(np.median(even)) <= float(np.median(odd)) else 1


def corner_square_is_dark(warped, corner: int, occupied: set[Cell] = frozenset()) -> bool:
    """Whether a corner square (0 = TL … 3 = BL) of the straightened board has the dark colour."""
    return _dark_parity(_gray(warped), occupied) == corner % 2


def suggest_a1_corner(warped, placements: list[Placement]) -> int:
    """a1 is a dark square on White's side: the dark corner nearest the white pieces.

    Rule 1: the dark checker colour decides the pair of dark corners (TL+BR or TR+BL): a1 and h8.
    Rule 2: pick the one nearest the centroid of the white pieces (or farthest from the black ones);
    without pieces the dark corner at the bottom is used."""
    gray = _gray(warped)
    size = gray.shape[0]
    occupied = {p.cell for p in placements if p.cell is not None}
    pair = (0, 2) if _dark_parity(gray, occupied) == 0 else (1, 3)

    def corner_xy(c: int) -> Point:
        ux, uy = _CORNER_UNITS[c]
        return (ux * size, uy * size)

    def centroid(white: bool) -> Optional[Point]:
        pts = [p.board_point for p in placements if p.cell is not None and p.symbol
               and p.symbol.isupper() == white]
        if not pts:
            return None
        scale = size / BOARD_PX
        return (sum(x for x, _ in pts) / len(pts) * scale, sum(y for _, y in pts) / len(pts) * scale)

    def dist(c: int, pt: Point) -> float:
        x, y = corner_xy(c)
        return (x - pt[0]) ** 2 + (y - pt[1]) ** 2

    white, black = centroid(True), centroid(False)
    if white is not None:
        return min(pair, key=lambda c: dist(c, white))
    if black is not None:
        return max(pair, key=lambda c: dist(c, black))
    return 2 if pair == (0, 2) else 3


@dataclass
class BoardState:
    """The reviewed board: one symbol per cell plus the problems found while placing pieces."""

    cells: dict[Cell, str] = field(default_factory=dict)
    conflicts: dict[Cell, list[Placement]] = field(default_factory=dict)
    uncertain: set[Cell] = field(default_factory=set)
    off_board: list[Placement] = field(default_factory=list)
    edited: set[Cell] = field(default_factory=set)
    winners: dict[Cell, Placement] = field(default_factory=dict)

    @classmethod
    def from_placements(cls, placements: list[Placement], min_conf: float = 0.5,
                        min_edge: float = 0.15) -> "BoardState":
        """One piece per cell (highest confidence wins, the others are conflicts); low confidence or
        a foot near a square edge marks the cell uncertain."""
        state = cls()
        by_cell: dict[Cell, list[Placement]] = {}
        for p in placements:
            if p.symbol is None:
                continue
            if p.cell is None:
                state.off_board.append(p)
            else:
                by_cell.setdefault(p.cell, []).append(p)
        for cell, items in by_cell.items():
            items.sort(key=lambda p: p.conf, reverse=True)
            best = items[0]
            state.cells[cell] = best.symbol
            state.winners[cell] = best
            if len(items) > 1:
                state.conflicts[cell] = items
            if best.conf < min_conf or best.edge < min_edge:
                state.uncertain.add(cell)
        return state

    def set_cell(self, cell: Cell, symbol: Optional[str]) -> None:
        """User correction (``None`` = empty). The cell is marked as edited (no longer a problem)."""
        if symbol is None:
            self.cells.pop(cell, None)
        else:
            if symbol not in SYMBOL_NAMES:
                raise ValueError(f"Unknown piece symbol {symbol!r}")
            self.cells[cell] = symbol
        self.edited.add(cell)
        self.uncertain.discard(cell)
        self.conflicts.pop(cell, None)

    def squares(self, a1_corner: int) -> dict[str, str]:
        """``{"e1": "K", …}`` for FEN and diagram."""
        return {square_name(cell, a1_corner): symbol for cell, symbol in self.cells.items()}

    def status(self, placement: Placement) -> str:
        """``ok``, ``uncertain`` or ``problem`` (conflict loser / off-board) for drawing."""
        if placement.cell is None or placement.symbol is None:
            return "problem"
        winner = self.winners.get(placement.cell)
        if placement.cell in self.edited:
            return "ok" if winner is placement else "problem"
        if winner is not placement:
            return "problem"
        if placement.cell in self.uncertain or placement.cell in self.conflicts:
            return "uncertain"
        return "ok"
