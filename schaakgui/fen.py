"""FEN notation and position checks (scan step 5, Qt independent), built on python-chess."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import chess

from .position import FILES


@dataclass
class GameInfo:
    """The FEN fields a photo cannot show; filled in by the user (castling can be suggested)."""

    side: str = "w"
    castling: str = "-"
    en_passant: str = "-"
    halfmove: int = 0
    fullmove: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


def placement_fen(squares: dict[str, str]) -> str:
    """Piece-placement field only, e.g. ``rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR``."""
    rows = []
    for rank in range(8, 0, -1):
        row, empty = "", 0
        for file in FILES:
            symbol = squares.get(f"{file}{rank}")
            if symbol:
                if empty:
                    row += str(empty)
                    empty = 0
                row += symbol
            else:
                empty += 1
        if empty:
            row += str(empty)
        rows.append(row)
    return "/".join(rows)


def full_fen(squares: dict[str, str], info: GameInfo | None = None) -> str:
    """All six FEN fields."""
    info = info or GameInfo()
    return (f"{placement_fen(squares)} {info.side or 'w'} {info.castling or '-'} "
            f"{info.en_passant or '-'} {int(info.halfmove)} {int(info.fullmove)}")


def suggest_castling(squares: dict[str, str]) -> str:
    """Castling rights that are still possible from the piece positions (kings/rooks at home)."""
    rights = ""
    if squares.get("e1") == "K":
        if squares.get("h1") == "R":
            rights += "K"
        if squares.get("a1") == "R":
            rights += "Q"
    if squares.get("e8") == "k":
        if squares.get("h8") == "r":
            rights += "k"
        if squares.get("a8") == "r":
            rights += "q"
    return rights or "-"


_STATUS_MESSAGES = [
    (chess.STATUS_NO_WHITE_KING, "White has no king"),
    (chess.STATUS_NO_BLACK_KING, "Black has no king"),
    (chess.STATUS_TOO_MANY_KINGS, "More than one king of a colour"),
    (chess.STATUS_TOO_MANY_WHITE_PAWNS, "White has more than 8 pawns"),
    (chess.STATUS_TOO_MANY_BLACK_PAWNS, "Black has more than 8 pawns"),
    (chess.STATUS_PAWNS_ON_BACKRANK, "Pawn on rank 1 or 8"),
    (chess.STATUS_TOO_MANY_WHITE_PIECES, "White has more than 16 pieces"),
    (chess.STATUS_TOO_MANY_BLACK_PIECES, "Black has more than 16 pieces"),
    (chess.STATUS_BAD_CASTLING_RIGHTS, "Castling rights do not match the kings/rooks"),
    (chess.STATUS_INVALID_EP_SQUARE, "Invalid en-passant square"),
    (chess.STATUS_OPPOSITE_CHECK, "The side not to move is in check"),
    (chess.STATUS_TOO_MANY_CHECKERS, "Too many pieces give check"),
    (chess.STATUS_IMPOSSIBLE_CHECK, "Impossible check"),
]


def check_position(squares: dict[str, str], info: GameInfo | None = None) -> list[str]:
    """Human-readable warnings for an impossible position; the position itself is not changed."""
    try:
        board = chess.Board(full_fen(squares, info))
    except ValueError as exc:
        return [f"Invalid FEN: {exc}"]
    status = board.status()
    return [message for flag, message in _STATUS_MESSAGES if status & flag]


def squares_from_fen(fen: str) -> dict[str, str]:
    """``{"e1": "K", …}`` of a FEN (only the placement field is needed)."""
    board = chess.Board(None)
    board.set_board_fen(fen.split()[0])
    return {chess.square_name(sq): piece.symbol() for sq, piece in board.piece_map().items()}
