"""Images of every scan step (Qt independent): corners, feet, annotated photo and board diagram."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from PIL import Image, ImageDraw, ImageFont

from .board import BOARD_PX, Corners, grid_lines
from .position import FILES, BoardState, Placement, square_name

GREEN = (0, 200, 0)
ORANGE = (255, 150, 0)
RED = (230, 30, 30)
STATUS_COLORS = {"ok": GREEN, "uncertain": ORANGE, "problem": RED}
MARK_COLORS = {"uncertain": ORANGE, "conflict": RED, "problem": RED}
LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
GLYPHS = {"K": "♚", "Q": "♛", "R": "♜", "B": "♝", "N": "♞", "P": "♟"}
_FONT_NAMES = ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "Arial Unicode.ttf", "seguisym.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


def font(size: int, glyphs: bool = False) -> ImageFont.ImageFont:
    """A TrueType font of ``size`` px (one that has chess glyphs when ``glyphs``), else the default."""
    for name in _FONT_NAMES:
        try:
            candidate = ImageFont.truetype(name, size)
        except OSError:
            continue
        if not glyphs or _has_glyph(candidate, "♚"):
            return candidate
    if glyphs:
        return None  # type: ignore[return-value]
    try:
        return ImageFont.load_default(size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _has_glyph(fnt, char: str) -> bool:
    try:
        mask = fnt.getmask(char)
        box = mask.getbbox()
        return box is not None and box[2] - box[0] > 1
    except Exception:  # noqa: BLE001 - any font issue means "no glyph"
        return False


def _rgb(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB").copy()
    return Image.fromarray(image).convert("RGB")


def _line_width(image: Image.Image) -> int:
    return max(2, round(max(image.size) / 400))


def _text(draw: ImageDraw.ImageDraw, xy, text: str, color, fnt) -> None:
    """Label with a filled background so it stays readable on any photo."""
    x, y = xy
    box = draw.textbbox((x, y), text, font=fnt)
    pad = 2
    draw.rectangle((box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad), fill=color)
    draw.text((x, y), text, fill=(255, 255, 255) if sum(color) < 500 else (0, 0, 0), font=fnt)


def _outline(draw: ImageDraw.ImageDraw, corners: Corners, color, width: int) -> None:
    pts = [tuple(map(float, p)) for p in corners]
    draw.line(pts + [pts[0]], fill=color, width=width)


def draw_corners(image, corners: Optional[Corners], color=GREEN,
                 labels: Sequence[str] = ("TL", "TR", "BR", "BL")) -> Image.Image:
    """Corner overlay: board outline + numbered corner dots (step 1, board model review)."""
    out = _rgb(image)
    if not corners:
        return out
    draw = ImageDraw.Draw(out)
    width = _line_width(out)
    fnt = font(max(12, width * 7))
    _outline(draw, corners, color, width)
    r = width * 3
    for (x, y), label in zip(corners, labels):
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=(0, 0, 0))
        _text(draw, (x + r + 2, y + r + 2), label, color, fnt)
    return out


def draw_grid(image, corners: Corners, color=GREEN, width: Optional[int] = None) -> Image.Image:
    out = _rgb(image)
    draw = ImageDraw.Draw(out)
    width = width or max(1, _line_width(out) // 2)
    for a, b in grid_lines(corners):
        draw.line((a, b), fill=color, width=width)
    return out


def draw_feet(warped, placements: list[Placement], size: int = BOARD_PX) -> Image.Image:
    """Straight board with the grid and the projected foot point of every detection (step 4)."""
    out = _rgb(warped)
    draw = ImageDraw.Draw(out)
    side = out.size[0]
    square = side / 8
    for i in range(9):
        t = i * square
        draw.line(((t, 0), (t, side)), fill=GREEN, width=1)
        draw.line(((0, t), (side, t)), fill=GREEN, width=1)
    fnt = font(max(12, side // 50))
    for p in placements:
        x, y = (p.board_point[0] * side / size, p.board_point[1] * side / size)
        color = GREEN if p.cell is not None else RED
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color, outline=(0, 0, 0))
        if p.symbol:
            draw.text((x + 8, y - 8), p.symbol, fill=color, font=fnt)
    return out


def draw_detections(image, placements: list[Placement]) -> Image.Image:
    """Raw piece boxes with label and confidence (step 3)."""
    out = _rgb(image)
    draw = ImageDraw.Draw(out)
    width = _line_width(out)
    fnt = font(max(12, width * 6))
    for p in placements:
        draw.rectangle(p.box, outline=GREEN, width=width)
        _text(draw, (p.box[0], max(0, p.box[1] - width * 8)), f"{p.label} {p.conf:.0%}", GREEN, fnt)
    return out


def draw_scan(image, corners: Optional[Corners], placements: list[Placement],
              state: Optional[BoardState], a1_corner: int, grid: bool = True) -> Image.Image:
    """Annotated photo at original resolution: outline, grid, boxes with ``label / square / %``.

    Colour = status: green ok, orange uncertain, red conflict/off-board."""
    out = _rgb(image)
    draw = ImageDraw.Draw(out)
    width = _line_width(out)
    fnt = font(max(12, width * 6))
    if corners:
        if grid:
            for a, b in grid_lines(corners):
                draw.line((a, b), fill=(0, 160, 255), width=max(1, width // 2))
        _outline(draw, corners, (0, 160, 255), width)
        a1 = corners[a1_corner]
        r = width * 3
        draw.ellipse((a1[0] - r, a1[1] - r, a1[0] + r, a1[1] + r), fill=(0, 160, 255))
        _text(draw, (a1[0] + r, a1[1] + r), "a1", (0, 160, 255), fnt)
    for p in placements:
        status = state.status(p) if state is not None else "ok"
        color = STATUS_COLORS[status]
        draw.rectangle(p.box, outline=color, width=width)
        square = square_name(p.cell, a1_corner) if p.cell is not None else "off-board"
        fx, fy = p.foot
        draw.ellipse((fx - width * 2, fy - width * 2, fx + width * 2, fy + width * 2), fill=color)
        _text(draw, (p.box[0], max(0, p.box[1] - width * 8)),
              f"{p.label} / {square} / {p.conf:.0%}", color, fnt)
    return out


def draw_diagram(squares: dict[str, str], size: int = 640, white_bottom: bool = True,
                 marks: Optional[dict[str, str]] = None) -> Image.Image:
    """Clean top-down diagram (piece glyph if a font has it, else the letter) with status marks."""
    marks = marks or {}
    margin = size // 20
    square = (size - 2 * margin) / 8
    out = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(out)
    glyph_font = font(int(square * 0.8), glyphs=True)
    letter_font = font(int(square * 0.55))
    coord_font = font(max(10, margin // 2))
    for col in range(8):
        for row in range(8):
            file = col if white_bottom else 7 - col
            rank = 7 - row if white_bottom else row
            name = f"{FILES[file]}{rank + 1}"
            x0, y0 = margin + col * square, margin + row * square
            box = (x0, y0, x0 + square, y0 + square)
            dark = (file + rank) % 2 == 0
            draw.rectangle(box, fill=DARK_SQUARE if dark else LIGHT_SQUARE)
            mark = marks.get(name)
            if mark in MARK_COLORS:
                draw.rectangle(box, outline=MARK_COLORS[mark], width=max(3, int(square / 14)))
            symbol = squares.get(name)
            if symbol:
                cx, cy = x0 + square / 2, y0 + square / 2
                white = symbol.isupper()
                if glyph_font is not None:
                    draw.text((cx, cy), GLYPHS[symbol.upper()], font=glyph_font, anchor="mm",
                              fill=(255, 255, 255) if white else (0, 0, 0),
                              stroke_width=max(1, int(square / 40)), stroke_fill=(0, 0, 0))
                else:
                    draw.text((cx, cy), symbol, font=letter_font, anchor="mm",
                              fill=(255, 255, 255) if white else (0, 0, 0),
                              stroke_width=2, stroke_fill=(0, 0, 0) if white else (255, 255, 255))
    for i in range(8):
        file = FILES[i if white_bottom else 7 - i]
        rank = str(8 - i if white_bottom else i + 1)
        x = margin + (i + 0.5) * square
        y = margin + (i + 0.5) * square
        draw.text((x, size - margin / 2), file, font=coord_font, anchor="mm", fill=(0, 0, 0))
        draw.text((margin / 2, y), rank, font=coord_font, anchor="mm", fill=(0, 0, 0))
    return out


def save_image(image: Image.Image, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path
