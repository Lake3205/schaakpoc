"""Controller of one garden chess scan (Qt independent): one method per user action.

Used by the scan screen (:mod:`schaakgui.scan_dialog`) and the command line (``scan.py``)::

    session = ScanSession(ImageTester("pieces.pt"), BoardFinder("board.pt"))
    session.load_photo("photo.jpg")
    session.auto_corners()          # or session.set_corners([...]) / move_corner(i, point)
    session.detect_pieces(0.25)     # slow: run it in a worker thread
    session.suggest_orientation()   # or set_a1_corner(i) / rotate()
    session.set_square("e4", "P")   # corrections
    print(session.fen())
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import yaml
from PIL import Image

from .board import CORNER_NAMES, BoardFinder, Cell, CornerProposal, Corners, Point, order_corners, warp_board
from .detector import ImageTester, load_picture
from .fen import GameInfo, check_position, full_fen, suggest_castling
from .keypoints import write_board_label, write_board_yaml, write_detect_yaml
from .position import (
    PIECE_CLASSES,
    SYMBOL_NAMES,
    BoardState,
    Placement,
    cell_of,
    piece_symbol,
    place_pieces,
    square_name,
    suggest_a1_corner,
)
from .render import draw_corners, draw_detections, draw_diagram, draw_feet, draw_scan, save_image

DETECT_FLOOR = 0.05  # detections are kept from this confidence so the slider can filter without re-running
SPLITS = ("train", "valid", "test")


class ScanSession:
    def __init__(self, tester: Optional[ImageTester] = None, finder: Optional[BoardFinder] = None,
                 conf: float = 0.25, lift: float = 0.08):
        self.tester = tester or ImageTester()
        self.finder = finder or BoardFinder()
        self.conf = conf
        self.lift = lift
        self.info = GameInfo()
        self.photo_path: Optional[str] = None
        self.image: Optional[Image.Image] = None
        self._reset()

    def _reset(self):
        self.corners: Optional[Corners] = None
        self.corner_method: Optional[str] = None
        self.corner_score: Optional[float] = None
        self.a1_corner = 3  # bottom-left: White at the bottom of the photo
        self.detections: list[dict] = []
        self.detected = False
        self.placements: list[Placement] = []
        self.state = BoardState()
        self.edits: dict[Cell, Optional[str]] = {}
        self.info = GameInfo()

    # ----------------------------------------------------------------- step 1: photo + corners
    def load_photo(self, path: str | Path) -> None:
        """Open a photo (EXIF rotation applied). Resets the scan."""
        image = load_picture(path)
        self.set_image(image, str(path))

    def set_image(self, image: Image.Image, path: Optional[str] = None) -> None:
        self.image = image.convert("RGB")
        self.photo_path = path
        self._reset()

    @property
    def has_photo(self) -> bool:
        return self.image is not None

    def _require_photo(self):
        if self.image is None:
            raise ValueError("Open a photo first.")

    def auto_corners(self) -> Optional[CornerProposal]:
        """Board model → OpenCV checkerboard finder; ``None`` when both fail (the user drags)."""
        self._require_photo()
        proposal = self.finder.propose(self.image)
        if proposal is not None:
            self.set_corners(proposal.corners, proposal.method)
            self.corner_score = proposal.score
        return proposal

    def set_corners(self, corners: Sequence[Point], method: str = "manual") -> None:
        """Corners from any source (sorted into image order). Re-places pieces without re-detecting."""
        self._require_photo()
        self.corners = order_corners(corners)
        self.corner_method = method
        self.corner_score = None
        self.replace()

    def default_corners(self, margin: float = 0.15) -> Corners:
        """A starting rectangle for manual corners when nothing was found."""
        self._require_photo()
        w, h = self.image.size
        return ((w * margin, h * margin), (w * (1 - margin), h * margin),
                (w * (1 - margin), h * (1 - margin)), (w * margin, h * (1 - margin)))

    def move_corner(self, index: int, point: Point) -> None:
        """Drag one handle (the order is kept while dragging)."""
        if self.corners is None:
            self.corners = self.default_corners()
        corners = list(self.corners)
        corners[index] = (float(point[0]), float(point[1]))
        self.corners = tuple(corners)  # type: ignore[assignment]
        self.corner_method = "manual"
        self.replace()

    # ----------------------------------------------------------------- step 3-4: pieces
    def detect_pieces(self, conf: Optional[float] = None) -> None:
        """Slow: run the piece detector (call it in a worker thread). Earlier box edits are lost,
        square corrections are kept."""
        self._require_photo()
        if self.tester.detector is None:
            raise ValueError("No piece detector loaded: choose or import a YOLO piece detector first.")
        if conf is not None:
            self.conf = float(conf)
        raw = self.tester.detect(self.image, min(DETECT_FLOOR, self.conf))
        self.detections = []
        for d in raw:
            label, score = d.get("detector", (d.get("label"), d.get("conf", 1.0)))
            self.detections.append({"box": tuple(float(v) for v in d["box"]), "label": str(label),
                                    "conf": float(score), "source": "model"})
        self.detected = True
        self.replace()

    def set_conf(self, conf: float) -> None:
        self.conf = float(conf)
        self.replace()

    def set_lift(self, lift: float) -> None:
        self.lift = float(lift)
        self.replace()

    def active_detections(self) -> list[dict]:
        """Detections above the confidence threshold, with their index in ``self.detections``."""
        return [dict(d, index=i) for i, d in enumerate(self.detections) if d["conf"] >= self.conf]

    def replace(self) -> None:
        """Rebuild placements and board state after corners/lift/conf/box changes (keeps user edits)."""
        if self.corners is None:
            self.placements = []
            self.state = BoardState()
        else:
            self.placements = place_pieces(self.active_detections(), self.corners, self.lift)
            self.state = BoardState.from_placements(self.placements)
        for cell, symbol in self.edits.items():
            self.state.set_cell(cell, symbol)

    # ----------------------------------------------------------------- step 4: orientation
    def warped(self) -> Optional[Image.Image]:
        if self.image is None or self.corners is None:
            return None
        return warp_board(self.image, self.corners)

    def suggest_orientation(self) -> int:
        """Dark corner nearest the white pieces becomes a1 (see :func:`suggest_a1_corner`)."""
        warped = self.warped()
        if warped is not None:
            self.a1_corner = suggest_a1_corner(warped, self.placements)
        return self.a1_corner

    def set_a1_corner(self, corner: int) -> None:
        if corner not in range(4):
            raise ValueError("a1 corner must be 0 (TL), 1 (TR), 2 (BR) or 3 (BL)")
        self.a1_corner = int(corner)

    def rotate(self) -> int:
        """⟳ Rotate: a1 moves to the next corner (clockwise)."""
        self.a1_corner = (self.a1_corner + 1) % 4
        return self.a1_corner

    # ----------------------------------------------------------------- step 5: corrections
    def set_square(self, square: str, symbol: Optional[str]) -> None:
        """Diagram click correction (``None`` = empty). Survives rotation and corner moves."""
        if symbol is not None and symbol not in SYMBOL_NAMES:
            raise ValueError(f"Unknown piece symbol {symbol!r}")
        cell = cell_of(square, self.a1_corner)
        self.edits[cell] = symbol
        self.state.set_cell(cell, symbol)

    def clear_edits(self) -> None:
        self.edits.clear()
        self.replace()

    def add_box(self, box: Sequence[float], label: str) -> int:
        """Box drawn by the user (training samples); returns its detection index."""
        x1, y1, x2, y2 = (float(v) for v in box)
        box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        self.detections.append({"box": box, "label": _class_name(label), "conf": 1.0, "source": "user"})
        self.replace()
        return len(self.detections) - 1

    def remove_detection(self, index: int) -> None:
        del self.detections[index]
        self.replace()

    def relabel(self, index: int, label: str) -> None:
        """New class for a box; the user confirmed it, so its confidence becomes 100 %."""
        self.detections[index].update(label=_class_name(label), conf=1.0, source="user")
        self.replace()

    def set_info(self, info: GameInfo) -> None:
        self.info = info

    def suggest_castling(self) -> str:
        self.info.castling = suggest_castling(self.squares())
        return self.info.castling

    # ----------------------------------------------------------------- results
    def squares(self) -> dict[str, str]:
        return self.state.squares(self.a1_corner)

    def fen(self) -> str:
        return full_fen(self.squares(), self.info)

    def warnings(self) -> list[str]:
        """python-chess checks of the position (never fixed automatically)."""
        return check_position(self.squares(), self.info)

    def problems(self) -> list[str]:
        """Scan problems to review: conflicts, uncertain squares and off-board pieces."""
        result = []
        for cell, items in sorted(self.state.conflicts.items()):
            labels = ", ".join(f"{p.label} {p.conf:.0%}" for p in items)
            result.append(f"{square_name(cell, self.a1_corner)}: {len(items)} pieces ({labels})")
        for cell in sorted(self.state.uncertain - set(self.state.conflicts)):
            p = self.state.winners.get(cell)
            detail = f"{p.label} {p.conf:.0%}" if p else ""
            result.append(f"{square_name(cell, self.a1_corner)}: uncertain {detail}".rstrip())
        for p in self.state.off_board:
            result.append(f"off-board: {p.label} {p.conf:.0%}")
        return result

    def marks(self) -> dict[str, str]:
        """``{square: "conflict" | "uncertain"}`` for the diagram."""
        marks = {square_name(c, self.a1_corner): "uncertain" for c in self.state.uncertain}
        marks.update({square_name(c, self.a1_corner): "conflict" for c in self.state.conflicts})
        return marks

    def annotated_image(self, grid: bool = True) -> Image.Image:
        self._require_photo()
        return draw_scan(self.image, self.corners, self.placements, self.state, self.a1_corner, grid)

    def diagram_image(self, size: int = 640, white_bottom: bool = True) -> Image.Image:
        return draw_diagram(self.squares(), size, white_bottom, self.marks())

    def debug_images(self) -> dict[str, Image.Image]:
        """The image of every step (for the report): ``1_corners`` … ``5_diagram``."""
        self._require_photo()
        images = {"1_corners": draw_corners(self.image, self.corners)}
        warped = self.warped()
        if warped is not None:
            images["2_straight"] = warped
        images["3_pieces"] = draw_detections(self.image, self.placements)
        if warped is not None:
            images["4_feet"] = draw_feet(warped, self.placements)
        images["5_diagram"] = self.diagram_image()
        return images

    def to_dict(self) -> dict:
        """JSON summary (photo, corners, method, orientation, FEN, warnings, problems)."""
        return {
            "photo": self.photo_path,
            "image_size": list(self.image.size) if self.image is not None else None,
            "corners": [list(p) for p in self.corners] if self.corners else None,
            "corner_method": self.corner_method,
            "a1_corner": CORNER_NAMES[self.a1_corner],
            "conf": self.conf,
            "lift": self.lift,
            "detections": len(self.active_detections()),
            "edits": {square_name(c, self.a1_corner): s for c, s in self.edits.items()},
            "game_info": self.info.to_dict(),
            "fen": self.fen(),
            "warnings": self.warnings(),
            "problems": self.problems(),
        }

    # ----------------------------------------------------------------- exports
    def export_fen(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.fen() + "\n", encoding="utf-8")
        return path

    def export_annotated(self, path: str | Path) -> Path:
        return save_image(self.annotated_image(), path)

    def export_diagram(self, path: str | Path, white_bottom: bool = True) -> Path:
        return save_image(self.diagram_image(white_bottom=white_bottom), path)

    def export_debug(self, folder: str | Path) -> list[Path]:
        return [save_image(image, Path(folder) / f"{name}.png")
                for name, image in self.debug_images().items()]

    def export_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    def export_all(self, folder: str | Path, debug: bool = False) -> dict[str, str]:
        """``fen.txt``, ``annotated.png``, ``diagram.png``, ``scan.json`` (+ step images)."""
        folder = Path(folder)
        written = {"fen": self.export_fen(folder / "fen.txt"),
                   "annotated": self.export_annotated(folder / "annotated.png"),
                   "diagram": self.export_diagram(folder / "diagram.png"),
                   "json": self.export_json(folder / "scan.json")}
        result = {k: str(v) for k, v in written.items()}
        if debug:
            result["debug"] = [str(p) for p in self.export_debug(folder)]  # type: ignore[assignment]
        return result

    # ----------------------------------------------------------------- training samples (phase C)
    def training_boxes(self) -> tuple[list[tuple[str, tuple[float, float, float, float]]], list[str]]:
        """Corrected piece boxes ``[(class name, box)]`` + notes on corrections without a box.

        A square correction relabels (or drops, for *empty*) the box that won the square; conflict
        losers are dropped; off-board pieces are kept (their boxes are still pieces)."""
        boxes, notes = [], []
        boxed_cells = set()
        for p in self.placements:
            name = _class_name(p.label) if p.symbol else None
            if name is None:
                continue
            cell = p.cell
            if cell is not None:
                winner = self.state.winners.get(cell) is p
                if cell in self.edits:
                    symbol = self.edits[cell]
                    if not winner or symbol is None:
                        continue
                    name = SYMBOL_NAMES[symbol]
                elif not winner:
                    continue
                boxed_cells.add(cell)
            boxes.append((name, p.box))
        for cell, symbol in self.edits.items():
            if symbol is not None and cell not in boxed_cells:
                notes.append(f"{square_name(cell, self.a1_corner)} ({symbol}) has no box: draw one "
                             "in box mode to include it in the training sample")
        return boxes, notes

    def save_training_sample(self, root: str | Path, split: str = "train") -> dict:
        """Write the photo + YOLO piece boxes to ``root/pieces/<split>`` and the corner label to
        ``root/board/<split>`` (+ ``data.yaml`` files). Returns the written paths and notes."""
        self._require_photo()
        if split not in SPLITS:
            raise ValueError(f"split must be one of {', '.join(SPLITS)}")
        root = Path(root)
        stem = Path(self.photo_path).stem if self.photo_path else "scan"
        stem = f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        w, h = self.image.size
        result: dict = {"stem": stem}

        boxes, notes = self.training_boxes()
        pieces = root / "pieces"
        names = _dataset_names(pieces) or list(PIECE_CLASSES)
        ids = {_class_name(n) or n: i for i, n in enumerate(names)}
        lines = []
        for name, (x1, y1, x2, y2) in boxes:
            if name not in ids:
                notes.append(f"class {name} is not in {pieces / 'data.yaml'}: box skipped")
                continue
            x1, x2 = max(0.0, x1), min(float(w), x2)
            y1, y2 = max(0.0, y1), min(float(h), y2)
            if x2 <= x1 or y2 <= y1:
                continue
            lines.append(f"{ids[name]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                         f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        image_path = save_image(self.image, pieces / split / "images" / f"{stem}.jpg")
        label_path = pieces / split / "labels" / f"{stem}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
        if not (pieces / "data.yaml").is_file() or _dataset_names(pieces) is None:
            write_detect_yaml(pieces, names)
        else:
            _add_split(pieces / "data.yaml", split)
        result.update(pieces_image=str(image_path), pieces_label=str(label_path), boxes=len(lines))

        if self.corners is not None:
            board = root / "board"
            board_image = save_image(self.image, board / split / "images" / f"{stem}.jpg")
            board_label = board / split / "labels" / f"{stem}.txt"
            write_board_label(board_label, self.corners, w, h)
            write_board_yaml(board)
            result.update(board_image=str(board_image), board_label=str(board_label))
        else:
            notes.append("No corners: no board-corner label written")
        result["notes"] = notes
        return result


def _class_name(label: str) -> Optional[str]:
    """Unified class name (``white-knight``) of a label or FEN symbol."""
    if label in SYMBOL_NAMES:
        return SYMBOL_NAMES[label]
    symbol = piece_symbol(label)
    return SYMBOL_NAMES[symbol] if symbol else None


def _dataset_names(root: Path) -> Optional[list[str]]:
    path = root / "data.yaml"
    if not path.is_file():
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    names = data.get("names") if isinstance(data, dict) else None
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    return [str(n) for n in names] if isinstance(names, list) and names else None


def _add_split(path: Path, split: str) -> None:
    """Make sure an existing ``data.yaml`` lists a newly written split."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    key = {"valid": "val"}.get(split, split)
    if key not in data:
        data[key] = f"{split}/images"
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
