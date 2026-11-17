"""Scan board screen: photo → corners → pieces → orientation → review & export (thin PyQt6 layer).

All scan logic lives in :class:`schaakgui.scan_session.ScanSession`; this dialog only shows its state
and forwards the user's actions (dragging corners, drawing boxes, clicking diagram squares)."""
from __future__ import annotations

import html
from pathlib import Path
from typing import Callable, Optional, Sequence

from PIL import Image
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QFont, QGuiApplication, QImage, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .board import CORNER_NAMES, Corners
from .fen import GameInfo
from .position import PIECE_SYMBOLS
from .render import GLYPHS, diagram_square
from .review_dialog import PICTURE_FILTER, model_label
from .scan_session import SPLITS, ScanSession
from .worker import start_call

STEPS = ("1 Photo", "2 Corners", "3 Pieces", "4 Orientation", "5 Review & Export")
CORNER_COLORS = ("#e53935", "#fb8c00", "#43a047", "#1e88e5")  # TL, TR, BR, BL handles
CORNER_LONG = {"TL": "top-left", "TR": "top-right", "BR": "bottom-right", "BL": "bottom-left"}
DISPLAY_MAX = 1600  # overlay images are shrunk to this size for drawing speed
DIAGRAM_PX = 640


def pil_to_qimage(image: Image.Image) -> QImage:
    rgb = image.convert("RGB")
    data = rgb.tobytes("raw", "RGB")
    return QImage(data, rgb.width, rgb.height, 3 * rgb.width, QImage.Format.Format_RGB888).copy()


def _shrink(image: Image.Image, max_side: int = DISPLAY_MAX) -> Image.Image:
    if max(image.size) <= max_side:
        return image
    image = image.copy()
    image.thumbnail((max_side, max_side))
    return image


class PhotoView(QWidget):
    """The photo with the scan overlay, draggable corner handles and a box-drawing mode.

    Coordinates in signals are original photo pixels; the photo keeps its aspect ratio."""

    cornerMoved = pyqtSignal(int, float, float)
    boxDrawn = pyqtSignal(float, float, float, float)
    detectionClicked = pyqtSignal(int)
    HANDLE = 9  # handle radius in widget pixels

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(560, 420)
        self.setMouseTracking(True)
        self.photo_size: Optional[tuple[int, int]] = None
        self.overlay: Optional[QImage] = None
        self.corners: Optional[list[tuple[float, float]]] = None
        self.boxes: list[tuple[int, tuple[float, float, float, float]]] = []
        self.selected: Optional[int] = None
        self.mode = "corners"
        self._drag: Optional[int] = None
        self._box_start: Optional[tuple[float, float]] = None
        self._box_end: Optional[tuple[float, float]] = None

    # ----------------------------------------------------------------- state
    def set_photo(self, size: Optional[tuple[int, int]]):
        self.photo_size = size
        self.overlay = None
        self.corners = None
        self.boxes = []
        self.selected = None
        self.update()

    def set_overlay(self, image: Optional[Image.Image]):
        self.overlay = pil_to_qimage(_shrink(image)) if image is not None else None
        self.update()

    def set_corners(self, corners: Optional[Corners]):
        self.corners = [tuple(map(float, p)) for p in corners] if corners else None
        self.update()

    def set_boxes(self, boxes):
        self.boxes = list(boxes)
        self.update()

    def set_mode(self, mode: str):
        if mode not in ("corners", "box"):
            raise ValueError(mode)
        self.mode = mode
        self.setCursor(Qt.CursorShape.CrossCursor if mode == "box" else Qt.CursorShape.ArrowCursor)

    # ----------------------------------------------------------------- coordinates
    def _target(self) -> QRectF:
        if not self.photo_size:
            return QRectF(0, 0, self.width(), self.height())
        w, h = self.photo_size
        scale = min(self.width() / w, self.height() / h)
        tw, th = w * scale, h * scale
        return QRectF((self.width() - tw) / 2, (self.height() - th) / 2, tw, th)

    def _to_photo(self, point: QPointF) -> tuple[float, float]:
        rect = self._target()
        w, h = self.photo_size or (1, 1)
        return ((point.x() - rect.x()) * w / rect.width(), (point.y() - rect.y()) * h / rect.height())

    def _to_widget(self, x: float, y: float) -> QPointF:
        rect = self._target()
        w, h = self.photo_size or (1, 1)
        return QPointF(rect.x() + x * rect.width() / w, rect.y() + y * rect.height() / h)

    def _handle_at(self, pos: QPointF) -> Optional[int]:
        if not self.corners:
            return None
        best, best_d = None, (self.HANDLE + 4) ** 2
        for i, (x, y) in enumerate(self.corners):
            p = self._to_widget(x, y)
            d = (p.x() - pos.x()) ** 2 + (p.y() - pos.y()) ** 2
            if d <= best_d:
                best, best_d = i, d
        return best

    def _box_at(self, x: float, y: float) -> Optional[int]:
        hits = [(abs((b[2] - b[0]) * (b[3] - b[1])), i) for i, b in self.boxes
                if b[0] <= x <= b[2] and b[1] <= y <= b[3]]
        return min(hits)[1] if hits else None

    # ----------------------------------------------------------------- painting
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#263238"))
        if not self.photo_size:
            painter.setPen(QColor("#cfd8dc"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "📷  Open a photo of the garden chess board\n"
                             "(camera high, all 4 board corners visible)")
            return
        rect = self._target()
        if self.overlay is not None:
            painter.drawImage(rect, self.overlay)
        if self.selected is not None:
            for i, (x1, y1, x2, y2) in self.boxes:
                if i == self.selected:
                    painter.setPen(QPen(QColor("#ffeb3b"), 3, Qt.PenStyle.DashLine))
                    painter.drawRect(QRectF(self._to_widget(x1, y1), self._to_widget(x2, y2)))
        if self.corners:
            points = [self._to_widget(x, y) for x, y in self.corners]
            if self._drag is not None:  # live outline while dragging (the grid follows on release)
                painter.setPen(QPen(QColor("#ffeb3b"), 2))
                for i in range(4):
                    painter.drawLine(points[i], points[(i + 1) % 4])
            for i, p in enumerate(points):
                painter.setPen(QPen(QColor("white"), 2))
                painter.setBrush(QColor(CORNER_COLORS[i]))
                painter.drawEllipse(p, self.HANDLE, self.HANDLE)
                painter.drawText(p + QPointF(self.HANDLE + 2, -self.HANDLE), CORNER_NAMES[i])
        if self._box_start and self._box_end:
            painter.setPen(QPen(QColor("#00e5ff"), 2, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRectF(self._to_widget(*self._box_start), self._to_widget(*self._box_end)))

    # ----------------------------------------------------------------- mouse
    def mousePressEvent(self, event):
        if not self.photo_size or event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.position()
        x, y = self._to_photo(pos)
        if self.mode == "corners":
            handle = self._handle_at(pos)
            if handle is not None:
                self._drag = handle
                return
            hit = self._box_at(x, y)
            if hit is not None:
                self.detectionClicked.emit(hit)
            return
        self._box_start = self._box_end = (x, y)

    def mouseMoveEvent(self, event):
        if self._drag is not None and self.corners:
            x, y = self._to_photo(event.position())
            w, h = self.photo_size
            self.corners[self._drag] = (min(max(x, 0.0), float(w)), min(max(y, 0.0), float(h)))
            self.update()
        elif self._box_start is not None:
            self._box_end = self._to_photo(event.position())
            self.update()

    def mouseReleaseEvent(self, event):
        if self._drag is not None:
            index, self._drag = self._drag, None
            x, y = self.corners[index]
            self.update()
            self.cornerMoved.emit(index, x, y)
        elif self._box_start is not None:
            (x1, y1), (x2, y2) = self._box_start, self._to_photo(event.position())
            self._box_start = self._box_end = None
            self.update()
            if abs(x2 - x1) >= 4 and abs(y2 - y1) >= 4:
                self.boxDrawn.emit(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            else:
                hit = self._box_at(x1, y1)
                if hit is not None:
                    self.detectionClicked.emit(hit)


class BoardDiagram(QWidget):
    """The 2D diagram; clicking a square emits its name (``e4``)."""

    squareClicked = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 320)
        self.image: Optional[QImage] = None
        self.white_bottom = True
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Click a square to correct it")

    def set_image(self, image: Image.Image, white_bottom: bool = True):
        self.image = pil_to_qimage(image)
        self.white_bottom = white_bottom
        self.update()

    def _rect(self) -> QRectF:
        side = min(self.width(), self.height())
        return QRectF((self.width() - side) / 2, (self.height() - side) / 2, side, side)

    def square_at(self, pos: QPointF) -> Optional[str]:
        rect = self._rect()
        if rect.width() <= 0:
            return None
        scale = DIAGRAM_PX / rect.width()
        return diagram_square((pos.x() - rect.x()) * scale, (pos.y() - rect.y()) * scale, DIAGRAM_PX,
                              self.white_bottom)

    def widget_point(self, square: str) -> QPointF:
        """Centre of a square in widget coordinates (used by tests and keyboard focus)."""
        files = "abcdefgh"
        file, rank = files.index(square[0]), int(square[1]) - 1
        col = file if self.white_bottom else 7 - file
        row = 7 - rank if self.white_bottom else rank
        rect = self._rect()
        margin = rect.width() / 20
        size = (rect.width() - 2 * margin) / 8
        return QPointF(rect.x() + margin + (col + 0.5) * size, rect.y() + margin + (row + 0.5) * size)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        if self.image is not None:
            painter.drawImage(self._rect(), self.image)

    def mousePressEvent(self, event):
        square = self.square_at(event.position())
        if square is not None:
            self.squareClicked.emit(square)


class PieceMenu(QMenu):
    """Empty + the 12 pieces. :meth:`choose` returns ``None`` (empty), a symbol, or ``...`` (cancel)."""

    def __init__(self, parent=None, title: str = "Set square", allow_empty: bool = True):
        super().__init__(title, parent)
        if allow_empty:
            action = self.addAction("∅  Empty")
            action.setData("")
            self.addSeparator()
        for color in ("white", "black"):
            for name in ("king", "queen", "rook", "bishop", "knight", "pawn"):
                symbol = PIECE_SYMBOLS[f"{color}-{name}"]
                action = self.addAction(f"{GLYPHS[symbol.upper()]}  {color} {name}  ({symbol})")
                action.setData(symbol)
            self.addSeparator()

    def choose(self, pos) -> object:
        action = self.exec(pos)
        if action is None:
            return ...
        data = action.data()
        return data or None


class GameInfoBox(QGroupBox):
    """Side to move, castling rights, en passant square and the move clocks."""

    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Game info", parent)
        self.side = QComboBox()
        self.side.addItem("White to move", "w")
        self.side.addItem("Black to move", "b")
        self.castling = {}
        castle_row = QHBoxLayout()
        for flag, tip in (("K", "White O-O"), ("Q", "White O-O-O"), ("k", "Black O-O"), ("q", "Black O-O-O")):
            box = QCheckBox(flag)
            box.setToolTip(tip)
            box.toggled.connect(self.changed)
            self.castling[flag] = box
            castle_row.addWidget(box)
        self.suggest_button = QPushButton("Suggest")
        self.suggest_button.setToolTip("Allow castling where king and rook stand on their start squares")
        castle_row.addWidget(self.suggest_button)
        self.en_passant = QLineEdit("-")
        self.en_passant.setMaximumWidth(60)
        self.halfmove = QSpinBox()
        self.halfmove.setRange(0, 999)
        self.fullmove = QSpinBox()
        self.fullmove.setRange(1, 999)
        clocks = QHBoxLayout()
        clocks.addWidget(QLabel("en passant"))
        clocks.addWidget(self.en_passant)
        clocks.addWidget(QLabel("halfmove"))
        clocks.addWidget(self.halfmove)
        clocks.addWidget(QLabel("move"))
        clocks.addWidget(self.fullmove)
        form = QFormLayout(self)
        form.addRow(self.side)
        form.addRow("Castling", castle_row)
        form.addRow(clocks)
        self.side.currentIndexChanged.connect(self.changed)
        self.en_passant.editingFinished.connect(self.changed)
        self.halfmove.valueChanged.connect(self.changed)
        self.fullmove.valueChanged.connect(self.changed)

    def info(self) -> GameInfo:
        castling = "".join(f for f, box in self.castling.items() if box.isChecked()) or "-"
        ep = self.en_passant.text().strip().lower() or "-"
        if ep != "-" and not (len(ep) == 2 and ep[0] in "abcdefgh" and ep[1] in "36"):
            ep = "-"
        return GameInfo(side=self.side.currentData(), castling=castling, en_passant=ep,
                        halfmove=self.halfmove.value(), fullmove=self.fullmove.value())

    def set_info(self, info: GameInfo):
        widgets = [self.side, self.en_passant, self.halfmove, self.fullmove, *self.castling.values()]
        for w in widgets:
            w.blockSignals(True)
        self.side.setCurrentIndex(max(0, self.side.findData(info.side)))
        for flag, box in self.castling.items():
            box.setChecked(flag in info.castling)
        self.en_passant.setText(info.en_passant)
        self.halfmove.setValue(info.halfmove)
        self.fullmove.setValue(info.fullmove)
        for w in widgets:
            w.blockSignals(False)


class ScanDialog(QDialog):
    """Garden chess scanner: five explainable steps with manual corrections at every step."""

    def __init__(self, session: ScanSession, detectors: Sequence[str] = (),
                 board_models: Sequence[str] = (), sample_root: Optional[str | Path] = None,
                 import_model: Optional[Callable[[str, str], str]] = None, parent=None):
        super().__init__(parent)
        self.session = session
        self.sample_root = Path(sample_root) if sample_root else None
        self.import_callback = import_model
        self.white_bottom = True
        self.orientation_confirmed = False
        self.thread = None
        self.worker = None
        self.last_error: Optional[str] = None
        self.setWindowTitle("Scan garden chess board")
        self.setModal(True)
        self.setAcceptDrops(True)
        self.resize(1400, 900)

        # step bar
        self.step_labels = []
        steps = QHBoxLayout()
        for i, name in enumerate(STEPS):
            label = QLabel(name, alignment=Qt.AlignmentFlag.AlignCenter)
            self.step_labels.append(label)
            steps.addWidget(label, 1)
            if i < len(STEPS) - 1:
                steps.addWidget(QLabel("→"))

        # models
        self.pieces_combo = QComboBox()
        for path in detectors:
            self.pieces_combo.addItem(model_label(path), path)
        if session.tester.detector is not None and getattr(session.tester, "detector_path", None) is None:
            self.pieces_combo.addItem("(loaded detector)", None)
        self.pieces_combo.currentIndexChanged.connect(self.on_pieces_model_changed)
        self.board_combo = QComboBox()
        self.board_combo.addItem("(none – OpenCV checkerboard / drag the corners)", None)
        for path in board_models:
            self.board_combo.addItem(model_label(path), path)
        self.board_combo.currentIndexChanged.connect(self.on_board_model_changed)
        add_pieces = QPushButton("➕ Add…")
        add_pieces.setToolTip("Add YOLO piece detector weights (.pt) to this project")
        add_pieces.clicked.connect(lambda: self.choose_import("detect"))
        add_board = QPushButton("➕ Add…")
        add_board.setToolTip("Add a YOLO-pose board corner model (.pt, 4 keypoints) to this project")
        add_board.clicked.connect(lambda: self.choose_import("board"))
        for button in (add_pieces, add_board):
            button.setEnabled(import_model is not None)
        models = QGridLayout()
        models.addWidget(QLabel("<b>Piece detector</b>"), 0, 0)
        models.addWidget(self.pieces_combo, 0, 1)
        models.addWidget(add_pieces, 0, 2)
        models.addWidget(QLabel("<b>Board corners</b>"), 0, 3)
        models.addWidget(self.board_combo, 0, 4)
        models.addWidget(add_board, 0, 5)
        models.setColumnStretch(1, 1)
        models.setColumnStretch(4, 1)

        # photo side
        self.view = PhotoView()
        self.view.cornerMoved.connect(self.on_corner_moved)
        self.view.boxDrawn.connect(self.on_box_drawn)
        self.view.detectionClicked.connect(self.on_detection_clicked)
        open_button = QPushButton("📂 Open photo…")
        open_button.clicked.connect(self.choose_photo)
        self.find_button = QPushButton("🔍 Find corners")
        self.find_button.setToolTip("Board model → OpenCV checkerboard; if both fail, drag the 4 handles")
        self.find_button.clicked.connect(self.find_corners)
        self.corner_mode = QPushButton("✥ Drag corners")
        self.box_mode = QPushButton("▭ Draw box")
        self.box_mode.setToolTip("Draw a box around a missed piece (used for training samples)")
        self.mode_group = QButtonGroup(self)
        for i, button in enumerate((self.corner_mode, self.box_mode)):
            button.setCheckable(True)
            self.mode_group.addButton(button, i)
        self.corner_mode.setChecked(True)
        self.mode_group.idClicked.connect(lambda i: self.view.set_mode("box" if i else "corners"))
        self.detect_button = QPushButton("♟ Detect pieces")
        self.detect_button.clicked.connect(self.detect_pieces)
        photo_buttons = QHBoxLayout()
        for w in (open_button, self.find_button, self.corner_mode, self.box_mode, self.detect_button):
            photo_buttons.addWidget(w)
        photo_buttons.addStretch()

        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(5, 95)
        self.conf_slider.setValue(int(round(session.conf * 100)))
        self.conf_slider.setToolTip("Minimum piece confidence (no new detection needed)")
        self.conf_label = QLabel()
        self.conf_slider.valueChanged.connect(self.on_conf_changed)
        self.lift_slider = QSlider(Qt.Orientation.Horizontal)
        self.lift_slider.setRange(0, 30)
        self.lift_slider.setValue(int(round(session.lift * 100)))
        self.lift_slider.setToolTip("Foot point lift: how far above the bottom of a box the piece stands "
                                    "(raise for tall pieces whose base looks lower)")
        self.lift_label = QLabel()
        self.lift_slider.valueChanged.connect(self.on_lift_changed)
        sliders = QHBoxLayout()
        sliders.addWidget(QLabel("Confidence"))
        sliders.addWidget(self.conf_slider, 1)
        sliders.addWidget(self.conf_label)
        sliders.addSpacing(16)
        sliders.addWidget(QLabel("Foot lift"))
        sliders.addWidget(self.lift_slider, 1)
        sliders.addWidget(self.lift_label)

        left = QVBoxLayout()
        left.addLayout(photo_buttons)
        left.addWidget(self.view, 1)
        left.addLayout(sliders)

        # board side
        self.diagram = BoardDiagram()
        self.diagram.squareClicked.connect(self.on_square_clicked)
        self.orientation_label = QLabel()
        rotate_button = QPushButton("⟳ Rotate a1")
        rotate_button.setToolTip("Move a1 to the next corner of the photo (no new detection)")
        rotate_button.clicked.connect(self.rotate)
        suggest_button = QPushButton("💡 Suggest")
        suggest_button.setToolTip("a1 = the dark corner on White's side")
        suggest_button.clicked.connect(self.suggest_orientation)
        ok_button = QPushButton("✔ Orientation OK")
        ok_button.clicked.connect(self.confirm_orientation)
        flip_button = QPushButton("⇅ Flip diagram")
        flip_button.clicked.connect(self.flip_diagram)
        orientation = QHBoxLayout()
        for w in (rotate_button, suggest_button, ok_button, flip_button):
            orientation.addWidget(w)
        self.info_box = GameInfoBox()
        self.info_box.changed.connect(self.on_info_changed)
        self.info_box.suggest_button.clicked.connect(self.suggest_castling)
        self.fen_edit = QLineEdit()
        self.fen_edit.setReadOnly(True)
        self.fen_edit.setFont(QFont("monospace"))
        copy_button = QPushButton("📋 Copy")
        copy_button.clicked.connect(self.copy_fen)
        fen_row = QHBoxLayout()
        fen_row.addWidget(QLabel("<b>FEN</b>"))
        fen_row.addWidget(self.fen_edit, 1)
        fen_row.addWidget(copy_button)
        self.problems = QListWidget()
        self.problems.setMaximumHeight(150)
        self.problems.setWordWrap(True)

        export_buttons = QGridLayout()
        exports = [("💾 FEN…", self.save_fen), ("🖼 Annotated photo…", self.save_annotated),
                   ("♜ Diagram…", self.save_diagram), ("🔎 Step images…", self.save_debug),
                   ("📦 Export all…", self.export_all)]
        for i, (text, slot) in enumerate(exports):
            button = QPushButton(text)
            button.clicked.connect(lambda _=False, s=slot: s())
            export_buttons.addWidget(button, i // 3, i % 3)
        self.split_combo = QComboBox()
        self.split_combo.addItems(SPLITS)
        self.split_combo.setToolTip("Split by photo session: train = sessions 1-3, valid = 4, test = 5")
        sample_button = QPushButton("🎓 Save as Training Sample")
        sample_button.setToolTip("Photo + corrected piece boxes + board corners as YOLO labels")
        sample_button.clicked.connect(lambda: self.save_training_sample())
        sample_row = QHBoxLayout()
        sample_row.addWidget(sample_button, 1)
        sample_row.addWidget(self.split_combo)

        right = QVBoxLayout()
        right.addWidget(self.diagram, 1)
        right.addWidget(self.orientation_label)
        right.addLayout(orientation)
        right.addWidget(self.info_box)
        right.addLayout(fen_row)
        right.addWidget(QLabel("<b>To review</b>"))
        right.addWidget(self.problems)
        right.addLayout(export_buttons)
        right.addLayout(sample_row)
        right_widget = QWidget()
        right_widget.setLayout(right)
        right_widget.setMaximumWidth(520)

        self.body = QWidget()
        body = QHBoxLayout(self.body)
        body.setContentsMargins(0, 0, 0, 0)
        body.addLayout(left, 1)
        body.addWidget(right_widget)

        self.status = QLabel()
        self.status.setWordWrap(True)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        bottom = QHBoxLayout()
        bottom.addWidget(self.status, 1)
        bottom.addWidget(close_button)

        layout = QVBoxLayout(self)
        layout.addLayout(steps)
        layout.addLayout(models)
        layout.addWidget(self.body, 1)
        layout.addLayout(bottom)

        self.on_conf_changed(self.conf_slider.value(), apply=False)
        self.on_lift_changed(self.lift_slider.value(), apply=False)
        if self.pieces_combo.count() and self.pieces_combo.currentData():
            self.on_pieces_model_changed(quiet=True)
        if getattr(session.finder, "model_path", None) is None and self.board_combo.count() > 1:
            self.board_combo.setCurrentIndex(1)
        self.refresh()

    # ----------------------------------------------------------------- helpers
    @property
    def busy(self) -> bool:
        return self.thread is not None

    def _wait_cursor(self, func, *args):
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            return func(*args)
        finally:
            QApplication.restoreOverrideCursor()

    def _warn(self, title: str, text: str):
        self.last_error = text
        QMessageBox.warning(self, title, text)

    def current_step(self) -> int:
        s = self.session
        if not s.has_photo:
            return 0
        if s.corners is None:
            return 1
        if not s.detected:
            return 2
        return 4 if self.orientation_confirmed else 3

    # ----------------------------------------------------------------- models
    def on_pieces_model_changed(self, *_, quiet: bool = False) -> bool:
        path = self.pieces_combo.currentData()
        if not path:
            return False
        try:
            self._wait_cursor(self.session.tester.load_detector, path)
        except Exception as e:
            self.session.tester.detector = None
            if not quiet:
                self._warn("Piece detector", f"Cannot load the detector:\n{e}")
            self.refresh()
            return False
        self.refresh()
        return True

    def on_board_model_changed(self, *_):
        path = self.board_combo.currentData()
        try:
            self._wait_cursor(self.session.finder.load_model, path)
        except Exception as e:
            self.session.finder.load_model(None)
            self.board_combo.blockSignals(True)
            self.board_combo.setCurrentIndex(0)
            self.board_combo.blockSignals(False)
            self._warn("Board corner model", f"Cannot use this board model:\n{e}")
        self.refresh()

    def choose_import(self, task: str):
        path, _ = QFileDialog.getOpenFileName(self, "Add YOLO weights", "", "YOLO weights (*.pt);;All files (*)")
        if path:
            self.import_model(path, task)

    def import_model(self, source: str, task: str) -> Optional[str]:
        try:
            path = self._wait_cursor(self.import_callback, source, task)
        except Exception as e:
            self._warn("Add model", f"Cannot add this model:\n{e}")
            return None
        combo = self.board_combo if task == "board" else self.pieces_combo
        index = combo.findData(path)
        if index < 0:
            combo.addItem(model_label(path), path)
            index = combo.count() - 1
        if combo.currentIndex() == index:
            combo.currentIndexChanged.emit(index)
        else:
            combo.setCurrentIndex(index)
        return path

    # ----------------------------------------------------------------- step 1-2: photo + corners
    def choose_photo(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open photo", "", PICTURE_FILTER)
        if path:
            self.open_photo(path)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if urls:
            self.open_photo(urls[0])

    def open_photo(self, path: str, detect: bool = True) -> bool:
        if self.busy:
            return False
        try:
            self._wait_cursor(self.session.load_photo, path)
        except Exception as e:
            self._warn("Open photo", f"Cannot open this photo:\n{e}")
            return False
        self.orientation_confirmed = False
        self.view.set_photo(self.session.image.size)
        self.session.info = self.info_box.info()
        self.find_corners(detect=detect)
        return True

    def find_corners(self, *_, detect: bool = True):
        if not self.session.has_photo or self.busy:
            return
        try:
            proposal = self._wait_cursor(self.session.auto_corners)
        except Exception as e:
            proposal = None
            self.status.setText(f"Corner finder failed: {html.escape(str(e))}")
        if proposal is None:
            self.session.set_corners(self.session.default_corners(), "manual")
            self.refresh(f"<b>No board found automatically</b> – drag the 4 coloured handles onto the "
                         "outer corners of the playing area.")
        else:
            self.refresh()
        if detect and self.session.tester.detector is not None and not self.session.detected:
            self.detect_pieces()

    def on_corner_moved(self, index: int, x: float, y: float):
        if self.busy:
            return
        self.session.move_corner(index, (x, y))
        self.refresh()

    # ----------------------------------------------------------------- step 3: pieces
    def detect_pieces(self, *_) -> bool:
        if self.busy or not self.session.has_photo:
            return False
        if self.session.tester.detector is None:
            self._warn("Detect pieces", "No piece detector loaded: choose or add a YOLO piece "
                                        "detector first.")
            return False
        self.body.setEnabled(False)
        self.status.setText("♟ Detecting pieces…")
        self.thread, self.worker = start_call(self.session.detect_pieces, self.session.conf)
        self.worker.finished.connect(self.on_detected)
        self.worker.failed.connect(self.on_detect_failed)
        self.thread.finished.connect(self._thread_done)
        self.thread.start()
        return True

    def _thread_done(self):
        if self.thread is not None:
            self.thread.deleteLater()
        self.thread = self.worker = None
        self.body.setEnabled(True)
        self.refresh()

    def on_detected(self, _result=None):
        self.session.suggest_orientation()
        self.session.suggest_castling()
        self.info_box.set_info(self.session.info)

    def on_detect_failed(self, message: str):
        self.last_error = message
        self.status.setText(f"<span style='color:#c62828'>Detection failed: {html.escape(message)}</span>")
        QMessageBox.warning(self, "Detect pieces", f"Piece detection failed:\n{message}")

    def wait_for_detection(self, timeout_ms: int = 60000):
        """Block until a running detection is done (used by tests and before closing)."""
        if self.thread is not None:
            self.thread.wait(timeout_ms)
            QApplication.processEvents()
            QApplication.processEvents()

    def on_conf_changed(self, value: int, apply: bool = True):
        self.conf_label.setText(f"{value}%")
        if apply and not self.busy:
            self.session.set_conf(value / 100)
            self.refresh()

    def on_lift_changed(self, value: int, apply: bool = True):
        self.lift_label.setText(f"{value}%")
        if apply and not self.busy:
            self.session.set_lift(value / 100)
            self.refresh()

    def on_box_drawn(self, x1: float, y1: float, x2: float, y2: float):
        if self.busy:
            return
        symbol = self.ask_piece(None, "Piece in this box", allow_empty=False)
        if symbol is ... or symbol is None:
            return
        self.session.add_box((x1, y1, x2, y2), symbol)
        self.refresh()

    def on_detection_clicked(self, index: int):
        if self.busy:
            return
        self.view.selected = index
        self.view.update()
        d = self.session.detections[index]
        menu = QMenu(self)
        menu.addSection(f"{d['label']} {d['conf']:.0%}")
        relabel = menu.addMenu("Change class")
        for name, symbol in PIECE_SYMBOLS.items():
            action = relabel.addAction(f"{GLYPHS[symbol.upper()]}  {name}")
            action.setData(("relabel", name))
        remove = QAction("🗑 Remove box", menu)
        remove.setData(("remove", None))
        menu.addAction(remove)
        action = menu.exec(self.view.mapToGlobal(self.view._to_widget(*d["box"][2:]).toPoint()))
        if action is not None and action.data():
            kind, name = action.data()
            if kind == "remove":
                self.session.remove_detection(index)
            else:
                self.session.relabel(index, name)
        self.view.selected = None
        self.refresh()

    # ----------------------------------------------------------------- step 4: orientation
    def rotate(self):
        self.session.rotate()
        self.orientation_confirmed = True
        self.refresh()

    def suggest_orientation(self):
        self.session.suggest_orientation()
        self.refresh()

    def confirm_orientation(self):
        self.orientation_confirmed = True
        self.refresh()

    def flip_diagram(self):
        self.white_bottom = not self.white_bottom
        self.refresh()

    # ----------------------------------------------------------------- step 5: review
    def ask_piece(self, square: Optional[str], title: str = "Set square", allow_empty: bool = True):
        """Piece menu at the mouse (``None`` = empty, ``...`` = cancelled). Patched in tests."""
        from PyQt6.QtGui import QCursor

        return PieceMenu(self, title if square is None else f"{square}", allow_empty).choose(QCursor.pos())

    def on_square_clicked(self, square: str):
        if self.busy or not self.session.has_photo:
            return
        choice = self.ask_piece(square)
        if choice is ...:
            return
        self.set_square(square, choice)

    def set_square(self, square: str, symbol: Optional[str]):
        self.session.set_square(square, symbol)
        self.refresh()

    def on_info_changed(self):
        self.session.set_info(self.info_box.info())
        self.refresh()

    def suggest_castling(self):
        self.session.suggest_castling()
        self.info_box.set_info(self.session.info)
        self.refresh()

    def copy_fen(self):
        QGuiApplication.clipboard().setText(self.session.fen())
        self.status.setText("FEN copied to the clipboard.")

    # ----------------------------------------------------------------- exports
    def _photo_stem(self) -> str:
        return Path(self.session.photo_path).stem if self.session.photo_path else "scan"

    def _save_path(self, path, title: str, suffix: str, filter_: str) -> Optional[str]:
        if not self.session.has_photo:
            self._warn(title, "Open a photo first.")
            return None
        if path:
            return str(path)
        path, _ = QFileDialog.getSaveFileName(self, title, f"{self._photo_stem()}{suffix}", filter_)
        return path or None

    def _folder(self, folder, title: str) -> Optional[str]:
        if not self.session.has_photo:
            self._warn(title, "Open a photo first.")
            return None
        if folder:
            return str(folder)
        return QFileDialog.getExistingDirectory(self, title) or None

    def _done(self, text: str):
        self.status.setText(f"✔ {html.escape(text)}")

    def save_fen(self, path=None) -> Optional[str]:
        path = self._save_path(path, "Save FEN", "_fen.txt", "Text (*.txt)")
        if path:
            self._done(f"Saved {self.session.export_fen(path)}")
        return path

    def save_annotated(self, path=None) -> Optional[str]:
        path = self._save_path(path, "Save annotated photo", "_annotated.png", "PNG (*.png)")
        if path:
            self._done(f"Saved {self.session.export_annotated(path)}")
        return path

    def save_diagram(self, path=None) -> Optional[str]:
        path = self._save_path(path, "Save diagram", "_diagram.png", "PNG (*.png)")
        if path:
            self._done(f"Saved {self.session.export_diagram(path, self.white_bottom)}")
        return path

    def save_debug(self, folder=None) -> Optional[str]:
        folder = self._folder(folder, "Folder for the step images")
        if folder:
            self.session.export_debug(folder)
            self._done(f"Saved the step images 1_corners.png … 5_diagram.png to {folder}")
        return folder

    def export_all(self, folder=None) -> Optional[str]:
        folder = self._folder(folder, "Export folder")
        if folder:
            self.session.export_all(folder, debug=True)
            self._done(f"Saved FEN, annotated photo, diagram, scan.json and step images to {folder}")
        return folder

    def save_training_sample(self, root=None, split: Optional[str] = None) -> Optional[dict]:
        if not self.session.has_photo:
            self._warn("Training sample", "Open a photo first.")
            return None
        root = root or self.sample_root
        if root is None:
            root = QFileDialog.getExistingDirectory(self, "Training sample dataset folder") or None
            if root is None:
                return None
            self.sample_root = Path(root)
        split = split or self.split_combo.currentText()
        try:
            result = self.session.save_training_sample(root, split)
        except Exception as e:
            self._warn("Training sample", f"Cannot save the training sample:\n{e}")
            return None
        text = (f"Saved to {Path(root) / 'pieces' / split} ({result['boxes']} boxes)"
                + (f" and {Path(root) / 'board' / split}" if "board_label" in result else ""))
        self._done(text)
        if result["notes"]:
            QMessageBox.information(self, "Training sample", text + "\n\n" + "\n".join(result["notes"]))
        return result

    # ----------------------------------------------------------------- refresh
    def refresh(self, message: Optional[str] = None):
        s = self.session
        step = self.current_step()
        for i, label in enumerate(self.step_labels):
            if i < step:
                style = "background:#c8e6c9; color:#1b5e20;"
            elif i == step:
                style = "background:#1565c0; color:white; font-weight:bold;"
            else:
                style = "background:#eceff1; color:#78909c;"
            label.setStyleSheet(style + " padding:6px; border-radius:6px;")
        has_photo = s.has_photo
        for w in (self.find_button, self.corner_mode, self.box_mode):
            w.setEnabled(has_photo)
        self.detect_button.setEnabled(has_photo and s.tester.detector is not None)
        if has_photo:
            if self.view.photo_size != s.image.size:
                self.view.set_photo(s.image.size)
            self.view.set_overlay(s.annotated_image() if s.corners else s.image)
            self.view.set_corners(s.corners)
            self.view.set_boxes([(p.index, p.box) for p in s.placements])
        self.diagram.set_image(s.diagram_image(DIAGRAM_PX, self.white_bottom), self.white_bottom)
        self.orientation_label.setText(
            f"a1 is at the <b>{CORNER_LONG[CORNER_NAMES[s.a1_corner]]}</b> corner of the photo "
            + ("(confirmed)" if self.orientation_confirmed else "(suggested – check it)"))
        self.fen_edit.setText(s.fen())
        self.problems.clear()
        for text in s.problems():
            item = QListWidgetItem(f"⚠ {text}")
            item.setForeground(QColor("#e65100"))
            self.problems.addItem(item)
        for text in s.warnings():
            item = QListWidgetItem(f"✖ {text}")
            item.setForeground(QColor("#c62828"))
            self.problems.addItem(item)
        if has_photo and s.detected and self.problems.count() == 0:
            item = QListWidgetItem("✔ Nothing to review")
            item.setForeground(QColor("#2e7d32"))
            self.problems.addItem(item)
        if message is not None:
            self.status.setText(message)
        elif not self.busy:
            self.status.setText(self._hint(step))

    def _hint(self, step: int) -> str:
        s = self.session
        if step == 0:
            return "Open a photo: camera high (≥ 2 m), all 4 outer corners of the board visible."
        if step == 1:
            return "Drag the 4 handles onto the outer corners of the playing area."
        if step == 2:
            if s.tester.detector is None:
                return "Choose or add a piece detector, then click ♟ Detect pieces."
            return "Check the green grid, then click ♟ Detect pieces."
        method = f"corners: {s.corner_method}"
        if step == 3:
            return f"{len(s.active_detections())} pieces ({method}). Check where a1 is: ⟳ Rotate a1 or ✔ OK."
        return (f"{len(s.active_detections())} pieces ({method}). Click a diagram square to correct it, "
                "click a box to relabel or remove it, then export.")

    def done(self, result: int):
        if self.thread is not None:
            self.thread.wait()
        super().done(result)
