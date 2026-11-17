"""Modal dialogs for manually checking predictions: random validation/test images (Manual Review)
and pictures chosen by the user (Test own picture: the YOLO detector boxes and names every piece)."""
from __future__ import annotations

import html
import io
import random
from pathlib import Path
from typing import Callable, Optional, Sequence

import torch
from PIL import Image
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

from .checkpoint import load_checkpoint, model_from_checkpoint
from .data import MultiFolderImageDataset, build_eval_transform, resolve_placeholder_classes
from .detector import (  # noqa: F401 - ImageTester is re-exported for older imports
    AGREE_RGB,
    DetectionReviewSession,
    ImageTester,
    draw_detections,
    load_picture,
    load_yolo_detector,
)
from .render import draw_corners

CORNER_PRED_RGB = (0, 200, 83)
CORNER_TRUTH_RGB = (41, 121, 255)
PICTURE_FILTER = "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff);;All files (*)"
EMPTY_AREA_STYLE = ("border: 3px dashed #90a4ae; border-radius: 12px; color: #607d8b; "
                    "font-size: 18px; background: #f5f7f8;")


@torch.no_grad()
def top_predictions(model, transform, classes: list[str], image: Image.Image,
                    k: int = 3) -> list[tuple[str, float]]:
    probs = torch.softmax(model(transform(image).unsqueeze(0)), dim=1)[0]
    conf, idx = probs.topk(min(k, len(classes)))
    return [(classes[i], float(c)) for c, i in zip(conf.tolist(), idx.tolist())]


def to_pixmap(image: Image.Image, size: int = 400) -> QPixmap:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    pixmap = QPixmap()
    pixmap.loadFromData(buffer.getvalue(), "PNG")
    if not pixmap.isNull():
        pixmap = pixmap.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)
    return pixmap


class ReviewSession:
    """Model + validation/test samples + random sample history (Qt independent)."""

    def __init__(self, checkpoint_path: str, folders: list[str], seed: Optional[int] = None):
        ckpt = load_checkpoint(checkpoint_path)
        self.classes = resolve_placeholder_classes(ckpt["classes"], folders)
        self.model = model_from_checkpoint(ckpt).eval()
        self.transform = build_eval_transform(int(ckpt["image_size"]))
        self.dataset = MultiFolderImageDataset(folders, self.classes)
        if len(self.dataset) == 0:
            raise ValueError("The selected folders contain no images of the model's classes.")
        self.rng = random.Random(seed)
        self.history: list[int] = []
        self.position = -1

    def predict(self, index: int) -> dict:
        path, target, box = self.dataset.samples[index]
        image = self.dataset.load_image(index)
        top = top_predictions(self.model, self.transform, self.classes, image)
        return {"path": path, "box": box, "image": image, "truth": self.classes[target], "top3": top,
                "correct": top[0][0] == self.classes[target]}

    def next(self) -> dict:
        if self.position + 1 < len(self.history):
            self.position += 1
        else:
            self.history.append(self.rng.randrange(len(self.dataset)))
            self.position = len(self.history) - 1
        return self.predict(self.history[self.position])

    def previous(self) -> Optional[dict]:
        if self.position <= 0:
            return None
        self.position -= 1
        return self.predict(self.history[self.position])


class ReviewDialog(QDialog):
    def __init__(self, session: ReviewSession, checkpoint_path: str, parent=None, source: str = ""):
        super().__init__(parent)
        self.session = session
        self.setWindowTitle("Manual Review")
        self.setModal(True)
        self.resize(560, 640)

        self.info = QLabel(f"Model: {checkpoint_path}" + (f"\nImages: {source}" if source else ""))
        self.info.setWordWrap(True)
        self.image = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumSize(420, 420)
        self.badge = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.truth = QLabel()
        self.predictions = QLabel()
        self.counter = QLabel()
        self.prev_button = QPushButton("◀ Previous")
        self.next_button = QPushButton("Next ▶")
        close_button = QPushButton("Close")
        self.prev_button.clicked.connect(self.show_previous)
        self.next_button.clicked.connect(self.show_next)
        close_button.clicked.connect(self.accept)

        buttons = QHBoxLayout()
        buttons.addWidget(self.prev_button)
        buttons.addWidget(self.counter, 1, Qt.AlignmentFlag.AlignCenter)
        buttons.addWidget(self.next_button)
        buttons.addWidget(close_button)
        layout = QVBoxLayout(self)
        layout.addWidget(self.info)
        layout.addWidget(self.image, 1)
        layout.addWidget(self.badge)
        layout.addWidget(self.truth)
        layout.addWidget(self.predictions)
        layout.addLayout(buttons)
        self.show_next()

    def show_next(self):
        self._show(self.session.next())

    def show_previous(self):
        result = self.session.previous()
        if result is not None:
            self._show(result)

    def _show(self, result: dict):
        color = "#2e7d32" if result["correct"] else "#c62828"
        self.image.setPixmap(to_pixmap(result["image"]))
        self.image.setStyleSheet(f"border: 6px solid {color};")
        self.badge.setText("CORRECT" if result["correct"] else "INCORRECT")
        self.badge.setStyleSheet(f"background: {color}; color: white; font-weight: bold; padding: 4px;")
        box = ""
        if result["box"] is not None:
            box = " – label box (x1, y1, x2, y2) = " + ", ".join(f"{v:.3f}" for v in result["box"])
        self.truth.setText(f"<b>Ground truth:</b> {html.escape(result['truth'])}"
                           f"<br><small>{html.escape(result['path'])}{box}</small>")
        lines = []
        for rank, (name, conf) in enumerate(result["top3"], 1):
            style = "color:#2e7d32;font-weight:bold" if name == result["truth"] else ""
            lines.append(f"<span style='{style}'>{rank}. {html.escape(name)}: {conf * 100:.1f}%</span>")
        self.predictions.setText("<b>Top-3 predictions:</b><br>" + "<br>".join(lines))
        self.counter.setText(f"sample {self.session.position + 1} / {len(self.session.history)}")
        self.prev_button.setEnabled(self.session.position > 0)




class DetectionReviewDialog(QDialog):
    """Manual Review for detectors: predicted boxes (left) vs. ground-truth labels (right)."""

    def __init__(self, session: DetectionReviewSession, weights: str, parent=None, source: str = ""):
        super().__init__(parent)
        self.session = session
        self.setWindowTitle("Manual Review – detection")
        self.setModal(True)
        self.resize(1250, 760)
        info = QLabel(f"Model: {html.escape(weights)}"
                      + (f"<br>Images: {html.escape(source)}" if source else "")
                      + "<br><small>Left: prediction (green = correct, orange = wrong class or extra "
                        "box). Right: ground truth (blue = found, orange = missed or wrong class)."
                        "</small>")
        info.setWordWrap(True)
        self.pred_image = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.truth_image = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        for label in (self.pred_image, self.truth_image):
            label.setMinimumSize(560, 520)
        images = QHBoxLayout()
        images.addWidget(self.pred_image, 1)
        images.addWidget(self.truth_image, 1)
        self.badge = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.details = QLabel()
        self.details.setWordWrap(True)
        self.counter = QLabel()
        self.prev_button = QPushButton("◀ Previous")
        self.next_button = QPushButton("Next ▶")
        close_button = QPushButton("Close")
        self.prev_button.clicked.connect(self.show_previous)
        self.next_button.clicked.connect(self.show_next)
        close_button.clicked.connect(self.accept)
        buttons = QHBoxLayout()
        buttons.addWidget(self.prev_button)
        buttons.addWidget(self.counter, 1, Qt.AlignmentFlag.AlignCenter)
        buttons.addWidget(self.next_button)
        buttons.addWidget(close_button)
        layout = QVBoxLayout(self)
        layout.addWidget(info)
        layout.addLayout(images, 1)
        layout.addWidget(self.badge)
        layout.addWidget(self.details)
        layout.addLayout(buttons)
        self.show_next()

    def show_next(self):
        self._show(self.session.next())

    def show_previous(self):
        result = self.session.previous()
        if result is not None:
            self._show(result)

    def _show(self, result: dict):
        c = result["counts"]
        color = "#2e7d32" if c["perfect"] else "#c62828"
        pred_image, truth_image = result["pred_image"], result["truth_image"]
        corner_text = ""
        if "pred_corners" in result:  # board corner (pose) model: show the 4 corners
            pred_image = draw_corners(pred_image, result["pred_corners"], CORNER_PRED_RGB)
            truth_image = draw_corners(truth_image, result["truth_corners"], CORNER_TRUTH_RGB)
            error = result.get("corner_error")
            corner_text = ("  ·  Corners: not found" if result["pred_corners"] is None else
                           f"  ·  Corner error: {error:.2f}% of the board diagonal" if error is not None
                           else "")
        self.pred_image.setPixmap(to_pixmap(pred_image, 580))
        self.truth_image.setPixmap(to_pixmap(truth_image, 580))
        self.badge.setText(("ALL CORRECT" if c["perfect"] else "MISTAKES")
                           + f" – {c['correct']} / {c['truths']} objects correct")
        self.badge.setStyleSheet(f"background: {color}; color: white; font-weight: bold; padding: 4px;")
        self.details.setText(f"Wrong class: {c['wrong_class']}  ·  Missed: {c['missed']}  ·  "
                             f"Extra boxes: {c['extra']}{corner_text}"
                             f"<br><small>{html.escape(result['path'])}</small>")
        self.counter.setText(f"image {self.session.position + 1} / {len(self.session.history)}")
        self.prev_button.setEnabled(self.session.position > 0)


def model_label(path: str) -> str:
    """Short name of a model file, relative to its project's ``checkpoints`` folder when possible."""
    parts = Path(path).parts
    if "checkpoints" in parts:
        rel = parts[len(parts) - parts[::-1].index("checkpoints"):]
        if rel and rel[0] == "detectors":
            rel = rel[1:]
        return "/".join(rel)
    return Path(path).name


class PictureArea(QLabel):
    """Picture area that opens a file dialog when clicked."""

    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()


class ImageTestDialog(QDialog):
    """Find every piece in your own picture: the detector draws the boxes and names them."""

    def __init__(self, tester: ImageTester, parent=None, detectors: Sequence[str] = (),
                 import_detector: Optional[Callable[[str], str]] = None):
        super().__init__(parent)
        self.tester = tester
        self.import_callback = import_detector
        self.current_path: Optional[str] = None
        self.detector_error: Optional[str] = None
        self.setWindowTitle("Test own picture")
        self.setModal(True)
        self.setAcceptDrops(True)
        self.resize(1150, 820)

        self.detector_combo = QComboBox()
        for path in detectors:
            self.detector_combo.addItem(model_label(path), path)
        self.detector_combo.currentIndexChanged.connect(self.on_detector_changed)
        add_button = QPushButton("➕ Add detector file…")
        add_button.setToolTip("Copy YOLO detector weights (e.g. runs/chess/weights/best.pt) into this "
                              "project. Classifier checkpoints of this app cannot draw boxes.")
        add_button.clicked.connect(self.choose_import)
        add_button.setEnabled(import_detector is not None)
        self.detector_combo.setMinimumWidth(420)
        self.status = QLabel()
        self.status.setWordWrap(True)

        steps = QGridLayout()
        steps.addWidget(QLabel("<b>Detector</b> – finds and names the pieces"), 0, 0)
        steps.addWidget(self.detector_combo, 0, 1)
        steps.addWidget(add_button, 0, 2)
        steps.setColumnStretch(1, 1)

        self.image = PictureArea("📷  Click here or drop a picture to test it",
                                 alignment=Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumSize(680, 560)
        self.image.setCursor(Qt.CursorShape.PointingHandCursor)
        self.image.setStyleSheet(EMPTY_AREA_STYLE)
        self.image.clicked.connect(self.choose_picture)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.pieces = QListWidget()
        self.pieces.setMinimumWidth(330)
        self.pieces.setWordWrap(True)
        self.pieces.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        side = QVBoxLayout()
        side.addWidget(QLabel("<b>Pieces found</b>"))
        side.addWidget(self.summary)
        side.addWidget(self.pieces, 1)
        body = QHBoxLayout()
        body.addWidget(self.image, 1)
        body.addLayout(side)

        open_button = QPushButton("📂 Open picture…")
        open_button.clicked.connect(self.choose_picture)
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(5, 95)
        self.conf_slider.setValue(25)
        self.conf_slider.setFixedWidth(180)
        self.conf_slider.setToolTip("Lower = find more pieces (more false boxes), "
                                    "higher = only confident boxes.")
        self.conf_label = QLabel()
        self.conf_slider.valueChanged.connect(self.on_conf_changed)
        self.conf_slider.sliderReleased.connect(self.refresh)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        buttons = QHBoxLayout()
        buttons.addWidget(open_button)
        buttons.addSpacing(20)
        buttons.addWidget(QLabel("Min. confidence:"))
        buttons.addWidget(self.conf_slider)
        buttons.addWidget(self.conf_label)
        buttons.addStretch()
        buttons.addWidget(close_button)

        layout = QVBoxLayout(self)
        layout.addLayout(steps)
        layout.addWidget(self.status)
        layout.addLayout(body, 1)
        layout.addLayout(buttons)
        self.on_conf_changed(self.conf_slider.value())
        if self.detector_combo.count():
            self.on_detector_changed(quiet=True)
        else:
            self.update_status()

    @property
    def conf(self) -> float:
        return self.conf_slider.value() / 100

    def update_status(self):
        if self.detector_combo.count() == 0:
            text, ok = ("This project has no detector yet. Click <b>➕ Add detector file…</b> (e.g. "
                        "<code>runs/chess/weights/best.pt</code>) or train one with <b>🎯 Train "
                        "detector</b> in the <i>Training &amp; Quotas</i> tab."), False
        elif self.detector_error:
            text, ok = f"Cannot use this detector: {html.escape(self.detector_error)}", False
        else:
            text, ok = "Ready – click the picture area or drop a picture.", True
        self.status.setText(text)
        self.status.setStyleSheet(f"color: {'#2e7d32' if ok else '#c62828'};")

    def _busy(self, func, *args):
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            return func(*args)
        finally:
            QApplication.restoreOverrideCursor()

    def on_detector_changed(self, *_, quiet: bool = False) -> bool:
        path = self.detector_combo.currentData()
        if not path:
            return False
        try:
            self._busy(self.tester.load_detector, path)
        except Exception as e:
            self.tester.detector, self.tester.detector_path = None, None
            self.detector_error = str(e)
            self.update_status()
            if not quiet:
                QMessageBox.warning(self, "Test own picture", f"Cannot load the detector:\n{e}")
            return False
        self.detector_error = None
        self.update_status()
        self.refresh()
        return True

    def on_conf_changed(self, value: int):
        self.conf_label.setText(f"{value}%")
        if not self.conf_slider.isSliderDown():
            self.refresh()

    def choose_import(self):
        path, _ = QFileDialog.getOpenFileName(self, "Add YOLO detector weights", "",
                                              "YOLO weights (*.pt);;All files (*)")
        if path:
            self.import_picture_detector(path)

    def import_picture_detector(self, source: str) -> Optional[str]:
        try:
            path = self._busy(self.import_callback, source)
        except Exception as e:
            QMessageBox.warning(self, "Add detector", f"Cannot add this detector:\n{e}")
            return None
        index = self.detector_combo.findData(path)
        if index < 0:
            self.detector_combo.blockSignals(True)
            self.detector_combo.insertItem(0, model_label(path), path)
            self.detector_combo.blockSignals(False)
            index = 0
        if self.detector_combo.currentIndex() == index:
            self.on_detector_changed()
        else:
            self.detector_combo.setCurrentIndex(index)
        return path if self.tester.detector_path == path else None

    def choose_picture(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open picture", "", PICTURE_FILTER)
        if path:
            self.test_picture(path)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if files:
            self.test_picture(files[0])

    def refresh(self):
        if self.current_path and self.tester.detector is not None:
            self.test_picture(self.current_path)

    def test_picture(self, path: str) -> Optional[dict]:
        try:
            result = self._busy(self.tester.predict_file, path, self.conf)
        except Exception as e:
            QMessageBox.warning(self, "Test own picture", f"Cannot test this picture:\n{e}")
            return None
        self.current_path = path
        detections = result["detections"]
        self.image.setPixmap(to_pixmap(result["annotated"], 800))
        self.image.setStyleSheet("")
        text = f"<b>{len(detections)}</b> found in <small>{html.escape(Path(path).name)}</small>"
        if not detections:
            text += "<br><small>Nothing found – move the confidence slider to the left.</small>"
        self.summary.setText(text)
        self.pieces.clear()
        for n, d in enumerate(detections, 1):
            item = QListWidgetItem(f"{n}. {d['label']}  {d['conf'] * 100:.0f}%")
            item.setForeground(QBrush(QColor(*AGREE_RGB)))
            self.pieces.addItem(item)
        return result
