"""Modal dialog for manually reviewing predictions on random validation images."""
from __future__ import annotations

import html
import random
from typing import Optional

import torch
from PIL import Image
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from .checkpoint import load_checkpoint, model_from_checkpoint
from .data import MultiFolderImageDataset, build_eval_transform


class ReviewSession:
    """Model + validation samples + random sample history (Qt independent)."""

    def __init__(self, checkpoint_path: str, folders: list[str], seed: Optional[int] = None):
        ckpt = load_checkpoint(checkpoint_path)
        self.classes = list(ckpt["classes"])
        self.model = model_from_checkpoint(ckpt).eval()
        self.transform = build_eval_transform(int(ckpt["image_size"]))
        self.dataset = MultiFolderImageDataset(folders, self.classes)
        if len(self.dataset) == 0:
            raise ValueError("The selected validation folders contain no images of the model's classes.")
        self.rng = random.Random(seed)
        self.history: list[int] = []
        self.position = -1

    @torch.no_grad()
    def predict(self, index: int) -> dict:
        path, target = self.dataset.samples[index]
        image = Image.open(path).convert("RGB")
        probs = torch.softmax(self.model(self.transform(image).unsqueeze(0)), dim=1)[0]
        k = min(3, len(self.classes))
        conf, idx = probs.topk(k)
        top = [(self.classes[i], float(c)) for c, i in zip(conf.tolist(), idx.tolist())]
        return {"path": path, "truth": self.classes[target], "top3": top,
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
    def __init__(self, session: ReviewSession, checkpoint_path: str, parent=None):
        super().__init__(parent)
        self.session = session
        self.setWindowTitle("Manual Review")
        self.setModal(True)
        self.resize(560, 640)

        self.info = QLabel(f"Model: {checkpoint_path}")
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
        pixmap = QPixmap(result["path"])
        if not pixmap.isNull():
            pixmap = pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio,
                                   Qt.TransformationMode.SmoothTransformation)
        self.image.setPixmap(pixmap)
        self.image.setStyleSheet(f"border: 6px solid {color};")
        self.badge.setText("CORRECT" if result["correct"] else "INCORRECT")
        self.badge.setStyleSheet(f"background: {color}; color: white; font-weight: bold; padding: 4px;")
        self.truth.setText(f"<b>Ground truth:</b> {html.escape(result['truth'])}"
                           f"<br><small>{html.escape(result['path'])}</small>")
        lines = []
        for rank, (name, conf) in enumerate(result["top3"], 1):
            style = "color:#2e7d32;font-weight:bold" if name == result["truth"] else ""
            lines.append(f"<span style='{style}'>{rank}. {html.escape(name)}: {conf * 100:.1f}%</span>")
        self.predictions.setText("<b>Top-3 predictions:</b><br>" + "<br>".join(lines))
        self.counter.setText(f"sample {self.session.position + 1} / {len(self.session.history)}")
        self.prev_button.setEnabled(self.session.position > 0)
