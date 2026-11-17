"""Data tools: generate synthetic board-corner data and import ChessReD2K (thin PyQt6 layer).

The work runs in :func:`schaakgui.worker.start_call` threads; the generators and importers themselves
are plain Python (:mod:`schaakgui.synthetic`, :mod:`schaakgui.importers`)."""
from __future__ import annotations

import html
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .importers import import_chessred2k
from .synthetic import SyntheticConfig, generate_board_dataset
from .worker import start_call

SPLITS = ("train", "valid", "test")


def _path_row(edit: QLineEdit, title: str, parent: QWidget, file_filter: Optional[str] = None) -> QHBoxLayout:
    button = QPushButton("…")
    button.setFixedWidth(32)

    def browse():
        if file_filter:
            path, _ = QFileDialog.getOpenFileName(parent, title, edit.text(), file_filter)
        else:
            path = QFileDialog.getExistingDirectory(parent, title, edit.text())
        if path:
            edit.setText(path)

    button.clicked.connect(browse)
    row = QHBoxLayout()
    row.addWidget(edit, 1)
    row.addWidget(button)
    return row


class DataToolsDialog(QDialog):
    """Two tabs: **Synthetic Boards** (YOLO-pose corner data) and **Import ChessReD2K**."""

    def __init__(self, output_root: str | Path, task: str = "detect", parent=None):
        super().__init__(parent)
        self.output_root = Path(output_root)
        self.task = task
        self.thread = None
        self.worker = None
        self._kind = ""
        self.results: list[dict] = []
        self.last_error: Optional[str] = None
        self.setWindowTitle("Data tools")
        self.setModal(True)
        self.resize(720, 460)

        # synthetic boards
        synthetic = QWidget()
        self.syn_out = QLineEdit(str(self.output_root / "synthetic_boards"))
        self.syn_backgrounds = QLineEdit()
        self.syn_backgrounds.setPlaceholderText("optional: your grass / paving photos (else procedural noise)")
        self.syn_count = QSpinBox()
        self.syn_count.setRange(1, 100000)
        self.syn_count.setValue(3000)
        self.syn_size = QSpinBox()
        self.syn_size.setRange(64, 2048)
        self.syn_size.setSingleStep(32)
        self.syn_size.setValue(640)
        self.syn_val = QDoubleSpinBox()
        self.syn_val.setRange(0.0, 0.5)
        self.syn_val.setSingleStep(0.05)
        self.syn_val.setValue(0.15)
        self.syn_rotation = QSpinBox()
        self.syn_rotation.setRange(0, 45)
        self.syn_rotation.setValue(30)
        self.syn_rotation.setSuffix("°")
        self.syn_occluders = QCheckBox("Piece-like occluders")
        self.syn_occluders.setChecked(True)
        self.syn_shadows = QCheckBox("Shadows and lighting")
        self.syn_shadows.setChecked(True)
        self.syn_seed = QSpinBox()
        self.syn_seed.setRange(-1, 2**31 - 1)
        self.syn_seed.setValue(-1)
        self.syn_seed.setSpecialValueText("random")
        form = QFormLayout(synthetic)
        form.addRow(QLabel("Random 8×8 boards seen by a tilted camera (20°–70°), with the 4 outer corners "
                           "as YOLO-pose keypoints. Use it to pre-train the board corner model."))
        form.addRow("Output folder", _path_row(self.syn_out, "Output folder", self))
        form.addRow("Backgrounds", _path_row(self.syn_backgrounds, "Background photos folder", self))
        form.addRow("Images", self.syn_count)
        form.addRow("Image size", self.syn_size)
        form.addRow("Validation fraction", self.syn_val)
        form.addRow("Max. rotation", self.syn_rotation)
        form.addRow(self.syn_occluders)
        form.addRow(self.syn_shadows)
        form.addRow("Seed", self.syn_seed)
        self.syn_button = QPushButton("⚙ Generate")
        self.syn_button.clicked.connect(lambda: self.start_synthetic())
        form.addRow(self.syn_button)

        # ChessReD2K
        chessred = QWidget()
        self.cr_json = QLineEdit()
        self.cr_json.setPlaceholderText("annotations.json from data.4tu.nl (ChessReD)")
        self.cr_images = QLineEdit()
        self.cr_images.setPlaceholderText("folder with the downloaded ChessReD images")
        self.cr_out = QLineEdit(str(self.output_root / "chessred2k"))
        self.cr_copy = QCheckBox("Copy images (uncheck to link to the originals)")
        self.cr_copy.setChecked(True)
        cr_form = QFormLayout(chessred)
        cr_form.addRow(QLabel("Converts ChessReD2K into <b>pieces/</b> (12-class YOLO boxes) and "
                              "<b>board/</b> (YOLO-pose corners), keeping its train/val/test split. "
                              "Download the images and annotations.json yourself from "
                              "<a href='https://data.4tu.nl/datasets/99b5c721-280b-450b-b058-b2900b69a90f'>"
                              "4TU.ResearchData</a>."))
        cr_form.itemAt(0).widget().setOpenExternalLinks(True)
        cr_form.itemAt(0).widget().setWordWrap(True)
        cr_form.addRow("annotations.json", _path_row(self.cr_json, "ChessReD annotations", self,
                                                     "JSON (*.json);;All files (*)"))
        cr_form.addRow("Images folder", _path_row(self.cr_images, "ChessReD images folder", self))
        cr_form.addRow("Output folder", _path_row(self.cr_out, "Output folder", self))
        cr_form.addRow(self.cr_copy)
        self.cr_button = QPushButton("📥 Import")
        self.cr_button.clicked.connect(lambda: self.start_chessred())
        cr_form.addRow(self.cr_button)
        for label in (form.itemAt(0).widget(),):
            label.setWordWrap(True)

        self.tabs = QTabWidget()
        self.tabs.addTab(synthetic, "Synthetic Boards")
        self.tabs.addTab(chessred, "Import ChessReD2K")
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.RichText)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        buttons = QHBoxLayout()
        buttons.addWidget(self.progress, 1)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.close_button)
        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs, 1)
        layout.addWidget(self.status)
        layout.addLayout(buttons)

    # ----------------------------------------------------------------- jobs
    @property
    def busy(self) -> bool:
        return self.thread is not None

    def synthetic_config(self) -> SyntheticConfig:
        return SyntheticConfig(
            count=self.syn_count.value(), image_size=self.syn_size.value(),
            val_fraction=self.syn_val.value(), backgrounds_dir=self.syn_backgrounds.text().strip() or None,
            max_rotation=float(self.syn_rotation.value()), occluders=self.syn_occluders.isChecked(),
            shadows=self.syn_shadows.isChecked(),
            seed=None if self.syn_seed.value() < 0 else self.syn_seed.value())

    def start_synthetic(self) -> bool:
        out = self.syn_out.text().strip()
        if not out:
            self.status.setText("Choose an output folder.")
            return False
        return self._start("synthetic", generate_board_dataset, out, self.synthetic_config())

    def start_chessred(self) -> bool:
        ann, images, out = (e.text().strip() for e in (self.cr_json, self.cr_images, self.cr_out))
        if not ann or not Path(ann).is_file():
            self.status.setText("Choose the ChessReD <b>annotations.json</b> file.")
            return False
        if not images or not Path(images).is_dir():
            self.status.setText("Choose the folder with the downloaded ChessReD images.")
            return False
        if not out:
            self.status.setText("Choose an output folder.")
            return False
        return self._start("chessred", import_chessred2k, ann, images, out,
                           copy_images=self.cr_copy.isChecked())

    def _start(self, kind: str, func, *args, **kwargs) -> bool:
        if self.busy:
            return False
        self.last_error = None
        self._set_running(True)
        self.progress.setRange(0, 0)
        self.status.setText("Working…")
        self.thread, self.worker = start_call(func, *args, with_progress=True, **kwargs)
        self.worker.progress.connect(self.on_progress)
        self._kind = kind
        self.worker.finished.connect(self.on_worker_finished)  # bound slot: runs in the GUI thread
        self.worker.failed.connect(self.on_failed)
        self.thread.finished.connect(self._thread_done)
        self.thread.start()
        return True

    def _set_running(self, running: bool):
        self.syn_button.setEnabled(not running)
        self.cr_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        self.close_button.setEnabled(not running)

    def _thread_done(self):
        if self.thread is not None:
            self.thread.deleteLater()
        self.thread = self.worker = None
        self._set_running(False)

    def on_progress(self, done: int, total: int):
        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(done)
        self.status.setText(f"{done} / {total}")

    def cancel(self):
        if self.worker is not None:
            self.worker.request_stop()
            self.status.setText("Stopping…")

    def on_failed(self, message: str):
        self.last_error = message
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.status.setText(f"<span style='color:#c62828'>Failed: {html.escape(message)}</span>")

    def on_worker_finished(self, result):
        self.on_finished(self._kind, result)

    def on_finished(self, kind: str, result: dict):
        result = dict(result, kind=kind)
        self.results.append(result)
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        counts = ", ".join(f"{k} {v}" for k, v in result["counts"].items())
        stopped = " <b>(stopped early)</b>" if result.get("stopped") else ""
        if kind == "synthetic":
            text = (f"✔ Synthetic boards: {counts}{stopped} in {html.escape(result['out_dir'])}. "
                    "Select the <b>Board corners</b> task to train on them.")
        else:
            skipped = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in result["skipped"].items() if v)
            text = (f"✔ ChessReD2K: {counts}{stopped} in {html.escape(result['out_dir'])}"
                    + (f" (skipped: {skipped})" if skipped else "")
                    + ". <b>pieces/</b> is for the piece detector, <b>board/</b> for the corner model.")
        self.status.setText(text)

    def wait(self, timeout_ms: int = 120000):
        if self.thread is not None:
            self.thread.wait(timeout_ms)

    # ----------------------------------------------------------------- results for the main window
    def created_folders(self, task: Optional[str] = None) -> dict[str, list[str]]:
        """Split folders made in this dialog for a task: board folders for ``board``, piece folders
        for the detector (``{"train": [...], "valid": [...], "test": [...]}``)."""
        task = task or self.task
        folders: dict[str, list[str]] = {split: [] for split in SPLITS}
        for result in self.results:
            if result["kind"] == "synthetic":
                if task != "board":
                    continue
                split_dirs = {s: paths for s, paths in result["folders"].items()}
            else:
                kind = "board" if task == "board" else "pieces"
                split_dirs = {s: [p] for s, p in result["folders"][kind].items()}
            for split, paths in split_dirs.items():
                for path in paths:
                    if path not in folders[split]:
                        folders[split].append(path)
        return folders

    def done(self, result: int):
        if self.thread is not None:
            self.cancel()
            self.thread.wait()
        super().done(result)
