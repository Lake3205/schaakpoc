"""Main desktop window: projects, datasets, augmentations, quotas, training and analytics."""
from __future__ import annotations

import html
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QThread
from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .checkpoint import ARCHITECTURES, Lineage, read_lineage, session_val_folders
from .data import class_counts, dataset_yaml, enabled_augmentations
from .detector import (
    DETECTOR_BASE_MODELS,
    DetectionReviewSession,
    DetectorJob,
    evaluate_detector,
    import_detector,
    list_detectors,
    read_run_info,
    ultralytics_augmentation,
)
from .engine import TrainJob, evaluate_checkpoint
from .plots import PlotCanvas
from .projects import TASKS, DatasetEntry, Project, ProjectManager, QuotaConfig
from .quota import OBJECTIVES, QuotaTracker, class_progress, objective_progress, objective_threshold
from .review_dialog import (
    DetectionReviewDialog,
    ImageTestDialog,
    ImageTester,
    ReviewDialog,
    ReviewSession,
)
from .sync_dialog import YamlSyncDialog
from .yaml_sync import unique_yaml_paths
from .worker import TrainingWorker, start_detector_worker, start_worker

STOP_REASONS = {
    "completed": "all epochs completed",
    "quota": "all quota targets satisfied",
    "early_stopping": "early stopping (no improvement)",
    "user": "stopped by user",
}
DETECTION_METRICS = {"precision": "Precision", "recall": "Recall", "mAP50": "mAP50",
                     "mAP50_95": "mAP50-95"}


def _scrollable(widget: QWidget) -> QScrollArea:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(widget)
    return area


def _spin(minimum, maximum, value=0, suffix="") -> QSpinBox:
    box = QSpinBox()
    box.setRange(minimum, maximum)
    box.setValue(value)
    box.setSuffix(suffix)
    return box


def _dspin(minimum, maximum, decimals=2, step=0.1, suffix="") -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(minimum, maximum)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setSuffix(suffix)
    return box


def _badge_icon(color: str, symbol: str) -> QIcon:
    """Small round status badge (e.g. green ✓ = .yaml detected)."""
    pixmap = QPixmap(16, 16)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor(color))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(1, 1, 14, 14)
    painter.setPen(QColor("white"))
    font = painter.font()
    font.setPixelSize(10)
    font.setBold(True)
    painter.setFont(font)
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, symbol)
    painter.end()
    return QIcon(pixmap)


def _progress_style(met: bool) -> str:
    color = "#43a047" if met else "#fb8c00"
    return f"QProgressBar::chunk {{ background-color: {color}; }} QProgressBar {{ text-align: center; }}"


class FolderList(QGroupBox):
    """Checkable folder list with Add/Remove buttons."""

    def __init__(self, title: str, on_change, parent=None):
        super().__init__(title, parent)
        self.on_change = on_change
        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list.itemChanged.connect(lambda _item: self.on_change())
        add = QPushButton("Add folder…")
        remove = QPushButton("Remove selected")
        add.clicked.connect(self.add_folder)
        remove.clicked.connect(self.remove_selected)
        buttons = QHBoxLayout()
        buttons.addWidget(add)
        buttons.addWidget(remove)
        buttons.addStretch()
        layout = QVBoxLayout(self)
        layout.addWidget(self.list)
        layout.addLayout(buttons)

    def add_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select dataset folder (class_name/xxx.jpg or YOLO images/ + labels/)")
        if folder:
            if folder in [e.path for e in self.entries()]:
                return
            self.add_entry(DatasetEntry(folder, True))
            self.on_change()

    def add_entry(self, entry: DatasetEntry):
        item = QListWidgetItem(entry.path)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked if entry.checked else Qt.CheckState.Unchecked)
        self._update_badge(item)
        self.list.addItem(item)

    @staticmethod
    def _update_badge(item: QListWidgetItem):
        """Status badge: green ✓ if a .yaml configuration was detected for the folder."""
        try:
            path = dataset_yaml(item.text())
        except OSError:
            path = None
        item.setData(Qt.ItemDataRole.UserRole, str(path) if path else None)
        if path:
            item.setIcon(_badge_icon("#2e7d32", "✓"))
            item.setToolTip(f"YAML detected: {path}")
        else:
            item.setIcon(_badge_icon("#9e9e9e", "–"))
            item.setToolTip("No .yaml configuration detected")

    def refresh_badges(self):
        self.list.blockSignals(True)
        for i in range(self.list.count()):
            self._update_badge(self.list.item(i))
        self.list.blockSignals(False)

    def yaml_paths(self) -> list[str]:
        """Detected .yaml files of all folders (checked or not)."""
        values = (self.list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.list.count()))
        return [v for v in values if v]

    def remove_selected(self):
        for item in self.list.selectedItems():
            self.list.takeItem(self.list.row(item))
        self.on_change()

    def set_entries(self, entries: list[DatasetEntry]):
        self.list.blockSignals(True)
        self.list.clear()
        for entry in entries:
            self.add_entry(entry)
        self.list.blockSignals(False)

    def entries(self) -> list[DatasetEntry]:
        return [DatasetEntry(self.list.item(i).text(),
                             self.list.item(i).checkState() == Qt.CheckState.Checked)
                for i in range(self.list.count())]

    def checked(self) -> list[str]:
        return [e.path for e in self.entries() if e.checked]


class MainWindow(QMainWindow):
    def __init__(self, manager: Optional[ProjectManager] = None):
        super().__init__()
        self.manager = manager or ProjectManager()
        self.project: Optional[Project] = None
        self.thread: Optional[QThread] = None
        self.worker: Optional[TrainingWorker] = None
        self.paused_checkpoint: Optional[str] = None
        self.last_checkpoint: Optional[str] = None
        self.detector_device = None  # None = GPU when available (tests use "cpu")
        self._form_task = "detect"
        self._loading = False
        # Matrix currently shown in the analytics ({confusion, classes, prefix}) and the previous
        # run it is compared with in the normalized confusion matrix.
        self.current_eval: Optional[dict] = None
        self.previous_baseline: Optional[dict] = None

        self.setWindowTitle("schaakpoc v3 – Object Detection Trainer")
        self.resize(1500, 950)
        central = QWidget()
        root = QVBoxLayout(central)
        root.addLayout(self._build_project_bar())
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.config_tabs = QTabWidget()
        self.config_tabs.addTab(_scrollable(self._build_dataset_tab()), "Datasets")
        self.config_tabs.addTab(_scrollable(self._build_augmentation_tab()), "Augmentation")
        self.config_tabs.addTab(_scrollable(self._build_training_tab()), "Training && Quotas")
        self.config_tabs.addTab(_scrollable(self._build_checkpoint_tab()), "Checkpoints && Provenance")
        splitter.addWidget(self.config_tabs)
        splitter.addWidget(self._build_run_panel())
        splitter.setSizes([600, 900])
        root.addWidget(splitter, 1)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")

        self._init_projects()

    # ------------------------------------------------------------------ layout
    def _build_project_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        self.project_combo = QComboBox()
        self.project_combo.setMinimumWidth(220)
        self.project_combo.currentTextChanged.connect(self.switch_project)
        self.new_project_button = QPushButton("New project…")
        self.save_project_button = QPushButton("Save project")
        self.delete_project_button = QPushButton("Delete project")
        self.new_project_button.clicked.connect(self.new_project)
        self.save_project_button.clicked.connect(self.save_project)
        self.delete_project_button.clicked.connect(self.delete_project)
        bar.addWidget(QLabel("<b>Project:</b>"))
        bar.addWidget(self.project_combo)
        bar.addWidget(self.new_project_button)
        bar.addWidget(self.save_project_button)
        bar.addWidget(self.delete_project_button)
        bar.addSpacing(20)
        bar.addWidget(QLabel("<b>Task:</b>"))
        self.task_combo = QComboBox()
        for key, label in TASKS.items():
            self.task_combo.addItem(label, key)
        self.task_combo.setToolTip("Detection: YOLO boxes on YOLO folders (images/ + labels/), many "
                                   "objects per image. Classification: one label per (cropped) image.")
        self.task_combo.currentIndexChanged.connect(self.on_task_changed)
        bar.addWidget(self.task_combo)
        bar.addStretch()
        return bar

    def _build_dataset_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.train_folders = FolderList("Training pools (check the folders to use)",
                                        self.on_datasets_changed)
        self.val_folders = FolderList("Validation sets (evaluated every epoch during training)",
                                      self.on_datasets_changed)
        self.test_folders = FolderList("Test sets (final exam – only used after training)",
                                       self.on_datasets_changed)
        layout.addWidget(self.train_folders)
        layout.addWidget(self.val_folders)
        layout.addWidget(self.test_folders)
        sync_row = QHBoxLayout()
        self.sync_yaml_button = QPushButton("🔄 Synchronize YAML class names…")
        self.sync_yaml_button.setToolTip(
            "Merge differently named classes (e.g. W-R and White-Rook) across the detected .yaml "
            "files and re-index the class ids of their YOLO label files.")
        self.sync_yaml_button.clicked.connect(self.open_yaml_sync)
        sync_row.addWidget(QLabel("<small>✓ = .yaml configuration detected</small>"))
        sync_row.addStretch()
        sync_row.addWidget(self.sync_yaml_button)
        layout.addLayout(sync_row)
        help_label = QLabel(
            "<small>Folders contain one sub-folder per class (<code>class_name/xxx.jpg</code>) or are "
            "a YOLO split (<code>images/</code> + <code>labels/</code>, class names from "
            "<code>data.yaml</code>; every labeled box becomes one sample). Without validation "
            "folders 20% of the training pool is held out for validation. Test folders are never "
            "used during training: use <i>Final exam</i> / <i>Manual Review</i> after training.</small>")
        help_label.setWordWrap(True)
        layout.addWidget(help_label)

        shuffle_row = QHBoxLayout()
        self.shuffle_check = QCheckBox("Shuffle training samples every epoch")
        self.seed_check = QCheckBox("Fixed seed:")
        self.seed_spin = _spin(0, 2_147_483_647, 42)
        self.seed_check.toggled.connect(self.seed_spin.setEnabled)
        shuffle_row.addWidget(self.shuffle_check)
        shuffle_row.addStretch()
        shuffle_row.addWidget(self.seed_check)
        shuffle_row.addWidget(self.seed_spin)
        layout.addLayout(shuffle_row)

        balance = QGroupBox("Class balance (selected training pools)")
        balance_layout = QVBoxLayout(balance)
        self.balance_canvas = PlotCanvas(width=5, height=3)
        self.balance_canvas.setMinimumHeight(260)
        balance_layout.addWidget(self.balance_canvas)
        layout.addWidget(balance)
        return tab

    def _build_augmentation_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        group = QGroupBox("Training-split augmentations (torchvision.transforms.Compose)")
        self.aug_group = group
        form = QFormLayout(group)
        self.aug_hflip = QCheckBox("Random Horizontal Flip (p=0.5)")
        self.aug_vflip = QCheckBox("Random Vertical Flip (p=0.5)")
        self.aug_rotation = QCheckBox("Random Rotation")
        self.aug_rotation_deg = _dspin(0, 180, 1, 1, "°")
        self.aug_jitter = QCheckBox("Color Jitter")
        self.aug_brightness = _dspin(0, 2, 2, 0.05)
        self.aug_contrast = _dspin(0, 2, 2, 0.05)
        self.aug_saturation = _dspin(0, 2, 2, 0.05)
        self.aug_affine = QCheckBox("Random Affine / Shear")
        self.aug_affine_deg = _dspin(0, 180, 1, 1, "°")
        self.aug_translate = _dspin(0, 1, 2, 0.05)
        self.aug_shear = _dspin(0, 90, 1, 1, "°")
        self.aug_crop = QCheckBox("Random Resized Crop / Scale")
        self.aug_crop_scale = _dspin(0.05, 1, 2, 0.05)

        form.addRow(self.aug_hflip)
        form.addRow(self.aug_vflip)
        form.addRow(self.aug_rotation)
        form.addRow("    degrees (±)", self.aug_rotation_deg)
        form.addRow(self.aug_jitter)
        form.addRow("    brightness", self.aug_brightness)
        form.addRow("    contrast", self.aug_contrast)
        form.addRow("    saturation", self.aug_saturation)
        form.addRow(self.aug_affine)
        form.addRow("    rotation degrees", self.aug_affine_deg)
        form.addRow("    translate (fraction)", self.aug_translate)
        form.addRow("    shear degrees", self.aug_shear)
        form.addRow(self.aug_crop)
        form.addRow("    minimum scale", self.aug_crop_scale)
        layout.addWidget(group)

        self.aug_preview = QLabel()
        self.aug_preview.setWordWrap(True)
        layout.addWidget(self.aug_preview)
        layout.addStretch()
        for widget in (self.aug_hflip, self.aug_vflip, self.aug_rotation, self.aug_jitter,
                       self.aug_affine, self.aug_crop):
            widget.toggled.connect(self.update_aug_preview)
        for widget in (self.aug_rotation_deg, self.aug_brightness, self.aug_contrast,
                       self.aug_saturation, self.aug_affine_deg, self.aug_translate,
                       self.aug_shear, self.aug_crop_scale):
            widget.valueChanged.connect(self.update_aug_preview)
        return tab

    def _build_training_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        training = QGroupBox("Training")
        form = QFormLayout(training)
        self.epochs_spin = _spin(1, 10_000, 20)
        self.batch_spin = _spin(1, 4096, 16)
        self.lr_spin = _dspin(1e-6, 1.0, 6, 1e-4)
        self.image_size_spin = _spin(32, 1024, 224, " px")
        self.arch_combo = QComboBox()
        self.arch_combo.addItems(ARCHITECTURES)
        self.pretrained_check = QCheckBox("Start from ImageNet weights (download)")
        self.workers_spin = _spin(0, 32, 0)
        self.crop_fraction_spin = _dspin(0, 100, 0, 5, " %")
        self.crop_fraction_spin.setValue(100)
        self.crop_fraction_spin.setToolTip(
            "Share of YOLO images cropped into one sample per piece, drawn at random each epoch. "
            "The other images are trained as full pictures with all their pieces as labels at once "
            "(multi-label).")
        form.addRow("Epochs (this session)", self.epochs_spin)
        form.addRow("Cropped images per epoch", self.crop_fraction_spin)
        form.addRow("Batch size", self.batch_spin)
        form.addRow("Learning rate", self.lr_spin)
        form.addRow("Image size", self.image_size_spin)
        form.addRow("Architecture", self.arch_combo)
        form.addRow(self.pretrained_check)
        form.addRow("Data loader workers", self.workers_spin)
        layout.addWidget(training)

        quota = QGroupBox("Target quota")
        form = QFormLayout(quota)
        self.min_acc_spin = _dspin(0, 100, 1, 1, " %")
        self.target_acc_spin = _dspin(0, 100, 1, 1, " %")
        self.min_conf_spin = _dspin(0, 100, 1, 1, " %")
        self.target_conf_spin = _dspin(0, 100, 1, 1, " %")
        self.window_spin = _spin(1, 1000, 5, " epochs")
        self.stop_on_quota_check = QCheckBox("Stop training once all targets are satisfied")
        form.addRow("Accuracy floor (epoch)", self.min_acc_spin)
        form.addRow("Accuracy target (moving avg)", self.target_acc_spin)
        form.addRow("Confidence floor (epoch mean)", self.min_conf_spin)
        form.addRow("Confidence target (moving avg)", self.target_conf_spin)
        form.addRow("Moving average window", self.window_spin)
        form.addRow(self.stop_on_quota_check)
        layout.addWidget(quota)

        snapshots = QGroupBox("Snapshots && early stopping")
        form = QFormLayout(snapshots)
        self.snapshot_spin = _spin(0, 10_000, 10, " epochs")
        self.snapshot_spin.setSpecialValueText("never")
        self.early_stop_check = QCheckBox("Enable early stopping")
        self.patience_spin = _spin(1, 1000, 5, " epochs")
        self.monitor_combo = QComboBox()
        self.monitor_combo.addItem("validation loss", "val_loss")
        self.monitor_combo.addItem("validation accuracy", "val_acc")
        self.monitor_combo.addItem("validation loss and accuracy (stop when both stall)", "both")
        self.early_stop_check.toggled.connect(self.patience_spin.setEnabled)
        self.early_stop_check.toggled.connect(self.monitor_combo.setEnabled)
        form.addRow("Export full weights every", self.snapshot_spin)
        form.addRow(self.early_stop_check)
        form.addRow("Patience", self.patience_spin)
        form.addRow("Monitor", self.monitor_combo)
        layout.addWidget(snapshots)
        self.classify_groups = (training, quota, snapshots)

        self.detect_box = QGroupBox("Detection training (Ultralytics YOLO, boxes on YOLO folders)")
        form = QFormLayout(self.detect_box)
        self.det_base_combo = QComboBox()
        self.det_base_combo.setEditable(True)
        self.det_base_combo.addItems(DETECTOR_BASE_MODELS)
        self.det_base_combo.setToolTip("Ultralytics weights to start from (downloaded automatically); "
                                       "n = fastest … x = most accurate.")
        self.det_epochs_spin = _spin(1, 10_000, 50)
        self.det_image_size_spin = _spin(32, 4096, 640, " px")
        self.det_batch_spin = _spin(1, 1024, 16)
        self.det_patience_spin = _spin(0, 10_000, 20, " epochs")
        self.det_patience_spin.setSpecialValueText("off")
        self.det_patience_spin.setToolTip("Stop when mAP has not improved for this many epochs.")
        self.det_snapshot_spin = _spin(0, 10_000, 0, " epochs")
        self.det_snapshot_spin.setSpecialValueText("never")
        self.det_workers_spin = _spin(0, 32, 0)
        self.det_mosaic_check = QCheckBox("Mosaic (combine 4 training images into one)")
        self.det_mosaic_check.setToolTip("Ultralytics' standard detection augmentation; turned off "
                                         "automatically for the last 10 epochs.")
        form.addRow("Base model", self.det_base_combo)
        form.addRow("Epochs", self.det_epochs_spin)
        form.addRow("Image size", self.det_image_size_spin)
        form.addRow("Batch size", self.det_batch_spin)
        form.addRow("Early stopping patience", self.det_patience_spin)
        form.addRow("Save weights every", self.det_snapshot_spin)
        form.addRow("Data loader workers", self.det_workers_spin)
        form.addRow(self.det_mosaic_check)
        help_label = QLabel("<small>Uses the checked YOLO training/validation folders (any classes; "
                            "run <i>Synchronize YAML class names</i> first if their names differ). "
                            "Augmentation tab settings are applied box-aware. Results are saved in "
                            "<code>checkpoints/detectors/</code> of the project.</small>")
        help_label.setWordWrap(True)
        form.addRow(help_label)
        layout.insertWidget(0, self.detect_box)
        layout.addStretch()
        return tab

    def _build_checkpoint_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.resume_box = QGroupBox("Resume training")
        resume_layout = QHBoxLayout(self.resume_box)
        self.resume_check = QCheckBox("Resume from:")
        self.resume_edit = QLineEdit()
        self.resume_edit.setPlaceholderText("checkpoint .pt / .pth")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self.browse_checkpoint)
        self.resume_edit.editingFinished.connect(
            lambda: self.show_provenance(self.resume_edit.text().strip()))
        resume_layout.addWidget(self.resume_check)
        resume_layout.addWidget(self.resume_edit, 1)
        resume_layout.addWidget(browse)
        layout.addWidget(self.resume_box)

        self.saved_box = saved = QGroupBox("Saved models in this project")
        saved_layout = QVBoxLayout(saved)
        self.checkpoint_list = QListWidget()
        self.checkpoint_list.currentTextChanged.connect(self.show_provenance)
        row = QHBoxLayout()
        refresh = QPushButton("Refresh")
        use = QPushButton("Use for resume")
        self.import_detector_button = QPushButton("Import detector…")
        self.import_detector_button.setToolTip("Copy existing YOLO detector weights (e.g. from "
                                               "train.py runs) into this project.")
        refresh.clicked.connect(self.refresh_checkpoints)
        use.clicked.connect(self.use_selected_for_resume)
        self.import_detector_button.clicked.connect(self.browse_import_detector)
        row.addWidget(refresh)
        row.addWidget(use)
        row.addWidget(self.import_detector_button)
        row.addStretch()
        saved_layout.addWidget(self.checkpoint_list)
        saved_layout.addLayout(row)
        layout.addWidget(saved)

        self.provenance_box = QGroupBox("Model Provenance")
        self.provenance_layout = QVBoxLayout(self.provenance_box)
        self.provenance_label = QLabel("Select a saved model to see its provenance.")
        self.provenance_label.setWordWrap(True)
        self.provenance_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.provenance_layout.addWidget(self.provenance_label)
        self.milestone_container = QVBoxLayout()
        self.provenance_layout.addLayout(self.milestone_container)
        layout.addWidget(self.provenance_box)
        layout.addStretch()
        return tab

    def _build_run_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        buttons = QHBoxLayout()
        self.start_button = QPushButton("▶ Start training")
        self.pause_button = QPushButton("⏸ Pause")
        self.stop_button = QPushButton("⏹ Stop")
        self.review_button = QPushButton("🔍 Manual Review")
        self.picture_button = QPushButton("🖼 Test own picture")
        self.exam_button = QPushButton("🎓 Final exam (test sets)")
        self.start_button.clicked.connect(self.start_training)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.stop_button.clicked.connect(self.stop_training)
        self.review_button.clicked.connect(self.open_review)
        self.picture_button.clicked.connect(self.open_picture_test)
        self.exam_button.clicked.connect(self.run_final_exam)
        for b in (self.start_button, self.pause_button, self.stop_button, self.review_button,
                  self.picture_button, self.exam_button):
            buttons.addWidget(b)
        layout.addLayout(buttons)

        progress = QFormLayout()
        self.batch_progress = QProgressBar()
        self.run_progress = QProgressBar()
        self.batch_progress.setFormat("%v / %m batches")
        self.run_progress.setFormat("%v / %m epochs")
        self.batch_progress.setMaximum(1)
        self.run_progress.setMaximum(1)
        self.batch_progress.setValue(0)
        self.run_progress.setValue(0)
        progress.addRow("Batch progress", self.batch_progress)
        progress.addRow("Total run progress", self.run_progress)
        layout.addLayout(progress)

        quota_box = QGroupBox("Live quota check (current / target)")
        quota_grid = QGridLayout(quota_box)
        quota_grid.setContentsMargins(6, 2, 6, 2)
        self.quota_bars: dict[str, QProgressBar] = {}
        for i, (key, label) in enumerate(OBJECTIVES.items()):
            bar = QProgressBar()
            bar.setRange(0, 100)
            self.quota_bars[key] = bar
            quota_grid.addWidget(QLabel(label), i // 2, (i % 2) * 2)
            quota_grid.addWidget(bar, i // 2, (i % 2) * 2 + 1)
        layout.addWidget(quota_box)
        self.quota_box = quota_box

        self.det_metrics_box = QGroupBox("Live detection metrics (validation, last epoch)")
        det_grid = QGridLayout(self.det_metrics_box)
        det_grid.setContentsMargins(6, 2, 6, 2)
        self.det_metric_bars: dict[str, QProgressBar] = {}
        for i, (key, label) in enumerate(DETECTION_METRICS.items()):
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setFormat("–")
            self.det_metric_bars[key] = bar
            det_grid.addWidget(QLabel(label), i // 2, (i % 2) * 2)
            det_grid.addWidget(bar, i // 2, (i % 2) * 2 + 1)
        layout.addWidget(self.det_metrics_box)

        baseline_row = QHBoxLayout()
        self.baseline_combo = QComboBox()
        self.baseline_combo.setMinimumWidth(260)
        self.baseline_combo.currentIndexChanged.connect(lambda _i: self.render_normalized_confusion())
        baseline_row.addWidget(QLabel("Normalized matrix Δ baseline:"))
        baseline_row.addWidget(self.baseline_combo)
        baseline_row.addStretch()
        layout.addLayout(baseline_row)

        splitter = QSplitter(Qt.Orientation.Vertical)
        self.analytics = QTabWidget()
        self.cm_canvas = PlotCanvas()
        self.cm_norm_canvas = PlotCanvas()
        self.count_canvas = PlotCanvas()
        self.precision_canvas = PlotCanvas()
        self.conf_canvas = PlotCanvas()
        self.class_quota_table = QTableWidget(0, 4)
        self.class_quota_table.setHorizontalHeaderLabels(
            ["Class", "Correct / needed", "Accuracy / target", "Progress"])
        self.class_quota_table.verticalHeader().setVisible(False)
        self.class_quota_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.class_quota_table.horizontalHeader()
        for column in range(3):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.analytics.addTab(self.cm_canvas, "Confusion Matrix")
        self.analytics.addTab(self.cm_norm_canvas, "Normalized Confusion Matrix")
        self.analytics.addTab(self.count_canvas, "Instance Count")
        self.analytics.addTab(self.precision_canvas, "Precision Curve")
        self.analytics.addTab(self.conf_canvas, "Recall vs. Confidence")
        self.analytics.addTab(self.class_quota_table, "Class Quotas")
        splitter.addWidget(self.analytics)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        splitter.addWidget(self.log_view)
        splitter.setSizes([600, 250])
        layout.addWidget(splitter, 1)
        self._reset_analytics()
        self._set_running(False)
        return panel

    # ---------------------------------------------------------------- projects
    def _init_projects(self):
        names = self.manager.list_projects()
        if not names:
            self.manager.create("default")
            names = self.manager.list_projects()
        self.project_combo.blockSignals(True)
        self.project_combo.addItems(names)
        self.project_combo.blockSignals(False)
        self.switch_project(names[0])

    def switch_project(self, name: str):
        if not name or (self.project is not None and self.project.name == name):
            return
        if self.project is not None:
            self.save_project()
        try:
            self.project = self.manager.load(name)
        except (OSError, ValueError) as e:
            QMessageBox.critical(self, "Load project", f"Could not load project '{name}':\n{e}")
            return
        self._load_form(self.project)
        self.last_checkpoint = None
        self.paused_checkpoint = None
        self.previous_baseline = None
        self._reset_analytics()
        self.refresh_baselines()
        self.refresh_checkpoints()
        self.statusBar().showMessage(f"Loaded project '{name}'")
        self.log(f"Project '{name}' loaded.")

    def new_project(self):
        name, ok = QInputDialog.getText(self, "New project", "Project name:")
        if not ok or not name.strip():
            return
        try:
            project = self.manager.create(name)
        except ValueError as e:
            QMessageBox.warning(self, "New project", str(e))
            return
        self.project_combo.addItem(project.name)
        self.project_combo.setCurrentText(project.name)

    def save_project(self):
        if self.project is None:
            return
        self._collect_form(self.project)
        self.manager.save(self.project)
        self.statusBar().showMessage(f"Project '{self.project.name}' saved", 3000)

    def delete_project(self):
        if self.project is None:
            return
        name = self.project.name
        answer = QMessageBox.question(
            self, "Delete project",
            f"Delete project '{name}' including all of its checkpoints?")
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.manager.delete(name)
        self.project = None
        index = self.project_combo.findText(name)
        self.project_combo.blockSignals(True)
        self.project_combo.removeItem(index)
        self.project_combo.blockSignals(False)
        if self.project_combo.count() == 0:
            self.manager.create("default")
            self.project_combo.blockSignals(True)
            self.project_combo.addItem("default")
            self.project_combo.blockSignals(False)
        self.switch_project(self.project_combo.currentText())

    # -------------------------------------------------------- form <-> project
    def _load_form(self, p: Project):
        self._loading = True
        self.train_folders.set_entries(p.train_folders)
        self.val_folders.set_entries(p.val_folders)
        self.test_folders.set_entries(p.test_folders)
        self.shuffle_check.setChecked(p.shuffle)
        self.seed_check.setChecked(p.seed is not None)
        self.seed_spin.setValue(p.seed if p.seed is not None else 42)
        self.seed_spin.setEnabled(p.seed is not None)

        a = p.augmentation
        self.aug_hflip.setChecked(a.hflip)
        self.aug_vflip.setChecked(a.vflip)
        self.aug_rotation.setChecked(a.rotation)
        self.aug_rotation_deg.setValue(a.rotation_degrees)
        self.aug_jitter.setChecked(a.color_jitter)
        self.aug_brightness.setValue(a.brightness)
        self.aug_contrast.setValue(a.contrast)
        self.aug_saturation.setValue(a.saturation)
        self.aug_affine.setChecked(a.affine)
        self.aug_affine_deg.setValue(a.affine_degrees)
        self.aug_translate.setValue(a.affine_translate)
        self.aug_shear.setValue(a.affine_shear)
        self.aug_crop.setChecked(a.resized_crop)
        self.aug_crop_scale.setValue(a.crop_scale_min)

        t = p.training
        self.epochs_spin.setValue(t.epochs)
        self.batch_spin.setValue(t.batch_size)
        self.lr_spin.setValue(t.learning_rate)
        self.image_size_spin.setValue(t.image_size)
        self.arch_combo.setCurrentText(t.architecture)
        self.pretrained_check.setChecked(t.pretrained)
        self.workers_spin.setValue(t.num_workers)
        self.crop_fraction_spin.setValue(t.crop_fraction * 100)

        q = p.quota
        self.min_acc_spin.setValue(q.min_accuracy * 100)
        self.target_acc_spin.setValue(q.target_avg_accuracy * 100)
        self.min_conf_spin.setValue(q.min_confidence * 100)
        self.target_conf_spin.setValue(q.target_avg_confidence * 100)
        self.window_spin.setValue(q.moving_average_window)
        self.stop_on_quota_check.setChecked(q.stop_on_quota)
        self.snapshot_spin.setValue(q.snapshot_every)
        self.early_stop_check.setChecked(q.early_stopping)
        self.patience_spin.setValue(q.patience)
        self.patience_spin.setEnabled(q.early_stopping)
        self.monitor_combo.setCurrentIndex(max(0, self.monitor_combo.findData(q.monitor)))
        self.monitor_combo.setEnabled(q.early_stopping)

        d = p.detector
        self.det_base_combo.setCurrentText(d.base_model)
        self.det_epochs_spin.setValue(d.epochs)
        self.det_image_size_spin.setValue(d.image_size)
        self.det_batch_spin.setValue(d.batch_size)
        self.det_patience_spin.setValue(d.patience)
        self.det_snapshot_spin.setValue(d.snapshot_every)
        self.det_workers_spin.setValue(d.num_workers)
        self.det_mosaic_check.setChecked(d.mosaic)

        self.task_combo.blockSignals(True)
        self.task_combo.setCurrentIndex(max(0, self.task_combo.findData(p.task)))
        self.task_combo.blockSignals(False)
        self._load_resume(p)
        self._loading = False
        self.apply_task_ui()
        self.update_aug_preview()
        self.on_datasets_changed()

    def _load_resume(self, p: Project):
        """Resume fields show the classifier checkpoint or the detector to continue from."""
        self._form_task = p.task
        resume = p.detector.resume_from if p.task == "detect" else p.resume_checkpoint
        self.resume_check.setChecked(bool(resume))
        self.resume_edit.setText(resume or "")

    def _collect_form(self, p: Project):
        p.train_folders = self.train_folders.entries()
        p.val_folders = self.val_folders.entries()
        p.test_folders = self.test_folders.entries()
        p.shuffle = self.shuffle_check.isChecked()
        p.seed = self.seed_spin.value() if self.seed_check.isChecked() else None

        a = p.augmentation
        a.hflip = self.aug_hflip.isChecked()
        a.vflip = self.aug_vflip.isChecked()
        a.rotation = self.aug_rotation.isChecked()
        a.rotation_degrees = self.aug_rotation_deg.value()
        a.color_jitter = self.aug_jitter.isChecked()
        a.brightness = self.aug_brightness.value()
        a.contrast = self.aug_contrast.value()
        a.saturation = self.aug_saturation.value()
        a.affine = self.aug_affine.isChecked()
        a.affine_degrees = self.aug_affine_deg.value()
        a.affine_translate = self.aug_translate.value()
        a.affine_shear = self.aug_shear.value()
        a.resized_crop = self.aug_crop.isChecked()
        a.crop_scale_min = self.aug_crop_scale.value()

        t = p.training
        t.epochs = self.epochs_spin.value()
        t.batch_size = self.batch_spin.value()
        t.learning_rate = self.lr_spin.value()
        t.image_size = self.image_size_spin.value()
        t.architecture = self.arch_combo.currentText()
        t.pretrained = self.pretrained_check.isChecked()
        t.num_workers = self.workers_spin.value()
        t.crop_fraction = self.crop_fraction_spin.value() / 100

        q = p.quota
        q.min_accuracy = self.min_acc_spin.value() / 100
        q.target_avg_accuracy = self.target_acc_spin.value() / 100
        q.min_confidence = self.min_conf_spin.value() / 100
        q.target_avg_confidence = self.target_conf_spin.value() / 100
        q.moving_average_window = self.window_spin.value()
        q.stop_on_quota = self.stop_on_quota_check.isChecked()
        q.snapshot_every = self.snapshot_spin.value()
        q.early_stopping = self.early_stop_check.isChecked()
        q.patience = self.patience_spin.value()
        q.monitor = self.monitor_combo.currentData()

        d = p.detector
        d.base_model = self.det_base_combo.currentText().strip() or d.base_model
        d.epochs = self.det_epochs_spin.value()
        d.image_size = self.det_image_size_spin.value()
        d.batch_size = self.det_batch_spin.value()
        d.patience = self.det_patience_spin.value()
        d.snapshot_every = self.det_snapshot_spin.value()
        d.num_workers = self.det_workers_spin.value()
        d.mosaic = self.det_mosaic_check.isChecked()

        resume = self.resume_edit.text().strip()
        resume = resume if self.resume_check.isChecked() and resume else None
        if getattr(self, "_form_task", p.task) == "detect":
            d.resume_from = resume
        else:
            p.resume_checkpoint = resume
        p.task = self.task_combo.currentData() or p.task

    # --------------------------------------------------------------- datasets
    @property
    def task(self) -> str:
        return self.task_combo.currentData() or "detect"

    def on_task_changed(self, _index: int = 0):
        if self._loading or self.project is None:
            return
        self._collect_form(self.project)  # stores the resume field of the previous task
        self._load_resume(self.project)
        self.last_checkpoint = None
        self.previous_baseline = None
        self.apply_task_ui()
        self._reset_analytics()
        self.refresh_baselines()
        self.refresh_checkpoints()
        self.update_aug_preview()
        self.save_project()
        self.log(f"Task: {TASKS[self.task]}")

    def apply_task_ui(self):
        detect = self.task == "detect"
        self.setWindowTitle("schaakpoc v3 – "
                            + ("Object Detection Trainer" if detect else "Image Classifier Trainer"))
        self.detect_box.setVisible(detect)
        for group in self.classify_groups:
            group.setVisible(not detect)
        self.quota_box.setVisible(not detect)
        self.det_metrics_box.setVisible(detect)
        self.analytics.setTabVisible(self.analytics.indexOf(self.class_quota_table), not detect)
        self.analytics.setTabText(self.analytics.indexOf(self.precision_canvas),
                                  "Metrics per Epoch" if detect else "Precision Curve")
        self.aug_group.setTitle("Training augmentations (applied box-aware by Ultralytics)" if detect
                                else "Training-split augmentations (torchvision.transforms.Compose)")
        self.aug_contrast.setEnabled(not detect)
        self.aug_contrast.setToolTip("Not available for detection (Ultralytics has no contrast "
                                     "augmentation)" if detect else "")
        self.resume_box.setTitle("Continue training from a detector (fine-tune)" if detect
                                 else "Resume training")
        self.resume_check.setText("Continue from:" if detect else "Resume from:")
        self.resume_edit.setPlaceholderText("detector weights .pt" if detect
                                            else "checkpoint .pt / .pth")
        self.saved_box.setTitle("Saved detectors in this project" if detect
                                else "Saved models in this project")
        self.import_detector_button.setVisible(detect)

    def _safe_counts(self, folders: list[str]) -> dict[str, int]:
        existing = [f for f in folders if Path(f).is_dir()]
        missing = sorted(set(folders) - set(existing))
        if missing:
            self.log("Warning: folder(s) not found: " + ", ".join(missing))
        counts: dict[str, int] = {}
        for folder in existing:
            try:
                folder_counts = class_counts([folder])
            except (OSError, ValueError) as e:
                self.log(f"Warning: {e}")
                continue
            for name, n in folder_counts.items():
                counts[name] = counts.get(name, 0) + n
        return dict(sorted(counts.items()))

    def on_datasets_changed(self):
        if self._loading:
            return
        train = self._safe_counts(self.train_folders.checked())
        val = self._safe_counts(self.val_folders.checked())
        test = self._safe_counts(self.test_folders.checked())
        self.balance_canvas.plot_counts(train, "Class balance – training pools")
        self.count_canvas.plot_counts(train, "Instance Count (selected datasets)",
                                      {"validation": val, "test": test})

    def open_yaml_sync(self):
        """Harmonize the class names of all detected .yaml files (and re-index their labels)."""
        lists = (self.train_folders, self.val_folders, self.test_folders)
        for folder_list in lists:
            folder_list.refresh_badges()
        paths = unique_yaml_paths(p for folder_list in lists for p in folder_list.yaml_paths())
        if not paths:
            QMessageBox.information(self, "Synchronize YAML class names",
                                    "No .yaml configuration was detected in the dataset folders.")
            return
        dialog = YamlSyncDialog(paths, self)
        if not dialog.exec() or dialog.report is None:
            return
        report = dialog.report
        self.log(f"YAML class names synchronized: {len(report['yaml_files'])} .yaml file(s) "
                 f"rewritten, {report['boxes']} label(s) in {report['label_files']} file(s) "
                 f"re-indexed. Classes: " + ", ".join(report["names"]))
        for folder_list in lists:
            folder_list.refresh_badges()
        self.on_datasets_changed()

    def update_aug_preview(self):
        if self._loading or self.project is None:
            return
        self._collect_form(self.project)
        if self.task == "detect":
            settings = ultralytics_augmentation(self.project.augmentation, self.project.detector.mosaic)
            active = [f"{k}={v:g}" for k, v in settings.items() if v]
            self.aug_preview.setText("<b>Ultralytics settings:</b> "
                                     + (html.escape(", ".join(active)) or "(no augmentation)"))
            return
        steps = enabled_augmentations(self.project.augmentation)
        self.aug_preview.setText("<b>Pipeline:</b> Resize/Crop → "
                                 + (" → ".join(html.escape(s) for s in steps) or "(no augmentation)")
                                 + " → ToTensor → Normalize")

    # -------------------------------------------------------------- training
    def _set_running(self, running: bool):
        self.start_button.setEnabled(not running)
        self.pause_button.setEnabled(running)
        self.stop_button.setEnabled(running)
        self.config_tabs.setEnabled(not running)
        for w in (self.project_combo, self.new_project_button, self.delete_project_button,
                  self.save_project_button, self.task_combo):
            w.setEnabled(not running)
        self.review_button.setEnabled(not running)
        self.picture_button.setEnabled(not running)
        self.exam_button.setEnabled(not running)
        if not running:
            self.pause_button.setText("⏸ Pause")

    def is_training(self) -> bool:
        return self.thread is not None and self.thread.isRunning()

    def build_job(self) -> TrainJob:
        p = self.project
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = self.manager.project_dir(p.name) / "checkpoints" / f"run_{stamp}"
        return TrainJob(
            train_folders=p.checked_train_folders(),
            val_folders=p.checked_val_folders(),
            output_dir=str(output),
            shuffle=p.shuffle,
            seed=p.seed,
            augmentation=p.augmentation,
            quota=p.quota,
            training=p.training,
            resume_checkpoint=p.resume_checkpoint,
        )

    def start_training(self):
        if self.project is None or self.is_training():
            return
        if self.task == "detect":
            self.start_detector_training()
            return
        self.save_project()
        job = self.build_job()
        if not job.train_folders:
            QMessageBox.warning(self, "Start training", "Check at least one training folder.")
            return
        if job.resume_checkpoint and not Path(job.resume_checkpoint).is_file():
            QMessageBox.warning(self, "Start training",
                                f"Checkpoint not found:\n{job.resume_checkpoint}")
            return
        self._reset_analytics(keep_counts=True)
        self.previous_baseline = self._latest_evaluation(test=False)
        self.batch_progress.setValue(0)
        self.run_progress.setMaximum(job.training.epochs)
        self.run_progress.setValue(0)
        self.paused_checkpoint = None
        self.log(f"=== Training session started ({job.output_dir}) ===")

        self.thread, self.worker = start_worker(job)
        self.worker.log.connect(self.log)
        self.worker.batch_progress.connect(self.on_batch_progress)
        self.worker.epoch_progress.connect(self.on_epoch_progress)
        self.worker.evaluated.connect(self.on_evaluated)
        self.worker.milestone.connect(self.on_milestone)
        self.worker.paused.connect(self.on_paused)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.thread.finished.connect(self.on_thread_finished)
        self._set_running(True)
        self.thread.start()

    def toggle_pause(self):
        if self.worker is None:
            return
        pausing = not self.worker.trainer.is_paused
        self.worker.set_paused(pausing)
        self.pause_button.setText("▶ Resume" if pausing else "⏸ Pause")
        if not pausing:
            self.paused_checkpoint = None
            self.review_button.setEnabled(False)
            self.picture_button.setEnabled(False)
        self.statusBar().showMessage("Pausing after the current batch…" if pausing else "Resumed")

    def stop_training(self):
        if self.worker is not None:
            self.worker.request_stop()
            self.statusBar().showMessage("Stopping after the current batch…")

    def on_batch_progress(self, i: int, total: int):
        self.batch_progress.setMaximum(max(1, total))
        self.batch_progress.setValue(i)

    def on_epoch_progress(self, e: int, total: int):
        self.run_progress.setMaximum(max(1, total))
        self.run_progress.setValue(e)

    def on_evaluated(self, payload: dict):
        classes = payload["classes"]
        self.cm_canvas.plot_confusion(payload["confusion"], classes)
        self.current_eval = {"confusion": payload["confusion"], "classes": classes, "prefix": ""}
        self.render_normalized_confusion()
        self.update_quota_counters(payload.get("quota_values", {}), payload["confusion"], classes)
        self.precision_canvas.plot_precision(payload["history"])
        self.conf_canvas.plot_confidence_curves(payload["curves"])
        self.last_checkpoint = payload["checkpoint"]

    def on_milestone(self, milestone: dict):
        self.statusBar().showMessage("🏁 " + milestone["message"], 10000)

    def on_paused(self, checkpoint_path: str):
        self.paused_checkpoint = checkpoint_path
        self.review_button.setEnabled(True)
        self.picture_button.setEnabled(True)
        self.statusBar().showMessage("Paused – use Manual Review to inspect the current model")

    def on_finished(self, summary: dict):
        self.paused_checkpoint = None
        if self.project is not None:
            record = {k: summary[k] for k in ("finished_at", "stop_reason", "total_epochs_completed",
                                              "epochs_this_session", "last_checkpoint",
                                              "best_checkpoint", "metrics", "milestones")}
            if self.current_eval is not None and not self.current_eval["prefix"]:
                record["confusion"] = self.current_eval["confusion"]
                record["classes"] = self.current_eval["classes"]
            self.project.evaluations.append(record)
            for path in (summary["last_checkpoint"], summary["best_checkpoint"]):
                if path and path not in self.project.checkpoints:
                    self.project.checkpoints.append(path)
            self.manager.save(self.project)
            self.refresh_baselines()
        if summary["last_checkpoint"]:
            self.last_checkpoint = summary["last_checkpoint"]
        self.refresh_checkpoints(select=self.last_checkpoint)
        reason = STOP_REASONS.get(summary["stop_reason"], summary["stop_reason"])
        self.log(f"=== Training finished: {reason} ===")
        message = (f"Training finished: {reason}.\n"
                   f"Total epochs trained: {summary['total_epochs_completed']}")
        if summary["milestones"]:
            message += "\n\nMilestones:\n" + "\n".join(summary["milestones"])
        QMessageBox.information(self, "Training finished", message)

    def on_failed(self, error: str):
        self.paused_checkpoint = None
        self.log(f"=== Training failed: {error} ===")
        QMessageBox.critical(self, "Training failed", error)

    def on_thread_finished(self):
        if self.thread is not None:
            self.thread.deleteLater()
        self.thread = None
        self.worker = None
        self._set_running(False)

    # ------------------------------------------------------------ checkpoints
    def browse_checkpoint(self):
        detect = self.task == "detect"
        path, _ = QFileDialog.getOpenFileName(
            self, "Select detector weights" if detect else "Select checkpoint", "",
            "YOLO weights (*.pt)" if detect else "PyTorch checkpoints (*.pt *.pth)")
        if path:
            self.resume_edit.setText(path)
            self.resume_check.setChecked(True)
            if detect:
                self.show_provenance(path)
                return
            if self.project is not None and path not in self.project.checkpoints:
                self.project.checkpoints.append(path)
            self.refresh_checkpoints(select=str(Path(path).resolve()))

    def refresh_checkpoints(self, select: Optional[str] = None):
        if self.project is None:
            return
        current = select or self.checkpoint_list.currentItem() and self.checkpoint_list.currentItem().text()
        self.checkpoint_list.blockSignals(True)
        self.checkpoint_list.clear()
        if self.task == "detect":
            paths = self.project_detectors()
            fallback = self.project.detector.resume_from
        else:
            paths = self.manager.list_checkpoints(self.project)
            fallback = self.project.resume_checkpoint
        self.checkpoint_list.addItems(paths)
        self.checkpoint_list.blockSignals(False)
        if current in paths:
            self.checkpoint_list.setCurrentRow(paths.index(current))
        else:
            self.show_provenance(fallback or "")

    def use_selected_for_resume(self):
        item = self.checkpoint_list.currentItem()
        if item is not None:
            self.resume_edit.setText(item.text())
            self.resume_check.setChecked(True)

    def selected_checkpoint(self) -> Optional[str]:
        item = self.checkpoint_list.currentItem()
        return item.text() if item is not None else None

    def _clear_milestones(self):
        while self.milestone_container.count():
            widget = self.milestone_container.takeAt(0).widget()
            if widget is not None:
                widget.deleteLater()

    def _milestone_card(self, title: str, text: str, achieved: bool) -> QFrame:
        card = QFrame()
        color, background = ("#2e7d32", "#e8f5e9") if achieved else ("#757575", "#f5f5f5")
        card.setStyleSheet(f"QFrame {{ border: 1px solid {color}; border-radius: 6px; "
                           f"background: {background}; }} QLabel {{ border: none; color: #212121; }}")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(8, 4, 8, 4)
        state = "✅ Achieved" if achieved else "⏳ Pending"
        label = QLabel(f"<b>{html.escape(title)}</b> – {state}<br>{html.escape(text)}")
        label.setWordWrap(True)
        layout.addWidget(label)
        return card

    def show_provenance(self, path: str):
        self._clear_milestones()
        if not path:
            self.provenance_label.setText("Select a saved model to see its provenance.")
            return
        if self.task == "detect":
            self.provenance_label.setText(self.detector_provenance_html(path))
            return
        try:
            lineage = read_lineage(path)
        except Exception as e:  # unreadable / foreign checkpoint
            self.provenance_label.setText(f"Could not read provenance of {html.escape(path)}:<br>"
                                          f"{html.escape(str(e))}")
            return
        self.provenance_label.setText(self.provenance_html(path, lineage))
        if self.project is not None:
            self._collect_form(self.project)
            tracker = QuotaTracker(self.project.quota, lineage.quota_history)
            for key, milestone in tracker.status().items():
                threshold = objective_threshold(self.project.quota, key) * 100
                if milestone is not None:
                    card = self._milestone_card(OBJECTIVES[key], milestone["message"], True)
                else:
                    card = self._milestone_card(OBJECTIVES[key], f"Target {threshold:.0f}% not yet reached",
                                                False)
                self.milestone_container.addWidget(card)

    @staticmethod
    def detector_provenance_html(path: str) -> str:
        esc = html.escape
        info = read_run_info(path)
        parts = [f"<b>Detector:</b> {esc(path)}"]
        if info is None:
            parts.append("Imported weights (no training record from this app).")
            return "<br>".join(parts)
        m = info.get("metrics") or {}
        reason = STOP_REASONS.get(info.get("stop_reason"), info.get("stop_reason"))
        parts += [f"<b>Trained:</b> {esc(str(info.get('finished_at')))} – {info.get('epochs')} "
                  f"epoch(s), {esc(str(reason))}, best epoch {info.get('best_epoch')}",
                  f"<b>Started from:</b> {esc(str(info.get('base_model')))}",
                  "<b>Validation (best weights):</b> " + ", ".join(
                      f"{label} {m.get(key, 0) * 100:.1f}%" for key, label in DETECTION_METRICS.items()),
                  "<b>Classes:</b> " + esc(", ".join(info.get("classes", []))),
                  "<b>Training sources:</b><br>"
                  + "<br>".join(f"• {esc(f)}" for f in info.get("train_folders", [])),
                  "<b>Validation sources:</b><br>"
                  + ("<br>".join(f"• {esc(f)}" for f in info.get("val_folders", []))
                     or "(training folders)"),
                  "<b>Augmentation:</b> " + esc(", ".join(
                      f"{k}={v:g}" for k, v in (info.get("augmentation") or {}).items() if v) or "none")]
        return "<br>".join(parts)

    @staticmethod
    def provenance_html(path: str, lineage: Lineage) -> str:
        esc = html.escape
        sources = lineage.dataset_sources()
        parts = [f"<b>Model:</b> {esc(path)}",
                 f"<b>Total cumulative epochs trained:</b> {lineage.total_epochs_completed}",
                 f"<b>Training sessions:</b> {len(lineage.sessions)}"]
        if lineage.sessions:
            active = lineage.sessions[-1]
            parts.append("<b>Active training sources:</b><br>"
                         + "<br>".join(f"• {esc(f)}" for f in active.get("train_folders", [])))
            parts.append("<b>Active validation sources:</b><br>"
                         + ("<br>".join(f"• {esc(f)}" for f in session_val_folders(active))
                            or "(held-out split of training pool)"))
        if len(lineage.sessions) > 1:
            parts.append("<b>Historical training sources:</b><br>"
                         + "<br>".join(f"• {esc(f)}" for f in sources["train"]))
            lines = []
            for s in lineage.sessions:
                lines.append(f"• epochs {s.get('start_epoch')}→{s.get('end_epoch')} "
                             f"({esc(str(s.get('stop_reason')))}): "
                             + esc(", ".join(Path(f).name for f in s.get("train_folders", []))))
            parts.append("<b>Session lineage:</b><br>" + "<br>".join(lines))
        if lineage.best:
            parts.append(f"<b>Best weights:</b> epoch {lineage.best['epoch']}, "
                         f"val acc {lineage.best['val_acc'] * 100:.1f}% – {esc(lineage.best['path'])}")
        if lineage.snapshots:
            parts.append("<b>Snapshots:</b> " + ", ".join(
                f"epoch {s['epoch']} ({s['val_acc'] * 100:.1f}%)" for s in lineage.snapshots))
        if lineage.quota_history:
            parts.append("<b>Quota history:</b><br>"
                         + "<br>".join(f"• {esc(m['message'])}" for m in lineage.quota_history))
        return "<br>".join(parts)

    # ----------------------------------------------------------------- review
    def review_checkpoint(self) -> Optional[str]:
        """Model to inspect: the paused model while training, else the selected or last one."""
        if self.is_training():
            return self.paused_checkpoint
        return self.selected_checkpoint() or self.last_checkpoint

    def open_review(self):
        if self.project is None:
            return
        checkpoint = self.review_checkpoint()
        if not checkpoint:
            QMessageBox.information(self, "Manual Review",
                                    "No model available yet. Train a model or select a saved one.")
            return
        self._collect_form(self.project)
        source, folders = self.review_folders()
        if not folders:
            QMessageBox.information(self, "Manual Review",
                                    "Check a test, validation or training folder first.")
            return
        if self.task == "detect":
            try:
                session = DetectionReviewSession(checkpoint, folders, self.project.seed)
            except Exception as e:
                QMessageBox.warning(self, "Manual Review", f"Cannot start review:\n{e}")
                return
            DetectionReviewDialog(session, checkpoint, self, source).exec()
            return
        try:
            session = ReviewSession(checkpoint, folders, self.project.seed)
        except Exception as e:
            QMessageBox.warning(self, "Manual Review", f"Cannot start review:\n{e}")
            return
        ReviewDialog(session, checkpoint, self, source).exec()

    def open_picture_test(self):
        """Box every object in a user picture with a project detector; optionally a classifier
        checkpoint re-labels each box (two-stage)."""
        if self.project is None:
            return
        self._collect_form(self.project)
        p = self.project
        classifiers = sorted(self.manager.list_checkpoints(p), key=lambda c: Path(c).stat().st_mtime,
                             reverse=True)
        detectors = self.project_detectors()
        if self.task == "detect":
            preferred = self.review_checkpoint()
            if preferred:
                detectors = [preferred] + [d for d in detectors if d != preferred]
            checkpoint = None  # the detector's own names; a classifier can still be chosen
        else:
            checkpoint = self.review_checkpoint() or (classifiers[0] if classifiers else None)
            if checkpoint and checkpoint not in classifiers:
                classifiers.insert(0, checkpoint)
        # Only used to give class_N placeholders their yaml names.
        folders = p.checked_train_folders() + p.checked_val_folders() + p.checked_test_folders()
        tester = ImageTester(folders=folders)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            tester.load_classifier(checkpoint)
        except Exception as e:
            self.log(f"Test own picture: cannot load classifier {checkpoint}: {e}")
        finally:
            QApplication.restoreOverrideCursor()
        ImageTestDialog(tester, self, detectors, classifiers, self.import_project_detector).exec()

    # --------------------------------------------------------------- detector
    def project_detectors(self) -> list[str]:
        """Detector weights stored in this project, newest first."""
        if self.project is None:
            return []
        return list_detectors(self.manager.project_dir(self.project.name))

    def import_project_detector(self, source: str) -> str:
        path = import_detector(self.manager.project_dir(self.project.name), source)
        self.log(f"Detector imported into the project: {path}")
        if self.task == "detect":
            self.refresh_checkpoints(select=path)
        return path

    def browse_import_detector(self):
        if self.project is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Import YOLO detector weights", "",
                                              "YOLO weights (*.pt);;All files (*)")
        if not path:
            return
        try:
            self.import_project_detector(path)
        except Exception as e:
            QMessageBox.warning(self, "Import detector", f"Cannot import the detector:\n{e}")

    def build_detector_job(self) -> DetectorJob:
        p = self.project
        return DetectorJob(train_folders=p.checked_train_folders(), val_folders=p.checked_val_folders(),
                           project_dir=str(self.manager.project_dir(p.name)), config=p.detector,
                           augmentation=p.augmentation, seed=p.seed, device=self.detector_device)

    def start_detector_training(self):
        self.save_project()
        job = self.build_detector_job()
        if not job.train_folders:
            QMessageBox.warning(self, "Start training", "Check at least one YOLO training folder.")
            return
        if job.config.resume_from and not Path(job.config.resume_from).is_file():
            QMessageBox.warning(self, "Start training",
                                f"Detector not found:\n{job.config.resume_from}")
            return
        self._reset_analytics(keep_counts=True)
        self.previous_baseline = self._latest_evaluation(test=False)
        self.batch_progress.setValue(0)
        self.run_progress.setMaximum(job.config.epochs)
        self.run_progress.setValue(0)
        self.paused_checkpoint = None
        self.log(f"=== Detection training started ({job.config.resume_from or job.config.base_model}, "
                 f"{job.config.epochs} epochs) ===")
        self.thread, self.worker = start_detector_worker(job)
        self.worker.log.connect(self.log)
        self.worker.batch_progress.connect(self.on_batch_progress)
        self.worker.epoch_progress.connect(self.on_epoch_progress)
        self.worker.evaluated.connect(self.on_detector_evaluated)
        self.worker.paused.connect(self.on_paused)
        self.worker.finished.connect(self.on_detector_finished)
        self.worker.failed.connect(self.on_failed)
        self.thread.finished.connect(self.on_thread_finished)
        self._set_running(True)
        self.thread.start()

    def show_detection_metrics(self, metrics: dict):
        for key, bar in self.det_metric_bars.items():
            value = metrics.get(key)
            bar.setValue(0 if value is None else round(value * 100))
            bar.setFormat("–" if value is None else f"{value * 100:.1f}%")

    def on_detector_evaluated(self, payload: dict):
        self.show_detection_metrics(payload["metrics"])
        self.precision_canvas.plot_detection_history(payload["history"])
        if payload.get("confusion"):
            self.cm_canvas.plot_confusion(payload["confusion"], payload["classes"])
            self.current_eval = {"confusion": payload["confusion"], "classes": payload["classes"],
                                 "prefix": ""}
            self.render_normalized_confusion()
        if payload.get("curves"):
            self.conf_canvas.plot_confidence_curves(payload["curves"])

    def on_detector_finished(self, summary: dict):
        self.paused_checkpoint = None
        if self.project is not None:
            record = {"type": "detect",
                      **{k: summary[k] for k in ("finished_at", "stop_reason", "epochs", "best_epoch",
                                                 "metrics", "weights", "output_dir")}}
            if self.current_eval is not None and not self.current_eval["prefix"]:
                record["confusion"] = self.current_eval["confusion"]
                record["classes"] = self.current_eval["classes"]
            self.project.evaluations.append(record)
            self.manager.save(self.project)
            self.refresh_baselines()
        if summary["weights"]:
            self.last_checkpoint = summary["weights"]
        self.refresh_checkpoints(select=self.last_checkpoint)
        if summary["metrics"]:
            self.show_detection_metrics(summary["metrics"])
        reason = STOP_REASONS.get(summary["stop_reason"], summary["stop_reason"])
        self.log(f"=== Detection training finished: {reason} – {summary['weights']} ===")
        metrics = "\n".join(f"{label}: {summary['metrics'].get(key, 0) * 100:.1f}%"
                            for key, label in DETECTION_METRICS.items()) if summary["metrics"] else ""
        QMessageBox.information(self, "Training finished",
                                f"Training finished: {reason}.\nEpochs trained: {summary['epochs']}"
                                f"\n\nBest weights (validation):\n{metrics}\n\n{summary['weights']}")

    def review_folders(self) -> tuple[str, list[str]]:
        """Images for Manual Review: validation sets while paused, test sets after training."""
        p = self.project
        if self.is_training():
            candidates = [("validation sets", p.checked_val_folders()),
                          ("training pools", p.checked_train_folders())]
        else:
            candidates = [("test sets (final exam)", p.checked_test_folders()),
                          ("validation sets", p.checked_val_folders()),
                          ("training pools", p.checked_train_folders())]
        for source, folders in candidates:
            if folders:
                return source, folders
        return "", []

    def run_final_exam(self):
        """Evaluate the selected model on the test sets (never used during training)."""
        if self.project is None or self.is_training():
            return
        checkpoint = self.selected_checkpoint() or self.last_checkpoint
        if not checkpoint:
            QMessageBox.information(self, "Final exam",
                                    "No model available yet. Train a model or select a saved one.")
            return
        self._collect_form(self.project)
        folders = self.project.checked_test_folders()
        if not folders:
            QMessageBox.information(self, "Final exam", "Check at least one test folder first.")
            return
        if self.task == "detect":
            self._final_exam_detection(checkpoint, folders)
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result = evaluate_checkpoint(checkpoint, folders, self.project.training.batch_size,
                                         self.project.training.num_workers)
        except Exception as e:
            QMessageBox.warning(self, "Final exam", f"Cannot evaluate the test sets:\n{e}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.cm_canvas.plot_confusion(result["confusion"], result["classes"], prefix="Test – ")
        confusion = [list(map(int, row)) for row in result["confusion"]]
        self.previous_baseline = self._latest_evaluation(test=True)
        self.current_eval = {"confusion": confusion, "classes": list(result["classes"]),
                             "prefix": "Test – "}
        self.render_normalized_confusion()
        self.update_quota_counters({"accuracy_floor": result["accuracy"],
                                    "confidence_floor": result["mean_confidence"]},
                                   confusion, list(result["classes"]))
        self.conf_canvas.plot_confidence_curves(result["curves"])
        summary = (f"Test samples: {result['samples']}\n"
                   f"Accuracy: {result['accuracy'] * 100:.1f}%\n"
                   f"Loss: {result['loss']:.4f}\n"
                   f"Mean confidence: {result['mean_confidence'] * 100:.1f}%\n"
                   f"Precision: {result['precision'] * 100:.1f}%  Recall: {result['recall'] * 100:.1f}%")
        if result["skipped_classes"]:
            summary += "\n\nSkipped classes unknown to the model: " + ", ".join(result["skipped_classes"])
        if result["seen_in_training"]:
            summary += ("\n\nWarning: these test folders were used during training/validation, so "
                        "this is not an independent exam:\n" + "\n".join(result["seen_in_training"]))
        self.log("=== Final exam on test sets ===\n" + summary)
        self.project.evaluations.append({
            "type": "test",
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            **{k: result[k] for k in ("checkpoint", "test_folders", "samples", "loss", "accuracy",
                                      "mean_confidence", "precision", "recall", "seen_in_training")},
            "confusion": confusion,
            "classes": list(result["classes"]),
        })
        self.manager.save(self.project)
        self.refresh_baselines()
        QMessageBox.information(self, "Final exam", summary)

    def _final_exam_detection(self, weights: str, folders: list[str]):
        d = self.project.detector
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result = evaluate_detector(weights, folders, d.image_size, d.batch_size,
                                       device=self.detector_device)
        except Exception as e:
            QMessageBox.warning(self, "Final exam", f"Cannot evaluate the test sets:\n{e}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.previous_baseline = self._latest_evaluation(test=True)
        self.cm_canvas.plot_confusion(result["confusion"], result["classes"], prefix="Test – ")
        self.current_eval = {"confusion": result["confusion"], "classes": result["classes"],
                             "prefix": "Test – "}
        self.render_normalized_confusion()
        if result["curves"]:
            self.conf_canvas.plot_confidence_curves(result["curves"])
        self.show_detection_metrics(result)
        used = {str(Path(f).resolve()) for f in
                self.project.checked_train_folders() + self.project.checked_val_folders()}
        seen = [f for f in folders if str(Path(f).resolve()) in used]
        summary = (f"Test images: {result['images']}\n" + "\n".join(
            f"{label}: {result[key] * 100:.1f}%" for key, label in DETECTION_METRICS.items()))
        if result["per_class"]:
            summary += "\n\nPer class (P / R / mAP50):\n" + "\n".join(
                f"{c['class']}: {c['precision'] * 100:.0f}% / {c['recall'] * 100:.0f}% / "
                f"{c['mAP50'] * 100:.0f}%" for c in result["per_class"])
        if seen:
            summary += ("\n\nWarning: these test folders were used during training/validation, so "
                        "this is not an independent exam:\n" + "\n".join(seen))
        self.log("=== Final exam on test sets ===\n" + summary)
        self.project.evaluations.append({
            "type": "detect_test",
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            **{k: result[k] for k in ("weights", "test_folders", "images", "precision", "recall",
                                      "mAP50", "mAP50_95", "per_class", "confusion", "classes")},
            "seen_in_training": seen,
        })
        self.manager.save(self.project)
        self.refresh_baselines()
        QMessageBox.information(self, "Final exam", summary)

    # ------------------------------------------------- baselines & live quota
    @staticmethod
    def _record_task(record: dict) -> str:
        return "detect" if record.get("type") in ("detect", "detect_test") else "classify"

    @staticmethod
    def _record_is_test(record: dict) -> bool:
        return record.get("type") in ("test", "detect_test")

    @staticmethod
    def _evaluation_label(record: dict) -> str:
        when = record.get("finished_at", "?")
        if record.get("type") == "detect_test":
            return f"Detector exam {when} (mAP50 {record.get('mAP50', 0) * 100:.1f}%)"
        if record.get("type") == "detect":
            map50 = (record.get("metrics") or {}).get("mAP50")
            return f"Detector run {when}" + (f" (mAP50 {map50 * 100:.1f}%)" if map50 is not None else "")
        if record.get("type") == "test":
            return f"Test exam {when} (acc {record.get('accuracy', 0) * 100:.1f}%)"
        acc = (record.get("metrics") or {}).get("val_acc")
        return f"Training run {when}" + (f" (val acc {acc * 100:.1f}%)" if acc is not None else "")

    def _latest_evaluation(self, test: bool) -> Optional[dict]:
        """The newest stored training run (or test exam) of the current task with a confusion matrix,
        as a baseline."""
        if self.project is None:
            return None
        for record in reversed(self.project.evaluations):
            if self._record_is_test(record) == test and self._record_task(record) == self.task \
                    and record.get("confusion") and record.get("classes"):
                return {"confusion": record["confusion"], "classes": record["classes"],
                        "label": "previous " + ("exam" if test else "run")}
        return None

    def refresh_baselines(self):
        current = self.baseline_combo.currentData()
        self.baseline_combo.blockSignals(True)
        self.baseline_combo.clear()
        self.baseline_combo.addItem("Previous run", "previous")
        self.baseline_combo.addItem("No comparison", "none")
        records = self.project.evaluations if self.project is not None else []
        for i in reversed(range(len(records))):
            if records[i].get("confusion") and records[i].get("classes") \
                    and self._record_task(records[i]) == self.task:
                self.baseline_combo.addItem(self._evaluation_label(records[i]), i)
        index = self.baseline_combo.findData(current) if current is not None else -1
        self.baseline_combo.setCurrentIndex(max(0, index))
        self.baseline_combo.blockSignals(False)

    def selected_baseline(self) -> Optional[dict]:
        data = self.baseline_combo.currentData()
        if data == "previous":
            return self.previous_baseline
        if isinstance(data, int) and self.project is not None \
                and 0 <= data < len(self.project.evaluations):
            record = self.project.evaluations[data]
            return {"confusion": record["confusion"], "classes": record["classes"],
                    "label": self.baseline_combo.currentText()}
        return None

    def render_normalized_confusion(self):
        if self.current_eval is None:
            return
        self.cm_norm_canvas.plot_confusion(self.current_eval["confusion"],
                                           self.current_eval["classes"], normalized=True,
                                           prefix=self.current_eval["prefix"],
                                           baseline=self.selected_baseline())

    def update_quota_counters(self, values: dict, confusion=None, classes=None):
        """Live ``current / target`` counters of the quota objectives and of every class."""
        quota = self.project.quota if self.project is not None else QuotaConfig()
        for row in objective_progress(quota, values):
            bar = self.quota_bars[row["objective"]]
            bar.setValue(0 if row["ratio"] is None else min(100, math.floor(row["ratio"] * 100)))
            bar.setFormat(row["text"])
            bar.setToolTip(f"{row['label']}: current / target (share of the target reached)")
            bar.setStyleSheet(_progress_style(row["met"]))
        rows = class_progress(confusion, classes, quota.min_accuracy) if confusion is not None else []
        table = self.class_quota_table
        table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            accuracy = "–" if row["accuracy"] is None else f"{row['accuracy'] * 100:.1f}%"
            table.setItem(r, 0, QTableWidgetItem(row["class"]))
            table.setItem(r, 1, QTableWidgetItem(f"{row['correct']} / {row['needed']} "
                                                 f"(of {row['total']})"))
            table.setItem(r, 2, QTableWidgetItem(f"{accuracy} / {quota.min_accuracy * 100:.1f}%"))
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0 if row["ratio"] is None else min(100, math.floor(row["ratio"] * 100)))
            bar.setFormat(row["text"])
            bar.setStyleSheet(_progress_style(row["met"]))
            table.setCellWidget(r, 3, bar)

    # ------------------------------------------------------------------ misc
    def _reset_analytics(self, keep_counts: bool = False):
        self.current_eval = None
        self.update_quota_counters({})
        self.show_detection_metrics({})
        self.cm_canvas.show_message("Confusion Matrix", "Waiting for the first validation step")
        self.cm_norm_canvas.show_message("Normalized Confusion Matrix",
                                         "Waiting for the first validation step")
        if self.task == "detect":
            self.precision_canvas.plot_detection_history([])
        else:
            self.precision_canvas.plot_precision([])
        self.conf_canvas.plot_confidence_curves({})
        if not keep_counts and not self._loading and self.project is not None:
            self.on_datasets_changed()

    def log(self, message: str):
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_view.append(f"[{stamp}] {message}")

    def closeEvent(self, event):
        if self.is_training():
            answer = QMessageBox.question(self, "Quit", "Training is running. Stop it and quit?")
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        # Training may have finished while the dialog was open, so re-check.
        thread, worker = self.thread, self.worker
        if worker is not None:
            worker.request_stop()
        if thread is not None:
            thread.quit()
            thread.wait()
        if self.project is not None:
            self.save_project()
        super().closeEvent(event)
