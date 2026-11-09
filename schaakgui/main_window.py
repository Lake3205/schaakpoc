"""Main desktop window: projects, datasets, augmentations, quotas, training and analytics."""
from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QThread
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
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
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .checkpoint import ARCHITECTURES, Lineage, read_lineage
from .data import class_counts, enabled_augmentations
from .engine import TrainJob
from .plots import PlotCanvas
from .projects import DatasetEntry, Project, ProjectManager
from .quota import OBJECTIVES, QuotaTracker, objective_threshold
from .review_dialog import ReviewDialog, ReviewSession
from .worker import TrainingWorker, start_worker

STOP_REASONS = {
    "completed": "all epochs completed",
    "quota": "all quota targets satisfied",
    "early_stopping": "early stopping (no improvement)",
    "user": "stopped by user",
}


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
        folder = QFileDialog.getExistingDirectory(self, "Select dataset folder (class_name/xxx.jpg)")
        if folder:
            if folder in [e.path for e in self.entries()]:
                return
            self.add_entry(DatasetEntry(folder, True))
            self.on_change()

    def add_entry(self, entry: DatasetEntry):
        item = QListWidgetItem(entry.path)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked if entry.checked else Qt.CheckState.Unchecked)
        self.list.addItem(item)

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
        self._loading = False

        self.setWindowTitle("schaakpoc v2 – Chess Piece Classifier Trainer")
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
        bar.addStretch()
        return bar

    def _build_dataset_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.train_folders = FolderList("Training pools (check the folders to use)",
                                        self.on_datasets_changed)
        self.test_folders = FolderList("Test / validation sets (evaluated every epoch)",
                                       self.on_datasets_changed)
        layout.addWidget(self.train_folders)
        layout.addWidget(self.test_folders)
        layout.addWidget(QLabel("<small>Each folder must contain one sub-folder per class: "
                                "<code>class_name/xxx.jpg</code>. Without test folders 20% of the "
                                "training pool is held out for validation.</small>"))

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
        form.addRow("Epochs (this session)", self.epochs_spin)
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
        self.early_stop_check.toggled.connect(self.patience_spin.setEnabled)
        self.early_stop_check.toggled.connect(self.monitor_combo.setEnabled)
        form.addRow("Export full weights every", self.snapshot_spin)
        form.addRow(self.early_stop_check)
        form.addRow("Patience", self.patience_spin)
        form.addRow("Monitor", self.monitor_combo)
        layout.addWidget(snapshots)
        layout.addStretch()
        return tab

    def _build_checkpoint_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        resume = QGroupBox("Resume training")
        resume_layout = QHBoxLayout(resume)
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
        layout.addWidget(resume)

        saved = QGroupBox("Saved models in this project")
        saved_layout = QVBoxLayout(saved)
        self.checkpoint_list = QListWidget()
        self.checkpoint_list.currentTextChanged.connect(self.show_provenance)
        row = QHBoxLayout()
        refresh = QPushButton("Refresh")
        use = QPushButton("Use for resume")
        refresh.clicked.connect(self.refresh_checkpoints)
        use.clicked.connect(self.use_selected_for_resume)
        row.addWidget(refresh)
        row.addWidget(use)
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
        self.start_button.clicked.connect(self.start_training)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.stop_button.clicked.connect(self.stop_training)
        self.review_button.clicked.connect(self.open_review)
        for b in (self.start_button, self.pause_button, self.stop_button, self.review_button):
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

        splitter = QSplitter(Qt.Orientation.Vertical)
        self.analytics = QTabWidget()
        self.cm_canvas = PlotCanvas()
        self.cm_norm_canvas = PlotCanvas()
        self.count_canvas = PlotCanvas()
        self.precision_canvas = PlotCanvas()
        self.conf_canvas = PlotCanvas()
        self.analytics.addTab(self.cm_canvas, "Confusion Matrix")
        self.analytics.addTab(self.cm_norm_canvas, "Normalized Confusion Matrix")
        self.analytics.addTab(self.count_canvas, "Instance Count")
        self.analytics.addTab(self.precision_canvas, "Precision Curve")
        self.analytics.addTab(self.conf_canvas, "Recall vs. Confidence")
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
        self._reset_analytics()
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

        self.resume_check.setChecked(bool(p.resume_checkpoint))
        self.resume_edit.setText(p.resume_checkpoint or "")
        self._loading = False
        self.update_aug_preview()
        self.on_datasets_changed()

    def _collect_form(self, p: Project):
        p.train_folders = self.train_folders.entries()
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

        resume = self.resume_edit.text().strip()
        p.resume_checkpoint = resume if self.resume_check.isChecked() and resume else None

    # --------------------------------------------------------------- datasets
    def _safe_counts(self, folders: list[str]) -> dict[str, int]:
        existing = [f for f in folders if Path(f).is_dir()]
        missing = sorted(set(folders) - set(existing))
        if missing:
            self.log("Warning: folder(s) not found: " + ", ".join(missing))
        return class_counts(existing)

    def on_datasets_changed(self):
        if self._loading:
            return
        train = self._safe_counts(self.train_folders.checked())
        test = self._safe_counts(self.test_folders.checked())
        self.balance_canvas.plot_counts(train, "Class balance – training pools")
        self.count_canvas.plot_counts(train, "Instance Count (selected datasets)", test or None)

    def update_aug_preview(self):
        if self._loading or self.project is None:
            return
        self._collect_form(self.project)
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
                  self.save_project_button):
            w.setEnabled(not running)
        self.review_button.setEnabled(not running)
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
            test_folders=p.checked_test_folders(),
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
        self.cm_norm_canvas.plot_confusion(payload["confusion"], classes, normalized=True)
        self.precision_canvas.plot_precision(payload["history"])
        self.conf_canvas.plot_confidence_curves(payload["curves"])
        self.last_checkpoint = payload["checkpoint"]

    def on_milestone(self, milestone: dict):
        self.statusBar().showMessage("🏁 " + milestone["message"], 10000)

    def on_paused(self, checkpoint_path: str):
        self.paused_checkpoint = checkpoint_path
        self.review_button.setEnabled(True)
        self.statusBar().showMessage("Paused – use Manual Review to inspect the current model")

    def on_finished(self, summary: dict):
        self.paused_checkpoint = None
        if self.project is not None:
            record = {k: summary[k] for k in ("finished_at", "stop_reason", "total_epochs_completed",
                                              "epochs_this_session", "last_checkpoint",
                                              "best_checkpoint", "metrics", "milestones")}
            self.project.evaluations.append(record)
            for path in (summary["last_checkpoint"], summary["best_checkpoint"]):
                if path and path not in self.project.checkpoints:
                    self.project.checkpoints.append(path)
            self.manager.save(self.project)
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
        path, _ = QFileDialog.getOpenFileName(self, "Select checkpoint", "",
                                              "PyTorch checkpoints (*.pt *.pth)")
        if path:
            self.resume_edit.setText(path)
            self.resume_check.setChecked(True)
            if self.project is not None and path not in self.project.checkpoints:
                self.project.checkpoints.append(path)
            self.refresh_checkpoints(select=str(Path(path).resolve()))

    def refresh_checkpoints(self, select: Optional[str] = None):
        if self.project is None:
            return
        current = select or self.checkpoint_list.currentItem() and self.checkpoint_list.currentItem().text()
        self.checkpoint_list.blockSignals(True)
        self.checkpoint_list.clear()
        paths = self.manager.list_checkpoints(self.project)
        self.checkpoint_list.addItems(paths)
        self.checkpoint_list.blockSignals(False)
        if current in paths:
            self.checkpoint_list.setCurrentRow(paths.index(current))
        else:
            self.show_provenance(self.project.resume_checkpoint or "")

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
            parts.append("<b>Active test sources:</b><br>"
                         + ("<br>".join(f"• {esc(f)}" for f in active.get("test_folders", []))
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
    def open_review(self):
        if self.project is None:
            return
        if self.is_training():
            checkpoint = self.paused_checkpoint
        else:
            checkpoint = self.selected_checkpoint() or self.last_checkpoint
        if not checkpoint:
            QMessageBox.information(self, "Manual Review",
                                    "No model available yet. Train a model or select a saved one.")
            return
        self._collect_form(self.project)
        folders = self.project.checked_test_folders() or self.project.checked_train_folders()
        if not folders:
            QMessageBox.information(self, "Manual Review", "Check a test (or training) folder first.")
            return
        try:
            session = ReviewSession(checkpoint, folders, self.project.seed)
        except Exception as e:
            QMessageBox.warning(self, "Manual Review", f"Cannot start review:\n{e}")
            return
        ReviewDialog(session, checkpoint, self).exec()

    # ------------------------------------------------------------------ misc
    def _reset_analytics(self, keep_counts: bool = False):
        self.cm_canvas.show_message("Confusion Matrix", "Waiting for the first validation step")
        self.cm_norm_canvas.show_message("Normalized Confusion Matrix",
                                         "Waiting for the first validation step")
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
            self.worker.request_stop()
            self.thread.quit()
            self.thread.wait()
        if self.project is not None:
            self.save_project()
        super().closeEvent(event)
