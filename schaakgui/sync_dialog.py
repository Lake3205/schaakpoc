"""Dialog for harmonizing class names across the ``.yaml`` files of several YOLO datasets."""
from __future__ import annotations

import html
from pathlib import Path
from typing import Optional

import yaml
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from .yaml_sync import apply_mapping, collect_tags, suggest_mapping, unified_names


class YamlSyncDialog(QDialog):
    """Mapping table ``tag | found in | unified class name``; *Apply* rewrites yaml + label files."""

    def __init__(self, yaml_paths: list[str | Path], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Synchronize YAML class names")
        self.resize(820, 520)
        self.yaml_paths = [Path(p) for p in yaml_paths]
        self.tags = collect_tags(self.yaml_paths)
        self.report: Optional[dict] = None
        suggestions = suggest_mapping(self.tags)

        layout = QVBoxLayout(self)
        files = "<br>".join(f"• {html.escape(str(p))}" for p in self.yaml_paths)
        intro = QLabel(f"<b>Detected .yaml files ({len(self.yaml_paths)}):</b><br>{files}<br><br>"
                       "Choose or type the unified class name for every tag. Tags with the same "
                       "unified name are merged; every .yaml gets the same <code>names</code> list "
                       "and the class ids in the YOLO label files are re-indexed accordingly.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.table = QTableWidget(len(self.tags), 3)
        self.table.setHorizontalHeaderLabels(["Tag", "Found in", "Unified class name"])
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.combos: dict[str, QComboBox] = {}
        for row, (tag, sources) in enumerate(self.tags.items()):
            self.table.setItem(row, 0, QTableWidgetItem(tag))
            found = QTableWidgetItem(", ".join(_short(p) for p in sources))
            found.setToolTip("\n".join(str(p) for p in sources))
            self.table.setItem(row, 1, found)
            combo = QComboBox()
            combo.setEditable(True)
            combo.addItems(list(self.tags))
            combo.setCurrentText(suggestions[tag])
            combo.currentTextChanged.connect(self.update_preview)
            self.table.setCellWidget(row, 2, combo)
            self.combos[tag] = combo
        layout.addWidget(self.table, 1)

        self.preview = QLabel()
        self.preview.setWordWrap(True)
        layout.addWidget(self.preview)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.confirm)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.update_preview()

    def mapping(self) -> dict[str, str]:
        return {tag: combo.currentText().strip() or tag for tag, combo in self.combos.items()}

    def update_preview(self, *_):
        names = unified_names(self.yaml_paths, self.mapping())
        self.preview.setText(f"<b>Unified classes ({len(names)}):</b> "
                             + html.escape(", ".join(f"{i}: {n}" for i, n in enumerate(names))))

    def apply_changes(self) -> dict:
        self.report = apply_mapping(self.yaml_paths, self.mapping())
        return self.report

    def confirm(self):
        answer = QMessageBox.question(
            self, "Synchronize YAML class names",
            f"Rewrite the names of {len(self.yaml_paths)} .yaml file(s) and re-index the class ids "
            "of their YOLO label files? This modifies the dataset files on disk.")
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.apply_changes()
        except (OSError, ValueError, yaml.YAMLError) as e:
            QMessageBox.critical(self, "Synchronize YAML class names", f"Synchronization failed:\n{e}")
            return
        self.accept()


def _short(path: Path) -> str:
    """``parent/name`` of a yaml path (enough to tell datasets apart in the table)."""
    return f"{path.parent.name}/{path.name}"
