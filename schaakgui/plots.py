"""Matplotlib canvases embedded in the Qt UI."""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from .metrics import normalize_confusion


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width: float = 5, height: float = 4):
        self.figure = Figure(figsize=(width, height), tight_layout=True)
        super().__init__(self.figure)
        self.setParent(parent)
        self.ax = self.figure.add_subplot(111)

    def _reset(self, title: str):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title(title)

    def show_message(self, title: str, message: str):
        self._reset(title)
        self.ax.text(0.5, 0.5, message, ha="center", va="center", transform=self.ax.transAxes)
        self.ax.set_axis_off()
        self.draw_idle()

    def plot_confusion(self, cm, classes: list[str], normalized: bool = False, prefix: str = ""):
        title = prefix + ("Normalized Confusion Matrix" if normalized else "Confusion Matrix")
        self._reset(title)
        cm = np.asarray(cm)
        data = normalize_confusion(cm) if normalized else cm
        image = self.ax.imshow(data, cmap="Blues", vmin=0, vmax=1 if normalized else None)
        self.figure.colorbar(image, ax=self.ax, fraction=0.046, pad=0.04)
        ticks = np.arange(len(classes))
        self.ax.set_xticks(ticks, classes, rotation=90, fontsize=7)
        self.ax.set_yticks(ticks, classes, fontsize=7)
        self.ax.set_xlabel("Predicted")
        self.ax.set_ylabel("True")
        if len(classes) <= 20:
            threshold = data.max() / 2 if data.size and data.max() > 0 else 0
            for r in range(data.shape[0]):
                for c in range(data.shape[1]):
                    text = f"{data[r, c] * 100:.0f}%" if normalized else str(int(data[r, c]))
                    self.ax.text(c, r, text, ha="center", va="center", fontsize=6,
                                 color="white" if data[r, c] > threshold else "black")
        self.draw_idle()

    def plot_counts(self, counts: dict[str, int], title: str = "Instance Count",
                    others: dict[str, dict[str, int]] | None = None):
        """Bar chart of samples per class (optionally side by side with e.g. validation/test counts)."""
        series = {"train": counts, **{k: v for k, v in (others or {}).items() if v}}
        if not any(series.values()):
            self.show_message(title, "No images in the selected folders")
            return
        self._reset(title)
        names = sorted(set().union(*series.values()))
        x = np.arange(len(names))
        width = 0.8 / len(series)
        colors = ["tab:blue", "tab:green", "tab:orange", "tab:red"]
        for i, (label, values) in enumerate(series.items()):
            offset = (i - (len(series) - 1) / 2) * width
            bars = self.ax.bar(x + offset, [values.get(n, 0) for n in names], width,
                               color=colors[i % len(colors)], label=label)
            self.ax.bar_label(bars, fontsize=6)
        if len(series) > 1:
            self.ax.legend()
        self.ax.set_xticks(x, names, rotation=90, fontsize=7)
        self.ax.set_ylabel("instances")
        self.draw_idle()

    def plot_precision(self, history: list[dict]):
        if not history:
            self.show_message("Precision Curve", "No epochs evaluated yet")
            return
        self._reset("Precision / Recall per Epoch")
        epochs = [h["epoch"] for h in history]
        self.ax.plot(epochs, [h["precision"] for h in history], marker="o", label="precision")
        self.ax.plot(epochs, [h["recall"] for h in history], marker=".", ls="--", label="recall")
        self.ax.plot(epochs, [h["val_acc"] for h in history], ls=":", label="val accuracy")
        self.ax.set_xlabel("epoch")
        self.ax.set_ylim(0, 1.05)
        self.ax.grid(alpha=0.3)
        self.ax.legend()
        self.draw_idle()

    def plot_confidence_curves(self, curves: dict):
        if not curves:
            self.show_message("Recall vs. Confidence", "No epochs evaluated yet")
            return
        self._reset("Precision / Recall vs. Confidence Threshold")
        t = curves["thresholds"]
        self.ax.plot(t, curves["recall"], label="recall")
        self.ax.plot(t, curves["precision"], label="precision")
        self.ax.set_xlabel("confidence threshold")
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1.05)
        self.ax.grid(alpha=0.3)
        self.ax.legend()
        self.draw_idle()
