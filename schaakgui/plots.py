"""Matplotlib canvases embedded in the Qt UI."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from .metrics import confusion_delta, normalize_confusion

# Precision plot: zoom into the upper range so e.g. 0.80 vs. 0.83 is clearly visible.
PRECISION_YLIM = (0.75, 1.01)
# Deltas smaller than this (in percentage points) are shown as neutral.
DELTA_EPSILON = 0.05


def precision_ylim(values, floor: float = PRECISION_YLIM[0],
                   top: float = PRECISION_YLIM[1]) -> tuple[float, float]:
    """Y range of the precision plot: ``[floor, top]``, lowered (in 0.05 steps) so that no plotted
    value below ``floor`` is clipped."""
    finite = [float(v) for v in values if v is not None and math.isfinite(v)]
    low = min(finite, default=floor)
    if low >= floor:
        return floor, top
    return max(0.0, math.floor((low - 0.01) * 20) / 20), top


def delta_color(delta: float, diagonal: bool, dark_background: bool = False) -> str:
    """Green for improvements (diagonal gain / off-diagonal drop), red for regressions."""
    if abs(delta) < DELTA_EPSILON:
        return "#e0e0e0" if dark_background else "#616161"
    good = (delta > 0) == diagonal
    if dark_background:
        return "#b9f6ca" if good else "#ff8a80"
    return "#1b5e20" if good else "#b71c1c"


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

    def plot_confusion(self, cm, classes: list[str], normalized: bool = False, prefix: str = "",
                       baseline: Optional[dict] = None):
        """``baseline`` (normalized only): ``{"confusion", "classes", "label"}`` of an earlier run;
        every cell then also shows the percentage-point change, e.g. ``0.85 (+3.2%)``."""
        title = prefix + ("Normalized Confusion Matrix" if normalized else "Confusion Matrix")
        cm = np.asarray(cm)
        delta = None
        if normalized and baseline:
            delta = confusion_delta(cm, classes, baseline["confusion"], baseline["classes"])
            title += f" (Δ vs. {baseline.get('label', 'baseline')})"
        self._reset(title)
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
                    dark = data[r, c] > threshold
                    color = "white" if dark else "black"
                    if delta is not None and not np.isnan(delta[r, c]):
                        self.ax.text(c, r - 0.17, f"{data[r, c]:.2f}", ha="center", va="center",
                                     fontsize=6, color=color)
                        self.ax.text(c, r + 0.22, f"({delta[r, c]:+.1f}%)", ha="center",
                                     va="center", fontsize=5,
                                     color=delta_color(delta[r, c], r == c, dark))
                        continue
                    text = f"{data[r, c] * 100:.0f}%" if normalized else str(int(data[r, c]))
                    self.ax.text(c, r, text, ha="center", va="center", fontsize=6, color=color)
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
        self.ax.set_ylim(*precision_ylim(
            [h[k] for h in history for k in ("precision", "recall", "val_acc")]))
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
