"""Training quotas (accuracy/confidence targets) and early stopping.

Quota objectives:

* ``accuracy_floor``     - validation accuracy of an epoch >= ``min_accuracy``
* ``accuracy_target``    - moving average of validation accuracy >= ``target_avg_accuracy``
* ``confidence_floor``   - mean top-1 confidence of an epoch >= ``min_confidence``
* ``confidence_target``  - moving average of mean confidence >= ``target_avg_confidence``

The moving average is taken over the last ``moving_average_window`` epochs.
"""
from __future__ import annotations

import math
from collections import deque
from datetime import datetime
from typing import Optional

from .projects import QuotaConfig

OBJECTIVES = {
    "accuracy_floor": "Val Acc Floor",
    "accuracy_target": "Val Acc Moving Avg",
    "confidence_floor": "Confidence Floor",
    "confidence_target": "Confidence Moving Avg",
}


def objective_threshold(cfg: QuotaConfig, key: str) -> float:
    return {
        "accuracy_floor": cfg.min_accuracy,
        "accuracy_target": cfg.target_avg_accuracy,
        "confidence_floor": cfg.min_confidence,
        "confidence_target": cfg.target_avg_confidence,
    }[key]


class QuotaTracker:
    """Tracks quota objectives across epochs and records milestones once achieved."""

    def __init__(self, cfg: QuotaConfig, history: Optional[list[dict]] = None):
        self.cfg = cfg
        self.history = list(history or [])
        window = max(1, int(cfg.moving_average_window))
        self._acc = deque(maxlen=window)
        self._conf = deque(maxlen=window)
        self.current: dict[str, bool] = {k: False for k in OBJECTIVES}

    def achieved(self, key: str) -> Optional[dict]:
        """First recorded milestone for an objective at (or above) the current threshold."""
        threshold = objective_threshold(self.cfg, key)
        for m in self.history:
            if m["objective"] == key and m["threshold"] >= threshold - 1e-9:
                return m
        return None

    def status(self) -> dict[str, Optional[dict]]:
        return {k: self.achieved(k) for k in OBJECTIVES}

    def update(self, epoch: int, val_acc: float, mean_conf: float) -> list[dict]:
        """Register an epoch's results. Returns newly achieved milestones."""
        self._acc.append(val_acc)
        self._conf.append(mean_conf)
        values = {
            "accuracy_floor": val_acc,
            "accuracy_target": sum(self._acc) / len(self._acc),
            "confidence_floor": mean_conf,
            "confidence_target": sum(self._conf) / len(self._conf),
        }
        new = []
        for key, value in values.items():
            threshold = objective_threshold(self.cfg, key)
            met = value >= threshold
            self.current[key] = met
            if met and self.achieved(key) is None:
                verb = "Achieved" if key.startswith("accuracy") else "Hit"
                milestone = {
                    "objective": key,
                    "epoch": epoch,
                    "value": float(value),
                    "threshold": float(threshold),
                    "message": f"{verb} {value * 100:.1f}% {OBJECTIVES[key]} "
                               f"(target {threshold * 100:.0f}%) at Epoch {epoch}",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
                self.history.append(milestone)
                new.append(milestone)
        return new

    def all_met(self) -> bool:
        """True if every objective is satisfied by the most recent epoch."""
        return all(self.current.values())

    def describe(self) -> str:
        return ", ".join(f"{OBJECTIVES[k]}: {'OK' if v else 'pending'}"
                         for k, v in self.current.items())


class EarlyStopping:
    """Stop when the monitored metric fails to improve for ``patience`` epochs."""

    def __init__(self, patience: int = 5, mode: str = "min", min_delta: float = 0.0):
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = max(1, int(patience))
        self.mode = mode
        self.min_delta = min_delta
        self.best = math.inf if mode == "min" else -math.inf
        self.counter = 0

    def improved(self, value: float) -> bool:
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def step(self, value: float) -> bool:
        """Register a value; returns True when training should stop."""
        if self.improved(value):
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience
