"""Classification metrics: confusion matrices, precision/recall and confidence curves."""
from __future__ import annotations

import numpy as np


def confusion_matrix(y_true, y_pred, num_classes: int) -> np.ndarray:
    """Absolute counts; rows = true class, columns = predicted class."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)), 1)
    return cm


def normalize_confusion(cm: np.ndarray) -> np.ndarray:
    """Normalize each true-class row so it sums to 1 (empty rows stay 0)."""
    cm = np.asarray(cm, dtype=float)
    sums = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, sums, out=np.zeros_like(cm), where=sums > 0)


def precision_recall(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-class precision and recall from a confusion matrix."""
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm)
    pred = cm.sum(axis=0)
    true = cm.sum(axis=1)
    precision = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    recall = np.divide(tp, true, out=np.zeros_like(tp), where=true > 0)
    return precision, recall


def macro_precision_recall(cm: np.ndarray) -> tuple[float, float]:
    """Macro averages over the classes present in the ground truth."""
    precision, recall = precision_recall(cm)
    present = np.asarray(cm).sum(axis=1) > 0
    if not present.any():
        return 0.0, 0.0
    return float(precision[present].mean()), float(recall[present].mean())


def confidence_curves(confidences, correct, num_points: int = 101) -> dict:
    """Precision and recall as a function of the confidence threshold.

    At threshold ``t`` only predictions with confidence >= t are kept:
    precision = correct kept / kept, recall = correct kept / all samples.
    """
    conf = np.asarray(confidences, dtype=float)
    ok = np.asarray(correct, dtype=bool)
    thresholds = np.linspace(0.0, 1.0, num_points)
    precision = np.zeros(num_points)
    recall = np.zeros(num_points)
    total = len(conf)
    for i, t in enumerate(thresholds):
        kept = conf >= t
        n_kept = int(kept.sum())
        n_ok = int((ok & kept).sum())
        precision[i] = n_ok / n_kept if n_kept else 1.0
        recall[i] = n_ok / total if total else 0.0
    return {"thresholds": thresholds.tolist(), "precision": precision.tolist(),
            "recall": recall.tolist()}
