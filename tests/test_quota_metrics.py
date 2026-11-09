import numpy as np
import pytest

from schaakgui.metrics import (
    confidence_curves,
    confusion_matrix,
    macro_precision_recall,
    normalize_confusion,
    precision_recall,
)
from schaakgui.projects import QuotaConfig
from schaakgui.quota import EarlyStopping, QuotaTracker


def test_quota_milestones_and_all_met():
    cfg = QuotaConfig(min_accuracy=0.8, target_avg_accuracy=0.9, min_confidence=0.7,
                      target_avg_confidence=0.8, moving_average_window=2)
    tracker = QuotaTracker(cfg)
    assert tracker.update(1, 0.5, 0.5) == []
    new = tracker.update(2, 0.82, 0.75)
    assert {m["objective"] for m in new} == {"accuracy_floor", "confidence_floor"}
    assert "Achieved 82.0% Val Acc Floor" in new[0]["message"] and "Epoch 2" in new[0]["message"]
    assert not tracker.all_met()
    tracker.update(3, 0.99, 0.9)
    assert tracker.all_met()
    assert all(v is not None for v in tracker.status().values())


def test_quota_history_is_reused_and_threshold_aware():
    history = QuotaTracker(QuotaConfig(min_accuracy=0.5)).history
    tracker = QuotaTracker(QuotaConfig(min_accuracy=0.5))
    tracker.update(1, 0.6, 0.0)
    history = tracker.history
    assert QuotaTracker(QuotaConfig(min_accuracy=0.5), history).achieved("accuracy_floor")
    assert QuotaTracker(QuotaConfig(min_accuracy=0.9), history).achieved("accuracy_floor") is None


def test_early_stopping():
    stopper = EarlyStopping(patience=2, mode="min")
    assert not stopper.step(1.0)
    assert not stopper.step(0.9)
    assert not stopper.step(0.95)
    assert stopper.step(0.91)
    maximize = EarlyStopping(patience=1, mode="max")
    assert not maximize.step(0.5)
    assert maximize.step(0.5)
    with pytest.raises(ValueError):
        EarlyStopping(mode="up")


def test_confusion_and_precision_recall():
    cm = confusion_matrix([0, 0, 1, 1, 2], [0, 1, 1, 1, 0], 3)
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 0]]
    norm = normalize_confusion(cm)
    assert np.allclose(norm.sum(axis=1), 1)
    precision, recall = precision_recall(cm)
    assert np.allclose(precision, [0.5, 2 / 3, 0])
    assert np.allclose(recall, [0.5, 1, 0])
    p, r = macro_precision_recall(cm)
    assert p == pytest.approx((0.5 + 2 / 3) / 3) and r == pytest.approx(0.5)
    assert normalize_confusion(np.zeros((2, 2))).sum() == 0


def test_confidence_curves():
    curves = confidence_curves([0.9, 0.6, 0.3], [True, False, True], num_points=11)
    assert curves["recall"][0] == pytest.approx(2 / 3)
    assert curves["precision"][0] == pytest.approx(2 / 3)
    assert curves["precision"][7] == 1.0  # threshold 0.7 keeps only the correct 0.9 prediction
    assert curves["recall"][7] == pytest.approx(1 / 3)
