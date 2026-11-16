import numpy as np
import pytest

from schaakgui.metrics import (
    confidence_curves,
    confusion_delta,
    confusion_matrix,
    macro_precision_recall,
    normalize_confusion,
    precision_recall,
)
from schaakgui.projects import QuotaConfig
from schaakgui.quota import (
    EarlyStopping,
    EarlyStoppingMonitor,
    QuotaTracker,
    class_progress,
    objective_progress,
)


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


def test_early_stopping_monitor_loss_accuracy_and_both():
    loss = EarlyStoppingMonitor(patience=2, monitor="val_loss")
    assert not loss.step(1.0, 0.5)
    assert not loss.step(1.1, 0.9)
    assert loss.step(1.2, 0.95)  # accuracy improves, but only loss is monitored

    acc = EarlyStoppingMonitor(patience=2, monitor="val_acc")
    assert not acc.step(1.0, 0.5)
    assert not acc.step(0.5, 0.5)
    assert acc.step(0.1, 0.4)  # loss improves, but only accuracy is monitored

    both = EarlyStoppingMonitor(patience=2, monitor="both")
    assert not both.step(1.0, 0.5)
    assert not both.step(1.1, 0.6)  # accuracy improved
    assert not both.step(0.9, 0.6)  # loss improved
    assert not both.step(0.95, 0.55)  # loss stalled 1, accuracy stalled 2
    assert both.step(0.95, 0.55)  # both stalled for 2 epochs
    assert both.describe() == "validation loss and accuracy"
    with pytest.raises(ValueError):
        EarlyStoppingMonitor(monitor="train_loss")


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


def test_quota_live_counters():
    cfg = QuotaConfig(min_accuracy=0.8, target_avg_accuracy=0.9, min_confidence=0.7,
                      target_avg_confidence=0.8, moving_average_window=2)
    tracker = QuotaTracker(cfg)
    assert [r["text"] for r in objective_progress(cfg, tracker.values)][0] == "– / 80.0%"
    tracker.update(1, 0.7, 0.7)
    tracker.update(2, 0.82, 0.75)
    rows = {r["objective"]: r for r in objective_progress(cfg, tracker.values)}
    assert rows["accuracy_floor"]["text"] == "82.0% / 80.0% (102%)" and rows["accuracy_floor"]["met"]
    assert rows["accuracy_target"]["value"] == pytest.approx(0.76)
    assert rows["accuracy_target"]["text"] == "76.0% / 90.0% (84%)" and not rows["accuracy_target"]["met"]

    cm = [[9, 1, 0], [3, 7, 0], [0, 0, 0]]
    rows = class_progress(cm, ["a", "b", "c"], 0.8)
    assert [(r["correct"], r["needed"], r["met"]) for r in rows] == [(9, 8, True), (7, 8, False),
                                                                     (0, 0, False)]
    assert rows[0]["text"] == "9 / 8 (112%)" and rows[1]["text"] == "7 / 8 (88%)"
    assert rows[2]["text"] == "0 / 0" and rows[2]["accuracy"] is None


def test_confusion_delta_aligned_by_class_name():
    cm = [[9, 1], [2, 8]]
    baseline = [[5, 0, 5], [0, 0, 0], [3, 0, 7]]  # classes a, x, b; class x has no samples
    delta = confusion_delta(cm, ["a", "b"], baseline, ["a", "x", "b"])
    assert delta[0, 0] == pytest.approx(40) and delta[0, 1] == pytest.approx(-40)
    assert delta[1, 0] == pytest.approx(-10) and delta[1, 1] == pytest.approx(10)
    delta = confusion_delta(cm, ["a", "new"], baseline, ["a", "x", "b"])
    assert delta[0, 0] == pytest.approx(40) and np.isnan(delta[0, 1]) and np.isnan(delta[1]).all()
    assert np.isnan(confusion_delta(cm, ["a", "b"], [], [])).all()
