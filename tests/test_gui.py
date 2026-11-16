import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

pytest.importorskip("PyQt6.QtWidgets", exc_type=ImportError)
from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

import schaakgui.detector as detector_module  # noqa: E402
import schaakgui.review_dialog as review_dialog_module  # noqa: E402
from schaakgui.main_window import MainWindow  # noqa: E402
from schaakgui.projects import DatasetEntry, ProjectManager  # noqa: E402
from schaakgui.review_dialog import (  # noqa: E402
    DetectionReviewDialog,
    ImageTestDialog,
    ReviewDialog,
    ReviewSession,
)
from schaakgui.sync_dialog import YamlSyncDialog  # noqa: E402


class FakeDetector:
    """Stands in for an Ultralytics YOLO detector: two boxes, returned out of reading order."""

    def __call__(self, image, conf, device, verbose):
        boxes = SimpleNamespace(xyxy=torch.tensor([[24.0, 4.0, 46.0, 22.0], [2.0, 0.0, 22.0, 20.0]]),
                                conf=torch.tensor([0.8, 0.9]), cls=torch.tensor([1.0, 0.0]))
        return [SimpleNamespace(names={0: "White_Rook", 1: "black-king"}, boxes=boxes)]


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def wait_until(app, predicate, timeout=120):
    end = time.time() + timeout
    while not predicate():
        app.processEvents()
        assert time.time() < end, "timeout"
        time.sleep(0.01)


def test_full_gui_workflow(app, datasets, tmp_path, monkeypatch):
    messages = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: messages.append(a[2]))
    manager = ProjectManager(tmp_path / ".projects")
    window = MainWindow(manager)
    assert window.project.name == "default" and window.task == "detect"
    window.task_combo.setCurrentIndex(window.task_combo.findData("classify"))
    assert window.project.task == "classify" and manager.load("default").task == "classify"

    window.train_folders.set_entries([DatasetEntry(str(datasets["a"])),
                                      DatasetEntry(str(datasets["b"]), False)])
    window.val_folders.set_entries([DatasetEntry(str(datasets["val"]))])
    window.test_folders.set_entries([DatasetEntry(str(datasets["test"]))])
    window.on_datasets_changed()
    window.aug_hflip.setChecked(True)
    window.epochs_spin.setValue(2)
    window.batch_spin.setValue(4)
    window.image_size_spin.setValue(32)
    window.snapshot_spin.setValue(1)
    window.seed_check.setChecked(True)
    window.monitor_combo.setCurrentIndex(window.monitor_combo.findData("both"))
    assert "RandomHorizontalFlip" in window.aug_preview.text()

    window.start_training()
    assert window.is_training() and not window.config_tabs.isEnabled()
    wait_until(app, lambda: window.thread is None)
    assert window.run_progress.value() == 2
    assert window.batch_progress.value() == window.batch_progress.maximum() == 3
    assert messages and "Total epochs trained: 2" in messages[-1]
    assert " / 80.0%" in window.quota_bars["accuracy_floor"].format()
    assert window.class_quota_table.rowCount() == 3
    assert " / " in window.class_quota_table.item(0, 1).text()
    assert "Δ" not in window.cm_norm_canvas.ax.get_title()  # first run: no previous run yet
    assert window.baseline_combo.count() == 3
    window.baseline_combo.setCurrentIndex(2)  # the run that just finished
    assert "Δ vs. Training run" in window.cm_norm_canvas.ax.get_title()
    assert "(+0.0%)" in [t.get_text() for t in window.cm_norm_canvas.ax.texts]
    window.baseline_combo.setCurrentIndex(0)

    project = manager.load("default")
    assert project.checked_train_folders() == [str(datasets["a"])]
    assert project.augmentation.hflip and project.seed == 42 and project.quota.monitor == "both"
    assert project.checked_val_folders() == [str(datasets["val"])]
    assert project.checked_test_folders() == [str(datasets["test"])]
    assert project.evaluations[-1]["total_epochs_completed"] == 2
    assert window.checkpoint_list.count() == 4  # last, best, 2 snapshots
    assert "Total cumulative epochs trained:</b> 2" in window.provenance_label.text()
    assert window.milestone_container.count() == 4

    assert "Active validation sources" in window.provenance_label.text()
    assert str(datasets["test"]) not in window.provenance_label.text()
    assert window.review_folders() == ("test sets (final exam)", [str(datasets["test"])])

    window.run_final_exam()
    assert "Test samples: 6" in messages[-1]
    exam = manager.load("default").evaluations[-1]
    assert exam["type"] == "test" and exam["test_folders"] == [str(datasets["test"])]
    assert exam["seen_in_training"] == []
    assert len(exam["confusion"]) == len(exam["classes"]) == 3
    assert window.baseline_combo.count() == 4
    assert window.class_quota_table.rowCount() == 3

    session = ReviewSession(window.selected_checkpoint(), project.checked_test_folders(), seed=0)
    dialog = ReviewDialog(session, window.selected_checkpoint(), window)
    assert len(session.history) == 1
    assert dialog.badge.text() in ("CORRECT", "INCORRECT")
    dialog.show_next()
    dialog.show_previous()
    assert session.position == 0 and not dialog.prev_button.isEnabled()
    dialog.close()

    exec_calls = []
    monkeypatch.setattr(ImageTestDialog, "exec", lambda d: exec_calls.append(d) or 0)
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a[2]))
    window.open_picture_test()
    picture_dialog = exec_calls[-1]
    assert picture_dialog.detector_combo.count() == 0
    assert "no detector yet" in picture_dialog.status.text()
    picture = tmp_path / "board.png"
    Image.new("RGB", (48, 24), (245, 245, 245)).save(picture)
    assert picture_dialog.test_picture(str(picture)) is None
    assert "No YOLO detector loaded" in warnings[-1]

    detectors_dir = manager.project_dir("default") / "checkpoints" / "detectors"
    assert picture_dialog.import_picture_detector(window.selected_checkpoint()) is None
    assert "classifier checkpoint" in warnings[-1] and not detectors_dir.exists()

    monkeypatch.setattr(detector_module, "load_yolo_detector", lambda path: FakeDetector())
    monkeypatch.setattr(review_dialog_module, "load_yolo_detector", lambda path: FakeDetector())
    outside = tmp_path / "weights.pt"
    outside.write_bytes(b"detector")
    imported = picture_dialog.import_picture_detector(str(outside))
    assert Path(imported).parent == detectors_dir.resolve()
    assert picture_dialog.tester.detector_path == imported
    assert picture_dialog.detector_combo.currentData() == imported
    assert "Ready" in picture_dialog.status.text()
    assert picture_dialog.import_picture_detector(imported) == imported  # no duplicate copy
    assert window.project_detectors() == [imported] and picture_dialog.detector_combo.count() == 1
    window.refresh_checkpoints()
    assert window.checkpoint_list.count() == 4  # detectors are not classifier checkpoints

    result = picture_dialog.test_picture(str(picture))
    detections = result["detections"]
    assert [d["detector"][0] for d in detections] == ["White_Rook", "black-king"]  # reading order
    assert [d["label"] for d in detections] == ["White_Rook", "black-king"]  # the detector's names
    assert picture_dialog.pieces.count() == 2
    assert result["annotated"].tobytes() != result["image"].tobytes()
    assert picture_dialog.pieces.item(0).text() == "1. White_Rook  90%"
    picture_dialog.conf_slider.setValue(50)
    assert picture_dialog.conf == 0.5 and picture_dialog.conf_label.text() == "50%"
    not_an_image = tmp_path / "notes.txt"
    not_an_image.write_text("no picture", encoding="utf-8")
    assert picture_dialog.test_picture(str(not_an_image)) is None
    assert "Cannot test this picture" in warnings[-1]
    picture_dialog.close()

    window.use_selected_for_resume()
    assert window.resume_check.isChecked()

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    manager.create("second")
    window.project_combo.addItem("second")
    window.project_combo.setCurrentText("second")
    assert window.project.name == "second" and window.train_folders.list.count() == 0
    window.delete_project()
    assert manager.list_projects() == ["default"] and window.project.name == "default"
    window.close()


def test_yaml_badges_and_sync(app, tmp_path, monkeypatch):
    from conftest import make_yolo_split

    a = make_yolo_split(tmp_path / "a" / "train", names=("W-K", "black-king"))
    b = make_yolo_split(tmp_path / "b" / "train", names=("black-king", "White_King"))
    plain = tmp_path / "plain" / "white-king"
    plain.mkdir(parents=True)
    window = MainWindow(ProjectManager(tmp_path / ".projects"))
    window.train_folders.set_entries([DatasetEntry(str(a)), DatasetEntry(str(plain.parent))])
    window.val_folders.set_entries([DatasetEntry(str(b))])
    items = [window.train_folders.list.item(i) for i in range(2)]
    assert "YAML detected" in items[0].toolTip() and not items[0].icon().isNull()
    assert "No .yaml" in items[1].toolTip()
    assert window.train_folders.entries()[0].path == str(a)

    dialogs = []

    def fake_exec(dialog):
        dialogs.append(dialog)
        dialog.combos["W-K"].setCurrentText("white-king")
        dialog.combos["White_King"].setCurrentText("white-king")
        assert "0: white-king, 1: black-king" in dialog.preview.text()
        dialog.apply_changes()
        return 1

    monkeypatch.setattr(YamlSyncDialog, "exec", fake_exec)
    window.open_yaml_sync()
    assert dialogs and dialogs[0].table.rowCount() == 3
    assert len(dialogs[0].tags["black-king"]) == 2
    assert "white-king" in window.log_view.toPlainText()
    assert window._safe_counts([str(a), str(b)]) == {"black-king": 6, "white-king": 6}
    assert (b / "labels" / "board0.txt").read_text(encoding="utf-8").startswith("1 ")
    window.close()


def test_detection_gui_workflow(app, tmp_path, monkeypatch):
    pytest.importorskip("ultralytics")
    from conftest import make_yolo_split

    messages, dialogs = [], []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: messages.append(a[2]))
    monkeypatch.setattr(DetectionReviewDialog, "exec", lambda d: dialogs.append(d) or 0)
    monkeypatch.setattr(ImageTestDialog, "exec", lambda d: dialogs.append(d) or 0)
    train = make_yolo_split(tmp_path / "data" / "train")
    test = make_yolo_split(tmp_path / "data" / "test")
    manager = ProjectManager(tmp_path / ".projects")
    window = MainWindow(manager)
    window.detector_device = "cpu"
    assert window.task == "detect"
    assert not window.detect_box.isHidden() and window.quota_box.isHidden()
    assert all(g.isHidden() for g in window.classify_groups)

    window.train_folders.set_entries([DatasetEntry(str(train))])
    window.test_folders.set_entries([DatasetEntry(str(test))])
    window.on_datasets_changed()
    window.det_base_combo.setCurrentText("yolo26n.yaml")  # built from scratch: no download
    window.det_epochs_spin.setValue(2)
    window.det_image_size_spin.setValue(64)
    window.det_batch_spin.setValue(3)
    window.det_patience_spin.setValue(0)
    window.aug_hflip.setChecked(True)
    assert "fliplr=0.5" in window.aug_preview.text()

    window.start_training()
    assert window.is_training() and window.pause_button.isEnabled()
    window.toggle_pause()
    wait_until(app, lambda: window.paused_checkpoint is not None)
    assert window.review_folders()[0] == "training pools"
    window.open_review()
    assert dialogs and "objects correct" in dialogs[-1].badge.text()
    window.toggle_pause()
    wait_until(app, lambda: window.thread is None)

    assert "Training finished: all epochs completed" in messages[-1]
    weights = window.project_detectors()[0]
    assert window.checkpoint_list.count() == 1 and window.selected_checkpoint() == weights
    assert "Started from:</b> yolo26n.yaml" in window.provenance_label.text()
    assert window.project.evaluations[-1]["type"] == "detect"
    assert window.project.evaluations[-1]["classes"] == ["black-king", "white-king", "background"]
    assert window.det_metric_bars["mAP50"].format().endswith("%")

    window.run_final_exam()
    assert "Test images: 3" in messages[-1]
    assert window.project.evaluations[-1]["type"] == "detect_test"
    assert window.baseline_combo.count() == 4  # previous, none, run, exam

    window.open_picture_test()
    picture_dialog = dialogs[-1]
    assert picture_dialog.detector_combo.currentData() == weights
    assert picture_dialog.test_picture(str(next((train / "images").iterdir()))) is not None

    window.use_selected_for_resume()
    window.task_combo.setCurrentIndex(window.task_combo.findData("classify"))
    assert window.resume_edit.text() == "" and window.checkpoint_list.count() == 0
    assert window.detect_box.isHidden() and not window.quota_box.isHidden()
    window.task_combo.setCurrentIndex(window.task_combo.findData("detect"))
    assert window.resume_edit.text() == weights
    assert manager.load("default").detector.resume_from == weights
    window.close()
