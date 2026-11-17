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


def test_scan_dialog_corrections_and_exports(app, tmp_path, monkeypatch):
    from PyQt6.QtCore import QPointF, Qt
    from PyQt6.QtTest import QTest

    from schaakgui.board import BoardFinder
    from schaakgui.scan_dialog import ScanDialog
    from schaakgui.scan_session import ScanSession
    from scan_helpers import PHOTO_CORNERS, FakeTester, board_photo, draw_pieces, piece_boxes

    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a[2]))
    boxes = piece_boxes()
    photo = tmp_path / "garden.png"
    draw_pieces(board_photo(), boxes).save(photo)
    tester = FakeTester(boxes)
    dialog = ScanDialog(ScanSession(tester, BoardFinder(None)), sample_root=tmp_path / "samples")
    dialog.resize(1400, 900)
    dialog.show()
    assert dialog.current_step() == 0 and not dialog.detect_button.isEnabled()

    # the checkerboard is hidden by pieces: OpenCV may fail, so set the true corners by hand
    assert dialog.open_photo(str(photo), detect=False)
    dialog.session.set_corners(PHOTO_CORNERS)
    dialog.refresh()
    assert dialog.current_step() == 2
    assert dialog.detect_pieces()
    assert not dialog.models_row.isEnabled()  # no model swaps during detection
    wait_until(app, lambda: not dialog.busy, 30)
    assert dialog.models_row.isEnabled()
    assert tester.calls == 1 and not warnings
    assert dialog.fen_edit.text().startswith("4k3/p7/2n5/8/8/8/P6P/3QK3 w")
    assert dialog.current_step() == 3 and "Nothing to review" in dialog.problems.item(0).text()

    # click c6 on the diagram and choose a bishop instead of the knight
    monkeypatch.setattr(dialog, "ask_piece", lambda square, *a, **k: "b" if square == "c6" else ...)
    QTest.mouseClick(dialog.diagram, Qt.MouseButton.LeftButton,
                     pos=dialog.diagram.widget_point("c6").toPoint())
    assert dialog.fen_edit.text().startswith("4k3/p7/2b5/")

    # rotation and flipping keep the correction and never re-run detection
    dialog.rotate()
    dialog.rotate()
    assert dialog.session.a1_corner == 1 and dialog.current_step() == 4
    assert dialog.session.squares()["f3"] == "b"
    dialog.flip_diagram()
    assert dialog.diagram.square_at(dialog.diagram.widget_point("f3")) == "f3"
    dialog.suggest_orientation()
    assert dialog.session.a1_corner == 3

    # drag the bottom-left handle a bit: the grid follows, no new detection
    view = dialog.view
    start = view._to_widget(*PHOTO_CORNERS[3])
    assert view._to_photo(start) == pytest.approx(PHOTO_CORNERS[3], abs=0.01)
    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=start.toPoint())
    QTest.mouseMove(view, (start + QPointF(3, 0)).toPoint())
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=(start + QPointF(3, 0)).toPoint())
    assert dialog.session.corners[3] != PHOTO_CORNERS[3] and dialog.session.corner_method == "manual"
    assert tester.calls == 1

    # sliders filter without detecting again
    dialog.conf_slider.setValue(95)
    assert dialog.session.squares() == {"c6": "b"}
    dialog.conf_slider.setValue(25)
    dialog.info_box.side.setCurrentIndex(1)
    assert " b " in dialog.fen_edit.text()

    out = tmp_path / "out"
    assert dialog.save_fen(out / "pos.txt")
    assert (out / "pos.txt").read_text().strip() == dialog.session.fen()
    dialog.save_annotated(out / "annotated.png")
    dialog.save_diagram(out / "diagram.png")
    dialog.export_all(out / "all")
    assert sorted(p.name for p in (out / "all").iterdir()) == [
        "1_corners.png", "2_straight.png", "3_pieces.png", "4_feet.png", "5_diagram.png",
        "annotated.png", "diagram.png", "fen.txt", "scan.json"]
    assert Image.open(out / "diagram.png").size == (640, 640)
    sample = dialog.save_training_sample(split="valid")
    assert sample["boxes"] == 7 and Path(sample["board_label"]).is_file()
    assert tester.calls == 1
    dialog.close()


def test_data_tools_dialog(app, tmp_path):
    from schaakgui.data_tools_dialog import DataToolsDialog
    from test_importers import make_chessred

    dialog = DataToolsDialog(tmp_path / "datasets", task="board")
    dialog.syn_count.setValue(4)
    dialog.syn_size.setValue(128)
    dialog.syn_val.setValue(0.25)
    dialog.syn_seed.setValue(7)
    assert dialog.start_synthetic()
    assert not dialog.start_chessred()  # busy
    wait_until(app, lambda: not dialog.busy, 60)
    assert dialog.last_error is None and "train 3, valid 1" in dialog.status.text()
    assert dialog.progress.value() == dialog.progress.maximum()

    assert not dialog.start_chessred() and "annotations.json" in dialog.status.text()
    dialog.cr_json.setText(str(make_chessred(tmp_path / "chessred")))
    dialog.cr_images.setText(str(tmp_path / "chessred"))
    assert dialog.start_chessred()
    wait_until(app, lambda: not dialog.busy, 60)
    assert "ChessReD2K: train 1, valid 1, test 1" in dialog.status.text()

    board = dialog.created_folders()
    assert len(board["train"]) == 2 and len(board["valid"]) == 2 and len(board["test"]) == 1
    assert all(Path(p).name in ("train", "valid", "test") for p in board["train"])
    pieces = dialog.created_folders("detect")
    assert [Path(p).parent.name for p in pieces["train"]] == ["pieces"]

    dialog.syn_count.setValue(5000)
    assert dialog.start_synthetic()
    dialog.cancel()
    wait_until(app, lambda: not dialog.busy, 60)
    assert "stopped early" in dialog.status.text()
    dialog.close()


def test_board_task_and_scanner_buttons(app, tmp_path, monkeypatch):
    import schaakgui.main_window as main_window_module

    manager = ProjectManager(tmp_path / ".projects")
    window = MainWindow(manager)
    window.det_base_combo.setCurrentText("yolo26n.pt")
    window.resume_check.setChecked(True)
    window.resume_edit.setText("pieces.pt")
    window.task_combo.setCurrentIndex(window.task_combo.findData("board"))
    assert window.task == "board" and window.yolo_task
    assert window.det_base_combo.currentText() == "yolo26s-pose.pt"
    assert "Board Corner" in window.windowTitle() and "boards" in window.detect_help.text()
    assert window.resume_edit.text() == "" and not window.resume_check.isChecked()
    window.det_freeze_spin.setValue(10)
    window.resume_check.setChecked(True)
    window.resume_edit.setText("corners.pt")
    window.save_project()  # start_detector_training saves the form before building the job
    job = window.build_detector_job()
    assert job.task == "board" and job.config.base_model == "yolo26s-pose.pt"
    assert job.config.freeze == 10 and job.config.resume_from == "corners.pt"
    window.task_combo.setCurrentIndex(window.task_combo.findData("detect"))
    d = manager.load("default").detector
    assert (d.base_model, d.resume_from, d.board_resume_from) == ("yolo26n.pt", "pieces.pt", "corners.pt")
    assert window.det_base_combo.currentText() == "yolo26n.pt" and window.resume_edit.text() == "pieces.pt"
    assert window.build_detector_job().task == "detect"
    assert window._record_task({"type": "board_test"}) == "board"
    assert "corner error 1.50%" in window._evaluation_label(
        {"type": "board_test", "mAP50": 0.9, "corner_error": 1.5})

    # Data tools: created split folders are appended to the matching dataset lists.
    created = {"train": [str(tmp_path / "b" / "train")], "valid": [str(tmp_path / "b" / "valid")],
               "test": [str(tmp_path / "b" / "test")]}
    seen = {}

    def fake_tools_exec(dialog):
        seen["tools"] = (dialog.output_root, dialog.task)
        return 0

    monkeypatch.setattr(main_window_module.DataToolsDialog, "exec", fake_tools_exec)
    monkeypatch.setattr(main_window_module.DataToolsDialog, "created_folders", lambda self: created)
    window.data_tools_button.click()
    assert seen["tools"][1] == "detect" and seen["tools"][0].name == "datasets"
    assert window.train_folders.checked() == created["train"]
    assert window.val_folders.checked() == created["valid"]
    window.data_tools_button.click()  # no duplicates
    assert window.test_folders.checked() == created["test"]
    window.task_combo.setCurrentIndex(window.task_combo.findData("classify"))
    window.train_folders.set_entries([])
    window.data_tools_button.click()  # YOLO folders are not added to classifier datasets
    assert window.train_folders.entries() == []
    window.task_combo.setCurrentIndex(window.task_combo.findData("detect"))

    def fake_scan_exec(dialog):
        seen["scan"] = dialog
        dialog.session.conf = 0.4
        return 0

    monkeypatch.setattr(main_window_module.ScanDialog, "exec", fake_scan_exec)
    window.scan_button.click()
    assert seen["scan"].import_callback == window.import_project_model
    assert seen["scan"].sample_root.name == "scan_samples"
    assert manager.load("default").scan.conf == pytest.approx(0.4)
    window.close()
