import time

import pytest

pytest.importorskip("PyQt6.QtWidgets", exc_type=ImportError)
from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from schaakgui.main_window import MainWindow  # noqa: E402
from schaakgui.projects import DatasetEntry, ProjectManager  # noqa: E402
from schaakgui.review_dialog import ReviewDialog, ReviewSession  # noqa: E402
from schaakgui.sync_dialog import YamlSyncDialog  # noqa: E402


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
    assert window.project.name == "default"

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
