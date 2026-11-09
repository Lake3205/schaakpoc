import time

import pytest

pytest.importorskip("PyQt6.QtWidgets", exc_type=ImportError)
from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from schaakgui.main_window import MainWindow  # noqa: E402
from schaakgui.projects import DatasetEntry, ProjectManager  # noqa: E402
from schaakgui.review_dialog import ReviewDialog, ReviewSession  # noqa: E402


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
