"""QThread worker wrapping :class:`schaakgui.engine.Trainer` (Worker-Signal-Slot pattern)."""
from __future__ import annotations

import traceback

from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from .engine import Callbacks, Trainer, TrainJob


class TrainingWorker(QObject):
    log = pyqtSignal(str)
    batch_progress = pyqtSignal(int, int)
    epoch_progress = pyqtSignal(int, int)
    evaluated = pyqtSignal(dict)
    milestone = pyqtSignal(dict)
    paused = pyqtSignal(str)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, job: TrainJob):
        super().__init__()
        self.trainer = Trainer(job, Callbacks(
            log=self.log.emit,
            batch_progress=self.batch_progress.emit,
            epoch_progress=self.epoch_progress.emit,
            evaluated=self.evaluated.emit,
            milestone=self.milestone.emit,
            paused=self.paused.emit,
        ))

    @pyqtSlot()
    def run(self):
        try:
            summary = self.trainer.run()
        except Exception as e:  # report every failure to the UI instead of killing the thread
            self.log.emit(traceback.format_exc())
            self.failed.emit(str(e))
        else:
            self.finished.emit(summary)

    # These are called directly from the GUI thread; the trainer uses thread-safe events.
    def request_stop(self):
        self.trainer.request_stop()

    def set_paused(self, paused: bool):
        if paused:
            self.trainer.pause()
        else:
            self.trainer.resume()


def start_worker(job: TrainJob) -> tuple[QThread, TrainingWorker]:
    """Create a worker moved to a new QThread. Connect signals, then call ``thread.start()``."""
    thread = QThread()
    worker = TrainingWorker(job)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    thread.finished.connect(worker.deleteLater)
    return thread, worker
