"""QThread worker wrapping :class:`schaakgui.engine.Trainer` (Worker-Signal-Slot pattern)."""
from __future__ import annotations

import threading
import traceback

from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from .detector import DetectorJob, DetectorTrainer
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
    return _start(TrainingWorker(job))


class DetectorWorker(QObject):
    """Runs :class:`schaakgui.detector.DetectorTrainer` (same signals as :class:`TrainingWorker`)."""

    log = pyqtSignal(str)
    batch_progress = pyqtSignal(int, int)
    epoch_progress = pyqtSignal(int, int)
    evaluated = pyqtSignal(dict)
    paused = pyqtSignal(str)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, job: DetectorJob):
        super().__init__()
        self.trainer = DetectorTrainer(job, self.log.emit, self.epoch_progress.emit,
                                       self.batch_progress.emit, self.evaluated.emit,
                                       self.paused.emit)

    @pyqtSlot()
    def run(self):
        try:
            summary = self.trainer.run()
        except Exception as e:
            self.log.emit(traceback.format_exc())
            self.failed.emit(str(e))
        else:
            self.finished.emit(summary)

    def request_stop(self):
        self.trainer.request_stop()

    def set_paused(self, paused: bool):
        if paused:
            self.trainer.pause()
        else:
            self.trainer.resume()


def start_detector_worker(job: DetectorJob) -> tuple[QThread, DetectorWorker]:
    return _start(DetectorWorker(job))


class CallWorker(QObject):
    """Runs one function off the GUI thread (piece detection, dataset generation, imports).

    With ``with_progress`` the function also gets ``progress=(i, total) -> None`` and
    ``stop=() -> bool`` keyword arguments (stop is set by :meth:`request_stop`)."""

    progress = pyqtSignal(int, int)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, func, *args, with_progress: bool = False, **kwargs):
        super().__init__()
        self.func, self.args, self.kwargs = func, args, kwargs
        self.with_progress = with_progress
        self._stop = threading.Event()

    @pyqtSlot()
    def run(self):
        kwargs = dict(self.kwargs)
        if self.with_progress:
            kwargs.update(progress=self.progress.emit, stop=self._stop.is_set)
        try:
            result = self.func(*self.args, **kwargs)
        except Exception as e:  # report every failure to the UI instead of killing the thread
            traceback.print_exc()
            self.failed.emit(str(e) or type(e).__name__)
        else:
            self.finished.emit(result)

    def request_stop(self):
        self._stop.set()


def start_call(func, *args, with_progress: bool = False, **kwargs) -> tuple[QThread, CallWorker]:
    """``func(*args, **kwargs)`` in a new QThread. Connect signals, then call ``thread.start()``."""
    return _start(CallWorker(func, *args, with_progress=with_progress, **kwargs))


def _start(worker):
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    thread.finished.connect(worker.deleteLater)
    return thread, worker
