"""Detection -> board state -> FEN pipeline (kept separate from the training code).

Stages: piece base points (``detection``), frame-0 homography calibration (``calibration``),
temporal legal-move tracking (``tracker``), annotated image rendering (``render``) and the
per-move export of images and FEN strings (``session``).
"""
from .calibration import BoardCalibration, CalibrationError, calibrate
from .detection import Detection, base_point, detections_from_result
from .render import render_annotated
from .session import FenSession, SessionFrame
from .tracker import FrameResult, GameTracker

__all__ = [
    "BoardCalibration", "CalibrationError", "calibrate",
    "Detection", "base_point", "detections_from_result",
    "render_annotated",
    "FenSession", "SessionFrame",
    "FrameResult", "GameTracker",
]
