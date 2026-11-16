"""Stage 4b: per-move export of the annotated image and the FEN session log."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .detection import Detection
from .render import render_annotated
from .tracker import FrameResult, GameTracker

FEN_LOG = "moves.fen"


@dataclass
class SessionFrame:
    """Export of one processed frame: the engine-ready FEN and the annotated image path."""

    fen: str
    image_path: Path
    result: FrameResult


class FenSession:
    """Processes the frames of one game and writes ``move_<num>.<ext>`` images plus a
    ``moves.fen`` log (one FEN per line) to ``output_dir``."""

    def __init__(self, output_dir: Path, tracker: GameTracker | None = None,
                 image_ext: str = ".jpg") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = tracker if tracker is not None else GameTracker()
        self.image_ext = image_ext if image_ext.startswith(".") else f".{image_ext}"
        self.fen_log = self.output_dir / FEN_LOG
        self.fen_log.write_text("", encoding="utf-8")

    def process(self, image: np.ndarray, detections: list[Detection]) -> SessionFrame:
        """Track the next frame, then export its annotated image and append its FEN."""
        result = self.tracker.process(detections)
        annotated = render_annotated(image, detections, result.squares, self.tracker.calibration,
                                     result.fen, result.frame, result.san)
        image_path = self.output_dir / f"move_{result.frame}{self.image_ext}"
        if not cv2.imwrite(str(image_path), annotated):
            raise OSError(f"Could not write image: {image_path}")
        with self.fen_log.open("a", encoding="utf-8") as f:
            f.write(result.fen + "\n")
        return SessionFrame(result.fen, image_path, result)
