"""Evaluation of the scan pipeline for the report (Qt independent).

* board model: corner error as % of the board diagonal (+ the pose mAP from Ultralytics);
* full pipeline: square accuracy, exact boards, boards with at most one error and the step
  where a scan failed (no corners / wrong orientation / wrong piece).
"""
from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Callable, Optional, Sequence

from .board import Corners, order_corners


def corner_error(pred: Corners, truth: Corners) -> float:
    """Mean distance between predicted and true corners as % of the true board diagonal."""
    diagonal = max(math.dist(truth[0], truth[2]), math.dist(truth[1], truth[3]), 1e-9)
    errors = [math.dist(p, t) for p, t in zip(pred, truth)]
    return 100.0 * sum(errors) / len(errors) / diagonal


def best_corners(result) -> Optional[Corners]:
    """The 4 keypoints (image order) of the most confident ``board`` in an Ultralytics result."""
    keypoints = getattr(result, "keypoints", None)
    boxes = getattr(result, "boxes", None)
    if keypoints is None or boxes is None or len(boxes) == 0:
        return None
    scores = boxes.conf.tolist()
    best = max(range(len(scores)), key=scores.__getitem__)
    points = keypoints.xy[best].tolist()
    return order_corners(points) if len(points) == 4 else None


def board_corner_errors(model, folders: Sequence[str], conf: float = 0.25,
                        device=None) -> list[Optional[float]]:
    """Corner error per labelled image of the YOLO-pose folders (``None`` = no board found)."""
    from .detector import load_picture, yolo_images
    from .keypoints import read_board_label

    if device is None:
        import torch

        device = 0 if torch.cuda.is_available() else "cpu"
    errors: list[Optional[float]] = []
    for image_path, label_file, _ in yolo_images(folders):
        image = load_picture(image_path)
        truth = read_board_label(label_file, *image.size)
        if truth is None:
            continue
        result = model(image, conf=conf, device=device, verbose=False)[0]
        pred = best_corners(result)
        errors.append(corner_error(pred, truth) if pred else None)
    return errors


def summarize_corner_errors(errors: Sequence[Optional[float]], within: float = 2.0) -> dict:
    """Mean/median corner error, % within ``within`` % of the diagonal and missed boards."""
    found = [e for e in errors if e is not None]
    return {
        "corner_images": len(errors),
        "corner_missed": len(errors) - len(found),
        "corner_error": statistics.fmean(found) if found else None,
        "corner_error_median": statistics.median(found) if found else None,
        "corner_within_2pct": 100.0 * sum(e <= within for e in found) / len(errors) if errors else 0.0,
    }


def evaluate_board_model(weights: str | Path, folders: Sequence[str], **kwargs) -> dict:
    """Pose mAP + mean/median corner error + % of boards within 2 % of the diagonal."""
    from .detector import evaluate_detector

    return evaluate_detector(weights, folders, **kwargs)


def square_accuracy(pred: dict[str, str], truth: dict[str, str]) -> float:
    """Correct squares (empty squares included) / 64."""
    from .position import FILES

    squares = [f"{f}{r}" for f in FILES for r in range(1, 9)]
    return sum(pred.get(s) == truth.get(s) for s in squares) / 64


def _rotations_match(session, truth: dict[str, str]) -> bool:
    """Whether some a1 corner gives the true position (then only the orientation was wrong)."""
    current = session.a1_corner
    try:
        for corner in range(4):
            session.set_a1_corner(corner)
            if session.squares() == truth:
                return True
        return False
    finally:
        session.set_a1_corner(current)


def evaluate_scans(cases: Sequence[dict], piece_weights: Optional[str] = None,
                   board_weights: Optional[str] = None, conf: float = 0.25,
                   session=None, progress: Optional[Callable[[int, int], None]] = None) -> dict:
    """Full pipeline on known positions.

    ``cases`` are ``{"image": path, "fen": placement or full FEN, "corners": optional}``. Without
    a board model (or when it fails) given ``corners`` are not used: a scan without corners counts as
    a ``no_corners`` failure, so the numbers show the whole automatic pipeline."""
    from .board import BoardFinder
    from .detector import ImageTester
    from .fen import squares_from_fen
    from .scan_session import ScanSession

    if session is None:
        session = ScanSession(ImageTester(piece_weights), BoardFinder(board_weights))
    failures = {"no_corners": 0, "orientation": 0, "pieces": 0}
    accuracies: list[float] = []
    exact = within_one = 0
    rows = []
    for i, case in enumerate(cases):
        truth = squares_from_fen(case["fen"])
        session.load_photo(case["image"])
        proposal = session.auto_corners()
        if proposal is None:
            failures["no_corners"] += 1
            accuracies.append(0.0)
            rows.append({"image": str(case["image"]), "failure": "no_corners", "accuracy": 0.0})
            if progress:
                progress(i + 1, len(cases))
            continue
        session.detect_pieces(conf)
        session.suggest_orientation()
        pred = session.squares()
        acc = square_accuracy(pred, truth)
        accuracies.append(acc)
        errors = round((1 - acc) * 64)
        exact += errors == 0
        within_one += errors <= 1
        failure = None
        if errors:
            failure = "orientation" if _rotations_match(session, truth) else "pieces"
            failures[failure] += 1
        rows.append({"image": str(case["image"]), "failure": failure, "accuracy": acc,
                     "fen": session.fen(), "truth": case["fen"]})
        if progress:
            progress(i + 1, len(cases))
    n = len(cases)
    return {
        "boards": n,
        "square_accuracy": 100.0 * statistics.fmean(accuracies) if accuracies else 0.0,
        "exact_boards": 100.0 * exact / n if n else 0.0,
        "within_one_error": 100.0 * within_one / n if n else 0.0,
        "failures": failures,
        "cases": rows,
    }
