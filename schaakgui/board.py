"""Board geometry (scan steps 1-2, Qt independent): corners, homography, 8x8 grid, corner finding.

Corners are always kept in *image order*: top-left, top-right, bottom-right, bottom-left (clockwise,
starting at the corner nearest the photo's top-left). Which corner is a1 is decided later
(:mod:`schaakgui.position`), so the geometry never depends on the chess orientation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
from PIL import Image

Point = tuple[float, float]  # pixel (x, y) in the original photo
Corners = tuple[Point, Point, Point, Point]  # image order: TL, TR, BR, BL
Cell = tuple[int, int]  # (col, row) in the straightened grid, 0..7, row 0 = top
BOARD_PX = 800  # side of the straightened board image
CORNER_NAMES = ("TL", "TR", "BR", "BL")


def _as_array(image) -> np.ndarray:
    """RGB uint8 array of a PIL image or numpy array."""
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    return np.asarray(image)


def order_corners(points: Sequence[Point]) -> Corners:
    """Sort 4 points clockwise (in image coordinates, y down), starting at the corner nearest (0, 0)."""
    pts = [(float(x), float(y)) for x, y in points]
    if len(pts) != 4:
        raise ValueError(f"Need exactly 4 corner points, got {len(pts)}")
    cx = sum(p[0] for p in pts) / 4
    cy = sum(p[1] for p in pts) / 4
    # With y pointing down, increasing atan2 angle runs clockwise on screen.
    pts.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    start = min(range(4), key=lambda i: pts[i][0] ** 2 + pts[i][1] ** 2)
    ordered = pts[start:] + pts[:start]
    return tuple(ordered)  # type: ignore[return-value]


def _square_points(size: int) -> np.ndarray:
    return np.float32([[0, 0], [size, 0], [size, size], [0, size]])


def board_homography(corners: Corners, size: int = BOARD_PX) -> np.ndarray:
    """3x3 matrix mapping photo pixels onto the straight ``size`` x ``size`` board."""
    return cv2.getPerspectiveTransform(np.float32(corners), _square_points(size))


def _apply(h: np.ndarray, point: Point) -> Point:
    x, y, w = h @ np.array([point[0], point[1], 1.0])
    if abs(w) < 1e-12:
        return (float("inf"), float("inf"))
    return (float(x / w), float(y / w))


def to_board(h: np.ndarray, point: Point) -> Point:
    """Map one photo point onto the straight board."""
    return _apply(h, point)


def to_image(h: np.ndarray, point: Point) -> Point:
    """Map one straight-board point back into the photo (inverse of :func:`to_board`)."""
    return _apply(np.linalg.inv(h), point)


def warp_board(image, corners: Corners, size: int = BOARD_PX) -> Image.Image:
    """Straightened top-down board image (debug step 2 and the orientation rule)."""
    array = _as_array(image)
    warped = cv2.warpPerspective(array, board_homography(corners, size), (size, size),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return Image.fromarray(warped)


def cell_at(board_point: Point, size: int = BOARD_PX) -> Optional[Cell]:
    """``(col, row)`` of a straight-board point, ``None`` outside the board."""
    x, y = board_point
    if not (0 <= x < size and 0 <= y < size):
        return None
    square = size / 8
    return (min(7, int(x // square)), min(7, int(y // square)))


def edge_distance(board_point: Point, size: int = BOARD_PX) -> float:
    """Distance to the nearest square edge as a fraction of a square (0 = on a line, 0.5 = centre)."""
    square = size / 8
    fx = board_point[0] / square
    fy = board_point[1] / square
    fx -= math.floor(fx)
    fy -= math.floor(fy)
    return float(min(fx, 1 - fx, fy, 1 - fy))


def grid_lines(corners: Corners, size: int = BOARD_PX) -> list[tuple[Point, Point]]:
    """The 9 vertical + 9 horizontal grid lines in photo coordinates (for the overlay)."""
    h = board_homography(corners, size)
    lines = []
    for i in range(9):
        t = size * i / 8
        lines.append((to_image(h, (t, 0)), to_image(h, (t, size))))
    for i in range(9):
        t = size * i / 8
        lines.append((to_image(h, (0, t)), to_image(h, (size, t))))
    return lines


def inner_to_outer(inner: np.ndarray) -> Corners:
    """7x7 inner checkerboard corners → the 4 outer board corners (one square further out).

    A homography is fitted from the ideal grid (1..7 squares) to the found points, so the
    extrapolation also follows the perspective."""
    pts = np.asarray(inner, dtype=np.float32).reshape(-1, 2)
    if len(pts) != 49:
        raise ValueError(f"Expected 49 inner corners, got {len(pts)}")
    ideal = np.float32([[c + 1, r + 1] for r in range(7) for c in range(7)])
    h, _ = cv2.findHomography(ideal, pts)
    if h is None:
        raise ValueError("Cannot fit the inner checkerboard corners")
    outer = [_apply(h, p) for p in ((0, 0), (8, 0), (8, 8), (0, 8))]
    return order_corners(outer)


@dataclass
class CornerProposal:
    corners: Corners
    method: str  # "model", "opencv" or "manual"
    score: float = 1.0


def find_corners_opencv(image, max_side: int = 1000) -> Optional[CornerProposal]:
    """Classical fallback: ``cv2.findChessboardCornersSB`` on the 7x7 inner corners of an (almost)
    empty board, extrapolated one square outward. ``None`` when no checkerboard is found."""
    array = _as_array(image)
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY) if array.ndim == 3 else array
    scale = min(1.0, max_side / max(gray.shape[:2]))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    try:
        found, inner = cv2.findChessboardCornersSB(gray, (7, 7), flags=cv2.CALIB_CB_EXHAUSTIVE)
    except cv2.error:
        return None
    if not found or inner is None:
        return None
    corners = inner_to_outer(np.asarray(inner).reshape(-1, 2) / scale)
    return CornerProposal(corners, "opencv", 1.0)


def model_kpt_shape(model) -> Optional[list[int]]:
    """``kpt_shape`` of an Ultralytics pose model, if known."""
    inner = getattr(model, "model", None)
    shape = getattr(inner, "kpt_shape", None)
    if shape is None:
        shape = (getattr(inner, "yaml", None) or {}).get("kpt_shape")
    return list(shape) if shape is not None else None


def load_board_model(path: str | Path):
    """Load a YOLO-pose board corner model; clear errors for other tasks or keypoint counts."""
    from .detector import load_yolo_model

    model = load_yolo_model(path, task="pose")
    shape = model_kpt_shape(model)
    if shape is None or int(shape[0]) != 4:
        raise ValueError(f"{Path(path).name} has keypoints {shape}; a board corner model needs "
                         "4 keypoints (TL, TR, BR, BL)")
    return model


class BoardFinder:
    """Tries the corner sources in order: board model → OpenCV checkerboard finder → ``None``."""

    def __init__(self, model_path: Optional[str | Path] = None, use_opencv: bool = True):
        self.model = None
        self.model_path: Optional[str] = None
        self.use_opencv = use_opencv
        if model_path:
            self.load_model(model_path)

    def load_model(self, path: Optional[str | Path]):
        """Select another board model (``None`` = no model, OpenCV only)."""
        if not path:
            self.model, self.model_path = None, None
            return
        self.model = load_board_model(path)
        self.model_path = str(path)

    def propose(self, image, conf: float = 0.25) -> Optional[CornerProposal]:
        if self.model is not None:
            proposal = self._from_model(image, conf)
            if proposal is not None:
                return proposal
        if self.use_opencv:
            return find_corners_opencv(image)
        return None

    def _from_model(self, image, conf: float) -> Optional[CornerProposal]:
        import torch

        device = 0 if torch.cuda.is_available() else "cpu"
        source = image if isinstance(image, Image.Image) else Image.fromarray(_as_array(image))
        result = self.model(source, conf=conf, device=device, verbose=False)[0]
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None or result.boxes is None or len(result.boxes) == 0:
            return None
        scores = result.boxes.conf.tolist()
        best = max(range(len(scores)), key=scores.__getitem__)
        points = keypoints.xy[best].tolist()
        if len(points) != 4:
            return None
        return CornerProposal(order_corners(points), "model", float(scores[best]))
