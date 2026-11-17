"""YOLO-pose labels for the board corner model (Qt independent).

One label line per board: ``0 xc yc w h x1 y1 v1 … x4 y4 v4`` (normalised), keypoints in image
order TL, TR, BR, BL. Visibility ``2`` = inside the picture, ``0`` = outside (stored as 0, 0).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import yaml

from .board import Corners, Point

BOARD_KPT_SHAPE = [4, 3]
BOARD_FLIP_IDX = [1, 0, 3, 2]  # a left-right flip swaps TL<->TR and BR<->BL
BOARD_NAMES = {0: "board"}
SPLIT_KEYS = {"train": "train", "valid": "val", "val": "val", "test": "test"}


def _clip(v: float) -> float:
    return min(1.0, max(0.0, v))


def board_label_line(corners: Sequence[Point], width: int, height: int) -> str:
    """``0 xc yc w h x1 y1 v …`` for the 4 corners of one board in a ``width`` x ``height`` image."""
    xs = [x / width for x, _ in corners]
    ys = [y / height for _, y in corners]
    x1, x2 = _clip(min(xs)), _clip(max(xs))
    y1, y2 = _clip(min(ys)), _clip(max(ys))
    values = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
    for x, y in zip(xs, ys):
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            values += [x, y, 2]
        else:
            values += [0.0, 0.0, 0]
    return "0 " + " ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in values)


def write_board_label(path: str | Path, corners: Sequence[Point], width: int, height: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(board_label_line(corners, width, height) + "\n", encoding="utf-8")


def read_board_label(path: str | Path, width: int, height: int) -> Optional[Corners]:
    """Pixel corners of the first board in a pose label file; ``None`` if missing or a corner is
    not visible."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5 + 4 * 3:
            continue
        try:
            values = [float(v) for v in parts[5:5 + 12]]
        except ValueError:
            continue
        corners = []
        for i in range(4):
            x, y, v = values[3 * i:3 * i + 3]
            if v <= 0:
                return None
            corners.append((x * width, y * height))
        return tuple(corners)  # type: ignore[return-value]
    return None


def write_board_yaml(dataset_root: str | Path, splits: Sequence[str] = ("train", "valid", "test")) -> Path:
    """``data.yaml`` for a board dataset: ``names: {0: board}``, ``kpt_shape`` and ``flip_idx``.

    Only splits whose ``images`` folder exists are listed."""
    root = Path(dataset_root)
    root.mkdir(parents=True, exist_ok=True)
    data: dict = {"path": str(root.resolve())}
    for split in splits:
        if (root / split / "images").is_dir():
            data.setdefault(SPLIT_KEYS.get(split, split), f"{split}/images")
    data["names"] = dict(BOARD_NAMES)
    data["kpt_shape"] = list(BOARD_KPT_SHAPE)
    data["flip_idx"] = list(BOARD_FLIP_IDX)
    path = root / "data.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def write_detect_yaml(dataset_root: str | Path, names: Sequence[str],
                      splits: Sequence[str] = ("train", "valid", "test")) -> Path:
    """``data.yaml`` for a box (detection) dataset with the given class names."""
    root = Path(dataset_root)
    root.mkdir(parents=True, exist_ok=True)
    data: dict = {"path": str(root.resolve())}
    for split in splits:
        if (root / split / "images").is_dir():
            data.setdefault(SPLIT_KEYS.get(split, split), f"{split}/images")
    data["nc"] = len(names)
    data["names"] = list(names)
    path = root / "data.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path
