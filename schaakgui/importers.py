"""Public data importers (Qt independent): ChessReD2K → YOLO piece boxes + YOLO-pose board corners.

ChessReD (Masouris & van Gemert, 4TU.ResearchData) ships one ``annotations.json`` with
``images``, ``categories``, ``annotations.pieces`` (``bbox`` = ``[x, y, w, h]`` in pixels, only for
the 2K subset), ``annotations.corners`` (4 White-relative board corners, 2K subset) and ``splits``.
Only images with both piece boxes and corners are imported; ChessReD's own train/val/test split is
kept. The images themselves must be downloaded by the user.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from .board import Corners, order_corners
from .keypoints import write_board_label, write_board_yaml, write_detect_yaml
from .position import PIECE_CLASSES, PIECE_SYMBOLS, piece_symbol

# ChessReD category id → unified name (used when the json has no category names).
CHESSRED_NAMES = {
    0: "white-pawn", 1: "white-rook", 2: "white-knight", 3: "white-bishop", 4: "white-queen",
    5: "white-king", 6: "black-pawn", 7: "black-rook", 8: "black-knight", 9: "black-bishop",
    10: "black-queen", 11: "black-king",
}
SPLIT_NAMES = {"train": "train", "val": "valid", "valid": "valid", "validation": "valid", "test": "test"}
_CORNER_KEYS = ("top_left", "top_right", "bottom_right", "bottom_left")


def load_chessred(annotation_json: str | Path) -> dict:
    """Read the annotation file once."""
    path = Path(annotation_json)
    if not path.is_file():
        raise FileNotFoundError(f"ChessReD annotation file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "images" not in data or "annotations" not in data:
        raise ValueError(f"{path.name} is not a ChessReD annotations.json (no images/annotations)")
    return data


def category_names(data: dict) -> dict[int, Optional[str]]:
    """Category id → unified class name (``None`` for ``empty`` or unknown categories)."""
    names: dict[int, Optional[str]] = {}
    for category in data.get("categories", []) or []:
        symbol = piece_symbol(str(category.get("name", "")))
        names[int(category["id"])] = next((n for n, s in PIECE_SYMBOLS.items() if s == symbol), None) \
            if symbol else None
    if not names:
        names = dict(CHESSRED_NAMES)
    return names


def chessred_corners(annotation: dict) -> Corners:
    """White-relative corners (dict with ``top_left`` … or a list of 4 points) → image order."""
    corners = annotation.get("corners", annotation)
    if isinstance(corners, dict):
        points = [corners[k] for k in _CORNER_KEYS if k in corners]
    else:
        points = list(corners)
    if len(points) != 4:
        raise ValueError(f"Expected 4 corners, got {len(points)}")
    return order_corners([(float(p[0]), float(p[1])) for p in points])


def _annotation_lists(data: dict) -> tuple[list, list]:
    ann = data.get("annotations", {})
    if isinstance(ann, dict):
        return list(ann.get("pieces", []) or []), list(ann.get("corners", []) or [])
    return list(ann), []


def _image_splits(data: dict) -> dict[int, str]:
    """Image id → ``train``/``valid``/``test``, preferring the ``chessred2k`` split."""
    splits = data.get("splits", {}) or {}
    source = splits.get("chessred2k") if isinstance(splits.get("chessred2k"), dict) else splits
    result: dict[int, str] = {}
    for key, value in (source or {}).items():
        split = SPLIT_NAMES.get(str(key).lower())
        if split is None or not isinstance(value, dict):
            continue
        for image_id in value.get("image_ids", []) or []:
            result[int(image_id)] = split
    return result


def _image_path(images_root: Path, image: dict) -> Optional[Path]:
    candidates = []
    for key in ("path", "file_name"):
        value = image.get(key)
        if value:
            rel = Path(str(value))
            candidates += [images_root / rel, images_root / rel.name]
            parts = rel.parts
            for i in range(1, len(parts)):
                candidates.append(images_root.joinpath(*parts[i:]))
    for path in candidates:
        if path.is_file():
            return path
    return None


def import_chessred2k(annotation_json: str | Path, images_root: str | Path, out_dir: str | Path,
                      progress: Optional[Callable[[int, int], None]] = None,
                      stop: Optional[Callable[[], bool]] = None, copy_images: bool = True) -> dict:
    """Write ``out_dir/pieces/{train,valid,test}`` (detect, 12 classes) and
    ``out_dir/board/{train,valid,test}`` (pose, 4 corners) from ChessReD2K.

    Images without piece boxes or without corners are skipped; missing image files are counted.
    Returns counts per split, the created split folders and skip reasons."""
    data = load_chessred(annotation_json)
    images_root = Path(images_root)
    out = Path(out_dir)
    names = category_names(data)
    class_ids = {name: i for i, name in enumerate(PIECE_CLASSES)}
    pieces, corners = _annotation_lists(data)
    boxes_by_image: dict[int, list] = {}
    for piece in pieces:
        if piece.get("bbox"):
            boxes_by_image.setdefault(int(piece["image_id"]), []).append(piece)
    corners_by_image = {int(c["image_id"]): c for c in corners if c.get("corners") or c.get("top_left")}
    splits = _image_splits(data)
    selected = [img for img in data["images"]
                if int(img["id"]) in boxes_by_image and int(img["id"]) in corners_by_image]
    counts = {"train": 0, "valid": 0, "test": 0}
    skipped = {"no_boxes_or_corners": len(data["images"]) - len(selected), "missing_file": 0,
               "no_split": 0}
    stopped = False
    for i, image in enumerate(selected):
        if stop is not None and stop():
            stopped = True
            break
        image_id = int(image["id"])
        split = splits.get(image_id) if splits else "train"
        if split is None:
            skipped["no_split"] += 1
            continue
        source = _image_path(images_root, image)
        if source is None:
            skipped["missing_file"] += 1
            continue
        width, height = image.get("width"), image.get("height")
        if not width or not height:
            with Image.open(source) as raw:
                width, height = raw.size
        stem = f"chessred_{image_id}_{source.stem}"
        lines = []
        for piece in boxes_by_image[image_id]:
            name = names.get(int(piece["category_id"]))
            if name is None:
                continue
            x, y, w, h = (float(v) for v in piece["bbox"])
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(float(width), x + w), min(float(height), y + h)
            if x2 <= x1 or y2 <= y1:
                continue
            lines.append(f"{class_ids[name]} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
                         f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")
        for kind in ("pieces", "board"):
            target = out / kind / split / "images" / f"{stem}{source.suffix.lower()}"
            target.parent.mkdir(parents=True, exist_ok=True)
            if copy_images:
                shutil.copy2(source, target)
            else:
                target.unlink(missing_ok=True)
                target.symlink_to(source.resolve())
        label = out / "pieces" / split / "labels" / f"{stem}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
        write_board_label(out / "board" / split / "labels" / f"{stem}.txt",
                          chessred_corners(corners_by_image[image_id]), width, height)
        counts[split] += 1
        if progress is not None:
            progress(i + 1, len(selected))
    if any(counts.values()):
        write_detect_yaml(out / "pieces", PIECE_CLASSES)
        write_board_yaml(out / "board")
    folders = {
        kind: {split: str((out / kind / split).resolve()) for split, n in counts.items() if n}
        for kind in ("pieces", "board")
    }
    return {"out_dir": str(out.resolve()), "counts": counts, "skipped": skipped,
            "folders": folders, "stopped": stopped}


def chessred_test_cases(annotation_json: str | Path, images_root: str | Path,
                        split: str = "test") -> list[dict]:
    """``{"image": path, "fen": placement}`` per (found) image of a split for ``evaluate_scans``."""
    data = load_chessred(annotation_json)
    images_root = Path(images_root)
    names = category_names(data)
    pieces, _ = _annotation_lists(data)
    splits = _image_splits(data)
    wanted = SPLIT_NAMES.get(split, split)
    squares: dict[int, dict[str, str]] = {}
    for piece in pieces:
        name = names.get(int(piece["category_id"]))
        square = piece.get("chessboard_position")
        if name and square:
            squares.setdefault(int(piece["image_id"]), {})[str(square).lower()] = PIECE_SYMBOLS[name]
    from .fen import placement_fen

    cases = []
    for image in data["images"]:
        image_id = int(image["id"])
        if splits and splits.get(image_id) != wanted:
            continue
        path = _image_path(images_root, image)
        if path is None or image_id not in squares:
            continue
        cases.append({"image": str(path), "fen": placement_fen(squares[image_id])})
    return cases
