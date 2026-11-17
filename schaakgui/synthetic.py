"""Synthetic board-corner dataset (Qt independent): perfect 8x8 boards warped into random photos.

The corner labels are exactly known, so no labelling is needed. Every image gets one YOLO-pose
label (class ``board``, 4 keypoints in image order TL, TR, BR, BL), see :mod:`schaakgui.keypoints`.
"""
from __future__ import annotations

import io
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .board import Corners, order_corners
from .keypoints import write_board_label, write_board_yaml

RGB = tuple[int, int, int]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class SyntheticConfig:
    count: int = 3000
    image_size: int = 640
    val_fraction: float = 0.15
    backgrounds_dir: Optional[str] = None
    max_rotation: float = 30.0  # degrees around the camera axis
    tilt: tuple[float, float] = (20.0, 70.0)  # camera angle above the horizon (degrees)
    occluders: bool = True
    shadows: bool = True
    seed: Optional[int] = None


def checkerboard(square_px: int = 48, border_px: int = 0, light: RGB = (235, 235, 230),
                 dark: RGB = (30, 30, 30), seams: bool = False,
                 border_color: RGB = (90, 90, 90)) -> Image.Image:
    """Flat 8x8 board (a1 dark, at the bottom-left), optionally with a border and tile seams.

    The returned image is the board *including* its border; the playing area starts at
    ``border_px``."""
    side = 8 * square_px + 2 * border_px
    image = Image.new("RGB", (side, side), border_color)
    draw = ImageDraw.Draw(image)
    for row in range(8):
        for col in range(8):
            rank = 7 - row
            color = dark if (col + rank) % 2 == 0 else light
            x0 = border_px + col * square_px
            y0 = border_px + row * square_px
            draw.rectangle((x0, y0, x0 + square_px - 1, y0 + square_px - 1), fill=color)
    if seams:
        seam = tuple(int(c * 0.6) for c in light)
        for i in range(9):
            t = border_px + i * square_px
            draw.line(((t, border_px), (t, side - border_px)), fill=seam, width=1)
            draw.line(((border_px, t), (side - border_px, t)), fill=seam, width=1)
    return image


def _project(points3d: np.ndarray, yaw: float, tilt: float, distance: float, focal: float,
             center: tuple[float, float]) -> np.ndarray:
    """Pinhole camera looking at a flat board (z = 0) from ``tilt`` degrees above the horizon."""
    a = math.radians(yaw)
    rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    t = math.radians(90 - tilt)  # 0 = straight down
    rx = np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]])
    pts = (rx @ (rz @ points3d.T)).T
    pts[:, 2] += distance
    return np.stack([focal * pts[:, 0] / pts[:, 2] + center[0],
                     focal * pts[:, 1] / pts[:, 2] + center[1]], axis=1)


def random_view(rng: random.Random, image_size: int = 640, tilt: tuple[float, float] = (20.0, 70.0),
                max_rotation: float = 30.0, margin: float = 0.03) -> Corners:
    """Camera-like corner positions (tilt, rotation, scale, position), all inside the image."""
    half = 0.5
    board = np.array([[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]], float)
    lo, hi = margin * image_size, (1 - margin) * image_size
    for _ in range(100):
        yaw = rng.uniform(-max_rotation, max_rotation)
        tlt = rng.uniform(*tilt)
        pts = _project(board, yaw, tlt, distance=rng.uniform(1.6, 3.0), focal=1.0, center=(0, 0))
        span = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))
        scale = rng.uniform(0.45, 0.92) * image_size / span
        pts = pts * scale
        width, height = np.ptp(pts[:, 0]), np.ptp(pts[:, 1])
        if width > hi - lo or height > hi - lo:
            continue
        ox = rng.uniform(lo - pts[:, 0].min(), hi - pts[:, 0].max())
        oy = rng.uniform(lo - pts[:, 1].min(), hi - pts[:, 1].max())
        pts = pts + [ox, oy]
        return order_corners([tuple(p) for p in pts.tolist()])
    s = image_size
    return ((0.2 * s, 0.2 * s), (0.8 * s, 0.2 * s), (0.8 * s, 0.8 * s), (0.2 * s, 0.8 * s))


def paste_board(background: Image.Image, board: Image.Image, corners: Corners) -> Image.Image:
    """Perspective-warp the flat ``board`` so its 4 corners land on ``corners`` (TL, TR, BR, BL)."""
    bg = np.asarray(background.convert("RGB")).copy()
    w, h = board.size
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    m = cv2.getPerspectiveTransform(src, np.float32(corners))
    size = (bg.shape[1], bg.shape[0])
    warped = cv2.warpPerspective(np.asarray(board.convert("RGB")), m, size, flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), m, size, flags=cv2.INTER_LINEAR)
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    out = bg * (1 - alpha) + warped * alpha
    return Image.fromarray(out.astype(np.uint8))


def add_occluders(image: Image.Image, corners: Corners, rng: random.Random,
                  count: Optional[int] = None) -> Image.Image:
    """Piece-like blobs standing on random squares (they cover lines and sometimes corners)."""
    out = image.copy()
    draw = ImageDraw.Draw(out)
    src = np.float32([[0, 0], [8, 0], [8, 8], [0, 8]])
    m = cv2.getPerspectiveTransform(src, np.float32(corners))
    side = max(math.dist(corners[0], corners[1]), math.dist(corners[1], corners[2])) / 8
    for _ in range(count if count is not None else rng.randint(0, 20)):
        col = rng.choice([0, 7]) if rng.random() < 0.2 else rng.randrange(8)
        row = rng.choice([0, 7]) if rng.random() < 0.2 else rng.randrange(8)
        x, y, wgt = m @ np.array([col + 0.5, row + 0.5, 1.0])
        x, y = x / wgt, y / wgt
        width = side * rng.uniform(0.35, 0.7)
        height = side * rng.uniform(0.6, 2.2)
        shade = rng.choice([rng.randint(10, 70), rng.randint(180, 250)])
        color = (shade, shade + rng.randint(-10, 10), shade + rng.randint(-10, 10))
        color = tuple(max(0, min(255, c)) for c in color)
        draw.ellipse((x - width / 2, y - height, x + width / 2, y + width / 4), fill=color)
        head = width * 0.45
        draw.ellipse((x - head, y - height - head, x + head, y - height + head), fill=color)
    return out


def add_lighting(image: Image.Image, rng: random.Random, shadows: bool = True) -> Image.Image:
    """Shadows, brightness/contrast, blur, noise and JPEG artefacts."""
    array = np.asarray(image.convert("RGB")).astype(np.float32)
    h, w = array.shape[:2]
    if shadows and rng.random() < 0.6:
        mask = Image.new("L", (w, h), 0)
        pts = [(rng.uniform(0, w), rng.uniform(0, h)) for _ in range(rng.randint(3, 6))]
        ImageDraw.Draw(mask).polygon(pts, fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(rng.uniform(5, 25)))
        strength = rng.uniform(0.25, 0.6)
        array *= 1 - strength * (np.asarray(mask, np.float32) / 255.0)[..., None]
    array = array * rng.uniform(0.6, 1.3) + rng.uniform(-30, 30)
    gradient = np.linspace(rng.uniform(0.8, 1.1), rng.uniform(0.9, 1.2), h, dtype=np.float32)
    array *= gradient[:, None, None]
    if rng.random() < 0.7:
        array += np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(2, 10), array.shape)
    out = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
    if rng.random() < 0.4:
        out = out.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.5)))
    if rng.random() < 0.5:
        buffer = io.BytesIO()
        out.save(buffer, format="JPEG", quality=rng.randint(35, 90))
        buffer.seek(0)
        out = Image.open(buffer).convert("RGB")
    return out


def load_backgrounds(folder: Optional[str | Path]) -> list[Path]:
    """Background photos (grass, paving, garden …) in a folder, recursively."""
    if not folder:
        return []
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"Backgrounds folder not found: {root}")
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def noise_background(rng: random.Random, size: int) -> Image.Image:
    """Procedural texture: coloured noise at a few scales (grass-, gravel- or paving-like)."""
    gen = np.random.default_rng(rng.randrange(2**32))
    base = np.array(rng.choice([(70, 120, 50), (110, 110, 105), (150, 130, 100), (60, 90, 60),
                                (170, 165, 155)]), np.float32)
    array = np.zeros((size, size, 3), np.float32) + base
    for cells, amp in ((4, 40), (16, 25), (64, 15), (size, 12)):
        noise = gen.normal(0, amp, (cells, cells, 3)).astype(np.float32)
        array += cv2.resize(noise, (size, size), interpolation=cv2.INTER_CUBIC)
    if rng.random() < 0.3:  # paving lines
        step = rng.randint(size // 10, size // 4)
        array[::step, :, :] *= 0.6
        array[:, ::step, :] *= 0.6
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def _background(rng: random.Random, size: int, backgrounds: list[Path]) -> Image.Image:
    if backgrounds:
        path = rng.choice(backgrounds)
        try:
            with Image.open(path) as raw:
                image = raw.convert("RGB")
        except OSError:
            return noise_background(rng, size)
        w, h = image.size
        crop = rng.uniform(0.5, 1.0) * min(w, h)
        x, y = rng.uniform(0, w - crop), rng.uniform(0, h - crop)
        image = image.crop((int(x), int(y), int(x + crop), int(y + crop))).resize((size, size))
        if rng.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image
    return noise_background(rng, size)


def make_sample(rng: random.Random, cfg: SyntheticConfig,
                backgrounds: list[Path]) -> tuple[Image.Image, Corners]:
    """One image + the exact outer corners of the playing area."""
    size = cfg.image_size
    light_v = rng.randint(190, 250)
    dark_v = rng.randint(10, 80)
    light = (light_v, light_v - rng.randint(0, 15), light_v - rng.randint(0, 25))
    dark = (dark_v, dark_v + rng.randint(0, 10), dark_v + rng.randint(0, 10))
    square = 48
    border = rng.choice([0, 0, rng.randint(2, 20)])
    board = checkerboard(square, border, light, dark, rng.random() < 0.3,
                         border_color=tuple(rng.randint(40, 200) for _ in range(3)))
    corners = random_view(rng, size, cfg.tilt, cfg.max_rotation)
    if border:
        # Warp the board with its border so the playing-area corners land on ``corners``.
        side = 8 * square
        src = np.float32([[border, border], [border + side, border], [border + side, border + side],
                          [border, border + side]])
        m = cv2.getPerspectiveTransform(src, np.float32(corners))
        full = board.size[0]
        outer = cv2.perspectiveTransform(np.float32([[[0, 0], [full, 0], [full, full], [0, full]]]), m)[0]
        image = paste_board(_background(rng, size, backgrounds), board, tuple(map(tuple, outer)))
    else:
        image = paste_board(_background(rng, size, backgrounds), board, corners)
    if cfg.occluders:
        image = add_occluders(image, corners, rng)
    image = add_lighting(image, rng, cfg.shadows)
    return image, corners


def generate_board_dataset(out_dir: str | Path, cfg: Optional[SyntheticConfig] = None,
                           progress: Optional[Callable[[int, int], None]] = None,
                           stop: Optional[Callable[[], bool]] = None) -> dict:
    """Write ``train/`` + ``valid/`` (images + YOLO-pose labels) and ``data.yaml``.

    Returns the counts, the split folders and whether the run was stopped."""
    cfg = cfg or SyntheticConfig()
    out = Path(out_dir)
    rng = random.Random(cfg.seed)
    backgrounds = load_backgrounds(cfg.backgrounds_dir)
    n_val = int(round(cfg.count * cfg.val_fraction)) if cfg.count > 1 else 0
    counts = {"train": 0, "valid": 0}
    stopped = False
    for i in range(cfg.count):
        if stop is not None and stop():
            stopped = True
            break
        split = "valid" if i < n_val else "train"
        image, corners = make_sample(rng, cfg, backgrounds)
        name = f"synthetic_{i:06d}"
        image_path = out / split / "images" / f"{name}.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(image_path, quality=92)
        write_board_label(out / split / "labels" / f"{name}.txt", corners, *image.size)
        counts[split] += 1
        if progress is not None:
            progress(i + 1, cfg.count)
    write_board_yaml(out, splits=("train", "valid"))
    folders = {split: [str((out / split).resolve())] for split, n in counts.items() if n}
    return {"out_dir": str(out.resolve()), "counts": counts, "folders": folders,
            "backgrounds": len(backgrounds), "stopped": stopped}
