"""Shared helpers for the scanner tests: a rendered garden board photo and a fake piece detector."""
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image, ImageDraw

from schaakgui.board import board_homography, to_image
from schaakgui.position import SYMBOL_NAMES, cell_of
from schaakgui.synthetic import checkerboard

# Perspective view of the board: the far side (top) is narrower.
PHOTO_SIZE = (900, 700)
PHOTO_CORNERS = ((260.0, 150.0), (640.0, 150.0), (820.0, 620.0), (80.0, 620.0))
POSITION = {"e1": "K", "d1": "Q", "a2": "P", "h2": "P", "e8": "k", "a7": "p", "c6": "n"}


def board_photo(corners=PHOTO_CORNERS, size=PHOTO_SIZE, a1_corner=3) -> Image.Image:
    """A grass-coloured photo with an empty 8x8 board warped onto ``corners`` (a1 at ``a1_corner``)."""
    board = checkerboard(64)
    # checkerboard() has a1 at the bottom-left (corner 3); rotate it clockwise to put a1 elsewhere.
    board = board.rotate(-90 * ((a1_corner - 3) % 4))
    side = board.size[0]
    src = np.float32([[0, 0], [side, 0], [side, side], [0, side]])
    m = cv2.getPerspectiveTransform(src, np.float32(corners))
    photo = np.full((size[1], size[0], 3), (70, 140, 60), np.uint8)
    warped = cv2.warpPerspective(np.asarray(board), m, size)
    mask = cv2.warpPerspective(np.full((side, side), 255, np.uint8), m, size)
    photo[mask > 0] = warped[mask > 0]
    return Image.fromarray(photo)


def piece_boxes(position=POSITION, corners=PHOTO_CORNERS, a1_corner=3, height=70.0):
    """Upright boxes whose foot point (with the default lift) lands in the centre of each square."""
    h = board_homography(corners)
    boxes = []
    for square, symbol in position.items():
        col, row = cell_of(square, a1_corner)
        fx, fy = to_image(h, ((col + 0.5) * 100, (row + 0.5) * 100))
        y2 = fy + 0.08 * height  # foot = y2 - lift * height
        boxes.append({"box": (fx - 18, y2 - height, fx + 18, y2), "label": SYMBOL_NAMES[symbol],
                      "conf": 0.9})
    return boxes


def draw_pieces(photo: Image.Image, boxes) -> Image.Image:
    photo = photo.copy()
    draw = ImageDraw.Draw(photo)
    for d in boxes:
        color = (250, 250, 250) if d["label"].startswith("white") else (15, 15, 15)
        draw.ellipse(d["box"], fill=color, outline=(200, 0, 0))
    return photo


class FakeTester:
    """Stands in for ``ImageTester``: returns fixed boxes and counts the (slow) detector calls."""

    def __init__(self, boxes):
        self.boxes = boxes
        self.detector = SimpleNamespace(names={})
        self.calls = 0

    def detect(self, image, conf=0.25):
        self.calls += 1
        return [{"box": d["box"], "detector": (d["label"], d["conf"])} for d in self.boxes
                if d["conf"] >= conf]


class NoFinder:
    def propose(self, image, conf=0.25):
        return None
