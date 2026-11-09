import os
import random
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

COLORS = {"black-king": (10, 10, 10), "white-king": (245, 245, 245), "white-rook": (200, 30, 30)}


def make_yolo_split(root: Path, names=("black-king", "white-king"), yaml_in_parent=True) -> Path:
    """YOLO split ``root/images`` + ``root/labels``: each 40x20 image has a black box (left)
    and a white box (right), labeled with class ids 0 and 1."""
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    for i in range(3):
        image = Image.new("RGB", (40, 20), (128, 128, 128))
        image.paste((0, 0, 0), (0, 0, 20, 20))
        image.paste((255, 255, 255), (20, 0, 40, 20))
        image.save(root / "images" / f"board{i}.jpg")
        (root / "labels" / f"board{i}.txt").write_text(
            "0 0.25 0.5 0.5 1.0\n1 0.75 0.5 0.5 1.0\n", encoding="utf-8")
    yaml_dir = root.parent if yaml_in_parent else root
    lines = [f"  {i}: {n}" for i, n in enumerate(names)]
    (yaml_dir / "data.yaml").write_text("train: train/images\nnames:\n" + "\n".join(lines) + "\n",
                                        encoding="utf-8")
    return root


def make_folder(root: Path, per_class: int = 4, classes=COLORS) -> Path:
    rng = random.Random(0)
    for name in classes:
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            color = tuple(max(0, min(255, c + rng.randint(-15, 15))) for c in COLORS[name])
            Image.new("RGB", (24, 24), color).save(d / f"{i}.png")
    return root


@pytest.fixture
def datasets(tmp_path):
    return {
        "a": make_folder(tmp_path / "a", 4),
        "b": make_folder(tmp_path / "b", 2, ["black-king", "white-king"]),
        "val": make_folder(tmp_path / "val", 2),
        "test": make_folder(tmp_path / "test", 2),
    }
