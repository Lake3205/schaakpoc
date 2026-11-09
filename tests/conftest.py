import os
import random
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

COLORS = {"black-king": (10, 10, 10), "white-king": (245, 245, 245), "white-rook": (200, 30, 30)}


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
        "test": make_folder(tmp_path / "test", 2),
    }
