import random

import numpy as np

from schaakgui.board import warp_board
from schaakgui.data import yolo_kpt_shape
from schaakgui.keypoints import read_board_label
from schaakgui.position import corner_square_is_dark
from schaakgui.synthetic import SyntheticConfig, generate_board_dataset, make_sample


def test_make_sample_corners_match_the_warped_board():
    rng = random.Random(3)
    cfg = SyntheticConfig(image_size=320, occluders=False, shadows=False)
    for _ in range(4):
        image, corners = make_sample(rng, cfg, [])
        straight = np.asarray(warp_board(image, corners, 400).convert("L"), dtype=float)
        # Straightened on the labeled corners the checker pattern is aligned with the grid:
        # the colour alternates between neighbouring squares in every row.
        centres = [[straight[r * 50 + 25, c * 50 + 25] for c in range(8)] for r in range(8)]
        diffs = [abs(row[c] - row[c + 1]) for row in centres for c in range(7)]
        assert np.median(diffs) > 60
        dark_corners = [corner_square_is_dark(straight, c) for c in range(4)]
        assert dark_corners in ([True, False, True, False], [False, True, False, True])


def test_generate_board_dataset(tmp_path):
    seen = []
    result = generate_board_dataset(tmp_path / "boards", SyntheticConfig(count=6, image_size=160,
                                                                          val_fraction=0.34, seed=1),
                                    progress=lambda i, n: seen.append((i, n)))
    assert result["counts"] == {"train": 4, "valid": 2}
    assert seen[-1] == (6, 6) and not result["stopped"]
    root = tmp_path / "boards"
    assert yolo_kpt_shape(root / "train") == [4, 3]
    for split in ("train", "valid"):
        images = sorted((root / split / "images").glob("*.jpg"))
        assert len(images) == result["counts"][split]
        for image in images:
            corners = read_board_label(root / split / "labels" / f"{image.stem}.txt", 160, 160)
            assert corners is not None


def test_generate_board_dataset_can_stop(tmp_path):
    result = generate_board_dataset(tmp_path / "b", SyntheticConfig(count=10, image_size=96, seed=1),
                                    stop=lambda: True)
    assert result["stopped"] and result["counts"] == {"train": 0, "valid": 0}
