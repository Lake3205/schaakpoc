import argparse
import subprocess
import sys
from pathlib import Path

import pytest

import train


@pytest.mark.parametrize("yolo, size, expected", [
    ("yolo11", "n", "yolo11n.pt"),
    ("yolo26", "n", "yolo26n.pt"),
    ("yolo26", "x", "yolo26x.pt"),
    ("yolov8", "s", "yolov8s.pt"),
])
def test_base_model_switches_between_yolo_versions(yolo, size, expected):
    assert train.base_model(yolo, size) == expected


def test_base_model_defaults_and_custom_override():
    assert train.base_model() == "yolo11n.pt"
    assert "yolo26" in train.YOLO_VERSIONS
    assert train.base_model("yolo26", "s", "my_weights.pt") == "my_weights.pt"


def test_base_model_rejects_unknown_values():
    with pytest.raises(ValueError):
        train.base_model("yolo99", "n")
    with pytest.raises(ValueError):
        train.base_model("yolo26", "q")


def test_cli_lists_yolo26_option():
    root = Path(__file__).resolve().parents[1]
    out = subprocess.run([sys.executable, str(root / "train.py"), "--help"],
                         capture_output=True, text=True, check=True).stdout
    assert "yolo26" in out and "--size" in out


def test_train_kwargs_use_camera_angle_augmentations(tmp_path):
    args = argparse.Namespace(data=Path("merged.yaml"), epochs=1, imgsz=640, batch=2, workers=0,
                              perspective=0.0005, degrees=180.0, scale=0.5,
                              project=tmp_path / "runs", name="exp1")
    kwargs = train.train_kwargs(args, "cpu")
    assert kwargs["perspective"] == 0.0005 and kwargs["degrees"] == 180 and kwargs["scale"] == 0.5
    assert kwargs["flipud"] == 0.0
    assert kwargs["data"] == "merged.yaml"
    assert Path(kwargs["project"]) == (tmp_path / "runs").resolve() and kwargs["name"] == "exp1"
    assert train.AUGMENTATION == {"perspective": 0.0005, "degrees": 180.0, "scale": 0.5}
