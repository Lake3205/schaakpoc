import threading

import pytest
import yaml
from conftest import make_folder, make_yolo_split

from schaakgui.detector import (
    DetectionReviewSession,
    DetectorJob,
    DetectorTrainer,
    build_data_yaml,
    evaluate_detector,
    list_detectors,
    load_yolo_detector,
    match_boxes,
    model_name,
    read_run_info,
    rename_model,
    ultralytics_augmentation,
)
from schaakgui.projects import AugmentationConfig, DetectorConfig
from schaakgui.review_dialog import model_label


def test_rename_model_names_without_moving_files(tmp_path):
    root = tmp_path / "proj" / "checkpoints" / "detectors"
    trained = root / "detector_1" / "weights" / "best.pt"
    imported = root / "imported_x.pt"
    board = tmp_path / "proj" / "checkpoints" / "boards" / "board_1" / "weights" / "best.pt"
    for path in (trained, imported, board):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

    rename_model(trained, "  pieces   v2 ")
    rename_model(imported, "old one")
    rename_model(board, "garden board")
    assert model_name(trained) == "pieces v2" and model_name(imported) == "old one"
    assert model_name(board) == "garden board"
    assert model_label(str(trained)) == "pieces v2  (detector_1/weights/best.pt)"
    assert trained.is_file() and list_detectors(tmp_path / "proj")  # names file is not a model

    rename_model(trained, "")
    assert model_name(trained) is None and model_name(imported) == "old one"
    assert model_label(str(trained)) == "detector_1/weights/best.pt"
    outside = tmp_path / "loose.pt"
    outside.write_bytes(b"x")
    assert model_name(outside) is None
    with pytest.raises(ValueError):
        rename_model(outside, "nope")


def test_build_data_yaml_merges_yolo_splits(tmp_path):
    train = make_yolo_split(tmp_path / "a" / "train")
    other = make_yolo_split(tmp_path / "b" / "train")
    val = make_yolo_split(tmp_path / "a" / "valid")
    image_folder = make_folder(tmp_path / "plain", 1)
    config = build_data_yaml([str(train), str(other / "images"), str(image_folder)], [str(val)],
                             tmp_path / "out" / "data.yaml")
    assert config["train"] == [str((train / "images").resolve()), str((other / "images").resolve())]
    assert config["val"] == [str((val / "images").resolve())]
    assert config["names"] == {0: "black-king", 1: "white-king"}
    assert yaml.safe_load((tmp_path / "out" / "data.yaml").read_text(encoding="utf-8")) == config

    without_val = build_data_yaml([str(train)], [], tmp_path / "out2" / "data.yaml")
    assert without_val["val"] == without_val["train"]


def test_build_data_yaml_rejects_unsynced_names_and_no_boxes(tmp_path):
    a = make_yolo_split(tmp_path / "a" / "train")
    b = make_yolo_split(tmp_path / "b" / "train", names=("W-K", "black-king"))
    with pytest.raises(ValueError, match="Synchronize YAML"):
        build_data_yaml([str(a), str(b)], [], tmp_path / "data.yaml")
    with pytest.raises(ValueError, match="no boxes"):
        build_data_yaml([str(make_folder(tmp_path / "plain", 1))], [], tmp_path / "data.yaml")


def test_augmentation_maps_to_box_aware_ultralytics_settings():
    off = ultralytics_augmentation(AugmentationConfig(), mosaic=False)
    assert all(v == 0.0 for v in off.values())
    aug = AugmentationConfig(hflip=True, rotation=True, rotation_degrees=10, color_jitter=True,
                             brightness=0.3, saturation=1.5, affine=True, affine_degrees=20,
                             affine_translate=0.2, affine_shear=5, resized_crop=True,
                             crop_scale_min=0.6)
    on = ultralytics_augmentation(aug)
    assert on == {"fliplr": 0.5, "flipud": 0.0, "degrees": 20.0, "hsv_h": 0.015, "hsv_s": 1.0,
                  "hsv_v": 0.3, "translate": 0.2, "shear": 5.0, "scale": 0.4, "mosaic": 1.0}


def test_match_boxes_counts_correct_wrong_extra_and_missed():
    truths = [{"box": (0, 0, 10, 10), "label": "white-king"},
              {"box": (20, 0, 30, 10), "label": "black-king"},
              {"box": (40, 0, 50, 10), "label": "white-rook"}]
    predictions = [{"box": (1, 0, 10, 10), "label": "White_King"},
                   {"box": (20, 0, 30, 10), "label": "white-king"},
                   {"box": (60, 0, 70, 10), "label": "black-king"}]
    counts = match_boxes(predictions, truths)
    assert counts == {"correct": 1, "wrong_class": 1, "extra": 1, "missed": 1, "truths": 3,
                      "perfect": False}
    assert [t["status"] for t in truths] == ["found", "wrong_class", "missed"]


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """One tiny CPU training run, paused once after the first batch and resumed."""
    pytest.importorskip("ultralytics")
    tmp = tmp_path_factory.mktemp("det")
    train = make_yolo_split(tmp / "data" / "train")
    project_dir = tmp / ".projects" / "p"
    events = {"epochs": [], "payloads": [], "paused": [], "paused_loadable": None}
    trainer = None

    def on_paused(path):
        events["paused"].append(path)
        events["paused_loadable"] = load_yolo_detector(path).task
        trainer.resume()

    # A yaml base model is built from scratch, so the test needs no download.
    job = DetectorJob([str(train)], [], str(project_dir),
                      DetectorConfig(base_model="yolo26n.yaml", epochs=2, image_size=64, batch_size=3,
                                     patience=0),
                      device="cpu", plots=False, seed=1)
    trainer = DetectorTrainer(job, lambda m: None, lambda e, t: events["epochs"].append((e, t)),
                              evaluated=events["payloads"].append, paused=on_paused)
    trainer.pause()
    result = {}
    thread = threading.Thread(target=lambda: result.update(trainer.run()))
    thread.start()
    thread.join(timeout=300)
    assert not thread.is_alive()
    return {"summary": result, "events": events, "project_dir": project_dir, "split": train}


def test_detector_training_saves_into_project(trained):
    summary, events = trained["summary"], trained["events"]
    assert events["epochs"] == [(1, 2), (2, 2)]
    assert summary["stop_reason"] == "completed" and summary["epochs"] == 2
    assert summary["weights"] in list_detectors(trained["project_dir"])
    assert read_run_info(summary["weights"])["classes"] == ["black-king", "white-king"]
    assert [h["epoch"] for h in events["payloads"][-1]["history"]] == [1, 2]
    assert events["payloads"][-1]["classes"] == ["black-king", "white-king", "background"]


def test_pause_saves_loadable_review_weights_and_resumes(trained):
    events = trained["events"]
    assert len(events["paused"]) == 1 and events["paused_loadable"] == "detect"
    assert trained["summary"]["epochs"] == 2  # resumed after the pause
    assert trained["summary"]["weights"] not in events["paused"]


def test_final_exam_and_review_session(trained):
    weights, split = trained["summary"]["weights"], str(trained["split"])
    exam = evaluate_detector(weights, [split], image_size=64, batch_size=3, device="cpu", plots=False)
    assert exam["images"] == 3 and 0.0 <= exam["mAP50"] <= 1.0
    assert exam["classes"] == ["black-king", "white-king", "background"]
    assert len(exam["confusion"]) == 3

    session = DetectionReviewSession(weights, [split], seed=0, conf=0.01)
    first = session.next()
    assert first["counts"]["truths"] == 2 and len(first["truths"]) == 2
    assert first["pred_image"].size == first["image"].size
    session.next()
    assert session.previous()["path"] == first["path"]
