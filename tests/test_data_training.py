import pytest
import torch

from schaakgui.checkpoint import load_checkpoint, model_from_checkpoint, read_lineage
from schaakgui.data import (
    MultiFolderImageDataset,
    build_train_transform,
    class_counts,
    discover_classes,
    enabled_augmentations,
    read_yolo_labels,
    scan_folder,
)
from schaakgui.engine import Callbacks, Trainer, TrainJob, evaluate_checkpoint
from schaakgui.projects import AugmentationConfig, QuotaConfig, TrainingConfig


def test_multi_folder_dataset(datasets):
    folders = [datasets["a"], datasets["b"]]
    assert discover_classes(folders) == ["black-king", "white-king", "white-rook"]
    assert class_counts(folders) == {"black-king": 6, "white-king": 6, "white-rook": 4}
    ds = MultiFolderImageDataset(folders, ["black-king", "white-king"])
    assert len(ds) == 12 and ds.skipped_classes == {"white-rook"}
    train, val = ds.split(0.25, 0, None, None)
    assert len(train) == 9 and len(val) == 3


def test_augmentation_pipeline_composition():
    aug = AugmentationConfig(hflip=True, vflip=True, rotation=True, color_jitter=True,
                             affine=True, resized_crop=True)
    names = [type(t).__name__ for t in build_train_transform(aug, 32).transforms]
    assert names == ["RandomResizedCrop", "RandomHorizontalFlip", "RandomVerticalFlip",
                     "RandomRotation", "ColorJitter", "RandomAffine", "ToTensor", "Normalize"]
    plain = [type(t).__name__ for t in build_train_transform(AugmentationConfig(), 32).transforms]
    assert plain == ["Resize", "ToTensor", "Normalize"]
    assert len(enabled_augmentations(aug)) == 6


def _job(datasets, out, **kwargs):
    defaults = dict(
        train_folders=[str(datasets["a"]), str(datasets["b"])],
        val_folders=[str(datasets["val"])],
        output_dir=str(out),
        seed=0,
        augmentation=AugmentationConfig(hflip=True),
        training=TrainingConfig(epochs=2, batch_size=8, image_size=32),
        quota=QuotaConfig(snapshot_every=1, stop_on_quota=False, early_stopping=False),
    )
    defaults.update(kwargs)
    return TrainJob(**defaults)


def test_training_resume_and_lineage(datasets, tmp_path):
    events = {"batch": [], "epoch": [], "evaluated": []}
    callbacks = Callbacks(batch_progress=lambda i, n: events["batch"].append((i, n)),
                          epoch_progress=lambda e, n: events["epoch"].append((e, n)),
                          evaluated=events["evaluated"].append)
    summary = Trainer(_job(datasets, tmp_path / "run1"), callbacks).run()
    assert summary["stop_reason"] == "completed"
    assert summary["total_epochs_completed"] == 2
    assert events["epoch"] == [(1, 2), (2, 2)]
    assert events["batch"][-1] == (2, 2)
    payload = events["evaluated"][-1]
    assert len(payload["confusion"]) == 3 and len(payload["history"]) == 2

    ckpt = load_checkpoint(summary["last_checkpoint"])
    model = model_from_checkpoint(ckpt).eval()
    assert model(torch.zeros(1, 3, 32, 32)).shape == (1, 3)
    lineage = read_lineage(summary["last_checkpoint"])
    assert [s["epoch"] for s in lineage.snapshots] == [1, 2]
    assert lineage.best is not None

    resumed = Trainer(_job(datasets, tmp_path / "run2", train_folders=[str(datasets["a"])],
                           val_folders=[], resume_checkpoint=summary["last_checkpoint"],
                           training=TrainingConfig(epochs=1, batch_size=8))).run()
    lineage = read_lineage(resumed["last_checkpoint"])
    assert lineage.total_epochs_completed == 3
    assert len(lineage.sessions) == 2
    assert lineage.sessions[1]["start_epoch"] == 2 and lineage.sessions[1]["val_folders"] == []
    assert lineage.dataset_sources() == {"train": [str(datasets["a"]), str(datasets["b"])],
                                         "val": [str(datasets["val"])]}
    assert [h["epoch"] for h in lineage.history] == [1, 2, 3]


def test_quota_and_early_stopping(datasets, tmp_path):
    easy = QuotaConfig(min_accuracy=0, target_avg_accuracy=0, min_confidence=0,
                       target_avg_confidence=0, early_stopping=False, snapshot_every=0)
    summary = Trainer(_job(datasets, tmp_path / "q", quota=easy,
                           training=TrainingConfig(epochs=5, batch_size=8, image_size=32))).run()
    assert summary["stop_reason"] == "quota" and summary["epochs_this_session"] == 1
    assert len(summary["milestones"]) == 4

    patient = QuotaConfig(stop_on_quota=False, early_stopping=True, patience=1, monitor="val_acc",
                          snapshot_every=0)
    summary = Trainer(_job(datasets, tmp_path / "e", quota=patient,
                           training=TrainingConfig(epochs=30, batch_size=8, image_size=32,
                                                   learning_rate=0.0))).run()
    assert summary["stop_reason"] == "early_stopping" and summary["epochs_this_session"] == 2


def test_stop_request_and_unknown_classes(datasets, tmp_path):
    trainer = Trainer(_job(datasets, tmp_path / "s"))
    trainer.request_stop()
    summary = trainer.run()
    assert summary["stop_reason"] == "user" and summary["last_checkpoint"] is None

    first = Trainer(_job(datasets, tmp_path / "k", train_folders=[str(datasets["b"])],
                         training=TrainingConfig(epochs=1, batch_size=8, image_size=32))).run()
    with pytest.raises(ValueError, match="white-rook"):
        Trainer(_job(datasets, tmp_path / "k2", resume_checkpoint=first["last_checkpoint"])).run()


def test_yolo_labels_are_cropped_per_box(tmp_path):
    from conftest import make_yolo_split

    split = make_yolo_split(tmp_path / "chess" / "train")
    # The dataset root (train/valid/test splits) must not be read as classes "train", "valid", ...
    with pytest.raises(ValueError, match="YOLO dataset"):
        scan_folder(tmp_path / "chess")
    for folder in (split, split / "images"):
        assert class_counts([folder]) == {"black-king": 3, "white-king": 3}
    ds = MultiFolderImageDataset([split], ["black-king", "white-king"])
    assert sorted(ds.targets) == [0, 0, 0, 1, 1, 1]
    for i, (_, target, box) in enumerate(ds.samples):
        crop = ds.load_image(i)
        assert crop.size == (20, 20) and box is not None
        expected = (0, 0, 0) if target == 0 else (255, 255, 255)
        assert all(abs(a - b) < 40 for a, b in zip(crop.getpixel((10, 10)), expected))

    # Unknown names fall back to class_<id>; polygons are converted to boxes; bad lines skipped.
    label = tmp_path / "x.txt"
    label.write_text("3 0.1 0.2 0.5 0.2 0.5 0.6\nbad line here x y\n1 0.5 0.5 2 2\n", encoding="utf-8")
    assert read_yolo_labels(label) == [(3, (0.1, 0.2, 0.5, 0.6)), (1, (0.0, 0.0, 1.0, 1.0))]
    other = make_yolo_split(tmp_path / "noyaml" / "train", names=())
    assert set(scan_folder(other)) == {"class_0", "class_1"}


def test_yolo_training_and_final_exam(datasets, tmp_path):
    from conftest import make_yolo_split

    train = make_yolo_split(tmp_path / "yolo" / "train")
    val = make_yolo_split(tmp_path / "yolo" / "valid")
    summary = Trainer(_job(datasets, tmp_path / "y", train_folders=[str(train)],
                           val_folders=[str(val)],
                           training=TrainingConfig(epochs=1, batch_size=4, image_size=32))).run()
    ckpt = load_checkpoint(summary["last_checkpoint"])
    assert ckpt["classes"] == ["black-king", "white-king"]

    test = make_yolo_split(tmp_path / "yolo" / "test")
    exam = evaluate_checkpoint(summary["last_checkpoint"], [str(test)])
    assert exam["samples"] == 6 and len(exam["confusion"]) == 2
    assert 0.0 <= exam["accuracy"] <= 1.0 and exam["seen_in_training"] == []
    assert evaluate_checkpoint(summary["last_checkpoint"], [str(val)])["seen_in_training"] == [str(val)]
    with pytest.raises(ValueError):
        evaluate_checkpoint(summary["last_checkpoint"], [])


def test_test_folders_are_not_used_during_training(datasets, tmp_path):
    logs = []
    Trainer(_job(datasets, tmp_path / "t", val_folders=[]), Callbacks(log=logs.append)).run()
    assert any("holding out 20%" in line for line in logs)
    assert not any(str(datasets["test"]) in line for line in logs)
