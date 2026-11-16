"""YOLO object detection (bounding boxes, many objects per image) for a project (Qt independent).

Detectors live in ``<project>/checkpoints/detectors/``: in-app trainings in
``detector_<stamp>/weights/best.pt`` (with ``schaakgui_run.json`` provenance) and imported weights as
``<name>.pt``. They are kept apart from the classifier checkpoints (a different checkpoint format).
"""
from __future__ import annotations

import json
import random
import shutil
import tempfile
import threading
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .checkpoint import load_checkpoint
from .data import is_image, read_yolo_labels, yolo_class_names, yolo_dirs
from .projects import AugmentationConfig, DetectorConfig

DETECTOR_BASE_MODELS = tuple(f"{family}{size}.pt" for family in ("yolo26", "yolo11")
                             for size in "nsmlx")
RUN_INFO = "schaakgui_run.json"
METRIC_KEYS = {"precision": "metrics/precision(B)", "recall": "metrics/recall(B)",
               "mAP50": "metrics/mAP50(B)", "mAP50_95": "metrics/mAP50-95(B)"}
AGREE_RGB = (46, 125, 50)
DISAGREE_RGB = (230, 81, 0)
TRUTH_RGB = (21, 101, 192)


def load_picture(path: str | Path) -> Image.Image:
    with Image.open(path) as raw:
        # Phone photos are stored sideways with an EXIF rotation flag.
        return ImageOps.exif_transpose(raw).convert("RGB")


def same_class(a: str, b: str) -> bool:
    """Class names match ignoring case and ``-``/``_``/space (``White-King`` == ``white_king``)."""
    def norm(s: str) -> str:
        return s.lower().replace("_", "-").replace(" ", "-")
    return norm(a) == norm(b)


def draw_detections(image: Image.Image, detections: list[dict]) -> Image.Image:
    """Copy of ``image`` with a numbered, labeled box per detection.

    Each detection has ``box`` (pixels), ``label``, optional ``conf`` and either ``color`` or
    ``agree`` (green / orange)."""
    out = image.copy()
    draw = ImageDraw.Draw(out)
    side = max(out.size)
    width = max(2, round(side / 400))
    font = ImageFont.load_default(size=max(11, round(side / 60)))
    for n, d in enumerate(detections, 1):
        color = d.get("color") or (AGREE_RGB if d.get("agree", True) else DISAGREE_RGB)
        x1, y1, x2, y2 = d["box"]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
        text = f"{n} {d['label']}" + (f" {d['conf'] * 100:.0f}%" if d.get("conf") is not None else "")
        _, _, tw, th = draw.textbbox((0, 0), text, font=font)
        h = th + 2 * width
        top = y1 - h if y1 - h >= 0 else y1
        draw.rectangle((x1, top, x1 + tw + 2 * width, top + h), fill=color)
        draw.text((x1 + width, top + width), text, fill=(255, 255, 255), font=font)
    return out


def detectors_dir(project_dir: str | Path) -> Path:
    return Path(project_dir) / "checkpoints" / "detectors"


def list_detectors(project_dir: str | Path) -> list[str]:
    """Detector weights of a project, newest first (``best.pt`` of trainings + imported files)."""
    root = detectors_dir(project_dir)
    if not root.is_dir():
        return []
    found = [p for p in root.glob("*/weights/best.pt")] + [p for p in root.glob("*.pt")]
    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p.resolve()) for p in found]


def load_yolo_detector(path: str | Path):
    """Load YOLO detector weights; clear errors for classifier checkpoints and non-detect models."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Detector not found: {path}")
    try:
        load_checkpoint(path)
    except Exception:
        pass
    else:
        raise ValueError(f"{path.name} is a classifier checkpoint of this app (one label per picture, "
                         "no boxes). Choose it as the classifier instead.")
    from ultralytics import YOLO

    try:
        detector = YOLO(str(path))
    except Exception as e:
        raise ValueError(f"{path.name} is not a YOLO model ({e})") from e
    if detector.task != "detect":
        raise ValueError(f"{path.name} is a YOLO '{detector.task}' model, not a detector")
    return detector


def import_detector(project_dir: str | Path, source: str | Path) -> str:
    """Validate detector weights and copy them into the project; returns the project path."""
    source = Path(source).resolve()
    root = detectors_dir(project_dir).resolve()
    if source.is_relative_to(root):
        load_yolo_detector(source)
        return str(source)
    load_yolo_detector(source)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = source.stem if source.parent.name != "weights" else f"{source.parent.parent.name}_{source.stem}"
    target = root / f"imported_{stem}_{stamp}.pt"
    shutil.copy2(source, target)
    target.touch()  # newest detector = the one just imported
    return str(target.resolve())


def _split_names(images_dir: Path) -> dict[int, str]:
    return yolo_class_names(images_dir.parent)


def build_data_yaml(train_folders: Sequence[str], val_folders: Sequence[str],
                    out_path: str | Path) -> dict:
    """Write an Ultralytics dataset yaml for the YOLO splits among the folders.

    ImageFolder datasets have no boxes and are skipped. All splits must share the same class
    names (use *Synchronize YAML class names* first). Without YOLO validation folders the
    training splits are also used for validation. Returns the written config.
    """
    train = [d[0] for d in (yolo_dirs(f) for f in train_folders) if d]
    val = [d[0] for d in (yolo_dirs(f) for f in val_folders) if d]
    if not train:
        raise ValueError("None of the checked training folders is a YOLO split "
                         "(images/ + labels/), so there are no boxes to train a detector on.")
    names: dict[int, str] = {}
    for images in train + val:
        split = _split_names(images)
        if not split:
            raise ValueError(f"No .yaml with class names found for {images.parent}")
        if names and split != names:
            raise ValueError("The YOLO folders use different class names. Use "
                             "'Synchronize YAML class names' first.\n"
                             f"{images.parent}: {list(split.values())}\nother: {list(names.values())}")
        names = split
    config = {
        "train": [str(p.resolve()) for p in train],
        "val": [str(p.resolve()) for p in (val or train)],
        "names": {i: names.get(i, f"class_{i}") for i in range(max(names) + 1)},
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return config




def ultralytics_augmentation(aug: AugmentationConfig, mosaic: bool = True) -> dict:
    """Box-aware Ultralytics equivalents of the Augmentation tab (unchecked = off).

    Brightness and saturation map to ``hsv_v``/``hsv_s`` (contrast has no equivalent), the resized crop
    minimum scale to ``scale`` and the affine settings to ``translate``/``shear``/``degrees``."""
    degrees = max(aug.rotation_degrees if aug.rotation else 0.0,
                  aug.affine_degrees if aug.affine else 0.0)
    return {
        "fliplr": 0.5 if aug.hflip else 0.0,
        "flipud": 0.5 if aug.vflip else 0.0,
        "degrees": float(degrees),
        "hsv_h": 0.015 if aug.color_jitter else 0.0,
        "hsv_s": min(1.0, float(aug.saturation)) if aug.color_jitter else 0.0,
        "hsv_v": min(1.0, float(aug.brightness)) if aug.color_jitter else 0.0,
        "translate": float(aug.affine_translate) if aug.affine else 0.0,
        "shear": float(aug.affine_shear) if aug.affine else 0.0,
        "scale": round(1.0 - float(aug.crop_scale_min), 3) if aug.resized_crop else 0.0,
        "mosaic": 1.0 if mosaic else 0.0,
    }


def detection_metrics(raw: Optional[dict]) -> dict:
    raw = raw or {}
    return {key: float(raw.get(name, 0.0)) for key, name in METRIC_KEYS.items()}


def confusion_payload(matrix, names: dict) -> tuple[list[list[int]], list[str]]:
    """Ultralytics' ``[predicted][true]`` matrix as ``[true][predicted]`` with a background class."""
    classes = [str(names[i]) for i in range(len(names))] + ["background"]
    m = np.asarray(matrix, dtype=float).T
    return np.rint(m).astype(int).tolist(), classes


def confidence_curves(box) -> dict:
    """Mean precision/recall over classes vs. confidence threshold (101 points)."""
    px, p, r = (getattr(box, k, None) for k in ("px", "p_curve", "r_curve"))
    if px is None or p is None or r is None or len(p) == 0:
        return {}
    idx = np.linspace(0, len(px) - 1, 101).astype(int)
    return {"thresholds": np.asarray(px)[idx].tolist(),
            "precision": np.asarray(p).mean(0)[idx].tolist(),
            "recall": np.asarray(r).mean(0)[idx].tolist()}


def save_review_weights(trainer, path: str | Path) -> str:
    """Save the current (EMA) weights mid-training as a loadable YOLO checkpoint."""
    import torch

    model = deepcopy(trainer.ema.ema if trainer.ema is not None else trainer.model).half()
    path = Path(path)
    torch.save({"model": model, "train_args": dict(vars(trainer.args)),
                "date": datetime.now().isoformat(timespec="seconds")}, path)
    return str(path.resolve())


def read_run_info(weights: str | Path) -> Optional[dict]:
    """Provenance of an in-app training (``<run>/schaakgui_run.json``) for its weights, if any."""
    weights = Path(weights)
    for directory in (weights.parent, weights.parent.parent):
        path = directory / RUN_INFO
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
    return None


@dataclass
class DetectorJob:
    train_folders: list[str]
    val_folders: list[str]
    project_dir: str
    config: DetectorConfig
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    seed: Optional[int] = None
    device: Optional[str | int] = None  # None = GPU 0 when available, else CPU
    plots: bool = True  # also needed for the live confusion matrix


class DetectorTrainer:
    """Trains an Ultralytics YOLO detector into ``<project>/checkpoints/detectors/detector_<stamp>``.

    Supports stop (after the current batch) and pause/resume between batches; while paused the
    current weights are saved as ``weights/paused.pt`` for Manual Review."""

    def __init__(self, job: DetectorJob, log: Callable[[str], None] = print,
                 epoch_progress: Callable[[int, int], None] = lambda e, t: None,
                 batch_progress: Callable[[int, int], None] = lambda i, t: None,
                 evaluated: Callable[[dict], None] = lambda payload: None,
                 paused: Callable[[str], None] = lambda path: None):
        self.job = job
        self.log = log
        self.epoch_progress = epoch_progress
        self.batch_progress = batch_progress
        self.evaluated = evaluated
        self.paused = paused
        self._stop = threading.Event()
        self._running = threading.Event()
        self._running.set()

    def request_stop(self):
        self._stop.set()
        self._running.set()

    def pause(self):
        self._running.clear()

    def resume(self):
        self._running.set()

    @property
    def is_paused(self) -> bool:
        return not self._running.is_set()

    def _wait_if_paused(self, trainer):
        if self._running.is_set():
            return
        path = save_review_weights(trainer, Path(trainer.wdir) / "paused.pt")
        self.log(f"Paused at epoch {trainer.epoch + 1}; current weights: {path}")
        self.paused(path)
        self._running.wait()
        if not self._stop.is_set():
            self.log("Resumed.")

    def run(self) -> dict:
        import torch
        from ultralytics import YOLO

        cfg = self.job.config
        if cfg.resume_from and not Path(cfg.resume_from).is_file():
            raise FileNotFoundError(f"Detector to continue from not found: {cfg.resume_from}")
        base = cfg.resume_from or cfg.base_model
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = detectors_dir(self.job.project_dir).resolve()
        name = f"detector_{stamp}"
        data = build_data_yaml(self.job.train_folders, self.job.val_folders,
                               root / name / "data.yaml")
        if data["val"] == data["train"]:
            self.log("No YOLO validation folders checked: validating on the training folders.")
        skipped = [f for f in [*self.job.train_folders, *self.job.val_folders] if yolo_dirs(f) is None]
        if skipped:
            self.log("Skipped (no YOLO images/ + labels/, so no boxes): " + ", ".join(skipped))
        self.log(f"Detector classes: {', '.join(data['names'].values())}")
        augmentation = ultralytics_augmentation(self.job.augmentation, cfg.mosaic)
        self.log("Augmentation: " + ", ".join(f"{k}={v}" for k, v in augmentation.items()))
        device = self.job.device
        if device is None:
            device = 0 if torch.cuda.is_available() else "cpu"

        model = load_yolo_detector(base) if cfg.resume_from else YOLO(base)
        history: list[dict] = []
        batch = {"i": 0}

        def on_batch_end(trainer):
            batch["i"] += 1
            self.batch_progress(batch["i"], len(trainer.train_loader))
            self._wait_if_paused(trainer)
            if self._stop.is_set():
                trainer.stop = True

        def on_epoch_end(trainer):
            batch["i"] = 0
            if trainer.epoch >= trainer.epochs:  # also fired for the final validation of best.pt
                return
            metrics = detection_metrics(trainer.metrics)
            history.append({"epoch": trainer.epoch + 1, **metrics})
            self.epoch_progress(trainer.epoch + 1, trainer.epochs)
            payload = {"history": list(history), "metrics": metrics, "curves": {}}
            validator = trainer.validator
            if validator is not None and getattr(validator, "confusion_matrix", None) is not None:
                payload["confusion"], payload["classes"] = confusion_payload(
                    validator.confusion_matrix.matrix, trainer.data["names"])
                payload["curves"] = confidence_curves(validator.metrics.box)
            self.evaluated(payload)
            self.log(f"Epoch {trainer.epoch + 1}/{trainer.epochs}: P {metrics['precision']:.3f}  "
                     f"R {metrics['recall']:.3f}  mAP50 {metrics['mAP50']:.3f}  "
                     f"mAP50-95 {metrics['mAP50_95']:.3f}")
            if self._stop.is_set():
                trainer.stop = True

        model.add_callback("on_train_batch_end", on_batch_end)
        model.add_callback("on_fit_epoch_end", on_epoch_end)
        self.log(f"Training detector from {base} on device {device} → {root / name}")
        seed = self.job.seed if self.job.seed is not None else random.randrange(2**31)
        model.train(data=str(root / name / "data.yaml"), epochs=cfg.epochs, imgsz=cfg.image_size,
                    batch=cfg.batch_size, device=device, workers=cfg.num_workers,
                    patience=cfg.patience, save_period=cfg.snapshot_every or -1, seed=seed,
                    deterministic=self.job.seed is not None, project=str(root), name=name,
                    exist_ok=True, plots=self.job.plots, verbose=False, **augmentation)

        trainer = model.trainer
        weights = root / name / "weights"
        best = weights / "best.pt" if (weights / "best.pt").is_file() else weights / "last.pt"
        (weights / "paused.pt").unlink(missing_ok=True)
        if self._stop.is_set():
            reason = "user"
        elif len(history) < cfg.epochs:
            reason = "early_stopping"
        else:
            reason = "completed"
        final = detection_metrics(getattr(trainer, "metrics", None)) if history else {}
        summary = {
            "weights": str(best.resolve()) if best.is_file() else None,
            "output_dir": str(root / name),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "stop_reason": reason,
            "epochs": len(history),
            "best_epoch": int(getattr(trainer.stopper, "best_epoch", 0)) + 1 if history else None,
            "metrics": final,
            "history": history,
            "base_model": base,
            "train_folders": list(self.job.train_folders),
            "val_folders": list(self.job.val_folders),
            "classes": list(data["names"].values()),
            "augmentation": augmentation,
            "config": asdict(cfg),
            "seed": seed,
        }
        (root / name / RUN_INFO).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary


def evaluate_detector(weights: str | Path, folders: Sequence[str], image_size: int = 640,
                      batch_size: int = 16, device: Optional[str | int] = None,
                      plots: bool = True) -> dict:
    """Final exam: Ultralytics validation of a detector on (test) YOLO folders."""
    import torch

    model = load_yolo_detector(weights)
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"
    with tempfile.TemporaryDirectory() as tmp:
        config = build_data_yaml(folders, folders, Path(tmp) / "data.yaml")
        model_names = [str(model.names[i]) for i in range(len(model.names))]
        if list(config["names"].values()) != model_names:
            raise ValueError("The test folders' classes differ from the detector's classes.\n"
                             f"test: {list(config['names'].values())}\ndetector: {model_names}")
        metrics = model.val(data=str(Path(tmp) / "data.yaml"), split="val", imgsz=image_size,
                            batch=batch_size, device=device, workers=0, plots=plots,
                            project=tmp, name="exam", exist_ok=True, verbose=False)
    box = metrics.box
    per_class = []
    for i, cls in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(i)
        per_class.append({"class": model_names[int(cls)], "precision": float(p), "recall": float(r),
                          "mAP50": float(ap50), "mAP50_95": float(ap)})
    confusion, classes = confusion_payload(metrics.confusion_matrix.matrix, dict(enumerate(model_names)))
    return {"weights": str(weights), "test_folders": list(folders),
            "images": len(yolo_images(folders)), "precision": float(box.mp),
            "recall": float(box.mr), "mAP50": float(box.map50), "mAP50_95": float(box.map),
            "per_class": per_class, "confusion": confusion, "classes": classes,
            "curves": confidence_curves(box)}


def yolo_images(folders: Sequence[str]) -> list[tuple[str, str, dict[int, str]]]:
    """``(image, label file, class names)`` of every image in the YOLO splits among the folders."""
    result = []
    for folder in folders:
        dirs = yolo_dirs(folder)
        if dirs is None:
            continue
        images_dir, labels_dir = dirs
        names = yolo_class_names(images_dir.parent)
        for image in sorted(p for p in images_dir.rglob("*") if is_image(p)):
            label = (labels_dir / image.relative_to(images_dir)).with_suffix(".txt")
            result.append((str(image), str(label), names))
    return result


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_boxes(predictions: list[dict], truths: list[dict], iou: float = 0.5) -> dict:
    """Greedy IoU matching; sets ``status`` on every box and returns the counts.

    Prediction status: ``correct``, ``wrong_class`` or ``extra``; truth status: ``found``,
    ``wrong_class`` or ``missed``."""
    pairs = sorted(((box_iou(p["box"], t["box"]), i, j) for i, p in enumerate(predictions)
                    for j, t in enumerate(truths)), reverse=True)
    used_p, used_t = set(), set()
    for value, i, j in pairs:
        if value < iou or i in used_p or j in used_t:
            continue
        used_p.add(i)
        used_t.add(j)
        ok = same_class(predictions[i]["label"], truths[j]["label"])
        predictions[i]["status"] = "correct" if ok else "wrong_class"
        truths[j]["status"] = "found" if ok else "wrong_class"
    for i, p in enumerate(predictions):
        p.setdefault("status", "extra")
    for t in truths:
        t.setdefault("status", "missed")
    counts = {"correct": sum(p["status"] == "correct" for p in predictions),
              "wrong_class": sum(p["status"] == "wrong_class" for p in predictions),
              "extra": sum(p["status"] == "extra" for p in predictions),
              "missed": sum(t["status"] == "missed" for t in truths),
              "truths": len(truths)}
    counts["perfect"] = counts["correct"] == len(truths) and counts["extra"] == 0
    return counts


class DetectionReviewSession:
    """Detector + labeled YOLO images + random history: predicted boxes vs. ground truth."""

    def __init__(self, weights: str, folders: Sequence[str], seed: Optional[int] = None,
                 conf: float = 0.25):
        self.detector = load_yolo_detector(weights)
        self.samples = yolo_images(folders)
        if not self.samples:
            raise ValueError("The selected folders contain no YOLO images (images/ + labels/).")
        self.conf = conf
        self.rng = random.Random(seed)
        self.history: list[int] = []
        self.position = -1

    def predict(self, index: int) -> dict:
        import torch

        path, label_file, names = self.samples[index]
        image = load_picture(path)
        w, h = image.size
        device = 0 if torch.cuda.is_available() else "cpu"
        result = self.detector(image, conf=self.conf, device=device, verbose=False)[0]
        boxes = result.boxes
        predictions = [{"box": tuple(float(v) for v in xyxy), "label": str(result.names[int(c)]),
                        "conf": float(p)}
                       for xyxy, p, c in zip(boxes.xyxy.tolist(), boxes.conf.tolist(),
                                             boxes.cls.tolist())]
        truths = [{"box": (x1 * w, y1 * h, x2 * w, y2 * h), "label": names.get(c, f"class_{c}")}
                  for c, (x1, y1, x2, y2) in read_yolo_labels(label_file)]
        counts = match_boxes(predictions, truths)
        for p in predictions:
            p["agree"] = p["status"] == "correct"
        for t in truths:
            t["color"] = TRUTH_RGB if t["status"] == "found" else DISAGREE_RGB
        return {"path": path, "image": image, "predictions": predictions, "truths": truths,
                "counts": counts, "pred_image": draw_detections(image, predictions),
                "truth_image": draw_detections(image, truths)}

    def next(self) -> dict:
        if self.position + 1 < len(self.history):
            self.position += 1
        else:
            self.history.append(self.rng.randrange(len(self.samples)))
            self.position = len(self.history) - 1
        return self.predict(self.history[self.position])

    def previous(self) -> Optional[dict]:
        if self.position <= 0:
            return None
        self.position -= 1
        return self.predict(self.history[self.position])
