"""Workspace projects persisted as JSON files in a ``.projects/`` directory."""
from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECTS_DIR = Path(".projects")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,63}$")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _from_dict(cls, data):
    """Build a dataclass from a dict, ignoring unknown keys (forward compatible)."""
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in (data or {}).items() if k in names})


@dataclass
class AugmentationConfig:
    hflip: bool = False
    vflip: bool = False
    rotation: bool = False
    rotation_degrees: float = 15.0
    color_jitter: bool = False
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    affine: bool = False
    affine_degrees: float = 0.0
    affine_translate: float = 0.1
    affine_shear: float = 10.0
    resized_crop: bool = False
    crop_scale_min: float = 0.8


@dataclass
class QuotaConfig:
    min_accuracy: float = 0.80
    target_avg_accuracy: float = 0.90
    min_confidence: float = 0.70
    target_avg_confidence: float = 0.80
    moving_average_window: int = 5
    stop_on_quota: bool = True
    snapshot_every: int = 10
    early_stopping: bool = True
    patience: int = 5
    monitor: str = "val_loss"  # "val_loss", "val_acc" or "both"


@dataclass
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 1e-3
    image_size: int = 224
    architecture: str = "resnet18"
    pretrained: bool = False
    num_workers: int = 0
    # Share of YOLO images cropped into one sample per piece, drawn at random each epoch; the other
    # images are trained as full pictures with all their piece classes as a multi-label target.
    crop_fraction: float = 1.0


@dataclass
class DatasetEntry:
    path: str
    checked: bool = True


@dataclass
class DetectorConfig:
    """Object detection training (bounding boxes, many objects per image) with Ultralytics YOLO."""

    base_model: str = "yolo26m.pt"
    epochs: int = 50
    image_size: int = 640
    batch_size: int = 16
    patience: int = 20  # Ultralytics early stopping (epochs without improvement), 0 = off
    snapshot_every: int = 0  # save weights every N epochs (0 = never)
    num_workers: int = 0
    mosaic: bool = True
    resume_from: Optional[str] = None  # fine-tune from earlier detector weights
    freeze: int = 0  # fine-tuning: keep the first N layers fixed (0 = train all layers)
    # The board corner (pose) task keeps its own base model and fine-tune source.
    board_base_model: str = "yolo26s-pose.pt"
    board_resume_from: Optional[str] = None


TASKS = {"detect": "Detection (boxes, many objects per image)",
         "classify": "Classification (one label per crop)",
         "board": "Board corners (4 keypoints per board)"}
DEFAULT_BASE_MODELS = {"detect": "yolo26m.pt", "board": "yolo26s-pose.pt"}


def default_base_model(task: str) -> str:
    """YOLO base model that fits a task (``-pose`` weights for the board corner model)."""
    return DEFAULT_BASE_MODELS.get(task, DEFAULT_BASE_MODELS["detect"])


def base_model_fits(base_model: str, task: str) -> bool:
    """Whether a base model name/weights path suits the task (pose for ``board``, else detect)."""
    is_pose = "-pose" in Path(str(base_model)).name
    return is_pose == (task == "board")


@dataclass
class ScanSettings:
    """Last choices of the garden chess scan screen."""

    piece_model: Optional[str] = None
    board_model: Optional[str] = None
    conf: float = 0.25
    lift: float = 0.08
    sample_root: Optional[str] = None


@dataclass
class Project:
    name: str
    task: str = "detect"  # "detect" (default), "classify" or "board" (corner keypoints)
    train_folders: list[DatasetEntry] = field(default_factory=list)
    val_folders: list[DatasetEntry] = field(default_factory=list)  # evaluated every epoch
    test_folders: list[DatasetEntry] = field(default_factory=list)  # final exam after training
    shuffle: bool = True
    seed: Optional[int] = None
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    quota: QuotaConfig = field(default_factory=QuotaConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    scan: ScanSettings = field(default_factory=ScanSettings)
    resume_checkpoint: Optional[str] = None
    checkpoints: list[str] = field(default_factory=list)
    evaluations: list[dict] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def checked_train_folders(self) -> list[str]:
        return [d.path for d in self.train_folders if d.checked]

    def checked_val_folders(self) -> list[str]:
        return [d.path for d in self.val_folders if d.checked]

    def checked_test_folders(self) -> list[str]:
        return [d.path for d in self.test_folders if d.checked]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Project":
        data = dict(data)
        if "val_folders" not in data:
            # Before v2.1 the "test" folders were evaluated every epoch, i.e. they were validation sets.
            data["val_folders"] = data.get("test_folders", [])
            data["test_folders"] = []
        data["train_folders"] = [_from_dict(DatasetEntry, d) for d in data.get("train_folders", [])]
        data["val_folders"] = [_from_dict(DatasetEntry, d) for d in data.get("val_folders", [])]
        data["test_folders"] = [_from_dict(DatasetEntry, d) for d in data.get("test_folders", [])]
        data["augmentation"] = _from_dict(AugmentationConfig, data.get("augmentation"))
        data["quota"] = _from_dict(QuotaConfig, data.get("quota"))
        data["training"] = _from_dict(TrainingConfig, data.get("training"))
        data["detector"] = _from_dict(DetectorConfig, data.get("detector"))
        data["scan"] = _from_dict(ScanSettings, data.get("scan"))
        return _from_dict(cls, data)


class ProjectManager:
    """Create, load, save and delete projects in a root directory."""

    def __init__(self, root: Path | str = PROJECTS_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def validate_name(name: str) -> str:
        name = name.strip()
        if not _NAME_RE.match(name):
            raise ValueError(
                "Project names must start with a letter or digit and may only contain "
                "letters, digits, spaces, '_' and '-' (max 64 characters)."
            )
        return name

    def json_path(self, name: str) -> Path:
        return self.root / f"{self.validate_name(name)}.json"

    def project_dir(self, name: str) -> Path:
        """Directory holding a project's checkpoints and other artifacts."""
        return self.root / self.validate_name(name)

    def list_projects(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.json") if p.is_file())

    def exists(self, name: str) -> bool:
        return self.json_path(name).exists()

    def create(self, name: str) -> Project:
        name = self.validate_name(name)
        if self.exists(name):
            raise ValueError(f"Project '{name}' already exists")
        project = Project(name=name)
        self.save(project)
        return project

    def load(self, name: str) -> Project:
        with open(self.json_path(name), encoding="utf-8") as f:
            return Project.from_dict(json.load(f))

    def save(self, project: Project) -> None:
        project.updated_at = _now()
        path = self.json_path(project.name)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(project.to_dict(), f, indent=2)
        tmp.replace(path)

    def delete(self, name: str) -> None:
        self.json_path(name).unlink(missing_ok=True)
        directory = self.project_dir(name)
        if directory.exists():
            shutil.rmtree(directory)

    def list_checkpoints(self, project: Project) -> list[str]:
        """All classifier checkpoint files of a project (saved runs + manually registered ones).

        YOLO weights in ``checkpoints/detectors`` and ``checkpoints/boards`` are a different format
        and excluded."""
        yolo_dirs = [(self.project_dir(project.name) / "checkpoints" / sub).resolve()
                     for sub in ("detectors", "boards")]
        found = {str(p.resolve()) for ext in ("*.pt", "*.pth")
                 for p in self.project_dir(project.name).rglob(ext)
                 if not any(p.resolve().is_relative_to(d) for d in yolo_dirs)}
        found.update(str(Path(c).resolve()) for c in project.checkpoints if Path(c).exists())
        return sorted(found)
