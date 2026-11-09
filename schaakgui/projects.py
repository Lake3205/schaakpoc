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
    monitor: str = "val_loss"  # "val_loss" or "val_acc"


@dataclass
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 1e-3
    image_size: int = 224
    architecture: str = "resnet18"
    pretrained: bool = False
    num_workers: int = 0


@dataclass
class DatasetEntry:
    path: str
    checked: bool = True


@dataclass
class Project:
    name: str
    train_folders: list[DatasetEntry] = field(default_factory=list)
    test_folders: list[DatasetEntry] = field(default_factory=list)
    shuffle: bool = True
    seed: Optional[int] = None
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    quota: QuotaConfig = field(default_factory=QuotaConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    resume_checkpoint: Optional[str] = None
    checkpoints: list[str] = field(default_factory=list)
    evaluations: list[dict] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def checked_train_folders(self) -> list[str]:
        return [d.path for d in self.train_folders if d.checked]

    def checked_test_folders(self) -> list[str]:
        return [d.path for d in self.test_folders if d.checked]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Project":
        data = dict(data)
        data["train_folders"] = [_from_dict(DatasetEntry, d) for d in data.get("train_folders", [])]
        data["test_folders"] = [_from_dict(DatasetEntry, d) for d in data.get("test_folders", [])]
        data["augmentation"] = _from_dict(AugmentationConfig, data.get("augmentation"))
        data["quota"] = _from_dict(QuotaConfig, data.get("quota"))
        data["training"] = _from_dict(TrainingConfig, data.get("training"))
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
        """All checkpoint files of a project (saved runs + manually registered ones)."""
        found = {str(p.resolve()) for ext in ("*.pt", "*.pth")
                 for p in self.project_dir(project.name).rglob(ext)}
        found.update(str(Path(c).resolve()) for c in project.checkpoints if Path(c).exists())
        return sorted(found)
