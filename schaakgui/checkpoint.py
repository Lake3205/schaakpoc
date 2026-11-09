"""Model construction and checkpoint save/load with lineage metadata."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torchvision import models

ARCHITECTURES = ("resnet18", "resnet34", "resnet50", "mobilenet_v3_small", "efficientnet_b0")


def build_model(architecture: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    """A torchvision classifier with its head replaced for ``num_classes`` outputs."""
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {architecture}")
    model = getattr(models, architecture)(weights="DEFAULT" if pretrained else None)
    if architecture.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        last = model.classifier[-1]
        model.classifier[-1] = nn.Linear(last.in_features, num_classes)
    return model


@dataclass
class Lineage:
    """Provenance stored inside every checkpoint."""

    total_epochs_completed: int = 0
    quota_history: list[dict] = field(default_factory=list)
    sessions: list[dict] = field(default_factory=list)
    snapshots: list[dict] = field(default_factory=list)
    best: Optional[dict] = None
    history: list[dict] = field(default_factory=list)

    def dataset_sources(self) -> dict[str, list[str]]:
        """Unique train/test folders used over all sessions (in first-use order)."""
        train, test = [], []
        for s in self.sessions:
            train += [f for f in s.get("train_folders", []) if f not in train]
            test += [f for f in s.get("test_folders", []) if f not in test]
        return {"train": train, "test": test}

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "Lineage":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (data or {}).items() if k in names})


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    classes: list[str],
    architecture: str,
    image_size: int,
    lineage: Lineage,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metrics: Optional[dict] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format": "schaakgui-v2",
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "classes": list(classes),
        "architecture": architecture,
        "image_size": int(image_size),
        "epoch": epoch if epoch is not None else lineage.total_epochs_completed,
        "metrics": metrics or {},
        "lineage": lineage.to_dict(),
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)
    return path


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    """Load a checkpoint saved by :func:`save_checkpoint`.

    Uses ``weights_only=True`` so untrusted files cannot execute code.
    """
    ckpt = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(ckpt, dict) or "model_state" not in ckpt or "classes" not in ckpt:
        raise ValueError(f"{path} is not a schaakgui checkpoint")
    return ckpt


def read_lineage(path: str | Path) -> Lineage:
    return Lineage.from_dict(load_checkpoint(path).get("lineage"))


def model_from_checkpoint(ckpt: dict) -> nn.Module:
    model = build_model(ckpt["architecture"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    return model
