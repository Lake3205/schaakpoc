"""Multi-folder image datasets (``class_name/xxx.jpg``) and augmentation pipelines."""
from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Callable, Optional, Sequence

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader

from .projects import AugmentationConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMG_EXTENSIONS


def scan_folder(root: str | Path) -> dict[str, list[str]]:
    """Return ``{class_name: [image paths]}`` for an ImageFolder-style directory."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {root}")
    result = {}
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        images = sorted(str(p) for p in class_dir.rglob("*") if is_image(p))
        if images:
            result[class_dir.name] = images
    return result


def discover_classes(folders: Sequence[str | Path]) -> list[str]:
    """Sorted union of class names over all folders."""
    names = set()
    for folder in folders:
        names.update(scan_folder(folder))
    return sorted(names)


def class_counts(folders: Sequence[str | Path]) -> dict[str, int]:
    """Aggregate number of images per class across folders (sorted by class name)."""
    counts: Counter = Counter()
    for folder in folders:
        for name, images in scan_folder(folder).items():
            counts[name] += len(images)
    return dict(sorted(counts.items()))


class MultiFolderImageDataset(Dataset):
    """Combines several ImageFolder-style roots into one dataset with a shared class index.

    Images whose class is not in ``classes`` are skipped (their class names are
    collected in ``skipped_classes``).
    """

    def __init__(
        self,
        folders: Sequence[str | Path],
        classes: Optional[Sequence[str]] = None,
        transform: Optional[Callable] = None,
        samples: Optional[list[tuple[str, int]]] = None,
    ):
        self.folders = [str(f) for f in folders]
        self.classes = list(classes) if classes is not None else discover_classes(folders)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform = transform
        self.loader = default_loader
        self.skipped_classes: set[str] = set()
        if samples is not None:
            self.samples = list(samples)
        else:
            self.samples = []
            for folder in self.folders:
                for name, images in scan_folder(folder).items():
                    if name not in self.class_to_idx:
                        self.skipped_classes.add(name)
                        continue
                    idx = self.class_to_idx[name]
                    self.samples.extend((p, idx) for p in images)

    @property
    def targets(self) -> list[int]:
        return [t for _, t in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def subset(self, indices: Sequence[int], transform: Optional[Callable] = None):
        """New dataset over a subset of samples, optionally with a different transform."""
        return MultiFolderImageDataset(
            self.folders, self.classes, transform, [self.samples[i] for i in indices]
        )

    def split(self, val_fraction: float, seed: Optional[int], train_transform, val_transform):
        """Random train/validation split (used when no test folders are selected)."""
        indices = list(range(len(self.samples)))
        random.Random(seed).shuffle(indices)
        n_val = max(1, int(round(len(indices) * val_fraction))) if len(indices) > 1 else 0
        return (self.subset(indices[n_val:], train_transform),
                self.subset(indices[:n_val], val_transform))


def augmentation_steps(aug: AugmentationConfig, image_size: int) -> list:
    """The torchvision transforms enabled by the augmentation checkboxes."""
    steps = []
    if aug.resized_crop:
        steps.append(transforms.RandomResizedCrop(image_size, scale=(aug.crop_scale_min, 1.0)))
    else:
        steps.append(transforms.Resize((image_size, image_size)))
    if aug.hflip:
        steps.append(transforms.RandomHorizontalFlip(p=0.5))
    if aug.vflip:
        steps.append(transforms.RandomVerticalFlip(p=0.5))
    if aug.rotation:
        steps.append(transforms.RandomRotation(degrees=aug.rotation_degrees))
    if aug.color_jitter:
        steps.append(transforms.ColorJitter(
            brightness=aug.brightness, contrast=aug.contrast, saturation=aug.saturation))
    if aug.affine:
        steps.append(transforms.RandomAffine(
            degrees=aug.affine_degrees,
            translate=(aug.affine_translate, aug.affine_translate),
            shear=aug.affine_shear,
        ))
    return steps


def build_train_transform(aug: AugmentationConfig, image_size: int) -> transforms.Compose:
    return transforms.Compose(augmentation_steps(aug, image_size) + [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def enabled_augmentations(aug: AugmentationConfig) -> list[str]:
    """Human readable names of the enabled augmentations."""
    names = []
    if aug.hflip:
        names.append("RandomHorizontalFlip(p=0.5)")
    if aug.vflip:
        names.append("RandomVerticalFlip(p=0.5)")
    if aug.rotation:
        names.append(f"RandomRotation(±{aug.rotation_degrees:g}°)")
    if aug.color_jitter:
        names.append(f"ColorJitter(b={aug.brightness:g}, c={aug.contrast:g}, s={aug.saturation:g})")
    if aug.affine:
        names.append(f"RandomAffine(deg={aug.affine_degrees:g}, translate={aug.affine_translate:g}, "
                     f"shear={aug.affine_shear:g})")
    if aug.resized_crop:
        names.append(f"RandomResizedCrop(scale={aug.crop_scale_min:g}-1.0)")
    return names
