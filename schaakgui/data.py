"""Multi-folder image datasets and augmentation pipelines.

Two folder layouts are supported:

* ImageFolder: ``class_name/xxx.jpg`` - the sub-folder name is the label.
* YOLO: ``images/xxx.jpg`` + ``labels/xxx.txt`` (e.g. ``datasets/chess/train``). Every labeled box is
  cropped from its image and becomes one sample; class names come from the ``names`` of a
  ``data.yaml`` in the folder or one of its two parent folders.
"""
from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Callable, Optional, Sequence

import yaml
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader

from .projects import AugmentationConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Normalized (x1, y1, x2, y2) crop box of a YOLO label, ``None`` for the whole image.
Box = Optional[tuple[float, float, float, float]]


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMG_EXTENSIONS


def yolo_dirs(root: str | Path) -> Optional[tuple[Path, Path]]:
    """``(images_dir, labels_dir)`` if ``root`` is a YOLO split folder (or its ``images`` folder)."""
    root = Path(root)
    if (root / "images").is_dir() and (root / "labels").is_dir():
        return root / "images", root / "labels"
    if root.name == "images" and (root.parent / "labels").is_dir():
        return root, root.parent / "labels"
    return None


def _yaml_names(path: Path) -> dict[int, str]:
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return {}
    names = data.get("names") if isinstance(data, dict) else None
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    if isinstance(names, dict):
        try:
            return {int(k): str(v) for k, v in names.items()}
        except (TypeError, ValueError):
            return {}
    return {}


def yolo_class_names(split_dir: str | Path) -> dict[int, str]:
    """Class id -> name from the first ``*.yaml`` with ``names`` in the split folder or its parents."""
    split_dir = Path(split_dir).resolve()
    for directory in [split_dir, *list(split_dir.parents)[:2]]:
        candidates = [directory / "data.yaml"] + sorted(directory.glob("*.yaml")) \
            + sorted(directory.glob("*.yml"))
        for path in candidates:
            if path.is_file():
                names = _yaml_names(path)
                if names:
                    return names
    return {}


def read_yolo_labels(label_file: str | Path) -> list[tuple[int, tuple[float, float, float, float]]]:
    """``[(class_id, (x1, y1, x2, y2))]`` with normalized corners from a YOLO label file.

    Supports box lines (``cls xc yc w h``) and segmentation lines (``cls x1 y1 x2 y2 ...``).
    """
    result = []
    try:
        text = Path(label_file).read_text(encoding="utf-8")
    except OSError:
        return result
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            values = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        if len(values) == 4:
            xc, yc, w, h = values
            x1, y1, x2, y2 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
        else:
            xs, ys = values[0::2], values[1::2]
            n = min(len(xs), len(ys))
            x1, x2, y1, y2 = min(xs[:n]), max(xs[:n]), min(ys[:n]), max(ys[:n])
        x1, y1, x2, y2 = (min(1.0, max(0.0, v)) for v in (x1, y1, x2, y2))
        if x2 > x1 and y2 > y1:
            result.append((cls, (x1, y1, x2, y2)))
    return result


def scan_yolo_folder(images_dir: Path, labels_dir: Path) -> dict[str, list[tuple[str, Box]]]:
    """``{class_name: [(image path, box)]}`` for every labeled box of a YOLO split."""
    names = yolo_class_names(images_dir.parent)
    result: dict[str, list[tuple[str, Box]]] = {}
    for image in sorted(p for p in images_dir.rglob("*") if is_image(p)):
        label_file = (labels_dir / image.relative_to(images_dir)).with_suffix(".txt")
        for cls, box in read_yolo_labels(label_file):
            result.setdefault(names.get(cls, f"class_{cls}"), []).append((str(image), box))
    return dict(sorted(result.items()))


def scan_folder(root: str | Path) -> dict[str, list[tuple[str, Box]]]:
    """Return ``{class_name: [(image path, box)]}`` for an ImageFolder or YOLO directory."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {root}")
    yolo = yolo_dirs(root)
    if yolo is not None:
        return scan_yolo_folder(*yolo)
    splits = [p.name for p in sorted(root.iterdir()) if p.is_dir() and yolo_dirs(p) is not None]
    if splits:
        raise ValueError(f"{root} is the root of a YOLO dataset; add its split folders "
                         f"({', '.join(splits)}) separately instead.")
    result = {}
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        images = sorted(str(p) for p in class_dir.rglob("*") if is_image(p))
        if images:
            result[class_dir.name] = [(p, None) for p in images]
    return result


def discover_classes(folders: Sequence[str | Path]) -> list[str]:
    """Sorted union of class names over all folders."""
    names = set()
    for folder in folders:
        names.update(scan_folder(folder))
    return sorted(names)


def class_counts(folders: Sequence[str | Path]) -> dict[str, int]:
    """Aggregate number of samples per class across folders (sorted by class name)."""
    counts: Counter = Counter()
    for folder in folders:
        for name, images in scan_folder(folder).items():
            counts[name] += len(images)
    return dict(sorted(counts.items()))


def crop_box(image: Image.Image, box: Box) -> Image.Image:
    """Crop a normalized ``(x1, y1, x2, y2)`` box (at least 1 pixel) from an image."""
    if box is None:
        return image
    w, h = image.size
    x1, y1 = int(box[0] * w), int(box[1] * h)
    x2, y2 = max(x1 + 1, round(box[2] * w)), max(y1 + 1, round(box[3] * h))
    return image.crop((x1, y1, min(w, x2), min(h, y2)))


class MultiFolderImageDataset(Dataset):
    """Combines several ImageFolder-style roots into one dataset with a shared class index.

    Images whose class is not in ``classes`` are skipped (their class names are
    collected in ``skipped_classes``). Samples are ``(path, class index, box)`` tuples;
    ``box`` is ``None`` for ImageFolder images and the crop box of a YOLO label otherwise.
    """

    def __init__(
        self,
        folders: Sequence[str | Path],
        classes: Optional[Sequence[str]] = None,
        transform: Optional[Callable] = None,
        samples: Optional[list[tuple[str, int, Box]]] = None,
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
                    self.samples.extend((p, idx, box) for p, box in images)

    @property
    def targets(self) -> list[int]:
        return [s[1] for s in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def load_image(self, index: int) -> Image.Image:
        """The (cropped) RGB image of a sample, without transform."""
        path, _, box = self.samples[index]
        image = self.loader(path)
        if box is not None:
            # YOLO boxes refer to the EXIF-rotated image (as shown by viewers and labeling tools).
            image = ImageOps.exif_transpose(image)
        return crop_box(image, box)

    def __getitem__(self, index: int):
        image = self.load_image(index)
        target = self.samples[index][1]
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def subset(self, indices: Sequence[int], transform: Optional[Callable] = None):
        """New dataset over a subset of samples, optionally with a different transform."""
        return MultiFolderImageDataset(
            self.folders, self.classes, transform, [self.samples[i] for i in indices]
        )

    def split(self, val_fraction: float, seed: Optional[int], train_transform, val_transform):
        """Random train/validation split (used when no validation folders are selected)."""
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
