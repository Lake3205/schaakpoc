"""Harmonize the class names of several YOLO datasets (``data.yaml`` + ``labels/*.txt``).

Different exports often name the same class differently (``W-R`` vs ``White-Rook``) and use
different class indices. :func:`apply_mapping` renames the tags to unified names, writes the same
unified ``names`` list to every ``.yaml`` file and rewrites the class ids of all YOLO label files
belonging to those ``.yaml`` files so they match the new unified indices.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional, Sequence

import yaml

from .data import _yaml_names, split_yaml, yolo_dirs

_LINE_RE = re.compile(r"^(\s*)(\S+)(.*)$", re.DOTALL)


def unique_yaml_paths(paths: Iterable[Optional[str | Path]]) -> list[Path]:
    """Resolved, de-duplicated ``.yaml`` paths in their original order (``None`` entries skipped)."""
    result: list[Path] = []
    for path in paths:
        if path is None:
            continue
        resolved = Path(path).resolve()
        if resolved not in result:
            result.append(resolved)
    return result


def yaml_class_names(path: str | Path) -> list[str]:
    """The ``names`` of a ``.yaml`` file ordered by class index."""
    names = _yaml_names(Path(path))
    return [names[i] for i in sorted(names)]


def collect_tags(yaml_paths: Sequence[str | Path]) -> dict[str, list[Path]]:
    """``{tag: [yaml files declaring it]}`` in order of first appearance."""
    tags: dict[str, list[Path]] = {}
    for path in unique_yaml_paths(yaml_paths):
        for name in yaml_class_names(path):
            sources = tags.setdefault(name, [])
            if path not in sources:
                sources.append(path)
    return tags


def normalize_tag(tag: str) -> str:
    """Comparison key ignoring case and ``-``/``_``/space separators (``White_Rook`` == ``white-rook``)."""
    return re.sub(r"[\s_\-]+", "", tag).lower()


def suggest_mapping(tags: Iterable[str]) -> dict[str, str]:
    """Default target per tag: tags that only differ in case/separators map to the first of them."""
    first: dict[str, str] = {}
    mapping = {}
    for tag in tags:
        mapping[tag] = first.setdefault(normalize_tag(tag), tag)
    return mapping


def _target(mapping: dict[str, str], name: str) -> str:
    return (mapping.get(name) or "").strip() or name


def unified_names(yaml_paths: Sequence[str | Path], mapping: dict[str, str]) -> list[str]:
    """The unified class list (order of first appearance of the target names)."""
    names: list[str] = []
    for tag in collect_tags(yaml_paths):
        target = _target(mapping, tag)
        if target not in names:
            names.append(target)
    return names


def yaml_label_dirs(yaml_path: str | Path) -> list[Path]:
    """``labels`` folders of the YOLO splits whose class names come from ``yaml_path``."""
    yaml_path = Path(yaml_path).resolve()
    base = yaml_path.parent
    candidates = [base / "labels", *base.glob("*/labels"), *base.glob("*/*/labels")]
    result = []
    for labels in sorted(set(candidates)):
        if not labels.is_dir() or yolo_dirs(labels.parent) is None:
            continue
        found = split_yaml(labels.parent)
        if found is not None and found.resolve() == yaml_path:
            result.append(labels)
    return result


def _label_files(yaml_path: Path) -> list[Path]:
    return sorted(p for d in yaml_label_dirs(yaml_path) for p in d.rglob("*.txt") if p.is_file())


def _class_id(token: str) -> Optional[int]:
    try:
        value = float(token)
    except ValueError:
        return None
    return int(value) if value.is_integer() and value >= 0 else None


def _write_all(contents: dict[Path, str]) -> None:
    """Write several files (nearly) all-or-nothing: every new content is staged in a ``.tmp`` file
    first; the originals are only replaced once all files were staged successfully."""
    staged: list[tuple[Path, Path]] = []
    try:
        for path, text in contents.items():
            tmp = path.with_name(path.name + ".sync.tmp")
            staged.append((tmp, path))
            tmp.write_text(text, encoding="utf-8")
    except BaseException:
        for tmp, _ in staged:
            tmp.unlink(missing_ok=True)
        raise
    for tmp, path in staged:
        tmp.replace(path)


def apply_mapping(yaml_paths: Sequence[str | Path], mapping: dict[str, str]) -> dict:
    """Rename tags and re-index all datasets to one unified class list.

    ``mapping`` maps a discovered tag to its unified target name (missing/empty = keep the tag).
    Label ids without a name in their ``.yaml`` become ``class_N`` (mappable like any tag).
    Returns a report with the unified ``names`` and the number of changed files/boxes.
    """
    paths = unique_yaml_paths(yaml_paths)
    per_yaml = {p: _yaml_names(p) for p in paths}
    files = {p: _label_files(p) for p in paths}

    names = unified_names(paths, mapping)
    for path in paths:  # ids used in labels but missing from the yaml
        for label_file in files[path]:
            for line in label_file.read_text(encoding="utf-8").splitlines():
                match = _LINE_RE.match(line)
                cls = _class_id(match.group(2)) if match else None
                if cls is not None and cls not in per_yaml[path]:
                    per_yaml[path][cls] = f"class_{cls}"
                    target = _target(mapping, f"class_{cls}")
                    if target not in names:
                        names.append(target)
    index = {name: i for i, name in enumerate(names)}

    contents: dict[Path, str] = {}
    changed_boxes = 0
    for path in paths:
        remap = {old: index[_target(mapping, name)] for old, name in per_yaml[path].items()}
        for label_file in files[path]:
            text = label_file.read_text(encoding="utf-8")
            lines = text.splitlines(keepends=True)
            changed = 0
            for i, line in enumerate(lines):
                match = _LINE_RE.match(line)
                cls = _class_id(match.group(2)) if match else None
                if cls is None or remap[cls] == cls and match.group(2) == str(cls):
                    continue
                lines[i] = f"{match.group(1)}{remap[cls]}{match.group(3)}"
                changed += 1
            if changed:
                contents[label_file] = "".join(lines)
                changed_boxes += changed
    changed_labels = len(contents)

    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        data["names"] = list(names)
        data["nc"] = len(names)
        contents[path] = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    _write_all(contents)

    return {"names": names, "yaml_files": [str(p) for p in paths],
            "label_files": changed_labels, "boxes": changed_boxes}
