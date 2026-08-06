"""Load the table-derived category whitelist used by Vision verification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


DEFAULT_PATH = Path(__file__).resolve().parent / "config" / "vision_categories.yaml"


@dataclass(frozen=True)
class VisionCategory:
    name: str
    description: str = ""
    dimension_keys: tuple[str, ...] = ()


def load_vision_categories(path: str | Path = DEFAULT_PATH) -> tuple[VisionCategory, ...]:
    source = Path(path)
    if not source.is_absolute():
        source = Path(__file__).resolve().parent / source
    with source.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    categories = []
    seen = set()
    for index, item in enumerate(data.get("categories", []) or [], start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Vision category row {index} must be an object")
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError(f"Vision category row {index} has no name")
        if name in seen:
            raise ValueError(f"Duplicate Vision category: {name}")
        seen.add(name)
        categories.append(VisionCategory(
            name=name,
            description=str(item.get("description", "")).strip(),
            dimension_keys=tuple(
                str(value).strip()
                for value in (item.get("dimension_keys") or [name])
                if str(value).strip()
            ),
        ))
    if not categories:
        raise ValueError(f"Vision category whitelist is empty: {source}")
    return tuple(categories)


def format_categories_for_prompt(categories: tuple[VisionCategory, ...]) -> str:
    lines = []
    for category in categories:
        suffix = f"：{category.description}" if category.description else ""
        lines.append(f"- {category.name}{suffix}")
    return "\n".join(lines)
