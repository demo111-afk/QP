"""Content-addressed persistent cache for successful Vision verification results."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from vision_candidate import VisionCandidate


@dataclass(frozen=True)
class VisionCacheLookup:
    key: str
    verification: dict | None
    original_api_seconds: float = 0.0

    @property
    def hit(self) -> bool:
        return self.verification is not None


class VisionResultCache:
    """Cache only verified model JSON; failures and skips are always retried."""

    def __init__(self, input_config: dict, vision_config: dict) -> None:
        config = vision_config.get("cache", {}) or {}
        self.enabled = bool(config.get("enabled", True))
        self.schema_version = str(config.get("schema_version", "vision-cache-v1"))
        self.output_root = Path(input_config.get("output_dir", "ai_inputs"))
        self.dirname = str(config.get("dirname", ".vision_cache"))
        self.model_name = str(vision_config.get("model_name", ""))
        self.provider = str(vision_config.get("provider", ""))
        self.api_base = str(vision_config.get("api_base", ""))
        self.label_aliases = (
            (vision_config.get("wrong_annotation", {}) or {}).get("label_aliases", {}) or {}
        )
        self.prompt_source_hash = _file_hash(Path(__file__).with_name("vision_verifier.py"))
        category_path = vision_config.get("category_whitelist_path")
        self.category_source_hash = _file_hash(_resolve_project_path(category_path))

    def lookup(
        self,
        task: VisionCandidate,
        metadata: dict,
        image_paths: Iterable[str | Path],
    ) -> VisionCacheLookup:
        if not self.enabled:
            return VisionCacheLookup(key="", verification=None)
        key = self._key(task, metadata, image_paths)
        path = self._path(task.scene_id, key)
        if not path.is_file():
            return VisionCacheLookup(key=key, verification=None)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return VisionCacheLookup(key=key, verification=None)
        verification = data.get("verification")
        if (
            data.get("schema_version") != self.schema_version
            or data.get("candidate_type") != task.candidate_type
            or not isinstance(verification, dict)
            or verification.get("status") != "verified"
        ):
            return VisionCacheLookup(key=key, verification=None)
        try:
            original_api_seconds = max(0.0, float(data.get("original_api_seconds", 0.0)))
        except (TypeError, ValueError):
            original_api_seconds = 0.0
        return VisionCacheLookup(
            key=key,
            verification=verification,
            original_api_seconds=original_api_seconds,
        )

    def save(
        self,
        task: VisionCandidate,
        key: str,
        verification: dict,
        original_api_seconds: float = 0.0,
    ) -> None:
        if (
            not self.enabled
            or not key
            or verification.get("status") != "verified"
        ):
            return
        path = self._path(task.scene_id, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.schema_version,
            "candidate_type": task.candidate_type,
            "model_name": self.model_name,
            "original_api_seconds": max(0.0, float(original_api_seconds)),
            "verification": verification,
        }
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{key}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _key(
        self,
        task: VisionCandidate,
        metadata: dict,
        image_paths: Iterable[str | Path],
    ) -> str:
        digest = hashlib.sha256()
        payload = {
            "schema_version": self.schema_version,
            "candidate_type": task.candidate_type,
            "model_name": self.model_name,
            "provider": self.provider,
            "api_base": self.api_base,
            "prompt_source_hash": self.prompt_source_hash,
            "category_source_hash": self.category_source_hash,
            "label_aliases": self.label_aliases,
            "metadata": metadata,
        }
        digest.update(json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8"))
        for path_value in image_paths:
            path = Path(path_value)
            digest.update(path.name.encode("utf-8"))
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()

    def _path(self, scene_id: str, key: str) -> Path:
        safe_scene = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(scene_id)).strip("._")
        return self.output_root / f"scene_{safe_scene or 'unknown'}" / self.dirname / f"{key}.json"


def _resolve_project_path(value) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else Path(__file__).resolve().parent / path


def _file_hash(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
