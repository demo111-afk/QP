"""Shared candidate types for missing- and wrong-annotation Vision checks."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any


RESIDUAL_CLUSTER = "residual_cluster"
EXISTING_BBOX = "existing_bbox"


@dataclass(frozen=True)
class VisionProjectionView:
    camera: str
    camera_id: str
    image_path: str
    roi: tuple[int, int, int, int]
    visible_points: int
    total_points: int
    clipped: bool
    image_size: tuple[int, int]


@dataclass(frozen=True)
class VisionCandidate:
    candidate_type: str
    candidate_id: str
    scene_id: str
    frame_index: int
    position: tuple[float, float, float] | None = None
    size: tuple[float, float, float] | None = None
    rotation: tuple[float, float, float] | None = None
    distance: float | None = None
    track_id: str = ""
    bbox_index: str = ""
    current_label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    projections: tuple[VisionProjectionView, ...] = ()

    # Compatibility properties used by the existing residual-Cluster path.
    @property
    def cluster_id(self) -> str:
        return self.candidate_id

    @property
    def center(self) -> tuple[float, float, float] | None:
        return self.position

    @property
    def point_count(self) -> int:
        return int(self.metadata.get("point_count", 0))

    @property
    def pca_features(self) -> dict[str, float]:
        return dict(self.metadata.get("pca_features", {}))

    @property
    def cluster_confidence(self) -> float:
        return float(self.metadata.get("cluster_confidence", 0.0))

    @property
    def merged_fragment_count(self) -> int:
        return int(self.metadata.get("merged_fragment_count", 1))

    @property
    def clustering_method(self) -> str:
        return str(self.metadata.get("clustering_method", ""))


def bbox_candidate_id(record) -> str:
    target = (
        getattr(record, "resolved_target_key", "")
        or getattr(record, "track_id", "")
        or f"index_{getattr(record, 'bbox_index', '')}"
    )
    safe_target = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(target)).strip("._") or "unknown"
    return f"bbox_{getattr(record, 'bbox_index', '')}_{safe_target}"


def distance_from_position(position: tuple[float, float, float] | None) -> float | None:
    if position is None:
        return None
    return math.sqrt(sum(value * value for value in position))
