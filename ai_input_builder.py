"""Build one-candidate-at-a-time image packages for vision verification."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

from cluster_detector import ClusterCandidate
from cluster_projection import ProjectedROI
from vision_candidate import (
    EXISTING_BBOX,
    RESIDUAL_CLUSTER,
    VisionCandidate,
    VisionProjectionView,
    bbox_candidate_id,
    distance_from_position,
)


ClusterProjectionView = VisionProjectionView
ClusterVisionTask = VisionCandidate


@dataclass(frozen=True)
class ProjectionQuality:
    camera: str
    quality: str
    reasons: tuple[str, ...]
    roi_area_px: int
    roi_area_ratio: float
    border_touch_sides: int


@dataclass(frozen=True)
class PreparedVisionInput:
    candidate: VisionCandidate
    projection_quality: str
    manifest_path: str
    context_paths: tuple[str, ...]
    crop_paths: tuple[str, ...]
    accepted_cameras: tuple[str, ...]
    rois: tuple[tuple[int, int, int, int], ...]

    @property
    def task(self) -> VisionCandidate:
        return self.candidate


PreparedClusterInput = PreparedVisionInput


def build_cluster_vision_tasks(
    scene_id: str,
    frame_index: int,
    clusters: Iterable[ClusterCandidate],
    projections_by_asset: dict[str, list[ProjectedROI]],
) -> tuple[VisionCandidate, ...]:
    """Convert detector/projection objects into lightweight serializable tasks."""
    views_by_cluster: dict[str, list[VisionProjectionView]] = {}
    for asset_name, rois in projections_by_asset.items():
        for projected in rois:
            if not projected.image_path or not projected.target_image_size:
                continue
            views_by_cluster.setdefault(projected.cluster_id, []).append(
                VisionProjectionView(
                    camera=asset_name,
                    camera_id=projected.camera_id,
                    image_path=projected.image_path,
                    roi=projected.roi,
                    visible_points=projected.visible_points,
                    total_points=projected.total_points,
                    clipped=projected.clipped,
                    image_size=projected.target_image_size,
                )
            )

    tasks = []
    for cluster in clusters:
        views = tuple(sorted(views_by_cluster.get(cluster.cluster_id, []), key=lambda item: item.camera))
        center = tuple(cluster.centroid) if cluster.centroid is not None else None
        size = None
        if cluster.bbox_hint and cluster.bbox_hint.get("size") is not None:
            size = tuple(float(value) for value in cluster.bbox_hint["size"])
        tasks.append(VisionCandidate(
            candidate_type=RESIDUAL_CLUSTER,
            candidate_id=cluster.cluster_id,
            scene_id=str(scene_id),
            frame_index=int(frame_index),
            position=center,
            size=size,
            distance=distance_from_position(center),
            metadata={
                "point_count": int(cluster.point_count),
                "pca_features": _pca_features(cluster.points_xyz),
                "cluster_confidence": float(cluster.cluster_confidence),
                "merged_fragment_count": int(cluster.merged_fragment_count),
                "clustering_method": cluster.clustering_method,
            },
            projections=views,
        ))
    return tuple(tasks)


def build_bbox_vision_tasks(
    scene_id: str,
    frame_index: int,
    bbox_records: Iterable,
    projections_by_asset: dict[str, list[ProjectedROI]],
) -> tuple[VisionCandidate, ...]:
    """Convert Existing BBoxes and their shared Projection output into candidates."""
    views_by_candidate: dict[str, list[VisionProjectionView]] = {}
    for asset_name, rois in projections_by_asset.items():
        for projected in rois:
            if not projected.image_path or not projected.target_image_size:
                continue
            views_by_candidate.setdefault(projected.candidate_id, []).append(VisionProjectionView(
                camera=asset_name,
                camera_id=projected.camera_id,
                image_path=projected.image_path,
                roi=projected.roi,
                visible_points=projected.visible_points,
                total_points=projected.total_points,
                clipped=projected.clipped,
                image_size=projected.target_image_size,
            ))

    tasks = []
    for record in bbox_records:
        candidate_id = bbox_candidate_id(record)
        values = (
            record.position_x, record.position_y, record.position_z,
            record.rotation_x, record.rotation_y, record.rotation_z,
            record.scale_x, record.scale_y, record.scale_z,
        )
        if any(value is None for value in values):
            continue
        position = tuple(float(value) for value in values[:3])
        rotation = tuple(float(value) for value in values[3:6])
        size = tuple(float(value) for value in values[6:9])
        label = str(record.className or record.label or "")
        track_id = str(record.resolved_target_key or record.track_id or "")
        tasks.append(VisionCandidate(
            candidate_type=EXISTING_BBOX,
            candidate_id=candidate_id,
            scene_id=str(scene_id),
            frame_index=int(frame_index),
            position=position,
            rotation=rotation,
            size=size,
            distance=distance_from_position(position),
            track_id=track_id,
            bbox_index=str(record.bbox_index),
            current_label=label,
            metadata={
                "current_label": label,
                "class_id": str(record.class_id or ""),
                "track_id": track_id,
                "bbox_index": str(record.bbox_index),
                "point_count": int(record.point_count or 0),
            },
            projections=tuple(sorted(
                views_by_candidate.get(candidate_id, []), key=lambda item: item.camera
            )),
        ))
    return tuple(tasks)


def assess_projection_quality(
    view: VisionProjectionView,
    quality_config: dict,
) -> ProjectionQuality:
    x1, y1, x2, y2 = view.roi
    roi_width = max(0, x2 - x1)
    roi_height = max(0, y2 - y1)
    roi_area = roi_width * roi_height
    image_width, image_height = view.image_size
    image_area = max(1, image_width * image_height)
    area_ratio = roi_area / image_area

    edge_margin = max(0, int(quality_config.get("edge_margin_px", 2)))
    border_touch_sides = sum((
        x1 <= edge_margin,
        y1 <= edge_margin,
        x2 >= image_width - 1 - edge_margin,
        y2 >= image_height - 1 - edge_margin,
    ))

    reasons = []
    if roi_area < int(quality_config.get("min_roi_area_px", 400)):
        reasons.append("roi_area_too_small")
    if roi_width < int(quality_config.get("min_roi_width_px", 1)):
        reasons.append("roi_width_too_small")
    if roi_height < int(quality_config.get("min_roi_height_px", 1)):
        reasons.append("roi_height_too_small")
    if area_ratio > float(quality_config.get("max_roi_area_ratio", 0.5)):
        reasons.append("roi_area_too_large")
    if view.visible_points < int(quality_config.get("min_visible_points", 8)):
        reasons.append("visible_points_too_few")
    if border_touch_sides > int(quality_config.get("max_border_touch_sides", 1)):
        reasons.append("roi_at_image_edge")

    return ProjectionQuality(
        camera=view.camera,
        quality="good" if not reasons else "projection_low_quality",
        reasons=tuple(reasons),
        roi_area_px=roi_area,
        roi_area_ratio=area_ratio,
        border_touch_sides=border_touch_sides,
    )


def prepare_vision_input(
    task: VisionCandidate,
    config: dict,
) -> PreparedVisionInput:
    """Write context/crop images and a manifest for one selected candidate."""
    output_root = Path(config.get("output_dir", "ai_inputs"))
    candidate_dir = (
        output_root
        / f"scene_{_safe_name(task.scene_id)}"
        / f"frame_{task.frame_index:04d}"
        / _safe_name(task.candidate_id)
    )
    candidate_dir.mkdir(parents=True, exist_ok=True)
    for stale in candidate_dir.glob("*.jpg"):
        stale.unlink()
    manifest_path = candidate_dir / "manifest.json"

    quality_config = (
        (config.get("quality_by_candidate", {}) or {}).get(task.candidate_type)
        or config.get("quality", {})
        or {}
    )
    checks = [assess_projection_quality(view, quality_config) for view in task.projections]
    checks_by_camera = {check.camera: check for check in checks}
    good_views = [
        view for view in task.projections
        if checks_by_camera[view.camera].quality == "good"
    ]
    projection_quality = "good" if good_views else "projection_low_quality"

    context_paths = []
    crop_paths = []
    image_entries = []
    accepted_cameras = []
    accepted_rois = []
    for view in good_views:
        image_config = dict(config)
        candidate_crop_ratios = config.get("crop_expand_ratio_by_candidate", {}) or {}
        if task.candidate_type in candidate_crop_ratios:
            image_config["crop_expand_ratio"] = candidate_crop_ratios[task.candidate_type]
        context_path, crop_path = _write_view_images(view, candidate_dir, image_config)
        context_paths.append(str(context_path))
        crop_paths.append(str(crop_path))
        accepted_cameras.append(view.camera)
        accepted_rois.append(view.roi)
        image_entries.append({
            "camera": view.camera,
            "camera_id": view.camera_id,
            "context": str(context_path),
            "crop": str(crop_path),
            "roi": list(view.roi),
            "visible_points": view.visible_points,
        })

    manifest = {
        "candidate_type": task.candidate_type,
        "candidate_id": task.candidate_id,
        "scene_id": task.scene_id,
        "frame_index": task.frame_index,
        "cluster_id": task.cluster_id if task.candidate_type == RESIDUAL_CLUSTER else None,
        "track_id": task.track_id,
        "bbox_index": task.bbox_index,
        "current_label": task.current_label,
        "point_count": task.point_count,
        "center": list(task.position) if task.position is not None else None,
        "size": list(task.size) if task.size is not None else None,
        "rotation": list(task.rotation) if task.rotation is not None else None,
        "distance": task.distance,
        "metadata": task.metadata,
        "pca_features": task.pca_features,
        "cluster_confidence": task.cluster_confidence,
        "merged_fragment_count": task.merged_fragment_count,
        "clustering_method": task.clustering_method,
        "projection_quality": projection_quality,
        "projection_checks": [
            {
                "camera": check.camera,
                "quality": check.quality,
                "reasons": list(check.reasons),
                "roi_area_px": check.roi_area_px,
                "roi_area_ratio": check.roi_area_ratio,
                "border_touch_sides": check.border_touch_sides,
            }
            for check in checks
        ],
        "images": image_entries,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return PreparedVisionInput(
        candidate=task,
        projection_quality=projection_quality,
        manifest_path=str(manifest_path),
        context_paths=tuple(context_paths),
        crop_paths=tuple(crop_paths),
        accepted_cameras=tuple(accepted_cameras),
        rois=tuple(accepted_rois),
    )


def prepare_cluster_input(task: VisionCandidate, config: dict) -> PreparedVisionInput:
    """Backward-compatible name for the original Cluster-only entry point."""
    return prepare_vision_input(task, config)


def update_manifest(manifest_path: str | Path, section: str, value) -> None:
    path = Path(manifest_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data[section] = value
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_view_images(
    view: VisionProjectionView,
    cluster_dir: Path,
    config: dict,
) -> tuple[Path, Path]:
    source_path = Path(view.image_path)
    with Image.open(source_path) as source:
        original = source.convert("RGB")

    camera_name = _safe_name(view.camera)
    context_path = cluster_dir / f"{camera_name}_context.jpg"
    context = original.copy()
    draw = ImageDraw.Draw(context)
    draw.rectangle(
        view.roi,
        outline=(255, 32, 32),
        width=max(1, int(config.get("context_line_width", 2))),
    )

    crop_bounds = _expanded_crop_bounds(
        view.roi,
        original.size,
        float(config.get("crop_expand_ratio", 0.5)),
    )
    crop = original.crop(crop_bounds)
    crop_path = cluster_dir / f"{camera_name}_crop.jpg"
    quality = max(1, min(100, int(config.get("jpeg_quality", 92))))
    context.save(context_path, quality=quality)
    crop.save(crop_path, quality=quality)
    return context_path, crop_path


def _expanded_crop_bounds(
    roi: tuple[int, int, int, int],
    image_size: tuple[int, int],
    expand_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = roi
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    margin_x = int(round(width * max(0.0, expand_ratio)))
    margin_y = int(round(height * max(0.0, expand_ratio)))
    image_width, image_height = image_size
    return (
        max(0, x1 - margin_x),
        max(0, y1 - margin_y),
        min(image_width, x2 + margin_x),
        min(image_height, y2 + margin_y),
    )


def _pca_features(points_xyz: np.ndarray | None) -> dict[str, float]:
    empty = {"linearity": 0.0, "planarity": 0.0, "flatness": 0.0, "thickness_ratio": 0.0}
    if points_xyz is None:
        return empty
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 3:
        return empty
    covariance = np.cov(points[:, :3] - points[:, :3].mean(axis=0), rowvar=False)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
    lam1, lam2, lam3 = (float(max(0.0, value)) for value in eigenvalues)
    if lam1 <= 1e-12:
        return empty
    linearity = (lam1 - lam2) / lam1
    planarity = (lam2 - lam3) / lam1
    return {
        "linearity": linearity,
        "planarity": planarity,
        "flatness": 1.0 - max(linearity, planarity),
        "thickness_ratio": lam3 / lam1,
    }


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return cleaned or "unknown"
