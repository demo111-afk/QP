"""
Single-frame residual-cluster projection pipeline.

This module coordinates existing Cluster and Projection APIs for one downloaded
asset frame. It does not alter Browser Automation, Rule Engine, or reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ai_input_builder import build_bbox_vision_tasks, build_cluster_vision_tasks
from bbox_extractor import BBoxRecord
from calibration import CalibrationManager
from cluster_detector import OrientedBox, detect_clusters
from cluster_projection import (
    project_clusters_to_asset_cameras,
    project_oriented_boxes_to_asset_cameras,
    save_projection_debug_images,
)
from vision_candidate import VisionCandidate, bbox_candidate_id


@dataclass(frozen=True)
class FrameProjectionResult:
    frame_index: int
    cluster_count: int = 0
    visible_roi_count: int = 0
    saved_images: tuple[str, ...] = ()
    bbox_count: int = 0
    bbox_visible_roi_count: int = 0
    vision_tasks: tuple[VisionCandidate, ...] = ()
    cluster_skipped_reason: str = ""
    skipped_reason: str = ""

    @property
    def skipped(self) -> bool:
        return bool(self.skipped_reason)


def clear_frame_projection_output(frame_dir: str | Path, projection_config: dict) -> None:
    """Remove stale debug JPGs without touching original downloaded images."""
    frame_path = Path(frame_dir)
    output_dir = frame_path / projection_config.get("output_dirname", "projection_debug")
    save_projection_debug_images({}, output_dir)


def bbox_records_to_oriented(records: Iterable[BBoxRecord]) -> list[OrientedBox]:
    """Convert successfully extracted page BBoxes into Cluster Detector boxes."""
    boxes = []
    for record in records:
        position = (record.position_x, record.position_y, record.position_z)
        rotation = (record.rotation_x, record.rotation_y, record.rotation_z)
        size = (record.scale_x, record.scale_y, record.scale_z)
        if any(value is None for value in (*position, *rotation, *size)):
            continue
        boxes.append(OrientedBox(
            center=tuple(float(value) for value in position),
            rotation_euler=tuple(float(value) for value in rotation),
            size=tuple(float(value) for value in size),
        ))
    return boxes


def bbox_records_with_oriented(
    records: Iterable[BBoxRecord],
) -> list[tuple[str, OrientedBox]]:
    """Keep Candidate IDs attached while reusing the detector's BBox geometry type."""
    output = []
    for record in records:
        position = (record.position_x, record.position_y, record.position_z)
        rotation = (record.rotation_x, record.rotation_y, record.rotation_z)
        size = (record.scale_x, record.scale_y, record.scale_z)
        if any(value is None for value in (*position, *rotation, *size)):
            continue
        output.append((
            bbox_candidate_id(record),
            OrientedBox(
                center=tuple(float(value) for value in position),
                rotation_euler=tuple(float(value) for value in rotation),
                size=tuple(float(value) for value in size),
            ),
        ))
    return output


def run_frame_projection(
    frame_index: int,
    frame_dir: str | Path,
    bbox_records: Iterable[BBoxRecord],
    calibration: CalibrationManager,
    cluster_config: dict,
    projection_config: dict,
    scene_id: str = "",
) -> FrameProjectionResult:
    """Run Cluster -> Camera Projection -> Debug JPG for one sampled frame."""
    frame_path = Path(frame_dir)
    images_dir = frame_path / "images"
    pcd_path = frame_path / "pointcloud.pcd"
    output_dir = frame_path / projection_config.get("output_dirname", "projection_debug")
    clear_frame_projection_output(frame_path, projection_config)

    if not images_dir.is_dir() or not any(images_dir.glob("*.jpg")):
        return FrameProjectionResult(frame_index=frame_index, skipped_reason="本帧 JPG 不存在")

    bbox_records = tuple(bbox_records)
    boxes = bbox_records_to_oriented(bbox_records)
    cluster_skipped_reason = ""
    if pcd_path.is_file():
        clusters = detect_clusters(str(pcd_path), boxes, cluster_config)
    else:
        clusters = []
        cluster_skipped_reason = "pointcloud.pcd 不存在"
    projections = project_clusters_to_asset_cameras(
        clusters,
        calibration,
        images_dir,
        min_depth=float(projection_config.get("min_depth", 0.1)),
        min_visible_points=int(projection_config.get("min_visible_points", 3)),
        padding_px=int(projection_config.get("padding_px", 8)),
    )
    visible_roi_count = sum(len(rois) for rois in projections.values())
    saved_images = tuple(save_projection_debug_images(
        projections,
        output_dir,
        line_width=int(projection_config.get("line_width", 1)),
    ))
    vision_tasks = build_cluster_vision_tasks(
        scene_id=scene_id,
        frame_index=frame_index,
        clusters=clusters,
        projections_by_asset=projections,
    )
    wrong_config = projection_config.get("wrong_annotation", {}) or {}
    bbox_projections = {}
    bbox_tasks = ()
    if wrong_config.get("enabled", True):
        bbox_geometry = bbox_records_with_oriented(bbox_records)
        bbox_projections = project_oriented_boxes_to_asset_cameras(
            bbox_geometry,
            calibration,
            images_dir,
            min_depth=float(projection_config.get("min_depth", 0.1)),
            min_visible_points=int(wrong_config.get("min_visible_points", 3)),
            padding_px=int(wrong_config.get("padding_px", projection_config.get("padding_px", 8))),
            edge_samples=int(wrong_config.get("edge_samples", 9)),
        )
        bbox_tasks = build_bbox_vision_tasks(
            scene_id=scene_id,
            frame_index=frame_index,
            bbox_records=bbox_records,
            projections_by_asset=bbox_projections,
        )
    return FrameProjectionResult(
        frame_index=frame_index,
        cluster_count=len(clusters),
        visible_roi_count=visible_roi_count,
        bbox_count=len(bbox_records),
        bbox_visible_roi_count=sum(len(rois) for rois in bbox_projections.values()),
        saved_images=saved_images,
        vision_tasks=vision_tasks + bbox_tasks,
        cluster_skipped_reason=cluster_skipped_reason,
    )
