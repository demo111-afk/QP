"""
cluster_projection.py
Projection/debug utilities for residual clusters.

This module is intentionally standalone. It does not modify or depend on the
Rule Engine, Browser Automation, or Cluster Detector internals beyond consuming
ClusterCandidate objects produced by cluster_detector.detect_clusters().
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from calibration import CalibrationManager, CameraCalibration
from cluster_detector import ClusterCandidate


@dataclass(frozen=True)
class ProjectedROI:
    camera_id: str
    location: str
    image_path: str | None
    cluster_id: str
    roi: tuple[int, int, int, int]
    visible_points: int
    total_points: int
    clipped: bool


@dataclass(frozen=True)
class AssetCameraMatch:
    camera: CameraCalibration
    image_path: Path
    asset_name: str


_ASSET_PREFIX = "cam_"


def asset_name_to_location(asset_stem: str) -> str:
    """Convert asset image stems such as cam_front_left to calibration locations.

    The current Assets Downloader names JPGs from URL camera names, lowercased.
    Calibration uses location values like front_left. Keep this conversion small
    and explicit so future naming differences can be handled in one place.
    """
    if asset_stem.startswith(_ASSET_PREFIX):
        return asset_stem[len(_ASSET_PREFIX):]
    return asset_stem


def build_asset_camera_matches(
    manager: CalibrationManager,
    images_dir: str | Path,
) -> list[AssetCameraMatch]:
    """Map available asset JPG files to calibration cameras.

    The calibration file can contain more cameras than the QP page currently
    downloads. Matching is therefore based on images that actually exist under
    frame_NNNN/images. If multiple calibration cameras share a location, prefer
    cameras with an extrinsic matrix and then the lowest numeric camera id.
    """
    base = Path(images_dir)
    if not base.is_dir():
        return []

    matches = []
    for image_path in sorted(base.glob("*.jpg")):
        location = asset_name_to_location(image_path.stem)
        candidates = list(manager.get_cameras_by_location(location))
        if not candidates:
            continue
        camera = sorted(candidates, key=_camera_match_sort_key)[0]
        matches.append(AssetCameraMatch(camera=camera, image_path=image_path, asset_name=image_path.stem))
    return matches


def project_cluster_to_camera(
    cluster: ClusterCandidate,
    camera: CameraCalibration,
    image_path: str | Path | None = None,
    min_depth: float = 0.1,
) -> ProjectedROI | None:
    """Project one cluster AABB hint to one camera and return a clipped ROI.

    This uses camera.extrinsic as the lidar/world-to-camera transform and camera
    intrinsic as the 3x3 pinhole matrix. Distortion is preserved in calibration
    data but intentionally not applied here; this stage creates the Projection
    infrastructure and debug ROI, not final pixel-perfect geometry.
    """
    if camera.extrinsic is None or camera.intrinsic is None or camera.image_size is None:
        return None
    if not cluster.bbox_hint:
        return None

    corners = _bbox_corners(cluster.bbox_hint)
    pixels, visible_count = project_points(corners, camera, min_depth=min_depth)
    if pixels.size == 0:
        return None

    width, height = camera.image_size.width, camera.image_size.height
    x_min, y_min = pixels.min(axis=0)
    x_max, y_max = pixels.max(axis=0)

    if x_max < 0 or y_max < 0 or x_min >= width or y_min >= height:
        return None

    clipped_x1 = int(max(0, min(width - 1, np.floor(x_min))))
    clipped_y1 = int(max(0, min(height - 1, np.floor(y_min))))
    clipped_x2 = int(max(0, min(width - 1, np.ceil(x_max))))
    clipped_y2 = int(max(0, min(height - 1, np.ceil(y_max))))
    if clipped_x2 <= clipped_x1 or clipped_y2 <= clipped_y1:
        return None

    clipped = (
        clipped_x1 != int(np.floor(x_min))
        or clipped_y1 != int(np.floor(y_min))
        or clipped_x2 != int(np.ceil(x_max))
        or clipped_y2 != int(np.ceil(y_max))
    )

    return ProjectedROI(
        camera_id=camera.camera_id,
        location=camera.location,
        image_path=str(image_path) if image_path is not None else None,
        cluster_id=cluster.cluster_id,
        roi=(clipped_x1, clipped_y1, clipped_x2, clipped_y2),
        visible_points=visible_count,
        total_points=len(corners),
        clipped=clipped,
    )


def project_points(
    points_xyz: np.ndarray,
    camera: CameraCalibration,
    min_depth: float = 0.1,
) -> tuple[np.ndarray, int]:
    """Project 3D points into one camera using the stored extrinsic/intrinsic."""
    if camera.extrinsic is None or camera.intrinsic is None:
        return np.empty((0, 2), dtype=np.float64), 0

    points = np.asarray(points_xyz, dtype=np.float64)
    homogeneous = np.column_stack([points, np.ones(points.shape[0], dtype=np.float64)])
    transform = np.asarray(camera.extrinsic.values, dtype=np.float64)
    camera_points = (transform @ homogeneous.T).T[:, :3]

    valid = camera_points[:, 2] > min_depth
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.float64), 0

    visible = camera_points[valid]
    intrinsic = np.asarray(camera.intrinsic.values, dtype=np.float64)
    projected = (intrinsic @ visible.T).T
    pixels = projected[:, :2] / projected[:, 2:3]
    return pixels, int(valid.sum())


def project_clusters_to_asset_cameras(
    clusters: list[ClusterCandidate],
    manager: CalibrationManager,
    images_dir: str | Path,
) -> dict[str, list[ProjectedROI]]:
    """Project clusters to every camera that has a matching asset image."""
    matches = build_asset_camera_matches(manager, images_dir)
    output: dict[str, list[ProjectedROI]] = {match.asset_name: [] for match in matches}
    for match in matches:
        for cluster in clusters:
            roi = project_cluster_to_camera(cluster, match.camera, match.image_path)
            if roi is not None:
                output[match.asset_name].append(roi)
    return output


def save_projection_debug_images(
    projections_by_asset: dict[str, list[ProjectedROI]],
    output_dir: str | Path,
) -> list[str]:
    """Draw ROI boxes on asset images and save one debug JPG per visible camera."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = []
    for asset_name, rois in projections_by_asset.items():
        visible_rois = [roi for roi in rois if roi.image_path]
        if not visible_rois:
            continue

        source = Path(visible_rois[0].image_path)
        with Image.open(source) as image:
            canvas = image.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()

        for index, roi in enumerate(visible_rois):
            color = _debug_color(index)
            draw.rectangle(roi.roi, outline=color, width=4)
            label = f"{roi.cluster_id} {roi.camera_id}"
            text_xy = (roi.roi[0] + 4, max(0, roi.roi[1] - 14))
            draw.text(text_xy, label, fill=color, font=font)

        target = out / f"projection_debug_{asset_name}.jpg"
        canvas.save(target, quality=92)
        saved.append(str(target))
    return saved


def _bbox_corners(bbox_hint: dict) -> np.ndarray:
    mins = np.asarray(bbox_hint["min"], dtype=np.float64)
    maxs = np.asarray(bbox_hint["max"], dtype=np.float64)
    return np.array([
        [mins[0], mins[1], mins[2]],
        [mins[0], mins[1], maxs[2]],
        [mins[0], maxs[1], mins[2]],
        [mins[0], maxs[1], maxs[2]],
        [maxs[0], mins[1], mins[2]],
        [maxs[0], mins[1], maxs[2]],
        [maxs[0], maxs[1], mins[2]],
        [maxs[0], maxs[1], maxs[2]],
    ], dtype=np.float64)


def _camera_match_sort_key(camera: CameraCalibration) -> tuple[int, int, str]:
    has_no_extrinsic = 0 if camera.extrinsic is not None else 1
    numeric = _camera_numeric_suffix(camera.camera_id)
    return (has_no_extrinsic, numeric, camera.camera_id)


def _camera_numeric_suffix(camera_id: str) -> int:
    match = __import__("re").search(r"(\d+)$", camera_id)
    return int(match.group(1)) if match else 10**9


def _debug_color(index: int) -> tuple[int, int, int]:
    palette = [
        (255, 64, 64),
        (64, 200, 255),
        (64, 255, 128),
        (255, 196, 64),
        (220, 128, 255),
        (255, 128, 192),
    ]
    return palette[index % len(palette)]
