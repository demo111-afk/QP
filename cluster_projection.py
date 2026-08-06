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
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from calibration import CalibrationManager, CameraCalibration
from cluster_detector import ClusterCandidate, OrientedBox


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
    calibration_image_size: tuple[int, int] | None = None
    target_image_size: tuple[int, int] | None = None

    @property
    def candidate_id(self) -> str:
        """Generic name; cluster_id remains for report/debug compatibility."""
        return self.cluster_id


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
    min_visible_points: int = 3,
    padding_px: int = 8,
) -> ProjectedROI | None:
    """Project one cluster's actual points and return a clipped image ROI.

    The observed calibration stores camera.extrinsic as camera-to-lidar/world,
    so projection uses its inverse as the lidar/world-to-camera transform. The
    camera intrinsic is then applied as the 3x3 pinhole matrix. Distortion is
    preserved in calibration data but intentionally not applied because the
    downloaded QP JPGs are already rectified.
    """
    projection_points = _cluster_projection_points(cluster)
    return project_geometry_to_camera(
        candidate_id=cluster.cluster_id,
        points_xyz=projection_points,
        camera=camera,
        image_path=image_path,
        min_depth=min_depth,
        min_visible_points=min_visible_points,
        padding_px=padding_px,
    )


def project_geometry_to_camera(
    candidate_id: str,
    points_xyz: np.ndarray,
    camera: CameraCalibration,
    image_path: str | Path | None = None,
    min_depth: float = 0.1,
    min_visible_points: int = 3,
    padding_px: int = 8,
) -> ProjectedROI | None:
    """Project arbitrary 3D evidence with the existing camera math."""
    if camera.extrinsic is None or camera.intrinsic is None or camera.image_size is None:
        return None

    projection_points = np.asarray(points_xyz, dtype=np.float64)
    if projection_points.ndim != 2 or projection_points.shape[1] < 3 or projection_points.size == 0:
        return None

    pixels, _ = project_points(projection_points[:, :3], camera, min_depth=min_depth)
    if pixels.size == 0:
        return None

    calibration_size = (camera.image_size.width, camera.image_size.height)
    target_size = _target_image_size(image_path, calibration_size)
    pixels = _scale_pixels_to_target_image(pixels, calibration_size, target_size)
    width, height = target_size
    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    visible_pixels = pixels[inside]
    required_points = max(1, min_visible_points)
    if len(visible_pixels) < required_points:
        return None

    x_min, y_min = visible_pixels.min(axis=0)
    x_max, y_max = visible_pixels.max(axis=0)
    raw_x1 = int(np.floor(x_min)) - max(0, padding_px)
    raw_y1 = int(np.floor(y_min)) - max(0, padding_px)
    raw_x2 = int(np.ceil(x_max)) + max(0, padding_px)
    raw_y2 = int(np.ceil(y_max)) + max(0, padding_px)

    clipped_x1 = max(0, min(width - 1, raw_x1))
    clipped_y1 = max(0, min(height - 1, raw_y1))
    clipped_x2 = max(0, min(width - 1, raw_x2))
    clipped_y2 = max(0, min(height - 1, raw_y2))
    if clipped_x2 <= clipped_x1 or clipped_y2 <= clipped_y1:
        return None

    clipped = (clipped_x1, clipped_y1, clipped_x2, clipped_y2) != (raw_x1, raw_y1, raw_x2, raw_y2)

    return ProjectedROI(
        camera_id=camera.camera_id,
        location=camera.location,
        image_path=str(image_path) if image_path is not None else None,
        cluster_id=str(candidate_id),
        roi=(clipped_x1, clipped_y1, clipped_x2, clipped_y2),
        visible_points=len(visible_pixels),
        total_points=len(projection_points),
        clipped=clipped,
        calibration_image_size=calibration_size,
        target_image_size=target_size,
    )


def project_oriented_boxes_to_asset_cameras(
    candidates: Iterable[tuple[str, OrientedBox]],
    manager: CalibrationManager,
    images_dir: str | Path,
    min_depth: float = 0.1,
    min_visible_points: int = 3,
    padding_px: int = 8,
    edge_samples: int = 9,
) -> dict[str, list[ProjectedROI]]:
    """Project Existing BBoxes through the same camera projection implementation."""
    matches = build_asset_camera_matches(manager, images_dir)
    output: dict[str, list[ProjectedROI]] = {match.asset_name: [] for match in matches}
    geometry = [
        (candidate_id, oriented_box_edge_points(box, edge_samples=edge_samples))
        for candidate_id, box in candidates
    ]
    for match in matches:
        for candidate_id, points in geometry:
            roi = project_geometry_to_camera(
                candidate_id,
                points,
                match.camera,
                match.image_path,
                min_depth=min_depth,
                min_visible_points=min_visible_points,
                padding_px=padding_px,
            )
            if roi is not None:
                output[match.asset_name].append(roi)
    return output


def oriented_box_edge_points(box: OrientedBox, edge_samples: int = 9) -> np.ndarray:
    """Sample all cuboid edges in world coordinates using Three.js XYZ Euler order."""
    half = np.asarray(box.size, dtype=np.float64) / 2.0
    corners = np.array([
        [sx * half[0], sy * half[1], sz * half[2]]
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ], dtype=np.float64)
    sample_count = max(2, int(edge_samples))
    samples = []
    for index, corner in enumerate(corners):
        for axis in range(3):
            neighbor = index ^ (1 << (2 - axis))
            if index < neighbor:
                weights = np.linspace(0.0, 1.0, sample_count)[:, None]
                samples.append(corner + weights * (corners[neighbor] - corner))
    local_points = np.vstack(samples)
    rotation = _euler_xyz_to_matrix(*box.rotation_euler)
    return local_points @ rotation.T + np.asarray(box.center, dtype=np.float64)


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
    camera_to_world = np.asarray(camera.extrinsic.values, dtype=np.float64)
    try:
        world_to_camera = np.linalg.inv(camera_to_world)
    except np.linalg.LinAlgError:
        return np.empty((0, 2), dtype=np.float64), 0
    camera_points = (world_to_camera @ homogeneous.T).T[:, :3]

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
    min_depth: float = 0.1,
    min_visible_points: int = 3,
    padding_px: int = 8,
) -> dict[str, list[ProjectedROI]]:
    """Project clusters to every camera that has a matching asset image."""
    matches = build_asset_camera_matches(manager, images_dir)
    output: dict[str, list[ProjectedROI]] = {match.asset_name: [] for match in matches}
    for match in matches:
        for cluster in clusters:
            roi = project_cluster_to_camera(
                cluster,
                match.camera,
                match.image_path,
                min_depth=min_depth,
                min_visible_points=min_visible_points,
                padding_px=padding_px,
            )
            if roi is not None:
                output[match.asset_name].append(roi)
    return output


def save_projection_debug_images(
    projections_by_asset: dict[str, list[ProjectedROI]],
    output_dir: str | Path,
    line_width: int = 1,
) -> list[str]:
    """Draw ROI boxes on asset images and save one debug JPG per visible camera."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stale_path in out.glob("projection_debug_*.jpg"):
        stale_path.unlink()

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

        draw_width = max(1, int(line_width))
        for index, roi in enumerate(visible_rois):
            color = _debug_color(index)
            draw.rectangle(roi.roi, outline=color, width=draw_width)
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


def _euler_xyz_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Match the Three.js XYZ convention used by Cluster point removal."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rot_x @ rot_y @ rot_z


def _cluster_projection_points(cluster: ClusterCandidate) -> np.ndarray:
    if cluster.points_xyz is not None:
        points = np.asarray(cluster.points_xyz, dtype=np.float64)
        if points.ndim == 2 and points.shape[1] >= 3:
            return points[:, :3]
    if cluster.bbox_hint:
        return _bbox_corners(cluster.bbox_hint)
    return np.empty((0, 3), dtype=np.float64)


def _camera_match_sort_key(camera: CameraCalibration) -> tuple[int, int, str]:
    has_no_extrinsic = 0 if camera.extrinsic is not None else 1
    numeric = _camera_numeric_suffix(camera.camera_id)
    return (has_no_extrinsic, numeric, camera.camera_id)


def _camera_numeric_suffix(camera_id: str) -> int:
    match = __import__("re").search(r"(\d+)$", camera_id)
    return int(match.group(1)) if match else 10**9


def _target_image_size(image_path: str | Path | None, calibration_size: tuple[int, int]) -> tuple[int, int]:
    if image_path is None:
        return calibration_size
    try:
        with Image.open(image_path) as image:
            return image.size
    except OSError:
        return calibration_size


def _scale_pixels_to_target_image(
    pixels: np.ndarray,
    calibration_size: tuple[int, int],
    target_size: tuple[int, int],
) -> np.ndarray:
    if calibration_size == target_size:
        return pixels

    calibration_width, calibration_height = calibration_size
    target_width, target_height = target_size
    if calibration_width <= 0 or calibration_height <= 0:
        return pixels

    scaled = pixels.copy()
    scaled[:, 0] *= target_width / calibration_width
    scaled[:, 1] *= target_height / calibration_height
    return scaled


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
