"""
test_cluster_projection.py
Standalone tests for cluster_projection.py. No pytest required.

Usage:
    .venv/bin/python test_cluster_projection.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from calibration import (
    CalibrationManager,
    CameraCalibration,
    DistortionParameters,
    ImageSize,
    Matrix3x3,
    Matrix3x4,
    Matrix4x4,
)
from cluster_detector import ClusterCandidate
from demo_cluster_projection import make_synthetic_cluster_for_camera
from cluster_projection import (
    build_asset_camera_matches,
    project_cluster_to_camera,
    project_clusters_to_asset_cameras,
    save_projection_debug_images,
)


def _camera(camera_id="camera0", location="front_mid", extrinsic=True):
    return CameraCalibration(
        camera_id=camera_id,
        location=location,
        image_size=ImageSize(width=640, height=480),
        intrinsic=Matrix3x3.from_flat([100, 0, 320, 0, 100, 240, 0, 0, 1], "intrinsic"),
        distortion=DistortionParameters(distortion_type=0, coefficients=()),
        extrinsic=Matrix4x4.from_flat([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], "extrinsic") if extrinsic else None,
        default_transform=Matrix3x4.from_flat([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], "default"),
    )


def _cluster(min_xyz=(-1, -1, 9), max_xyz=(1, 1, 11)):
    return ClusterCandidate(
        cluster_id="cluster_0",
        point_indices=list(range(8)),
        centroid=(0, 0, 10),
        point_count=8,
        bbox_hint={"min": min_xyz, "max": max_xyz, "size": tuple(max_xyz[i] - min_xyz[i] for i in range(3))},
        status="Possible Missing Annotation",
    )


def test_project_cluster_to_camera_returns_roi():
    roi = project_cluster_to_camera(_cluster(), _camera())
    assert roi is not None
    x1, y1, x2, y2 = roi.roi
    assert 250 <= x1 < 320
    assert 320 < x2 <= 390
    assert 170 <= y1 < 240
    assert 240 < y2 <= 310
    assert roi.visible_points == 8


def test_cluster_behind_camera_is_not_visible():
    roi = project_cluster_to_camera(_cluster(min_xyz=(-1, -1, -11), max_xyz=(1, 1, -9)), _camera())
    assert roi is None


def test_asset_camera_matching_prefers_camera_with_extrinsic_then_low_id():
    manager = CalibrationManager(cameras=(
        _camera("camera9", "front_mid", extrinsic=False),
        _camera("camera0", "front_mid", extrinsic=True),
        _camera("camera1", "front_left", extrinsic=True),
    ))
    with tempfile.TemporaryDirectory() as tmp:
        images = Path(tmp)
        Image.new("RGB", (640, 480)).save(images / "cam_front_mid.jpg")
        Image.new("RGB", (640, 480)).save(images / "cam_front_left.jpg")
        matches = build_asset_camera_matches(manager, images)

    mapping = {m.asset_name: m.camera.camera_id for m in matches}
    assert mapping["cam_front_mid"] == "camera0"
    assert mapping["cam_front_left"] == "camera1"



def test_synthetic_cluster_projects_to_source_camera():
    camera = _camera("camera0", "front_mid", extrinsic=True)
    cluster = make_synthetic_cluster_for_camera(camera)
    roi = project_cluster_to_camera(cluster, camera)
    assert roi is not None
    x1, y1, x2, y2 = roi.roi
    assert x1 < 320 < x2
    assert y1 < 240 < y2


def test_projection_roi_scales_to_actual_asset_image_size():
    with tempfile.TemporaryDirectory() as tmp:
        image_path = Path(tmp) / "cam_front_mid.jpg"
        Image.new("RGB", (320, 240), color=(10, 10, 10)).save(image_path)
        roi = project_cluster_to_camera(_cluster(), _camera(), image_path)

    assert roi is not None
    assert roi.calibration_image_size == (640, 480)
    assert roi.target_image_size == (320, 240)
    x1, y1, x2, y2 = roi.roi
    assert 125 <= x1 < 160
    assert 160 < x2 <= 195
    assert 85 <= y1 < 120
    assert 120 < y2 <= 155


def test_save_projection_debug_images():
    manager = CalibrationManager(cameras=(_camera("camera0", "front_mid", extrinsic=True),))
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        images = base / "images"
        images.mkdir()
        Image.new("RGB", (640, 480), color=(10, 10, 10)).save(images / "cam_front_mid.jpg")
        projections = project_clusters_to_asset_cameras([_cluster()], manager, images)
        saved = save_projection_debug_images(projections, base / "projection_debug")
        assert len(saved) == 1
        assert Path(saved[0]).is_file()
        assert Image.open(saved[0]).size == (640, 480)


def main() -> None:
    tests = [
        test_project_cluster_to_camera_returns_roi,
        test_cluster_behind_camera_is_not_visible,
        test_asset_camera_matching_prefers_camera_with_extrinsic_then_low_id,
        test_synthetic_cluster_projects_to_source_camera,
        test_projection_roi_scales_to_actual_asset_image_size,
        test_save_projection_debug_images,
    ]
    for test in tests:
        test()
        print(f"[通过] {test.__name__}")
    print(f"\n全部 {len(tests)} 个测试通过。")


if __name__ == "__main__":
    main()
