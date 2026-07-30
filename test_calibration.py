"""
test_calibration.py
Independent tests for calibration.py. No pytest required.

Usage:
    python test_calibration.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from calibration import CalibrationError, load_calibration


def test_load_current_wps_calibration():
    manager = load_calibration("cali.wps")
    cameras = manager.get_all_cameras()
    assert len(cameras) == 11, len(cameras)
    assert manager.has_camera("camera0")
    assert manager.has_camera("camera11")

    camera0 = manager.get_camera("camera0")
    assert camera0.location == "front_mid"
    assert camera0.image_size.width == 1920
    assert camera0.image_size.height == 1536
    assert camera0.intrinsic is not None
    assert len(camera0.intrinsic.flat()) == 9
    assert camera0.extrinsic is not None
    assert len(camera0.extrinsic.flat()) == 16
    assert camera0.distortion is not None
    assert len(camera0.distortion.coefficients) == 8


def test_empty_extrinsic_is_preserved_as_none():
    manager = load_calibration("cali.wps")
    camera10 = manager.get_camera("camera10")
    camera11 = manager.get_camera("camera11")
    assert camera10.extrinsic is None
    assert camera11.extrinsic is None


def test_location_is_not_unique():
    manager = load_calibration("cali.wps")
    front_mid = manager.get_cameras_by_location("front_mid")
    rear_left = manager.get_cameras_by_location("rear_left")
    assert {camera.camera_id for camera in front_mid} == {"camera0", "camera9"}
    assert {camera.camera_id for camera in rear_left} == {"camera2", "camera8"}


def test_extra_extrinsic_fields_are_loaded():
    manager = load_calibration("cali.wps")
    camera2 = manager.get_camera("camera2")
    camera6 = manager.get_camera("camera6")
    assert "extrinsic4" in camera2.extra_extrinsics
    assert len(camera2.extra_extrinsics["extrinsic4"].flat()) == 16
    assert "extrinsic4" in camera6.extra_extrinsics


def test_load_plain_yaml_file():
    yaml_text = """
hardware:
  sensors:
    camera_labels:
      - cam_a
    cameras:
      cam_a:
        device: "device-a"
        location: "front"
        type: "test-camera"
        image_size: [640, 480]
        intrinsic: [1, 0, 320, 0, 1, 240, 0, 0, 1]
        distort: [0, 0, 0, 0, 0]
        distortion_type: 1
        extrinsic: []
        default: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "calibration.yaml"
        path.write_text(yaml_text, encoding="utf-8")
        manager = load_calibration(path)

    cam = manager.get_camera("cam_a")
    assert cam.location == "front"
    assert cam.extrinsic is None
    assert cam.default_transform is not None
    assert cam.image_size.width == 640
    assert cam.image_size.height == 480


def test_invalid_intrinsic_length_raises():
    yaml_text = """
hardware:
  sensors:
    camera_labels: [cam_a]
    cameras:
      cam_a:
        image_size: [640, 480]
        intrinsic: [1, 2, 3]
"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad.yaml"
        path.write_text(yaml_text, encoding="utf-8")
        try:
            load_calibration(path)
        except CalibrationError as exc:
            assert "intrinsic" in str(exc)
        else:
            raise AssertionError("Expected CalibrationError")


def main() -> None:
    tests = [
        test_load_current_wps_calibration,
        test_empty_extrinsic_is_preserved_as_none,
        test_location_is_not_unique,
        test_extra_extrinsic_fields_are_loaded,
        test_load_plain_yaml_file,
        test_invalid_intrinsic_length_raises,
    ]
    for test in tests:
        test()
        print(f"[通过] {test.__name__}")
    print(f"\n全部 {len(tests)} 个测试通过。")


if __name__ == "__main__":
    main()
