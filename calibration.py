"""
calibration.py
Phase 1 calibration infrastructure.

This module only loads and structures calibration data. It does not implement
projection, cropping, vision, browser automation, rule checks, or cluster logic.

Supported inputs:
- Plain YAML files (.yaml / .yml)
- Current WPS export (.wps) that contains an embedded UTF-16LE YAML block

The stable identity for a camera is the YAML camera key such as "camera0".
The "location" field is not unique in the observed calibration file, so it is
exposed as metadata and can return multiple cameras.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml


class CalibrationError(ValueError):
    """Raised when calibration content is missing or structurally invalid."""


@dataclass(frozen=True)
class ImageSize:
    width: int
    height: int

    @classmethod
    def from_list(cls, values: list[Any], field_name: str) -> "ImageSize":
        if len(values) != 2:
            raise CalibrationError(f"{field_name} must contain 2 values, got {len(values)}")
        return cls(width=int(values[0]), height=int(values[1]))


@dataclass(frozen=True)
class Matrix3x3:
    values: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]

    @classmethod
    def from_flat(cls, values: list[Any], field_name: str) -> "Matrix3x3":
        floats = _expect_float_list(values, 9, field_name)
        return cls(values=(
            (floats[0], floats[1], floats[2]),
            (floats[3], floats[4], floats[5]),
            (floats[6], floats[7], floats[8]),
        ))

    def flat(self) -> tuple[float, ...]:
        return tuple(v for row in self.values for v in row)


@dataclass(frozen=True)
class Matrix3x4:
    values: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ]

    @classmethod
    def from_flat(cls, values: list[Any], field_name: str) -> "Matrix3x4":
        floats = _expect_float_list(values, 12, field_name)
        return cls(values=(
            (floats[0], floats[1], floats[2], floats[3]),
            (floats[4], floats[5], floats[6], floats[7]),
            (floats[8], floats[9], floats[10], floats[11]),
        ))

    def flat(self) -> tuple[float, ...]:
        return tuple(v for row in self.values for v in row)


@dataclass(frozen=True)
class Matrix4x4:
    values: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ]

    @classmethod
    def from_flat(cls, values: list[Any], field_name: str) -> "Matrix4x4":
        floats = _expect_float_list(values, 16, field_name)
        return cls(values=(
            (floats[0], floats[1], floats[2], floats[3]),
            (floats[4], floats[5], floats[6], floats[7]),
            (floats[8], floats[9], floats[10], floats[11]),
            (floats[12], floats[13], floats[14], floats[15]),
        ))

    def flat(self) -> tuple[float, ...]:
        return tuple(v for row in self.values for v in row)


@dataclass(frozen=True)
class DistortionParameters:
    distortion_type: int | None
    coefficients: tuple[float, ...]


@dataclass(frozen=True)
class CameraCalibration:
    camera_id: str
    location: str = ""
    device: str = ""
    camera_type: str = ""
    image_size: ImageSize | None = None
    intrinsic: Matrix3x3 | None = None
    distortion: DistortionParameters | None = None
    extrinsic: Matrix4x4 | None = None
    default_transform: Matrix3x4 | None = None
    extra_extrinsics: dict[str, Matrix4x4] = field(default_factory=dict)
    raw_extra: dict[str, Any] = field(default_factory=dict)

    def has_extrinsic(self) -> bool:
        return self.extrinsic is not None


@dataclass(frozen=True)
class LidarCalibration:
    lidar_id: str
    extrinsic_matrix: Matrix4x4 | Matrix3x4 | None = None
    default_extrinsic_matrix: Matrix4x4 | Matrix3x4 | None = None
    raw_extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CalibrationManager:
    cameras: tuple[CameraCalibration, ...]
    lidars: tuple[LidarCalibration, ...] = ()
    raw_top_level_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        camera_ids = [c.camera_id for c in self.cameras]
        if len(camera_ids) != len(set(camera_ids)):
            raise CalibrationError("camera_id values must be unique")

    def get_camera(self, name: str) -> CameraCalibration:
        for camera in self.cameras:
            if camera.camera_id == name:
                return camera
        raise KeyError(f"Camera not found: {name}")

    def has_camera(self, name: str) -> bool:
        return any(camera.camera_id == name for camera in self.cameras)

    def get_all_cameras(self) -> tuple[CameraCalibration, ...]:
        return self.cameras

    def get_cameras_by_location(self, location: str) -> tuple[CameraCalibration, ...]:
        return tuple(camera for camera in self.cameras if camera.location == location)

    def get_lidar(self, name: str) -> LidarCalibration:
        for lidar in self.lidars:
            if lidar.lidar_id == name:
                return lidar
        raise KeyError(f"Lidar not found: {name}")

    def get_all_lidars(self) -> tuple[LidarCalibration, ...]:
        return self.lidars

    def summary(self) -> str:
        lines: list[str] = []
        for camera in self.cameras:
            lines.append("=" * 60)
            lines.append(f"Camera: {camera.camera_id}")
            lines.append(f"Location: {camera.location or '-'}")
            lines.append(f"Device: {camera.device or '-'}")
            lines.append(f"Type: {camera.camera_type or '-'}")
            if camera.image_size:
                lines.append(f"Image Size: {camera.image_size.width} x {camera.image_size.height}")
            else:
                lines.append("Image Size: -")
            lines.append("Intrinsic:")
            lines.extend(_format_matrix(camera.intrinsic.values if camera.intrinsic else None))
            if camera.distortion:
                lines.append(f"Distortion Type: {camera.distortion.distortion_type}")
                lines.append(f"Distortion: {list(camera.distortion.coefficients)}")
            else:
                lines.append("Distortion: -")
            lines.append("Extrinsic:")
            lines.extend(_format_matrix(camera.extrinsic.values if camera.extrinsic else None))
            lines.append("Default Transform:")
            lines.extend(_format_matrix(camera.default_transform.values if camera.default_transform else None))
            if camera.extra_extrinsics:
                lines.append(f"Extra Extrinsics: {', '.join(sorted(camera.extra_extrinsics))}")
            else:
                lines.append("Extra Extrinsics: -")
        if not self.cameras:
            lines.append("No cameras loaded.")
        lines.append("=" * 60)
        return "\n".join(lines)


class CalibrationLoader:
    """Loads calibration YAML/WPS content into structured calibration objects."""

    _WPS_TEXT_RE = re.compile(rb"(?:[\x09\x0a\x0d\x20-\x7e]\x00){4,}")

    @classmethod
    def load(cls, path: str | Path) -> CalibrationManager:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)

        if source.suffix.lower() == ".wps":
            text = cls._extract_yaml_from_wps(source)
        else:
            text = source.read_text(encoding="utf-8")

        return cls.loads(text)

    @classmethod
    def loads(cls, text: str) -> CalibrationManager:
        """Load pasted YAML text into the same structured objects as file loading."""
        if not text or not text.strip():
            raise CalibrationError("Calibration YAML is empty")
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise CalibrationError(f"Invalid Calibration YAML: {exc}") from exc
        if not isinstance(data, dict):
            raise CalibrationError("Calibration YAML root must be a mapping")
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationManager:
        hardware = data.get("hardware") or {}
        if not isinstance(hardware, dict):
            raise CalibrationError("hardware must be a mapping")
        sensors = hardware.get("sensors") or {}
        if not isinstance(sensors, dict):
            raise CalibrationError("hardware.sensors must be a mapping")
        cameras_raw = sensors.get("cameras") or {}
        if not isinstance(cameras_raw, dict):
            raise CalibrationError("hardware.sensors.cameras must be a mapping")
        camera_labels = sensors.get("camera_labels") or list(cameras_raw.keys())
        if not isinstance(camera_labels, list):
            raise CalibrationError("hardware.sensors.camera_labels must be a list")

        cameras = []
        for camera_id in camera_labels:
            raw = cameras_raw.get(camera_id)
            if raw is None:
                raise CalibrationError(f"camera_labels contains {camera_id!r}, but cameras has no matching entry")
            cameras.append(cls._parse_camera(str(camera_id), raw))

        # Preserve additional camera entries even if camera_labels omitted them.
        labeled = set(camera_labels)
        for camera_id, raw in cameras_raw.items():
            if camera_id not in labeled:
                cameras.append(cls._parse_camera(str(camera_id), raw))

        lidars_raw = sensors.get("lidars") or {}
        if not isinstance(lidars_raw, dict):
            raise CalibrationError("hardware.sensors.lidars must be a mapping")
        lidars = tuple(cls._parse_lidar(str(lidar_id), raw) for lidar_id, raw in lidars_raw.items())

        return CalibrationManager(
            cameras=tuple(cameras),
            lidars=lidars,
            raw_top_level_keys=tuple(data.keys()),
        )

    @classmethod
    def _extract_yaml_from_wps(cls, path: Path) -> str:
        content = path.read_bytes()
        chunks = [m.group(0).decode("utf-16le", errors="ignore") for m in cls._WPS_TEXT_RE.finditer(content)]
        text = "\n".join(chunks).replace("\r", "\n")

        start = text.find("hardware:")
        if start == -1:
            raise CalibrationError(f"Could not find embedded YAML starting with 'hardware:' in {path}")

        end_markers = ["\nCalibri\n", "\nDejaVu Sans\n", "\nRoot Entry\n", "\nWordDocument\n"]
        end_candidates = [idx for marker in end_markers if (idx := text.find(marker, start)) != -1]
        end = min(end_candidates) if end_candidates else len(text)
        yaml_text = text[start:end].strip() + "\n"

        if "cameras:" not in yaml_text:
            raise CalibrationError(f"Embedded YAML in {path} does not contain a cameras section")
        return yaml_text

    @classmethod
    def _parse_camera(cls, camera_id: str, raw: dict[str, Any]) -> CameraCalibration:
        known = {
            "default", "device", "distort", "distortion_type", "extrinsic", "image_size",
            "intrinsic", "location", "type",
        }

        image_size = None
        if "image_size" in raw:
            image_size = ImageSize.from_list(_as_list(raw["image_size"], f"{camera_id}.image_size"), f"{camera_id}.image_size")

        intrinsic = None
        if "intrinsic" in raw:
            intrinsic = Matrix3x3.from_flat(_as_list(raw["intrinsic"], f"{camera_id}.intrinsic"), f"{camera_id}.intrinsic")

        distortion = None
        if "distort" in raw or "distortion_type" in raw:
            coefficients = tuple(float(v) for v in _as_list(raw.get("distort") or [], f"{camera_id}.distort"))
            distortion = DistortionParameters(
                distortion_type=raw.get("distortion_type"),
                coefficients=coefficients,
            )

        extrinsic = _optional_matrix4x4(raw.get("extrinsic") or [], f"{camera_id}.extrinsic")
        default_transform = _optional_matrix3x4(raw.get("default") or [], f"{camera_id}.default")

        extra_extrinsics = {}
        for key, value in raw.items():
            if key.startswith("extrinsic") and key != "extrinsic" and value:
                extra_extrinsics[key] = Matrix4x4.from_flat(_as_list(value, f"{camera_id}.{key}"), f"{camera_id}.{key}")

        raw_extra = {key: value for key, value in raw.items() if key not in known and key not in extra_extrinsics}

        return CameraCalibration(
            camera_id=camera_id,
            location=str(raw.get("location") or ""),
            device=str(raw.get("device") or ""),
            camera_type=str(raw.get("type") or ""),
            image_size=image_size,
            intrinsic=intrinsic,
            distortion=distortion,
            extrinsic=extrinsic,
            default_transform=default_transform,
            extra_extrinsics=extra_extrinsics,
            raw_extra=raw_extra,
        )

    @classmethod
    def _parse_lidar(cls, lidar_id: str, raw: dict[str, Any]) -> LidarCalibration:
        known = {"extrinct_matrix", "default_extrinct_matrix"}
        return LidarCalibration(
            lidar_id=lidar_id,
            extrinsic_matrix=_optional_matrix(raw.get("extrinct_matrix") or [], f"{lidar_id}.extrinct_matrix"),
            default_extrinsic_matrix=_optional_matrix(
                raw.get("default_extrinct_matrix") or [], f"{lidar_id}.default_extrinct_matrix"
            ),
            raw_extra={key: value for key, value in raw.items() if key not in known},
        )


def load_calibration(path: str | Path) -> CalibrationManager:
    """Load calibration data from a YAML/YML file or the current WPS export."""
    return CalibrationLoader.load(path)


def load_calibration_text(text: str) -> CalibrationManager:
    """Load Calibration YAML pasted by the UI."""
    return CalibrationLoader.loads(text)


def _expect_float_list(values: list[Any], expected_len: int, field_name: str) -> tuple[float, ...]:
    values = _as_list(values, field_name)
    if len(values) != expected_len:
        raise CalibrationError(f"{field_name} must contain {expected_len} values, got {len(values)}")
    return tuple(float(v) for v in values)


def _as_list(value: Any, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise CalibrationError(f"{field_name} must be a list")
    return value


def _optional_matrix4x4(values: list[Any], field_name: str) -> Matrix4x4 | None:
    if not values:
        return None
    return Matrix4x4.from_flat(values, field_name)


def _optional_matrix3x4(values: list[Any], field_name: str) -> Matrix3x4 | None:
    if not values:
        return None
    return Matrix3x4.from_flat(values, field_name)


def _optional_matrix(values: list[Any], field_name: str) -> Matrix4x4 | Matrix3x4 | None:
    if not values:
        return None
    if len(values) == 16:
        return Matrix4x4.from_flat(values, field_name)
    if len(values) == 12:
        return Matrix3x4.from_flat(values, field_name)
    raise CalibrationError(f"{field_name} must contain either 12 or 16 values, got {len(values)}")


def _format_matrix(rows: Iterable[Iterable[float]] | None) -> list[str]:
    if rows is None:
        return ["  -"]
    return ["  " + "[" + ", ".join(f"{value:.9g}" for value in row) + "]" for row in rows]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print a Calibration YAML/WPS summary.")
    parser.add_argument("calibration", help="Calibration YAML/YML/WPS path")
    args = parser.parse_args()
    manager = load_calibration(args.calibration)
    print(manager.summary())
