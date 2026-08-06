"""Deterministic post-model gates for conservative Vision reporting."""

from __future__ import annotations

from dataclasses import dataclass

import vehicle_dimension_config
from vision_category_config import VisionCategory


@dataclass(frozen=True)
class QualityGateResult:
    accepted: bool
    reason: str
    details: dict


def normalize_category(label: str, aliases: dict[str, str] | None = None) -> str:
    value = str(label or "").strip()
    return (aliases or {}).get(value, value)


def assess_category_size(
    category: str,
    measured_size: tuple[float, float, float] | None,
    categories: tuple[VisionCategory, ...],
    config: dict,
    dimensions: dict | None = None,
) -> QualityGateResult:
    """Reject model categories that are physically incompatible with Cluster size.

    X/Y orientation is unknown for residual AABBs, so horizontal dimensions are
    sorted before comparing them with each configured reference variant. The gate
    enforces generous lower/upper ratios. A residual fragment may be incomplete,
    but a fragment that is only a very thin slice of the selected reference object
    cannot support the claim that it is one independent, complete missing target.
    """
    if not config.get("enabled", True):
        return QualityGateResult(True, "disabled", {})
    if measured_size is None or len(measured_size) != 3:
        return QualityGateResult(False, "missing_measured_size", {})

    category_config = next((item for item in categories if item.name == category), None)
    if category_config is None:
        return QualityGateResult(False, "unknown_vision_category", {"category": category})

    dimension_db = dimensions if dimensions is not None else vehicle_dimension_config.load()
    measured_xy = sorted((float(measured_size[0]), float(measured_size[1])), reverse=True)
    measured = (measured_xy[0], measured_xy[1], float(measured_size[2]))
    max_axis_ratio = max(1.0, float(config.get("max_axis_ratio", 2.5)))
    min_axis_ratio = min(1.0, max(0.0, float(config.get("min_axis_ratio", 0.2))))
    checked = []

    for key in category_config.dimension_keys:
        entry = dimension_db.get(key) or {}
        reference = entry.get("reference_size") or {}
        values = tuple(reference.get(axis) for axis in ("length", "width", "height"))
        if any(value is None or float(value) <= 0 for value in values):
            continue
        reference_xy = sorted((float(values[0]), float(values[1])), reverse=True)
        normalized_reference = (reference_xy[0], reference_xy[1], float(values[2]))
        ratios = tuple(
            measured_value / reference_value
            for measured_value, reference_value in zip(measured, normalized_reference)
        )
        checked.append({"dimension_key": key, "ratios": ratios})
        if all(min_axis_ratio <= ratio <= max_axis_ratio for ratio in ratios):
            return QualityGateResult(True, "within_reference_multiplier", {
                "dimension_key": key,
                "ratios": ratios,
                "min_axis_ratio": min_axis_ratio,
                "max_axis_ratio": max_axis_ratio,
            })

    if not checked:
        return QualityGateResult(True, "no_fixed_reference", {"category": category})
    return QualityGateResult(False, "category_size_mismatch", {
        "category": category,
        "measured_size": measured,
        "min_axis_ratio": min_axis_ratio,
        "max_axis_ratio": max_axis_ratio,
        "checked_variants": checked,
    })
