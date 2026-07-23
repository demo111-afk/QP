"""
vehicle_dimension_config.py
车辆/设施尺寸参考库的读取与比对接口（Phase 1：只建立结构，不接入 Rule Engine）。

跟 rule_engine.py 完全独立——这一版任何函数都不会被 rule_engine.py 调用，
只是先把"以后规则要用的尺寸参考数据从哪读、怎么比对"这件事的接口定下来。
接入 Rule Engine 是后续阶段的事，不在本次改动范围内。

数据结构（config/vehicle_dimensions.yaml）：每个类别是 reference_size（经验参考尺寸）
+ tolerance（允许偏差比例），不是固定的 min/max。原因：项目目前只有各类别的经验参考尺寸，
没有统计意义上的最小值/最大值，编造 min/max 没有依据。允许范围在 check_dimension() 里
按 reference_size × (1 ± tolerance) 现算，不落盘存成固定数字——以后只改 tolerance，
所有判断自动跟着变，不需要碰 reference_size，也不需要改这里的代码。
"""

from __future__ import annotations

import yaml

DEFAULT_PATH = "config/vehicle_dimensions.yaml"

_AXES = ("length", "width", "height")


def load(path: str = DEFAULT_PATH) -> dict:
    """读取 vehicle_dimensions.yaml，返回
    {class_name: {id, note, size_basis, reference_size, tolerance}} 字典。"""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("classes", {}) or {}


def get_dimension(class_name: str, dimensions: dict) -> dict | None:
    """按类别名取这个类别的完整配置项（reference_size/tolerance/size_basis 等），
    找不到这个类别返回 None（区别于"类别存在但 reference_size 是 null"这种情况，
    后者要走 check_dimension() 里的 "no_reference" 分支，不能跟"类别压根不存在"混为一谈）。"""
    return dimensions.get(class_name)


def check_dimension(
    class_name: str,
    measured_lwh: tuple[float, float, float],
    dimensions: dict,
) -> dict:
    """比对实测尺寸（length, width, height）是否落在 reference_size × (1 ± tolerance)
    算出来的允许范围内——范围是现算的，不是读一个写死的 min/max。

    tolerance 直接从这个类别自己的配置里读（不是外部传入的固定参数），
    以后调整 config/vehicle_dimensions.yaml 里某个类别的 tolerance，
    这里的判断范围会自动跟着变，不需要改代码。

    reference_size 任一轴是 null（比如"其他车辆"/"组合车辆"这类"不设固定值"的类别）时
    返回 {"status": "no_reference", ...}，不判定通过/不通过，也不报错——
    调用方（未来的 Rule Engine 或别的模块）自己决定"没有参考数据"时怎么处理。
    """
    reference = get_dimension(class_name, dimensions)
    if reference is None:
        return {"status": "unknown_class", "class_name": class_name}

    ref_size = reference.get("reference_size") or {}
    tolerance = reference.get("tolerance") or {}
    ref_lwh = tuple(ref_size.get(axis) for axis in _AXES)
    tol_lwh = tuple(tolerance.get(axis) for axis in _AXES)

    if any(v is None for v in ref_lwh) or any(v is None for v in tol_lwh):
        return {
            "status": "no_reference",
            "class_name": class_name,
            "size_basis": reference.get("size_basis"),
        }

    diffs = {}
    within_tolerance = True
    for axis, measured, ref, tol in zip(_AXES, measured_lwh, ref_lwh, tol_lwh):
        low, high = ref * (1 - tol), ref * (1 + tol)
        in_range = low <= measured <= high
        diffs[axis] = {
            "measured": measured,
            "reference": ref,
            "tolerance": tol,
            "allowed_range": (low, high),
            "within_tolerance": in_range,
        }
        if not in_range:
            within_tolerance = False

    return {
        "status": "ok",
        "class_name": class_name,
        "size_basis": reference.get("size_basis"),
        "within_tolerance": within_tolerance,
        "diffs": diffs,
    }
