"""
cluster_detector.py
点云聚类候选检测——Phase 2：漏标（Missing Annotation）检测的核心模块。

流程（本模块负责 PCD 读取 -> BBox 内部点剔除 -> DBSCAN -> 噪声过滤这几步；
调用方 rule_engine.py 负责把结果写进 rule_report.csv，两者职责分开）：

    PCD
     |
     v
    读取已有 BBox（调用方从 bbox_data.csv 转换成 OrientedBox，本模块不碰 CSV）
     |
     v
    remove_bbox_points()：删除落在任一 BBox 内部的点
     |
     v
    Residual Point Cloud（剩余点云）
     |
     v
    DBSCAN（sklearn.cluster.DBSCAN，只用 x/y/z，不用 intensity）
     |
     v
    过滤噪声（点数 / AABB 体积）
     |
     v
    ClusterCandidate 列表（status 固定是 "Possible Missing Annotation" 或
    "Suspicious Residual Cluster"，不判断类别、不调用 Vision/LLM）

跟 bbox_extractor.py 完全无关：BBox 的读取/保存逻辑保持现状，不会被这里调用或修改。
本模块消费的是 assets_downloader.py 下载下来的 PCD 文件（本地路径），不重新连浏览器、
不重新监听网络。

`OrientedBox` 是本模块自己的小结构（center/rotation_euler/size），不直接依赖
bbox_data.csv 的字符串/字典格式——调用方负责把 CSV 行转换成 OrientedBox，这样本模块
可以被其它调用方复用（比如以后接 Vision AI 时直接喂 PCD + BBox，不需要经过 CSV）。
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import DBSCAN

# PCD FIELDS 的 (TYPE, SIZE) -> numpy dtype 映射，ascii/binary/binary_compressed 三种模式共用。
_PCD_TYPE_MAP = {
    ("F", 4): np.float32, ("F", 8): np.float64,
    ("U", 1): np.uint8, ("U", 2): np.uint16, ("U", 4): np.uint32,
    ("I", 1): np.int8, ("I", 2): np.int16, ("I", 4): np.int32,
}


def _lzf_decompress(data: bytes, expected_size: int) -> bytes:
    """liblzf 解压（PCD 的 binary_compressed 模式用的就是这个格式）。

    格式：一串控制字节，ctrl < 32 表示接下来 ctrl+1 个字节是原样字面量；
    ctrl >= 32 表示一次回溯复制（长度、偏移量从 ctrl 和后续 1~2 个字节算出来）。
    这是标准、公开的压缩格式，没有现成的纯 Python 库可以直接复用，自己实现。
    """
    out = bytearray()
    i, n = 0, len(data)
    while i < n:
        ctrl = data[i]
        i += 1
        if ctrl < 32:
            run_len = ctrl + 1
            out += data[i:i + run_len]
            i += run_len
        else:
            length = ctrl >> 5
            if length == 7:
                length += data[i]
                i += 1
            ref_offset = ((ctrl & 0x1F) << 8) | data[i]
            i += 1
            ref = len(out) - ref_offset - 1
            length += 2
            for _ in range(length):
                out.append(out[ref])
                ref += 1

    if len(out) != expected_size:
        raise ValueError(f"LZF 解压后长度不符：期望 {expected_size}，实际 {len(out)}")
    return bytes(out)


@dataclass
class OrientedBox:
    """一个 BBox 的几何信息，跟 bbox_data.csv 的具体列名/字符串格式解耦。"""
    center: tuple[float, float, float]
    rotation_euler: tuple[float, float, float]   # (rotation_x, rotation_y, rotation_z)，Three.js 'XYZ' Euler 顺序
    size: tuple[float, float, float]              # (scale_x, scale_y, scale_z)，即完整的长/宽/高，不是半长


@dataclass
class ClusterCandidate:
    """一个残留点云聚类候选。不含任何类别判断——status 只表示"这个候选离噪声过滤边界
    有多近"，不是"这是什么物体"。"""
    cluster_id: str
    point_indices: list = field(default_factory=list)   # 在传入的 residual 点云里的行索引
    centroid: tuple[float, float, float] | None = None
    point_count: int = 0
    bbox_hint: dict | None = None    # 残留点的轴对齐包围盒：{"min":.., "max":.., "size":..}
    status: str = ""                  # "Possible Missing Annotation" | "Suspicious Residual Cluster"


def _euler_xyz_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """按 Three.js 默认的 'XYZ' Euler 顺序构造旋转矩阵（local -> world），
    即 R = Rx(rx) @ Ry(ry) @ Rz(rz)（跟 bbox_extractor.py 读出来的 node.rotation 顺序一致）。"""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    return rot_x @ rot_y @ rot_z


def remove_bbox_points(points: np.ndarray, boxes: list[OrientedBox], margin: float = 0.0) -> np.ndarray:
    """从点云里删除落在任一 BBox 内部的点，返回剩余点（Residual Point Cloud）。

    判断方式：把点变换到每个 BBox 的局部坐标系（先平移到 BBox 中心，再乘旋转矩阵的
    转置——因为旋转矩阵是正交矩阵，逆矩阵就是转置），局部坐标的每个分量绝对值都不超过
    对应半长（scale/2，Three.js 的 BoxGeometry(1,1,1) 按 scale 缩放后，尺寸就是 scale 本身）
    就算在框内。margin 给这个半长再放大一点比例，避免框边缘的真实点因为旋转/浮点误差被
    误判成"残留点"。

    points: (N, 3) 的 x/y/z 数组。boxes: 这一帧全部 BBox 转换后的 OrientedBox 列表。
    """
    if points.shape[0] == 0 or not boxes:
        return points

    inside_any = np.zeros(points.shape[0], dtype=bool)
    for box in boxes:
        rotation_matrix = _euler_xyz_to_matrix(*box.rotation_euler)
        centered = points - np.array(box.center, dtype=np.float64)
        local = centered @ rotation_matrix   # 等价于 R^T @ centered，把点转到 BBox 局部坐标系
        half_extent = np.array(box.size, dtype=np.float64) / 2.0 * (1.0 + margin)
        inside = np.all(np.abs(local) <= half_extent, axis=1)
        inside_any |= inside

    return points[~inside_any]


def _read_pcd(pcd_path: str) -> np.ndarray:
    """读取 PCD 文件，只提取 x/y/z（不读 intensity，本阶段不需要），返回 (N, 3) 数组。

    只支持 ascii 和 binary 两种 DATA 模式（这个项目的 lidar 输出是这两种之一，见
    bbox_probe.py 的说明：PCD 里只有 x/y/z/intensity）；不支持 PCL 的 binary_compressed
    （LZF 压缩），遇到会抛异常，调用方（rule_engine.py）会捕获并跳过这一帧，不中断整体流程。
    """
    with open(pcd_path, "rb") as f:
        header: dict[str, str] = {}
        while True:
            line = f.readline()
            if not line:
                raise ValueError("PCD 文件在读到 DATA 行之前就结束了，文件可能损坏或不完整")
            text = line.decode("ascii", errors="ignore").strip()
            if not text or text.startswith("#"):
                continue
            key, _, rest = text.partition(" ")
            if key == "DATA":
                data_mode = rest.strip()
                break
            header[key] = rest.strip()

        fields = header.get("FIELDS", "x y z").split()
        sizes = [int(s) for s in header.get("SIZE", "4 4 4").split()]
        types = header.get("TYPE", "F F F").split()
        counts = (
            [int(c) for c in header["COUNT"].split()]
            if "COUNT" in header
            else [1] * len(fields)
        )
        n_points = int(header.get("POINTS", header.get("WIDTH", "0")))

        for axis in ("x", "y", "z"):
            if axis not in fields:
                raise ValueError(f"PCD 文件缺少 {axis} 字段，FIELDS={fields}")

        if data_mode == "ascii":
            xyz_positions = [fields.index(a) for a in ("x", "y", "z")]
            points = np.zeros((n_points, 3), dtype=np.float64)
            for i in range(n_points):
                raw_line = f.readline()
                if not raw_line:
                    raise ValueError(f"PCD ascii 数据提前结束（期望 {n_points} 个点，实际读到 {i} 个）")
                values = raw_line.decode("ascii", errors="ignore").split()
                points[i] = [float(values[p]) for p in xyz_positions]
            return points

        if data_mode == "binary":
            struct_fields = []
            for name, size, type_, count in zip(fields, sizes, types, counts):
                np_type = _PCD_TYPE_MAP.get((type_, size))
                if np_type is None:
                    raise ValueError(f"PCD 字段 {name} 的类型 {type_}{size} 暂不支持解析")
                for c in range(count):
                    field_name = name if count == 1 else f"{name}_{c}"
                    struct_fields.append((field_name, np_type))

            raw = np.frombuffer(f.read(), dtype=np.dtype(struct_fields), count=n_points)
            return np.stack(
                [raw["x"].astype(np.float64), raw["y"].astype(np.float64), raw["z"].astype(np.float64)],
                axis=1,
            )

        if data_mode == "binary_compressed":
            # PCL 的 binary_compressed 格式：8 字节头（compressed_size, uncompressed_size，
            # 均为小端 uint32），后面跟 compressed_size 字节的 LZF 压缩数据。解压后不是按点
            # 交错存储的，而是按字段分栏存储（所有点的 x 连在一起，再是所有点的 y，...），
            # 这是 PCL 写这种格式时的固定布局，不是本项目自己定的。
            compressed_size, uncompressed_size = struct.unpack("<II", f.read(8))
            raw_bytes = _lzf_decompress(f.read(compressed_size), uncompressed_size)

            offset = 0
            field_arrays: dict[str, np.ndarray] = {}
            for name, size, type_, count in zip(fields, sizes, types, counts):
                np_type = _PCD_TYPE_MAP.get((type_, size))
                if np_type is None:
                    raise ValueError(f"PCD 字段 {name} 的类型 {type_}{size} 暂不支持解析")
                n_values = n_points * count
                field_arrays[name] = np.frombuffer(raw_bytes, dtype=np_type, count=n_values, offset=offset)
                offset += n_values * size

            return np.stack(
                [field_arrays["x"].astype(np.float64), field_arrays["y"].astype(np.float64),
                 field_arrays["z"].astype(np.float64)],
                axis=1,
            )

        raise ValueError(f"暂不支持的 PCD DATA 模式: {data_mode!r}（只支持 ascii/binary/binary_compressed）")


def detect_clusters(pcd_path: str, boxes: list[OrientedBox], config: dict) -> list[ClusterCandidate]:
    """对一份 PCD 文件跑 Remove BBox Points + DBSCAN + 噪声过滤，返回候选聚类列表。

    不判断类别、不调用 Vision/LLM——每个候选只有几何信息（点数/质心/AABB）和一个固定的
    status 字符串。

        pcd_path: assets_downloader.py 下载下来的本地 PCD 文件路径
                  （例如 assets/scene_xxx/frame_0011/pointcloud.pcd）
        boxes:    这一帧全部 BBox 转换成的 OrientedBox 列表（调用方负责从 bbox_data.csv 转换）
        config:   config.yaml -> cluster_detector 这个子配置块（eps/min_points/bbox_margin/
                  min_cluster_point_count/min_cluster_volume/max_cluster_volume）
    """
    points = _read_pcd(pcd_path)
    if points.shape[0] == 0:
        return []

    margin = config.get("bbox_margin", 0.05)
    residual = remove_bbox_points(points, boxes, margin)
    if residual.shape[0] == 0:
        return []

    eps = config.get("eps", 0.5)
    min_points = config.get("min_points", 10)
    labels = DBSCAN(eps=eps, min_samples=min_points).fit_predict(residual)

    min_point_count = config.get("min_cluster_point_count", 15)
    min_volume = config.get("min_cluster_volume", 0.05)
    max_volume = config.get("max_cluster_volume", 200.0)

    candidates: list[ClusterCandidate] = []
    for label in sorted(set(labels)):
        if label == -1:
            continue  # DBSCAN 自己的噪声标签，不是候选

        member_indices = np.where(labels == label)[0]
        cluster_points = residual[member_indices]
        point_count = len(cluster_points)
        if point_count < min_point_count:
            continue

        mins = cluster_points.min(axis=0)
        maxs = cluster_points.max(axis=0)
        size = maxs - mins
        volume = float(size[0] * size[1] * size[2])
        if volume < min_volume or volume > max_volume:
            continue

        centroid = cluster_points.mean(axis=0)

        # 越靠近噪声过滤阈值边界，越"可疑"——不是分类，只是标出这个候选离阈值有多近，
        # 供人工判断优先看哪些候选。
        near_boundary = (
            point_count < min_point_count * 2
            or volume < min_volume * 2
            or volume > max_volume * 0.5
        )
        status = "Suspicious Residual Cluster" if near_boundary else "Possible Missing Annotation"

        candidates.append(
            ClusterCandidate(
                cluster_id=f"cluster_{int(label)}",
                point_indices=member_indices.tolist(),
                centroid=tuple(float(v) for v in centroid),
                point_count=point_count,
                bbox_hint={
                    "min": tuple(float(v) for v in mins),
                    "max": tuple(float(v) for v in maxs),
                    "size": tuple(float(v) for v in size),
                },
                status=status,
            )
        )

    return candidates
