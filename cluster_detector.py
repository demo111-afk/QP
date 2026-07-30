"""
cluster_detector.py
点云聚类候选检测——Phase 2：漏标（Missing Annotation）检测的核心模块。

流程是两级聚类（本模块负责全部这几步；调用方 rule_engine.py 负责把结果写进
rule_report.csv，两者职责分开）：

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
    range_filter()：只保留 LiDAR 周围 max_detection_range 米水平半径内的点
     |
     v
    voxel_downsample()：体素降采样，纯粹减少点数，不做任何"是不是地面"的语义判断
     |
     v
    height_filter()：只保留 z > height_threshold 的点，粗略滤掉地面附近最密集的一层
     |    （这不是完整 Ground Removal——没有 RANSAC/平面拟合/CSF/Patchwork++，
     |     只是点云预处理，目的仅仅是让 DBSCAN 的输入规模降到能跑得动）
     v
    第一次 DBSCAN（eps 较小，只用 x/y/z，不用 intensity）：精细但容易把一个真实物体
     |    因为遮挡/稀疏回波/BBox 裕量削点等原因切成好几个局部碎片——这是刻意的代价，
     |    不是参数没调好，避免把两个真实独立的物体粘连成一个。
     v
    宽松初筛：只丢弃 DBSCAN 自己判定的噪声（-1），不做任何点数/体积/扁平度判断——
     |    严格几何过滤放到第二次聚类合并之后，避免在合并前就误杀本该合并的碎片。
     v
    第二次 Cluster-level 聚类（_merge_nearby_candidates）：按碎片 AABB 在 XY 平面上的
     |    最近距离合并（不是质心距离，也不看 Z 重叠——灯塔/场桥/岸桥这类高瘦结构不同
     |    高度的碎片本该合并）。合并后用真实点重新计算 point_count/centroid/AABB/
     |    volume/flatness（PCA 特征值），统一跑一轮严格过滤（含合并后整体尺寸上限，
     |    挡住链式合并出的不合理结果），不管候选是不是被合并过都要过这一关。
     v
    ClusterCandidate 列表（status 固定是 "Possible Missing Annotation" 或
    "Suspicious Residual Cluster"，不判断类别、不调用 Vision/LLM）

背景：真实数据实测过，如果跳过 voxel_downsample/height_filter 直接对全场景残留点云
（30万级点，其中一半以上是地面）跑 DBSCAN，sklearn 在算近邻列表时会直接内存溢出——
地面点密度极高，任何一个点附近可能有成百上千个邻居。这两步预处理是让流程在真实点云
规模下能跑通的必要条件，不是可选优化。

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
    point_indices: list = field(default_factory=list)   # 在体素降采样+高度过滤后点云里的行索引
    centroid: tuple[float, float, float] | None = None
    point_count: int = 0
    bbox_hint: dict | None = None    # 残留点的轴对齐包围盒：{"min":.., "max":.., "size":..}
    status: str = ""                  # "Possible Missing Annotation" | "Suspicious Residual Cluster"
    merged_fragment_count: int = 1     # 由几个原始 DBSCAN 碎片合并而来；1 表示没有被合并过


@dataclass
class _Fragment:
    """第一次 DBSCAN 产出的一个原始局部碎片（内部使用，不对外暴露）。只记录必要的
    几何信息（AABB）供第二次聚类判断要不要合并，不在这一步做任何点数/体积/扁平度过滤。"""
    label: int
    indices: np.ndarray   # 在 filtered_points 里的行索引
    mins: np.ndarray
    maxs: np.ndarray


def _pairwise_xy_gap_matrix(mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    """算 N 个 AABB 两两之间在 XY 平面上的最近距离（重叠记为 0），返回 (N, N) 距离矩阵。

    故意只看 X/Y，不看 Z——像灯塔/场桥/岸桥这类真实高瘦结构，不同高度的碎片
    水平投影往往重合，但 Z 跨度可能有好几米，用 3D 距离反而会把这些本该合并的
    同一物体的上下碎片挡在外面。
    """
    mins_xy = mins[:, :2]
    maxs_xy = maxs[:, :2]
    gap = np.maximum(
        0.0,
        np.maximum(
            mins_xy[:, None, :] - maxs_xy[None, :, :],
            mins_xy[None, :, :] - maxs_xy[:, None, :],
        ),
    )
    return np.sqrt((gap ** 2).sum(axis=2))


def _shape_features(points: np.ndarray) -> tuple[float, float]:
    """算一个点集的体积（AABB）和形状扁平度。

    扁平度用 PCA 特征值算：对点集做协方差矩阵特征分解，得到 λ1≥λ2≥λ3。
    linearity = (λ1-λ2)/λ1，接近 1 说明点云主要沿一个方向延展（像一根线/杆）；
    planarity = (λ2-λ3)/λ1，接近 1 说明点云基本躺在一个平面上（像一面墙/薄片）；
    flatness = 1 - max(linearity, planarity)，越接近 1 越像"敦实"的三维物体
    （三个方向的延展程度比较接近），越接近 0 越像退化的线状/面状结构。

    这比单纯用"AABB 最短边/最长边"更准——一根斜着放置、没有对齐坐标轴的杆，
    AABB 在 x/y 上可能看起来差不多方正，但 PCA 能正确识别出它其实是线状的。
    """
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    size = maxs - mins
    volume = float(size[0] * size[1] * size[2])

    if len(points) < 3:
        return volume, 0.0   # 点太少，PCA 不稳定，保守地当作最扁处理

    centered = points - points.mean(axis=0)
    cov = (centered.T @ centered) / len(points)
    eigenvalues = np.linalg.eigvalsh(cov)   # 升序返回：lam3 <= lam2 <= lam1
    lam3, lam2, lam1 = eigenvalues

    if lam1 <= 1e-12:
        return volume, 0.0

    linearity = (lam1 - lam2) / lam1
    planarity = (lam2 - lam3) / lam1
    flatness = 1.0 - max(linearity, planarity)
    return volume, float(flatness)


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


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """体素降采样：把空间切成 voxel_size 大小的网格，同一格子里的点合并成它们的质心。

    纯粹是为了减少后续 DBSCAN 的输入规模，不做任何"这是不是地面"之类的语义判断——
    跟真实场景里占大多数的地面点、离散噪声点，都是同样按网格合并，不做区分。
    """
    if points.shape[0] == 0 or voxel_size <= 0:
        return points

    keys = np.floor(points / voxel_size).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    sorted_points = points[order]

    changed = np.any(np.diff(sorted_keys, axis=0) != 0, axis=1)
    boundaries = np.concatenate(([0], np.where(changed)[0] + 1, [len(sorted_keys)]))

    out = np.empty((len(boundaries) - 1, 3), dtype=np.float64)
    for i in range(len(boundaries) - 1):
        out[i] = sorted_points[boundaries[i]:boundaries[i + 1]].mean(axis=0)
    return out


def range_filter(points: np.ndarray, max_range: float | None = None) -> np.ndarray:
    """距离过滤：只保留 LiDAR 周围 max_range 米水平半径内的点。

    这里使用 XY 平面距离 sqrt(x^2 + y^2)，不把 z 算进去；高度范围由
    height_filter(height_threshold/max_height) 单独负责。max_range 为空或 <=0 时不启用。
    """
    if points.shape[0] == 0 or max_range is None or max_range <= 0:
        return points
    xy_distance = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    return points[xy_distance <= max_range]


def height_filter(points: np.ndarray, height_threshold: float, max_height: float | None = None) -> np.ndarray:
    """简单高度过滤：只保留 height_threshold < z（< max_height，如果给了的话）的点。

    这不是完整的 Ground Removal——没有做 RANSAC 平面拟合、没有 CSF、没有 Patchwork++，
    只是固定阈值的点云预处理步骤。下限（height_threshold）滤掉地面附近最密集的一层；
    上限（max_height）滤掉明显高于参考表里最高目标（岸桥/灯塔/场桥都是 8m 量级）的点——
    这些多半是周围建筑物/高层结构，不太可能是漏标的目标物体，同时它们镂空的结构容易被
    DBSCAN 切成一堆小碎片，是"漏标候选数量偏多"的一个主要来源。
    两个阈值都是这一帧点云自己坐标系里的绝对 z 值，不是拟合出来的地面/建筑高度——
    如果不同 scene/雷达安装方式导致 z 原点不一致，这两个值可能需要跟着调。
    """
    if points.shape[0] == 0:
        return points
    mask = points[:, 2] > height_threshold
    if max_height is not None:
        mask &= points[:, 2] < max_height
    return points[mask]


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
    """两级聚类：第一次 DBSCAN 精细分割（容易把一个真实物体拆成多个局部碎片），
    第二次按碎片 AABB 的 XY 距离做 Cluster-level 合并，合并后统一做一轮严格几何复检。

    不判断类别、不调用 Vision/LLM——每个候选只有几何信息（点数/质心/AABB/形状）和一个
    固定的 status 字符串。

        pcd_path: assets_downloader.py 下载下来的本地 PCD 文件路径
                  （例如 assets/scene_xxx/frame_0011/pointcloud.pcd）
        boxes:    这一帧全部 BBox 转换成的 OrientedBox 列表（调用方负责从 bbox_data.csv 转换）
        config:   config.yaml -> cluster_detector 这个子配置块（max_detection_range/voxel_size/
                  height_threshold/max_height/eps/min_points/bbox_margin/merge_distance/max_merged_extent/
                  min_cluster_point_count/min_cluster_volume/max_cluster_volume/min_flatness）
    """
    points = _read_pcd(pcd_path)
    if points.shape[0] == 0:
        return []

    margin = config.get("bbox_margin", 0.05)
    residual = remove_bbox_points(points, boxes, margin)
    if residual.shape[0] == 0:
        return []

    max_detection_range = config.get("max_detection_range")
    ranged = range_filter(residual, max_detection_range)
    if ranged.shape[0] == 0:
        return []

    # 真实点云（30万级）实测过，跳过这两步直接对 residual 跑 DBSCAN 会内存溢出——
    # 地面点密度太高，见模块开头的说明。这两步是必须的，不是可选优化。
    voxel_size = config.get("voxel_size", 0.2)
    downsampled = voxel_downsample(ranged, voxel_size)
    if downsampled.shape[0] == 0:
        return []

    height_threshold = config.get("height_threshold", 0.3)
    max_height = config.get("max_height")
    filtered_points = height_filter(downsampled, height_threshold, max_height)
    if filtered_points.shape[0] == 0:
        return []

    # ---- 第一次 DBSCAN：精细分割 ----
    eps = config.get("eps", 0.4)
    min_points = config.get("min_points", 5)
    labels = DBSCAN(eps=eps, min_samples=min_points).fit_predict(filtered_points)

    # 宽松初筛：只丢弃 DBSCAN 自己判定的噪声（-1）。不在这里做任何点数/体积/扁平度
    # 判断——严格过滤放到合并之后统一执行，避免在合并前就误杀本该合并的碎片。
    fragments: list[_Fragment] = []
    for label in sorted(set(labels)):
        if label == -1:
            continue
        member_indices = np.where(labels == label)[0]
        member_points = filtered_points[member_indices]
        fragments.append(_Fragment(
            label=int(label),
            indices=member_indices,
            mins=member_points.min(axis=0),
            maxs=member_points.max(axis=0),
        ))

    return _merge_nearby_candidates(fragments, filtered_points, config)


def _merge_nearby_candidates(
    fragments: list[_Fragment], filtered_points: np.ndarray, config: dict
) -> list[ClusterCandidate]:
    """第二次 Cluster-level 聚类：把可能属于同一个真实实体、但被第一次 DBSCAN 拆开的碎片
    重新合并，然后对合并结果（不管由 1 个还是多个碎片组成）统一做一轮严格几何复检。

    合并判据：两个碎片的 AABB 在 XY 平面上的最近距离 <= merge_distance（见
    _pairwise_xy_gap_matrix，不是质心距离，也不看 Z 方向重叠）。实现上复用 DBSCAN——
    把两两之间的 XY 距离矩阵喂给 metric="precomputed"，min_samples=1 让每个碎片都能
    自成一组。这个实现仍然具备链式传递性（A-B、B-C 各自在 eps 内会连成一组，即使
    A-C 相距较远）——不打算消除传递性本身（任何基于连通分量的合并方式都有这个特性），
    而是靠下面的 max_merged_extent 挡住链式合并出的不合理结果。

    合并后用真实点（不是包围盒角点）重新计算 point_count/centroid/AABB/volume/flatness，
    再统一跑一次严格过滤——这一轮过滤对"没有被合并、自己单独成一组"的候选同样生效，
    不是只筛合并后的结果。
    """
    if not fragments:
        return []

    min_point_count = config.get("min_cluster_point_count", 15)
    min_volume = config.get("min_cluster_volume", 0.05)
    max_volume = config.get("max_cluster_volume", 200.0)
    min_flatness = config.get("min_flatness", 0.25)
    max_merged_extent = config.get("max_merged_extent", 30.0)

    if len(fragments) == 1:
        group_labels = np.array([0])
    else:
        mins = np.array([f.mins for f in fragments])
        maxs = np.array([f.maxs for f in fragments])
        merge_distance = config.get("merge_distance", 1.5)
        distance_matrix = _pairwise_xy_gap_matrix(mins, maxs)
        group_labels = DBSCAN(
            eps=merge_distance, min_samples=1, metric="precomputed"
        ).fit_predict(distance_matrix)

    candidates: list[ClusterCandidate] = []
    for group_label in sorted(set(group_labels)):
        members = [f for f, g in zip(fragments, group_labels) if g == group_label]
        group_indices = np.concatenate([m.indices for m in members])
        group_points = filtered_points[group_indices]

        point_count = len(group_points)
        if point_count < min_point_count:
            continue

        group_mins = group_points.min(axis=0)
        group_maxs = group_points.max(axis=0)
        size = group_maxs - group_mins
        if size.max() > max_merged_extent:
            continue  # 合并结果明显不合理（比如链式合并出几十米长），整条丢弃

        volume, flatness = _shape_features(group_points)
        if volume < min_volume or volume > max_volume:
            continue
        if flatness < min_flatness:
            continue

        centroid = group_points.mean(axis=0)
        merged_fragment_count = len(members)

        if merged_fragment_count > 1:
            # 多个独立碎片能拼成一个通过所有几何检查的连贯结构，是比单个碎片更强的证据。
            status = "Possible Missing Annotation"
        else:
            # 没有被合并：沿用"离过滤阈值有多近"这套模糊判断，供人工判断优先看哪些候选。
            near_boundary = (
                point_count < min_point_count * 2
                or volume < min_volume * 2
                or volume > max_volume * 0.5
            )
            status = "Suspicious Residual Cluster" if near_boundary else "Possible Missing Annotation"

        candidates.append(
            ClusterCandidate(
                cluster_id="+".join(f"cluster_{m.label}" for m in members),
                point_indices=group_indices.tolist(),
                centroid=tuple(float(v) for v in centroid),
                point_count=point_count,
                bbox_hint={
                    "min": tuple(float(v) for v in group_mins),
                    "max": tuple(float(v) for v in group_maxs),
                    "size": tuple(float(v) for v in size),
                },
                status=status,
                merged_fragment_count=merged_fragment_count,
            )
        )

    return candidates
