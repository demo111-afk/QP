"""
test_cluster_detector.py
cluster_detector.py 的独立测试（跟 test_assets_downloader.py 一样，纯 assert，不用 pytest）。
用合成的点云/PCD 文件测试，不需要真实抓取的数据。

用法：
    python test_cluster_detector.py
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np

from cluster_detector import (
    OrientedBox,
    _euler_xyz_to_matrix,
    _read_pcd,
    detect_clusters,
    height_filter,
    remove_bbox_points,
    voxel_downsample,
)


def _write_ascii_pcd(path: Path, points: np.ndarray) -> None:
    n = len(points)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z intensity\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA ascii\n"
    )
    with open(path, "w") as f:
        f.write(header)
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]} 0.5\n")


def _write_binary_pcd(path: Path, points: np.ndarray) -> None:
    n = len(points)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z intensity\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA binary\n"
    )
    arr = np.zeros(n, dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32), ("intensity", np.float32)])
    arr["x"], arr["y"], arr["z"] = points[:, 0], points[:, 1], points[:, 2]
    arr["intensity"] = 0.5
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes())


def test_rotation_matrix_is_orthogonal():
    R = _euler_xyz_to_matrix(0.3, -0.2, 1.1)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)


def test_remove_bbox_points_axis_aligned():
    box = OrientedBox(center=(0, 0, 0), rotation_euler=(0, 0, 0), size=(2, 2, 2))
    points = np.array([[0, 0, 0], [0.9, 0.9, 0.9], [5, 5, 5], [-5, 0, 0]])
    residual = remove_bbox_points(points, [box], margin=0.0)
    assert residual.shape[0] == 2
    assert np.allclose(sorted(residual.tolist()), sorted([[5, 5, 5], [-5, 0, 0]]))


def test_remove_bbox_points_rotated():
    box = OrientedBox(center=(10, 10, 0), rotation_euler=(0, 0, math.pi / 4), size=(4, 1, 1))
    Rz = _euler_xyz_to_matrix(0, 0, math.pi / 4)
    world_pt = np.array([10, 10, 0]) + Rz @ np.array([1.5, 0, 0])  # 局部坐标在box内，世界坐标应被删除
    points = np.array([world_pt, [100, 100, 100]])
    residual = remove_bbox_points(points, [box], margin=0.0)
    assert residual.shape[0] == 1
    assert np.allclose(residual[0], [100, 100, 100])


def test_read_pcd_ascii_and_binary_match():
    rng = np.random.default_rng(1)
    points = rng.normal(size=(50, 3))
    with tempfile.TemporaryDirectory() as tmp:
        ascii_path = Path(tmp) / "a.pcd"
        binary_path = Path(tmp) / "b.pcd"
        _write_ascii_pcd(ascii_path, points)
        _write_binary_pcd(binary_path, points)

        parsed_ascii = _read_pcd(str(ascii_path))
        parsed_binary = _read_pcd(str(binary_path))

    assert parsed_ascii.shape == points.shape
    assert np.allclose(parsed_ascii, points, atol=1e-4)
    assert parsed_binary.shape == points.shape
    assert np.allclose(parsed_binary, points, atol=1e-3)


def test_voxel_downsample_merges_points_in_same_cell():
    # 4 个点都落在 [0,0.2) x [0,0.2) x [0,0.2) 这同一个 0.2m 格子里，应该合并成 1 个质心点
    points = np.array([[0.01, 0.01, 0.01], [0.05, 0.02, 0.03], [0.1, 0.15, 0.05], [0.19, 0.19, 0.19]])
    down = voxel_downsample(points, voxel_size=0.2)
    assert down.shape[0] == 1
    assert np.allclose(down[0], points.mean(axis=0))

    # 2 个点分别落在不同格子里，不应该被合并
    far_points = np.array([[0, 0, 0], [10, 10, 10]])
    down_far = voxel_downsample(far_points, voxel_size=0.2)
    assert down_far.shape[0] == 2


def test_height_filter_keeps_only_points_above_threshold():
    points = np.array([[0, 0, -1], [0, 0, 0.1], [0, 0, 0.5], [0, 0, 5]])
    filtered = height_filter(points, height_threshold=0.3)
    assert filtered.shape[0] == 2
    assert np.allclose(sorted(filtered[:, 2].tolist()), [0.5, 5])


# detect_clusters 端到端测试用的默认 config：显式关掉 voxel/height 这两步的实际效果
# （voxel_size 设得比测试点云的间距小很多，height_threshold 设得比所有测试点的 z 都低），
# 这样这几个测试专注验证 DBSCAN + 噪声过滤本身的行为，不跟体素化/高度过滤混在一起判断。
_BASE_CFG = {
    "voxel_size": 0.01, "height_threshold": -1000.0,
    "eps": 0.5, "min_points": 10, "bbox_margin": 0.05,
    "min_cluster_point_count": 15, "min_cluster_volume": 0.0001, "max_cluster_volume": 200.0,
}


def test_detect_clusters_end_to_end():
    rng = np.random.default_rng(42)
    covered = rng.normal(loc=[0, 0, 0], scale=0.3, size=(200, 3))          # 落在 BBox 内，应被删除
    residual_cluster = rng.normal(loc=[20, 20, 0], scale=0.2, size=(80, 3))  # 真实的漏标聚类
    noise = rng.uniform(low=-50, high=50, size=(5, 3))                       # 孤立噪声，DBSCAN 应标 -1

    all_points = np.vstack([covered, residual_cluster, noise])
    box = OrientedBox(center=(0, 0, 0), rotation_euler=(0, 0, 0), size=(2, 2, 2))

    with tempfile.TemporaryDirectory() as tmp:
        pcd_path = Path(tmp) / "scene.pcd"
        _write_ascii_pcd(pcd_path, all_points)
        candidates = detect_clusters(str(pcd_path), [box], _BASE_CFG)

    assert len(candidates) == 1, f"应该只找到 1 个候选，实际 {len(candidates)}"
    assert candidates[0].point_count >= 70
    assert np.allclose(candidates[0].centroid, [20, 20, 0], atol=1.0)
    assert candidates[0].status in ("Possible Missing Annotation", "Suspicious Residual Cluster")


def test_detect_clusters_no_residual_returns_empty():
    rng = np.random.default_rng(7)
    covered = rng.normal(loc=[0, 0, 0], scale=0.3, size=(100, 3))
    box = OrientedBox(center=(0, 0, 0), rotation_euler=(0, 0, 0), size=(3, 3, 3))

    with tempfile.TemporaryDirectory() as tmp:
        pcd_path = Path(tmp) / "scene.pcd"
        _write_ascii_pcd(pcd_path, covered)
        candidates = detect_clusters(str(pcd_path), [box], _BASE_CFG)

    assert candidates == []


def test_detect_clusters_height_filter_removes_ground_band():
    rng = np.random.default_rng(99)
    # "地面"：大量落在 z < 0.3 的密集点，铺满一大片区域，没有被任何 BBox 覆盖
    ground = rng.normal(loc=[0, 0, 0.05], scale=[20, 20, 0.05], size=(2000, 3))
    # 真实的漏标目标：z 明显高于地面（比如一个 1.8m 高的物体主体部分）
    real_object = rng.normal(loc=[5, 5, 1.8], scale=0.2, size=(80, 3))

    all_points = np.vstack([ground, real_object])
    cfg = dict(_BASE_CFG)
    cfg["height_threshold"] = 0.3   # 恢复真实的高度过滤阈值，验证地面被滤掉

    with tempfile.TemporaryDirectory() as tmp:
        pcd_path = Path(tmp) / "scene.pcd"
        _write_ascii_pcd(pcd_path, all_points)
        candidates = detect_clusters(str(pcd_path), [], cfg)

    assert len(candidates) == 1, f"高度过滤后应该只剩下真实目标这一个候选，实际 {len(candidates)}"
    assert candidates[0].centroid[2] > 1.0, "候选的质心高度应该在地面之上"


def main() -> None:
    tests = [
        test_rotation_matrix_is_orthogonal,
        test_remove_bbox_points_axis_aligned,
        test_remove_bbox_points_rotated,
        test_read_pcd_ascii_and_binary_match,
        test_voxel_downsample_merges_points_in_same_cell,
        test_height_filter_keeps_only_points_above_threshold,
        test_detect_clusters_end_to_end,
        test_detect_clusters_no_residual_returns_empty,
        test_detect_clusters_height_filter_removes_ground_band,
    ]
    for test in tests:
        test()
        print(f"[通过] {test.__name__}")
    print(f"\n全部 {len(tests)} 个测试通过。")


if __name__ == "__main__":
    main()
