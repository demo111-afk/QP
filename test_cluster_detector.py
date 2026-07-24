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

from cluster_detector import OrientedBox, _euler_xyz_to_matrix, _read_pcd, detect_clusters, remove_bbox_points


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


def test_detect_clusters_end_to_end():
    rng = np.random.default_rng(42)
    covered = rng.normal(loc=[0, 0, 0], scale=0.3, size=(200, 3))          # 落在 BBox 内，应被删除
    residual_cluster = rng.normal(loc=[20, 20, 0], scale=0.2, size=(80, 3))  # 真实的漏标聚类
    noise = rng.uniform(low=-50, high=50, size=(5, 3))                       # 孤立噪声，DBSCAN 应标 -1

    all_points = np.vstack([covered, residual_cluster, noise])
    box = OrientedBox(center=(0, 0, 0), rotation_euler=(0, 0, 0), size=(2, 2, 2))
    cfg = {
        "eps": 0.5, "min_points": 10, "bbox_margin": 0.05,
        "min_cluster_point_count": 15, "min_cluster_volume": 0.0001, "max_cluster_volume": 200.0,
    }

    with tempfile.TemporaryDirectory() as tmp:
        pcd_path = Path(tmp) / "scene.pcd"
        _write_ascii_pcd(pcd_path, all_points)
        candidates = detect_clusters(str(pcd_path), [box], cfg)

    assert len(candidates) == 1, f"应该只找到 1 个候选，实际 {len(candidates)}"
    assert candidates[0].point_count >= 70
    assert np.allclose(candidates[0].centroid, [20, 20, 0], atol=1.0)
    assert candidates[0].status in ("Possible Missing Annotation", "Suspicious Residual Cluster")


def test_detect_clusters_no_residual_returns_empty():
    rng = np.random.default_rng(7)
    covered = rng.normal(loc=[0, 0, 0], scale=0.3, size=(100, 3))
    box = OrientedBox(center=(0, 0, 0), rotation_euler=(0, 0, 0), size=(3, 3, 3))
    cfg = {"eps": 0.5, "min_points": 10, "bbox_margin": 0.05,
           "min_cluster_point_count": 15, "min_cluster_volume": 0.0001, "max_cluster_volume": 200.0}

    with tempfile.TemporaryDirectory() as tmp:
        pcd_path = Path(tmp) / "scene.pcd"
        _write_ascii_pcd(pcd_path, covered)
        candidates = detect_clusters(str(pcd_path), [box], cfg)

    assert candidates == []


def main() -> None:
    tests = [
        test_rotation_matrix_is_orthogonal,
        test_remove_bbox_points_axis_aligned,
        test_remove_bbox_points_rotated,
        test_read_pcd_ascii_and_binary_match,
        test_detect_clusters_end_to_end,
        test_detect_clusters_no_residual_returns_empty,
    ]
    for test in tests:
        test()
        print(f"[通过] {test.__name__}")
    print(f"\n全部 {len(tests)} 个测试通过。")


if __name__ == "__main__":
    main()
