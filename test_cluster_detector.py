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
    _Fragment,
    _merge_nearby_candidates,
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


def test_height_filter_upper_bound_drops_tall_points():
    points = np.array([[0, 0, -1], [0, 0, 0.5], [0, 0, 5], [0, 0, 15]])
    filtered = height_filter(points, height_threshold=0.3, max_height=10.0)
    assert filtered.shape[0] == 2
    assert np.allclose(sorted(filtered[:, 2].tolist()), [0.5, 5])


# detect_clusters 端到端测试用的默认 config：显式关掉 voxel/height 这两步的实际效果
# （voxel_size 设得比测试点云的间距小很多，height_threshold 设得比所有测试点的 z 都低），
# 这样这几个测试专注验证 DBSCAN + 噪声过滤本身的行为，不跟体素化/高度过滤混在一起判断。
_BASE_CFG = {
    "voxel_size": 0.01, "height_threshold": -1000.0,
    "eps": 0.5, "min_points": 10, "bbox_margin": 0.05,
    "min_cluster_point_count": 15, "min_cluster_volume": 0.0001, "max_cluster_volume": 200.0,
    "min_flatness": 0.0,   # 默认关掉，跟扁平度无关的测试不受这个新过滤影响
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


def _build_fragments(point_groups: list[np.ndarray]) -> tuple[list[_Fragment], np.ndarray]:
    """把几组点直接拼成 (fragments, filtered_points)，绕开 PCD 文件/第一次 DBSCAN，
    直接单测 _merge_nearby_candidates（第二次聚类）本身的合并/复检逻辑。"""
    filtered_points = np.vstack(point_groups)
    fragments = []
    offset = 0
    for i, group in enumerate(point_groups):
        n = len(group)
        indices = np.arange(offset, offset + n)
        fragments.append(_Fragment(label=i, indices=indices, mins=group.min(axis=0), maxs=group.max(axis=0)))
        offset += n
    return fragments, filtered_points


# _merge_nearby_candidates 单测用的宽松 config：只关心合并/尺寸/形状判据本身，
# 点数/体积门槛故意放得很低很宽，不让它们干扰这几个测试。
_MERGE_CFG = {
    "merge_distance": 1.5, "max_merged_extent": 30.0,
    "min_cluster_point_count": 5, "min_cluster_volume": 0.0, "max_cluster_volume": 1e6,
    "min_flatness": 0.0,
}


def test_merge_nearby_candidates_combines_xy_close_ignores_z():
    rng = np.random.default_rng(1)
    # a、b 的 XY 位置几乎重合，Z 差 8 米（模拟灯塔上下两段断开的碎片）——
    # 判据故意不看 Z，应该合并。
    frag_a = rng.normal(loc=[10, 10, 0], scale=0.1, size=(30, 3))
    frag_b = rng.normal(loc=[10, 10, 8], scale=0.1, size=(30, 3))
    # c 的 XY 位置离得很远（90 米），不应该被卷进来
    frag_c = rng.normal(loc=[100, 100, 0], scale=0.1, size=(30, 3))

    fragments, filtered_points = _build_fragments([frag_a, frag_b, frag_c])
    candidates = _merge_nearby_candidates(fragments, filtered_points, _MERGE_CFG)

    assert len(candidates) == 2, f"a+b 应该合并，c 保持独立，实际 {len(candidates)}"
    combined = next(c for c in candidates if c.merged_fragment_count == 2)
    untouched = next(c for c in candidates if c.merged_fragment_count == 1)
    assert combined.point_count == 60
    assert untouched.point_count == 30


def test_merge_nearby_candidates_xy_far_apart_no_merge():
    rng = np.random.default_rng(2)
    frag_a = rng.normal(loc=[0, 0, 0], scale=0.1, size=(30, 3))
    frag_b = rng.normal(loc=[10, 0, 0], scale=0.1, size=(30, 3))  # XY 距离 10m，远超 merge_distance=1.5
    fragments, filtered_points = _build_fragments([frag_a, frag_b])
    candidates = _merge_nearby_candidates(fragments, filtered_points, _MERGE_CFG)
    assert len(candidates) == 2
    assert all(c.merged_fragment_count == 1 for c in candidates)


def test_merge_nearby_candidates_rejects_oversized_chain_merge():
    rng = np.random.default_rng(3)
    # 模拟一排间距 1.2m 的围栏立柱：相邻两根的 XY 间隙都在 merge_distance=1.5 内，
    # 会链式合并成一整条（这正是链式传递性的体现，不打算消除），但整排跨度约 22.8m，
    # 超过 max_merged_extent=15，应该被整体丢弃。
    posts = [rng.normal(loc=[i * 1.2, 0, 0], scale=0.05, size=(10, 3)) for i in range(20)]
    fragments, filtered_points = _build_fragments(posts)
    cfg = dict(_MERGE_CFG)
    cfg["max_merged_extent"] = 15.0
    candidates = _merge_nearby_candidates(fragments, filtered_points, cfg)
    assert candidates == [], f"链式合并结果超过尺寸上限，应该整体丢弃，实际保留了 {len(candidates)} 条"


def test_merge_nearby_candidates_rejects_linear_shape_after_merge():
    rng = np.random.default_rng(4)
    # 4 个各自还算敦实的小碎片，沿一条直线排开、彼此间隙都在 merge_distance 内会合并，
    # 但合并后整体点云呈线状——用来验证"合并后重新算 flatness"这一步真的在起作用
    # （如果只在合并前检查扁平度，这几个小圆球状碎片各自都能轻松通过）。
    blobs = [rng.normal(loc=[i * 1.0, 0, 0], scale=0.15, size=(20, 3)) for i in range(4)]
    fragments, filtered_points = _build_fragments(blobs)
    cfg = dict(_MERGE_CFG)
    cfg["min_flatness"] = 0.3
    candidates = _merge_nearby_candidates(fragments, filtered_points, cfg)
    assert candidates == [], f"合并后呈线状，应该被扁平度过滤挡掉，实际保留了 {len(candidates)} 条"


def test_detect_clusters_merges_fragmented_structure():
    rng = np.random.default_rng(123)
    # 一个真实结构因为内部有空隙，被切成两段紧挨着的点云（间距略大于 eps=0.4，
    # 第一遍 DBSCAN 会把它们分成 2 个独立候选），但两段 XY 位置几乎重合、只是 Z 差 1.5m
    # （模拟灯塔上下两段），应该被第二次聚类重新合并成 1 条——判据故意不看 Z 差距。
    fragment_1 = rng.normal(loc=[10, 10, 5.0], scale=0.05, size=(60, 3))
    fragment_2 = rng.normal(loc=[10, 10, 6.5], scale=0.05, size=(60, 3))
    all_points = np.vstack([fragment_1, fragment_2])

    with tempfile.TemporaryDirectory() as tmp:
        pcd_path = Path(tmp) / "scene.pcd"
        _write_ascii_pcd(pcd_path, all_points)
        candidates = detect_clusters(str(pcd_path), [], _BASE_CFG)

    assert len(candidates) == 1, f"两段应该被合并成 1 条，实际 {len(candidates)}"
    assert candidates[0].merged_fragment_count == 2, "应该是由 2 个碎片合并而来"
    # 体素降采样会让点数略微减少（同一个体素里的点合并成 1 个），不要求跟原始 120 完全相等
    assert candidates[0].point_count >= 110


def test_detect_clusters_flatness_filter_drops_thin_structures():
    rng = np.random.default_rng(55)
    # 一段"围栏杆"：x 方向跨 4 米，y/z 方向只有 0.1 米左右，扁平度约 0.1/4 = 0.025，
    # 明显低于 min_flatness=0.25，应该被滤掉。
    fence = np.column_stack([
        rng.uniform(0, 4, 100),
        rng.normal(0, 0.03, 100),
        rng.normal(0, 0.03, 100),
    ])
    # 一个"敦实"的目标：三个方向尺寸接近，扁平度高，应该保留。
    compact = rng.normal(loc=[20, 20, 1.0], scale=0.3, size=(80, 3))

    all_points = np.vstack([fence, compact])
    cfg = dict(_BASE_CFG)
    cfg["min_flatness"] = 0.25

    with tempfile.TemporaryDirectory() as tmp:
        pcd_path = Path(tmp) / "scene.pcd"
        _write_ascii_pcd(pcd_path, all_points)
        candidates = detect_clusters(str(pcd_path), [], cfg)

    assert len(candidates) == 1, f"围栏杆应该被扁平度过滤掉，只剩敦实目标，实际 {len(candidates)}"
    assert np.allclose(candidates[0].centroid, [20, 20, 1.0], atol=1.0)


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
        test_height_filter_upper_bound_drops_tall_points,
        test_detect_clusters_end_to_end,
        test_detect_clusters_no_residual_returns_empty,
        test_merge_nearby_candidates_combines_xy_close_ignores_z,
        test_merge_nearby_candidates_xy_far_apart_no_merge,
        test_merge_nearby_candidates_rejects_oversized_chain_merge,
        test_merge_nearby_candidates_rejects_linear_shape_after_merge,
        test_detect_clusters_merges_fragmented_structure,
        test_detect_clusters_flatness_filter_drops_thin_structures,
        test_detect_clusters_height_filter_removes_ground_band,
    ]
    for test in tests:
        test()
        print(f"[通过] {test.__name__}")
    print(f"\n全部 {len(tests)} 个测试通过。")


if __name__ == "__main__":
    main()
