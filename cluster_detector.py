"""
cluster_detector.py
点云聚类候选检测——Phase 1：只建立接口，不实现聚类算法。

未来完整流程（本文件只是这条流程的入口占位，具体算法留给后续阶段）：

    PCD
     |
     v
    Ground Removal（地面点剔除）
     |
     v
    Remove BBox Points（剔除已经被现有 BBox 标注覆盖的点，只留下"未被标注"的点）
     |
     v
    DBSCAN（对剩余点做密度聚类）
     |
     v
    Candidate Cluster（聚类结果 -> ClusterCandidate）
     |
     v
    Geometry Filter（按尺寸/点数等几何特征过滤明显不合理的候选）
     |
     v
    Vision Model（候选交给 vision_classifier.py 做视觉分类，见该文件）

跟 bbox_extractor.py 完全无关：BBox 的读取/保存逻辑保持现状，不会被这里调用或修改。
本模块消费的是 assets_downloader.py 下载下来的 PCD 文件（本地路径），不重新连浏览器。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ClusterCandidate:
    """一个聚类候选的占位结构，字段先按未来大概率需要的信息定，具体算法接入时再调整。"""
    cluster_id: str
    point_indices: list = field(default_factory=list)   # 属于这个候选的点云索引
    centroid: tuple[float, float, float] | None = None
    point_count: int = 0
    bbox_hint: dict | None = None    # 粗略的轴对齐包围盒（min/max），供 Geometry Filter 用


def remove_bbox_points(points, bboxes) -> None:
    """从点云里剔除已经落在现有 BBox 内的点，只保留未被标注覆盖的点用于聚类。

    Phase 1 不实现，只定义签名。
        points: 点云数据（未来接入时确定具体格式，比如 (N,4) 的 numpy 数组 x/y/z/intensity）
        bboxes: 复用 bbox_extractor.py 产出的 bbox_data.csv 里的位置/朝向/尺寸，
                不重新定义 BBox 的表示方式。
    """
    raise NotImplementedError(
        "remove_bbox_points 尚未实现——Phase 1 只建立接口，见模块文档的未来流程。"
    )


def detect_clusters(pcd_path: str, config: dict | None = None) -> list[ClusterCandidate]:
    """对一份 PCD 文件跑 Ground Removal + Remove BBox Points + DBSCAN，返回候选聚类列表。

    Phase 1 不实现，只定义签名。
        pcd_path: assets_downloader.py 下载下来的本地 PCD 文件路径
                  （例如 assets/scene_xxx/frame_0001/pointcloud.pcd）
        config:   预留给未来的算法参数（ground_removal 阈值、DBSCAN eps/min_samples 等），
                  跟现有 config.yaml 的风格保持一致，具体 key 在实现时再定义。
    """
    raise NotImplementedError(
        "detect_clusters 尚未实现——Phase 1 只建立接口，见模块文档的未来流程。"
    )
