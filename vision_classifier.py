"""
vision_classifier.py
视觉分类接口占位——Phase 1：只定义接口，不调用任何模型，不写任何 API 请求代码。

未来用途：cluster_detector.py 产出的候选聚类，结合对应的截图裁剪区域、BBox 尺寸、
点云强度等信息，交给一个视觉模型判断"这个候选到底是什么物体"，辅助发现漏标/错标。

跟现有 analyzers.py 的"判断分析可插拔框架"是同一层级的东西，但服务的对象不同：
analyzers.py 面向的是 capture_report.csv 里已有的一帧记录，本模块面向的是
cluster_detector.py 产出的候选聚类，两者未来可能会通过 ANALYZER_REGISTRY
以类似方式接入，但这是后续阶段的事，本次不做任何接入。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClassificationResult:
    """一次分类的结果占位结构。"""
    predicted_class: str
    confidence: float
    reason: str


def classify_cluster(
    cluster,
    crop_image: str | None = None,
    bbox: dict | None = None,
    dimension: dict | None = None,
    intensity=None,
) -> ClassificationResult:
    """对一个候选聚类做视觉分类。

    Phase 1 不实现，只定义签名。
        cluster:     cluster_detector.ClusterCandidate 实例
        crop_image:  对应截图上裁剪出来的局部图像路径（复用 capture.py 已经生成的截图，
                     不重新截图）
        bbox:        如果这个候选跟某个现有 BBox 相关，传入 bbox_data.csv 里对应的行
                     （复用现有输出，不重新定义 BBox 结构）
        dimension:   vehicle_dimension_config.get_dimension() 的结果，供模型参考尺寸范围
        intensity:   点云强度统计（比如均值/分布），辅助区分材质相近的类别

    输出（未来实现后）：
        ClassificationResult(predicted_class, confidence, reason)
    """
    raise NotImplementedError(
        "classify_cluster 尚未实现——Phase 1 只建立接口，不调用任何模型/API。"
    )
