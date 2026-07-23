"""
assets_downloader.py
AI Quality Inspection Phase 1 —— Assets（PCD + JPG）下载器。

跟 BBox 完全无关：BBox 的获取/保存逻辑在 bbox_extractor.py，本模块不碰、不重新实现。

设计原则（复用已有 Network Listener，不新增浏览器监听）：
  main.py 每一帧已经会调用 network_recorder.NetworkRecorder.snapshot(scene_id, frame_index)，
  拿到这一帧真正认领到的 AssetRecord（pcd_urls / jpg_urls，已经按时间戳分组算好，
  见 network_recorder.py）。本模块直接消费这个现成的 AssetRecord，不再挂第二个
  page.on("response") 监听器，也不重新判断"这个 URL 属于哪一帧"——那件事
  NetworkRecorder 已经做过了。

下载方式：page.context.request.get(url)（Playwright 的 APIRequestContext），
跟当前页面共用同一个浏览器 context，自动带上已登录的 cookie/session，
不需要另外处理账号密码，也不是"重新设计一套浏览器自动化"。

sample_interval 只控制"这一帧要不要下载 PCD/JPG 文件本体"，跟 BBox 采集、
Rule Engine、Report 完全无关——那几个模块每一帧都照常跑，不受这里影响。

保存目录（跟 outputs/ 完全独立，不放进 Report 也不放进 BBox 文件夹）：
  assets/
    scene_<scene_id>/
      metadata.json
      frame_0001/
        pointcloud.pcd
        images/
          camera_xxx.jpg
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from playwright.sync_api import Page

from network_recorder import AssetRecord

# URL 路径里形如 /camera/CAM_FRONT_MID/... 的段，尽量提取出语义化的相机名字用于命名 JPG。
# 提取不到时退化成 camera_1.jpg / camera_2.jpg ...（按认领顺序编号），不报错、不影响下载。
_CAMERA_NAME_RE = re.compile(r"/camera/([A-Za-z0-9_\-]+)/", re.IGNORECASE)


def should_download_frame(frame_index: int, start_frame_index: int, sample_interval: int) -> bool:
    """按 sample_interval 决定这一帧要不要下载 PCD/JPG 文件本体。

    规则：(frame_index - start_frame_index) % sample_interval == 0。
    例如 81 帧、start_frame_index=1、sample_interval=10 时，命中 1,11,21,...,81，
    跟需求里给的例子完全一致。sample_interval=1 时每一帧都下载。

    这个函数只影响 Assets 下载，不影响 BBox 采集 / Rule Engine / Report——那几个
    模块的循环完全不读这个函数的返回值。
    """
    if sample_interval <= 1:
        return True
    return (frame_index - start_frame_index) % sample_interval == 0


@dataclass
class FrameDownloadResult:
    """一帧 Assets 下载的结果，同时用于 metadata.json 的一条记录和终端日志打印，
    避免"文件里记一遍、日志里又单独拼一遍字段"这种重复。"""
    scene_id: str
    frame_index: int
    pcd_success: bool = False
    image_count: int = 0
    image_total_expected: int = 0
    download_time: str = ""
    errors: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "frame_index": self.frame_index,
            "pcd_success": self.pcd_success,
            "image_count": self.image_count,
            "image_total_expected": self.image_total_expected,
            "download_time": self.download_time,
            "errors": self.errors,
        }


def _extract_camera_name(url: str, fallback_index: int) -> str:
    match = _CAMERA_NAME_RE.search(url)
    if match:
        return match.group(1).lower()
    return f"camera_{fallback_index}"


class AssetsDownloader:
    """复用已连接的 Page（及其 BrowserContext）下载 PCD/JPG 文件本体。

    不重新连接浏览器、不重新登录、不新增 page.on("response") 监听——
    URL 列表全部来自调用方已经拿到的 AssetRecord（network_recorder.py 的产出）。
    """

    def __init__(self, page: Page, assets_cfg: dict):
        self.page = page
        self.cfg = assets_cfg
        self.output_dir = Path(assets_cfg.get("output_dir", "assets"))
        self.download_pcd = assets_cfg.get("download_pcd", True)
        self.download_jpg = assets_cfg.get("download_jpg", True)
        self.timeout_ms = assets_cfg.get("request_timeout_ms", 30000)
        self.sample_interval = max(1, assets_cfg.get("sample_interval", 10))
        self._results: list[FrameDownloadResult] = []

    def should_download(self, frame_index: int, start_frame_index: int) -> bool:
        return should_download_frame(frame_index, start_frame_index, self.sample_interval)

    def download_frame(self, scene_id: str, frame_index: int, asset_record: AssetRecord) -> FrameDownloadResult:
        """下载这一帧 AssetRecord 里记录的 PCD/JPG，写入 assets/scene_<scene_id>/frame_NNNN/。
        单个文件下载失败只记进 errors，不抛异常，不影响其它文件/其它帧。"""
        frame_dir = self.output_dir / f"scene_{scene_id}" / f"frame_{frame_index:04d}"
        images_dir = frame_dir / "images"

        result = FrameDownloadResult(
            scene_id=scene_id,
            frame_index=frame_index,
            image_total_expected=len(asset_record.jpg_urls) if self.download_jpg else 0,
            download_time=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        if self.download_pcd and asset_record.pcd_urls:
            frame_dir.mkdir(parents=True, exist_ok=True)
            # 理论上每帧只有 1 个 PCD；万一平台某帧异常返回了多个，只取第一个作为 pointcloud.pcd，
            # 不改变 AssetRecord 本身、也不影响 network_assets.csv 里已经记录的完整 URL 列表。
            pcd_url = asset_record.pcd_urls[0]
            try:
                self._download_to_file(pcd_url, frame_dir / "pointcloud.pcd")
                result.pcd_success = True
            except Exception as exc:
                result.errors.append(f"PCD 下载失败 ({pcd_url}): {exc}")

        if self.download_jpg and asset_record.jpg_urls:
            images_dir.mkdir(parents=True, exist_ok=True)
            for i, jpg_url in enumerate(asset_record.jpg_urls, start=1):
                camera_name = _extract_camera_name(jpg_url, i)
                try:
                    self._download_to_file(jpg_url, images_dir / f"{camera_name}.jpg")
                    result.image_count += 1
                except Exception as exc:
                    result.errors.append(f"JPG 下载失败 ({jpg_url}): {exc}")

        self._results.append(result)
        print(
            f"[资源下载] 第 {frame_index} 帧: "
            f"PCD {'成功' if result.pcd_success else '失败/跳过'}, "
            f"JPG {result.image_count}/{result.image_total_expected}"
            + (f", {len(result.errors)} 个错误" if result.errors else "")
        )
        return result

    def _download_to_file(self, url: str, filepath: Path) -> None:
        response = self.page.context.request.get(url, timeout=self.timeout_ms)
        if not response.ok:
            raise RuntimeError(f"HTTP {response.status}")
        filepath.write_bytes(response.body())

    def write_metadata(self, scene_id: str) -> str:
        """把这个场景所有已下载帧的结果汇总写成 assets/scene_<scene_id>/metadata.json。"""
        scene_dir = self.output_dir / f"scene_{scene_id}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        filepath = scene_dir / "metadata.json"

        metadata = {
            "scene_id": scene_id,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sample_interval": self.sample_interval,
            "frames": [r.as_dict() for r in self._results],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return str(filepath)
