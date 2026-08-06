"""QP Copilot V1.0 pipeline entry point.

The UI supplies Scene, Frame Count, and Calibration once. This module connects the
existing Edge session, collects every-frame BBox data and sampled Assets, releases the
browser session, then orchestrates existing Rule, Cluster/Projection, Vision, and Report
modules. Module failures are isolated where recovery is possible; final findings are
merged into rule_report.csv and rule_summary.json.
"""

from __future__ import annotations

import argparse
import copy
import shutil
import sys
from pathlib import Path

import yaml

from analyzers import run_analyzers
from assets_downloader import AssetsDownloader
from bbox_extractor import extract_bbox, save_mapping_report, write_bbox_csv
from bbox_probe import probe_frame, build_probe_report, save_probe_report
from browser import BrowserSession
from calibration import CalibrationError, CalibrationManager, load_calibration
from capture import take_frame_screenshot
from frame_ready import wait_for_frame_ready
from navigator import FrameNavigationError, FrameNavigator
from network_recorder import NetworkRecorder, write_asset_csv
from projection_pipeline import clear_frame_projection_output, run_frame_projection
from report import build_frame_record, write_report
from rule_engine import missing_bbox_finding, print_rule_summary, run_rules, write_rule_report, write_rule_summary
from runtime_env import load_runtime_environment
from vision_pipeline import print_vision_stats, run_vision_pipeline


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prompt_scene_id(config: dict) -> str:
    """场景 ID 用于截图/报告命名。config 里没填的话，运行时手动输入。"""
    scene_id = (config["scene"].get("scene_id") or "").strip()
    if scene_id:
        return scene_id

    scene_id = input("请输入 scene_id（用于截图和报告命名，可留空）: ").strip()
    return scene_id or "scene"


def prompt_frame_count(config: dict) -> int:
    """询问本次场景的总帧数。直接回车用 config.yaml 里 scene.frame_count 的默认值。

    切帧主要靠"可见文本+底部区域过滤"动态定位，不依赖固定坐标，帧数变化不影响这条路径；
    只有文本方式找不到候选、触发坐标 fallback 时才可能受帧栏布局变化影响（见 README）。
    """
    default = config["scene"].get("frame_count", 81)
    raw = input(f"请输入总帧数（默认{default}）：").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        print(f"[警告] 输入的不是数字，使用默认值 {default}")
        return default
    if value <= 0:
        print(f"[警告] 帧数必须大于 0，使用默认值 {default}")
        return default
    return value


def confirm_ready_to_start(nav_mode: str) -> None:
    print()
    print("请确认：")
    print("  1) 已经手动登录 QP 平台")
    print("  2) 已经打开目标 scene 的质检页面")
    print(f"  3) navigation.mode = {nav_mode}：脚本将从第 2 帧开始自动切帧")
    print("  4) 脚本运行期间不会点击『合格/驳回/提交』等按钮，只会截图、翻页和只读探测")
    input("准备好后按 Enter 开始自动浏览...")


def save_used_calibration(
    calibration_path: str,
    assets_cfg: dict,
    scene_id: str,
) -> str:
    """保存本次实际使用的 UI Calibration，供投影结果复盘。"""
    scene_dir = Path(assets_cfg.get("output_dir", "assets")) / f"scene_{scene_id}"
    scene_dir.mkdir(parents=True, exist_ok=True)
    target = scene_dir / "calibration_used.yaml"
    shutil.copyfile(calibration_path, target)
    return str(target)


PIPELINE_STAGE_COUNT = 9


def log_stage(index: int, message: str) -> None:
    print(f"[{index}/{PIPELINE_STAGE_COUNT}] {message}", flush=True)


def build_rule_run_config(config: dict, ai_vision_enabled: bool) -> dict:
    """Return a per-run config without mutating config.yaml-derived data.

    Projection already performs residual Cluster Detection for Vision. When Vision is
    enabled, running PossibleMissingAnnotation again inside Rule Engine would duplicate
    that expensive work and its geometric findings would be discarded anyway.
    """
    if not ai_vision_enabled:
        return config
    rule_config = copy.deepcopy(config)
    enabled = rule_config.get("rule_engine", {}).get("enabled_rules", []) or []
    rule_config["rule_engine"]["enabled_rules"] = [
        name for name in enabled if name != "PossibleMissingAnnotation"
    ]
    return rule_config


def run_rules_with_bbox_status(
    bbox_csv_path: str,
    scene_id: str,
    config: dict,
    expected_frame_indices: list[int],
    missing_bbox_indices: set[int],
):
    """Run existing rules while keeping MissingBBox distinct from EmptyFrame.

    BrokenTrack still receives the complete expected frame range so whole-frame read
    failures suppress per-track false positives. EmptyFrame is evaluated separately
    only for frames whose BBox source was read successfully.
    """
    enabled = config.get("rule_engine", {}).get("enabled_rules", []) or []
    if not missing_bbox_indices or "EmptyFrame" not in enabled:
        return run_rules(bbox_csv_path, scene_id, config, expected_frame_indices)

    non_empty_config = copy.deepcopy(config)
    non_empty_config["rule_engine"]["enabled_rules"] = [
        name for name in enabled if name != "EmptyFrame"
    ]
    findings = run_rules(
        bbox_csv_path,
        scene_id,
        non_empty_config,
        expected_frame_indices,
    )

    readable_frames = [
        frame_index for frame_index in expected_frame_indices
        if frame_index not in missing_bbox_indices
    ]
    if readable_frames:
        empty_config = copy.deepcopy(config)
        empty_config["rule_engine"]["enabled_rules"] = ["EmptyFrame"]
        findings.extend(run_rules(
            bbox_csv_path,
            scene_id,
            empty_config,
            readable_frames,
        ))
    findings.sort(key=lambda finding: (finding.frame_index, finding.track_id, finding.rule_id))
    return findings


def run(config_path: str = "config.yaml", calibration_path: str | None = None) -> None:
    log_stage(1, "Loading Configuration")
    config = load_config(config_path)

    projection_cfg = config.get("projection", {}) or {}
    ai_input_cfg = config.get("ai_inputs", {}) or {}
    ai_vision_cfg = config.get("ai_vision", {}) or {}
    api_key_env = str(ai_vision_cfg.get("api_key_env", "DASHSCOPE_API_KEY"))
    env_status = load_runtime_environment(Path(config_path).resolve().parent, api_key_env)
    if ai_vision_cfg.get("enabled", False):
        if env_status.api_key_available:
            source = ".env/环境变量" if env_status.env_file_exists else "环境变量"
            print(f"[信息] Vision API Key 已从{source}加载: {api_key_env}")
        else:
            print(f"[错误] {env_status.error_message}；Vision 将跳过，其他规则和报告继续执行。")

    calibration: CalibrationManager | None = None
    if projection_cfg.get("enabled", False):
        if calibration_path:
            try:
                calibration = load_calibration(calibration_path)
                print(f"[信息] Calibration 已读取: {len(calibration.get_all_cameras())} 个 Camera")
            except (CalibrationError, OSError, TypeError, ValueError) as exc:
                print(f"[错误] Calibration YAML 读取失败: {exc}；Projection/Vision 将跳过。")
        else:
            print("[错误] 未提供 Calibration YAML；Projection/Vision 将跳过，其他规则继续执行。")

    scene_id = prompt_scene_id(config)
    frame_count = prompt_frame_count(config)
    start_index = config["scene"]["start_frame_index"]
    scene_frame_indices = list(range(start_index, start_index + frame_count))
    skip_first_frame = bool(config["scene"].get("skip_first_frame", False))
    inspection_frame_indices = scene_frame_indices[1:] if skip_first_frame else scene_frame_indices
    sampling_start_index = (
        inspection_frame_indices[0] if inspection_frame_indices else start_index
    )
    inspection_frame_count = len(inspection_frame_indices)
    if skip_first_frame and scene_frame_indices:
        print(
            f"[信息] 跳过首帧 {scene_frame_indices[0]}：不导航、不等待、不采集；"
            f"从第 {sampling_start_index} 帧开始质检。"
        )
    screenshot_cfg = config["screenshot"]
    report_cfg = config["report"]
    network_cfg = config["network"]
    frame_ready_cfg = config["frame_ready"]
    bbox_cfg = config["bbox_probe"]
    bbox_extract_cfg = config.get("bbox_extract", {})
    rule_engine_cfg = config.get("rule_engine", {})
    assets_cfg = config.get("assets", {})
    expected_pcd = network_cfg.get("expected_pcd_count", 1)
    expected_jpg = network_cfg.get("expected_jpg_count", 5)

    if calibration and calibration_path:
        try:
            saved = save_used_calibration(calibration_path, assets_cfg, scene_id)
            print(f"[信息] 本次 Calibration 已保存: {saved}")
        except OSError as exc:
            print(f"[警告] Calibration 归档失败（不影响本次内存中的标定）: {exc}")

    confirm_ready_to_start(config["navigation"].get("mode", "visible_text_bottom"))

    records = []
    asset_records = []
    probe_results = []
    bbox_records = []
    missing_bbox_frames: list[tuple[int, str]] = []
    projection_jobs = []
    has_probed_once = False
    recorder = None
    assets_downloader = None
    capture_available = False

    def mark_missing_bbox(frame_index: int, reason: str) -> None:
        if not bbox_extract_cfg.get("enabled", True):
            return
        if any(item[0] == frame_index for item in missing_bbox_frames):
            return
        missing_bbox_frames.append((frame_index, reason))

    log_stage(2, "Connecting Edge Browser")
    session = BrowserSession(config)
    page = None
    try:
        page = session.start()
        capture_available = True
        print(f"[信息] 已连接页面: {page.url}")
    except Exception as exc:
        print(f"[错误] 浏览器连接失败: {exc}；将生成 MissingBBox 报告。")
        for frame_index in inspection_frame_indices:
            mark_missing_bbox(frame_index, f"浏览器连接失败: {exc}")
            records.append(build_frame_record(
                scene_id=scene_id,
                frame_index=frame_index,
                screenshot_path="",
                navigation_status="warning",
                ready_status="error",
                pcd_count=0,
                jpg_count=0,
                warning=f"浏览器连接失败: {exc}",
            ))

    log_stage(3, "Collecting Frames, BBox and Assets")
    if page is not None:
        try:
            navigator = FrameNavigator(page, config)
            recorder = NetworkRecorder(page, network_cfg) if network_cfg.get("enabled", True) else None
            assets_downloader = (
                AssetsDownloader(page, assets_cfg) if assets_cfg.get("enabled", False) else None
            )
        except Exception as exc:
            capture_available = False
            print(f"[错误] 采集模块初始化失败: {exc}；将生成 MissingBBox 报告。")
            for frame_index in inspection_frame_indices:
                mark_missing_bbox(frame_index, f"采集模块初始化失败: {exc}")
                records.append(build_frame_record(
                    scene_id=scene_id,
                    frame_index=frame_index,
                    screenshot_path="",
                    navigation_status="warning",
                    ready_status="error",
                    pcd_count=0,
                    jpg_count=0,
                    warning=f"采集模块初始化失败: {exc}",
                ))
        else:
            print(
                f"[信息] 开始自动浏览 {inspection_frame_count} 个质检帧"
                f"（scene_id={scene_id}，Scene 总帧数={frame_count}）..."
            )
            for offset, frame_index in enumerate(inspection_frame_indices):
                is_first = not skip_first_frame and offset == 0
                warnings = []
                navigation_status = "ok"
                ready_status = "ok"
                screenshot_path = ""
                screenshot_success = False
                pcd_count = 0
                jpg_count = 0
                frame_bbox_records = None
                asset_record = None
                print(
                    f"[进度] 第 {offset + 1}/{inspection_frame_count} 帧 "
                    f"(frame_index={frame_index}) ...",
                    end=" ",
                )

                try:
                    if recorder:
                        recorder.mark_frame_start()

                    try:
                        nav_result = navigator.goto_frame(frame_index, is_first=is_first)
                        if nav_result.used_fallback:
                            navigation_status = "warning"
                            warnings.append(nav_result.warning)
                    except FrameNavigationError as exc:
                        navigation_status = "warning"
                        warnings.append(f"切帧失败: {exc}")
                        print(f"\n[警告] 第 {frame_index} 帧切帧失败: {exc}")

                    try:
                        if recorder:
                            ready_result = wait_for_frame_ready(page, recorder, frame_ready_cfg)
                            ready_status = ready_result.ready_status
                            if ready_status != "ok":
                                warnings.append(f"就绪等待超时: {ready_result.reason}")
                                print(f"\n[警告] 第 {frame_index} 帧就绪等待超时: {ready_result.reason}")
                        else:
                            page.wait_for_timeout(frame_ready_cfg.get("render_settle_ms", 500))
                    except Exception as exc:
                        ready_status = "error"
                        warnings.append(f"等待页面就绪失败: {exc}")
                        print(f"\n[警告] 第 {frame_index} 帧等待页面就绪失败: {exc}")

                    try:
                        screenshot_result = take_frame_screenshot(
                            page, scene_id, frame_index, screenshot_cfg
                        )
                        screenshot_success = screenshot_result.success
                        screenshot_path = (
                            screenshot_result.screenshot_path if screenshot_result.success else ""
                        )
                        if screenshot_result.success:
                            print(f"完成 -> {screenshot_path}")
                        else:
                            warnings.append(f"截图失败: {screenshot_result.error_message}")
                    except Exception as exc:
                        warnings.append(f"截图异常: {exc}")
                        print(f"\n[警告] 第 {frame_index} 帧截图异常: {exc}")

                    if recorder:
                        try:
                            asset_record = recorder.snapshot(scene_id, frame_index)
                            asset_records.append(asset_record)
                            pcd_count, jpg_count = asset_record.pcd_count, asset_record.jpg_count
                            if asset_record.status(expected_pcd, expected_jpg) == "warning":
                                warning_text = asset_record.warning_text(expected_pcd, expected_jpg)
                                warnings.append(warning_text)
                                print(f"  [警告] 第 {frame_index} 帧资源数量异常：{warning_text}")
                        except Exception as exc:
                            warnings.append(f"网络资源记录失败: {exc}")
                            print(f"  [警告] 第 {frame_index} 帧网络资源记录失败: {exc}")

                    probe_mode = bbox_cfg.get("mode", "disabled")
                    should_probe = bbox_cfg.get("enabled", False) and screenshot_success and (
                        probe_mode == "every_frame"
                        or (probe_mode == "first_frame_only" and not has_probed_once)
                    )
                    if should_probe:
                        try:
                            probe_results.append(probe_frame(page, frame_index, bbox_cfg))
                            has_probed_once = True
                        except Exception as exc:
                            print(f"  [警告] 第 {frame_index} 帧 BBox 探测失败: {exc}")

                    if bbox_extract_cfg.get("enabled", True):
                        try:
                            bbox_result = extract_bbox(page, scene_id, frame_index)
                            if bbox_result.success:
                                frame_bbox_records = bbox_result.records
                                bbox_records.extend(bbox_result.records)
                            else:
                                mark_missing_bbox(frame_index, bbox_result.error)
                        except Exception as exc:
                            mark_missing_bbox(frame_index, f"BBox 提取异常: {exc}")
                            print(f"  [警告] 第 {frame_index} 帧 BBox 提取异常: {exc}")

                    if (
                        assets_downloader
                        and asset_record is not None
                        and assets_downloader.should_download(frame_index, sampling_start_index)
                    ):
                        frame_dir = (
                            assets_downloader.output_dir
                            / f"scene_{scene_id}"
                            / f"frame_{frame_index:04d}"
                        )
                        if calibration:
                            try:
                                clear_frame_projection_output(frame_dir, projection_cfg)
                            except Exception as exc:
                                print(f"  [警告] 第 {frame_index} 帧旧 Projection 清理失败: {exc}")
                        try:
                            download_result = assets_downloader.download_frame(
                                scene_id, frame_index, asset_record
                            )
                            warnings.extend(download_result.errors)
                        except Exception as exc:
                            download_result = None
                            warnings.append(f"Assets 下载异常: {exc}")
                            print(f"  [警告] 第 {frame_index} 帧 Assets 下载异常: {exc}")

                        if calibration and download_result is not None:
                            if frame_bbox_records is None:
                                print(
                                    f"  [警告] 第 {frame_index} 帧 BBox 未成功读取，"
                                    "不加入 Projection 队列。"
                                )
                            elif download_result.image_count == 0:
                                print(f"  [警告] 第 {frame_index} 帧 JPG 下载失败，跳过 Projection。")
                            else:
                                if not download_result.pcd_success:
                                    print(
                                        f"  [警告] 第 {frame_index} 帧 PCD 下载失败，"
                                        "漏标 Cluster 将跳过，错标 BBox 仍继续。"
                                    )
                                projection_jobs.append((
                                    frame_index,
                                    frame_dir,
                                    tuple(frame_bbox_records),
                                ))
                except Exception as exc:
                    warnings.append(f"单帧未预期异常: {exc}")
                    if frame_bbox_records is None:
                        mark_missing_bbox(frame_index, f"单帧未预期异常: {exc}")
                    print(f"\n[警告] 第 {frame_index} 帧未预期异常，继续下一帧: {exc}")
                finally:
                    records.append(build_frame_record(
                        scene_id=scene_id,
                        frame_index=frame_index,
                        screenshot_path=screenshot_path,
                        navigation_status=navigation_status,
                        ready_status=ready_status,
                        pcd_count=pcd_count,
                        jpg_count=jpg_count,
                        warning="; ".join(filter(None, warnings)),
                    ))

            ok_count = sum(1 for record in records if not record.warning)
            print(
                f"[信息] 自动浏览结束，共 {len(records)} 帧，其中 {ok_count} 帧完全正常，"
                f"{len(records) - ok_count} 帧有 warning。"
            )

    try:
        session.close(close_browser=False)
    except Exception as exc:
        print(f"[警告] 浏览器会话释放失败（不关闭用户 Edge）: {exc}")

    log_stage(4, "Writing Capture and BBox Data")
    if recorder:
        try:
            failed = recorder.total_failed_requests()
            if failed:
                print(f"[信息] 网络层失败请求: {len(failed)} 个。")
        except Exception as exc:
            print(f"[警告] 网络失败请求统计失败: {exc}")
        if asset_records:
            try:
                path = write_asset_csv(asset_records, network_cfg, report_cfg)
                print(f"[信息] 网络资源记录已生成: {path}")
            except Exception as exc:
                print(f"[警告] 网络资源记录写入失败: {exc}")

    if assets_downloader:
        try:
            path = assets_downloader.write_metadata(scene_id)
            print(f"[信息] Assets metadata 已生成: {path}")
        except Exception as exc:
            print(f"[警告] Assets metadata 写入失败: {exc}")

    if probe_results:
        try:
            probe_report = build_probe_report(probe_results, scene_id)
            path = save_probe_report(probe_report, bbox_cfg, report_cfg)
            print(f"[信息] BBox 探测报告已生成: {path}")
        except Exception as exc:
            print(f"[警告] BBox 探测报告生成失败: {exc}")

    bbox_csv_path = None
    if bbox_extract_cfg.get("enabled", True):
        try:
            bbox_csv_path = write_bbox_csv(bbox_records, report_cfg, bbox_extract_cfg)
            print(f"[信息] BBox 几何数据已生成: {bbox_csv_path}（共 {len(bbox_records)} 条）")
        except Exception as exc:
            print(f"[错误] BBox CSV 写入失败: {exc}")
        try:
            mapping_path = save_mapping_report(bbox_records, report_cfg)
            print(f"[信息] 字段映射报告已生成: {mapping_path}")
        except Exception as exc:
            print(f"[警告] 字段映射报告写入失败: {exc}")

    try:
        analyzed_records = [run_analyzers(record, config) for record in records]
    except Exception as exc:
        analyzed_records = records
        print(f"[警告] Analyzer 运行失败，保留原始采集记录: {exc}")
    try:
        capture_report_path = write_report(analyzed_records, report_cfg)
        print(f"[信息] 采集诊断报告已生成: {capture_report_path}")
    except Exception as exc:
        print(f"[警告] 采集诊断报告写入失败: {exc}")

    log_stage(5, "Running Rule Engine")
    findings = []
    if bbox_csv_path and capture_available:
        try:
            rule_run_config = build_rule_run_config(
                config, bool(ai_vision_cfg.get("enabled", False))
            )
            findings = run_rules_with_bbox_status(
                bbox_csv_path,
                scene_id,
                rule_run_config,
                inspection_frame_indices,
                {frame_index for frame_index, _ in missing_bbox_frames},
            )
            print(f"[信息] 普通规则命中 {len(findings)} 条。")
        except Exception as exc:
            print(f"[错误] Rule Engine 失败: {exc}；继续 Projection/Vision 和最终报告。")
    elif not capture_available:
        print("[警告] 浏览器采集不可用，跳过常规规则，使用 MissingBBox 记录。")
    else:
        print("[警告] BBox CSV 不可用，跳过常规规则。")

    log_stage(6, "Running Cluster Detection and Projection")
    vision_tasks = []
    if calibration:
        for frame_index, frame_dir, frame_bbox_records in projection_jobs:
            try:
                result = run_frame_projection(
                    frame_index=frame_index,
                    frame_dir=frame_dir,
                    bbox_records=frame_bbox_records,
                    calibration=calibration,
                    cluster_config=rule_engine_cfg.get("cluster_detector", {}) or {},
                    projection_config=projection_cfg,
                    scene_id=scene_id,
                )
                if result.skipped:
                    print(f"  [警告] 第 {frame_index} 帧 Projection 跳过: {result.skipped_reason}")
                    continue
                if result.cluster_skipped_reason:
                    print(
                        f"  [警告] 第 {frame_index} 帧 Cluster 跳过: "
                        f"{result.cluster_skipped_reason}；BBox Vision 继续。"
                    )
                vision_tasks.extend(result.vision_tasks)
                print(
                    f"  [投影] 第 {frame_index} 帧: Cluster {result.cluster_count}, "
                    f"Cluster 可见 ROI {result.visible_roi_count}, "
                    f"BBox {result.bbox_count}, BBox 可见 ROI {result.bbox_visible_roi_count}, "
                    f"Debug JPG {len(result.saved_images)}"
                )
            except Exception as exc:
                print(f"  [警告] 第 {frame_index} 帧 Cluster/Projection 失败，继续下一帧: {exc}")
    else:
        print("[警告] Calibration 不可用，Cluster Projection 已跳过。")

    log_stage(7, "Running Vision Verification")
    vision_result = None
    if not ai_input_cfg.get("enabled", False):
        print("[警告] ai_inputs 未启用，Vision 已跳过。")
    elif ai_vision_cfg.get("enabled", False) and not env_status.api_key_available:
        print(f"[错误] {env_status.error_message}；Vision 已跳过。")
    else:
        try:
            cluster_count = sum(
                task.candidate_type == "residual_cluster" for task in vision_tasks
            )
            bbox_count = sum(task.candidate_type == "existing_bbox" for task in vision_tasks)
            print(
                f"[信息] 准备单 Candidate Vision 输入"
                f"（Cluster {cluster_count}，BBox {bbox_count}）..."
            )
            vision_result = run_vision_pipeline(vision_tasks, ai_input_cfg, ai_vision_cfg)
            print_vision_stats(vision_result.stats)
        except Exception as exc:
            print(f"[错误] Vision Pipeline 失败: {exc}；继续生成最终报告。")

    log_stage(8, "Merging Rules and Writing Final Report")
    if vision_result is not None:
        findings.extend(vision_result.findings)
    for frame_index, reason in missing_bbox_frames:
        findings.append(missing_bbox_finding(scene_id, frame_index, reason))

    final_errors = []
    try:
        rule_report_path = write_rule_report(findings, report_cfg, rule_engine_cfg)
        print(f"[信息] 最终规则报告已生成，共 {len(findings)} 条: {rule_report_path}")
    except Exception as exc:
        final_errors.append(f"rule_report.csv: {exc}")
        print(f"[错误] rule_report.csv 写入失败: {exc}")

    if bbox_csv_path:
        try:
            summary_path, summary = write_rule_summary(
                bbox_csv_path,
                findings,
                [frame_index for frame_index, _ in missing_bbox_frames],
                scene_id,
                report_cfg,
                scene_frame_indices,
                vision_result.stats if vision_result is not None else None,
            )
            print(f"[信息] 最终规则汇总已生成: {summary_path}")
            print_rule_summary(summary)
        except Exception as exc:
            final_errors.append(f"rule_summary.json: {exc}")
            print(f"[错误] rule_summary.json 写入失败: {exc}")
    else:
        final_errors.append("rule_summary.json: bbox_data.csv 不可用")
        print("[错误] bbox_data.csv 不可用，无法生成 rule_summary.json。")

    log_stage(9, "Inspection Finished")
    if final_errors:
        raise RuntimeError("最终报告未完整生成: " + "; ".join(final_errors))
    print("Inspection Finished.")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QP Copilot inspection runner")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--calibration", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(config_path=args.config, calibration_path=args.calibration)
