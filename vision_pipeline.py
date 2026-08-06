"""Orchestrate one-candidate Vision verification for missing and wrong labels."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from ai_input_builder import (
    PreparedVisionInput,
    assess_projection_quality,
    prepare_vision_input,
    update_manifest,
)
from rule_engine import RuleFinding
from temporal_cluster_dedup import deduplicate_stationary_clusters
from vision_cache import VisionResultCache
from vision_candidate import EXISTING_BBOX, RESIDUAL_CLUSTER, VisionCandidate
from vision_verifier import (
    QwenVLVerifier,
    VisionVerificationResult,
    WrongAnnotationVerificationResult,
)
from vision_category_config import load_vision_categories
from vision_quality_gate import assess_category_size, normalize_category


@dataclass(frozen=True)
class VisionRunResult:
    findings: tuple[RuleFinding, ...]
    prepared_inputs: tuple[PreparedVisionInput, ...]
    verifications: tuple[object, ...]
    stats: dict


def run_vision_pipeline(
    tasks: list[VisionCandidate],
    input_config: dict,
    vision_config: dict,
    verifier: QwenVLVerifier | None = None,
) -> VisionRunResult:
    vision_started = time.perf_counter()
    cluster_tasks = [task for task in tasks if task.candidate_type == RESIDUAL_CLUSTER]
    bbox_tasks = [task for task in tasks if task.candidate_type == EXISTING_BBOX]
    cluster_limit = int(vision_config.get("vision_test_limit", 20))
    wrong_config = vision_config.get("wrong_annotation", {}) or {}
    missing_config = vision_config.get("missing_annotation", {}) or {}
    bbox_limit = int(wrong_config.get("vision_test_limit", 0))
    temporal_result = deduplicate_stationary_clusters(
        cluster_tasks,
        bbox_tasks,
        missing_config.get("temporal_dedup", {}) or {},
        input_config,
    )
    selected = select_vision_test_tasks(
        list(temporal_result.candidates), cluster_limit, input_config
    )
    selected.extend(select_bbox_vision_tasks(bbox_tasks, bbox_limit, input_config))
    enabled = bool(vision_config.get("enabled", False))
    missing_confidence_threshold = float(missing_config.get(
        "confidence_threshold",
        vision_config.get("uncertain_confidence_threshold", 0.9),
    ))
    category_path = vision_config.get("category_whitelist_path")
    categories = load_vision_categories(category_path) if category_path else load_vision_categories()

    stats = {
        "raw_cluster_candidates": temporal_result.raw_candidates,
        "temporal_dedup_groups": temporal_result.duplicate_groups,
        "temporal_dedup_observations": temporal_result.duplicate_observations,
        "temporal_dedup_saved_calls": temporal_result.saved_calls,
        "temporal_registered_frames": temporal_result.registered_frames,
        "temporal_unregistered_frames": list(temporal_result.unregistered_frames),
        "total_test_clusters": len([
            task for task in selected if task.candidate_type == RESIDUAL_CLUSTER
        ]),
        "ai_real_objects": 0,
        "ai_background_or_noise": 0,
        "ai_out_of_scope": 0,
        "uncertain": 0,
        "report_findings": 0,
        "api_failures": 0,
        "api_skipped": 0,
        "projection_low_quality": 0,
        "input_failures": 0,
        "category_size_rejected": 0,
        "wrong_inconsistent_rejected": 0,
        "wrong_consensus_rejected": 0,
        "total_wrong_annotation_candidates": len([
            task for task in selected if task.candidate_type == EXISTING_BBOX
        ]),
        "wrong_label_mismatches": 0,
        "wrong_bbox_mismatches": 0,
        "wrong_unknown": 0,
        "wrong_report_findings": 0,
        "vision_workers": max(1, int(vision_config.get("vision_workers", 2))),
        "vision_cache_hits": 0,
        "missing_cache_hits": 0,
        "wrong_cache_hits": 0,
        "missing_api_requests": 0,
        "wrong_api_requests": 0,
        "api_attempts": 0,
        "api_retries": 0,
        "api_elapsed_seconds": 0.0,
        "request_phase_wall_seconds": 0.0,
        "estimated_parallel_saved_seconds": 0.0,
        "estimated_cache_saved_seconds": 0.0,
        "vision_wall_seconds": 0.0,
    }
    prepared_inputs = []
    verifications = []
    findings = []
    wrong_observations = []
    jobs = []

    for task in selected:
        try:
            prepared = prepare_vision_input(task, input_config)
        except Exception as exc:
            stats["input_failures"] += 1
            print(
                f"  [警告] Frame {task.frame_index} {task.candidate_id} AI 图片生成失败: {exc}"
            )
            continue
        prepared_inputs.append(prepared)

        if prepared.projection_quality != "good":
            stats["projection_low_quality"] += 1
            update_manifest(prepared.manifest_path, "vision", {
                "status": "skipped",
                "error": "projection_low_quality",
            })
            continue
        if not enabled:
            update_manifest(prepared.manifest_path, "vision", {
                "status": "disabled",
                "error": "ai_vision_disabled",
            })
            continue
        if task.candidate_type == EXISTING_BBOX and not wrong_config.get("enabled", True):
            update_manifest(prepared.manifest_path, "vision", {
                "status": "disabled",
                "error": "wrong_annotation_disabled",
            })
            continue

        metadata = _verification_metadata(task)
        image_paths = []
        for context_path, crop_path in zip(prepared.context_paths, prepared.crop_paths):
            image_paths.extend((context_path, crop_path))
        jobs.append(_VerificationJob(
            prepared=prepared,
            metadata=metadata,
            image_paths=tuple(image_paths),
        ))

    cache = VisionResultCache(input_config, vision_config)
    outcomes = []
    if jobs:
        request_phase_started = time.perf_counter()
        active_verifier = verifier or QwenVLVerifier(vision_config)
        owns_verifier = verifier is None
        workers = min(stats["vision_workers"], len(jobs))
        try:
            if workers == 1:
                outcomes = [
                    _run_verification_job(job, active_verifier, cache)
                    for job in jobs
                ]
            else:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    outcomes = list(executor.map(
                        lambda job: _run_verification_job(job, active_verifier, cache),
                        jobs,
                    ))
        finally:
            if owns_verifier:
                active_verifier.close()
        stats["request_phase_wall_seconds"] = time.perf_counter() - request_phase_started

    for outcome in outcomes:
        prepared = outcome.job.prepared
        task = prepared.task
        if outcome.cache_error:
            print(
                f"  [警告] Frame {task.frame_index} {task.candidate_id} "
                f"Vision 缓存异常，已继续处理: {outcome.cache_error}"
            )
        if outcome.error:
            stats["api_failures"] += 1
            update_manifest(prepared.manifest_path, "vision", {
                "status": "error",
                "error": f"verifier_exception:{outcome.error}",
            })
            print(
                f"  [警告] Frame {task.frame_index} {task.candidate_id} "
                f"Vision 调用异常，继续下一 Candidate: {outcome.error}"
            )
            continue

        verification = outcome.verification
        verifications.append(verification)
        update_manifest(prepared.manifest_path, "vision", verification.as_dict())
        update_manifest(prepared.manifest_path, "vision_cache", {
            "hit": outcome.cache_hit,
            "key": outcome.cache_key,
        })
        if outcome.cache_hit:
            stats["vision_cache_hits"] += 1
            stats["estimated_cache_saved_seconds"] += outcome.cached_api_seconds
            if task.candidate_type == EXISTING_BBOX:
                stats["wrong_cache_hits"] += 1
            else:
                stats["missing_cache_hits"] += 1
        elif verification.attempts > 0:
            if task.candidate_type == EXISTING_BBOX:
                stats["wrong_api_requests"] += 1
            else:
                stats["missing_api_requests"] += 1
            stats["api_attempts"] += verification.attempts
            stats["api_retries"] += max(0, verification.attempts - 1)
            stats["api_elapsed_seconds"] += outcome.api_elapsed_seconds

        if verification.status == "error":
            stats["api_failures"] += 1
            continue
        if verification.status != "verified":
            stats["api_skipped"] += 1
            continue

        if task.candidate_type == EXISTING_BBOX:
            wrong_observations.append((prepared, verification))
        elif (
            verification.possible_category == "unknown"
            or verification.confidence < missing_confidence_threshold
        ):
            stats["uncertain"] += 1
        elif not verification.category_in_scope:
            stats["ai_out_of_scope"] += 1
            stats["ai_background_or_noise"] += 1
        elif verification.contains_real_object:
            stats["ai_real_objects"] += 1
        else:
            stats["ai_background_or_noise"] += 1

        if task.candidate_type == RESIDUAL_CLUSTER and (
            verification.category_in_scope
            and verification.contains_real_object
            and verification.missing_annotation_suspected
        ):
            if verification.confidence < missing_confidence_threshold:
                update_manifest(prepared.manifest_path, "vision_decision", {
                    "accepted": False,
                    "reason": "confidence_below_report_threshold",
                    "threshold": missing_confidence_threshold,
                })
                continue
            size_gate = assess_category_size(
                verification.possible_category,
                task.size,
                categories,
                missing_config.get("category_size_gate", {}) or {},
            )
            update_manifest(prepared.manifest_path, "vision_decision", {
                "accepted": size_gate.accepted,
                "reason": size_gate.reason,
                "details": size_gate.details,
            })
            if not size_gate.accepted:
                stats["category_size_rejected"] += 1
                continue
            findings.append(_to_rule_finding(prepared, verification))
            stats["report_findings"] += 1

    _finalize_wrong_annotation_results(
        wrong_observations, wrong_config, stats, findings
    )

    api_requests = stats["missing_api_requests"] + stats["wrong_api_requests"]
    stats["api_requests"] = api_requests
    stats["average_api_seconds"] = (
        stats["api_elapsed_seconds"] / api_requests if api_requests else 0.0
    )
    if api_requests > 1:
        stats["estimated_parallel_saved_seconds"] = max(
            0.0,
            stats["api_elapsed_seconds"] - stats["request_phase_wall_seconds"],
        )
    stats["vision_wall_seconds"] = time.perf_counter() - vision_started
    return VisionRunResult(
        findings=tuple(findings),
        prepared_inputs=tuple(prepared_inputs),
        verifications=tuple(verifications),
        stats=stats,
    )


@dataclass(frozen=True)
class _VerificationJob:
    prepared: PreparedVisionInput
    metadata: dict
    image_paths: tuple[str, ...]


@dataclass(frozen=True)
class _VerificationOutcome:
    job: _VerificationJob
    verification: object | None = None
    cache_key: str = ""
    cache_hit: bool = False
    api_elapsed_seconds: float = 0.0
    cached_api_seconds: float = 0.0
    cache_error: str = ""
    error: str = ""


def _run_verification_job(
    job: _VerificationJob,
    verifier: QwenVLVerifier,
    cache: VisionResultCache,
) -> _VerificationOutcome:
    task = job.prepared.task
    cache_key = ""
    cache_error = ""
    try:
        lookup = cache.lookup(task, job.metadata, job.image_paths)
        cache_key = lookup.key
        if lookup.hit:
            try:
                verification = _verification_from_cache(task, lookup.verification)
            except (TypeError, ValueError):
                verification = None
            if verification is not None:
                return _VerificationOutcome(
                    job=job,
                    verification=verification,
                    cache_key=cache_key,
                    cache_hit=True,
                    cached_api_seconds=lookup.original_api_seconds,
                )
    except Exception as exc:
        cache_error = str(exc)

    started = time.perf_counter()
    try:
        if task.candidate_type == EXISTING_BBOX:
            verification = verifier.verify_bbox(job.metadata, job.image_paths)
        else:
            verification = verifier.verify_cluster(job.metadata, job.image_paths)
    except Exception as exc:
        return _VerificationOutcome(
            job=job,
            cache_key=cache_key,
            api_elapsed_seconds=time.perf_counter() - started,
            cache_error=cache_error,
            error=str(exc),
        )

    elapsed = time.perf_counter() - started
    try:
        cache.save(
            task,
            cache_key,
            verification.as_dict(),
            original_api_seconds=elapsed,
        )
    except Exception as exc:
        cache_error = "; ".join(filter(None, (cache_error, str(exc))))
    return _VerificationOutcome(
        job=job,
        verification=verification,
        cache_key=cache_key,
        api_elapsed_seconds=elapsed,
        cache_error=cache_error,
    )


def _verification_from_cache(task: VisionCandidate, data: dict):
    if task.candidate_type == EXISTING_BBOX:
        return WrongAnnotationVerificationResult(**data)
    return VisionVerificationResult(**data)


def select_vision_test_tasks(
    tasks: list[VisionCandidate],
    limit: int,
    input_config: dict,
) -> list[VisionCandidate]:
    """Select deterministic geometry/quality-diverse tasks before creating images."""
    candidates = [task for task in tasks if task.projections]
    candidates.sort(key=lambda task: (task.frame_index, task.cluster_id))
    if limit <= 0 or len(candidates) <= limit:
        return candidates

    quality_config = input_config.get("quality", {}) or {}
    groups: dict[str, list[VisionCandidate]] = {
        "low_quality": [],
        "near": [],
        "far": [],
        "large": [],
        "thin": [],
        "other": [],
    }
    for task in candidates:
        qualities = [assess_projection_quality(view, quality_config) for view in task.projections]
        if not any(item.quality == "good" for item in qualities):
            groups["low_quality"].append(task)
        if task.distance is not None and task.distance <= 30.0:
            groups["near"].append(task)
        if task.distance is not None and task.distance >= 60.0:
            groups["far"].append(task)
        if task.size is not None and max(task.size) >= 8.0:
            groups["large"].append(task)
        if task.pca_features.get("thickness_ratio", 0.0) <= 0.02:
            groups["thin"].append(task)
        groups["other"].append(task)

    selected = []
    seen = set()
    group_order = ("low_quality", "near", "far", "large", "thin", "other")
    positions = {name: 0 for name in group_order}
    while len(selected) < limit:
        added = False
        for name in group_order:
            group = groups[name]
            while positions[name] < len(group):
                task = group[positions[name]]
                positions[name] += 1
                key = (task.frame_index, task.cluster_id)
                if key in seen:
                    continue
                selected.append(task)
                seen.add(key)
                added = True
                break
            if len(selected) >= limit:
                break
        if not added:
            break
    return selected


def select_bbox_vision_tasks(
    tasks: list[VisionCandidate],
    limit: int,
    input_config: dict,
) -> list[VisionCandidate]:
    """Select one best projected observation for each stable BBox track ID."""
    quality_config = (
        (input_config.get("quality_by_candidate", {}) or {}).get(EXISTING_BBOX)
        or input_config.get("quality", {})
        or {}
    )
    by_track: dict[str, list[VisionCandidate]] = defaultdict(list)
    for task in tasks:
        track_key = str(task.track_id or task.candidate_id)
        by_track[track_key].append(task)

    representatives = [
        max(track_tasks, key=lambda task: _bbox_representative_score(task, quality_config))
        for track_tasks in by_track.values()
    ]
    representatives.sort(
        key=lambda task: (task.frame_index, task.track_id, task.bbox_index, task.candidate_id)
    )
    return representatives if limit <= 0 else representatives[:limit]


def _bbox_representative_score(task: VisionCandidate, quality_config: dict) -> tuple:
    checks = [assess_projection_quality(view, quality_config) for view in task.projections]
    good_cameras = {
        check.camera for check in checks if check.quality == "good"
    }
    good_views = [view for view in task.projections if view.camera in good_cameras]
    roi_areas = [
        max(0, view.roi[2] - view.roi[0]) * max(0, view.roi[3] - view.roi[1])
        for view in good_views
    ]
    return (
        len(good_views),
        sum(view.visible_points for view in good_views),
        max(roi_areas, default=0),
        sum(view.visible_points for view in task.projections),
        -task.frame_index,
    )


def print_vision_stats(stats: dict) -> None:
    print("[信息] Vision 测试统计:")
    print(f"  - 总测试 Cluster: {stats['total_test_clusters']}")
    print(f"  - AI 判断真实目标: {stats['ai_real_objects']}")
    print(f"  - 原始可投影 Cluster: {stats.get('raw_cluster_candidates', 0)}")
    print(f"  - 跨帧静止重复组: {stats.get('temporal_dedup_groups', 0)}")
    print(f"  - 去重节省漏标 API: {stats.get('temporal_dedup_saved_calls', 0)}")
    print(f"  - 自车运动配准帧: {stats.get('temporal_registered_frames', 0)}")
    if stats.get("temporal_unregistered_frames"):
        print(
            "  - 未配准帧（保持逐个检查）: "
            f"{stats['temporal_unregistered_frames']}"
        )
    print(f"  - AI 判断背景/噪声: {stats['ai_background_or_noise']}")
    print(f"  - 表格范围外排除: {stats['ai_out_of_scope']}")
    print(f"  - uncertain: {stats['uncertain']}")
    print(f"  - 成功写入报告: {stats['report_findings']}")
    print(f"  - Vision 并发数: {stats.get('vision_workers', 1)}")
    print(
        f"  - 缓存命中: {stats.get('vision_cache_hits', 0)} "
        f"(漏标 {stats.get('missing_cache_hits', 0)}, "
        f"错标 {stats.get('wrong_cache_hits', 0)})"
    )
    print(
        f"  - 实际 API Candidate: {stats.get('api_requests', 0)} "
        f"(漏标 {stats.get('missing_api_requests', 0)}, "
        f"错标 {stats.get('wrong_api_requests', 0)})"
    )
    print(
        f"  - API 尝试/重试: {stats.get('api_attempts', 0)}/"
        f"{stats.get('api_retries', 0)}"
    )
    print(
        f"  - API 累计/平均耗时: {stats.get('api_elapsed_seconds', 0.0):.1f}s/"
        f"{stats.get('average_api_seconds', 0.0):.2f}s"
    )
    print(
        f"  - 请求阶段墙钟时间: {stats.get('request_phase_wall_seconds', 0.0):.1f}s"
    )
    print(
        f"  - 预计并发/缓存节省: "
        f"{stats.get('estimated_parallel_saved_seconds', 0.0):.1f}s/"
        f"{stats.get('estimated_cache_saved_seconds', 0.0):.1f}s"
    )
    print(f"  - Vision 总耗时: {stats.get('vision_wall_seconds', 0.0):.1f}s")
    print(f"  - API 失败: {stats['api_failures']}")
    print(f"  - API 跳过: {stats['api_skipped']}")
    print(f"  - Projection 低质量: {stats['projection_low_quality']}")
    print(f"  - AI 图片生成失败: {stats['input_failures']}")
    print(f"  - 类别/尺寸矛盾排除: {stats.get('category_size_rejected', 0)}")
    print(f"  - 错标测试 BBox: {stats.get('total_wrong_annotation_candidates', 0)}")
    print(f"  - Label 明显不一致: {stats.get('wrong_label_mismatches', 0)}")
    print(f"  - BBox 明显偏离/无目标: {stats.get('wrong_bbox_mismatches', 0)}")
    print(f"  - 错标 Unknown: {stats.get('wrong_unknown', 0)}")
    print(f"  - 错标自相矛盾排除: {stats.get('wrong_inconsistent_rejected', 0)}")
    print(f"  - 错标多帧一致性不足: {stats.get('wrong_consensus_rejected', 0)}")
    print(f"  - 错标写入报告: {stats.get('wrong_report_findings', 0)}")


def _verification_metadata(task: VisionCandidate) -> dict:
    public_metadata = {
        key: value
        for key, value in task.metadata.items()
        if not str(key).startswith("_")
    }
    metadata = {
        "candidate_type": task.candidate_type,
        "candidate_id": task.candidate_id,
        "scene_id": task.scene_id,
        "frame_index": task.frame_index,
        "position": list(task.position) if task.position is not None else None,
        "size": list(task.size) if task.size is not None else None,
        "rotation": list(task.rotation) if task.rotation is not None else None,
        "distance": task.distance,
        **public_metadata,
    }
    if task.candidate_type == RESIDUAL_CLUSTER:
        metadata["cluster_id"] = task.cluster_id
        metadata["center"] = metadata["position"]
    else:
        metadata.update({
            "track_id": task.track_id,
            "bbox_index": task.bbox_index,
            "current_label": task.current_label,
        })
    return metadata


def _to_rule_finding(
    prepared: PreparedVisionInput,
    verification: VisionVerificationResult,
) -> RuleFinding:
    task = prepared.task
    center = list(task.center) if task.center is not None else None
    size = list(task.size) if task.size is not None else None
    cameras = list(prepared.accepted_cameras)
    rois = [list(roi) for roi in prepared.rois]
    evidence = (
        f"point_count={task.point_count}, center={task.center}, estimated_size={task.size}, "
        f"distance={task.distance}, pca_features={task.pca_features}, "
        f"cluster_confidence={task.cluster_confidence:.4f}, "
        f"merged_fragment_count={task.merged_fragment_count}"
    )
    temporal = task.metadata.get("_temporal_dedup", {}) or {}
    observed_frames = [int(frame) for frame in temporal.get("observed_frames", [])]
    if observed_frames:
        evidence += (
            f", temporal_dedup={temporal.get('method', '')}, "
            f"observed_frames={observed_frames}"
        )
    return RuleFinding(
        scene_id=task.scene_id,
        frame_index=task.frame_index,
        track_id=f"vision_cluster_{task.frame_index}_{task.cluster_id}",
        bbox_index=task.cluster_id,
        label=verification.possible_category,
        rule_id="PossibleMissingAnnotation",
        severity="Warning",
        message="AI-confirmed possible missing annotation",
        evidence=evidence,
        cluster_id=task.cluster_id,
        cluster_center=json.dumps(center, ensure_ascii=False),
        cluster_size=json.dumps(size, ensure_ascii=False),
        cameras=json.dumps(cameras, ensure_ascii=False),
        rois=json.dumps(rois, ensure_ascii=False),
        possible_category=verification.possible_category,
        first_frame=min(observed_frames) if observed_frames else task.frame_index,
        last_frame=max(observed_frames) if observed_frames else task.frame_index,
        occurrence_count=len(observed_frames) if observed_frames else 1,
        ai_confidence=f"{verification.confidence:.4f}",
        ai_reason=verification.reason,
        context_paths=json.dumps(list(prepared.context_paths), ensure_ascii=False),
        crop_paths=json.dumps(list(prepared.crop_paths), ensure_ascii=False),
    )


def _finalize_wrong_annotation_results(
    observations: list[tuple[PreparedVisionInput, WrongAnnotationVerificationResult]],
    config: dict,
    stats: dict,
    findings: list[RuleFinding],
) -> None:
    """Finalize wrong-annotation results using the configured aggregation mode."""
    aliases = {
        str(source): str(target)
        for source, target in (config.get("label_aliases", {}) or {}).items()
    }
    threshold = float(config.get("confidence_threshold", 0.9))
    aggregation_mode = str(config.get("aggregation_mode", "per_frame")).strip().lower()
    if aggregation_mode == "per_frame":
        _finalize_per_frame_wrong_annotation_results(
            observations, threshold, aliases, stats, findings
        )
        return
    if aggregation_mode != "track_consensus":
        raise ValueError(
            "ai_vision.wrong_annotation.aggregation_mode must be "
            "'per_frame' or 'track_consensus'"
        )

    min_observations = max(1, int(config.get("min_consistent_observations", 2)))
    min_ratio = min(1.0, max(0.0, float(config.get("min_consensus_ratio", 0.67))))
    by_track = defaultdict(list)
    for prepared, verification in observations:
        task = prepared.task
        track_key = str(task.track_id or task.candidate_id)
        by_track[track_key].append((prepared, verification))

    for track_key, track_observations in by_track.items():
        votes = defaultdict(list)
        classifications = {}
        for prepared, verification in track_observations:
            vote, reason = _classify_wrong_annotation_vote(
                prepared, verification, threshold, aliases
            )
            classifications[prepared.manifest_path] = reason
            if reason == "model_response_internally_inconsistent":
                stats["wrong_inconsistent_rejected"] += 1
            if vote is not None:
                votes[vote].append((prepared, verification))

        winning_vote = None
        winning_items = []
        if votes:
            winning_vote, winning_items = max(
                votes.items(), key=lambda item: (len(item[1]), item[0])
            )
        observation_count = len(track_observations)
        vote_count = len(winning_items)
        ratio = vote_count / observation_count if observation_count else 0.0
        accepted = (
            winning_vote is not None
            and vote_count >= min_observations
            and ratio >= min_ratio
        )

        consensus = {
            "accepted": accepted,
            "track_id": track_key,
            "verified_observations": observation_count,
            "matching_error_observations": vote_count,
            "consensus_ratio": ratio,
            "required_observations": min_observations,
            "required_ratio": min_ratio,
            "winning_rule": winning_vote[0] if winning_vote else None,
            "winning_suggestion": winning_vote[1] if winning_vote else None,
        }
        for prepared, _verification in track_observations:
            decision = dict(consensus)
            decision["observation_reason"] = classifications[prepared.manifest_path]
            update_manifest(prepared.manifest_path, "vision_decision", decision)

        if not accepted:
            if votes:
                stats["wrong_consensus_rejected"] += 1
            else:
                stats["wrong_unknown"] += 1
            continue

        representative, verification = max(
            winning_items, key=lambda item: item[1].confidence
        )
        rule_id = winning_vote[0]
        findings.append(_wrong_annotation_finding(representative, verification, rule_id))
        if rule_id == "VisionBBoxMismatch":
            stats["wrong_bbox_mismatches"] += 1
        else:
            stats["wrong_label_mismatches"] += 1
        stats["wrong_report_findings"] += 1
        stats["report_findings"] += 1


def _finalize_per_frame_wrong_annotation_results(
    observations: list[tuple[PreparedVisionInput, WrongAnnotationVerificationResult]],
    threshold: float,
    aliases: dict[str, str],
    stats: dict,
    findings: list[RuleFinding],
) -> None:
    for prepared, verification in observations:
        vote, reason = _classify_wrong_annotation_vote(
            prepared, verification, threshold, aliases
        )
        accepted = vote is not None
        update_manifest(prepared.manifest_path, "vision_decision", {
            "accepted": accepted,
            "aggregation_mode": "per_frame",
            "observation_reason": reason,
            "winning_rule": vote[0] if vote else None,
            "winning_suggestion": vote[1] if vote else None,
        })
        if reason == "model_response_internally_inconsistent":
            stats["wrong_inconsistent_rejected"] += 1
        if not accepted:
            if reason in {
                "confidence_below_report_threshold",
                "model_returned_unknown",
                "label_mismatch_without_valid_suggestion",
            }:
                stats["wrong_unknown"] += 1
            continue

        rule_id = vote[0]
        findings.append(_wrong_annotation_finding(prepared, verification, rule_id))
        if rule_id == "VisionBBoxMismatch":
            stats["wrong_bbox_mismatches"] += 1
        else:
            stats["wrong_label_mismatches"] += 1
        stats["wrong_report_findings"] += 1
        stats["report_findings"] += 1


def _classify_wrong_annotation_vote(
    prepared: PreparedVisionInput,
    verification: WrongAnnotationVerificationResult,
    threshold: float,
    aliases: dict[str, str],
) -> tuple[tuple[str, str] | None, str]:
    if verification.confidence < threshold:
        return None, "confidence_below_report_threshold"
    if verification.label_match is None and verification.bbox_match is None:
        return None, "model_returned_unknown"

    current = normalize_category(prepared.task.current_label or "", aliases)
    suggested = normalize_category(verification.suggested_category, aliases)
    if verification.label_match is False and suggested and suggested == current:
        return None, "model_response_internally_inconsistent"
    if verification.bbox_match is False:
        return ("VisionBBoxMismatch", ""), "bbox_mismatch_vote"
    if verification.label_match is False:
        if not verification.category_in_scope or suggested in {"", "unknown", "out_of_scope"}:
            return None, "label_mismatch_without_valid_suggestion"
        return ("VisionLabelMismatch", suggested), "label_mismatch_vote"
    return None, "annotation_consistent"


def _wrong_annotation_finding(
    prepared: PreparedVisionInput,
    verification: WrongAnnotationVerificationResult,
    rule_id: str,
) -> RuleFinding:
    task = prepared.candidate
    cameras = list(prepared.accepted_cameras)
    rois = [list(roi) for roi in prepared.rois]
    evidence = (
        f"current_label={task.current_label}, position={task.position}, "
        f"rotation={task.rotation}, size={task.size}, distance={task.distance}, "
        f"label_match={verification.label_match}, bbox_match={verification.bbox_match}"
    )
    message = (
        "AI-confirmed BBox does not cover the labelled target"
        if rule_id == "VisionBBoxMismatch"
        else "AI-confirmed annotation label mismatch"
    )
    return RuleFinding(
        scene_id=task.scene_id,
        frame_index=task.frame_index,
        track_id=task.track_id,
        bbox_index=task.bbox_index,
        label=task.current_label,
        rule_id=rule_id,
        severity="Warning",
        message=message,
        evidence=evidence,
        cluster_center=json.dumps(list(task.position) if task.position else None, ensure_ascii=False),
        cluster_size=json.dumps(list(task.size) if task.size else None, ensure_ascii=False),
        cameras=json.dumps(cameras, ensure_ascii=False),
        rois=json.dumps(rois, ensure_ascii=False),
        possible_category=verification.suggested_category,
        ai_confidence=f"{verification.confidence:.4f}",
        ai_reason=verification.reason,
        context_paths=json.dumps(list(prepared.context_paths), ensure_ascii=False),
        crop_paths=json.dumps(list(prepared.crop_paths), ensure_ascii=False),
    )
