"""Shared Qwen-VL adapter for one Vision Candidate per request."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

from vision_category_config import (
    VisionCategory,
    format_categories_for_prompt,
    load_vision_categories,
)


OUT_OF_SCOPE_CATEGORY = "out_of_scope"
UNKNOWN_CATEGORY = "unknown"


def _vision_prompt(categories: tuple[VisionCategory, ...]) -> str:
    category_text = format_categories_for_prompt(categories)
    return f"""你是一名自动驾驶数据质检助手。

请综合分析同一个 Residual Cluster 在多个相机中的图片和点云几何信息。
每张 context 图片中只标出了当前 Cluster，crop 图片是该区域的无标记局部图。
context 图的红框内部才是待判断目标；crop 图只为看清该红框中心区域，边缘包含少量上下文。
红框外或 crop 边缘出现的车辆、设备、雪糕筒等都不能作为当前 Cluster 的证据。

本项目只关注下列白名单类别：
{category_text}

严格判断规则：
1. 必须先确认 ROI 对应一个真实、独立且完整程度足以归类的目标。
2. possible_category 只能返回上面某个完整类别名称、out_of_scope 或 unknown。
3. 船舶、集装箱、建筑、护栏、路面、植被及其他表外目标必须返回 out_of_scope。
4. 岸桥、场桥、车辆或其他大型目标的局部边缘、车顶附件、轮子、支架、吊臂、
   控制箱等附属部件不是独立目标，必须返回 out_of_scope。
5. 禁止为了命中白名单而把船舶归为轿车/其他车辆，或把普通固定机械部件归为岸桥/场桥。
6. 只有确认属于白名单中的独立目标时，missing_annotation_suspected 才能为 true。
7. 表外目标返回 out_of_scope；图片不足以判断时返回 unknown；这两种情况的
   missing_annotation_suspected 必须为 false。
8. 点云尺寸与所选类别的常见物理尺寸明显冲突时返回 unknown 或 out_of_scope；不得用
   “点云重建误差”解释数倍的尺寸差异。
9. ROI 同时覆盖多个目标、成排设施或大型结构的一部分时，不是单个独立目标，返回 out_of_scope。
10. 如果红框内没有清楚目标，即使红框附近存在白名单物体，也必须返回 unknown，
    contains_real_object 和 missing_annotation_suspected 均返回 false。

只返回 JSON：
{{
  "cluster_id": "...",
  "contains_real_object": true,
  "missing_annotation_suspected": true,
  "possible_category": "轿车",
  "confidence": 0.87,
  "reason": "..."
}}

不要输出 JSON 之外的内容。"""


def _wrong_annotation_prompt(
    categories: tuple[VisionCategory, ...],
    label_aliases: dict[str, str],
) -> str:
    category_text = format_categories_for_prompt(categories)
    alias_text = "\n".join(
        f"- {source} = {target}" for source, target in sorted(label_aliases.items())
    ) or "- 无"
    return f"""你是一名自动驾驶数据质检助手。

请判断一个已有三维 BBox 的当前标签、框的位置与多相机图片内容是否一致。
每张 context 图片只画了当前 BBox 的投影 ROI，crop 图片是该区域的无标记局部图。
context 图红框内部才是当前 BBox；crop 边缘的邻近目标不能作为标签或框位置的证据。

项目关注的标准类别如下：
{category_text}

平台标签别名（判断 label_match 时必须视为同一类别）：
{alias_text}

严格判断规则：
1. bbox_match 判断投影框是否覆盖当前标签所描述的真实目标；框住背景、建筑、空区域、
   明显偏离目标或只覆盖无关局部时返回 false。
2. label_match 只在视觉证据足够时判断；明显错误返回 false，正确返回 true，无法判断返回 null。
3. bbox_match 无法判断时返回 null。遮挡、距离过远或图片证据不足时不要强行判断。
4. suggested_category 只能返回上面的完整标准类别、out_of_scope 或 unknown。
5. 不要求识别精确车型；没有明确证据时必须返回 unknown。
6. 多相机内容有冲突时采用更保守结论，并在 reason 中说明。
7. suggested_category 与当前标签或其别名相同时，label_match 必须为 true，禁止输出自相矛盾结果。
8. 目标太小、过远、被遮挡或只看到局部时，label_match/bbox_match 返回 null，置信度不得高于 0.5。

只返回 JSON：
{{
  "candidate_id": "...",
  "label_match": true,
  "bbox_match": true,
  "suggested_category": "轿车",
  "confidence": 0.93,
  "reason": "..."
}}

不要输出 JSON 之外的内容。"""


@dataclass(frozen=True)
class VisionVerificationResult:
    cluster_id: str
    status: str
    contains_real_object: bool = False
    missing_annotation_suspected: bool = False
    possible_category: str = UNKNOWN_CATEGORY
    category_in_scope: bool = False
    confidence: float = 0.0
    reason: str = ""
    error: str = ""
    attempts: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class WrongAnnotationVerificationResult:
    candidate_id: str
    status: str
    label_match: bool | None = None
    bbox_match: bool | None = None
    suggested_category: str = UNKNOWN_CATEGORY
    category_in_scope: bool = False
    confidence: float = 0.0
    reason: str = ""
    error: str = ""
    attempts: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


class QwenVLVerifier:
    def __init__(
        self,
        config: dict,
        request_fn: Callable[[list[dict]], str] | None = None,
    ) -> None:
        self.config = config
        self.request_fn = request_fn
        self._client = None
        self._client_lock = threading.Lock()
        category_path = self.config.get("category_whitelist_path")
        self.categories = load_vision_categories(category_path) if category_path else load_vision_categories()
        self.allowed_categories = {category.name for category in self.categories}
        wrong_config = self.config.get("wrong_annotation", {}) or {}
        self.label_aliases = {
            str(source): str(target)
            for source, target in (wrong_config.get("label_aliases", {}) or {}).items()
        }

    def verify_cluster(
        self,
        cluster_metadata: dict,
        image_paths: Iterable[str | Path],
    ) -> VisionVerificationResult:
        cluster_id = str(cluster_metadata.get("cluster_id", ""))
        paths = [Path(path) for path in image_paths]
        if not paths:
            return VisionVerificationResult(
                cluster_id=cluster_id,
                status="skipped",
                error="no_image_paths",
            )

        if self.request_fn is None:
            api_key_env = str(self.config.get("api_key_env", "DASHSCOPE_API_KEY"))
            if not os.getenv(api_key_env):
                return VisionVerificationResult(
                    cluster_id=cluster_id,
                    status="skipped",
                    error=f"missing_api_key:{api_key_env}",
                )
            if not str(self.config.get("api_base", "")).strip():
                return VisionVerificationResult(
                    cluster_id=cluster_id,
                    status="skipped",
                    error="missing_api_base",
                )

        try:
            content = _build_content(cluster_metadata, paths, self.categories)
        except OSError as exc:
            return VisionVerificationResult(
                cluster_id=cluster_id,
                status="error",
                error=f"image_read_failed:{exc}",
            )

        parsed, attempts, error = self._request_with_retries(
            content,
            lambda raw: _parse_response(raw, cluster_id, self.allowed_categories),
        )
        if parsed is None:
            return VisionVerificationResult(
                cluster_id=cluster_id, status="error", error=error, attempts=attempts
            )
        return VisionVerificationResult(
            cluster_id=cluster_id,
            status="verified",
            contains_real_object=parsed["contains_real_object"],
            missing_annotation_suspected=parsed["missing_annotation_suspected"],
            possible_category=parsed["possible_category"],
            category_in_scope=parsed["category_in_scope"],
            confidence=parsed["confidence"],
            reason=parsed["reason"],
            attempts=attempts,
        )

    def verify_bbox(
        self,
        bbox_metadata: dict,
        image_paths: Iterable[str | Path],
    ) -> WrongAnnotationVerificationResult:
        candidate_id = str(bbox_metadata.get("candidate_id", ""))
        paths = [Path(path) for path in image_paths]
        unavailable = self._availability_error(paths)
        if unavailable:
            return WrongAnnotationVerificationResult(
                candidate_id=candidate_id,
                status="skipped",
                error=unavailable,
            )

        try:
            content = _build_prompt_content(
                _wrong_annotation_prompt(self.categories, self.label_aliases),
                "BBox metadata",
                bbox_metadata,
                paths,
            )
        except OSError as exc:
            return WrongAnnotationVerificationResult(
                candidate_id=candidate_id,
                status="error",
                error=f"image_read_failed:{exc}",
            )

        parsed, attempts, error = self._request_with_retries(
            content,
            lambda raw: _parse_wrong_annotation_response(
                raw, candidate_id, self.allowed_categories
            ),
        )
        if parsed is None:
            return WrongAnnotationVerificationResult(
                candidate_id=candidate_id, status="error", error=error, attempts=attempts
            )
        return WrongAnnotationVerificationResult(
            candidate_id=candidate_id,
            status="verified",
            label_match=parsed["label_match"],
            bbox_match=parsed["bbox_match"],
            suggested_category=parsed["suggested_category"],
            category_in_scope=parsed["category_in_scope"],
            confidence=parsed["confidence"],
            reason=parsed["reason"],
            attempts=attempts,
        )

    def _request_with_retries(
        self,
        content: list[dict],
        parser: Callable[[str], dict],
    ) -> tuple[dict | None, int, str]:
        max_retries = max(0, int(self.config.get("max_retries", 2)))
        last_error = ""
        for attempt in range(1, max_retries + 2):
            try:
                return parser(self._request(content)), attempt, ""
            except Exception as exc:
                last_error = str(exc)
                if attempt <= max_retries:
                    delay = max(0.0, float(self.config.get("retry_delay_seconds", 1.0)))
                    if delay:
                        time.sleep(delay)
        return None, max_retries + 1, last_error or "verification_failed"

    def _availability_error(self, paths: list[Path]) -> str:
        if not paths:
            return "no_image_paths"
        if self.request_fn is not None:
            return ""
        api_key_env = str(self.config.get("api_key_env", "DASHSCOPE_API_KEY"))
        if not os.getenv(api_key_env):
            return f"missing_api_key:{api_key_env}"
        if not str(self.config.get("api_base", "")).strip():
            return "missing_api_base"
        return ""

    def _request(self, content: list[dict]) -> str:
        if self.request_fn is not None:
            return self.request_fn(content)

        client = self._get_client()
        response = client.chat.completions.create(
            model=str(self.config.get("model_name", "qwen3-vl-plus")),
            messages=[{"role": "user", "content": content}],
            temperature=0,
        )
        result = response.choices[0].message.content
        if not isinstance(result, str) or not result.strip():
            raise ValueError("model returned empty content")
        return result

    def _get_client(self):
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is not None:
                return self._client
            try:
                from openai import OpenAI
                import httpx
            except ImportError as exc:
                raise RuntimeError("openai/httpx package is not installed") from exc

            api_key_env = str(self.config.get("api_key_env", "DASHSCOPE_API_KEY"))
            timeout = float(self.config.get("timeout", 60))
            http_client = httpx.Client(
                timeout=timeout,
                trust_env=bool(self.config.get("use_environment_proxy", False)),
            )
            self._client = OpenAI(
                api_key=os.environ[api_key_env],
                base_url=str(self.config["api_base"]).rstrip("/"),
                timeout=timeout,
                max_retries=0,
                http_client=http_client,
            )
            return self._client

    def close(self) -> None:
        with self._client_lock:
            client = self._client
            self._client = None
        if client is not None:
            client.close()


def verify_cluster(
    cluster_metadata: dict,
    image_paths: Iterable[str | Path],
    config: dict,
) -> VisionVerificationResult:
    verifier = QwenVLVerifier(config)
    try:
        return verifier.verify_cluster(cluster_metadata, image_paths)
    finally:
        verifier.close()


def verify_bbox(
    bbox_metadata: dict,
    image_paths: Iterable[str | Path],
    config: dict,
) -> WrongAnnotationVerificationResult:
    verifier = QwenVLVerifier(config)
    try:
        return verifier.verify_bbox(bbox_metadata, image_paths)
    finally:
        verifier.close()


def _build_content(
    cluster_metadata: dict,
    image_paths: list[Path],
    categories: tuple[VisionCategory, ...],
) -> list[dict]:
    return _build_prompt_content(
        _vision_prompt(categories), "Cluster metadata", cluster_metadata, image_paths
    )


def _build_prompt_content(
    prompt: str,
    metadata_label: str,
    metadata: dict,
    image_paths: list[Path],
) -> list[dict]:
    metadata_text = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
    content: list[dict] = [
        {"type": "text", "text": f"{prompt}\n\n{metadata_label}:\n{metadata_text}"}
    ]
    for path in image_paths:
        mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        content.append({"type": "text", "text": f"Image: {path.name}"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
        })
    return content


def _parse_wrong_annotation_response(
    raw: str,
    expected_candidate_id: str,
    allowed_categories: set[str],
) -> dict:
    value = _load_json_object(raw)
    for field in ("label_match", "bbox_match"):
        if value.get(field) is not None and type(value.get(field)) is not bool:
            raise ValueError(f"{field}_must_be_boolean_or_null")

    category = value.get("suggested_category")
    valid_categories = allowed_categories | {OUT_OF_SCOPE_CATEGORY, UNKNOWN_CATEGORY}
    if category not in valid_categories:
        raise ValueError("invalid_suggested_category")
    confidence, reason = _parse_confidence_and_reason(value)
    returned_id = str(value.get("candidate_id", ""))
    if returned_id and returned_id != expected_candidate_id:
        raise ValueError("candidate_id_mismatch")
    return {
        "candidate_id": expected_candidate_id,
        "label_match": value.get("label_match"),
        "bbox_match": value.get("bbox_match"),
        "suggested_category": category,
        "category_in_scope": category in allowed_categories,
        "confidence": confidence,
        "reason": reason,
    }


def _parse_response(
    raw: str,
    expected_cluster_id: str,
    allowed_categories: set[str],
) -> dict:
    value = _load_json_object(raw)

    for field in ("contains_real_object", "missing_annotation_suspected"):
        if type(value.get(field)) is not bool:
            raise ValueError(f"{field}_must_be_boolean")

    category = value.get("possible_category")
    valid_categories = allowed_categories | {OUT_OF_SCOPE_CATEGORY, UNKNOWN_CATEGORY}
    if category not in valid_categories:
        raise ValueError("invalid_possible_category")
    confidence, reason = _parse_confidence_and_reason(value)

    returned_id = str(value.get("cluster_id", ""))
    if returned_id and returned_id != expected_cluster_id:
        raise ValueError("cluster_id_mismatch")
    category_in_scope = category in allowed_categories
    return {
        "cluster_id": expected_cluster_id,
        "contains_real_object": value["contains_real_object"],
        "missing_annotation_suspected": (
            value["missing_annotation_suspected"] if category_in_scope else False
        ),
        "possible_category": category,
        "category_in_scope": category_in_scope,
        "confidence": confidence,
        "reason": reason,
    }


def _load_json_object(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid_json:{exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError("response_json_must_be_object")
    return value


def _parse_confidence_and_reason(value: dict) -> tuple[float, str]:
    confidence = value.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("confidence_must_be_number")
    confidence = float(confidence)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence_out_of_range")
    reason = value.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason_must_be_non_empty_string")
    return confidence, reason.strip()
