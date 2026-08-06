"""Conservatively deduplicate stationary residual Clusters before Vision calls.

Cluster centers are expressed in each LiDAR frame's ego coordinate system.  The
ego vehicle can move, so raw centers must never be compared across frames.  This
module robustly estimates frame-to-reference SE(2) transforms from existing BBox
track correspondences, then collapses only repeatedly observed, world-stationary
Cluster signatures.  Moving or unregistered candidates remain untouched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from ai_input_builder import assess_projection_quality
from vision_candidate import VisionCandidate


@dataclass(frozen=True)
class TemporalDedupResult:
    candidates: tuple[VisionCandidate, ...]
    raw_candidates: int
    duplicate_groups: int
    duplicate_observations: int
    saved_calls: int
    registered_frames: int
    unregistered_frames: tuple[int, ...]


@dataclass(frozen=True)
class _FrameTransform:
    rotation_xy: np.ndarray
    translation_xy: np.ndarray
    translation_z: float
    reference_frame: int

    def apply(self, position: tuple[float, float, float]) -> np.ndarray:
        point = np.asarray(position, dtype=float)
        xy = self.rotation_xy @ point[:2] + self.translation_xy
        return np.array((xy[0], xy[1], point[2] + self.translation_z))


@dataclass
class _Observation:
    candidate: VisionCandidate
    common_position: np.ndarray


def deduplicate_stationary_clusters(
    cluster_tasks: list[VisionCandidate],
    bbox_tasks: list[VisionCandidate],
    config: dict,
    input_config: dict,
) -> TemporalDedupResult:
    """Return one representative for stable repeated Clusters.

    No raw-coordinate fallback is used.  If ego-motion registration is unavailable
    or unreliable, all candidates from that frame pass through unchanged.
    """
    candidates = sorted(
        (task for task in cluster_tasks if task.projections),
        key=lambda task: (task.frame_index, task.cluster_id),
    )
    raw_count = len(candidates)
    if not config.get("enabled", False) or raw_count < 2:
        return _unchanged_result(candidates)

    frames = sorted(
        {task.frame_index for task in candidates}
        | {task.frame_index for task in bbox_tasks}
    )
    transforms = _estimate_frame_transforms(bbox_tasks, frames, config)
    matching = config.get("matching", {}) or {}
    max_step = float(matching.get("max_center_step_m", 0.30))
    max_diameter = float(matching.get("max_center_diameter_m", 0.50))
    max_z_delta = float(matching.get("max_z_delta_m", 0.50))
    max_size_ratio = float(matching.get("max_size_ratio", 3.0))
    max_pca_distance = float(matching.get("max_pca_distance", 0.50))
    min_observations = max(2, int(matching.get("min_observations", 3)))

    tracks: list[list[_Observation]] = []
    unregistered = set()
    by_frame: dict[int, list[VisionCandidate]] = {}
    for task in candidates:
        by_frame.setdefault(task.frame_index, []).append(task)

    for frame_index in frames:
        transform = transforms.get(frame_index)
        frame_tasks = by_frame.get(frame_index, [])
        if transform is None:
            unregistered.add(frame_index)
            tracks.extend([
                [_Observation(task, np.asarray(task.position, dtype=float))]
                for task in frame_tasks
            ])
            continue

        observations = [
            _Observation(task, transform.apply(task.position))
            for task in frame_tasks
            if task.position is not None
        ]
        missing_position = [task for task in frame_tasks if task.position is None]
        proposals = []
        for observation_index, observation in enumerate(observations):
            for track_index, track in enumerate(tracks):
                if track[-1].candidate.frame_index == frame_index:
                    continue
                if track[-1].candidate.frame_index in unregistered:
                    continue
                score = _association_score(
                    observation,
                    track,
                    max_step=max_step,
                    max_diameter=max_diameter,
                    max_z_delta=max_z_delta,
                    max_size_ratio=max_size_ratio,
                    max_pca_distance=max_pca_distance,
                )
                if score is not None:
                    proposals.append((*score, observation_index, track_index))

        used_observations = set()
        used_tracks = set()
        for _distance, _size_ratio, _pca_distance, observation_index, track_index in sorted(proposals):
            if observation_index in used_observations or track_index in used_tracks:
                continue
            tracks[track_index].append(observations[observation_index])
            used_observations.add(observation_index)
            used_tracks.add(track_index)
        tracks.extend([
            [observation]
            for index, observation in enumerate(observations)
            if index not in used_observations
        ])
        tracks.extend([
            [_Observation(task, np.zeros(3, dtype=float))]
            for task in missing_position
        ])

    quality_config = (
        (input_config.get("quality_by_candidate", {}) or {}).get("residual_cluster")
        or input_config.get("quality", {})
        or {}
    )
    output = []
    duplicate_groups = 0
    duplicate_observations = 0
    reference_frame = min(transforms) if transforms else 0
    for track in tracks:
        if len(track) < min_observations:
            output.extend(item.candidate for item in track)
            continue
        duplicate_groups += 1
        duplicate_observations += len(track)
        representative = max(
            (item.candidate for item in track),
            key=lambda task: _representative_score(task, quality_config),
        )
        frames_seen = [item.candidate.frame_index for item in track]
        candidate_ids = [item.candidate.candidate_id for item in track]
        metadata = dict(representative.metadata)
        metadata["_temporal_dedup"] = {
            "method": "bbox_track_ego_motion_compensation",
            "reference_frame": reference_frame,
            "observed_frames": frames_seen,
            "observation_count": len(track),
            "candidate_ids": candidate_ids,
        }
        output.append(replace(representative, metadata=metadata))

    output.sort(key=lambda task: (task.frame_index, task.cluster_id))
    saved_calls = duplicate_observations - duplicate_groups
    return TemporalDedupResult(
        candidates=tuple(output),
        raw_candidates=raw_count,
        duplicate_groups=duplicate_groups,
        duplicate_observations=duplicate_observations,
        saved_calls=saved_calls,
        registered_frames=len(transforms),
        unregistered_frames=tuple(sorted(unregistered)),
    )


def _unchanged_result(candidates: list[VisionCandidate]) -> TemporalDedupResult:
    return TemporalDedupResult(
        candidates=tuple(candidates),
        raw_candidates=len(candidates),
        duplicate_groups=0,
        duplicate_observations=0,
        saved_calls=0,
        registered_frames=0,
        unregistered_frames=(),
    )


def _estimate_frame_transforms(
    bbox_tasks: list[VisionCandidate],
    required_frames: list[int],
    config: dict,
) -> dict[int, _FrameTransform]:
    registration = config.get("registration", {}) or {}
    landmarks: dict[int, dict[str, np.ndarray]] = {}
    for task in bbox_tasks:
        if not task.track_id or task.position is None:
            continue
        landmarks.setdefault(task.frame_index, {})[task.track_id] = np.asarray(
            task.position, dtype=float
        )
    available = [frame for frame in required_frames if landmarks.get(frame)]
    if not available:
        return {}

    reference = available[0]
    transforms = {
        reference: _FrameTransform(
            rotation_xy=np.eye(2),
            translation_xy=np.zeros(2),
            translation_z=0.0,
            reference_frame=reference,
        )
    }
    pending = set(available[1:])
    while pending:
        progress = False
        for frame in sorted(pending):
            anchors = sorted(
                transforms,
                key=lambda anchor: len(set(landmarks[frame]) & set(landmarks[anchor])),
                reverse=True,
            )
            for anchor in anchors:
                pair = _estimate_pair_transform(
                    landmarks[frame], landmarks[anchor], registration
                )
                if pair is None:
                    continue
                rotation, translation, translation_z = pair
                anchor_transform = transforms[anchor]
                transforms[frame] = _FrameTransform(
                    rotation_xy=anchor_transform.rotation_xy @ rotation,
                    translation_xy=(
                        anchor_transform.rotation_xy @ translation
                        + anchor_transform.translation_xy
                    ),
                    translation_z=translation_z + anchor_transform.translation_z,
                    reference_frame=reference,
                )
                pending.remove(frame)
                progress = True
                break
        if not progress:
            break
    return transforms


def _estimate_pair_transform(
    source_by_track: dict[str, np.ndarray],
    target_by_track: dict[str, np.ndarray],
    config: dict,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    common = sorted(set(source_by_track) & set(target_by_track))
    min_common = max(2, int(config.get("min_common_tracks", 6)))
    if len(common) < min_common:
        return None
    source = np.array([source_by_track[key][:2] for key in common])
    target = np.array([target_by_track[key][:2] for key in common])
    source_z = np.array([source_by_track[key][2] for key in common])
    target_z = np.array([target_by_track[key][2] for key in common])
    threshold = float(config.get("inlier_threshold_m", 0.50))
    max_hypotheses = max(1, int(config.get("max_hypotheses", 5000)))

    hypotheses = []
    for first in range(len(common)):
        for second in range(first + 1, len(common)):
            source_delta = source[second] - source[first]
            target_delta = target[second] - target[first]
            if np.linalg.norm(source_delta) < 1.0 or np.linalg.norm(target_delta) < 1.0:
                continue
            hypotheses.append((first, second))
    if len(hypotheses) > max_hypotheses:
        step = len(hypotheses) / max_hypotheses
        hypotheses = [hypotheses[int(index * step)] for index in range(max_hypotheses)]

    best_inliers = None
    best_score = None
    for first, second in hypotheses:
        source_delta = source[second] - source[first]
        target_delta = target[second] - target[first]
        angle = math.atan2(target_delta[1], target_delta[0]) - math.atan2(
            source_delta[1], source_delta[0]
        )
        rotation = np.array((
            (math.cos(angle), -math.sin(angle)),
            (math.sin(angle), math.cos(angle)),
        ))
        translation = target[first] - rotation @ source[first]
        residuals = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
        inliers = residuals <= threshold
        score = (
            int(inliers.sum()),
            -float(np.median(residuals[inliers])) if inliers.any() else -math.inf,
        )
        if best_score is None or score > best_score:
            best_score = score
            best_inliers = inliers
    if best_inliers is None:
        return None

    min_inliers = max(2, int(config.get("min_inliers", 4)))
    min_ratio = float(config.get("min_inlier_ratio", 0.50))
    if best_inliers.sum() < min_inliers or best_inliers.mean() < min_ratio:
        return None
    rotation, translation = _fit_rigid_xy(source[best_inliers], target[best_inliers])
    residuals = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
    inliers = residuals <= threshold
    if inliers.sum() < min_inliers or inliers.mean() < min_ratio:
        return None
    rotation, translation = _fit_rigid_xy(source[inliers], target[inliers])
    translation_z = float(np.median(target_z[inliers] - source_z[inliers]))
    return rotation, translation, translation_z


def _fit_rigid_xy(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    u, _singular, vt = np.linalg.svd(
        (source - source_center).T @ (target - target_center)
    )
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def _association_score(
    observation: _Observation,
    track: list[_Observation],
    *,
    max_step: float,
    max_diameter: float,
    max_z_delta: float,
    max_size_ratio: float,
    max_pca_distance: float,
) -> tuple[float, float, float] | None:
    current = observation.candidate
    previous = track[-1].candidate
    if current.size is None or previous.size is None:
        return None
    xy_distance = float(np.linalg.norm(
        observation.common_position[:2] - track[-1].common_position[:2]
    ))
    diameter = max(
        float(np.linalg.norm(observation.common_position[:2] - item.common_position[:2]))
        for item in track
    )
    z_delta = max(
        abs(float(observation.common_position[2] - item.common_position[2]))
        for item in track
    )
    size_ratio = _size_ratio(current.size, previous.size)
    pca_distance = _pca_distance(current, previous)
    if (
        xy_distance > max_step
        or diameter > max_diameter
        or z_delta > max_z_delta
        or size_ratio > max_size_ratio
        or pca_distance > max_pca_distance
    ):
        return None
    return xy_distance, size_ratio, pca_distance


def _size_ratio(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> float:
    left_values = np.maximum(np.sort(np.asarray(left, dtype=float)), 1e-6)
    right_values = np.maximum(np.sort(np.asarray(right, dtype=float)), 1e-6)
    return float(np.max(np.maximum(
        left_values / right_values, right_values / left_values
    )))


def _pca_distance(left: VisionCandidate, right: VisionCandidate) -> float:
    keys = ("linearity", "planarity", "flatness")
    left_values = np.array([left.pca_features.get(key, 0.0) for key in keys])
    right_values = np.array([right.pca_features.get(key, 0.0) for key in keys])
    return float(np.linalg.norm(left_values - right_values))


def _representative_score(task: VisionCandidate, quality_config: dict) -> tuple:
    checks = [assess_projection_quality(view, quality_config) for view in task.projections]
    good_cameras = {check.camera for check in checks if check.quality == "good"}
    good_views = [view for view in task.projections if view.camera in good_cameras]
    roi_areas = [
        max(0, view.roi[2] - view.roi[0]) * max(0, view.roi[3] - view.roi[1])
        for view in good_views
    ]
    return (
        len(good_views),
        sum(view.visible_points for view in good_views),
        max(roi_areas, default=0),
        task.point_count,
        task.cluster_confidence,
        -task.frame_index,
    )
