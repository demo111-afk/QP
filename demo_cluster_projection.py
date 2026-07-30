"""
demo_cluster_projection.py
Standalone demo for projecting residual clusters onto available asset cameras.

This script is intentionally outside the main QP Copilot workflow. It does not
modify Rule Engine, Browser Automation, or Cluster Detector behavior.

Real-cluster demo:
    .venv/bin/python demo_cluster_projection.py --scene-id 25255 --frame-index 41

Synthetic projection demo for mismatched calibration/images:
    .venv/bin/python demo_cluster_projection.py --scene-id 25255 --frame-index 41 --synthetic-demo
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml

from calibration import CameraCalibration, load_calibration
from cluster_detector import ClusterCandidate, OrientedBox, detect_clusters
from cluster_projection import (
    build_asset_camera_matches,
    project_clusters_to_asset_cameras,
    save_projection_debug_images,
)


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_frame_boxes(bbox_csv_path: str | Path, scene_id: str, frame_index: int) -> list[OrientedBox]:
    path = Path(bbox_csv_path)
    if not path.is_file():
        return []

    boxes = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("scene_id", "")) != str(scene_id):
                continue
            if int(row.get("frame_index") or -1) != frame_index:
                continue
            box = _row_to_oriented_box(row)
            if box is not None:
                boxes.append(box)
    return boxes


def _row_to_oriented_box(row: dict) -> OrientedBox | None:
    try:
        center = (
            float(row["position_x"]),
            float(row["position_y"]),
            float(row["position_z"]),
        )
        rotation = (
            float(row["rotation_x"]),
            float(row["rotation_y"]),
            float(row["rotation_z"]),
        )
        size = (
            float(row["scale_x"]),
            float(row["scale_y"]),
            float(row["scale_z"]),
        )
    except (KeyError, TypeError, ValueError):
        return None
    return OrientedBox(center=center, rotation_euler=rotation, size=size)


def choose_demo_frame(scene_dir: Path) -> int:
    frame_dirs = sorted(p for p in scene_dir.glob("frame_*") if p.is_dir() and (p / "pointcloud.pcd").is_file())
    if not frame_dirs:
        raise FileNotFoundError(f"No frame_NNNN/pointcloud.pcd found under {scene_dir}")
    return int(frame_dirs[0].name.split("_", 1)[1])


def make_synthetic_cluster_for_camera(camera: CameraCalibration, size_m: float = 1.0, depth_m: float = 10.0) -> ClusterCandidate:
    """Create a small demo cluster on the camera optical axis.

    This is only for validating projection/debug-image plumbing when the local
    calibration and asset images are known to come from different scenes.
    """
    if camera.extrinsic is None:
        raise ValueError(f"Camera {camera.camera_id} has no extrinsic")

    half = size_m / 2.0
    camera_points = np.array([
        [-half, -half, depth_m - half],
        [-half, -half, depth_m + half],
        [-half, half, depth_m - half],
        [-half, half, depth_m + half],
        [half, -half, depth_m - half],
        [half, -half, depth_m + half],
        [half, half, depth_m - half],
        [half, half, depth_m + half],
    ], dtype=np.float64)
    transform = np.asarray(camera.extrinsic.values, dtype=np.float64)
    inverse = np.linalg.inv(transform)
    homogeneous = np.column_stack([camera_points, np.ones(len(camera_points), dtype=np.float64)])
    world_points = (inverse @ homogeneous.T).T[:, :3]
    mins = world_points.min(axis=0)
    maxs = world_points.max(axis=0)
    centroid = world_points.mean(axis=0)

    return ClusterCandidate(
        cluster_id=f"synthetic_{camera.camera_id}",
        point_indices=list(range(len(world_points))),
        centroid=tuple(float(v) for v in centroid),
        point_count=len(world_points),
        bbox_hint={
            "min": tuple(float(v) for v in mins),
            "max": tuple(float(v) for v in maxs),
            "size": tuple(float(v) for v in (maxs - mins)),
        },
        status="Synthetic Projection Demo",
    )


def run_demo(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    calibration = load_calibration(args.calibration)

    assets_dir = Path(args.assets_dir)
    scene_dir = assets_dir / f"scene_{args.scene_id}"
    frame_index = args.frame_index if args.frame_index is not None else choose_demo_frame(scene_dir)
    frame_dir = scene_dir / f"frame_{frame_index:04d}"
    images_dir = frame_dir / "images"
    pcd_path = frame_dir / "pointcloud.pcd"

    if not args.synthetic_demo and not pcd_path.is_file():
        raise FileNotFoundError(pcd_path)
    if not images_dir.is_dir():
        raise FileNotFoundError(images_dir)

    matches = build_asset_camera_matches(calibration, images_dir)
    print(f"[Projection Demo] scene={args.scene_id}, frame={frame_index}")
    print(f"[Projection Demo] available images: {len(list(images_dir.glob('*.jpg')))}")
    print(f"[Projection Demo] matched calibration cameras: {len(matches)}")
    for match in matches:
        print(f"  - {match.asset_name}.jpg -> {match.camera.camera_id} ({match.camera.location})")

    if args.synthetic_demo:
        clusters = [make_synthetic_cluster_for_camera(match.camera) for match in matches if match.camera.extrinsic is not None]
        print(f"[Projection Demo] synthetic clusters: {len(clusters)}")
    else:
        report_cfg = config.get("report", {}) or {}
        bbox_extract_cfg = config.get("bbox_extract", {}) or {}
        bbox_csv_path = args.bbox_csv or Path(report_cfg.get("output_dir", "outputs/reports")) / bbox_extract_cfg.get(
            "output_filename", "bbox_data.csv"
        )
        boxes = load_frame_boxes(bbox_csv_path, args.scene_id, frame_index)
        print(f"[Projection Demo] loaded bbox boxes for cluster removal: {len(boxes)}")

        cluster_cfg = ((config.get("rule_engine") or {}).get("cluster_detector") or {})
        clusters = detect_clusters(str(pcd_path), boxes, cluster_cfg)
        if args.max_clusters > 0:
            clusters = clusters[:args.max_clusters]
        print(f"[Projection Demo] residual clusters: {len(clusters)}")

    for i, cluster in enumerate(clusters):
        print(
            f"  - cluster[{i}] id={cluster.cluster_id}, points={cluster.point_count}, "
            f"center={cluster.centroid}, size={cluster.bbox_hint['size'] if cluster.bbox_hint else None}"
        )

    projections = project_clusters_to_asset_cameras(clusters, calibration, images_dir)
    visible_count = sum(len(rois) for rois in projections.values())
    print(f"[Projection Demo] visible projected ROIs: {visible_count}")
    for asset_name, rois in projections.items():
        if rois:
            print(f"  - {asset_name}: {len(rois)} ROI(s)")
            for roi in rois:
                print(f"      {roi.cluster_id} -> {roi.roi}, camera={roi.camera_id}, clipped={roi.clipped}")

    debug_dir = frame_dir / ("projection_debug_synthetic" if args.synthetic_demo else "projection_debug")
    saved = save_projection_debug_images(projections, debug_dir)
    print(f"[Projection Demo] saved debug images: {len(saved)}")
    for path in saved:
        print(f"  - {path}")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project residual clusters onto asset camera images.")
    parser.add_argument("--scene-id", default="25255")
    parser.add_argument("--frame-index", type=int, default=None)
    parser.add_argument("--assets-dir", default="assets")
    parser.add_argument("--calibration", default="cali.wps")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--bbox-csv", default=None)
    parser.add_argument("--max-clusters", type=int, default=20)
    parser.add_argument("--synthetic-demo", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_demo(parse_args()))
