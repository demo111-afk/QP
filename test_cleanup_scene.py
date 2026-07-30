"""
test_cleanup_scene.py
Independent tests for cleanup_scene.py. No pytest required.

Usage:
    python test_cleanup_scene.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from cleanup_scene import (
    _load_scan_dirs,
    find_asset_scene_dirs,
    find_asset_scene_files,
    find_matching_files,
    remove_empty_asset_scene_dirs,
)


def test_find_asset_scene_files_matches_nested_asset_files():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "assets"
        scene_dir = base / "scene_123" / "frame_0001" / "images"
        scene_dir.mkdir(parents=True)
        pcd = base / "scene_123" / "frame_0001" / "pointcloud.pcd"
        jpg = scene_dir / "cam_front_mid.jpg"
        pcd.write_text("pcd", encoding="utf-8")
        jpg.write_text("jpg", encoding="utf-8")
        (base / ".gitkeep").write_text("", encoding="utf-8")

        matches = find_asset_scene_files("123", str(base))
        matched_paths = {path for path, _reason in matches}
        assert pcd in matched_paths
        assert jpg in matched_paths
        assert base / ".gitkeep" not in matched_paths



def test_find_asset_scene_dirs_reports_empty_leftover_dirs():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "assets"
        image_dir = base / "scene_123" / "frame_0001" / "images"
        image_dir.mkdir(parents=True)

        dirs = find_asset_scene_dirs("123", str(base))

        assert image_dir in dirs
        assert image_dir.parent in dirs
        assert base / "scene_123" in dirs


def test_remove_empty_asset_scene_dirs_removes_scene_tree_after_files_deleted():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "assets"
        image_dir = base / "scene_123" / "frame_0001" / "images"
        image_dir.mkdir(parents=True)
        file_path = image_dir / "cam_front_mid.jpg"
        file_path.write_text("jpg", encoding="utf-8")

        file_path.unlink()
        removed = remove_empty_asset_scene_dirs("123", str(base))

        assert removed == 3  # images, frame_0001, scene_123
        assert not (base / "scene_123").exists()
        assert base.exists()


def test_remove_empty_asset_scene_dirs_keeps_non_empty_scene_dir():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "assets"
        image_dir = base / "scene_123" / "frame_0001" / "images"
        image_dir.mkdir(parents=True)
        file_path = image_dir / "cam_front_mid.jpg"
        file_path.write_text("jpg", encoding="utf-8")

        removed = remove_empty_asset_scene_dirs("123", str(base), warn=False)

        assert removed == 0
        assert (base / "scene_123").is_dir()
        assert file_path.is_file()


def test_load_scan_dirs_ignores_obsolete_network_assets_dir():
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "config.yaml"
        config_path.write_text(
            """
report:
  output_dir: custom/reports
screenshot:
  output_dir: custom/screenshots
network:
  assets_dir: outputs/assets
""".strip(),
            encoding="utf-8",
        )

        dirs = _load_scan_dirs(str(config_path))

        assert dirs == ["custom/reports", "custom/screenshots"]
        assert "outputs/assets" not in dirs


def test_find_matching_files_only_scans_configured_output_dirs():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        reports = base / "reports"
        screenshots = base / "screenshots"
        old_assets = base / "outputs" / "assets"
        reports.mkdir()
        screenshots.mkdir()
        old_assets.mkdir(parents=True)
        report = reports / "rule_report.csv"
        report.write_text("scene_id\n123\n", encoding="utf-8")
        stale = old_assets / "123_stale.txt"
        stale.write_text("123", encoding="utf-8")

        matches = find_matching_files("123", [str(reports), str(screenshots)])
        matched_paths = {path for path, _reason in matches}

        assert report in matched_paths
        assert stale not in matched_paths


def main() -> None:
    tests = [
        test_find_asset_scene_files_matches_nested_asset_files,
        test_find_asset_scene_dirs_reports_empty_leftover_dirs,
        test_remove_empty_asset_scene_dirs_removes_scene_tree_after_files_deleted,
        test_remove_empty_asset_scene_dirs_keeps_non_empty_scene_dir,
        test_load_scan_dirs_ignores_obsolete_network_assets_dir,
        test_find_matching_files_only_scans_configured_output_dirs,
    ]
    for test in tests:
        test()
        print(f"[通过] {test.__name__}")
    print(f"\n全部 {len(tests)} 个测试通过。")


if __name__ == "__main__":
    main()
