"""
cleanup_scene.py
按 scene_id 清理 outputs/ 目录下这个场景产生的本地文件（截图/报告/BBox 数据/网络记录等），
以及 assets/ 目录下这个场景下载的 PCD/JPG 文件（见 assets_downloader.py）。

用于人工判断完一个 scene 之后，清掉这个 scene 的本地测试数据，给下一个 scene 腾地方。
只读扫描 + 删除文件：
  - 在 outputs/reports、outputs/screenshots 这两个目录里
    按「文件名/内容匹配」找（目录路径优先读 config.yaml，读不到才退回默认值，见 _load_scan_dirs()）
  - 另外在顶层 assets/（assets_downloader.py 的下载目录）里按「scene_<scene_id> 目录名」
    匹配（见 find_asset_scene_files()）——这里的文件名本身不带 scene_id
    （pointcloud.pcd / camera_xxx.jpg），scene_id 只体现在父目录名里，
    所以不能用上面那套按文件名/内容的匹配逻辑，需要单独处理。
  - outputs/ 下只删匹配文件，不删 reports/screenshots 根目录；根级 assets/ 下会删除
    assets/scene_<scene_id>/ 目录树，避免清理后留下空的 frame_NNNN/images 目录
  - 不会碰任何项目代码/配置文件（.py/.yaml/.md 等）——因为压根不扫描这几个目录以外的地方
  - 不连浏览器、不改采集/规则逻辑，是一个完全独立的小工具

匹配规则（outputs/ 下：文件名 或 文件内容包含 scene_id 就算相关）：
  - 图片文件（.png/.jpg/.jpeg）：只按文件名匹配（截图命名是 {scene_id}_frame_NNN.png，
    内容是二进制，扫描内容既没意义也慢）
  - 其它文本文件（.csv/.json/.txt 等）：文件名包含 scene_id，或者文件内容里包含
    scene_id 都算匹配——现在大部分报告文件（capture_report.csv、bbox_data.csv、
    rule_report.csv、rule_summary.json、network_assets.csv、bbox_probe_report.json）
    文件名是固定的、不带 scene_id，只能靠内容里的 scene_id 列/字段判断是不是这个场景的数据。
  - 兜底（SCENE_AGNOSTIC_COMPANION_FILES）：极少数报告文件内容里完全不带 scene_id
    （比如 mapping_report.txt，是"这次抓取整体质量"的统计，不是某个场景自己的数据），
    文件名/内容都匹配不上。这种文件不改它的生成逻辑，而是在这里认：如果它所在的目录
    已经因为别的文件命中了这个 scene_id，说明这一批报告确实是这次场景生成的（同一次
    main.py 运行会把这些报告文件一起写出来），就把它也一并纳入清理范围。

匹配规则（assets/ 下：目录名匹配，见 find_asset_scene_files()）：
  - assets/scene_<scene_id>/ 目录存在，就把它下面递归找到的所有文件（pointcloud.pcd、
    images/*.jpg、metadata.json）都当作这个场景的数据一并清理，不需要逐个文件按名字/内容判断。

用法：
    python cleanup_scene.py              # 交互式：列出将删除的文件，输入 DELETE 二次确认才真删
    python cleanup_scene.py --dry-run     # 只列出会删除哪些文件，不会真的删除，不需要输入 DELETE

scene_id 留空直接回车 -> 什么都不做，直接退出。
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

# 图片按文件名匹配即可，不读二进制内容（既没意义又浪费时间）
BINARY_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# 内容匹配时单个文件的读取大小上限（字节），避免不小心扫到异常巨大的文件卡住
MAX_CONTENT_SCAN_BYTES = 20 * 1024 * 1024  # 20MB

# 找不到 config.yaml 或读取失败时的默认目录（跟 config.yaml 里的默认值保持一致）
DEFAULT_SCAN_DIRS = ["outputs/reports", "outputs/screenshots"]

# assets_downloader.py 的下载目录默认值，跟 config.yaml -> assets.output_dir 保持一致。
# 这是顶层的 assets/，即 assets_downloader.py 实际保存 PCD/JPG 的目录。
DEFAULT_ASSETS_DOWNLOAD_DIR = "assets"
DEFAULT_AI_INPUTS_DIR = "ai_inputs"

# 内容里不带 scene_id、没法直接按文件名/内容匹配的报告文件——如果它所在目录已经有
# 别的文件因为这个 scene_id 匹配上了，就一并纳入清理（见模块开头「匹配规则」的说明）。
SCENE_AGNOSTIC_COMPANION_FILES = {
    "bbox_data.csv",
    "bbox_probe_report.json",
    "capture_report.csv",
    "mapping_report.txt",
    "network_assets.csv",
    "rule_report.csv",
    "rule_summary.json",
}


def _load_scan_dirs(config_path: str = "config.yaml") -> list[str]:
    """优先读 config.yaml 里配置的实际输出目录，读不到就用默认值——不写死路径，
    避免以后有人在 config.yaml 里改了 output_dir，这个工具却删错地方（或者删不到）。"""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return DEFAULT_SCAN_DIRS

    dirs = []
    report_dir = config.get("report", {}).get("output_dir")
    screenshot_dir = config.get("screenshot", {}).get("output_dir")
    for d in (report_dir, screenshot_dir):
        if d:
            dirs.append(d)

    return dirs or DEFAULT_SCAN_DIRS


def _load_assets_download_dir(config_path: str = "config.yaml") -> str:
    """读 config.yaml -> assets.output_dir（assets_downloader.py 的下载目录），
    读不到就用默认值 "assets"。"""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return DEFAULT_ASSETS_DOWNLOAD_DIR

    return config.get("assets", {}).get("output_dir") or DEFAULT_ASSETS_DOWNLOAD_DIR


def _load_ai_inputs_dir(config_path: str = "config.yaml") -> str:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return DEFAULT_AI_INPUTS_DIR
    return config.get("ai_inputs", {}).get("output_dir") or DEFAULT_AI_INPUTS_DIR


def find_asset_scene_files(
    scene_id: str,
    assets_download_dir: str,
    label: str = "Assets",
) -> list[tuple[Path, str]]:
    """在 assets/ 下按 scene_<scene_id> 目录名匹配，而不是按文件名/内容——
    assets_downloader.py 产出的 pointcloud.pcd / camera_xxx.jpg 文件名本身不带 scene_id，
    scene_id 只体现在父目录名（scene_<scene_id>）里。目录存在就把它下面递归找到的所有文件
    都当作这个场景的数据，一并纳入清理范围。"""
    matches: list[tuple[Path, str]] = []
    scene_dir = Path(assets_download_dir) / f"scene_{scene_id}"
    if not scene_dir.is_dir():
        return matches

    for path in sorted(scene_dir.rglob("*")):
        if path.is_file() and path.name != ".gitkeep":
            matches.append((path, f"{label} 场景目录匹配"))

    return matches




def find_asset_scene_dirs(scene_id: str, assets_download_dir: str) -> list[Path]:
    """返回 assets/scene_<scene_id>/ 下需要清理的目录列表，包含 scene 目录本身。

    这个函数用于 dry-run 展示，也覆盖“文件已经删掉但空目录还残留”的场景。
    """
    scene_dir = Path(assets_download_dir) / f"scene_{scene_id}"
    if not scene_dir.is_dir():
        return []
    dirs = [p for p in scene_dir.rglob("*") if p.is_dir()]
    dirs.sort(key=lambda p: len(p.parts), reverse=True)
    dirs.append(scene_dir)
    return dirs

def find_matching_files(scene_id: str, scan_dirs: list[str]) -> list[tuple[Path, str]]:
    """在 scan_dirs 里找文件名或内容包含 scene_id 的文件，返回 [(路径, 匹配原因), ...]。
    另外把 SCENE_AGNOSTIC_COMPANION_FILES 里那种"内容不带 scene_id 但跟其它匹配文件
    同批次生成"的文件也纳入（前提是它所在目录确实有其它文件匹配上了这个 scene_id，
    不会凭空把一个没有任何关联证据的目录里的文件也算进来）。"""
    matches: list[tuple[Path, str]] = []
    matched_parent_dirs: set[Path] = set()

    for scan_dir in scan_dirs:
        base = Path(scan_dir)
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            if path.name == ".gitkeep":
                continue  # 目录占位文件，不是数据，不参与匹配

            if scene_id in path.name:
                matches.append((path, "文件名匹配"))
                matched_parent_dirs.add(path.parent)
                continue

            if path.suffix.lower() in BINARY_EXTENSIONS:
                continue  # 图片不扫内容

            try:
                if path.stat().st_size > MAX_CONTENT_SCAN_BYTES:
                    continue
                content = path.read_text(encoding="utf-8-sig", errors="ignore")
            except (OSError, UnicodeDecodeError):
                continue

            if scene_id in content:
                matches.append((path, "内容匹配"))
                matched_parent_dirs.add(path.parent)

    matched_paths = {p for p, _ in matches}
    for parent_dir in matched_parent_dirs:
        for name in SCENE_AGNOSTIC_COMPANION_FILES:
            companion = parent_dir / name
            if companion.is_file() and companion not in matched_paths:
                matches.append((companion, "同批次报告文件（内容不含 scene_id，随其它匹配文件一起清理）"))
                matched_paths.add(companion)

    return matches


def delete_files(matches: list[tuple[Path, str]]) -> int:
    """实际删除文件，返回成功删除的数量。"""
    deleted = 0
    for path, _reason in matches:
        try:
            path.unlink()
            deleted += 1
        except OSError as exc:
            print(f"[警告] 删除失败: {path} ({exc})")
    return deleted


def remove_empty_asset_scene_dirs(scene_id: str, assets_download_dir: str, warn: bool = True) -> int:
    """删除 assets/scene_<scene_id>/ 下已经清空的目录，最后删除 scene 目录本身。

    只处理这个精确匹配的 scene 目录，不碰 assets/ 根目录和其它 scene。目录里如果还
    有未删除文件（例如权限问题导致 unlink 失败），对应目录会保留并打印 warning。
    """
    scene_dir = Path(assets_download_dir) / f"scene_{scene_id}"
    if not scene_dir.is_dir():
        return 0

    removed = 0
    for path in sorted((p for p in scene_dir.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        try:
            path.rmdir()
            removed += 1
        except OSError:
            pass

    try:
        scene_dir.rmdir()
        removed += 1
    except OSError as exc:
        if warn:
            print(f"[警告] Assets 场景目录未能删除（可能仍有文件）: {scene_dir} ({exc})")

    return removed


def main() -> None:
    dry_run = "--dry-run" in sys.argv

    scene_id = input("请输入 scene_id: ").strip()
    if not scene_id:
        print("[信息] scene_id 为空，退出，没有做任何操作。")
        return

    scan_dirs = _load_scan_dirs()
    matches = find_matching_files(scene_id, scan_dirs)

    assets_download_dir = _load_assets_download_dir()
    matches += find_asset_scene_files(scene_id, assets_download_dir)
    asset_dirs = find_asset_scene_dirs(scene_id, assets_download_dir)

    ai_inputs_dir = _load_ai_inputs_dir()
    matches += find_asset_scene_files(scene_id, ai_inputs_dir, label="AI Inputs")
    ai_input_dirs = find_asset_scene_dirs(scene_id, ai_inputs_dir)

    print()
    if not matches and not asset_dirs and not ai_input_dirs:
        print(f"[信息] 没有找到跟 scene_id={scene_id!r} 相关的文件或 Assets 目录。")
        return

    total_scene_dirs = len(asset_dirs) + len(ai_input_dirs)
    print(f"将删除（scene_id={scene_id!r}，共 {len(matches)} 个文件，{total_scene_dirs} 个场景目录）：")
    for path, reason in matches:
        print(f"  {path.as_posix()}（{reason}）")
    for path in asset_dirs:
        print(f"  {path.as_posix()}/（空目录清理）")
    for path in ai_input_dirs:
        print(f"  {path.as_posix()}/（AI Inputs 空目录清理）")

    if dry_run:
        print()
        print(f"[信息] dry-run 模式，以上 {len(matches)} 个文件和 {total_scene_dirs} 个目录不会被真正删除。")
        return

    print()
    confirm = input("确认删除请输入 DELETE: ").strip()
    if confirm != "DELETE":
        print("[信息] 输入不是 DELETE，已取消，没有删除任何文件。")
        return

    deleted = delete_files(matches)
    removed_dirs = remove_empty_asset_scene_dirs(scene_id, assets_download_dir)
    removed_dirs += remove_empty_asset_scene_dirs(scene_id, ai_inputs_dir)
    print()
    print(f"[信息] 已删除 {deleted} / {len(matches)} 个文件，清理空 Assets 目录 {removed_dirs} 个。")


if __name__ == "__main__":
    main()
