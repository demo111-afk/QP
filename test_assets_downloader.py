"""
test_assets_downloader.py
sample_interval 选帧逻辑的独立测试（不需要 pytest，跟项目现有"无外部测试框架"的风格一致）。

用法：
    python test_assets_downloader.py
"""

from __future__ import annotations

from assets_downloader import should_download_frame


def _selected_frames(frame_count: int, start_frame_index: int, sample_interval: int) -> list[int]:
    return [
        frame_index
        for frame_index in range(start_frame_index, start_frame_index + frame_count)
        if should_download_frame(frame_index, start_frame_index, sample_interval)
    ]


def test_81_frames_interval_10():
    frames = _selected_frames(frame_count=81, start_frame_index=1, sample_interval=10)
    assert frames == [1, 11, 21, 31, 41, 51, 61, 71, 81], frames


def test_80_frames_interval_10():
    frames = _selected_frames(frame_count=80, start_frame_index=1, sample_interval=10)
    assert frames == [1, 11, 21, 31, 41, 51, 61, 71], frames


def test_5_frames_interval_10():
    frames = _selected_frames(frame_count=5, start_frame_index=1, sample_interval=10)
    assert frames == [1], frames


def test_1_frame_interval_10():
    frames = _selected_frames(frame_count=1, start_frame_index=1, sample_interval=10)
    assert frames == [1], frames


def test_interval_1_downloads_every_frame():
    frames = _selected_frames(frame_count=5, start_frame_index=1, sample_interval=1)
    assert frames == [1, 2, 3, 4, 5], frames


def test_start_frame_index_0():
    # start_frame_index=0 的平台（帧号栏从 0 开始显示）也要对齐到起始帧，而不是绝对帧号 0。
    frames = _selected_frames(frame_count=81, start_frame_index=0, sample_interval=10)
    assert frames == [0, 10, 20, 30, 40, 50, 60, 70, 80], frames


def main() -> None:
    tests = [
        test_81_frames_interval_10,
        test_80_frames_interval_10,
        test_5_frames_interval_10,
        test_1_frame_interval_10,
        test_interval_1_downloads_every_frame,
        test_start_frame_index_0,
    ]
    for test in tests:
        test()
        print(f"[通过] {test.__name__}")
    print(f"\n全部 {len(tests)} 个测试通过。")


if __name__ == "__main__":
    main()
