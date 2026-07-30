# UNSOLVED

## 1. 多 Tab 场景下自动识别 QP 质检页面

当前状态：`browser.py` 在 `browser.page_title_hint` 为空时默认使用当前 CDP context 的第一个 tab。用户目前使用时只保留 QP 质检页面，所以主流程可正常运行。

待改目标：即使浏览器同时打开多个页面，也应自动找到真正的 QP 质检页面，而不是依赖第一个 tab。

建议方向：增强 `BrowserSession._pick_target_page()` 的候选评分逻辑，按 URL/title、Scene ID、页面是否存在 `window.viewer`、是否能识别点云查看器或 `BoxVolume` 结构等信号选择页面。该改动只影响 tab 选择，不重做浏览器连接、切帧、BBox 提取或网络监听逻辑。

## 2. MissingBBox 与 EmptyFrame 报告语义应保持互斥

当前语义：
- `MissingBBox` 表示这一帧 BBox 提取失败，例如 `window.viewer` 不存在、页面未准备好或 JS evaluate 失败。
- `EmptyFrame` 表示这一帧 BBox 提取成功，但结果是 0 个框。

待确认/待改点：当前 `EmptyFrame` 规则只根据 `bbox_data.csv` 中某帧是否没有行来判断；而 BBox 提取失败帧在 `bbox_data.csv` 中同样没有行。虽然 `main.py` 另外记录了 `missing_bbox_frames` 并追加 `MissingBBox` finding，但后续应确保同一帧不会同时输出 `MissingBBox` 和 `EmptyFrame`。

期望行为：
- BBox 提取失败帧：只报 `MissingBBox`。
- BBox 提取成功但 0 个框：只报 `EmptyFrame`。

建议方向：Rule Engine 运行 `EmptyFrame` 时应接收或识别 `missing_bbox_frames`，并从 EmptyFrame 判断集合中排除这些帧；不要改变 BBox 提取和切帧逻辑。
