# QP Copilot V1.0 Integration 修改记录

更新时间：2026-08-03

## 1. 本次目标

将已完成的 Browser Automation、BBox、Assets、Rule Engine、Residual Cluster、Projection、
Qwen Vision 和 Report 串成一次 UI Start 即可完成的 V1.0 Pipeline。此次不修改各业务模块的
算法和数学逻辑，只调整入口编排、环境加载、错误隔离、日志和文档。

## 2. 修改前的实际状态

修改前已经能够从 UI 启动 `main.py` 并最终写入统一 Rule Report，但存在以下集成问题：

1. Qwen Key 只从进程环境读取，每次启动 UI 前需要手动 `export DASHSCOPE_API_KEY`。
2. Projection 阶段已经执行 Residual Cluster，之后 Rule Engine 的
   `PossibleMissingAnnotation` 又执行一次 Cluster；第二次结果在 AI 开启时被丢弃。
3. Cluster/Projection 位于逐帧浏览器循环内，而普通规则在全部采集完成后运行，阶段边界不清晰。
4. 单 Cluster API 失败可以继续，但 Vision Pipeline 初始化失败会阻止最终报告生成。
5. 浏览器连接失败直接退出，无法生成解释失败原因的最终报告。
6. UI 只显示帧进度，不显示阶段日志和子进程关键输出。
7. `vision_test_limit` 仍为 20，只属于小规模验证配置。
8. README 仍写着“不接 AI”，与实际代码不一致。

## 3. V1.0 最终编排

`main.py` 当前按九个可机器解析的阶段输出日志：

```text
[1/9] Loading Configuration
[2/9] Connecting Edge Browser
[3/9] Collecting Frames, BBox and Assets
[4/9] Writing Capture and BBox Data
[5/9] Running Rule Engine
[6/9] Running Cluster Detection and Projection
[7/9] Running Vision Verification
[8/9] Merging Rules and Writing Final Report
[9/9] Inspection Finished
```

完整顺序：

```text
UI 输入并校验 Scene / Frame Count / Calibration
→ 自动加载 .env
→ 连接已登录 Edge
→ 全帧切换、BBox 收集、采样下载 PCD/JPG
→ 释放 Playwright 会话但不关闭用户 Edge
→ 写 bbox_data.csv 等内部采集数据
→ 运行普通 BBox Rule
→ 对成功下载且 BBox 读取成功的采样帧运行 Cluster + Projection
→ 生成单 Cluster AI 输入
→ 逐 Cluster 调用 Qwen
→ 合并普通 Rule + AI PossibleMissingAnnotation + MissingBBox
→ 写 rule_report.csv
→ 写 rule_summary.json
→ Inspection Finished
```

## 4. `.env` 和 API Key

新增：

- `runtime_env.py`
- `.env.example`
- `python-dotenv>=1.0`

项目根目录 `.env` 格式：

```dotenv
DASHSCOPE_API_KEY=实际Key
```

规则：

- 系统环境变量优先，`.env` 不覆盖已有变量。
- `.env` 已在 `.gitignore` 中，不进入 Git。
- UI 每次点击 Start 时重新检查 Key，并在启动子进程时动态复制环境。
- UI 发现 Vision 已启用但 Key 缺失时给出明确错误，不启动不完整的 V1.0 Run。
- 直接运行 `main.py` 时，Key 缺失不会崩溃；Vision 跳过，普通规则和最终报告继续。
- 日志只显示变量名和状态，永不输出 Key 内容。

## 5. 避免重复 Cluster

没有修改 `rule_engine.py` 或 `cluster_detector.py`。

`main.py` 新增 `build_rule_run_config()`：当 Vision 开启时，复制当前 config，并只在本次副本的
`enabled_rules` 中排除 `PossibleMissingAnnotation`。原始 `config.yaml` 不变，Projection 仍调用
原有 `run_frame_projection()` 完成唯一一次 Cluster Detection。Vision 关闭时，Rule Engine 的
原几何漏标规则仍可照常独立运行。

## 6. 错误隔离

- 单帧切帧、就绪等待、截图、网络记录、BBox 和 Assets 异常：记录警告，继续下一帧。
- 单帧 Cluster/Projection 异常：跳过该采样帧，继续下一帧。
- 单 Cluster Qwen/API/JSON 异常：由 Vision 模块记录，继续下一个 Cluster。
- Vision Pipeline 整体异常：记录错误，继续合并普通规则并写最终报告。
- Edge 连接失败：所有预期帧写为 `MissingBBox`，仍生成两个最终报告。
- Rule Engine 整体异常：保留 MissingBBox 和可用 AI 结果，继续写最终报告。
- `rule_report.csv` 或 `rule_summary.json` 自身无法写入属于不可恢复的最终输出错误，UI 显示失败。

这里继续严格区分：页面成功读取且 BBox 数量为 0 是 EmptyFrame；页面/BBox 数据源没有成功读取
才是 MissingBBox。Integration 会让 BrokenTrack 使用完整帧范围以抑制帧级失败造成的假断轨，同时
单独只在 BBox 成功读取的帧集合上运行现有 EmptyFrame 规则，因此失败帧不会同时出现两个矛盾结论。

## 7. UI Integration

`ui_app.py` 和 `ui_app.pyw` 保持同步，增加：

- `.env`/API Key 预检。
- `[x/9]` 阶段状态解析。
- 实时 Run Log 文本区域。
- 子进程环境在点击 Start 时动态构建。
- 正式结果区只展示 `rule_report.csv` 和 `rule_summary.json`。

`bbox_data.csv`、capture/network/probe 文件仍作为内部输入和故障诊断文件保留，不建立第二套报告。

## 8. Config 变化

`config.yaml`：

```yaml
ai_vision:
  api_key_env: "DASHSCOPE_API_KEY"
  vision_test_limit: 0
```

`vision_test_limit: 0` 表示正式 V1.0 审核所有可投影候选。Projection 质量不合格、表格白名单外、
unknown 或 API 失败的候选仍不会进入最终漏标报告。完整 Run 的 API 调用量会高于之前的 20 个测试模式。

## 9. 未修改的业务模块

本次没有修改以下模块的内部逻辑：

- `browser.py` / `navigator.py`
- `bbox_extractor.py` / `bbox_probe.py`
- `assets_downloader.py`
- `cluster_detector.py`
- `cluster_projection.py` / `projection_pipeline.py`
- `rule_engine.py` 的规则实现
- `vision_verifier.py` 的模型判断逻辑

## 10. 测试结果

全量 13 个测试脚本、77 项测试全部通过：

- Assets：6
- Calibration：8
- Cleanup：6
- Cluster Detector：28
- Cluster Projection：7
- Projection Pipeline：3
- Rule Cluster 去重：3
- AI Input：2
- Vision Verifier：4
- Vision Pipeline：1
- Runtime Environment：3
- Main Integration：5
- Calibration Persistence：1

新增 Integration 测试已经验证：模拟 Edge 连接失败且 `.env` 缺失时，程序仍生成
`rule_report.csv` 和 `rule_summary.json`，两帧均以 `MissingBBox` 记录，并正常输出 `[9/9]`。

## 11. 已知前置条件和风险

- Edge 仍需提前以 Remote Debugging 模式启动并完成登录。
- 用户仍需在 UI 填写 Scene Number、Frame Count 和 Calibration YAML。
- 多标签页时当前 Browser 依赖 `page_title_hint`；为空时使用第一个页面。该问题仍按 `UNSOLVED.md`
  记录，本次遵守要求未修改 Browser Automation。
- 本次没有在真实 QP 页面上重新跑完整 81 帧和全部付费 Qwen 请求；代码级全量回归和失败降级测试已完成。
- 当前项目根目录尚未由程序写入真实 `.env` Key；真实 Key 需要用户一次性放入 `.env`。


## 12. V1.0 交付清理

根据用户要求，V1.0 集成与 77 项回归测试完成后执行交付清理：

- 删除项目内原始 `cali.wps`。正式 UI 使用用户粘贴内容生成临时 YAML，并按 Scene 保存
  `assets/scene_<scene_id>/calibration_used.yaml`，运行时不依赖 `cali.wps`。
- 删除项目根目录全部 13 个 `test_*.py` 测试脚本。
- 保留本节和第 10 节的测试结果作为删除前的验证记录。
- `calibration.py` 的独立命令行入口和 Projection Demo 不再引用已删除的默认 WPS，改为要求
  显式提供 Calibration 路径。
- 桌面的 `ucas_joint_v1.wps` 未确认属于本项目，未删除。
- 桌面的《车辆及设施尺寸参考表.xlsx》是类别白名单和尺寸库来源资料，未删除。


## 13. Cleanup 完整覆盖与 DimensionMismatch 降噪

更新时间：2026-08-03

### Cleanup

审计后确认原 Cleanup 已能删除：

- `outputs/screenshots` 中目标 Scene 截图；
- `assets/scene_<scene_id>/` 整棵目录，包括 PCD、原始 JPG、Projection Debug、metadata 和
  `calibration_used.yaml`；
- `ai_inputs/scene_<scene_id>/` 整棵目录，包括 context、crop 和 manifest；
- 报告目录中内容或文件名包含 Scene ID 的文件。

修复了只有表头的固定报告残留问题。`SCENE_AGNOSTIC_COMPANION_FILES` 现在覆盖
`bbox_data.csv`、`bbox_probe_report.json`、`capture_report.csv`、`mapping_report.txt`、
`network_assets.csv`、`rule_report.csv` 和 `rule_summary.json`。只要报告目录中的任一正式文件
确认属于目标 Scene，这批固定报告都会一起清理。

使用临时完整 Scene 树验证：报告、截图、Assets、Projection Debug、Calibration 归档和 AI Inputs
文件全部删除，两个 `scene_<scene_id>` 目录均无残留。

### DimensionMismatch

根据用户要求，不删除规则、不改变报告字段，只统一放宽尺寸数据库：

```yaml
tolerance: {length: 100.0, width: 100.0, height: 100.0}
```

34 个有 Reference Size 的类别全部使用该值；两个 `no_fixed_value` 类别继续保持 `null`。
在当前 `SizeOutlier` 的合法尺寸域 `0.1–30m` 内，逐类别验证最小值和最大值均
`within_tolerance=true`，因此 DimensionMismatch 不再产生误报告警。SizeOutlier 仍独立负责
明显小于 0.1m 或大于 30m 的异常尺寸。

### `__pycache__`

`__pycache__` 是 Python 自动生成的字节码缓存目录，里面的 `.pyc` 用于加快模块启动。它不包含
Scene、BBox、Assets、Calibration、API Key 或报告数据，可以随时删除，但下次运行 Python 时会
自动重新生成。Scene Cleanup 不删除它，因为它属于程序运行缓存而不是某个 Scene 的数据。


## 14. Qwen Vision 判定材料

更新时间：2026-08-03

Vision Pipeline 每次只让 Qwen 判断一个 Residual Cluster，不会把整帧所有 Cluster 一次性交给模型。

### 14.1 图片输入

一个 Cluster 在每个通过 Projection 质量门禁的 Camera 下最多生成并发送两张图片：

- `cam_<name>_context.jpg`：保留较大场景范围，只画当前 Cluster 的框，不显示其他 Cluster。
- `cam_<name>_crop.jpg`：从原始 JPG 裁剪当前 ROI，四周各外扩约 50%，不画框、不加文字，
  并裁剪到图像边界内。

如果同一个 Cluster 在多个 Camera 中可见，则这些 Camera 的 context/crop 会在同一次请求中一起
发送。原始 JPG 不修改、不覆盖。文件组织示例：

```text
ai_inputs/
  scene_29729/
    frame_0081/
      cluster_166_m8/
        manifest.json
        cam_rear_right_context.jpg
        cam_rear_right_crop.jpg
```

### 14.2 点云与几何元数据

模型请求同时包含当前 Cluster 的：

- `scene_id`、`frame_index`、`cluster_id`；
- `point_count`；
- 三维中心 `center=[x, y, z]`；
- 三维尺寸 `size=[length, width, height]`；
- 与自车的距离 `distance`；
- PCA 特征：`linearity`、`planarity`、`flatness`、`thickness_ratio`；
- `cluster_confidence`；
- `merged_fragment_count`；
- `clustering_method`。

每个 Camera 的名称、ROI、visible points、context/crop 路径和 Projection 检查结果记录在同目录
`manifest.json` 中。请求使用这些结构化元数据和实际图片；不会把 manifest 文件本身当图片发送。

### 14.3 Projection 质量门禁

以下候选标记为 `projection_low_quality`，不调用 Qwen：

- ROI 面积过小；
- ROI 面积占整图比例异常大；
- visible points 太少；
- ROI 严重贴边或被边界截断；
- 无法稳定投影到任何 Camera。

阈值统一来自 `config.yaml -> ai_inputs.quality`。

### 14.4 不发送给 Qwen 的内容

- 整帧多 Cluster Projection Debug JPG；
- 其他 Cluster 的框或一次性全部 Cluster；
- 原始 PCD 文件；
- Calibration YAML；
- 浏览器会话、BBox CSV 或 API Key。

### 14.5 类别与报告准入

Qwen 的 `possible_category` 受 `config/vision_categories.yaml` 白名单约束，来源是
《车辆及设施尺寸参考表.xlsx》的31个唯一类别。表外物体必须返回 `out_of_scope`，无法判断返回
`unknown`。只有同时满足以下条件才转换成现有 `PossibleMissingAnnotation` RuleFinding：

1. Projection 质量为 good；
2. 类别位于白名单；
3. `contains_real_object=true`；
4. `missing_annotation_suspected=true`；
5. 置信度不低于配置阈值。

表外目标、unknown、背景/噪声、低质量 Projection 和 API 失败只保留调试记录，不进入最终
`rule_report.csv`。

## 15. Wrong Annotation Vision Verification

更新时间：2026-08-03

### 15.1 目标与采样范围

新增 Existing BBox 错标检查，但没有建立第二套 Vision 或 Report。漏标 Cluster 和错标 BBox
都只处理 `assets.sample_interval` 命中的资产帧；当前间隔为 10，因此检查 1、11、21……等帧。
全帧 BBox 收集和原有 Rule Engine 不受该间隔影响。

统一流程：

```text
Residual Cluster -> VisionCandidate -> Projection -> context/crop -> Qwen -> RuleFinding
Existing BBox    -> VisionCandidate -> Projection -> context/crop -> Qwen -> RuleFinding
```

新增 `vision_candidate.py`，统一保存 Candidate 类型、ID、Scene、Frame、三维位置/尺寸/旋转、
Track ID、当前 Label、业务元数据和多 Camera Projection。Cluster 原有构造和公共入口保留为兼容层。

### 15.2 BBox Projection

Existing BBox 使用页面读取的 Position、Rotation、Scale。BBox 按与 Cluster 删除框内点相同的
Three.js XYZ Euler 顺序构造旋转矩阵，对长方体 12 条边逐边采样，再调用现有
`cluster_projection.project_points()`。没有复制或改变 Calibration、外参方向、内参和像素缩放数学。

每个 Camera 仍输出统一 ROI、可见采样点数、边界裁剪状态和目标图片尺寸。BBox Projection
阈值来自 `projection.wrong_annotation`。PCD 下载失败时仅跳过漏标 Cluster；只要 JPG 和 BBox
可用，同一采样帧的错标检查仍继续。

### 15.3 AI 输入和模型接口

`ai_input_builder.py` 现在接受通用 Candidate。每个 BBox 在独立目录生成只画当前框的 context、
无标记 crop 和 manifest；原始 JPG 不修改。无法投到 Camera 的 BBox 也生成
`projection_low_quality` manifest，不调用模型，不会静默丢失。

`vision_verifier.py` 继续使用同一个 Qwen 客户端、图片 Base64 编码、API Key、超时、重试和 JSON
清理逻辑。漏标与错标只分别提供 Prompt 和响应解析器。错标响应字段为：

```json
{
  "candidate_id": "bbox_0_track_x",
  "label_match": true,
  "bbox_match": true,
  "suggested_category": "岸桥",
  "confidence": 0.98,
  "reason": "..."
}
```

证据不足时 `label_match` / `bbox_match` 可以为 `null`。类别仍受表格白名单约束。配置中记录
`托架 -> 卡车拖架`、`其他车 -> 其他车辆`、`飞机拖车A/B -> 飞机拖车甲型/乙型`，Prompt 明确
要求将这些平台别名视为同一类别，避免名称差异误报。

### 15.4 Report 准入

- 高置信度且 `bbox_match=false`：写入 `VisionBBoxMismatch`。
- 高置信度、`bbox_match` 非 false 且 `label_match=false`：写入 `VisionLabelMismatch`。
- Unknown、低置信度、Projection 低质量、API/JSON 失败：不写主报告，保留 manifest 和统计。

结果继续写入现有 `rule_report.csv` 和 `rule_summary.json`。Current Label 使用原字段 `label`，
AI 建议使用 `possible_category`，并复用 `ai_confidence`、`ai_reason`、`cameras`、`rois`、
`context_paths`、`crop_paths`，没有增加第二套报告格式。

### 15.5 验证记录

- 修改模块全部通过 `py_compile`，`config.yaml` 可正常解析。
- 合成 BBox 的 12 条边产生 108 个采样点，经原投影函数得到稳定 ROI `(88, 88, 112, 112)`。
- 模拟双任务验证同时生成 `PossibleMissingAnnotation` 和 `VisionLabelMismatch`。
- Scene 29729 Frame 81 真实数据：18 个 BBox 均有完整几何，17 个可见，产生 21 个多相机 ROI；
  17 个 good Candidate 成功生成 21 组 context/crop，输出只写入 `/tmp` 验证目录。
- 完整 `run_frame_projection()` 真实回归同时得到 59 个 Cluster、54 个 Cluster ROI、18 个 BBox、
  21 个 BBox ROI 和 77 个统一 VisionCandidate；Projection Debug 仅写入 `/tmp`。
- 使用其中一个真实岸桥 BBox 完成一次 Qwen 调用：`label_match=true`、`bbox_match=true`、
  `suggested_category=岸桥`、`confidence=0.98`，JSON 协议解析成功且未产生错标 Finding。

本次没有修改 Browser Automation、BBox 收集、Cluster Detection、Calibration 数学、Assets Downloader、
原有 Rule Engine 规则或已有 Report 字段。

## 16. Scene 25204 Cluster / Vision 误报审计与修正

更新时间：2026-08-03

### 16.1 用户反馈与审计结论

本次完整运行结束后，用户反馈 Cluster 投影框混乱、Qwen 对明显不是目标的区域仍给出高置信度类别，
并且本应在不确定时返回 Unknown。对 Scene 25204 的 Report、751 个 AI manifest、Cluster 几何和
代表性 context/crop 图片进行了逐层审计。

旧 `rule_report.csv` 共 356 条，其中：

- `PossibleMissingAnnotation` 300 条；
- `VisionLabelMismatch` 35 条；
- `VisionBBoxMismatch` 8 条；
- 其他原有规则 13 条。

8 个采样帧共产生 491 个 Residual Cluster。119 个候选最长轴超过 5m，21 个超过 10m；242 个由
多个 HDBSCAN 叶簇合并而来，其中 59 个至少合并 5 个碎片。视觉抽查确认 Projection 红框与
Cluster AABB 一致，未发现本次问题由 Calibration 外参方向反转导致。主要问题是源 Cluster 本身
覆盖护栏、门架、成排设施或大型结构局部，Qwen 又把框附近而非框内的物体当成当前候选。

典型错误包括：

- Frame 51 `cluster_94` 尺寸约 `6.02 x 0.62 x 1.82m`，图中混有标志、雪糕筒和护栏，Qwen
  判为雪糕筒并用“点云重建放大”解释明显尺寸矛盾；
- Frame 41 `cluster_386_m8` 合并 8 个碎片、覆盖成排设施，Qwen 将多个设施当成一个雪糕筒目标；
- Existing BBox Track 17 当前标签和建议标签均为“灯塔”，模型却返回 `label_match=false`；
- 同一稳定 Track 在相邻采样帧得到不同建议类别，旧逻辑仍将单帧异常直接写入报告。

因此，本次运行产生的旧 Vision 报告不能作为可靠质检结论；旧文件未被离线脚本篡改，必须重跑后
才能得到应用新准入规则的正式 Report。

### 16.2 根因

1. Cluster 是高召回 Candidate Generator，不是语义识别器。固定高度过滤不能完整移除地面/背景，
   HDBSCAN 叶簇和二级空间合并仍可能形成结构碎片或成排物体的 AABB。
2. 原有 Cluster 最终几何过滤过宽，对极端水平细长结构和大面积低矮合并组没有专门门禁。
3. `visible_points` 仅表示投影后落在图像范围内的点，不代表经过遮挡判断后的真实可见点。
4. Cluster crop 外扩 50% 容易混入邻近车辆和设施，模型会错误借用框外视觉证据。
5. 漏标置信度阈值原先只参与统计，模型自报高置信度即可绕过尺寸常识进入主报告。
6. 错标只看单帧响应，没有校验“当前类别等于建议类别但 label_match=false”的内部矛盾，也没有
   利用 Track 的多帧稳定性。

### 16.3 Cluster 源头修正

在二级合并完成、重新计算真实点集 AABB 后增加两条类别无关的通用几何门禁：

- `max_horizontal_aspect_ratio: 20.0`：排除护栏、边线等极端水平细长结构；
- 当高度不超过 `low_height_threshold: 1.0m` 时，水平占地面积不得超过
  `max_low_height_horizontal_area: 12.0m²`，排除成排设施或地面结构被合成的大面积低矮候选。

这两条规则不针对灯塔、车辆或某个特定类别，也不改变第一级 HDBSCAN、二级 Merge、BBox 删除、
Projection 或 Calibration 数学。使用同一批 8 帧 PCD/BBox 回放得到 480 个候选，较旧运行的
491 个减少 11 个。由于 HDBSCAN 和残余点云仍以高召回为目标，候选数量下降有限；最终报告还必须
经过后续 Projection、尺寸和 Vision 保守准入，不能把每个 Cluster 直接解释为真实漏标。

### 16.4 Projection 与 AI 输入修正

Projection 质量门禁新增 ROI 最小宽度和高度检查：Cluster 至少 `48 x 32px`，Existing BBox
至少 `64 x 48px`；Existing BBox 同时要求更大的最小面积和至少 12 个投影采样点。

Cluster crop 外扩比例由默认 50% 单独收紧为 20%，Existing BBox 为 25%。context 仍保留完整原图，
只画当前 Candidate。Prompt 明确规定：

- 只有 context 红框内部才是待判断目标；
- crop 边缘和红框外的邻近目标不得作为当前 Candidate 证据；
- 红框覆盖多个物体、成排设施或大型结构局部时返回 `out_of_scope`；
- 物理尺寸冲突时不得用“点云重建误差”强行解释；
- 看不清、过远、遮挡或多相机冲突时返回 Unknown/null，并降低置信度。

### 16.5 漏标主报告确定性门禁

新增 `vision_quality_gate.py`。Qwen 返回表内类别后，程序根据
`config/vision_categories.yaml -> dimension_keys` 查找 `config/vehicle_dimensions.yaml` 的
Reference Size。由于 Residual Cluster 使用轴对齐 AABB，X/Y 先排序再与参考长宽比较。

进入 `PossibleMissingAnnotation` 主报告现在必须同时满足：

1. Qwen 返回表内类别、真实目标且疑似漏标；
2. Qwen 置信度至少 `0.95`；
3. 三维任一轴不得小于对应参考尺寸的 20%，避免把物体的一条边或一个附件当成完整目标；
4. 三维任一轴不得大于对应参考尺寸的 2.5 倍，避免雪糕筒对应 6m Cluster 等物理矛盾；
5. 表格中没有固定参考尺寸的类别不伪造尺寸，记录 `no_fixed_reference` 后继续依赖视觉判断。

对旧 300 条 AI 漏标结果做只读离线回放：289 条因置信度低于 0.95 排除，9 条因类别/尺寸矛盾
排除，剩余 2 条尺寸合理的雪糕筒候选。该数字是对旧 Qwen 响应应用新门禁的离线估算，不是新一轮
正式运行结果。

### 16.6 错标自洽校验与 Track 多帧一致性

Existing BBox 不再以单帧 Qwen 响应直接写报告。新增确定性处理：

- 当前标签先通过别名归一化，新增 `平台车1 -> 平台车`、`皮带车 -> 行李传送带车`、
  `车辆 -> 其他车辆`；
- 若归一化后的 Current Label 与 Suggested Category 相同，但模型返回 `label_match=false`，直接标记
  `model_response_internally_inconsistent`，不得写报告；
- `VisionLabelMismatch` 的投票必须同时具有明确、表内且不同于当前标签的建议类别；
- 同一 Track 至少 2 个已验证采样帧给出同一种错误，且占该 Track 已验证观察的至少 67%；
- Label Mismatch 还必须对建议类别一致；BBox Mismatch 按“框明显偏离/无目标”一致投票；
- 一个 Track 最终最多写一条错标 Finding，代表帧取获胜票中置信度最高的一次；原始逐帧响应和
  Track 共识结果继续保存在各自 manifest 中。

旧数据离线回放覆盖 42 个有成功响应的 Track：发现 2 次内部自相矛盾响应；经过多帧一致性后仅
Track 44 的两次“建议货车”达到报告门槛，其余 41 个 Track 均被 Unknown/无有效票或一致性不足
排除。该结果同样是只读离线估算。

### 16.7 验证

- `vision_pipeline.py`、`vision_verifier.py`、`vision_quality_gate.py`、
  `vision_category_config.py`、`ai_input_builder.py`、`cluster_detector.py` 通过 `py_compile`；
- `config.yaml` 和类别/尺寸配置可正常解析；
- 尺寸门禁断言覆盖正常雪糕筒、6m 雪糕筒矛盾、0.1m 宽货车碎片和无固定参考尺寸类别；
- 临时错标回归覆盖：两帧一致错误保留、同类别自相矛盾排除、单帧错误不进入报告；
- `git diff --check` 通过；
- 未调用 Vision API，未修改旧 Report，未修改 Browser Automation、BBox 收集、Calibration、
  Projection 数学、Assets Downloader、Rule Engine 原有规则和已有 Report 字段。

## 17. 错标恢复单帧判定与数量口径说明

更新时间：2026-08-04

### 17.1 单帧错标模式

根据用户确认，Wrong Annotation 从 Track 多帧共识改为采样帧单帧独立判定：

- `ai_vision.wrong_annotation.aggregation_mode: per_frame`；
- 每个命中 `assets.sample_interval` 的帧独立产生或拒绝错标 Finding；
- 当前 `sample_interval=10`，所以仍检查 1、11、21……，不是改为检查全部 81 帧；
- 同一 Track 在不同采样帧可以分别产生 Finding，不再要求至少两帧一致；
- 保留 `confidence_threshold=0.90`、标签别名归一化、同类别自相矛盾响应排除，以及 Unknown/null
  不写报告的门禁；
- `track_consensus` 模式和对应参数仍作为配置选项保留，但当前不启用。

对 Scene 25204 的旧 Existing BBox 响应进行只读离线估算：单帧模式会接受 16 条，其中
`VisionLabelMismatch` 15 条、`VisionBBoxMismatch` 1 条；104 条判断为一致，75 条低于 0.90，
2 条因 Current Label 与 Suggested Category 实际相同但 `label_match=false` 被判为自相矛盾。
正式数量必须以重新 Run 后的新 Qwen 响应为准。

### 17.2 Reference Size 的含义

Reference Size 来自 `config/vehicle_dimensions.yaml`，其原始业务来源为
《车辆及设施尺寸参考表.xlsx》。它是类别的典型完整三维参考尺寸，不是当前 Cluster 的测量值，
也不是人为拼出的 min/max。例如雪糕筒为 `0.45 x 0.45 x 0.75m`、轿车为
`5.7 x 2.3 x 2.1m`、灯塔为 `2 x 2 x 8m`。

Vision 漏标尺寸门禁使用 Reference Size 做合理性复核：Residual Cluster 是轴对齐 AABB，先将
实测 X/Y 和参考 Length/Width 分别排序，再逐轴计算 `measured / reference`。当前要求每一轴比例
位于 `[0.20, 2.50]`；低于 20% 说明更像局部碎片，高于 2.5 倍说明类别与物理尺度明显矛盾。
表中 Reference Size 为空的类别不编造尺寸，记录 `no_fixed_reference` 后跳过该尺寸门禁。

### 17.3 为什么报告数量大降而 Cluster 数量变化小

这是两个不同层级的数量，不能直接等同：

```text
Residual Cluster（高召回候选）
  -> Projection 质量
  -> Qwen 分类
  -> 置信度硬阈值
  -> Reference Size 门禁
  -> 最终 Rule Report
```

Cluster Detector 的目标仍是尽量不漏掉可疑残余点云，因此源头几何过滤只将旧回放的 491 个候选
降到约 480 个。此前看到的最终漏标从 300 条离线估算降到 2 条，主要不是 Cluster 算法突然只剩
2 个，而是 289 条旧 Qwen 响应低于新的 0.95 主报告阈值、9 条类别与三维尺寸矛盾。因此当前变化
本质上是“最终报告准入显著变保守”，不是“残余聚类数量显著减少”。

错标数量此前从旧报告 43 条降到 Track 共识估算 1 条，主要来自跨帧一致性。恢复单帧模式后，
旧数据估算回升到 16 条，但仍受置信度、别名和自相矛盾门禁限制。

## 18. 使用 Scene 25204 真实 BBox 回校尺寸数据库

更新时间：2026-08-04

### 18.1 数据和统计方法

用户确认原尺寸表中不少值来自网络，不一定符合当前平台 BBox。此次使用
`outputs/reports/bbox_data.csv` 的真实运行结果回校有实际样本的类别：

- 数据全部来自 Scene 25204，共 2,549 行；
- 共 47 个 `className + track_id`，覆盖 12 个平台 Label；
- 同一 Track 的 `scale_x/scale_y/scale_z` 在所有观测帧中保持不变；
- 先对每个 Track 的所有帧取尺寸中位数，再对类别内 Track 等权取中位数，避免长 Track 因帧数
  更多而获得更高权重；
- 本 Scene 没有出现的类别不根据网络资料重新猜测，暂时保留原数据库值；
- 仅 1 个 Track 的类别在 note 中明确标记“样本较少”，等待后续 Scene 累积回校。

### 18.2 本次更新值

本次实测更新的主要 Reference Size（长/宽/高，单位米）：

| 平台 Label / 标准类别 | Track 数 | Reference Size |
|---|---:|---:|
| 雪糕筒 | 5 | 0.45 / 0.45 / 0.75 |
| 轿车 | 6 | 4.79 / 2.08 / 1.94 |
| 卡车头 | 1 | 2.53 / 2.80 / 3.13 |
| 工程车 | 5 | 7.36 / 2.80 / 3.16 |
| 大巴车 | 1 | 8.00 / 2.36 / 2.62 |
| 灯塔 | 1 | 2.26 / 2.25 / 8.00 |
| 牵引车 | 4 | 2.88 / 1.70 / 2.13 |
| 货车 | 2 | 6.72 / 2.54 / 3.06 |
| 皮带车 / 行李传送带车 | 1 | 7.65 / 2.44 / 1.95 |
| 平台车1 / 平台车 | 1 | 9.29 / 4.26 / 3.50 |
| 车辆 / 其他车辆 | 2 | 8.37 / 3.07 / 2.80 |

平台原始 Label 和 Vision 标准类别分别保留数据库入口，例如 `皮带车/行李传送带车`、
`平台车1/平台车`、`车辆/其他车`，使 Rule Engine 和 Vision 都能读取同一实测参考。

### 18.3 托架两类

用户确认平台中存在两类尺寸不同但 `className` 都叫“托架”的目标。为配合统一 ±50% 容差并覆盖
本次真实 BBox，按高度状态分为两组：

- `托架_高型`：13 Track，Reference Size `5.55 / 2.40 / 2.44m`。该中心按观测范围与
  ±50% 容差共同校准，可覆盖高型内部不同长度、宽度和装载高度；
- `托架_低型`：5 Track，高度均约 0.75m，Reference Size `5.15 / 2.90 / 0.75m`。

此前临时拆出的六个状态变体已撤销。高型内部的尺寸变化视为同一类别不同装载或结构状态，不再
错误解释成六个业务类别。Rule Engine 继续利用 `托架_` 前缀执行 any-variant 判断，Vision 的
“卡车拖架”标准类别只映射这两个 Reference Size。

### 18.4 阈值调整

最终配置同时调整漏标 Vision 尺寸上限和 DimensionMismatch tolerance：

- `missing_annotation.confidence_threshold: 0.95 -> 0.92`；
- `category_size_gate.min_axis_ratio: 0.20 -> 0.10`；
- `category_size_gate.max_axis_ratio: 2.50 -> 2.00`，即最大允许 Reference Size 的 200%；
- 所有有 Reference Size 的类别统一使用 `tolerance=0.50`，即 DimensionMismatch 允许 ±50%；
- 无固定 Reference Size 的组合类别 tolerance 继续为 null。

使用旧 Qwen 响应和最终数据库做只读离线回放：旧 300 条 AI 漏标中，198 条低于 0.92，54 条类别/
尺寸仍矛盾，48 条通过。该结果只表示新门禁对旧响应的影响，正式结果必须重新 Run。

### 18.5 验证

- Scene 25204 出现的 12 个平台 Label、2,549 行 BBox 全部得到 `status=ok`，无 unknown class；
- 托架最终仅保留高型、低型两个变体；13 个高型 Track、5 个低型 Track；
- 36 个有固定参考尺寸的数据库类别全部为 ±50%，1 个组合类别保持 null；
- 使用更新后的 Reference Size 与 any-variant 逻辑回放 2,549 行真实 BBox，DimensionMismatch 为 0；
- Vision 类别配置中的所有尺寸键均指向现有数据库项；
- 数据库和 `config.yaml` 可正常解析；
- 未修改 BBox 收集、Cluster、Projection、Calibration、Assets、Browser Automation 和 Report 格式。

## 19. UI V1.1：一键启动与 Reset

更新时间：2026-08-04

### 19.1 修改范围

本次只修改 UI 和启动入口，没有修改 Browser Automation、BBox、Assets、Rule Engine、Cluster、
Projection、Vision、Report 等后端算法或业务逻辑。

原 UI 使用 `sys.executable` 启动 `main.py` 和 `cleanup_scene.py`。当用户没有先激活项目虚拟环境时，
这里可能变成系统 Python，导致依赖缺失。`ui_app.py` 与 `ui_app.pyw` 原先还各自保存了一份完整 UI
实现，存在两份入口行为逐渐不一致的风险。

### 19.2 项目虚拟环境与一键启动

UI 启动时现在会按顺序查找项目根目录中的 `.venv`、`venv`：

- Ubuntu 使用 `bin/python`；
- Windows 后端使用 `Scripts/python.exe`，UI 优先使用无控制台窗口的 `Scripts/pythonw.exe`；
- 找不到时显示 `Virtual Environment not found.`，并给出一次性初始化命令，不进入不完整 UI；
- 找到后先检查 YAML、Playwright、NumPy、scikit-learn、Pillow、OpenAI SDK 和 python-dotenv 等运行依赖；
- 当前 UI 如果不是由该项目环境运行，会自动使用项目解释器重新启动自身；
- Inspection 和 Cleanup 子进程都显式使用同一个项目解释器，不再依赖用户是否执行过 `activate`。

新增双击入口：

- Windows：`Start_QP_Copilot.vbs`；
- Ubuntu：`start_qp_copilot.sh`。

`ui_app.pyw` 改为调用 `ui_app.py` 的轻量入口，UI 逻辑只保留一份。虚拟环境仍只需首次创建和安装
依赖，日常使用不再需要打开 Terminal 或手动激活环境。

### 19.3 Reset 行为

新增 Reset 按钮。Reset 只恢复当前 UI 内存状态：

- 清空 Scene Number、Frame Count、Cleanup Scene Number 和 Calibration YAML；
- 进度恢复为 `0 / 0` 和 0%；
- 状态恢复为 `Ready`；
- 清空运行日志；
- 清空 Rule Report、BBox Data、Rule Summary 的 UI 路径，并禁用对应 Open 按钮；
- 清空当前 Scene、任务线程引用和待处理 UI 事件；
- 恢复 Start、Cleanup、Reset 按钮状态，并将输入焦点放回 Scene Number。

Reset 不启动后端进程，不通知 Browser，不执行 Cleanup，也不删除 Assets、Report 或任何磁盘文件。
Inspection 或 Cleanup 正在运行时 Reset 保持禁用，避免后台进程继续写入已经被重置的界面。

### 19.4 验证

- `ui_app.py`、`ui_app.pyw` Python 编译检查通过；
- Ubuntu 启动脚本 Bash 语法和可执行权限检查通过；
- 当前 `.venv` 能被正确发现，核心依赖检查通过；
- 使用真实 Tk UI 填充所有输入、日志、进度和报告状态后执行 Reset，所有状态恢复到初始值；
- Reset 前后现有 Report 文件的存在状态和修改时间完全不变；
- 使用模拟后端进程验证 Inspection 与 Cleanup 命令均以项目 `.venv/bin/python` 启动；
- `git diff --check` 通过。

### 19.5 Ubuntu 双击启动修正

初版让用户直接双击 `start_qp_copilot.sh`。Ubuntu 文件管理器会根据本机偏好将 `.sh` 当作文本文件，
或要求右键选择运行，并可能打开一个随 UI 一直存在的 Terminal，因此不能达到真正的图形化双击启动。

项目新增可上传 GitHub 的 `QP_Copilot.desktop.in` 模板和 `install_ubuntu_launcher.sh` 安装脚本：

- 安装时根据项目实际目录生成桌面 `QP Copilot.desktop`，不在仓库中写死本机路径；
- `Terminal=false`，启动 UI 时不创建额外 Terminal；
- `Exec` 指向项目中的 `start_qp_copilot.sh`，继续复用已有虚拟环境检查和错误提示；
- 同时安装到用户应用菜单，并将桌面启动器设置为可执行和 GNOME 可信；
- README 的 Ubuntu 日常入口改为桌面 `QP Copilot` 图标，`.sh` 只作为底层启动脚本保留。

## 20. 错标 Vision 按 Track 去重

更新时间：2026-08-04

Scene 25204 的旧流程在 8 个采样资产帧中生成 260 个 BBox manifest，其中 229 个实际进入 AI，
但同一个 `track_id` 会在多个采样帧重复调用。错标检测改为每个稳定 Track ID 只检查一次：

- 仍然从原有采样帧收集 Existing BBox Candidate，不修改 BBox、Assets 或 Projection；
- 按 `track_id` 分组；没有 Track ID 时才回退使用 `candidate_id`，避免错误合并未知目标；
- 每组优先选择合格投影视角更多、可见点更多、ROI 面积更清楚的代表帧；
- 完全相同时选择更早帧，保证结果稳定可复现；
- `wrong_annotation.vision_test_limit` 现在限制 Track 数，而不是跨帧 BBox 观测数；
- Missing Annotation Cluster 的选择和调用方式不变。

## 21. 漏标 Vision 跨帧去重

更新时间：2026-08-04

新增 Vision 前静止 Cluster 去重。由于 PCD 使用逐帧自车坐标系，程序不直接比较原始 Cluster
center，而是用同 Track ID 的已有 BBox 通过 RANSAC 估计自车运动，将 Cluster 变换到公共坐标系
后，再结合中心、尺寸、PCA 和至少 3 帧重复条件分组。真实运动目标在公共坐标系中仍会移动，因此
保持逐帧调用；配准失败的帧也保持逐个调用。

Scene 25204 历史回放从 491 个漏标候选中识别 55 个稳定重复组，节省 168 次 API，预计剩余 323
次。该改动仅位于 Vision Candidate 选择层，不修改 Cluster Detection、二级合并、Projection、
BBox 或 Rule Engine。完整参数和风险记录见桌面 Cluster 逻辑文档第 39 节及修改报告第 13 节。

### 21.1 连续两采样帧规则

漏标 Vision 去重门槛改为相邻两个采样帧。程序使用完整采样帧时间线，禁止跨过缺少对应 Cluster
的中间帧进行关联。Scene 25204 新回放为 71 个连续重复组、节省 132 次 API，预计调用量
`491→359`。第 21 节的 `55组/168次` 是此前至少三帧方案的历史结果。

### 21.2 最终恢复三帧方案

经同一 Scene 回放对比，连续两采样帧方案预计节省 132 次 API，至少三次稳定观测方案预计节省
168 次。最终以减少调用量为目标恢复三帧方案，允许中间采样帧 Cluster 暂时缺失；第 21.1 节仅
保留为对比记录，当前有效结果重新为 `491→323`。

## 22. Vision 调用性能优化

更新时间：2026-08-06

新增代码级 Vision 优化，GitHub 用户提交后可直接获得：

- `vision_workers=2` 两路 Candidate 并发，结果仍按原任务顺序汇总；
- 单个 Run 复用一个 HTTP/OpenAI Client，结束时统一关闭；
- `vision_cache.py` 按模型、Prompt/Verifier 源码、类别配置、Metadata 和图片字节生成 SHA256；
- 只缓存 verified 结果，错误和跳过继续重试；
- 缓存位于 `ai_inputs/scene_<id>/.vision_cache`，Cleanup 随 Scene 删除；
- 日志和 `vision_test_stats` 增加实际 API、缓存、重试及耗时字段。

无网络固定延迟测试中，两路并发相对单线程从 0.879 秒降至 0.505 秒，下降 42.5%；相同输入重跑
为 4 个缓存命中、0 请求、0.068 秒；单图变化只失效对应一个 Key。Scene 25204 当前估算首次实际
请求约 141 次，两路并发理论节省约 70 个平均请求延迟，全缓存重跑节省约 141 个平均请求延迟。
没有使用真实 Qwen 做性能测试，下一次完整 UI Run 将通过新增统计给出真实分钟数。

## 23. Scene 29694 运行分析、尺寸库回校与 Static 规则取消

更新时间：2026-08-06

### 23.1 本次运行与耗时

本次 UI 完整运行的 Scene 为 29694，共 81 帧、1737 个 BBox。项目没有单独保存整次 Run 的起止
日志时间戳，因此总时长采用落盘产物估算：最早的 `calibration_used.yaml` 为 14:42:08.897，最终
`rule_summary.json` 为 14:57:08.213，可观测总时长约 899.3 秒，即 14 分 59 秒。真实点击 Start
时间可能略早，因此该数字是基于文件时间戳的下界，不冒充精确 UI 计时。

Vision 阶段统计来自 `rule_summary.json`，是程序直接测量值：

- 167 个真实 API Candidate：漏标 141、错标 26；缓存命中 0，重试 0，失败 0；
- 单请求平均 4.681 秒，API 串行累计耗时 781.69 秒；
- 两路并发请求阶段墙钟时间 392.31 秒，Vision 总墙钟时间 393.98 秒；
- 相比顺序请求，本次估算节省 389.38 秒，即约 6 分 29 秒；
- Vision 占可观测总时长约 43.8%，非 Vision 阶段约 8 分 25 秒。

漏标侧共得到 380 个原始 Cluster；跨帧去重形成 7 组、覆盖 23 个观测并节省 16 个 Candidate。
Projection 质量门过滤 233 个，141 个漏标 Candidate 实际调用 Qwen。AI 判为真实目标 23 个、超出
目标类别范围 10 个、尺寸/类别后门控排除 22 个，最终 1 个 PossibleMissingAnnotation 写入主报告。
用户人工抽查认为本次漏标效果较好。

Assets metadata 显示 Frame 1 为 `pcd_success=false, image_count=0`，Frame 11 至 81 的 8 个采样帧
均下载 1 个 PCD 和 5 个 JPG。因此本次 Cluster 实际覆盖 8 个资产采样帧。该事实本次只记录，未修改
Assets Downloader。

### 23.2 DimensionMismatch 原因与数据库回校

修改前报告共有 131 条 DimensionMismatch，全部来自旧 Scene Reference 与本次真实 BBox 尺寸不一致：
卡车头 43 条、托架 44 条、工程车 44 条，不是本次 BBox 自身发生尺寸跳变。

使用本次 `bbox_data.csv` 的 1737 行、36 个 Track、7 个 Label 回校。统计仍采用“同一 Track 各帧先
取中位数，再让类别内各 Track 等权”的方法，避免长 Track 因帧数多而支配 Reference。数据库更新为：

- 卡车头：本次 9 Track，Reference 更新为约 4.24 x 3.23 x 3.49m；
- 托架：保持高型/低型两类，不按 Scene 增加类别；Reference 取能同时覆盖 Scene 25204 和 29694
  真实尺寸的中心，高型约 9.56 x 2.75 x 3.20m，低型约 9.50 x 3.00 x 1.10m；
- 工程车：平台 Label 不区分具体设备，数据库保留“常规”和“大型港区设备”两个变体，
  `any_variant` 命中任一真实变体即通过；
- 轿车、大巴车、灯塔按本次真实 Track 回校，同时确保此前真实范围仍在 +/-50% 容差内；
- 新增平台完整 Label `场桥 轮胎吊 轨道吊`，避免它只能依赖简写名称。

容差仍统一为每轴 50%，没有改阈值算法。对本次 1737 个 BBox 逐行复算结果为：
`dimension_failures=0`、`unknown_or_skipped=0`。现有 `rule_report.csv` 保留修改前 131 条历史结果，
用于审计；下一次 Run 才会按新数据库生成报告。

### 23.3 取消 StaticObjectPosition

StaticObjectPosition 直接比较相邻帧中的原始 BBox position，没有补偿自车运动。即使灯塔、场桥等
物体在世界坐标中静止，自车移动后它们在 LiDAR/自车坐标中的数值也会变化，因此该规则会产生系统性
误报。本次报告中的 17 条 StaticObjectPosition 均不再作为有效告警。

已从三个层面完整取消：

- `config.yaml -> rule_engine.enabled_rules` 不再启用；
- 删除 `static_object_position_threshold` 和 `static_object_classes` 配置；
- 删除规则函数及 `RULE_REGISTRY` 注册项，并同步移除 README 规则说明。

验证结果：配置未启用、注册表不存在、全项目业务代码无 StaticObjectPosition 引用。若用新代码对同一
数据重新生成核心规则结果，DimensionMismatch 131 条和 StaticObjectPosition 17 条都不会再出现；
历史总告警 159 条中其余 11 条为 EmptyFrame 6、PossibleMissingAnnotation 1、VisionLabelMismatch 4。

### 23.4 验证

- `config.yaml` 与 `config/vehicle_dimensions.yaml` YAML 解析通过；
- `rule_engine.py`、`vehicle_dimension_config.py`、`main.py`、`vision_pipeline.py` 编译通过；
- Scene 29694 全部 1737 个 BBox 尺寸复算为 0 条失败、0 条未知类别；
- `git diff --check` 通过。

## 24. 默认跳过 Scene 首帧

更新时间：2026-08-06

Scene 29694 的 Frame 1 出现 `pcd_success=false, image_count=0`，原因是程序连接现有浏览器页面后
才安装 Network Listener，而当前首帧的 PCD/JPG 通常已经加载完成。Listener 无法补获启动前的请求，
继续对首帧执行 Frame Ready 只会等待到超时。

新增 `scene.skip_first_frame: true`，默认行为调整为：

- UI 输入的 Frame Count 仍代表 Scene 总帧数，81 帧在 `rule_summary.json` 中仍记为 81；
- Frame 1 不导航、不等待、不截图、不提取 BBox、不记录网络资源、不下载 Assets；
- Frame 1 不加入 MissingBBox 或 EmptyFrame 判断，不制造人为告警；
- 首个实际质检帧为 Frame 2，并强制调用 Navigator 切到 Frame 2，不使用原先“首循环假定已在当前帧”
  的快速路径；
- BBox 和普通规则检查 Frame 2 至 Frame 81，共 80 帧；UI 进度为 1/80 至 80/80；
- Assets 的采样锚点同步改为 Frame 2。interval=10 时采样 2、12、22、32、42、52、62、72，
  Frame 81 不在新采样序列内；
- 若以后需要恢复旧行为，将 `skip_first_frame` 设为 `false` 即可，代码无需修改。

本次没有修改 Navigator、Frame Ready、BBox、Assets 下载器内部下载逻辑、Cluster、Projection、Vision
或 Rule Engine 算法，只调整主 Pipeline 传入这些模块的帧列表和采样起点。

验证：Scene 范围为 1–81 共81帧，质检范围为2–81共80帧，Frame 1 不在质检列表；采样序列精确为
`[2,12,22,32,42,52,62,72]`；UI 进度正则可解析 `[进度] 第 1/80 帧`；Python 编译、YAML 解析和
`git diff --check` 均通过。

## 25. README 定稿与发布清理

更新时间：2026-08-06

README 按当前生产代码完整重写为 17 节，使用顺序为：项目用途与 Pipeline、软件要求、Qwen API
配置、Windows 安装、Ubuntu 安装与桌面入口、UI 使用、Calibration、Browser/Frame Ready、Assets、
Cluster/Projection/Vision、Rule Engine、输出、Cleanup、配置索引、文件结构、故障处理和最终 Run 清单。

API 文档明确说明 Workspace/地域与 API Base 必须匹配，Key 只放项目根目录 `.env`，并分别给出
Windows/Ubuntu 的创建和连通性检查命令。为保证命令有效，`check_qwen_connection.py` 现在与 UI/
main.py 一样自动读取 `.env`，不再要求手动 `export`。

桌面 UI 文档覆盖：

- Windows 首次创建 `.venv`、依赖安装、Edge Remote Debugging 和双击
  `Start_QP_Copilot.vbs`；
- Ubuntu 系统依赖、`.venv`、Edge Remote Debugging、运行
  `install_ubuntu_launcher.sh` 生成无 Terminal 桌面图标；
- 两个平台日常启动都不需要手动 activate，也不需要手动执行 `python main.py`。

发布清理删除 `demo_cluster_projection.py`、`run_vision_test.py`、全部 `test_*.py`、`*.pyc` 和
`__pycache__`。保留 `check_qwen_connection.py`，因为它是正式 API 配置诊断工具；保留
`analyze.py/analyzers.py`，因为 main.py 仍调用兼容分析层。整理 `.gitignore` 的重复项，并新增
`.pytest_cache/`、`.coverage`。

发布前未启动浏览器、未调用 Qwen，避免影响最后一次真实 Run。离线检查结果：31 个 Python 源文件
编译通过、30 个生产模块导入通过、3 个 YAML 解析通过、Ubuntu 两个 Shell 启动脚本语法通过、
`.env` 被 Git 忽略且运行时可检测到 Key、无真实 API Key 泄漏、无测试/字节码残留、
`git diff --check` 通过。

### 25.1 Edge 命令、CDP 解释与单页面要求

README 的 Ubuntu Edge 命令改为当前实际验证版本：

```bash
microsoft-edge-stable \\
  --remote-debugging-port=9222 \\
  --user-data-dir="$HOME/edge-qp-debug"
```

删除此前额外写入的 `--remote-debugging-address`。CDP 明确解释为 Chrome DevTools Protocol，即
Playwright 用来连接 Edge 9222 调试端口并复用登录 Session 的浏览器协议，不是 CDG，也不需要单独
安装。

当前 `browser.py` 在 `page_title_hint` 为空时使用第一个标签页，hint 匹配失败时也会回退第一个
标签页。因此 README 将以下内容设为强制运行条件：Edge 中关闭所有其他标签页，只保留当前 QP 质检
Scene 页面，运行期间不得打开新标签页；`page_title_hint` 不能替代该条件。

## 26. Scene 29696 DimensionMismatch 回校

更新时间：2026-08-06

最新报告中有 10 条 `DimensionMismatch`，实际只涉及托架 Track 6、托架 Track 11 和灯塔
Track 16。同一 Track 因中间帧缺失被按连续区间折叠为多条报告，不代表存在 10 个独立尺寸错误。

对照本次真实 BBox 后确认长宽高字段映射正常，误报来自 Reference Size 尚未覆盖新 Scene：

- 托架实测长度为 15.20m 和 15.45m，旧高型参考 9.56m 的 +50% 上限仅 14.34m；
- 灯塔实测为 2.5 x 2.5 x 8.0m，旧底面参考约 1.5 x 1.5m 的 +50% 上限约 2.25m。

保持全部类别统一 `tolerance=0.50` 不变，仅使用真实数据回校参考中心：

- `托架_高型` length：9.56m -> 10.50m，新范围 5.25–15.75m；
- `灯塔` length/width：1.51/1.50m -> 2.00/2.00m，新底面范围 1.00–3.00m；
- 托架的高/低型变体结构、`any_variant` 策略、Rule Engine 算法和报告格式均未修改。
