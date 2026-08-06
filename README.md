# QP Copilot V2.0

QP Copilot 是用于自动驾驶标注质检的桌面辅助工具，检查范围包括漏标、错标以及跨帧一致性、
轨迹连续性、尺寸异常和空帧等规则。用户在 UI 中输入 Scene Number、Frame Count，粘贴
LiDAR-Camera Calibration YAML，然后点击 **Start Inspection**。程序复用已登录的 Edge 浏览器会话，
自动完成 BBox 采集、PCD/JPG 下载、规则检查、Residual Cluster、相机投影、Qwen Vision 复核和
统一报告生成。

最终质检结论只写入：

- `outputs/reports/rule_report.csv`
- `outputs/reports/rule_summary.json`

工具只读取 QP 页面并在本地分析，不会点击“合格、驳回、提交”，不会修改平台标注或评审状态。

---

## 1. 当前完整流程

```text
打开已登录的 QP Scene 质检页面
-> 双击桌面 QP Copilot
-> 输入 Scene Number + Frame Count + Calibration YAML
-> Start Inspection
-> 通过 CDP（浏览器远程调试协议）连接 Edge 端口 9222
-> 从 Frame 2 开始自动切帧、等待资源、采集 BBox
-> 按 sample_interval 下载 PCD + 5 路 JPG
-> 普通 Rule Engine
-> 删除已有 BBox 内点并运行两级 Residual Cluster
-> Cluster / Existing BBox 投影到可见 Camera
-> 生成单 Candidate context/crop/manifest
-> Qwen Vision 漏标与错标复核
-> 合并全部 RuleFinding
-> rule_report.csv + rule_summary.json
```

UI 显示固定 9 个阶段：

1. Loading Configuration
2. Connecting Edge Browser
3. Collecting Frames, BBox and Assets
4. Writing Capture and BBox Data
5. Running Rule Engine
6. Running Cluster Detection and Projection
7. Running Vision Verification
8. Merging Rules and Writing Final Report
9. Inspection Finished

单帧、单 Cluster、单 BBox 或单次 Vision 请求失败时会记录错误并继续。只要报告仍可写，整个
Scene 会继续生成最终报告。

---

## 2. 运行前准备

### 2.1 软件要求

- Python 3.10-3.12，推荐 Python 3.12
- Microsoft Edge 或 Google Chrome
- Qwen Model Studio API Key
- 与当前 Scene 匹配的 Calibration YAML
- Windows 10/11，或带桌面环境的 Ubuntu

依赖由 `requirements.txt` 统一管理，包括 Playwright、PyYAML、NumPy、scikit-learn、Pillow、
OpenAI SDK 和 python-dotenv。

### 2.2 获取项目

```bash
git clone https://github.com/hhanting1-ux/QP-copilot.git
cd QP-copilot
```

也可以下载 ZIP。后续命令都在能看到 `config.yaml`、`ui_app.py` 和
`requirements.txt` 的项目根目录执行。

---

## 3. Vision API 配置

### 3.1 API Key、Workspace 和地域

在 Alibaba Cloud Model Studio 中确认：

- API Key
- Workspace ID
- Workspace 地域
- 可用模型，例如 `qwen3-vl-plus`

API Key 和 API Base 必须属于兼容的 Workspace/地域。地域不一致可能返回 HTTP 401，即使 Key
格式正确。

OpenAI-compatible API Base 通常为：

```text
https://<workspace-id>.<region>.maas.aliyuncs.com/compatible-mode/v1
```

北京地域示例：

```text
https://<workspace-id>.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
```

在 `config.yaml` 填写实际配置：

```yaml
ai_vision:
  enabled: true
  provider: "qwen"
  model_name: "qwen3-vl-plus"
  api_base: "https://<workspace-id>.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
  api_key_env: "DASHSCOPE_API_KEY"
```

### 3.2 创建 .env

API Key 只保存在项目根目录 `.env`，不要写入代码或 `config.yaml`。

Windows PowerShell：

```powershell
Copy-Item .env.example .env
notepad .env
```

Ubuntu：

```bash
cp .env.example .env
nano .env
```

内容：

```dotenv
DASHSCOPE_API_KEY=你的实际API_KEY
```

`.env` 已被 `.gitignore` 忽略，程序自动读取。

### 3.3 验证 API

该命令会发送一张 64x64 临时灰色图片，产生一次最小 API 调用。

Windows：

```powershell
.\.venv\Scripts\python.exe check_qwen_connection.py
```

Ubuntu：

```bash
.venv/bin/python check_qwen_connection.py
```

成功输出：

```text
status=verified
model=qwen3-vl-plus
api_base=...
Qwen-VL connection verified.
```

错误处理：

- `Invalid API-key`：Key 错误，或 Key 与 API Base 的 Workspace/地域不匹配。
- `Unknown scheme for proxy URL socks://...`：保持
  `ai_vision.use_environment_proxy: false`，除非代理兼容 HTTPX。
- `.env 不存在`：确认文件名是 `.env`，不是 `.env.txt`。
- `missing_api_key`：确认变量名与 `api_key_env` 相同。

---

## 4. Windows 安装与启动

### 4.1 首次创建环境

打开 PowerShell 并进入项目目录：

```powershell
cd C:\path\to\QP-copilot
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m playwright install chromium
```

Windows 官方 Python 通常包含 Tkinter。若 `import tkinter` 失败，重新运行 Python 安装器并启用
Tcl/Tk。然后按第 3 节创建 `.env` 并验证 Qwen。

### 4.2 Edge Remote Debugging

先完全关闭所有 Edge 窗口，再在 PowerShell 执行。常见 32 位安装目录：

```powershell
& "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe" `
  --remote-debugging-port=9222 `
  --user-data-dir="$env:USERPROFILE\edge-qp-debug"
```

64 位安装目录：

```powershell
& "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe" `
  --remote-debugging-port=9222 `
  --user-data-dir="$env:USERPROFILE\edge-qp-debug"
```

独立 `user-data-dir` 会保存 QP 登录状态。首次启动需要登录一次。

浏览器启动后：

1. 登录 QP。
2. 打开需要质检的 Scene。
3. **关闭所有其他标签页，只保留当前 QP 质检 Scene 页面。**
4. 运行期间不要再打开新标签页。

这是当前版本的强制使用条件。脚本通过 Remote Debugging 连接整个 Edge Context；存在其他页面时
可能选择第一个标签页并把普通网页误判为质检页。即使配置了 `browser.page_title_hint`，匹配失败后
仍会回退到第一个标签页，因此不能用它替代“只保留质检页面”。

### 4.3 桌面 UI

双击项目根目录的：

```text
Start_QP_Copilot.vbs
```

启动器自动使用 `.venv\Scripts\pythonw.exe`，不会打开额外 Terminal，不需要手动 activate。
虚拟环境不存在或依赖不完整时，UI 会显示初始化命令并停止。

---

## 5. Ubuntu 安装与桌面 UI

### 5.1 首次创建环境

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv python3-tk git zenity

cd /path/to/QP-copilot
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m playwright install chromium
```

然后按第 3 节创建 `.env` 并验证 Qwen。

### 5.2 安装桌面图标

只需执行一次：

```bash
chmod +x start_qp_copilot.sh install_ubuntu_launcher.sh
./install_ubuntu_launcher.sh
```

安装脚本按当前项目绝对路径生成：

- 桌面 `QP Copilot.desktop`
- 应用菜单 `~/.local/share/applications/qp-copilot.desktop`

启动器使用 `Terminal=false`。不要直接双击 `start_qp_copilot.sh`，它是桌面入口调用的底层脚本。
如果移动项目目录，需要在新目录重新运行安装脚本。部分桌面环境首次使用时还需要右键图标并选择
**Allow Launching**。

### 5.3 Edge Remote Debugging


```bash
microsoft-edge-stable \
  --remote-debugging-port=9222 \
  --user-data-dir="$HOME/edge-qp-debug"
```

这就是当前项目实际验证使用的 Ubuntu 启动命令。Edge 启动后登录 QP、打开
目标 Scene，然后**关闭所有其他标签页，只保留当前质检 Scene 页面**。运行期间也不要打开新标签页，
否则脚本可能连接错误页面。

### 5.4 日常启动

双击桌面 **QP Copilot**，或从应用菜单打开。不需要：

```bash
source .venv/bin/activate
python main.py
```

---

## 6. 使用桌面 UI

必填项：

- **Scene Number**：QP Scene ID。
- **Frame Count**：Scene 总帧数，例如 81。
- **Calibration YAML**：完整 YAML 文本。

任何一项为空会立即报错。Calibration 无法解析或没有 Camera 时也会在启动前报错。

操作顺序：

1. 确认 Edge 已用调试端口启动并打开正确 Scene。
2. 输入 Scene Number 和 Frame Count。
3. 粘贴与当前 PCD/JPG 匹配的 Calibration YAML。
4. 点击 **Start Inspection**。
5. 等待 `[9/9] Inspection Finished`。
6. 点击两个报告旁的 **Open**。

按钮行为：

- **Open Report Folder**：打开 `outputs/reports/`。
- **Reset**：只清空 UI 状态，不删除报告/Assets，不关闭 Edge。
- **Delete Scene Files**：输入 Scene Number，再输入 `DELETE`，清理该 Scene 本地数据。

### 首帧策略

```yaml
scene:
  start_frame_index: 1
  skip_first_frame: true
```

程序连接后才安装 Network Listener，当前首帧资源通常已经加载完，等待它会产生假超时。因此：

- Scene 总帧数仍按 UI 输入记录。
- Frame 1 不导航、不等待、不截图、不提取 BBox、不下载 Assets。
- Frame 1 不记 MissingBBox 或 EmptyFrame。
- 从 Frame 2 开始质检。

81 帧 Scene 检查 Frame 2-81，共 80 帧，UI 显示 `1/80` 到 `80/80`。

---

## 7. Calibration 要求

从任务检索中找到当前 Scene ID（32 位）对应的数据集，复制 Calibration 参数并粘贴到 UI 的
**Calibration YAML** 输入框。

UI 使用 `CalibrationLoader.loads()`。YAML 根节点必须是 Mapping，参数结构包含：

```text
hardware
  sensors
    camera_labels
    cameras
    lidars
```

Camera 可包含 `image_size`、`intrinsic`、`distort`、`distortion_type`、`extrinsic`、
`default`、`location`、`device` 和 `type`。

新增 Camera 只修改 YAML，不修改代码。Calibration 可以包含多于 5 个 Camera；Projection 只处理
当前采样帧中存在对应 JPG 且投影可见的 Camera。

本次输入归档到：

```text
assets/scene_<scene_id>/calibration_used.yaml
```

---

## 8. 浏览器、切帧与等待

### 8.1 切帧

`navigator.py` 依次执行：

1. 帧号文本精确匹配。
2. 只保留可见元素。
3. 只保留页面底部候选。
4. 多候选取最靠近底部者。
5. 找不到时坐标 fallback。

```yaml
navigation:
  bottom_region_ratio: 0.75
  timeline_start_x: 200
  timeline_y: 780
  frame_step_px: 12
```

### 8.2 页面就绪

`frame_ready.py` 等待期望 PCD/JPG、网络安静期和额外渲染时间。超时只记 Warning，不停止 Scene。

---

## 9. Assets、Cluster、Projection 与 Vision

### 9.1 Assets

```yaml
assets:
  enabled: true
  sample_interval: 10
  download_pcd: true
  download_jpg: true
```

`sample_interval` 只影响 PCD/JPG、Cluster、Projection 和 Vision 的资产帧数量，不影响 Frame 2
起的逐帧 BBox 和普通规则。

默认采样锚点是 Frame 2。81 帧、interval=10：

```text
2, 12, 22, 32, 42, 52, 62, 72
```

### 9.2 Residual Cluster

```text
读取 binary / binary_compressed PCD
-> 删除已有 Oriented BBox 内点
-> 自车区域过滤
-> 100m 水平范围过滤
-> Voxel Downsample
-> Height Filter
-> 点级 HDBSCAN（可配置 DBSCAN）
-> Cluster Fragment AABB / Z 二级合并
-> 重新计算 point count、center、AABB、volume
-> PCA linearity / planarity / flatness
-> 几何质量过滤
-> Residual Cluster Candidate
```

第二级合并用于组合同一物体被点级聚类拆开的碎片，不负责类别识别。

### 9.3 Projection 与 AI 图片

`projection_pipeline.py` 消费现有 Cluster/BBox 和 Calibration。每个可见 Candidate 生成：

- `context.jpg`：全场景，只画当前 Candidate。
- `crop.jpg`：局部无框图片。
- `manifest.json`：3D 几何、ROI、Camera、visible_points 和路径。

原始 JPG 不覆盖。Projection 低质量 Candidate 不调用 Qwen。

### 9.4 Vision

漏标：

```text
Residual Cluster -> Projection -> AI Input -> Qwen -> PossibleMissingAnnotation
```

错标：

```text
Existing BBox -> Projection -> AI Input -> Qwen
-> VisionLabelMismatch / VisionBBoxMismatch
```

两种 Candidate 共用 Projection、AI Input、Verifier 和 Report。当前优化：

- 每次只判断一个 Candidate。
- 同一 BBox Track 只选择投影质量最佳的一帧。
- 漏标先用已有 Track 估计自车运动，再按至少 3 次稳定观测去重。
- 两路 API 并发并复用 HTTP Client。
- 成功结果按内容 SHA256 缓存到 `ai_inputs/scene_<scene_id>/.vision_cache/`。
- 类别必须位于 `config/vision_categories.yaml`。
- 不确定、背景、噪声、超出类别或尺寸门控失败不写成最终漏标。

---

## 10. 质检规则

最终报告统一合并三类结果：普通 Rule Engine 规则、Residual Cluster 漏标复核，以及 Existing BBox
错标复核。`VisionLabelMismatch` 和 `VisionBBoxMismatch` 由 Vision Pipeline 生成，不属于普通
`RULE_REGISTRY`，但使用相同的 `RuleFinding` 数据结构并写入同一份报告。

| Rule ID | 职责 |
|---|---|
| `LabelConsistency` | 同一 Track 跨帧 Label 变化 |
| `PositionJump` | 相邻帧位置跳变 |
| `RotationJump` | 相邻帧 rotation_z 跳变 |
| `SizeConsistency` | 同一 Track 尺寸比例变化 |
| `BrokenTrack` | 短暂消失后重新出现 |
| `EmptyFrame` | 成功读取但 BBox 数量为 0 |
| `SizeOutlier` | 单轴尺寸超出全局范围 |
| `DimensionMismatch` | 实测尺寸超出 Reference Size +/- Tolerance |
| `PossibleMissingAnnotation` | Residual Cluster 疑似漏标；Vision 开启时由 Vision 生成最终结论 |
| `VisionLabelMismatch` | Existing BBox 覆盖目标，但 Qwen 复核认为当前 Label 与图像内容明显不一致 |
| `VisionBBoxMismatch` | Qwen 复核认为 Existing BBox 未正确覆盖所标目标，例如框偏离、框住背景或无对应目标 |
| `MissingBBox` | 读取失败 |



尺寸库位于 `config/vehicle_dimensions.yaml`，判断公式是
`reference_size * (1 +/- tolerance)`。同一 Label 有多种真实尺寸时使用变体，
`multi_variant_strategy: any_variant` 表示任一变体通过即不报警。

---

## 11. 输出目录

### 最终报告

```text
outputs/reports/rule_report.csv
outputs/reports/rule_summary.json
```

`rule_report.csv` 包含 Scene/Frame/Track/Cluster、3D center/size、Camera/ROI、类别、置信度、
原因和图片路径。

`rule_summary.json` 包含总帧/BBox/Rule 统计，以及漏标/错标 API 次数、缓存、重试、失败、
平均耗时、墙钟时间和并发节省。

### 中间文件

```text
outputs/
  reports/
    bbox_data.csv
    capture_report.csv
    network_assets.csv
    mapping_report.txt
    bbox_probe_report.json
  screenshots/

assets/
  scene_<scene_id>/
    calibration_used.yaml
    metadata.json
    frame_NNNN/
      pointcloud.pcd
      images/
      projection_debug/

ai_inputs/
  scene_<scene_id>/
    frame_NNNN/
      cluster_xxx/
      bbox_xxx/
    .vision_cache/
```

固定报告会被下一次 Run 覆盖；Assets 和 AI Inputs 按 Scene/Frame/Candidate 保存。

---

## 12. Reset 与 Cleanup

Reset 只恢复 UI，不删除文件、不关闭浏览器。

Cleanup 删除目标 Scene 的：

- `outputs/reports` 当前批次报告
- `outputs/screenshots`
- `assets/scene_<scene_id>`
- `ai_inputs/scene_<scene_id>`
- Vision Cache

不会删除代码、配置、`.env`、虚拟环境或其他 Scene。

命令行 dry-run：

Windows：

```powershell
echo 29694 | .\.venv\Scripts\python.exe cleanup_scene.py --dry-run
```

Ubuntu：

```bash
printf '29694\n' | .venv/bin/python cleanup_scene.py --dry-run
```

---

## 13. 重要配置

| 配置 | 作用 |
|---|---|
| `scene.skip_first_frame` | 跳过首帧，从 Frame 2 质检 |
| `browser.cdp_url` | Remote Debugging 地址 |
| `browser.page_title_hint` | 多标签页匹配 |
| `frame_ready.timeout_seconds` | 单帧最大等待 |
| `frame_ready.render_settle_ms` | 网络完成后的渲染等待 |
| `assets.sample_interval` | PCD/JPG/Projection/Vision 采样 |
| `rule_engine.*` | 普通规则和 Cluster 参数 |
| `projection.*` | 可见点、Padding、Debug |
| `ai_inputs.quality*.` | ROI 质量门 |
| `ai_vision.api_base` | Qwen Workspace 地址 |
| `ai_vision.model_name` | Vision 模型 |
| `ai_vision.vision_workers` | API 并发，当前为 2 |
| `ai_vision.cache` | 成功结果缓存 |
| `ai_vision.*.confidence_threshold` | 报告准入置信度 |
| `report.output_dir` | 报告目录 |

详细阈值在 `config.yaml` 中有注释。正式 Run 前不要同时大幅调整多个阈值。

---

## 14. 项目结构

```text
QP-copilot/
├── main.py                         # 9阶段 Pipeline
├── ui_app.py / ui_app.pyw          # 桌面 UI / Windows 无控制台入口
├── Start_QP_Copilot.vbs            # Windows 双击启动
├── start_qp_copilot.sh             # Ubuntu UI 启动
├── install_ubuntu_launcher.sh      # Ubuntu 桌面安装
├── QP_Copilot.desktop.in           # 桌面入口模板
├── runtime_env.py                  # .env 加载
├── browser.py / navigator.py       # 浏览器会话与切帧
├── network_recorder.py             # PCD/JPG 网络监听
├── frame_ready.py / capture.py     # 就绪等待与截图
├── report.py                       # capture_report.csv 数据结构
├── analyzers.py / analyze.py       # 兼容的采集诊断分析层
├── bbox_extractor.py / bbox_probe.py
├── assets_downloader.py
├── calibration.py
├── cluster_detector.py             # HDBSCAN、二级合并、PCA
├── cluster_projection.py
├── projection_pipeline.py
├── vision_candidate.py
├── ai_input_builder.py
├── temporal_cluster_dedup.py
├── vision_verifier.py
├── vision_quality_gate.py
├── vision_cache.py
├── vision_category_config.py       # Vision 类别白名单加载
├── vision_classifier.py            # 保留的分类接口定义
├── vision_pipeline.py
├── rule_engine.py
├── vehicle_dimension_config.py
├── cleanup_scene.py
├── check_qwen_connection.py
├── config.yaml
├── config/
│   ├── vehicle_dimensions.yaml
│   └── vision_categories.yaml
├── .env.example
└── requirements.txt
```

---

## 15. 常见问题

### Virtual Environment not found

按 Windows 第 4.1 节或 Ubuntu 第 5.1 节创建根目录 `.venv`。不需要 activate。

### 无法连接 127.0.0.1:9222

- 完全关闭普通 Edge 后用调试参数重启。
- 确认端口与 `browser.cdp_url` 一致。
- 确认没有进程占用相同 user-data-dir。
- 在浏览器访问 `http://127.0.0.1:9222/json/version`。

### 连接错误标签页

关闭 Edge 中所有其他标签页，只保留当前 QP 质检 Scene 页面，然后重新运行。不要依赖
`browser.page_title_hint` 兜底，因为匹配失败仍会选择第一个标签页。

### Frame 2 切帧失败

检查 `bottom_region_ratio` 和 fallback 坐标。运行中不要改变浏览器缩放或窗口布局。

### PCD/JPG 超时

调大 `frame_ready.timeout_seconds` 或 `render_settle_ms`。单帧超时不会中断。

### Projection 无输出

确认采样帧有 PCD、JPG、成功读取的 BBox、匹配的 Calibration 和有效 Camera 外参。低质量投影不调用
Vision。

### Vision 很慢

查看 `rule_summary.json -> vision_test_stats` 中的请求数、缓存命中、平均 API 秒数、请求墙钟时间
和并发节省。遇到 429 时把 `vision_workers` 降为 1。

### 报告仍是上一 Scene

新 Run 完成前固定报告仍是旧结果。等待 `[9/9] Inspection Finished`，或运行前 Cleanup。运行中
不要 Cleanup。

---

## 16. 安全与提交

`.gitignore` 排除：

- `.env`
- `.venv/`
- `__pycache__/` 和 `*.pyc`
- `outputs/` 运行结果
- `assets/` PCD/JPG
- `ai_inputs/` 图片、Manifest、Cache
- `*.log`

提交前：

```bash
git status --short
git diff --check
```

不要提交真实 API Key、Cookie、PCD/JPG、客户 Calibration 或 Scene 报告。

---

## 17. 最终 Run 检查表

- [ ] `.venv` 已创建，依赖完整
- [ ] `.env` 已填写 `DASHSCOPE_API_KEY`
- [ ] API 检查返回 `status=verified`
- [ ] API Base 与 Key 的 Workspace/地域一致
- [ ] Edge 已用 Remote Debugging 启动
- [ ] QP 已登录，正确 Scene 质检页面已打开
- [ ] Edge 中其他标签页已全部关闭，只保留当前质检 Scene 页面
- [ ] Scene Number、Frame Count 正确
- [ ] Calibration YAML 与当前 PCD/JPG 匹配
- [ ] 运行中不移动窗口、不改变缩放、不关闭质检页
- [ ] 完成后核对两个最终报告的 Scene ID
