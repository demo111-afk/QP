"""QP Copilot V1.0 Tkinter entry point.

The UI validates Scene, Frame Count, and pasted Calibration, loads the project .env,
starts main.py once, displays frame/stage logs, and exposes the two final Rule reports.
Business logic remains in main.py and its existing modules.
"""

from __future__ import annotations

import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk

BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "outputs" / "reports"
MAIN_SCRIPT = BASE_DIR / "main.py"
CLEANUP_SCRIPT = BASE_DIR / "cleanup_scene.py"

# 子进程不新开控制台窗口（Windows 下 pythonw.exe 启动本 App 时，Popen 默认会给
# 控制台子系统的子进程新分配一个黑窗口，这里禁用掉）
_CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

PROGRESS_RE = re.compile(r"\[进度\]\s*第\s*(\d+)/(\d+)\s*帧")
STAGE_RE = re.compile(r"\[(\d+)/(\d+)\]\s*(.+)")

# (显示名, 文件名) —— 都在 outputs/reports/ 下，跟 main.py 生成的文件名保持一致
REPORT_FILES = [
    "rule_report.csv",
    "rule_summary.json",
]

REQUIRED_RUNTIME_IMPORTS = (
    "yaml",
    "playwright",
    "numpy",
    "sklearn",
    "PIL",
    "openai",
    "dotenv",
)


@dataclass(frozen=True)
class ProjectPython:
    environment_dir: Path
    backend_executable: Path
    ui_executable: Path


def _find_project_python() -> ProjectPython | None:
    """Locate the project-owned virtual environment without activating it."""
    for directory_name in (".venv", "venv"):
        environment_dir = BASE_DIR / directory_name
        if sys.platform == "win32":
            scripts_dir = environment_dir / "Scripts"
            backend = scripts_dir / "python.exe"
            ui = scripts_dir / "pythonw.exe"
            if backend.is_file():
                return ProjectPython(
                    environment_dir=environment_dir,
                    backend_executable=backend,
                    ui_executable=ui if ui.is_file() else backend,
                )
        else:
            executable = environment_dir / "bin" / "python"
            if executable.is_file():
                return ProjectPython(environment_dir, executable, executable)
    return None


def _virtual_environment_instructions() -> str:
    if sys.platform == "win32":
        return (
            "Open PowerShell in the project folder once and run:\n\n"
            "py -m venv .venv\n"
            ".venv\\Scripts\\python.exe -m pip install -r requirements.txt\n"
            ".venv\\Scripts\\python.exe -m playwright install chromium"
        )
    return (
        "Open Terminal in the project folder once and run:\n\n"
        "python3 -m venv .venv\n"
        ".venv/bin/python -m pip install -r requirements.txt\n"
        ".venv/bin/python -m playwright install chromium"
    )


def _show_bootstrap_error(title: str, message: str) -> None:
    root = tk.Tk()
    root.withdraw()
    try:
        messagebox.showerror(title, message, parent=root)
    finally:
        root.destroy()


def _validate_project_python(project_python: ProjectPython) -> str:
    imports = "; ".join(f"import {name}" for name in REQUIRED_RUNTIME_IMPORTS)
    try:
        result = subprocess.run(
            [str(project_python.backend_executable), "-c", imports],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            creationflags=_CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)
    if result.returncode == 0:
        return ""
    return (result.stderr or result.stdout or f"exit code {result.returncode}").strip()


def _bootstrap_project_environment() -> ProjectPython | None:
    project_python = _find_project_python()
    if project_python is None:
        _show_bootstrap_error(
            "Virtual Environment not found",
            "Virtual Environment not found.\n\n" + _virtual_environment_instructions(),
        )
        return None

    validation_error = _validate_project_python(project_python)
    if validation_error:
        _show_bootstrap_error(
            "Virtual Environment incomplete",
            "The project virtual environment is missing required packages.\n\n"
            + _virtual_environment_instructions()
            + f"\n\nDetails:\n{validation_error}",
        )
        return None

    try:
        running_prefix = Path(sys.prefix).resolve()
        project_prefix = project_python.environment_dir.resolve()
    except OSError:
        running_prefix = Path(sys.prefix)
        project_prefix = project_python.environment_dir
    if running_prefix != project_prefix:
        try:
            subprocess.Popen(
                [str(project_python.ui_executable), str(Path(__file__).resolve())],
                cwd=str(BASE_DIR),
                creationflags=_CREATE_NO_WINDOW,
                env=_build_child_environment(),
            )
        except OSError as exc:
            _show_bootstrap_error(
                "UI launch failed",
                f"Failed to start UI with project Python:\n{exc}",
            )
        return None
    return project_python


def _load_ai_vision_settings() -> tuple[bool, str]:
    try:
        import yaml

        with open(BASE_DIR / "config.yaml", "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        vision = config.get("ai_vision", {}) or {}
        return bool(vision.get("enabled", False)), str(
            vision.get("api_key_env", "DASHSCOPE_API_KEY")
        )
    except Exception:
        return False, "DASHSCOPE_API_KEY"


def _load_ui_runtime_environment():
    from runtime_env import load_runtime_environment

    _enabled, api_key_env = _load_ai_vision_settings()
    return load_runtime_environment(BASE_DIR, api_key_env)


def _build_child_environment() -> dict[str, str]:
    # Build this at process launch time, after .env has been loaded.
    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"
    return child_env


def _open_path(path: Path) -> None:
    """用系统默认程序打开文件/文件夹（跨平台：Windows / macOS / Linux）。"""
    if not path.exists():
        messagebox.showerror("Open failed", f"文件不存在:\n{path}")
        return
    try:
        if sys.platform == "win32":
            os.startfile(str(path))  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=True)
        else:
            subprocess.run(["xdg-open", str(path)], check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        messagebox.showerror("Open failed", f"无法打开:\n{path}\n\n{exc}")


class DeleteConfirmDialog(tk.Toplevel):
    """Cleanup 二次确认弹窗：必须手动输入 DELETE 才会返回 confirmed=True。"""

    def __init__(self, parent: tk.Tk, scene_id: str):
        super().__init__(parent)
        self.title("Confirm Delete")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        self.confirmed = False

        tk.Label(
            self,
            text=f"Delete ALL files related to this Scene?\n\nScene Number: {scene_id}",
            justify="center",
            padx=24,
            pady=12,
        ).pack()
        tk.Label(self, text='Type "DELETE" to confirm:').pack(pady=(0, 4))

        self.entry = tk.Entry(self, justify="center")
        self.entry.pack(padx=24, pady=4, fill="x")
        self.entry.focus_set()

        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=12)
        tk.Button(btn_frame, text="Cancel", width=10, command=self._cancel).pack(side="left", padx=6)
        tk.Button(btn_frame, text="Confirm", width=10, command=self._confirm).pack(side="left", padx=6)

        self.bind("<Return>", lambda _e: self._confirm())
        self.bind("<Escape>", lambda _e: self._cancel())
        self.protocol("WM_DELETE_WINDOW", self._cancel)

    def _confirm(self) -> None:
        if self.entry.get().strip() == "DELETE":
            self.confirmed = True
            self.destroy()
        else:
            messagebox.showwarning(
                "Not confirmed", 'Please type "DELETE" exactly to proceed.', parent=self
            )

    def _cancel(self) -> None:
        self.confirmed = False
        self.destroy()


class QPCopilotApp:
    def __init__(self, root: tk.Tk, project_python: ProjectPython):
        self.root = root
        self.project_python = project_python
        self.root.title("QP Copilot")
        self.root.resizable(False, False)

        self.event_queue: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self.inspection_running = False
        self.cleanup_running = False
        self.inspection_thread: threading.Thread | None = None
        self.cleanup_thread: threading.Thread | None = None
        self.current_scene_id = ""
        self.report_rows: dict[str, tuple[tk.Label, tk.Button]] = {}

        self._build_ui()
        self.root.after(100, self._poll_queue)

    # ---------------------------------------------------------- UI 构建 ----
    def _build_ui(self) -> None:
        pad = {"padx": 16, "pady": 6}

        tk.Label(self.root, text="QP Copilot", font=("Segoe UI", 18, "bold")).pack(pady=(16, 8))

        form = tk.Frame(self.root)
        form.pack(fill="x", **pad)

        tk.Label(form, text="Scene Number:").grid(row=0, column=0, sticky="w")
        self.scene_entry = tk.Entry(form, width=30)
        self.scene_entry.grid(row=0, column=1, sticky="w", padx=(8, 0))

        tk.Label(form, text="Frame Count:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.frame_count_entry = tk.Entry(form, width=10)
        self.frame_count_entry.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

        tk.Label(form, text="Calibration YAML:").grid(row=2, column=0, sticky="nw", pady=(8, 0))
        calibration_frame = tk.Frame(form)
        calibration_frame.grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(8, 0))
        self.calibration_text = tk.Text(calibration_frame, width=52, height=10, wrap="none")
        calibration_scroll_y = ttk.Scrollbar(
            calibration_frame, orient="vertical", command=self.calibration_text.yview
        )
        calibration_scroll_x = ttk.Scrollbar(
            calibration_frame, orient="horizontal", command=self.calibration_text.xview
        )
        self.calibration_text.configure(
            yscrollcommand=calibration_scroll_y.set,
            xscrollcommand=calibration_scroll_x.set,
        )
        self.calibration_text.grid(row=0, column=0, sticky="nsew")
        calibration_scroll_y.grid(row=0, column=1, sticky="ns")
        calibration_scroll_x.grid(row=1, column=0, sticky="ew")

        action_frame = tk.Frame(self.root)
        action_frame.pack(pady=10)
        self.start_btn = tk.Button(
            action_frame, text="Start Inspection", width=20, command=self._on_start_inspection
        )
        self.start_btn.pack(side="left", padx=4)
        self.reset_btn = tk.Button(
            action_frame, text="Reset", width=10, command=self._on_reset
        )
        self.reset_btn.pack(side="left", padx=4)

        progress_frame = tk.Frame(self.root)
        progress_frame.pack(fill="x", **pad)
        tk.Label(progress_frame, text="Progress").pack(anchor="w")
        self.progress_bar = ttk.Progressbar(progress_frame, length=320, mode="determinate")
        self.progress_bar.pack(fill="x", pady=(4, 2))
        self.progress_label = tk.Label(progress_frame, text="0 / 0")
        self.progress_label.pack(anchor="w")

        self.status_label = tk.Label(self.root, text="Status: Ready", fg="#555555")
        self.status_label.pack(pady=(4, 6))

        log_frame = tk.Frame(self.root)
        log_frame.pack(fill="x", **pad)
        tk.Label(log_frame, text="Run Log").pack(anchor="w")
        self.log_text = tk.Text(log_frame, width=67, height=8, wrap="word", state="disabled")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

        self.report_frame = tk.Frame(self.root)
        self.report_frame.pack(fill="x", **pad)
        for filename in REPORT_FILES:
            row = tk.Frame(self.report_frame)
            row.pack(fill="x", pady=2)
            status_lbl = tk.Label(row, text=f"  {filename}", anchor="w", width=22)
            status_lbl.pack(side="left")
            open_btn = tk.Button(
                row,
                text="Open",
                width=8,
                state="disabled",
                command=lambda f=filename: _open_path(REPORTS_DIR / f),
            )
            open_btn.pack(side="left")
            self.report_rows[filename] = (status_lbl, open_btn)

        tk.Button(
            self.root,
            text="Open Report Folder",
            width=24,
            command=lambda: _open_path(REPORTS_DIR),
        ).pack(pady=(4, 12))

        ttk.Separator(self.root, orient="horizontal").pack(fill="x", padx=16)

        tk.Label(self.root, text="Cleanup Scene", font=("Segoe UI", 12, "bold")).pack(pady=(12, 6))

        cleanup_form = tk.Frame(self.root)
        cleanup_form.pack(fill="x", **pad)
        tk.Label(cleanup_form, text="Scene Number:").grid(row=0, column=0, sticky="w")
        self.cleanup_scene_entry = tk.Entry(cleanup_form, width=30)
        self.cleanup_scene_entry.grid(row=0, column=1, sticky="w", padx=(8, 0))

        self.cleanup_btn = tk.Button(
            self.root, text="Delete Scene Files", width=24, command=self._on_delete_scene_files
        )
        self.cleanup_btn.pack(pady=(6, 16))

    # ---------------------------------------------------- Start Inspection ----
    def _on_start_inspection(self) -> None:
        if self.inspection_running:
            return

        from calibration import CalibrationError, CalibrationLoader

        scene_id = self.scene_entry.get().strip()
        frame_count_raw = self.frame_count_entry.get().strip()
        calibration_yaml = self.calibration_text.get("1.0", "end-1c").strip()
        if not scene_id:
            messagebox.showerror("Scene Number required", "请输入 Scene Number。")
            self.scene_entry.focus_set()
            return
        if not frame_count_raw:
            messagebox.showerror("Frame Number required", "请输入 Frame Number。")
            self.frame_count_entry.focus_set()
            return
        try:
            frame_count = int(frame_count_raw)
        except ValueError:
            messagebox.showerror("Invalid Frame Number", "Frame Number 必须是整数。")
            self.frame_count_entry.focus_set()
            return
        if frame_count <= 0:
            messagebox.showerror("Invalid Frame Number", "Frame Number 必须大于 0。")
            self.frame_count_entry.focus_set()
            return
        if not calibration_yaml:
            messagebox.showerror("Calibration required", "请粘贴 Calibration YAML。")
            self.calibration_text.focus_set()
            return
        try:
            calibration = CalibrationLoader.loads(calibration_yaml)
        except (CalibrationError, TypeError, ValueError) as exc:
            messagebox.showerror("Invalid Calibration YAML", str(exc))
            self.calibration_text.focus_set()
            return
        if not calibration.get_all_cameras():
            messagebox.showerror("Invalid Calibration YAML", "Calibration YAML 中没有 Camera。")
            self.calibration_text.focus_set()
            return

        vision_enabled, _api_key_env = _load_ai_vision_settings()
        env_status = _load_ui_runtime_environment()
        if vision_enabled and not env_status.api_key_available:
            messagebox.showerror(
                "Vision API Key required",
                f"{env_status.error_message}。\n\n"
                "请在项目根目录根据 .env.example 创建 .env，然后重新点击 Start。",
            )
            return

        for filename, (status_lbl, open_btn) in self.report_rows.items():
            status_lbl.config(text=f"  {filename}", fg="black")
            open_btn.config(state="disabled")

        self.inspection_running = True
        self.current_scene_id = scene_id
        self.start_btn.config(state="disabled")
        self.reset_btn.config(state="disabled")
        self.calibration_text.config(state="disabled")
        self.progress_bar.config(value=0, maximum=100)
        self.progress_label.config(text="0 / 0")
        self.status_label.config(text="Status: Running...", fg="#008800")
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")

        self.inspection_thread = threading.Thread(
            target=self._run_main_process,
            args=(scene_id, frame_count_raw, calibration_yaml),
            daemon=True,
        )
        self.inspection_thread.start()

    def _run_main_process(
        self, scene_id: str, frame_count_raw: str, calibration_yaml: str
    ) -> None:
        # main.py 依次用 input() 问：scene_id -> frame_count -> 确认回车（无内容），
        # 三行喂给 stdin，跟人在终端里敲的效果完全一样。
        stdin_payload = f"{scene_id}\n{frame_count_raw}\n\n"
        calibration_path = None
        process = None
        try:
            fd, calibration_path = tempfile.mkstemp(prefix="qp_calibration_", suffix=".yaml")
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as calibration_file:
                calibration_file.write(calibration_yaml)

            process = subprocess.Popen(
                [
                    str(self.project_python.backend_executable),
                    "-u",
                    str(MAIN_SCRIPT),
                    "--calibration",
                    calibration_path,
                ],
                cwd=str(BASE_DIR),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=_CREATE_NO_WINDOW,
                env=_build_child_environment(),
            )
        except OSError as exc:
            if calibration_path:
                Path(calibration_path).unlink(missing_ok=True)
            self.event_queue.put(("error", f"启动 main.py 失败: {exc}"))
            return

        output_tail = []
        try:
            assert process is not None
            assert process.stdin is not None and process.stdout is not None
            process.stdin.write(stdin_payload)
            process.stdin.close()

            for line in process.stdout:
                stripped = line.strip()
                if stripped:
                    output_tail.append(stripped)
                    output_tail = output_tail[-30:]
                    self.event_queue.put(("log", stripped))
                progress_match = PROGRESS_RE.search(line)
                if progress_match:
                    current, total = int(progress_match.group(1)), int(progress_match.group(2))
                    self.event_queue.put(("progress", (current, total)))
                stage_match = STAGE_RE.search(line)
                if stage_match:
                    current, total = int(stage_match.group(1)), int(stage_match.group(2))
                    self.event_queue.put(("stage", (current, total, stage_match.group(3).strip())))

            returncode = process.wait()
            if returncode == 0:
                self.event_queue.put(("done", None))
            else:
                details = "\n".join(output_tail)
                message = f"main.py 退出码 {returncode}"
                if details:
                    message += f"\n\n{details}"
                self.event_queue.put(("error", message))
        finally:
            if calibration_path:
                Path(calibration_path).unlink(missing_ok=True)

    # ------------------------------------------------------------- Cleanup ----
    def _on_delete_scene_files(self) -> None:
        if self.cleanup_running:
            return

        scene_id = self.cleanup_scene_entry.get().strip()
        if not scene_id:
            messagebox.showwarning("Scene Number required", "请输入 Scene Number。")
            return

        dialog = DeleteConfirmDialog(self.root, scene_id)
        self.root.wait_window(dialog)
        if not dialog.confirmed:
            return

        self.cleanup_running = True
        self.cleanup_btn.config(state="disabled")
        self.reset_btn.config(state="disabled")
        self.status_label.config(text="Status: Running...", fg="#008800")

        self.cleanup_thread = threading.Thread(
            target=self._run_cleanup_process, args=(scene_id,), daemon=True
        )
        self.cleanup_thread.start()

    def _run_cleanup_process(self, scene_id: str) -> None:
        # cleanup_scene.py 依次问：scene_id -> 确认输入 DELETE。UI 这边已经弹窗确认过一次，
        # 这里把 "DELETE" 一起喂进去，触发它自己已有的二次确认+删除逻辑。
        stdin_payload = f"{scene_id}\nDELETE\n"
        try:
            result = subprocess.run(
                [str(self.project_python.backend_executable), "-u", str(CLEANUP_SCRIPT)],
                cwd=str(BASE_DIR),
                input=stdin_payload,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=_CREATE_NO_WINDOW,
                env=_build_child_environment(),
            )
        except OSError as exc:
            self.event_queue.put(("cleanup_error", f"启动 cleanup_scene.py 失败: {exc}"))
            return

        if result.returncode == 0:
            self.event_queue.put(("cleanup_done", result.stdout))
        else:
            self.event_queue.put(("cleanup_error", result.stdout or f"退出码 {result.returncode}"))

    # --------------------------------------------------------------- 队列轮询 ----
    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self.event_queue.get_nowait()
                if kind == "progress":
                    current, total = payload
                    self.progress_bar.config(value=current, maximum=max(total, 1))
                    self.progress_label.config(text=f"{current} / {total}")
                elif kind == "stage":
                    current, total, message = payload
                    self.status_label.config(
                        text=f"Status: [{current}/{total}] {message}", fg="#008800"
                    )
                elif kind == "log":
                    self.log_text.config(state="normal")
                    self.log_text.insert("end", str(payload) + "\n")
                    self.log_text.see("end")
                    self.log_text.config(state="disabled")
                elif kind == "done":
                    self.status_label.config(text="Status: Completed.", fg="#008800")
                    self.start_btn.config(state="normal")
                    self.calibration_text.config(state="normal")
                    self.inspection_running = False
                    self.inspection_thread = None
                    self._refresh_reset_button()
                    self._refresh_report_files()
                elif kind == "error":
                    self.status_label.config(text="Status: Error", fg="#cc0000")
                    self.start_btn.config(state="normal")
                    self.calibration_text.config(state="normal")
                    self.inspection_running = False
                    self.inspection_thread = None
                    self._refresh_reset_button()
                    messagebox.showerror("Inspection failed", str(payload))
                elif kind == "cleanup_done":
                    self.status_label.config(text="Status: Ready", fg="#555555")
                    self.cleanup_btn.config(state="normal")
                    self.cleanup_running = False
                    self.cleanup_thread = None
                    self._refresh_reset_button()
                    messagebox.showinfo("Cleanup", "Cleanup completed.")
                elif kind == "cleanup_error":
                    self.status_label.config(text="Status: Ready", fg="#555555")
                    self.cleanup_btn.config(state="normal")
                    self.cleanup_running = False
                    self.cleanup_thread = None
                    self._refresh_reset_button()
                    messagebox.showerror("Cleanup failed", str(payload))
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self._poll_queue)

    def _refresh_report_files(self) -> None:
        for filename, (status_lbl, open_btn) in self.report_rows.items():
            path = REPORTS_DIR / filename
            if path.is_file():
                status_lbl.config(text=f"✓ {filename}", fg="#008800")
                open_btn.config(state="normal")
            else:
                status_lbl.config(text=f"  {filename} (not found)", fg="#cc0000")
                open_btn.config(state="disabled")

    def _refresh_reset_button(self) -> None:
        state = "disabled" if self.inspection_running or self.cleanup_running else "normal"
        self.reset_btn.config(state=state)

    def _on_reset(self) -> None:
        if self.inspection_running or self.cleanup_running:
            messagebox.showwarning(
                "Reset unavailable",
                "Wait for the current Inspection or Cleanup task to finish.",
            )
            return

        self.scene_entry.delete(0, "end")
        self.frame_count_entry.delete(0, "end")
        self.cleanup_scene_entry.delete(0, "end")

        self.calibration_text.config(state="normal")
        self.calibration_text.delete("1.0", "end")

        self.progress_bar.config(value=0, maximum=100)
        self.progress_label.config(text="0 / 0")
        self.status_label.config(text="Status: Ready", fg="#555555")

        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")

        for filename, (status_lbl, open_btn) in self.report_rows.items():
            status_lbl.config(text=f"  {filename}", fg="black")
            open_btn.config(state="disabled")

        while True:
            try:
                self.event_queue.get_nowait()
            except queue.Empty:
                break

        self.current_scene_id = ""
        self.inspection_thread = None
        self.cleanup_thread = None
        self.start_btn.config(state="normal")
        self.cleanup_btn.config(state="normal")
        self._refresh_reset_button()
        self.scene_entry.focus_set()


def main() -> None:
    project_python = _bootstrap_project_environment()
    if project_python is None:
        return
    root = tk.Tk()
    QPCopilotApp(root, project_python)
    root.mainloop()


if __name__ == "__main__":
    main()
