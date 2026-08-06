"""Project-level runtime environment loading for UI and CLI entry points."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class RuntimeEnvironmentStatus:
    env_path: Path
    env_file_exists: bool
    api_key_env: str
    api_key_available: bool

    @property
    def error_message(self) -> str:
        if not self.env_file_exists:
            return f".env 不存在: {self.env_path}"
        if not self.api_key_available:
            return f".env 已读取，但 {self.api_key_env} 未设置或为空"
        return ""


def load_runtime_environment(
    base_dir: str | Path,
    api_key_env: str = "DASHSCOPE_API_KEY",
) -> RuntimeEnvironmentStatus:
    """Load `<base_dir>/.env` without overriding an existing process variable."""
    env_path = Path(base_dir).resolve() / ".env"
    exists = env_path.is_file()
    if exists:
        load_dotenv(dotenv_path=env_path, override=False)
    return RuntimeEnvironmentStatus(
        env_path=env_path,
        env_file_exists=exists,
        api_key_env=api_key_env,
        api_key_available=bool(os.getenv(api_key_env, "").strip()),
    )
