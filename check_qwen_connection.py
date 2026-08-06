"""Make one minimal Qwen-VL request using config.yaml and DASHSCOPE_API_KEY."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import yaml
from PIL import Image

from runtime_env import load_runtime_environment
from vision_verifier import QwenVLVerifier


def main() -> int:
    config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8")) or {}
    vision_config = config.get("ai_vision", {}) or {}
    api_key_env = str(vision_config.get("api_key_env", "DASHSCOPE_API_KEY"))
    env_status = load_runtime_environment(Path.cwd(), api_key_env)
    if not env_status.api_key_available:
        print("status=error")
        print(f"error={env_status.error_message}")
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        image_path = Path(tmp) / "connection_test.jpg"
        Image.new("RGB", (64, 64), color=(128, 128, 128)).save(image_path)
        verifier = QwenVLVerifier({**vision_config, "max_retries": 0})
        result = verifier.verify_cluster(
            {
                "scene_id": "connection_test",
                "frame_index": 0,
                "cluster_id": "connection_test",
                "point_count": 1,
                "center": [0.0, 0.0, 1.0],
                "size": [0.1, 0.1, 0.1],
                "distance": 1.0,
                "pca_features": {
                    "linearity": 0.0,
                    "planarity": 0.0,
                    "flatness": 0.0,
                },
            },
            [image_path],
        )

    print(f"status={result.status}")
    print(f"model={vision_config.get('model_name', '')}")
    print(f"api_base={vision_config.get('api_base', '')}")
    if result.status == "verified":
        print("Qwen-VL connection verified.")
        return 0
    print(f"error={result.error}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
