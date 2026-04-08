from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def _get_default_device() -> str:
    return os.getenv("DEFAULT_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_path: str
    images_dir: str
    output_dir: str
    default_local_model: str
    default_eval_model: str
    default_device: str
    hf_endpoint: str | None
    zhipuai_api_key: str | None
    chatanywhere_api_key: str | None
    base_url: str | None


SETTINGS = Settings(
    project_root=PROJECT_ROOT,
    data_path=os.getenv("DATA_PATH", "data/annotations/annotation.jsonl"),
    images_dir=os.getenv("IMAGES_DIR", "data/images"),
    output_dir=os.getenv("OUTPUT_DIR", "outputs"),
    default_local_model=os.getenv("DEFAULT_LOCAL_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct"),
    default_eval_model=os.getenv("DEFAULT_EVAL_MODEL", "glm-4.6v"),
    default_device=_get_default_device(),
    hf_endpoint=os.getenv("HF_ENDPOINT"),
    zhipuai_api_key=os.getenv("ZHIPUAI_API_KEY"),
    chatanywhere_api_key=os.getenv("CHATANYWHERE_API_KEY"),
    base_url=os.getenv("BASE_URL"),
)

if SETTINGS.hf_endpoint and "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = SETTINGS.hf_endpoint

if SETTINGS.chatanywhere_api_key and "CHATANYWHERE_API_KEY" not in os.environ:
    os.environ["CHATANYWHERE_API_KEY"] = SETTINGS.chatanywhere_api_key


def require_zhipuai_api_key(api_key: str | None = None) -> str:
    value = api_key or SETTINGS.zhipuai_api_key
    if not value:
        raise ValueError(
            "ZHIPUAI_API_KEY is required for GLM-based workflows. Set it in the environment or in .env."
        )
    return value
