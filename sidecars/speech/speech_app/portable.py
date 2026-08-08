from __future__ import annotations

import os
from pathlib import Path


def sens_data_root() -> Path:
    """Mutable Sens data root, independent from the bundled code directory."""
    configured = os.environ.get("SENS_DATA_ROOT")
    if configured:
        return Path(configured)
    local = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    if local:
        return Path(local) / "Sens" / "Sens" / "data"
    return Path.home() / ".sens"


def portable_data_dir() -> Path | None:
    value = os.environ.get("SPEECH_DATA_DIR")
    if not value:
        return None
    return Path(value)


def portable_models_dir() -> Path | None:
    value = os.environ.get("SPEECH_MODELS_DIR")
    if not value:
        return None
    return Path(value)


def build_portable_env(root: Path) -> dict[str, str]:
    data_root = portable_data_dir() or sens_data_root() / "speech"
    model_root = portable_models_dir() or sens_data_root() / "models" / "speech"
    hf_home = model_root / "huggingface"
    return {
        "SPEECH_HOME": str(root),
        "SPEECH_DATA_DIR": str(data_root),
        "SPEECH_MODELS_DIR": str(model_root),
        "HF_HOME": str(hf_home),
        "HF_HUB_CACHE": str(hf_home / "hub"),
        "TRANSFORMERS_CACHE": str(hf_home / "transformers"),
        "TORCH_HOME": str(model_root / "torch"),
        "XDG_CACHE_HOME": str(sens_data_root() / "cache" / "speech"),
    }
