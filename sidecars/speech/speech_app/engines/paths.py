"""Sens-owned data/model paths for Hearing."""

from __future__ import annotations

import os
from pathlib import Path

from ..models import ModelPreset
from ..portable import portable_data_dir, portable_models_dir, sens_data_root
from ..settings import default_data_dir


def models_root() -> Path:
    configured = portable_models_dir()
    if configured is not None:
        return configured
    return sens_data_root() / "models" / "speech"


def model_dir(preset: ModelPreset) -> Path:
    return models_root() / preset.key


def installed_marker(preset: ModelPreset) -> Path:
    return model_dir(preset) / "INSTALLED.json"


def whisper_model_dir(preset: ModelPreset) -> Path:
    return model_dir(preset)


def whisper_installed_marker(preset: ModelPreset) -> Path:
    return installed_marker(preset)


def pack_download_dir() -> Path:
    return models_root() / ".downloads"


def data_dir() -> Path:
    portable = portable_data_dir()
    return portable if portable is not None else default_data_dir()
