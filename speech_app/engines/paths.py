"""Shared helpers for locating model files on disk.

Parakeet weights live in the Hugging Face hub cache (``models/huggingface``);
Whisper (CTranslate2) models live in their own folder tree
(``models/whisper/<key>``) because they are converted locally at install time.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..models import ModelPreset
from ..portable import portable_data_dir
from ..settings import default_data_dir


def models_root() -> Path:
    """Root directory for all model caches.

    Mirrors the layout used by the launcher scripts: when running in the
    portable environment (``SPEECH_HOME``/``HF_HOME`` set) we place the whisper
    cache next to the Hugging Face cache under ``models/``; otherwise we fall
    back to the data directory.
    """
    speech_home = os.environ.get("SPEECH_HOME")
    if speech_home:
        return Path(speech_home) / "models"
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).parent
    return default_data_dir() / "models"


def huggingface_home() -> Path:
    """Hugging Face cache root used for parakeet weights."""
    fallback = models_root() / "huggingface"
    return Path(os.environ.get("HF_HOME", str(fallback)))


def whisper_model_dir(preset: ModelPreset) -> Path:
    """Directory holding the converted CTranslate2 weights for a whisper preset."""
    return models_root() / "whisper" / preset.key


def whisper_installed_marker(preset: ModelPreset) -> Path:
    """Marker file written after a successful CT2 conversion."""
    return whisper_model_dir(preset) / "INSTALLED.json"


def data_dir() -> Path:
    """Runtime data directory (history, runtime state, api.port)."""
    portable = portable_data_dir()
    if portable is not None:
        return portable
    return default_data_dir()
