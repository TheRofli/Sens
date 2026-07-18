from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from .portable import portable_data_dir


APP_NAME = "Speech"


@dataclass(slots=True)
class AppSettings:
    # Model selection. `model` is the preset key (see speech_app/models.py);
    # `model_id` is kept for back-compat and as the resolved HF repo id for the
    # parakeet backend.
    model: str = "parakeet"
    model_id: str = "nvidia/parakeet-tdt-0.6b-v3"
    backend: str = "auto"
    device: str = "cpu"
    hotkey: str = "ctrl+win"
    engine_enabled: bool = True
    copy_to_clipboard: bool = True
    paste_to_active_input: bool = True
    suppress_hotkey: bool = False
    preload_model: bool = True
    sample_rate: int = 16000
    vad_sensitivity: float = 0.02
    history_limit: int = 100
    # Quality / generation params (applied where supported by each engine).
    beam_size: int = 5
    temperature: float = 0.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    postprocess_text: bool = True


def default_data_dir() -> Path:
    portable = portable_data_dir()
    if portable is not None:
        return portable

    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / APP_NAME
    return Path.home() / f".{APP_NAME.lower()}"


class SettingsStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_data_dir() / "settings.json"

    def load(self) -> AppSettings:
        if not self.path.exists():
            return AppSettings()

        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return AppSettings()

        allowed = {field.name for field in fields(AppSettings)}
        known: dict[str, Any] = {
            key: value for key, value in payload.items() if key in allowed
        }
        return AppSettings(**known)

    def save(self, settings: AppSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(asdict(settings), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
