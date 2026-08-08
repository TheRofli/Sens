from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from .portable import portable_data_dir, sens_data_root


APP_NAME = "Sens"
_MIGRATED_FILES = ("settings.json", "history.jsonl")


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
    # Gate whisper segments behind faster-whisper's Silero VAD. Tuned for
    # clean dictation speech; file transcription turns it off because the
    # VAD rejects vocoded/sung vocals on top of music.
    vad_filter: bool = True
    # Video still extraction limits (used by agent file transcription).
    # The model may ask for at / frames / every, but never more than
    # max_frames stills, never larger than frame_size, and `every` falls
    # back to default_every when the request does not specify an interval.
    max_frames: int = 12
    frame_size: int = 640
    default_every: float = 0.0
    history_limit: int = 100
    # Quality / generation params (applied where supported by each engine).
    beam_size: int = 5
    temperature: float = 0.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    postprocess_text: bool = True
    # Remote (OpenRouter-compatible) transcription. The API key lives in this
    # settings file only and must never be logged or passed through argv.
    remote_api_key: str = ""
    remote_base_url: str = "https://openrouter.ai/api/v1"
    remote_model_id: str = "openai/gpt-4o-transcribe"


def default_data_dir() -> Path:
    portable = portable_data_dir()
    if portable is not None:
        return portable

    return sens_data_root() / "speech"


def legacy_data_dir() -> Path:
    configured = os.environ.get("SENS_LEGACY_SPEECH_ROOT")
    return Path(configured or r"D:\Speech") / "data"


def migrate_legacy_user_data(destination: Path | None = None) -> list[str]:
    """Import compatible user files once, without touching runtime secrets.

    The allowlist intentionally excludes API ports, process locks, tokens, and
    logs. Each file is promoted atomically and only when the Sens-owned target
    does not already exist.
    """
    target_root = destination or default_data_dir()
    source_root = legacy_data_dir()
    migrated: list[str] = []
    if not source_root.is_dir() or source_root.resolve() == target_root.resolve():
        return migrated
    target_root.mkdir(parents=True, exist_ok=True)
    for name in _MIGRATED_FILES:
        source = source_root / name
        target = target_root / name
        if not source.is_file() or target.exists():
            continue
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{name}.", suffix=".migrating", dir=target_root
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            shutil.copyfile(source, temporary)
            os.replace(temporary, target)
            migrated.append(name)
        finally:
            temporary.unlink(missing_ok=True)
    return migrated


class SettingsStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_data_dir() / "settings.json"

    def load(self) -> AppSettings:
        if self.path == default_data_dir() / "settings.json":
            migrate_legacy_user_data(self.path.parent)
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
