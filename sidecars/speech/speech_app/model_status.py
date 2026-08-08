from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .models import ModelPreset


@dataclass(frozen=True, slots=True)
class ModelStatus:
    installed: bool
    snapshot: str
    path: Path | None
    size_mb: float

    @property
    def label(self) -> str:
        return f"Installed - {self.size_label}" if self.installed else "Not installed"

    @property
    def size_label(self) -> str:
        if self.size_mb >= 1024:
            return f"{self.size_mb / 1024:.2f} GB"
        return f"{self.size_mb:.1f} MB"


def _empty_status() -> ModelStatus:
    return ModelStatus(False, "", None, 0.0)


def find_model_status_for_preset(preset: ModelPreset) -> ModelStatus:
    if preset.engine == "remote":
        return ModelStatus(True, "remote", None, 0.0)
    from .engines.paths import installed_marker, model_dir

    root = model_dir(preset)
    marker = installed_marker(preset)
    if not marker.is_file() or not all(
        (root / relative).is_file() for relative in preset.required_files
    ):
        return _empty_status()
    try:
        metadata = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_status()
    if metadata.get("preset") != preset.key:
        return _empty_status()
    if preset.revision and metadata.get("revision") != preset.revision:
        return _empty_status()
    if (
        preset.download_sha256
        and metadata.get("archive_sha256") != preset.download_sha256
    ):
        return _empty_status()
    size = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    return ModelStatus(
        installed=True,
        snapshot=preset.revision or preset.download_sha256 or preset.key,
        path=root,
        size_mb=round(size / (1024 * 1024), 3),
    )


def find_whisper_model_status(preset: ModelPreset) -> ModelStatus:
    return find_model_status_for_preset(preset)


def find_gigaam_model_status(preset: ModelPreset) -> ModelStatus:
    return find_model_status_for_preset(preset)
