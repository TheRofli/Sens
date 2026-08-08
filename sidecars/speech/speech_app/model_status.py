from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import ModelPreset


@dataclass(frozen=True, slots=True)
class ModelStatus:
    installed: bool
    snapshot: str
    path: Path | None
    size_mb: float

    @property
    def label(self) -> str:
        if not self.installed:
            return "Not installed"
        return f"Installed - {self.size_label}"

    @property
    def size_label(self) -> str:
        if self.size_mb >= 1024:
            return f"{self.size_mb / 1024:.2f} GB"
        return f"{self.size_mb:.1f} MB"


def _empty_status() -> ModelStatus:
    return ModelStatus(False, "", None, 0.0)


def find_model_status(hf_home: Path, model_id: str) -> ModelStatus:
    cache_name = "models--" + model_id.replace("/", "--")
    snapshots_dir = hf_home / "hub" / cache_name / "snapshots"
    if not snapshots_dir.exists():
        return _empty_status()

    snapshots = sorted(
        [path for path in snapshots_dir.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not snapshots:
        return _empty_status()

    snapshot = snapshots[0]
    # A snapshot without weight files is an incomplete download (metadata only);
    # reporting it as "installed" misleads the UI and the preload path. Pick the
    # most recent snapshot that actually has weights.
    snapshot = next(
        (s for s in snapshots if _has_model_weights(s)), None
    )
    if snapshot is None:
        return _empty_status()

    size = sum(path.stat().st_size for path in snapshot.rglob("*") if path.is_file())
    return ModelStatus(
        installed=True,
        snapshot=snapshot.name,
        path=snapshot,
        size_mb=round(size / (1024 * 1024), 3),
    )


# File-name suffixes that count as model weights across the ASR models Speech
# supports (Parakeet TDT uses .safetensors/.bin; Whisper CT2 uses model.bin).
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".gguf", ".onnx")


def _has_model_weights(snapshot_dir: Path) -> bool:
    """True if the snapshot contains at least one weight file."""
    try:
        for path in snapshot_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in _WEIGHT_SUFFIXES:
                return True
    except OSError:
        return False
    return False


def find_whisper_model_status(preset: "ModelPreset") -> ModelStatus:
    """Installation status for a converted CTranslate2 Whisper model.

    The model directory must contain at least one ``model.bin`` and a marker
    file written by the install step. Size is the on-disk size of the model
    directory.
    """
    # Imported lazily to avoid a circular import at module load time.
    from .engines.paths import whisper_installed_marker, whisper_model_dir

    model_dir = whisper_model_dir(preset)
    marker = whisper_installed_marker(preset)
    if not model_dir.exists() or not marker.exists():
        return _empty_status()

    # Require a weights file so a partially-converted dir does not count.
    has_weights = any(
        (model_dir / name).is_file()
        for name in ("model.bin", "model.int8.bin")
    )
    if not has_weights:
        return _empty_status()

    size = sum(path.stat().st_size for path in model_dir.rglob("*") if path.is_file())
    return ModelStatus(
        installed=True,
        snapshot=preset.key,
        path=model_dir,
        size_mb=round(size / (1024 * 1024), 3),
    )


def find_gigaam_model_status(preset: "ModelPreset") -> ModelStatus:
    """Installation status for a locally patched GigaAM v3 model.

    The model directory must contain ``pytorch_model.bin`` (weights) plus the
    patched remote-code module. Size is the on-disk size of the directory.
    """
    # Imported lazily to avoid a circular import at module load time.
    from .engines.paths import gigaam_model_dir

    model_dir = gigaam_model_dir(preset)
    if not model_dir.exists() or not (model_dir / "pytorch_model.bin").is_file():
        return _empty_status()

    size = sum(path.stat().st_size for path in model_dir.rglob("*") if path.is_file())
    return ModelStatus(
        installed=True,
        snapshot=preset.key,
        path=model_dir,
        size_mb=round(size / (1024 * 1024), 3),
    )


def find_model_status_for_preset(preset: "ModelPreset", hf_home: Path) -> ModelStatus:
    """Dispatch installation-status lookup by preset engine family."""
    if preset.engine == "whisper":
        return find_whisper_model_status(preset)
    if preset.engine == "gigaam":
        return find_gigaam_model_status(preset)
    return find_model_status(hf_home, preset.model_id)
