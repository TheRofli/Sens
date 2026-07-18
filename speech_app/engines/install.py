"""Model installation: parakeet (HF snapshot) and whisper (CT2 conversion).

This module backs the ``speech model install <key>`` CLI. Parakeet is a plain
Hugging Face snapshot download. Whisper presets point at a Transformers
(PyTorch) checkpoint, which faster-whisper cannot consume directly; we convert
it to CTranslate2 INT8 once, locally, and write a marker file so the engine and
status detector treat it as installed.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from ..models import ModelPreset, get_preset
from .paths import (
    huggingface_home,
    whisper_installed_marker,
    whisper_model_dir,
)


def install_parakeet_model(model_id: str) -> int:
    """Download a parakeet model snapshot into the HF cache."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "huggingface_hub is not installed. Run: speech install",
            file=sys.stderr,
        )
        return 1

    print(f"Downloading {model_id} into the configured Hugging Face cache...")
    path = snapshot_download(repo_id=model_id)
    print(f"Parakeet is ready at: {path}")
    return 0


def install_whisper_model(preset: ModelPreset) -> int:
    """Convert a Transformers whisper checkpoint to CTranslate2 INT8 locally."""
    if _ensure_whisper_installed(preset):
        print(f"Whisper model already converted at {whisper_model_dir(preset)}")
        return 0

    try:
        from transformers import AutoTokenizer  # noqa: F401  (import check)
    except ImportError:
        print(
            "transformers is required to convert the Whisper model. "
            "Run: speech install",
            file=sys.stderr,
        )
        return 1

    out_dir = whisper_model_dir(preset)
    tmp_dir = out_dir.with_name(out_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Converting {preset.model_id} to CTranslate2 INT8 (this needs torch and "
        "may take several minutes and several GB of RAM)..."
    )
    rc = _run_ct2_converter(preset.model_id, tmp_dir)
    if rc != 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(
            "Conversion failed. See the output above. Common causes: not enough "
            "RAM, incompatible transformers/ctranslate2 versions, or network "
            "issues downloading the source checkpoint.",
            file=sys.stderr,
        )
        return rc

    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    tmp_dir.rename(out_dir)
    _write_installed_marker(preset)
    print(f"Whisper model is ready at: {out_dir}")
    return 0


def _run_ct2_converter(model_id: str, out_dir: Path) -> int:
    """Convert a Transformers checkpoint to CTranslate2 INT8 in-process.

    We deliberately avoid the ``ct2-transformers-converter`` CLI entry point.
    On a machine with several Python projects, ``shutil.which`` can resolve
    that script from a *different* virtualenv (e.g. another agent's venv),
    whose ctranslate2/transformers versions are incompatible with this one —
    that surfaces as cryptic ``NameError``s from inside ctranslate2. Running
    the conversion inside the current process guarantees the same libraries
    that Speech uses at inference time are the ones doing the conversion.
    """
    try:
        from ctranslate2.converters import TransformersConverter
    except ImportError as exc:
        print(
            f"ctranslate2 is not installed, cannot convert: {exc}",
            file=sys.stderr,
        )
        return 1

    # transformers must be importable for the converter to load the source
    # checkpoint; surface a clear error if it is missing.
    try:
        import transformers  # noqa: F401
    except ImportError as exc:
        print(
            f"transformers is not installed, cannot convert: {exc}",
            file=sys.stderr,
        )
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        TransformersConverter(model_id).convert(
            str(out_dir), quantization="int8", force=True
        )
    except Exception as exc:
        print(f"CTranslate2 conversion failed: {exc}", file=sys.stderr)
        return 1
    return 0


def _write_installed_marker(preset: ModelPreset) -> None:
    marker = whisper_installed_marker(preset)
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "preset": preset.key,
        "model_id": preset.model_id,
        "installed_at": datetime.now(timezone.utc).isoformat(),
    }
    marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _ensure_whisper_installed(preset: ModelPreset) -> bool:
    marker = whisper_installed_marker(preset)
    model_dir = whisper_model_dir(preset)
    if not marker.exists() or not model_dir.exists():
        return False
    has_weights = any(
        (model_dir / name).is_file() for name in ("model.bin", "model.int8.bin")
    )
    return has_weights


def install_model(preset_key: str) -> int:
    """Install the model for ``preset_key`` (dispatches by engine family)."""
    preset = get_preset(preset_key)
    if preset.engine == "whisper":
        return install_whisper_model(preset)
    return install_parakeet_model(preset.model_id)


def list_models() -> int:
    """Print installation status for every preset."""
    # Imported here to avoid a circular import at module load.
    from ..model_status import find_model_status, find_whisper_model_status

    for preset in (get_preset("parakeet"), get_preset("whisper-ru")):
        if preset.engine == "whisper":
            status = find_whisper_model_status(preset)
        else:
            status = find_model_status(huggingface_home(), preset.model_id)
        state = status.label if status.installed else "Not installed"
        active = " (active)" if False else ""
        print(f"{preset.key}\t{preset.label}\t{state}{active}")
    return 0
