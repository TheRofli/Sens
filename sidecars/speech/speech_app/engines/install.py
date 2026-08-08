"""Verified, staged installers for Sens Hearing model packs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from ..models import ModelPreset, available_presets, get_preset
from ..vad import (
    SILERO_VAD_BYTES,
    SILERO_VAD_SHA256,
    SILERO_VAD_URL,
    silero_vad_path,
)
from .paths import installed_marker, model_dir, models_root, pack_download_dir

ProgressCallback = Callable[[dict[str, object]], None]


def _print(message: str, *, error: bool = False) -> None:
    """Write human CLI status without failing on a legacy Windows code page."""
    stream = sys.stderr if error else sys.stdout
    encoding = getattr(stream, "encoding", None) or "utf-8"
    safe = message.encode(encoding, errors="backslashreplace").decode(encoding)
    print(safe, file=stream)


def _required_files_exist(preset: ModelPreset, root: Path) -> bool:
    return bool(preset.required_files) and all(
        (root / relative).is_file() for relative in preset.required_files
    )


def _write_installed_marker(root: Path, preset: ModelPreset) -> None:
    payload = {
        "schema_version": 1,
        "preset": preset.key,
        "model_id": preset.model_id,
        "revision": preset.revision or None,
        "archive_sha256": preset.download_sha256 or None,
        "installed_at": datetime.now(timezone.utc).isoformat(),
    }
    (root / "INSTALLED.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _is_installed(preset: ModelPreset) -> bool:
    root = model_dir(preset)
    return installed_marker(preset).is_file() and _required_files_exist(preset, root)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_archive(
    preset: ModelPreset,
    target: Path,
    progress: ProgressCallback | None = None,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = target.stat().st_size if target.is_file() else 0
    if preset.download_bytes and existing == preset.download_bytes:
        if _sha256(target).lower() == preset.download_sha256.lower():
            return
        target.unlink()
        existing = 0
    elif preset.download_bytes and existing > preset.download_bytes:
        target.unlink()
        existing = 0
    headers = {"User-Agent": "Sens/1.3.5"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    request = urllib.request.Request(preset.download_url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        resumed = existing > 0 and getattr(response, "status", 200) == 206
        mode = "ab" if resumed else "wb"
        received = existing if resumed else 0
        with target.open(mode) as stream:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
                received += len(chunk)
                if progress is not None:
                    progress(
                        {
                            "phase": "downloading",
                            "model": preset.key,
                            "bytes_present": received,
                            "bytes_required": preset.download_bytes,
                        }
                    )
    actual_size = target.stat().st_size
    if preset.download_bytes and actual_size != preset.download_bytes:
        if actual_size > preset.download_bytes:
            target.unlink(missing_ok=True)
        raise RuntimeError(
            f"Incomplete {preset.key} download: {actual_size} of "
            f"{preset.download_bytes} bytes"
        )
    actual_digest = _sha256(target)
    if actual_digest.lower() != preset.download_sha256.lower():
        target.unlink(missing_ok=True)
        raise RuntimeError(f"{preset.key} model archive failed SHA-256 verification")


def install_vad_model() -> int:
    destination = silero_vad_path()
    if (
        destination.is_file()
        and destination.stat().st_size == SILERO_VAD_BYTES
        and _sha256(destination).lower() == SILERO_VAD_SHA256
    ):
        return 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".onnx.part")
    try:
        asset = ModelPreset(
            key="vad",
            label="Silero VAD",
            engine="vad",
            model_id="silero-vad",
            family="silero-vad",
            download_url=SILERO_VAD_URL,
            download_sha256=SILERO_VAD_SHA256,
            download_bytes=SILERO_VAD_BYTES,
        )
        _download_archive(asset, temporary)
        os.replace(temporary, destination)
        return 0
    except Exception as exc:
        _print(f"Could not install Silero VAD: {exc}", error=True)
        return 1


def _safe_extract(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with tarfile.open(archive, "r:bz2") as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise RuntimeError("Model archive contains an unsafe path") from exc
            if member.issym() or member.islnk() or member.isdev():
                raise RuntimeError("Model archive contains an unsupported link or device")
        bundle.extractall(destination)


def _payload_root(extraction_root: Path, preset: ModelPreset) -> Path:
    if _required_files_exist(preset, extraction_root):
        return extraction_root
    candidates = [
        path
        for path in extraction_root.iterdir()
        if path.is_dir() and _required_files_exist(preset, path)
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"{preset.key} archive is missing required model files")
    return candidates[0]


def _promote_model(source: Path, destination: Path) -> None:
    backup = destination.with_name(destination.name + ".previous")
    if backup.exists():
        shutil.rmtree(backup)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.rename(backup)
    try:
        source.rename(destination)
    except Exception:
        if backup.exists() and not destination.exists():
            backup.rename(destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def install_archive_model(
    preset: ModelPreset,
    progress: ProgressCallback | None = None,
) -> int:
    if _is_installed(preset):
        if progress is not None:
            progress({"phase": "ready", "model": preset.key})
        _print(f"{preset.label} is already installed at {model_dir(preset)}")
        return 0
    downloads = pack_download_dir()
    archive_name = preset.download_url.rsplit("/", 1)[-1]
    archive = downloads / (archive_name + ".part")
    extraction_parent = models_root()
    extraction_parent.mkdir(parents=True, exist_ok=True)
    extraction = Path(
        tempfile.mkdtemp(prefix=f".{preset.key}-install-", dir=extraction_parent)
    )
    try:
        _print(
            f"Downloading {preset.label} ({preset.download_bytes / 1024**2:.0f} MiB)..."
        )
        if progress is None:
            _download_archive(preset, archive)
        else:
            _download_archive(preset, archive, progress)
        if progress is not None:
            progress({"phase": "installing", "model": preset.key})
        _safe_extract(archive, extraction)
        payload = _payload_root(extraction, preset)
        _write_installed_marker(payload, preset)
        _promote_model(payload, model_dir(preset))
        archive.unlink(missing_ok=True)
        if progress is not None:
            progress({"phase": "ready", "model": preset.key})
        _print(f"{preset.label} is ready at {model_dir(preset)}")
        return 0
    except Exception as exc:
        _print(f"Could not install {preset.label}: {exc}", error=True)
        return 1
    finally:
        shutil.rmtree(extraction, ignore_errors=True)


def install_whisper_model(
    preset: ModelPreset,
    progress: ProgressCallback | None = None,
) -> int:
    if _is_installed(preset):
        if progress is not None:
            progress({"phase": "ready", "model": preset.key})
        _print(f"{preset.label} is already installed at {model_dir(preset)}")
        return 0
    try:
        from faster_whisper.utils import download_model
    except ImportError:
        _print("faster-whisper is missing from the Sens runtime", error=True)
        return 1

    root = models_root()
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".whisper-install-", dir=root))
    try:
        if progress is not None:
            progress(
                {
                    "phase": "downloading",
                    "model": preset.key,
                    "bytes_present": 0,
                    "bytes_required": preset.download_bytes,
                }
            )
        _print(f"Downloading pinned {preset.label}...")
        download_model(
            "small",
            output_dir=str(staging),
            revision=preset.revision,
            use_auth_token=False,
        )
        if not _required_files_exist(preset, staging):
            raise RuntimeError("Whisper download is missing required model files")
        if progress is not None:
            progress({"phase": "installing", "model": preset.key})
        _write_installed_marker(staging, preset)
        _promote_model(staging, model_dir(preset))
        if progress is not None:
            progress({"phase": "ready", "model": preset.key})
        _print(f"{preset.label} is ready at {model_dir(preset)}")
        return 0
    except Exception as exc:
        _print(f"Could not install {preset.label}: {exc}", error=True)
        return 1
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def install_model(
    preset_key: str,
    progress: ProgressCallback | None = None,
) -> int:
    preset = get_preset(preset_key)
    if progress is not None:
        progress({"phase": "preparing", "model": preset.key})
    if preset.engine != "remote" and install_vad_model() != 0:
        return 1
    if preset.engine in {"qwen", "gigaam"}:
        return install_archive_model(preset, progress)
    if preset.engine == "whisper":
        return install_whisper_model(preset, progress)
    _print("Remote transcription does not install a local model", error=True)
    return 1


def list_models() -> int:
    from ..model_status import find_model_status_for_preset

    for preset in available_presets():
        status = find_model_status_for_preset(preset)
        _print(f"{preset.key}\t{preset.label}\t{status.label}")
    return 0
