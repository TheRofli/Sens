"""Persistent NDJSON adapter from the Sens broker to Speech ASR."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# The broker does not set PYTHONUTF8, so stdout defaults to the ANSI code
# page (cp1252 on this machine). GigaAM returns Cyrillic, and printing it
# with `ensure_ascii=False` would raise UnicodeEncodeError and kill the
# worker. Pin UTF-8 for the protocol channel regardless of the environment.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


speech_root_value = os.environ.get("SENS_SPEECH_ROOT")
if not speech_root_value:
    print("SENS_SPEECH_ROOT is required", file=sys.stderr)
    raise SystemExit(2)

speech_root = Path(speech_root_value).resolve()
sys.path.insert(0, str(speech_root))

from speech_app.portable import build_portable_env  # noqa: E402

for key, value in build_portable_env(speech_root).items():
    os.environ.setdefault(key, value)

from speech_app.engine_manager import EngineManager  # noqa: E402
from speech_app.history import TranscriptHistory  # noqa: E402
from speech_app.settings import SettingsStore  # noqa: E402
from speech_app.transcription import (  # noqa: E402
    extract_frames,
    settings_for_request,
    transcribe_audio_file,
)


engine = EngineManager()
settings_store = SettingsStore()


def _frames_output_dir(request_id: str) -> Path:
    """Per-request frame output under the shared Sens artifacts root."""
    root = os.environ.get("SENS_ARTIFACTS_ROOT")
    if not root:
        local = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        root = str(Path(local) / "Sens" / "artifacts") if local else None
    if not root:
        raise RuntimeError("no artifacts root available for frames")
    out = Path(root) / "hearing-frames" / request_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def handle(message: dict[str, object]) -> dict[str, object]:
    operation = str(message.get("operation", ""))
    payload = message.get("input") or {}
    if not isinstance(payload, dict):
        raise ValueError("Hearing input must be an object")
    if operation == "dictation_status":
        settings = settings_store.load()
        return {
            "enabled": settings.engine_enabled,
            "model": settings.model,
            "engine": engine.kind,
            "modelLoaded": engine.is_loaded,
            "modelControlledMicrophone": False,
        }
    if operation != "hear":
        raise ValueError(f"Unsupported Hearing operation: {operation}")

    audio_path = str(payload.get("audioPath", "")).strip()
    if not audio_path:
        raise ValueError("audioPath is required")
    model = str(payload["model"]) if payload.get("model") else None
    base_settings = settings_store.load()
    if not base_settings.engine_enabled:
        raise ValueError("Hearing is disabled in Sens settings")
    settings = settings_for_request(base_settings, model=model)
    result = transcribe_audio_file(audio_path, settings=settings, engine=engine)
    frames = int(payload.get("frames", 0) or 0)
    if frames > 0:
        try:
            request_id = str(message.get("requestId", "unknown"))
            result["framePaths"] = extract_frames(
                audio_path,
                count=frames,
                out_dir=_frames_output_dir(request_id),
            )
        except Exception as error:  # noqa: BLE001 - frames are best-effort
            result["framesError"] = f"{type(error).__name__}: {error}"
    save_to_history = bool(payload.get("saveToHistory", False))
    if save_to_history and result["text"]:
        TranscriptHistory(max_entries=settings.history_limit).add(str(result["text"]))
    result["savedToHistory"] = save_to_history and bool(result["text"])
    result["clipboardWritten"] = False
    result["pastedToActiveInput"] = False
    return result


for line in sys.stdin:
    if not line.strip():
        continue
    request_id = None
    try:
        message = json.loads(line)
        request_id = message.get("requestId")
        result = handle(message)
        response = {"ok": True, "requestId": request_id, "result": result}
    except Exception as error:  # noqa: BLE001 - protocol boundary
        response = {
            "ok": False,
            "requestId": request_id,
            "error": {"message": str(error), "type": type(error).__name__},
        }
    try:
        print(json.dumps(response, ensure_ascii=False), flush=True)
    except Exception as error:  # noqa: BLE001 - stdout is broken; exit for the broker to restart us
        sys.stderr.write(f"could not write response: {error}\n")
        sys.stderr.flush()
        raise SystemExit(3)
