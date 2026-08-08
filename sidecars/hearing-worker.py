"""Persistent NDJSON adapter from the Sens broker to Speech ASR."""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

# The broker does not set PYTHONUTF8, so stdout defaults to the ANSI code
# page (cp1252 on this machine). GigaAM returns Cyrillic, and printing it
# with `ensure_ascii=False` would raise UnicodeEncodeError and kill the
# worker. Pin UTF-8 for the protocol channel regardless of the environment.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Keep the one protocol stream private. Imported runtimes occasionally print
# progress or diagnostics to regular stdout; route those to stderr so they can
# never corrupt the broker's NDJSON channel.
protocol_stdout = sys.stdout
sys.stdout = sys.stderr


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
from speech_app.settings import SettingsStore, migrate_legacy_user_data  # noqa: E402
from speech_app.transcription import (  # noqa: E402
    extract_frames,
    extract_frames_at,
    extract_frames_every,
    settings_for_request,
    transcribe_audio_file,
)
from speech_app.fetch_media import fetch_video  # noqa: E402


migrate_legacy_user_data()

engine = EngineManager()
engine_lock = threading.Lock()
settings_store = SettingsStore()
dictation_lock = threading.Lock()
dictation_ready = threading.Event()
dictation_app = None
dictation_thread: threading.Thread | None = None
dictation_error = ""


def _dictation_status() -> dict[str, object]:
    settings = settings_store.load()
    with dictation_lock:
        app = dictation_app
        error = dictation_error
        thread = dictation_thread
    running = bool(app is not None and thread is not None and thread.is_alive())
    if app is not None:
        settings = app.settings
    return {
        "running": running,
        "managed": True,
        "engine_enabled": settings.engine_enabled,
        "hotkey": settings.hotkey,
        "model": settings.model,
        "model_state": app._model_state_label() if app is not None else "unloaded",  # noqa: SLF001
        "model_loaded": engine.is_loaded,
        "model_loading": bool(app is not None and app.model_loading),
        "transcribing": bool(app is not None and app.transcribing),
        "error": error or None,
        "modelControlledMicrophone": False,
    }


def _run_dictation() -> None:
    global dictation_app, dictation_error
    try:
        from speech_app.app import SpeechApp

        app = SpeechApp(engine=engine, engine_lock=engine_lock)
        with dictation_lock:
            dictation_app = app
            dictation_error = ""
        dictation_ready.set()
        app.run(managed=True, control_api=False)
    except Exception as error:  # noqa: BLE001 - surfaced through status
        with dictation_lock:
            dictation_error = f"{type(error).__name__}: {error}"
        dictation_ready.set()
    finally:
        with dictation_lock:
            dictation_app = None


def _start_dictation(settings_values: dict[str, object]) -> dict[str, object]:
    global dictation_thread, dictation_error
    with dictation_lock:
        current = dictation_thread
        if current is None or not current.is_alive():
            dictation_ready.clear()
            dictation_error = ""
            current = threading.Thread(
                target=_run_dictation,
                name="sens-dictation-ui",
                daemon=True,
            )
            dictation_thread = current
            current.start()
    if not dictation_ready.wait(timeout=8.0):
        raise RuntimeError("Dictation UI did not become ready")
    with dictation_lock:
        app = dictation_app
        error = dictation_error
    if app is None:
        raise RuntimeError(error or "Dictation UI failed to start")
    if settings_values:
        current_values = app.get_settings_values()
        merged = {**current_values, **settings_values}
        app.post_ui_sync(lambda: app.save_settings_values(merged))
    return _dictation_status()


def _stop_dictation() -> dict[str, object]:
    with dictation_lock:
        app = dictation_app
        thread = dictation_thread
    if app is not None:
        app.quit()
    if thread is not None and thread.is_alive():
        thread.join(timeout=5.0)
    return _dictation_status()


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


def _fetch_cache_dir() -> Path:
    """Shared cache for media fetched from URLs, keyed by video id."""
    root = os.environ.get("SENS_ARTIFACTS_ROOT")
    if not root:
        local = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        root = str(Path(local) / "Sens" / "artifacts") if local else None
    if not root:
        raise RuntimeError("no artifacts root available for fetched media")
    out = Path(root) / "fetched-media"
    out.mkdir(parents=True, exist_ok=True)
    return out


def handle(message: dict[str, object]) -> dict[str, object]:
    operation = str(message.get("operation", ""))
    payload = message.get("input") or {}
    if not isinstance(payload, dict):
        raise ValueError("Hearing input must be an object")
    if operation == "dictation_status":
        return _dictation_status()
    if operation == "dictation_start":
        return _start_dictation(payload)
    if operation == "dictation_settings":
        with dictation_lock:
            app = dictation_app
        if app is not None:
            current_values = app.get_settings_values()
            merged = {**current_values, **payload}
            app.post_ui_sync(lambda: app.save_settings_values(merged))
        return _dictation_status()
    if operation == "dictation_stop":
        return _stop_dictation()
    if operation == "fetch":
        url = str(payload.get("url", "")).strip()
        if not url:
            raise ValueError("url is required")
        return fetch_video(url, _fetch_cache_dir())
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
    with engine_lock:
        result = transcribe_audio_file(audio_path, settings=settings, engine=engine)
    at = payload.get("at")
    frames = int(payload.get("frames", 0) or 0)
    every = payload.get("every")
    every_s = float(every) if isinstance(every, (int, float)) and float(every) > 0 else 0.0
    if every_s <= 0 and settings.default_every > 0:
        every_s = float(settings.default_every)
    max_frames = max(1, int(getattr(settings, "max_frames", 12) or 12))
    frame_size = max(320, int(getattr(settings, "frame_size", 640) or 640))
    try:
        request_id = str(message.get("requestId", "unknown"))
        out_dir = _frames_output_dir(request_id)
        if isinstance(at, list) and any(isinstance(v, (int, float)) for v in at):
            # Exact-second stills requested by the model; highest priority.
            result["framePaths"] = extract_frames_at(
                audio_path,
                [float(v) for v in at if isinstance(v, (int, float))][:max_frames],
                out_dir=out_dir,
                max_side=frame_size,
            )
        elif every_s > 0:
            # One still every N seconds (model or default setting).
            result["framePaths"] = extract_frames_every(
                audio_path,
                every_s,
                max_count=max_frames,
                out_dir=out_dir,
                max_side=frame_size,
            )
        elif frames > 0:
            result["framePaths"] = extract_frames(
                audio_path,
                count=min(frames, max_frames),
                out_dir=out_dir,
                max_side=frame_size,
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
        print(
            json.dumps(response, ensure_ascii=False),
            file=protocol_stdout,
            flush=True,
        )
    except Exception as error:  # noqa: BLE001 - stdout is broken; exit for the broker to restart us
        sys.stderr.write(f"could not write response: {error}\n")
        sys.stderr.flush()
        raise SystemExit(3)
