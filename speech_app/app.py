from __future__ import annotations

import argparse
import importlib.util
import os
import queue
import sys
import threading
import traceback
import tkinter as tk
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from .audio import AudioRecorder
from .engine_manager import EngineManager
from .engines.base import EngineUnavailable
from .history import TranscriptHistory
from .hotkeys import GlobalHotkeyListener
from .model_status import (
    ModelStatus,
    find_gigaam_model_status,
    find_model_status,
    find_whisper_model_status,
)
from .models import available_presets, get_preset, resolve_engine, resolve_model_id
from .output import TranscriptPublisher
from .overlay import VoiceOverlay
from .portable import build_portable_env
from .resources import ProcessResourceMonitor, ResourceSnapshot
from .runtime_state import write_runtime_state
from .settings import AppSettings, SettingsStore
from .settings import default_data_dir
from .single_instance import SingleInstanceLock
from .system import SystemActions
from .textpost import postprocess
from .transcription import settings_for_request, transcribe_audio_file
from .tray import TrayController
from .vad import trim_silence
from .visuals import enable_dpi_awareness, set_windows_app_id


class SpeechApp:
    def __init__(self) -> None:
        set_windows_app_id()
        enable_dpi_awareness()
        self.root = tk.Tk()
        self.root.withdraw()
        self.root.title("Speech")
        self.ui_queue: queue.Queue[Callable[[], None]] = queue.Queue()

        self.settings_store = SettingsStore()
        self.settings = self.settings_store.load()
        self.history = TranscriptHistory(max_entries=self.settings.history_limit)
        self.system = SystemActions()
        self.publisher = TranscriptPublisher(
            history=self.history,
            set_clipboard=self.system.copy_to_clipboard,
            paste_active_input=self.system.paste_into_active_input,
        )
        self.engine = EngineManager()
        self.engine_lock = threading.Lock()
        self.resource_monitor = ProcessResourceMonitor()
        self.overlay = VoiceOverlay(self.root)
        self.tray = TrayController(self)
        self.recorder = AudioRecorder(
            sample_rate=self.settings.sample_rate,
            level_callback=lambda level: self.post_ui(
                lambda: self.overlay.set_level(level)
            ),
        )
        self.hotkey_listener: GlobalHotkeyListener | None = None
        self.transcribing = False
        self.model_loading = False
        self.last_error = ""
        self.api_server = None
        self._write_runtime_state("unloaded")

    def run(self, show_window: bool = False, managed: bool = False) -> None:
        if managed:
            os.environ["SPEECH_MANAGED"] = "1"
        self.root.after(30, self._pump_ui_queue)
        # In Sens-managed mode the Sens shell owns the only tray and primary
        # window. Speech keeps hotkeys, the lightweight recording overlay,
        # local ASR, and its authenticated control API.
        tray_started = True if managed else self.tray.start()
        self._start_hotkeys()
        if not tray_started:
            self.last_error = "pystray is not installed; tray mode is unavailable."
        elif show_window:
            self._show_primary_window()
        if self.settings.preload_model and self.settings.engine_enabled:
            self.load_model_background()
        self._start_api()
        self.root.mainloop()

    def post_ui(self, callback: Callable[[], None]) -> None:
        self.ui_queue.put(callback)

    def post_ui_sync(self, callback: Callable[[], object], timeout: float = 5.0) -> object:
        """Run ``callback`` on the UI thread and wait for its result.

        Used by the HTTP API server (which runs in its own thread) to mutate
        application state safely: tkinter is not thread-safe, so every state
        change must hop onto the UI loop. Raises ``TimeoutError`` if the UI
        thread does not service the callback within ``timeout`` seconds.
        """
        import threading

        done = threading.Event()
        box: dict[str, object] = {}

        def runner() -> None:
            try:
                box["result"] = callback()
            except BaseException as exc:  # noqa: BLE001 - re-raised to caller
                box["error"] = exc
            finally:
                done.set()

        self.ui_queue.put(runner)
        if not done.wait(timeout):
            raise TimeoutError("UI thread did not service the callback in time")
        if "error" in box:
            raise box["error"]  # type: ignore[misc]
        return box.get("result")

    def _start_api(self) -> None:
        """Start the local HTTP API server on an ephemeral port.

        The port is written to ``data/api.port`` so the Tauri shell (or any
        other local client) can discover it. Failures are non-fatal: the tray
        app keeps working without the API.
        """
        try:
            from .api import SpeechAPIServer

            self.api_server = SpeechAPIServer(self)
            self.api_server.start()
        except Exception as exc:
            self.last_error = f"API server failed to start: {exc}"

    def show_window(self) -> None:
        self.post_ui(self._show_primary_window)

    def show_history(self) -> None:
        self.post_ui(self._show_primary_window)

    def copy_last_transcript(self) -> None:
        entries = self.history.list()
        if not entries:
            self.post_ui(lambda: self.overlay.show_notice("No transcript yet"))
            return
        self.system.copy_to_clipboard(entries[0].text)
        self.post_ui(lambda: self.overlay.show_notice("Copied"))

    def _show_primary_window(self) -> None:
        """Open the GUI. Tauri is the only window; if it is not built, notify."""
        speech_home = Path(__file__).resolve().parents[1]
        if self.system.open_tauri_ui(speech_home):
            self.overlay.show_notice("Opening Speech")
            return
        self.overlay.show_notice("Build Tauri: see README", timeout_ms=2600)
        self.tray.notify(
            "Speech",
            "GUI window needs Tauri. Build it with: npm run tauri:build",
        )

    def toggle_engine(self) -> None:
        self.settings.engine_enabled = not self.settings.engine_enabled
        self.settings_store.save(self.settings)
        if not self.settings.engine_enabled:
            self.unload_model()
            self.post_ui(lambda: self.overlay.show_notice("Engine off"))
        else:
            self.post_ui(lambda: self.overlay.show_notice("Engine on"))

    def load_model_background(self) -> None:
        label = self.current_model_label()
        if self.engine.is_loaded:
            self.model_loading = False
            self._write_runtime_state("loaded")
            self.post_ui(lambda: self._model_state_changed(f"{label} loaded"))
            return
        if self.model_loading:
            self.post_ui(lambda: self._model_state_changed(f"{label} loading"))
            return
        self.model_loading = True
        self._write_runtime_state("loading")
        self.post_ui(lambda: self._model_state_changed(f"{label} loading"))
        settings_snapshot = replace(self.settings)
        threading.Thread(
            target=self._load_model_worker,
            args=(settings_snapshot,),
            daemon=True,
        ).start()

    def unload_model(self) -> None:
        label = self.current_model_label()
        self.model_loading = False
        engine_lock = getattr(self, "engine_lock", None)
        if engine_lock is None:
            self.engine.unload()
        else:
            with engine_lock:
                self.engine.unload()
        self._write_runtime_state("unloaded")
        self.post_ui(lambda: self._model_state_changed(f"{label} unloaded"))

    def set_device(self, device: str) -> None:
        self.settings.device = device
        self.settings_store.save(self.settings)
        self.unload_model()

    def set_backend(self, backend: str) -> None:
        self.settings.backend = backend
        self.settings_store.save(self.settings)
        self.unload_model()

    def set_model(self, key: str) -> None:
        """Switch the active model preset, saving settings and unloading.

        The model is reloaded lazily on the next transcription (or on preload).
        """
        preset = get_preset(key)
        previous_engine = resolve_engine(self.settings)
        self.settings.model = preset.key
        self.settings.model_id = preset.model_id
        self.settings_store.save(self.settings)
        # Engine kind changed -> must unload so the right one loads next.
        if resolve_engine(self.settings) != previous_engine or self.engine.is_loaded:
            self.unload_model()
            if self.settings.preload_model and self.settings.engine_enabled:
                self.load_model_background()
            else:
                self.post_ui(
                    lambda: self._model_state_changed(f"{preset.label} selected")
                )

    def available_models(self) -> list[dict[str, object]]:
        """Return all presets with installation status for UI display."""
        out: list[dict[str, object]] = []
        for preset in available_presets():
            status = self.model_status_for(preset.key)
            out.append(
                {
                    "key": preset.key,
                    "label": preset.label,
                    "engine": preset.engine,
                    "model_id": preset.model_id,
                    "description": preset.description,
                    "installed": status.installed,
                    "size_label": status.size_label,
                    "active": preset.key == self.settings.model,
                }
            )
        return out

    def current_model(self) -> str:
        return self.settings.model

    def current_model_label(self) -> str:
        preset = get_preset(self.settings.model) if self.settings.model else None
        return preset.label if preset is not None else self.settings.model

    def engine_enabled(self) -> bool:
        return self.settings.engine_enabled

    def current_device(self) -> str:
        return self.settings.device

    def current_backend(self) -> str:
        return self.settings.backend

    def model_loaded(self) -> bool:
        return self.engine.is_loaded

    def model_is_loading(self) -> bool:
        return self.model_loading

    def status_text(self) -> str:
        engine = "on" if self.settings.engine_enabled else "off"
        state = self._model_state_label()
        label = self.current_model_label()
        return f"Engine {engine} | {label} {state} | {self.settings.device}"

    def get_settings_values(self) -> dict[str, object]:
        return {
            "model": self.settings.model,
            "engine_enabled": self.settings.engine_enabled,
            "copy_to_clipboard": self.settings.copy_to_clipboard,
            "paste_to_active_input": self.settings.paste_to_active_input,
            "preload_model": self.settings.preload_model,
            "device": self.settings.device,
            "backend": self.settings.backend,
            "hotkey": self.settings.hotkey,
            "beam_size": self.settings.beam_size,
            "temperature": self.settings.temperature,
            "repetition_penalty": self.settings.repetition_penalty,
            "no_repeat_ngram_size": self.settings.no_repeat_ngram_size,
            "vad_sensitivity": self.settings.vad_sensitivity,
            "postprocess_text": self.settings.postprocess_text,
        }

    def save_settings_values(self, values: dict[str, object]) -> None:
        previous_hotkey = self.settings.hotkey
        previous_model = self.settings.model
        if "model" in values:
            self.settings.model = str(values["model"])
            preset = get_preset(self.settings.model)
            self.settings.model_id = preset.model_id
        self.settings.engine_enabled = bool(values["engine_enabled"])
        self.settings.copy_to_clipboard = bool(values["copy_to_clipboard"])
        self.settings.paste_to_active_input = bool(values["paste_to_active_input"])
        self.settings.preload_model = bool(values["preload_model"])
        self.settings.device = str(values["device"])
        self.settings.backend = str(values["backend"])
        self.settings.hotkey = str(values["hotkey"])
        # Quality params (optional in values for back-compat with old callers).
        if "beam_size" in values:
            self.settings.beam_size = int(values["beam_size"])
        if "temperature" in values:
            self.settings.temperature = float(values["temperature"])
        if "repetition_penalty" in values:
            self.settings.repetition_penalty = float(values["repetition_penalty"])
        if "no_repeat_ngram_size" in values:
            self.settings.no_repeat_ngram_size = int(values["no_repeat_ngram_size"])
        if "vad_sensitivity" in values:
            self.settings.vad_sensitivity = float(values["vad_sensitivity"])
        if "postprocess_text" in values:
            self.settings.postprocess_text = bool(values["postprocess_text"])
        # Remote (OpenRouter) transcription settings; absent for callers that
        # predate the feature, so keep the current value in that case.
        if "remote_api_key" in values:
            self.settings.remote_api_key = str(values["remote_api_key"])
        if "remote_base_url" in values:
            self.settings.remote_base_url = str(values["remote_base_url"])
        if "remote_model_id" in values:
            self.settings.remote_model_id = str(values["remote_model_id"])
        self.settings_store.save(self.settings)
        if self.settings.hotkey != previous_hotkey:
            self._restart_hotkeys()
        if self.settings.model != previous_model:
            self.unload_model()

    def history_rows(self) -> list[tuple[str, str]]:
        return [(entry.id, entry.text) for entry in self.history.list()]

    def copy_history_entry(self, entry_id: str) -> None:
        for entry in self.history.list():
            if entry.id == entry_id:
                self.system.copy_to_clipboard(entry.text)
                self.overlay.show_notice("Copied")
                return

    def quit(self) -> None:
        self.post_ui(self._quit_ui)

    def _quit_ui(self) -> None:
        if self.hotkey_listener is not None:
            self.hotkey_listener.stop()
        self.model_loading = False
        api_server = getattr(self, "api_server", None)
        if api_server is not None:
            try:
                api_server.stop()
            except Exception:
                pass
            self.api_server = None
        engine_lock = getattr(self, "engine_lock", None)
        if engine_lock is None:
            self.engine.unload()
        else:
            with engine_lock:
                self.engine.unload()
        self._write_runtime_state("unloaded", running=False)
        self.tray.stop()
        self.root.quit()
        self.root.destroy()

    def _start_hotkeys(self) -> None:
        try:
            self.hotkey_listener = GlobalHotkeyListener(
                hotkey=self.settings.hotkey,
                on_start=lambda: self.post_ui(self._begin_recording),
                on_stop=lambda: self.post_ui(self._finish_recording),
                suppress=self.settings.suppress_hotkey,
            )
            self.hotkey_listener.start()
        except Exception as exc:
            self.last_error = str(exc)
            self.overlay.show_notice("Hotkey unavailable", timeout_ms=2200)

    def _restart_hotkeys(self) -> None:
        if self.hotkey_listener is not None:
            self.hotkey_listener.stop()
            self.hotkey_listener = None
        self._start_hotkeys()

    def _begin_recording(self) -> None:
        if not self.settings.engine_enabled:
            self.overlay.show_notice("Engine off")
            return
        if self.recorder.is_recording or self.transcribing:
            return
        try:
            self.recorder.start()
            self.system.release_hotkey_modifiers()
            self.overlay.show_recording()
        except Exception as exc:
            self.last_error = str(exc)
            self.overlay.show_notice("Microphone error", timeout_ms=2200)
            self.tray.notify("Speech", str(exc))

    def _finish_recording(self) -> None:
        if not self.recorder.is_recording:
            return
        samples = self.recorder.stop()
        self.overlay.show_transcribing()
        self.transcribing = True
        settings_snapshot = replace(self.settings)
        threading.Thread(
            target=self._transcribe_worker,
            args=(samples, settings_snapshot.sample_rate, settings_snapshot),
            daemon=True,
        ).start()

    def _load_model_worker(
        self, settings_snapshot: AppSettings | None = None
    ) -> None:
        settings_snapshot = settings_snapshot or replace(self.settings)
        try:
            with self.engine_lock:
                self.engine.load(settings_snapshot)
        except Exception as exc:
            self.last_error = str(exc)
            self.post_ui(lambda error=exc: self._model_load_failed(error))
            return
        self.post_ui(self._model_load_succeeded)

    def _model_load_succeeded(self) -> None:
        self.model_loading = False
        self._write_runtime_state("loaded")
        self._model_state_changed(f"{self.current_model_label()} loaded")

    def _model_load_failed(self, exc: Exception) -> None:
        self.model_loading = False
        self._write_runtime_state("error", last_error=str(exc))
        self.tray.refresh_menu()
        self._show_error(f"{self.current_model_label()} load failed", exc)

    def _model_state_changed(self, notice: str) -> None:
        self.tray.refresh_menu()
        self.overlay.show_notice(notice)

    def _model_state_label(self) -> str:
        if self.model_loading:
            return "loading"
        if self.engine.is_loaded:
            return "loaded"
        return "unloaded"

    def _write_runtime_state(
        self,
        model_state: str,
        running: bool = True,
        last_error: str = "",
    ) -> None:
        try:
            write_runtime_state(
                model_state=model_state,
                settings=self.settings,
                running=running,
                last_error=last_error,
            )
        except Exception:
            pass

    def _transcribe_worker(
        self, samples, sample_rate: int, settings_snapshot: AppSettings
    ) -> None:
        text = ""
        failed: tuple[str, Exception] | None = None
        try:
            # Voice-activity trim: drop leading/trailing silence so keyboard
            # clicks and breath do not feed the model (major Whisper
            # hallucination cause on near-empty audio).
            trimmed = trim_silence(
                samples,
                sample_rate=sample_rate,
                sensitivity=settings_snapshot.vad_sensitivity,
            )
            if trimmed.size == 0:
                # No speech detected; publish empty so the overlay clears and
                # the "No speech detected" notice shows.
                self.post_ui(
                    lambda: setattr(self, "transcribing", False)
                )
                self.post_ui(
                    lambda: self._publish_transcript("", settings_snapshot)
                )
                return
            with self.engine_lock:
                raw_text = self.engine.transcribe(
                    trimmed, sample_rate, settings_snapshot
                )
            text = (
                postprocess(raw_text)
                if settings_snapshot.postprocess_text
                else (raw_text or "").strip()
            )
        except EngineUnavailable as exc:
            self.last_error = str(exc)
            failed = ("Engine unavailable", exc)
        except Exception as exc:
            self.last_error = traceback.format_exc()
            failed = ("Transcription failed", exc)

        # Always clear the transcribing flag so the overlay never gets stuck.
        self.post_ui(lambda: setattr(self, "transcribing", False))

        if failed is not None:
            # Bind the tuple eagerly to avoid the late-binding NameError that
            # used to swallow the real error inside a lambda closure.
            title, exc = failed
            self.post_ui(lambda t=title, e=exc: self._show_error(t, e))
            return

        self.post_ui(lambda: self._publish_transcript(text, settings_snapshot))

    def transcribe_file(
        self,
        audio_path: str,
        *,
        model: str | None = None,
        language: str | None = None,
        save_to_history: bool = False,
    ) -> dict[str, object]:
        """Transcribe one local file without clipboard or paste side effects."""
        del language  # Reserved for engines that support an explicit language.
        settings_snapshot = settings_for_request(self.settings, model=model)
        with self.engine_lock:
            result = transcribe_audio_file(
                audio_path,
                settings=settings_snapshot,
                engine=self.engine,
            )
        if save_to_history and result["text"]:
            self.history.add(str(result["text"]))
        return result

    def _publish_transcript(
        self, text: str, settings_snapshot: AppSettings
    ) -> None:
        self.overlay.hide()
        self.root.after(140, lambda: self._publish_transcript_after_focus(text, settings_snapshot))

    def _publish_transcript_after_focus(
        self, text: str, settings_snapshot: AppSettings
    ) -> None:
        entry = self.publisher.publish(text, settings_snapshot)
        if entry is None:
            self.overlay.show_notice("No speech detected")
        else:
            self.overlay.show_notice("Inserted")

    def _show_error(self, title: str, exc: Exception) -> None:
        message = str(exc)
        self.overlay.show_notice(title, timeout_ms=2400)
        self.tray.notify(title, message)

    def _pump_ui_queue(self) -> None:
        while True:
            try:
                callback = self.ui_queue.get_nowait()
            except queue.Empty:
                break
            callback()
        self.root.after(30, self._pump_ui_queue)

    def resource_snapshot(self) -> ResourceSnapshot:
        return self.resource_monitor.snapshot()

    def model_status(self) -> ModelStatus:
        """Status of the currently active model preset."""
        return self.model_status_for(self.settings.model)

    def model_status_for(self, key: str) -> ModelStatus:
        """Installation status for a given preset key."""
        preset = get_preset(key)
        if preset.engine == "whisper":
            return find_whisper_model_status(preset)
        if preset.engine == "gigaam":
            return find_gigaam_model_status(preset)
        fallback = Path(__file__).resolve().parents[1] / "models" / "huggingface"
        hf_home = Path(os.environ.get("HF_HOME", str(fallback)))
        return find_model_status(hf_home, preset.model_id)


def diagnose() -> int:
    modules = [
        "tkinter",
        "numpy",
        "sounddevice",
        "pystray",
        "PIL",
        "pynput",
        "pyperclip",
        "torch",
        "transformers",
        "librosa",
        "nemo",
    ]
    print(f"Python: {sys.version}")
    for module in modules:
        print(f"{module}: {'ok' if importlib.util.find_spec(module) else 'missing'}")
    try:
        import torch

        print(f"torch cuda available: {torch.cuda.is_available()}")
    except ImportError:
        pass
    return 0


def install_parakeet_model(model_id: str | None = None) -> int:
    """Back-compat wrapper around the shared installer.

    Kept so legacy callers (and tests) that import this name keep working.
    """
    from .engines.install import install_model

    if model_id is None:
        return install_model(AppSettings().model)
    # Resolve the preset key from a raw model id if possible, else treat the
    # value as a parakeet id directly.
    settings = AppSettings()
    for preset in available_presets():
        if preset.model_id == model_id:
            return install_model(preset.key)
    settings.model_id = model_id
    from .engines.install import install_parakeet_model as _install

    return _install(model_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="speech")
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Compatibility alias for: speech diagnose",
    )
    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser("run", help="Start the Speech tray app.")
    run_parser.add_argument(
        "--show-window",
        action="store_true",
        help="Open the Speech window immediately after starting the tray app.",
    )
    run_parser.add_argument(
        "--managed",
        action="store_true",
        help="Run under Sens without starting the legacy Speech tray/window.",
    )
    subparsers.add_parser("diagnose", help="Check Python and dependency state.")

    parakeet = subparsers.add_parser(
        "parakeet", help="(legacy) Manage the Parakeet model."
    )
    parakeet_sub = parakeet.add_subparsers(dest="parakeet_command")
    parakeet_sub.add_parser(
        "install",
        help="Download Parakeet into the configured local model cache.",
    )

    model = subparsers.add_parser("model", help="Manage ASR models.")
    model_sub = model.add_subparsers(dest="model_command")
    install_parser = model_sub.add_parser(
        "install", help="Download/convert a model (parakeet or whisper-ru)."
    )
    install_parser.add_argument(
        "key",
        nargs="?",
        default=None,
        help="Preset key (parakeet, whisper-ru). Defaults to the active model.",
    )
    model_sub.add_parser("list", help="Show installation status of all models.")
    return parser


def apply_portable_env_if_present() -> None:
    speech_home = Path(__file__).resolve().parents[1]
    env = build_portable_env(speech_home)
    for key, value in env.items():
        if key not in sys.modules.get("os").environ:
            sys.modules.get("os").environ[key] = value


def main(argv: list[str] | None = None) -> int:
    import os

    speech_home = Path(__file__).resolve().parents[1]
    for key, value in build_portable_env(speech_home).items():
        os.environ.setdefault(key, value)

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.diagnose or args.command == "diagnose":
        return diagnose()
    if args.command == "parakeet" and args.parakeet_command == "install":
        return install_parakeet_model()
    if args.command == "parakeet":
        parser.error("Choose a Parakeet command, for example: speech parakeet install")
    if args.command == "model":
        from .engines.install import install_model, list_models

        if args.model_command == "install":
            key = args.key or AppSettings().model
            return install_model(key)
        if args.model_command == "list":
            return list_models()
        parser.error(
            "Choose a model command, for example: speech model install whisper-ru"
        )

    lock = SingleInstanceLock(default_data_dir() / "speech.lock")
    if not lock.acquire():
        print("Speech is already running.")
        return 0
    try:
        app = SpeechApp()
        app.run(
            show_window=bool(getattr(args, "show_window", False)),
            managed=bool(getattr(args, "managed", False)),
        )
        return 0
    finally:
        lock.release()
