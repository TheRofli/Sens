import unittest
from unittest.mock import patch

from speech_app.app import SpeechApp
from speech_app.settings import AppSettings


class FakeWindow:
    def __init__(self) -> None:
        self.refresh_count = 0
        self.show_count = 0

    def refresh(self) -> None:
        self.refresh_count += 1

    def show(self) -> None:
        self.show_count += 1


class FakeTray:
    def __init__(self) -> None:
        self.refresh_count = 0
        self.stop_count = 0

    def refresh_menu(self) -> None:
        self.refresh_count += 1

    def stop(self) -> None:
        self.stop_count += 1


class FakeOverlay:
    def __init__(self) -> None:
        self.notices = []
        self.recording_count = 0

    def show_notice(self, message: str) -> None:
        self.notices.append(message)

    def show_recording(self) -> None:
        self.recording_count += 1


class FakeSystem:
    def __init__(self) -> None:
        self.remember_count = 0
        self.release_count = 0

    def remember_active_window(self) -> None:
        self.remember_count += 1

    def release_hotkey_modifiers(self) -> None:
        self.release_count += 1


class FakeRecorder:
    def __init__(self) -> None:
        self.is_recording = False
        self.start_count = 0

    def start(self) -> None:
        self.start_count += 1
        self.is_recording = True


class FakeHotkeyListener:
    def __init__(self) -> None:
        self.ignore_windows = []
        self.stop_count = 0

    def ignore_releases_for(self, seconds: float) -> None:
        self.ignore_windows.append(seconds)

    def stop(self) -> None:
        self.stop_count += 1


class FakeRoot:
    def __init__(self) -> None:
        self.quit_count = 0
        self.destroy_count = 0

    def quit(self) -> None:
        self.quit_count += 1

    def destroy(self) -> None:
        self.destroy_count += 1


class FakeEngine:
    def __init__(self, is_loaded: bool = False) -> None:
        self.is_loaded = is_loaded
        self.unload_count = 0

    def unload(self) -> None:
        self.unload_count += 1


class FakeThread:
    def __init__(self, target, daemon: bool = False, args=()) -> None:
        self.target = target
        self.daemon = daemon
        self.args = args
        self.started = False

    def start(self) -> None:
        self.started = True


class AppStateTests(unittest.TestCase):
    def test_model_state_change_refreshes_tray_and_notice(self):
        app = SpeechApp.__new__(SpeechApp)
        app.tray = FakeTray()
        app.overlay = FakeOverlay()

        app._model_state_changed("Parakeet loaded")

        self.assertEqual(app.tray.refresh_count, 1)
        self.assertEqual(app.overlay.notices, ["Parakeet loaded"])

    def test_begin_recording_releases_hotkey_modifiers_without_remembering_window(self):
        app = SpeechApp.__new__(SpeechApp)
        app.settings = AppSettings()
        app.recorder = FakeRecorder()
        app.overlay = FakeOverlay()
        app.tray = None
        app.transcribing = False
        app.system = FakeSystem()
        app.hotkey_listener = FakeHotkeyListener()

        app._begin_recording()

        self.assertEqual(app.system.remember_count, 0)
        self.assertEqual(app.system.release_count, 1)
        self.assertEqual(app.hotkey_listener.ignore_windows, [])
        self.assertEqual(app.recorder.start_count, 1)
        self.assertEqual(app.overlay.recording_count, 1)

    def test_quit_unloads_model_before_destroying_root(self):
        app = SpeechApp.__new__(SpeechApp)
        app.hotkey_listener = FakeHotkeyListener()
        app.tray = FakeTray()
        app.root = FakeRoot()
        app.engine = FakeEngine()

        app._quit_ui()

        self.assertEqual(app.engine.unload_count, 1)
        self.assertEqual(app.hotkey_listener.stop_count, 1)
        self.assertEqual(app.tray.stop_count, 1)
        self.assertEqual(app.root.quit_count, 1)
        self.assertEqual(app.root.destroy_count, 1)

    def test_status_text_reports_loading_while_model_loads(self):
        app = SpeechApp.__new__(SpeechApp)
        app.settings = AppSettings()
        app.engine = FakeEngine(is_loaded=False)
        app.model_loading = True

        # Status text carries the active model label and the loading state.
        self.assertIn(app.current_model_label(), app.status_text())
        self.assertIn("loading", app.status_text())

    def test_load_model_background_marks_loading_before_worker_starts(self):
        app = SpeechApp.__new__(SpeechApp)
        app.settings = AppSettings()
        app.engine = FakeEngine(is_loaded=False)
        app.model_loading = False
        app.posted_callbacks = []
        app.post_ui = app.posted_callbacks.append
        app._write_runtime_state = lambda *_args, **_kwargs: None

        with patch("speech_app.app.threading.Thread", FakeThread):
            app.load_model_background()

        self.assertTrue(app.model_loading)
        self.assertIn(app.current_model_label(), app.status_text())
        self.assertIn("loading", app.status_text())
        self.assertEqual(len(app.posted_callbacks), 1)

    def test_load_model_background_does_not_start_duplicate_worker_while_loading(self):
        app = SpeechApp.__new__(SpeechApp)
        app.settings = AppSettings()
        app.engine = FakeEngine(is_loaded=False)
        app.model_loading = True
        app.posted_callbacks = []
        app.post_ui = app.posted_callbacks.append

        with patch("speech_app.app.threading.Thread") as thread_cls:
            app.load_model_background()

        thread_cls.assert_not_called()

    def test_save_settings_keeps_remote_api_fields(self):
        """The Sens sync payload must not wipe the OpenRouter key."""

        class FakeStore:
            def __init__(self) -> None:
                self.saved: AppSettings | None = None

            def save(self, settings: AppSettings) -> None:
                self.saved = settings

        app = SpeechApp.__new__(SpeechApp)
        app.settings = AppSettings(model="remote")
        app.settings_store = FakeStore()
        values = {
            "model": "remote",
            "remote_api_key": "sk-or-v1-test",
            "remote_base_url": "https://openrouter.ai/api/v1",
            "remote_model_id": "openai/gpt-4o-mini-transcribe",
            "engine_enabled": True,
            "copy_to_clipboard": False,
            "paste_to_active_input": False,
            "preload_model": False,
            "device": "cpu",
            "backend": "auto",
            "hotkey": "ctrl+win",
        }

        app.save_settings_values(values)

        self.assertEqual(app.settings.remote_api_key, "sk-or-v1-test")
        self.assertEqual(app.settings.remote_base_url, "https://openrouter.ai/api/v1")
        self.assertEqual(
            app.settings.remote_model_id, "openai/gpt-4o-mini-transcribe"
        )
        assert app.settings_store.saved is not None
        self.assertEqual(
            app.settings_store.saved.remote_api_key, "sk-or-v1-test"
        )
        # Absent keys keep their current values (old callers predate the fields).
        app.settings.remote_api_key = "sk-or-v1-kept"
        app.save_settings_values(
            {
                "engine_enabled": True,
                "copy_to_clipboard": False,
                "paste_to_active_input": False,
                "preload_model": False,
                "device": "cpu",
                "backend": "auto",
                "hotkey": "ctrl+win",
            }
        )
        self.assertEqual(app.settings.remote_api_key, "sk-or-v1-kept")


if __name__ == "__main__":
    unittest.main()
