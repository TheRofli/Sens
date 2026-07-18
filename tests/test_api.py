import json
import os
import tempfile
import time
import unittest
import urllib.error
import urllib.request

from speech_app.api import SpeechAPIServer
from speech_app.settings import AppSettings


class FakeApp:
    """In-memory app double for API tests."""

    def __init__(self):
        self.settings = AppSettings()
        self.copied: list[str] = []
        self.set_model_calls: list[str] = []
        self.load_calls = 0
        self.unload_calls = 0
        self.transcribing = False

    # settings
    def get_settings_values(self):
        s = self.settings
        return {
            "model": s.model,
            "engine_enabled": s.engine_enabled,
            "copy_to_clipboard": s.copy_to_clipboard,
            "paste_to_active_input": s.paste_to_active_input,
            "preload_model": s.preload_model,
            "device": s.device,
            "backend": s.backend,
            "hotkey": s.hotkey,
            "beam_size": s.beam_size,
            "vad_sensitivity": s.vad_sensitivity,
            "postprocess_text": s.postprocess_text,
        }

    def save_settings_values(self, values):
        for key in ("device", "backend", "hotkey", "model"):
            if key in values:
                setattr(self.settings, key, str(values[key]))
        for key in (
            "engine_enabled",
            "copy_to_clipboard",
            "paste_to_active_input",
            "preload_model",
            "postprocess_text",
        ):
            if key in values:
                setattr(self.settings, key, bool(values[key]))
        if "beam_size" in values:
            self.settings.beam_size = int(values["beam_size"])

    # status
    def engine_enabled(self):
        return self.settings.engine_enabled

    def _model_state_label(self):
        return "unloaded"

    def current_model(self):
        return self.settings.model

    def current_model_label(self):
        return self.settings.model

    def model_loaded(self):
        return False

    def model_is_loading(self):
        return False

    def current_device(self):
        return self.settings.device

    def current_backend(self):
        return self.settings.backend

    def status_text(self):
        return "ok"

    def model_status(self):
        from speech_app.model_status import find_model_status
        from pathlib import Path

        return find_model_status(Path(os.environ.get("HF_HOME", "/tmp")), "x/y")

    # actions
    def available_models(self):
        return [
            {"key": "parakeet", "label": "Parakeet", "engine": "parakeet",
             "installed": True, "size_label": "1 GB", "active": True},
            {"key": "whisper-ru", "label": "Whisper RU", "engine": "whisper",
             "installed": False, "size_label": "Not installed", "active": False},
        ]

    def set_model(self, key):
        self.set_model_calls.append(key)
        self.settings.model = key

    def load_model_background(self):
        self.load_calls += 1

    def unload_model(self):
        self.unload_calls += 1

    def history_rows(self):
        return [("id1", "hello world"), ("id2", "second transcript")]

    def copy_history_entry(self, entry_id):
        self.copied.append(entry_id)

    def copy_last_transcript(self):
        self.copied.append("LAST")

    # UI threading helper: execute synchronously (tests run single-threaded).
    def post_ui_sync(self, callback, timeout=5.0):
        return callback()


class SpeechAPIServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        os.environ["SPEECH_DATA_DIR"] = cls._tmp.name
        cls.app = FakeApp()
        cls.server = SpeechAPIServer(cls.app)
        cls.server.start()
        # Give the daemon a moment to bind.
        time.sleep(0.1)
        cls.base = f"http://127.0.0.1:{cls.server.port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()
        del os.environ["SPEECH_DATA_DIR"]
        cls._tmp.cleanup()

    def _get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=3) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))

    def _post(self, path, body):
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.base + path, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))

    def test_status_endpoint_returns_state(self):
        status, body = self._get("/api/status")
        self.assertEqual(status, 200)
        self.assertTrue(body["running"])
        # model mirrors whatever the (shared) app currently holds; the exact
        # value depends on test execution order, so assert consistency.
        self.assertEqual(body["model"], self.app.settings.model)

    def test_get_settings_returns_defaults(self):
        status, body = self._get("/api/settings")
        self.assertEqual(status, 200)
        self.assertEqual(body["model"], "parakeet")
        self.assertEqual(body["beam_size"], 5)

    def test_post_settings_merges_and_applies(self):
        status, body = self._post("/api/settings", {"beam_size": 8, "device": "cuda"})
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertEqual(self.app.settings.beam_size, 8)
        self.assertEqual(self.app.settings.device, "cuda")

    def test_get_models_lists_presets(self):
        status, body = self._get("/api/models")
        self.assertEqual(status, 200)
        keys = [item["key"] for item in body]
        self.assertIn("parakeet", keys)
        self.assertIn("whisper-ru", keys)

    def test_post_model_selects_preset(self):
        try:
            status, body = self._post("/api/model", {"key": "whisper-ru"})
            self.assertEqual(status, 200)
            self.assertEqual(self.app.settings.model, "whisper-ru")
            self.assertEqual(self.app.set_model_calls, ["whisper-ru"])
        finally:
            # Restore default so test ordering does not bleed into other tests.
            self.app.settings.model = "parakeet"

    def test_post_model_load_triggers_load(self):
        status, _ = self._post("/api/model/load", {})
        self.assertEqual(status, 200)
        self.assertGreater(self.app.load_calls, 0)

    def test_post_model_unload_triggers_unload(self):
        status, _ = self._post("/api/model/unload", {})
        self.assertEqual(status, 200)
        self.assertGreater(self.app.unload_calls, 0)

    def test_history_endpoint_supports_search(self):
        status, body = self._get("/api/history?q=second")
        self.assertEqual(status, 200)
        self.assertEqual(len(body), 1)
        self.assertIn("second", body[0]["text"])

    def test_history_copy_calls_app(self):
        status, _ = self._post("/api/history/copy", {"id": "id2"})
        self.assertEqual(status, 200)
        self.assertIn("id2", self.app.copied)

    def test_unknown_path_returns_404(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(self.base + "/api/nope", timeout=3)
        self.assertEqual(ctx.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
