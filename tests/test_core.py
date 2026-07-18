import json
import tempfile
import unittest
from pathlib import Path

from speech_app.history import TranscriptHistory
from speech_app.output import TranscriptPublisher
from speech_app.runtime_state import write_runtime_state
from speech_app.settings import AppSettings, SettingsStore


class SettingsStoreTests(unittest.TestCase):
    def test_loads_defaults_when_file_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SettingsStore(Path(tmp) / "settings.json")

            settings = store.load()

            self.assertEqual(settings.model, "parakeet")
            self.assertEqual(settings.model_id, "nvidia/parakeet-tdt-0.6b-v3")
            self.assertEqual(settings.device, "cpu")
            self.assertEqual(settings.hotkey, "ctrl+win")
            self.assertTrue(settings.copy_to_clipboard)
            self.assertTrue(settings.paste_to_active_input)
            self.assertTrue(settings.preload_model)
            # Quality params have documented defaults.
            self.assertEqual(settings.beam_size, 5)
            self.assertEqual(settings.temperature, 0.0)
            self.assertTrue(settings.postprocess_text)

    def test_round_trips_known_settings_and_ignores_unknown_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            path.write_text(
                json.dumps(
                    {
                        "device": "cuda",
                        "backend": "nemo",
                        "engine_enabled": False,
                        "unknown_future_field": "kept out",
                    }
                ),
                encoding="utf-8",
            )
            store = SettingsStore(path)

            settings = store.load()
            store.save(settings)
            reloaded = json.loads(path.read_text(encoding="utf-8"))

            self.assertEqual(settings.device, "cuda")
            self.assertEqual(settings.backend, "nemo")
            self.assertFalse(settings.engine_enabled)
            self.assertNotIn("unknown_future_field", reloaded)


class TranscriptHistoryTests(unittest.TestCase):
    def test_add_records_newest_entries_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = TranscriptHistory(Path(tmp) / "history.jsonl", max_entries=3)

            first = history.add("first")
            second = history.add("second")

            entries = history.list()
            self.assertEqual([entry.text for entry in entries], ["second", "first"])
            self.assertNotEqual(first.id, second.id)

    def test_history_is_trimmed_to_max_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = TranscriptHistory(Path(tmp) / "history.jsonl", max_entries=2)

            history.add("one")
            history.add("two")
            history.add("three")

            self.assertEqual([entry.text for entry in history.list()], ["three", "two"])


class TranscriptPublisherTests(unittest.TestCase):
    def test_publish_saves_to_clipboard_history_and_pastes_when_enabled(self):
        clipboard = []
        pasted = []
        with tempfile.TemporaryDirectory() as tmp:
            history = TranscriptHistory(Path(tmp) / "history.jsonl")
            publisher = TranscriptPublisher(
                history=history,
                set_clipboard=clipboard.append,
                paste_active_input=lambda text: pasted.append(text),
            )

            entry = publisher.publish(" hello world ", AppSettings())

            self.assertEqual(entry.text, "hello world")
            self.assertEqual(clipboard, ["hello world"])
            self.assertEqual(pasted, ["hello world"])
            self.assertEqual(history.list()[0].text, "hello world")

    def test_publish_skips_empty_transcripts(self):
        clipboard = []
        with tempfile.TemporaryDirectory() as tmp:
            history = TranscriptHistory(Path(tmp) / "history.jsonl")
            publisher = TranscriptPublisher(
                history=history,
                set_clipboard=clipboard.append,
                paste_active_input=lambda text: None,
            )

            entry = publisher.publish("   ", AppSettings())

            self.assertIsNone(entry)
            self.assertEqual(clipboard, [])
            self.assertEqual(history.list(), [])


class RuntimeStateTests(unittest.TestCase):
    def test_runtime_state_records_model_loading_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runtime_state.json"

            write_runtime_state(
                "loading",
                AppSettings(device="cpu", backend="auto"),
                path=path,
            )

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(payload["running"])
            self.assertEqual(payload["model_state"], "loading")
            self.assertEqual(payload["device"], "cpu")
            self.assertEqual(payload["backend"], "auto")


if __name__ == "__main__":
    unittest.main()
