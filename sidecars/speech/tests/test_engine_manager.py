import unittest
from unittest import mock

import numpy as np

from speech_app import engine_manager
from speech_app.engine_manager import EngineManager
from speech_app.engines.base import EngineUnavailable
from speech_app.settings import AppSettings


class _FakeEngine:
    """Minimal engine double that records calls."""

    def __init__(self, kind: str):
        self.kind = kind
        self._loaded = False
        self.model_id = f"{kind}-model-id"
        self.load_calls = 0
        self.unload_calls = 0

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self, settings):
        self.load_calls += 1
        self._loaded = True
        self.model_id = settings.model_id

    def unload(self):
        self.unload_calls += 1
        self._loaded = False

    def transcribe(self, samples, sample_rate, settings):
        return f"{self.kind}:{samples.size}"


class _SegmentEngine(_FakeEngine):
    def __init__(self, kind: str):
        super().__init__(kind)
        self.segment_calls = 0

    def transcribe_segments(self, samples, sample_rate, settings):
        self.segment_calls += 1
        return [
            {"start": 0.0, "end": 0.5, "text": " hello "},
            {"start": 0.5, "end": 1.0, "text": "world"},
        ]


class EngineManagerTests(unittest.TestCase):
    def _patch_make(self, mapping):
        def factory(kind):
            if kind not in mapping:
                mapping[kind] = _FakeEngine(kind)
            return mapping[kind]

        return mock.patch.object(engine_manager, "make_engine", side_effect=factory)

    def test_load_uses_qwen_engine_for_default_model(self):
        engines: dict[str, _FakeEngine] = {}
        manager = EngineManager()
        with self._patch_make(engines):
            manager.load(AppSettings(model="qwen"))
        self.assertEqual(manager.kind, "qwen")
        self.assertIn("qwen", engines)
        self.assertTrue(manager.is_loaded)

    def test_load_uses_whisper_engine_for_whisper_model(self):
        engines: dict[str, _FakeEngine] = {}
        manager = EngineManager()
        with self._patch_make(engines):
            manager.load(AppSettings(model="whisper"))
        self.assertEqual(manager.kind, "whisper")
        self.assertIn("whisper", engines)

    def test_switching_model_unloads_previous_engine(self):
        engines: dict[str, _FakeEngine] = {}
        manager = EngineManager()
        with self._patch_make(engines):
            manager.load(AppSettings(model="qwen"))
            qwen = engines["qwen"]
            self.assertEqual(qwen.load_calls, 1)
            manager.load(AppSettings(model="whisper"))
        self.assertEqual(qwen.unload_calls, 1)
        self.assertEqual(manager.kind, "whisper")

    def test_load_same_kind_keeps_engine_loaded(self):
        engines: dict[str, _FakeEngine] = {}
        manager = EngineManager()
        with self._patch_make(engines):
            manager.load(AppSettings(model="qwen"))
            manager.load(AppSettings(model="qwen"))
        self.assertEqual(engines["qwen"].load_calls, 1)

    def test_transcribe_routes_to_current_engine(self):
        engines: dict[str, _FakeEngine] = {}
        manager = EngineManager()
        samples = np.ones(160, dtype=np.float32)
        with self._patch_make(engines):
            text = manager.transcribe(samples, 16000, AppSettings(model="whisper"))
        self.assertEqual(text, "whisper:160")

    def test_transcribe_empty_samples_short_circuits(self):
        manager = EngineManager()
        text = manager.transcribe(np.array([], dtype=np.float32), 16000, AppSettings())
        self.assertEqual(text, "")

    def test_transcribe_with_segments_runs_timestamp_engine_once(self):
        whisper = _SegmentEngine("whisper")
        manager = EngineManager()
        samples = np.ones(16000, dtype=np.float32)
        with mock.patch.object(engine_manager, "make_engine", return_value=whisper):
            text, segments = manager.transcribe_with_segments(
                samples, 16000, AppSettings(model="whisper")
            )

        self.assertEqual(text, "hello world")
        self.assertEqual(len(segments or []), 2)
        self.assertEqual(whisper.segment_calls, 1)

    def test_unload_clears_state(self):
        engines: dict[str, _FakeEngine] = {}
        manager = EngineManager()
        with self._patch_make(engines):
            manager.load(AppSettings(model="qwen"))
            manager.unload()
        self.assertFalse(manager.is_loaded)
        self.assertEqual(manager.kind, "")


if __name__ == "__main__":
    unittest.main()
