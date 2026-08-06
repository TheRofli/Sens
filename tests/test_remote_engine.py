import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

from speech_app.engines.base import EngineUnavailable
from speech_app.engines.remote import (
    DEFAULT_BASE_URL,
    RemoteEngine,
    _normalise_segments,
)
from speech_app.settings import AppSettings


def _sample_file(name: str = "sample.mp3") -> str:
    """Real file on disk; the engine uploads the path as-is."""
    path = Path(tempfile.gettempdir()) / name
    path.write_bytes(b"fake-audio-bytes")
    return str(path)


def _write_wav(path: Path) -> None:
    """Short tone with real energy, mono 16 kHz PCM."""
    samples = np.zeros(16000, dtype=np.float32)
    samples[500:9000] = 0.5
    pcm = (samples * 32767).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(pcm.tobytes())


def _settings(**overrides) -> AppSettings:
    defaults = {
        "model": "remote",
        "remote_api_key": "sk-or-test-key",
        "remote_base_url": DEFAULT_BASE_URL,
        "remote_model_id": "openai/gpt-4o-transcribe",
    }
    defaults.update(overrides)
    return AppSettings(**defaults)


class _FakeResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if isinstance(self._payload, dict) and "raw" in self._payload:
            raise ValueError("not json")
        return self._payload

    @property
    def text(self):
        return str(self._payload)


class RemoteEngineTests(unittest.TestCase):
    def test_transcribe_file_posts_multipart_to_audio_endpoint(self):
        engine = RemoteEngine()
        settings = _settings()
        with mock.patch("speech_app.engines.remote.requests.post") as post:
            post.return_value = _FakeResponse(
                200,
                {
                    "text": "Привет, мир",
                    "duration": 3.5,
                    "language": "ru",
                    "segments": [
                        {"id": 0, "start": 0.0, "end": 1.2, "text": "Привет,"},
                        {"id": 1, "start": 1.2, "end": 3.5, "text": "мир"},
                    ],
                },
            )
            result = engine.transcribe_file(_sample_file(), settings)

        self.assertEqual(result["text"], "Привет, мир")
        self.assertEqual(result["duration"], 3.5)
        self.assertEqual(
            result["segments"],
            [
                {"start": 0.0, "end": 1.2, "text": "Привет,"},
                {"start": 1.2, "end": 3.5, "text": "мир"},
            ],
        )
        url = post.call_args.args[0]
        self.assertEqual(url, f"{DEFAULT_BASE_URL}/audio/transcriptions")
        headers = post.call_args.kwargs["headers"]
        self.assertEqual(headers["Authorization"], "Bearer sk-or-test-key")
        form = post.call_args.kwargs["data"]
        self.assertEqual(form["model"], "openai/gpt-4o-transcribe")
        self.assertEqual(form["response_format"], "verbose_json")
        files = post.call_args.kwargs["files"]
        self.assertEqual(files["file"][0], "sample.mp3")

    def test_missing_api_key_raises_actionable_error(self):
        engine = RemoteEngine()
        with mock.patch("speech_app.engines.remote.requests.post") as post:
            with self.assertRaises(EngineUnavailable) as raised:
                engine.transcribe_file(_sample_file(), _settings(remote_api_key=""))
        self.assertIn("ключ не задан", str(raised.exception))
        post.assert_not_called()

    def test_status_codes_map_to_actionable_errors(self):
        engine = RemoteEngine()
        settings = _settings()
        cases = [
            (401, "отверг API ключ"),
            (402, "недостаточно средств"),
            (404, "не найдена"),
        ]
        for status, expected in cases:
            with self.subTest(status=status), mock.patch(
                "speech_app.engines.remote.requests.post"
            ) as post:
                post.return_value = _FakeResponse(
                    status, {"error": {"message": "detail"}}
                )
                with self.assertRaises(EngineUnavailable) as raised:
                    engine.transcribe_file(_sample_file(), settings)
                self.assertIn(expected, str(raised.exception))

    def test_non_json_success_raises(self):
        engine = RemoteEngine()
        with mock.patch("speech_app.engines.remote.requests.post") as post:
            post.return_value = _FakeResponse(200, {"raw": True})
            with self.assertRaises(EngineUnavailable):
                engine.transcribe_file(_sample_file(), _settings())

    def test_network_failure_raises_actionable_error(self):
        import requests

        engine = RemoteEngine()
        with mock.patch("speech_app.engines.remote.requests.post") as post:
            post.side_effect = requests.ConnectionError("connection refused")
            with self.assertRaises(EngineUnavailable) as raised:
                engine.transcribe_file(_sample_file(), _settings())
        self.assertIn("connection refused", str(raised.exception))

    def test_transcribe_with_samples_uploads_temp_wav(self):
        engine = RemoteEngine()
        samples = np.zeros(16000, dtype=np.float32)
        samples[100:500] = 0.5
        with mock.patch("speech_app.engines.remote.requests.post") as post:
            post.return_value = _FakeResponse(
                200, {"text": "тест", "duration": 1.0, "segments": []}
            )
            text = engine.transcribe(samples, 16000, _settings())
        self.assertEqual(text, "тест")
        uploaded = post.call_args.kwargs["files"]["file"][0]
        self.assertTrue(uploaded.endswith(".wav"))

    def test_empty_samples_return_empty_without_network(self):
        engine = RemoteEngine()
        with mock.patch("speech_app.engines.remote.requests.post") as post:
            self.assertEqual(engine.transcribe(np.zeros(0), 16000, _settings()), "")
            post.assert_not_called()

    def test_engine_stays_loaded_without_download(self):
        engine = RemoteEngine()
        engine.load(_settings())
        self.assertTrue(engine.is_loaded)
        self.assertEqual(engine.model_id, "openai/gpt-4o-transcribe")
        engine.unload()
        self.assertFalse(engine.is_loaded)

    def test_normalise_segments_filters_and_rounds(self):
        raw = [
            {"id": 0, "start": 0.123, "end": 1.456, "text": "  один  "},
            {"id": 1, "start": 1.5, "end": 2.0, "text": "   "},
            {"id": 2, "start": 2.0, "end": 3.0, "text": "два", "tokens": []},
            "not-a-dict",
        ]
        self.assertEqual(
            _normalise_segments(raw),
            [
                {"start": 0.12, "end": 1.46, "text": "один"},
                {"start": 2.0, "end": 3.0, "text": "два"},
            ],
        )
        self.assertIsNone(_normalise_segments(None))

    def test_transcribe_audio_file_uses_remote_branch(self):
        from speech_app.engine_manager import EngineManager
        from speech_app.transcription import transcribe_audio_file

        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "voice.wav"
            _write_wav(wav)
            engine = EngineManager()
            settings = _settings()
            with mock.patch("speech_app.engines.remote.requests.post") as post:
                post.return_value = _FakeResponse(
                    200,
                    {
                        "text": "Привет",
                        "duration": 1.25,
                        "language": "ru",
                        "segments": [
                            {"id": 0, "start": 0.0, "end": 1.25, "text": "Привет"}
                        ],
                    },
                )
                result = transcribe_audio_file(
                    str(wav), settings=settings, engine=engine
                )
            self.assertEqual(result["text"], "Привет")
            self.assertEqual(result["model"], "openai/gpt-4o-transcribe")
            self.assertEqual(result["engine"], "remote")
            self.assertEqual(result["duration_seconds"], 1.25)
            self.assertEqual(result["container"], "audio")
            self.assertTrue(result["audioTrack"])
            self.assertEqual(
                result["segments"], [{"start": 0.0, "end": 1.25, "text": "Привет"}]
            )
            # The file was uploaded as-is: no local decode/resample happened.
            uploaded = post.call_args.kwargs["files"]["file"][0]
            self.assertEqual(uploaded, "voice.wav")

    def test_transcribe_audio_file_local_engine_skips_remote_branch(self):
        from speech_app import engine_manager as engine_manager_module
        from speech_app.engine_manager import EngineManager
        from speech_app.transcription import transcribe_audio_file

        class _LocalFake:
            kind = "parakeet"

            @property
            def is_loaded(self) -> bool:
                return True

            @property
            def model_id(self) -> str:
                return "fake-parakeet"

            def load(self, settings) -> None:
                pass

            def unload(self) -> None:
                pass

            def transcribe(self, samples, sample_rate, settings) -> str:
                return ""

        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "voice.wav"
            _write_wav(wav)
            engine = EngineManager()
            settings = AppSettings(model="parakeet", remote_api_key="")
            with mock.patch.object(
                engine_manager_module, "make_engine", return_value=_LocalFake()
            ):
                result = transcribe_audio_file(
                    str(wav), settings=settings, engine=engine
                )
            # Falls through to the local parakeet path (no network anywhere).
            self.assertEqual(result["engine"], "parakeet")
            self.assertEqual(result["text"], "")


if __name__ == "__main__":
    unittest.main()
