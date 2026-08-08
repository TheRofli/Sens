import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from speech_app.engines.base import EngineUnavailable
from speech_app.engines.sherpa import SherpaEngine
from speech_app.models import get_preset
from speech_app.settings import AppSettings


class _Recognizer:
    def __init__(self):
        self.stream = SimpleNamespace(result=SimpleNamespace(text=" распознано "))

    def create_stream(self):
        self.stream.accept_waveform = mock.Mock()
        return self.stream

    def decode_stream(self, stream):
        self.decoded = stream


class SherpaEngineTests(unittest.TestCase):
    def _make_model_files(self, root: Path, key: str) -> None:
        for relative in get_preset(key).required_files:
            path = root / key / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")

    def test_qwen_load_uses_cpu_and_adaptive_threads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_model_files(root, "qwen")
            recognizer = _Recognizer()
            factory = mock.Mock(return_value=recognizer)
            module = SimpleNamespace(
                OfflineRecognizer=SimpleNamespace(from_qwen3_asr=factory)
            )
            with mock.patch.dict(
                sys.modules, {"sherpa_onnx": module}
            ), mock.patch.dict("os.environ", {"SPEECH_MODELS_DIR": tmp}), mock.patch(
                "speech_app.engines.sherpa.inference_threads", return_value=7
            ):
                engine = SherpaEngine("qwen")
                engine.load(AppSettings(model="qwen"))

            kwargs = factory.call_args.kwargs
            self.assertEqual(kwargs["num_threads"], 7)
            self.assertEqual(kwargs["provider"], "cpu")
            self.assertEqual(engine.model_id, "Qwen/Qwen3-ASR-0.6B")

    def test_gigaam_load_and_transcribe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_model_files(root, "gigaam")
            recognizer = _Recognizer()
            factory = mock.Mock(return_value=recognizer)
            module = SimpleNamespace(
                OfflineRecognizer=SimpleNamespace(from_transducer=factory)
            )
            with mock.patch.dict(
                sys.modules, {"sherpa_onnx": module}
            ), mock.patch.dict("os.environ", {"SPEECH_MODELS_DIR": tmp}):
                engine = SherpaEngine("gigaam")
                text = engine.transcribe(
                    np.ones(160, dtype=np.float32),
                    16_000,
                    AppSettings(model="gigaam"),
                )

            self.assertEqual(text, "распознано")
            factory.assert_called_once()
            recognizer.stream.accept_waveform.assert_called_once()

    def test_missing_model_file_is_actionable(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            "os.environ", {"SPEECH_MODELS_DIR": tmp}
        ), mock.patch.dict(
            sys.modules,
            {"sherpa_onnx": SimpleNamespace(OfflineRecognizer=SimpleNamespace())},
        ):
            with self.assertRaisesRegex(EngineUnavailable, "not installed"):
                SherpaEngine("qwen").load(AppSettings(model="qwen"))

    def test_qwen_long_audio_is_split_before_decoder_context_limit(self):
        class ChunkRecognizer:
            def __init__(self):
                self.waveform_sizes = []
                self.texts = iter(["один два три", "два три четыре", "пять"])

            def create_stream(self):
                stream = SimpleNamespace(
                    result=SimpleNamespace(text=next(self.texts)),
                    accept_waveform=mock.Mock(),
                )
                original = stream.accept_waveform

                def accept(rate, audio):
                    self.waveform_sizes.append(len(audio))
                    original(rate, audio)

                stream.accept_waveform = accept
                return stream

            def decode_stream(self, stream):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_model_files(root, "qwen")
            recognizer = ChunkRecognizer()
            factory = mock.Mock(return_value=recognizer)
            module = SimpleNamespace(
                OfflineRecognizer=SimpleNamespace(from_qwen3_asr=factory)
            )
            with mock.patch.dict(
                sys.modules, {"sherpa_onnx": module}
            ), mock.patch.dict("os.environ", {"SPEECH_MODELS_DIR": tmp}):
                text = SherpaEngine("qwen").transcribe(
                    np.ones(60 * 16_000, dtype=np.float32),
                    16_000,
                    AppSettings(model="qwen"),
                )

        self.assertEqual(text, "один два три четыре пять")
        self.assertEqual(len(recognizer.waveform_sizes), 3)
        self.assertTrue(all(size <= 25 * 16_000 for size in recognizer.waveform_sizes))


if __name__ == "__main__":
    unittest.main()
