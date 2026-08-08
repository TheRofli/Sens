"""CPU-only Qwen3-ASR and GigaAM engines through sherpa-onnx."""

from __future__ import annotations

import gc
from typing import Any

import numpy as np

from ..cpu import inference_threads
from ..longform import merge_overlapping_transcripts
from ..models import get_preset
from ..settings import AppSettings
from ..vad import split_audio
from .base import EngineUnavailable
from .paths import model_dir


class SherpaEngine:
    def __init__(self, kind: str) -> None:
        if kind not in {"qwen", "gigaam"}:
            raise ValueError(f"Unsupported sherpa engine: {kind}")
        self.kind = kind
        self._recognizer: Any = None
        self._model_id = ""

    @property
    def is_loaded(self) -> bool:
        return self._recognizer is not None

    @property
    def model_id(self) -> str:
        return self._model_id

    def unload(self) -> None:
        self._recognizer = None
        self._model_id = ""
        gc.collect()

    def load(self, settings: AppSettings) -> None:
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise EngineUnavailable(
                "sherpa-onnx is missing from the Sens Hearing runtime"
            ) from exc

        preset = get_preset(settings.model)
        if preset.engine != self.kind:
            raise EngineUnavailable(
                f"Preset {preset.key} does not use the {self.kind} engine"
            )
        root = model_dir(preset)
        missing = [name for name in preset.required_files if not (root / name).is_file()]
        if missing:
            raise EngineUnavailable(
                f"{preset.label} is not installed. Missing {missing[0]} in {root}"
            )
        threads = inference_threads()
        try:
            if self.kind == "qwen":
                self._recognizer = sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
                    conv_frontend=str(root / "conv_frontend.onnx"),
                    encoder=str(root / "encoder.int8.onnx"),
                    decoder=str(root / "decoder.int8.onnx"),
                    tokenizer=str(root / "tokenizer"),
                    num_threads=threads,
                    sample_rate=16_000,
                    feature_dim=128,
                    max_new_tokens=512,
                    provider="cpu",
                )
            else:
                self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                    encoder=str(root / "encoder.int8.onnx"),
                    decoder=str(root / "decoder.onnx"),
                    joiner=str(root / "joiner.onnx"),
                    tokens=str(root / "tokens.txt"),
                    num_threads=threads,
                    sample_rate=16_000,
                    feature_dim=80,
                    decoding_method="greedy_search",
                    provider="cpu",
                    model_type="nemo_transducer",
                )
        except Exception as exc:
            self._recognizer = None
            raise EngineUnavailable(f"Failed to load {preset.label}: {exc}") from exc
        self._model_id = preset.model_id

    def transcribe(
        self, samples: np.ndarray, sample_rate: int, settings: AppSettings
    ) -> str:
        if samples.size == 0:
            return ""
        if self._recognizer is None:
            self.load(settings)
        assert self._recognizer is not None
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if sample_rate != 16_000:
            from .whisper import _resample_linear

            audio = _resample_linear(audio, sample_rate, 16_000)
        if self.kind == "qwen":
            # The released decoder has a fixed 512-token context. Keep enough
            # room for generated text and use overlapping, silence-aware
            # chunks instead of letting sherpa truncate long input silently.
            chunks = split_audio(
                audio,
                sample_rate=16_000,
                max_duration_s=25.0,
                sensitivity=settings.vad_sensitivity,
                min_chunk_s=3.0,
                overlap_s=1.0,
            )
            return merge_overlapping_transcripts(
                [self._transcribe_once(chunk) for chunk in chunks]
            )
        return self._transcribe_once(audio)

    def _transcribe_once(self, audio: np.ndarray) -> str:
        assert self._recognizer is not None
        stream = self._recognizer.create_stream()
        stream.accept_waveform(16_000, audio)
        self._recognizer.decode_stream(stream)
        return str(stream.result.text or "").strip()
