"""Whisper engine via faster-whisper (CTranslate2).

Uses the converted CTranslate2 weights produced at install time (see
:mod:`speech_app.engines.install`). CPU + INT8 is the intended deployment:
fast, low-memory, and accurate enough for dictation.

Anti-hallucination knobs forwarded to ``transcribe``:

* ``condition_on_previous_text=False`` — Whisper otherwise re-feeds its own
  output into the next chunk, which is the main cause of looping phrases on
  silence;
* ``compression_ratio_threshold`` and ``log_prob_threshold`` discard segments
  that look like garbage;
* ``language=None`` keeps automatic detection so RU/EN code-switching works.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..models import get_preset
from ..settings import AppSettings
from .base import EngineUnavailable
from .paths import whisper_model_dir


class WhisperEngine:
    """Whisper ASR via faster-whisper, CPU + INT8."""

    def __init__(self) -> None:
        self._model: Any = None
        self._model_id: str = ""

    # -- SpeechEngine-compatible API ---------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_id(self) -> str:
        return self._model_id

    def unload(self) -> None:
        # faster-whisper's WhisperModel has no explicit close; dropping the
        # reference lets the CTranslate2 model be garbage collected.
        self._model = None
        self._model_id = ""
        import gc

        gc.collect()

    def load(self, settings: AppSettings) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise EngineUnavailable(
                "faster-whisper is not installed. Run install.ps1 or install "
                "requirements-whisper.txt, then `speech model install whisper-ru`."
            ) from exc

        preset = get_preset(settings.model)
        model_dir = whisper_model_dir(preset)
        if not model_dir.exists():
            raise EngineUnavailable(
                f"Whisper model is not installed at {model_dir}. "
                "Run: speech model install whisper-ru"
            )

        # CPU + INT8 is the documented sweet spot for faster-whisper on CPU.
        device = "cpu"
        compute_type = "int8"
        cpu_threads = _cpu_threads()
        try:
            self._model = WhisperModel(
                str(model_dir),
                device=device,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
            )
        except Exception as exc:
            raise EngineUnavailable(f"Failed to load Whisper model: {exc}") from exc
        self._model_id = preset.model_id

    def transcribe(
        self, samples: np.ndarray, sample_rate: int, settings: AppSettings
    ) -> str:
        if samples.size == 0:
            return ""
        if self._model is None:
            self.load(settings)
            assert self._model is not None

        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        # faster-whisper expects 16 kHz mono float32; resample defensively if
        # the recorder ever changes sample_rate away from the 16 kHz default.
        if sample_rate != 16000:
            audio = _resample_linear(audio, sample_rate, 16000)

        segments, _info = self._model.transcribe(
            audio,
            language=None,  # automatic detection (RU/EN code-switching)
            beam_size=settings.beam_size,
            temperature=settings.temperature,
            # Anti-hallucination: do not carry previous output forward.
            condition_on_previous_text=False,
            compression_ratio_threshold=settings.compression_ratio_threshold,
            log_prob_threshold=settings.log_prob_threshold,
            vad_filter=True,  # faster-whisper's built-in Silero VAD as a safety net
        )
        return " ".join(segment.text.strip() for segment in segments).strip()


def _cpu_threads() -> int:
    """Pick a thread count that leaves headroom for the UI thread.

    CTranslate2 defaults to using every logical core; on a desktop that makes
    the tray/tkinter UI janky during transcription. Cap at physical cores and
    leave at least one core free.
    """
    try:
        import os

        total = os.cpu_count() or 4
        return max(1, min(total - 1, 4))
    except Exception:
        return 2


def _resample_linear(
    audio: np.ndarray, from_rate: int, to_rate: int
) -> np.ndarray:
    """Crude linear resampler. faster-whisper only needs this if the recorder
    sample rate ever differs from 16 kHz; the app default is 16 kHz so this is
    a defensive fallback, not a hot path.
    """
    if from_rate == to_rate or audio.size == 0:
        return audio
    duration = audio.size / from_rate
    out_samples = int(round(duration * to_rate))
    if out_samples <= 0:
        return np.array([], dtype=np.float32)
    step = audio.size / out_samples
    indices = np.minimum(
        (np.arange(out_samples) * step).astype(np.int64), audio.size - 1
    )
    return audio[indices].astype(np.float32, copy=False)
