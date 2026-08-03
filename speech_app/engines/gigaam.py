"""GigaAM v3 e2e engine (Sber, Transformers backend).

GigaAM v3 is a Conformer-based Russian-first ASR (220-240M params, 700k hours
of Russian speech). The ``e2e_rnnt`` revision is end-to-end: it emits
punctuated, text-normalised output directly, so no separate punctuation or
LM step is needed.

Two implementation notes:

* The model is remote code (``trust_remote_code``) and its ``transcribe()``
  takes a *file path*, not raw samples. We write the numpy buffer to a
  temporary WAV and pass that, exactly like the NeMo path in the parakeet
  engine.
* The local copy under ``models/gigaam/e2e_rnnt`` ships a small patch of
  ``modeling_gigaam.py`` so it works with transformers v5: the feature
  extractor is built on CPU (torchaudio's MelSpectrogram calls ``.item()``
  during ``__init__`` and crashes under transformers v5's meta-device init
  context), ``load_audio`` uses soundfile instead of an ffmpeg subprocess,
  and ``all_tied_weights_keys`` is provided for v5's loading finalizer.
"""

from __future__ import annotations

import gc
from typing import Any

import numpy as np

from ..settings import AppSettings
from ..vad import split_audio
from .base import EngineUnavailable
from .parakeet import _write_temp_wav

# GigaAM's own transcribe() rejects audio longer than LONGFORM_THRESHOLD
# (25 s in modeling_gigaam.py). Chunk below that limit and join the parts;
# the app's RMS VAD places cuts in silence gaps so words stay intact, and a
# small overlap lets the next chunk hear a cut word whole (the duplicate seam
# is removed by _merge_transcript_parts).
_MAX_CHUNK_SECONDS = 24.0
_MIN_CHUNK_SECONDS = 2.0
_OVERLAP_SECONDS = 1.5


def _normalise_word(word: str) -> str:
    return word.strip(".,!?;:…\"'()[]«»—-").lower()


def _merge_transcript_parts(parts: list[str]) -> str:
    """Join chunk transcripts, dropping the duplicated overlap region.

    Consecutive chunks share an audio overlap, so the same words appear at
    the end of one part and the start of the next. The longest matching
    suffix/prefix (case- and punctuation-insensitive) is removed from the
    join; at least two words must match so a naturally repeated word is not
    eaten. A word mangled differently at the seam (rare) is left as-is.
    """
    merged = ""
    for part in parts:
        if not part:
            continue
        if not merged:
            merged = part
            continue
        a_words = merged.split()
        b_words = part.split()
        max_match = min(len(a_words), len(b_words))
        match = 0
        for size in range(max_match, 0, -1):
            if all(
                _normalise_word(left) == _normalise_word(right)
                for left, right in zip(a_words[-size:], b_words[:size])
            ):
                match = size
                break
        if match >= 2:
            merged = " ".join(a_words[:-match]) + " " + part
        else:
            merged += " " + part
    return merged.strip()


class GigaAMEngine:
    """GigaAM v3 e2e ASR via Transformers, CPU-only."""

    def __init__(self) -> None:
        self._model: Any = None
        self._model_id: str = ""
        self._model_dir: Any = None

    # -- SpeechEngine-compatible API ---------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_id(self) -> str:
        return self._model_id

    def unload(self) -> None:
        self._model = None
        self._model_id = ""
        self._model_dir = None
        gc.collect()

    def load(self, settings: AppSettings) -> None:
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise EngineUnavailable(
                "transformers is not installed. Run install.ps1 first."
            ) from exc

        from ..models import get_preset
        from .paths import gigaam_model_dir

        preset = get_preset(settings.model)
        model_dir = gigaam_model_dir(preset)
        if not (model_dir / "pytorch_model.bin").is_file():
            raise EngineUnavailable(
                f"GigaAM model is not installed at {model_dir}. "
                "Run: speech model install gigaam"
            )

        try:
            # Local copy with the transformers-v5 compatibility patch; see the
            # module docstring. CPU is the intended device (no CUDA needed).
            self._model = AutoModel.from_pretrained(
                str(model_dir),
                trust_remote_code=True,
            )
        except Exception as exc:
            raise EngineUnavailable(f"Failed to load GigaAM model: {exc}") from exc
        self._model_id = preset.model_id
        self._model_dir = model_dir

    def transcribe(
        self, samples: np.ndarray, sample_rate: int, settings: AppSettings
    ) -> str:
        if samples.size == 0:
            return ""
        if self._model is None:
            self.load(settings)
            assert self._model is not None

        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        parts = []
        for chunk in split_audio(
            audio,
            sample_rate,
            max_duration_s=_MAX_CHUNK_SECONDS,
            sensitivity=settings.vad_sensitivity,
            min_chunk_s=_MIN_CHUNK_SECONDS,
            overlap_s=_OVERLAP_SECONDS,
        ):
            wav_path = _write_temp_wav(chunk, sample_rate)
            try:
                text = self._model.transcribe(str(wav_path))
            finally:
                try:
                    wav_path.unlink()
                except OSError:
                    pass
            if text:
                parts.append(str(text).strip())
        return _merge_transcript_parts(parts)
