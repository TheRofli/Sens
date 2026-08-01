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
from .base import EngineUnavailable
from .parakeet import _write_temp_wav


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
        wav_path = _write_temp_wav(audio, sample_rate)
        try:
            text = self._model.transcribe(str(wav_path))
        finally:
            try:
                wav_path.unlink()
            except OSError:
                pass
        return str(text or "").strip()
