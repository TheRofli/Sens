"""Parakeet TDT engine (Transformers / NeMo backends).

Migrated from the previous monolithic ``parakeet_engine.py`` with two additions:

* quality parameters from :class:`~speech_app.settings.AppSettings`
  (``beam_size``, ``temperature``, ``repetition_penalty``,
  ``no_repeat_ngram_size``) are forwarded to ``model.generate``;
* ``language=None`` keeps automatic language detection (required for RU+EN
  code-switching), matching the prior behaviour.
"""

from __future__ import annotations

import gc
import tempfile
import wave
from typing import Any

import numpy as np

from ..settings import AppSettings
from .base import EngineUnavailable, LoadedEngine


class ParakeetEngine:
    """Parakeet TDT ASR via Transformers (primary) or NeMo (fallback)."""

    def __init__(self) -> None:
        self._loaded: LoadedEngine | None = None

    # -- SpeechEngine-compatible API ---------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._loaded is not None

    @property
    def model_id(self) -> str:
        return self._loaded.model_id if self._loaded is not None else ""

    def unload(self) -> None:
        self._loaded = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def load(self, settings: AppSettings) -> None:
        backend = settings.backend.lower()
        if backend not in {"auto", "transformers", "nemo"}:
            raise EngineUnavailable(f"Unsupported backend: {settings.backend}")

        device = self._resolve_device(settings.device)
        model_id = settings.model_id
        if backend in {"auto", "transformers"}:
            try:
                self._loaded = self._load_transformers(model_id, device)
                return
            except EngineUnavailable:
                if backend == "transformers":
                    raise
        self._loaded = self._load_nemo(model_id, device)

    def transcribe(
        self, samples: np.ndarray, sample_rate: int, settings: AppSettings
    ) -> str:
        if samples.size == 0:
            return ""

        loaded = self._loaded
        if (
            loaded is None
            or loaded.model_id != settings.model_id
            or loaded.device != self._resolve_device(settings.device)
            or (settings.backend != "auto" and loaded.backend != settings.backend)
        ):
            self.load(settings)
            loaded = self._loaded

        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        assert loaded is not None
        if loaded.backend == "transformers":
            return self._transcribe_transformers(audio, sample_rate, loaded, settings)
        return self._transcribe_nemo(audio, sample_rate, loaded)

    # -- internals ---------------------------------------------------------

    def _resolve_device(self, configured: str) -> str:
        device = configured.lower()
        if device == "auto":
            try:
                import torch

                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        if device in {"gpu", "cuda"}:
            return "cuda"
        return "cpu"

    def _load_transformers(self, model_id: str, device: str) -> LoadedEngine:
        try:
            import torch
            import transformers
            from transformers import AutoProcessor
        except ImportError as exc:
            raise EngineUnavailable(
                "Transformers backend is not installed. Run install.ps1 or "
                "install requirements-parakeet.txt."
            ) from exc

        AutoModelForTDT = getattr(transformers, "AutoModelForTDT", None)
        if AutoModelForTDT is None:
            raise EngineUnavailable(
                "This Transformers build does not include AutoModelForTDT. "
                "Install Transformers from source as in requirements-parakeet.txt."
            )

        if device == "cuda" and not torch.cuda.is_available():
            raise EngineUnavailable("CUDA was selected, but torch cannot see a GPU.")

        dtype = torch.float32 if device == "cpu" else torch.float16
        processor = AutoProcessor.from_pretrained(model_id)
        try:
            model = AutoModelForTDT.from_pretrained(model_id, dtype=dtype)
        except TypeError:
            model = AutoModelForTDT.from_pretrained(model_id, torch_dtype=dtype)
        model.to(device)
        model.eval()
        return LoadedEngine(
            backend="transformers",
            device=device,
            model_id=model_id,
            model=model,
            processor=processor,
        )

    def _transcribe_transformers(
        self,
        audio: np.ndarray,
        sample_rate: int,
        loaded: LoadedEngine,
        settings: AppSettings,
    ) -> str:
        import torch

        processor = loaded.processor
        model = loaded.model
        inputs = processor(
            [audio],
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        if hasattr(inputs, "to"):
            inputs = inputs.to(device=model.device, dtype=model.dtype)
        else:
            inputs = {
                key: value.to(device=model.device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }

        generate_kwargs: dict[str, Any] = {"return_dict_in_generate": True}
        # language=None preserves automatic detection (important for RU+EN
        # code-switching). Parakeet accepts ``language`` at generate time.
        generate_kwargs["language"] = None
        if settings.beam_size and settings.beam_size > 1:
            generate_kwargs["num_beams"] = settings.beam_size
        if settings.temperature and settings.temperature > 0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = settings.temperature
        if settings.repetition_penalty and settings.repetition_penalty != 1.0:
            generate_kwargs["repetition_penalty"] = settings.repetition_penalty
        if settings.no_repeat_ngram_size and settings.no_repeat_ngram_size > 0:
            generate_kwargs["no_repeat_ngram_size"] = settings.no_repeat_ngram_size

        with torch.inference_mode():
            output = model.generate(**inputs, **generate_kwargs)

        decoded = processor.decode(output.sequences, skip_special_tokens=True)
        return _normalize_decoded_text(decoded)

    def _load_nemo(self, model_id: str, device: str) -> LoadedEngine:
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError as exc:
            raise EngineUnavailable(
                "NeMo backend is not installed. On Windows the Transformers "
                "backend is usually the easier route."
            ) from exc

        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
        if device == "cuda":
            model = model.cuda()
        else:
            model = model.cpu()
        model.eval()
        return LoadedEngine(
            backend="nemo",
            device=device,
            model_id=model_id,
            model=model,
        )

    def _transcribe_nemo(
        self, audio: np.ndarray, sample_rate: int, loaded: LoadedEngine
    ) -> str:
        wav_path = _write_temp_wav(audio, sample_rate)
        try:
            output = loaded.model.transcribe([str(wav_path)])
        finally:
            try:
                wav_path.unlink()
            except OSError:
                pass
        if not output:
            return ""
        first = output[0]
        return getattr(first, "text", str(first)).strip()


def _normalize_decoded_text(decoded: Any) -> str:
    if isinstance(decoded, str):
        return decoded.strip()
    if isinstance(decoded, tuple) and decoded:
        return _normalize_decoded_text(decoded[0])
    if isinstance(decoded, list):
        return " ".join(str(item).strip() for item in decoded if str(item).strip())
    return str(decoded).strip()


def _write_temp_wav(audio: np.ndarray, sample_rate: int) -> Any:
    from pathlib import Path

    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    handle.close()
    path = Path(handle.name)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return path
