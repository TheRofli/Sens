"""Shared engine interface and types.

Concrete engines live in :mod:`speech_app.engines.sherpa` and
:mod:`speech_app.engines.whisper`. The :class:`SpeechEngine` protocol keeps them
interchangeable for :class:`~speech_app.engine_manager.EngineManager`.
"""

from __future__ import annotations

from dataclasses import dataclass
import tempfile
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from ..settings import AppSettings


class EngineUnavailable(RuntimeError):
    """Raised when a backend's optional dependencies are missing or unusable."""


def write_temp_wav(samples: np.ndarray, sample_rate: int) -> Path:
    descriptor, name = tempfile.mkstemp(prefix="sens-hearing-", suffix=".wav")
    import os

    os.close(descriptor)
    path = Path(name)
    pcm = np.clip(np.asarray(samples).reshape(-1), -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm.tobytes())
    return path


@dataclass(slots=True)
class LoadedEngine:
    """Describes a currently loaded model instance.

    Kept for compatibility with the previous single-engine API; engines set
    these fields so existing callers (status text, debug) keep working.
    """

    backend: str
    device: str
    model_id: str
    model: Any
    processor: Any | None = None


@runtime_checkable
class SpeechEngine(Protocol):
    """Minimal contract every ASR engine satisfies."""

    @property
    def is_loaded(self) -> bool: ...

    @property
    def model_id(self) -> str: ...

    def load(self, settings: "AppSettings") -> None: ...

    def unload(self) -> None: ...

    def transcribe(
        self, samples: np.ndarray, sample_rate: int, settings: "AppSettings"
    ) -> str: ...
