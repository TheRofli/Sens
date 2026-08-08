"""Engine manager: selects and drives the active ASR engine.

Holds a single loaded engine at a time. When the selected model changes
(:class:`~speech_app.settings.AppSettings.model`) the next ``transcribe`` or
explicit ``load`` swaps to the right engine kind, unloading the previous one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .engines.base import EngineUnavailable, SpeechEngine
from .engines.remote import RemoteEngine
from .engines.sherpa import SherpaEngine
from .engines.whisper import WhisperEngine
from .models import resolve_engine

if TYPE_CHECKING:
    from .settings import AppSettings


def make_engine(kind: str) -> SpeechEngine:
    """Instantiate the concrete engine for ``kind``
    ("qwen" | "gigaam" | "whisper" | "remote")."""
    if kind == "whisper":
        return WhisperEngine()
    if kind in {"qwen", "gigaam"}:
        return SherpaEngine(kind)
    if kind == "remote":
        return RemoteEngine()
    raise EngineUnavailable(f"Unsupported Hearing engine: {kind}")


class EngineManager:
    """Owns the currently loaded engine and routes calls to it."""

    def __init__(self) -> None:
        self._current: SpeechEngine | None = None
        self._kind: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._current is not None and self._current.is_loaded

    @property
    def model_id(self) -> str:
        return self._current.model_id if self._current is not None else ""

    @property
    def kind(self) -> str:
        """Engine kind of the currently loaded engine."""
        return self._kind or ""

    def load(self, settings: "AppSettings") -> None:
        kind = resolve_engine(settings)
        # If the right engine is already loaded and healthy, keep it.
        if self._current is not None and self._kind == kind and self._current.is_loaded:
            return
        # Switching engine kind requires unloading the previous one.
        if self._current is not None and self._kind != kind:
            self.unload()
        if self._current is None:
            self._current = make_engine(kind)
            self._kind = kind
        assert self._current is not None
        self._current.load(settings)

    def unload(self) -> None:
        if self._current is not None:
            self._current.unload()
        self._current = None
        self._kind = None

    def transcribe(
        self, samples: np.ndarray, sample_rate: int, settings: "AppSettings"
    ) -> str:
        if samples.size == 0:
            return ""
        kind = resolve_engine(settings)
        # Ensure the right engine is loaded; reload if the model changed.
        if (
            self._current is None
            or self._kind != kind
            or not self._current.is_loaded
        ):
            self.load(settings)
        assert self._current is not None
        return self._current.transcribe(samples, sample_rate, settings)

    def transcribe_segments(
        self, samples: np.ndarray, sample_rate: int, settings: "AppSettings"
    ) -> list[dict[str, Any]] | None:
        """Timestamped segments when the active engine can produce them."""
        if samples.size == 0:
            return []
        kind = resolve_engine(settings)
        if (
            self._current is None
            or self._kind != kind
            or not self._current.is_loaded
        ):
            self.load(settings)
        assert self._current is not None
        getter = getattr(self._current, "transcribe_segments", None)
        if getter is None:
            return None
        return getter(samples, sample_rate, settings)

    def transcribe_with_segments(
        self, samples: np.ndarray, sample_rate: int, settings: "AppSettings"
    ) -> tuple[str, list[dict[str, Any]] | None]:
        """Transcribe once and return timestamps when the engine supports them.

        Whisper's segment call already performs recognition. Keeping this
        combined path prevents file transcription from running Whisper twice.
        """
        if samples.size == 0:
            return "", []
        kind = resolve_engine(settings)
        if (
            self._current is None
            or self._kind != kind
            or not self._current.is_loaded
        ):
            self.load(settings)
        assert self._current is not None
        getter = getattr(self._current, "transcribe_segments", None)
        if getter is None:
            return self._current.transcribe(samples, sample_rate, settings), None
        segments = getter(samples, sample_rate, settings)
        if segments is None:
            return self._current.transcribe(samples, sample_rate, settings), None
        text = " ".join(str(segment.get("text", "")).strip() for segment in segments)
        return text.strip(), segments

    def transcribe_file(
        self, path: Any, settings: "AppSettings"
    ) -> dict[str, Any] | None:
        """Remote engines: transcribe the file as-is in one API call.

        Local engines have no ``transcribe_file`` and return None so callers
        fall back to the sample-based path. The remote branch is selected by
        ``settings.model`` (the "remote" preset key), not by what is currently
        loaded, so a fresh worker handles it before any engine is loaded.
        """
        if resolve_engine(settings) != "remote":
            return None
        self.load(settings)
        assert self._current is not None
        getter = getattr(self._current, "transcribe_file", None)
        if getter is None:
            return None
        return getter(path, settings)
