"""ASR engine package.

Each concrete engine (sherpa, whisper, remote) implements :class:`SpeechEngine`.
:class:`~speech_app.engine_manager.EngineManager` selects one based on the
active model preset.
"""

from __future__ import annotations

from .base import EngineUnavailable, LoadedEngine, SpeechEngine

__all__ = ["EngineUnavailable", "LoadedEngine", "SpeechEngine"]
