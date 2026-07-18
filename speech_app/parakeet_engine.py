"""Back-compat shim.

The concrete Parakeet engine now lives in :mod:`speech_app.engines.parakeet`
and is driven through :class:`~speech_app.engine_manager.EngineManager`. This
module re-exports the previously public names so legacy imports
(``from .parakeet_engine import ParakeetEngine, EngineUnavailable``) keep
working.
"""

from __future__ import annotations

from .engines.base import EngineUnavailable, LoadedEngine
from .engines.parakeet import ParakeetEngine

__all__ = ["EngineUnavailable", "LoadedEngine", "ParakeetEngine"]
