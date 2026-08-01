"""Registry of selectable ASR model presets.

A :class:`ModelPreset` describes one model that the user can pick from the tray
or window UI. Each preset maps to a concrete engine (parakeet / whisper) and a
Hugging Face repo id. ``EngineManager`` and the install flow both read from this
registry so the model list has a single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .settings import AppSettings


@dataclass(frozen=True, slots=True)
class ModelPreset:
    """One selectable ASR model."""

    key: str
    label: str
    engine: str  # "parakeet" | "whisper"
    model_id: str  # Hugging Face repo id
    family: str  # cache-family identifier, e.g. "parakeet-tdt" | "whisper"
    description: str = ""


MODELS: dict[str, ModelPreset] = {
    "parakeet": ModelPreset(
        key="parakeet",
        label="Parakeet (быстрая)",
        engine="parakeet",
        model_id="nvidia/parakeet-tdt-0.6b-v3",
        family="parakeet-tdt",
        description="600M, мультиязычная, быстрая на CPU.",
    ),
    "whisper-ru": ModelPreset(
        key="whisper-ru",
        label="Whisper RU codeswitch (точная)",
        engine="whisper",
        model_id="coriollon/whisper-large-v3-turbo-russian-codeswitch",
        family="whisper",
        description="809M, файн-тюн large-v3-turbo под RU+EN код-свичинг.",
    ),
    "gigaam": ModelPreset(
        key="gigaam",
        label="GigaAM v3 (русский, точная)",
        engine="gigaam",
        model_id="ai-sage/GigaAM-v3",
        family="gigaam",
        description="230M, Sber, лучший русский на CPU, e2e с пунктуацией.",
    ),
}


class UnknownModel(KeyError):
    """Raised when a model key is not in the registry."""


def get_preset(key: str) -> ModelPreset:
    """Return the preset for ``key`` or raise :class:`UnknownModel`."""
    try:
        return MODELS[key]
    except KeyError as exc:
        raise UnknownModel(key) from exc


def available_presets() -> list[ModelPreset]:
    """Return all presets in a stable display order."""
    return [MODELS[key] for key in ("parakeet", "whisper-ru", "gigaam")]


def resolve_engine(settings: "AppSettings") -> str:
    """Return the engine kind ("parakeet" | "whisper") for the active model.

    Falls back to "parakeet" for unknown ``settings.model`` values so a stale
    settings file from an older install never hard-blocks startup.
    """
    preset = MODELS.get(settings.model)
    if preset is not None:
        return preset.engine
    # Back-compat: older settings had no `model` field but carried a model_id.
    return "parakeet"


def resolve_model_id(settings: "AppSettings") -> str:
    """Return the canonical HF repo id for the active model preset.

    Falls back to ``settings.model_id`` for unknown presets so an older config
    that pinned a custom parakeet model id keeps working.
    """
    preset = MODELS.get(settings.model)
    if preset is not None:
        return preset.model_id
    return settings.model_id
