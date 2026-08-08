"""Single source of truth for selectable Hearing model presets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .settings import AppSettings


@dataclass(frozen=True, slots=True)
class ModelPreset:
    key: str
    label: str
    engine: str
    model_id: str
    family: str
    description: str = ""
    download_url: str = ""
    download_sha256: str = ""
    download_bytes: int = 0
    revision: str = ""
    required_files: tuple[str, ...] = ()


MODELS: dict[str, ModelPreset] = {
    "qwen": ModelPreset(
        key="qwen",
        label="Qwen3-ASR 0.6B INT8 (авто, 30 языков)",
        engine="qwen",
        model_id="Qwen/Qwen3-ASR-0.6B",
        family="sherpa-qwen3-asr",
        description=(
            "Сбалансированная локальная CPU-модель: русский, английский, "
            "code-switching и ещё 28 языков."
        ),
        download_url=(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
            "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2"
        ),
        download_sha256=(
            "393f8a14e2f5fb96746aaab342997a40641001fbd5bf9592a080a8329178ee96"
        ),
        download_bytes=878_702_423,
        required_files=(
            "conv_frontend.onnx",
            "encoder.int8.onnx",
            "decoder.int8.onnx",
            "tokenizer/vocab.json",
            "tokenizer/merges.txt",
            "tokenizer/tokenizer_config.json",
        ),
    ),
    "gigaam": ModelPreset(
        key="gigaam",
        label="GigaAM v3 INT8 (русский)",
        engine="gigaam",
        model_id=(
            "csukuangfj/"
            "sherpa-onnx-nemo-transducer-punct-giga-am-v3-russian-2025-12-16"
        ),
        family="sherpa-transducer",
        description=(
            "Компактная быстрая русская модель с пунктуацией, без 25-секундного "
            "ограничения старого Transformers backend."
        ),
        download_url=(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
            "sherpa-onnx-nemo-transducer-punct-giga-am-v3-russian-2025-12-16.tar.bz2"
        ),
        download_sha256=(
            "f9620a0099019c6afcee26525ef9ed3297fa50dd5691c1902af0c948fc1a470b"
        ),
        download_bytes=170_197_019,
        required_files=(
            "encoder.int8.onnx",
            "decoder.onnx",
            "joiner.onnx",
            "tokens.txt",
        ),
    ),
    "whisper": ModelPreset(
        key="whisper",
        label="Whisper Small INT8 (99 языков)",
        engine="whisper",
        model_id="Systran/faster-whisper-small",
        family="faster-whisper",
        description=(
            "Быстрый мультиязычный fallback для языков вне набора Qwen; "
            "значительно легче прежнего Whisper Large."
        ),
        revision="536b0662742c02347bc0e980a01041f333bce120",
        download_bytes=483_546_902,
        required_files=(
            "config.json",
            "model.bin",
            "tokenizer.json",
            "vocabulary.txt",
        ),
    ),
    "remote": ModelPreset(
        key="remote",
        label="OpenRouter API (онлайн)",
        engine="remote",
        model_id="openai/gpt-4o-transcribe",
        family="remote",
        description="Опциональная транскрипция через пользовательский API-ключ.",
    ),
}

LEGACY_MODEL_ALIASES = {
    "parakeet": "qwen",
    "whisper-ru": "whisper",
}


class UnknownModel(KeyError):
    pass


def normalize_model_key(key: str) -> str:
    return LEGACY_MODEL_ALIASES.get(key, key)


def get_preset(key: str) -> ModelPreset:
    try:
        return MODELS[normalize_model_key(key)]
    except KeyError as exc:
        raise UnknownModel(key) from exc


def available_presets() -> list[ModelPreset]:
    return [MODELS[key] for key in ("qwen", "gigaam", "whisper", "remote")]


def resolve_engine(settings: "AppSettings") -> str:
    preset = MODELS.get(normalize_model_key(settings.model))
    return preset.engine if preset is not None else "qwen"


def resolve_model_id(settings: "AppSettings") -> str:
    preset = MODELS.get(normalize_model_key(settings.model))
    return preset.model_id if preset is not None else MODELS["qwen"].model_id
