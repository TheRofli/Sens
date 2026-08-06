"""Remote transcription engine for OpenAI-compatible audio APIs.

Sends the whole audio file to a ``/audio/transcriptions`` endpoint (OpenRouter
by default) instead of running a local model. The API key is read from the
Speech settings file and is never logged, written to argv, or included in
error messages.

The engine has no local model: ``is_loaded`` is always true once settings are
available, and every ``transcribe`` call is a network round trip. Live
dictation through this engine works (a temporary WAV is uploaded per
utterance) but adds network latency; it is intended mainly for agent file
transcription via Sens hearing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import requests

from ..settings import AppSettings
from .base import EngineUnavailable
from .parakeet import _write_temp_wav

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL_ID = "openai/gpt-4o-transcribe"
# Long files take a while server-side; the broker's own request timeout is the
# backstop (sens_hear defaults to 180 s).
_REQUEST_TIMEOUT_S = 300.0
_MAX_ERROR_CHARS = 300


class RemoteEngine:
    """Transcription via an OpenAI-compatible ``/audio/transcriptions`` API."""

    def __init__(self) -> None:
        self._settings: AppSettings | None = None

    # -- SpeechEngine-compatible API ---------------------------------------

    @property
    def is_loaded(self) -> bool:
        # Nothing to preload: availability is verified per request.
        return self._settings is not None

    @property
    def model_id(self) -> str:
        if self._settings is not None and self._settings.remote_model_id:
            return self._settings.remote_model_id
        return DEFAULT_MODEL_ID

    def load(self, settings: AppSettings) -> None:
        self._settings = settings

    def unload(self) -> None:
        self._settings = None

    def transcribe(
        self, samples: np.ndarray, sample_rate: int, settings: AppSettings
    ) -> str:
        if samples.size == 0:
            return ""
        wav_path = _write_temp_wav(samples, sample_rate)
        try:
            return self.transcribe_file(wav_path, settings)["text"]
        finally:
            try:
                wav_path.unlink()
            except OSError:
                pass

    def transcribe_segments(
        self, samples: np.ndarray, sample_rate: int, settings: AppSettings
    ) -> list[dict[str, Any]] | None:
        """Transcript with per-segment timestamps, like the whisper engine."""
        if samples.size == 0:
            return []
        wav_path = _write_temp_wav(samples, sample_rate)
        try:
            result = self.transcribe_file(wav_path, settings)
        finally:
            try:
                wav_path.unlink()
            except OSError:
                pass
        return _normalise_segments(result.get("segments"))

    # -- Remote-specific API --------------------------------------------------

    def transcribe_file(self, path: str | Path, settings: AppSettings) -> dict[str, Any]:
        """Transcribe a local audio file in one API call.

        Returns ``{"text", "segments", "language", "duration"}`` parsed from
        the OpenAI ``verbose_json`` response. Segments are normalized to the
        same ``{"start", "end", "text"}`` shape the whisper engine produces.
        """
        base_url = (settings.remote_base_url or DEFAULT_BASE_URL).rstrip("/")
        model = settings.remote_model_id or DEFAULT_MODEL_ID
        api_key = settings.remote_api_key
        if not api_key:
            raise EngineUnavailable(
                "OpenRouter API ключ не задан. Добавьте его в настройках слуха "
                "(модель «OpenRouter API»)."
            )
        url = f"{base_url}/audio/transcriptions"
        headers = {"Authorization": f"Bearer {api_key}"}
        form = {"model": model, "response_format": "verbose_json"}
        try:
            with open(path, "rb") as handle:
                files = {"file": (Path(path).name, handle, "application/octet-stream")}
                response = requests.post(
                    url,
                    headers=headers,
                    files=files,
                    data=form,
                    timeout=_REQUEST_TIMEOUT_S,
                )
        except requests.RequestException as exc:
            raise EngineUnavailable(
                f"Не удалось связаться с {base_url}: {exc}"
            ) from exc
        _raise_for_status(response, model, base_url)
        try:
            payload = response.json()
        except ValueError as exc:
            raise EngineUnavailable(
                f"{base_url} вернул не-JSON ответ: {response.text[:_MAX_ERROR_CHARS]}"
            ) from exc
        text = str(payload.get("text") or "").strip()
        segments = _normalise_segments(payload.get("segments"))
        duration = payload.get("duration")
        return {
            "text": text,
            "segments": segments,
            "language": payload.get("language"),
            "duration": float(duration) if isinstance(duration, (int, float)) else 0.0,
        }


def _normalise_segments(segments: Any) -> list[dict[str, Any]] | None:
    """Map OpenAI segments to the whisper ``{"start", "end", "text"}`` shape."""
    if not isinstance(segments, list):
        return None
    normalized: list[dict[str, Any]] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        start = segment.get("start")
        end = segment.get("end")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        normalized.append(
            {
                "start": round(float(start), 2),
                "end": round(float(end), 2),
                "text": text,
            }
        )
    return normalized


def _raise_for_status(response: requests.Response, model: str, base_url: str) -> None:
    """Map common API failures to actionable Russian messages."""
    if response.status_code == 200:
        return
    detail = ""
    try:
        error = response.json().get("error")
        if isinstance(error, dict):
            detail = str(error.get("message") or "")
    except ValueError:
        detail = response.text
    detail = (detail or response.text or "").strip()[:_MAX_ERROR_CHARS]
    if response.status_code == 401:
        raise EngineUnavailable(
            "OpenRouter отверг API ключ (401). Проверьте ключ в настройках слуха."
        )
    if response.status_code == 402:
        raise EngineUnavailable(
            "На балансе OpenRouter недостаточно средств (402). Пополните баланс."
        )
    if response.status_code == 404:
        raise EngineUnavailable(
            f"Модель {model} не найдена на {base_url} (404). "
            "Проверьте название модели — например, openai/gpt-4o-transcribe."
        )
    raise EngineUnavailable(
        f"{base_url} вернул {response.status_code} для модели {model}"
        + (f": {detail}" if detail else "")
    )
