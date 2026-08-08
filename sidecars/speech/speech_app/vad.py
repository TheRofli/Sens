"""RMS-based voice activity trimming.

For a push-to-talk flow the most useful voice-activity work is trimming silence
from the start and end of the captured buffer. That removes the mechanical
keyboard clicks from pressing/releasing the hotkey and the breath before/after
speech, both of which feed hallucinations in Whisper-family models on near-empty
audio. Internal pauses are intentionally left intact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ~30 ms frame at 16 kHz is a good balance between responsiveness and a stable
# energy estimate for speech.
_FRAME_MS = 30.0

SILERO_VAD_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "asr-models/silero_vad.onnx"
)
SILERO_VAD_SHA256 = (
    "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6"
)
SILERO_VAD_BYTES = 643_854


@dataclass(frozen=True, slots=True)
class SpeechSpan:
    start: int
    end: int


def silero_vad_path() -> Path:
    from .engines.paths import models_root

    return models_root() / "vad" / "silero_vad.onnx"


def _silero_threshold(sensitivity: float) -> float:
    # Preserve the old UI scale: 0.01 is permissive, 0.02 balanced, and
    # 0.04 confident. Silero itself expects a probability threshold.
    return max(0.2, min(0.8, 0.35 + float(sensitivity) * 7.5))


def detect_speech_spans(
    samples: np.ndarray,
    sample_rate: int,
    sensitivity: float = 0.02,
) -> list[SpeechSpan] | None:
    """Return Silero speech spans, or ``None`` when neural VAD is unavailable."""
    model = silero_vad_path()
    if not model.is_file():
        return None
    try:
        import sherpa_onnx
    except ImportError:
        return None

    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return []
    vad_rate = 16_000
    if sample_rate != vad_rate:
        from .engines.whisper import _resample_linear

        vad_audio = _resample_linear(audio, sample_rate, vad_rate)
    else:
        vad_audio = audio
    try:
        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = str(model)
        config.silero_vad.threshold = _silero_threshold(sensitivity)
        config.silero_vad.min_silence_duration = 0.25
        config.silero_vad.min_speech_duration = 0.20
        config.silero_vad.max_speech_duration = 30.0
        config.sample_rate = vad_rate
        config.num_threads = 1
        config.provider = "cpu"
        window = int(config.silero_vad.window_size)
        detector = sherpa_onnx.VoiceActivityDetector(
            config,
            buffer_size_in_seconds=max(30.0, vad_audio.size / vad_rate + 5.0),
        )
        for offset in range(0, vad_audio.size, window):
            frame = vad_audio[offset : offset + window]
            if frame.size < window:
                frame = np.pad(frame, (0, window - frame.size))
            detector.accept_waveform(frame)
        detector.flush()
        scale = float(sample_rate) / vad_rate
        padding = int(sample_rate * 0.12)
        spans: list[SpeechSpan] = []
        while not detector.empty():
            segment = detector.front
            start = max(0, int(segment.start * scale) - padding)
            end = min(
                audio.size,
                int((segment.start + len(segment.samples)) * scale) + padding,
            )
            if end > start:
                spans.append(SpeechSpan(start, end))
            detector.pop()
        return spans
    except Exception:
        # A damaged optional VAD must not break dictation; RMS remains the
        # deterministic degraded path and model status will expose repair.
        return None


def trim_for_recognition(
    samples: np.ndarray,
    sample_rate: int,
    sensitivity: float = 0.02,
    *,
    use_neural: bool = True,
) -> np.ndarray:
    """Trim dictation with Silero when available, otherwise use RMS fallback."""
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if use_neural:
        spans = detect_speech_spans(audio, sample_rate, sensitivity)
        if spans is not None:
            if not spans:
                return np.array([], dtype=np.float32)
            return audio[spans[0].start : spans[-1].end]
    return trim_silence(audio, sample_rate, sensitivity)


def _rms(frame: np.ndarray) -> float:
    if frame.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(frame))))


def _energy_frames(
    samples: np.ndarray, sample_rate: int
) -> tuple[np.ndarray, int]:
    """Split ``samples`` into non-overlapping frames and return their RMS.

    Returns ``(energies, frame_size)`` where ``energies[i]`` is the RMS of the
    i-th frame and ``frame_size`` is the number of samples per frame.
    """
    if samples.size == 0:
        return np.array([], dtype=np.float32), 0
    frame_size = max(1, int(sample_rate * _FRAME_MS / 1000.0))
    usable = (samples.size // frame_size) * frame_size
    if usable == 0:
        return np.array([_rms(samples)], dtype=np.float32), samples.size
    trimmed = samples[:usable].reshape(-1, frame_size)
    energies = np.sqrt(np.mean(np.square(trimmed), axis=1))
    return energies.astype(np.float32, copy=False), frame_size


def trim_silence(
    samples: np.ndarray,
    sample_rate: int,
    sensitivity: float = 0.02,
    min_duration_s: float = 0.3,
) -> np.ndarray:
    """Trim leading/trailing silence based on an RMS energy threshold.

    A frame is considered "speech" when its RMS exceeds ``sensitivity``. The
    returned buffer spans from the first speech frame to the last speech frame
    inclusive, so internal pauses are preserved.

    Parameters
    ----------
    samples:
        Mono float32 PCM in the range [-1, 1]. Other dtypes are coerced.
    sample_rate:
        Sample rate of ``samples`` in Hz.
    sensitivity:
        RMS energy threshold below which a frame counts as silence. Smaller
        values are more permissive (keep more audio); larger values trim more.
    min_duration_s:
        If the trimmed audio is shorter than this, an empty array is returned.
        This lets callers treat "almost silent" captures as no-speech instead
        of feeding a click into the model.

    Returns
    -------
    np.ndarray
        float32 mono buffer with silence trimmed, possibly empty.
    """
    if samples is None:
        return np.array([], dtype=np.float32)
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return audio.astype(np.float32, copy=False)

    energies, frame_size = _energy_frames(audio, sample_rate)
    if energies.size == 0:
        return audio.astype(np.float32, copy=False)

    # Always include the midpoint anchor: never trim to nothing based on a
    # threshold of 0.
    threshold = max(float(sensitivity), 1e-6)
    speech = np.where(energies >= threshold)[0]
    if speech.size == 0:
        # Nothing crossed the threshold; treat as silence.
        return np.array([], dtype=np.float32)

    start_sample = int(speech[0] * frame_size)
    end_sample = int((speech[-1] + 1) * frame_size)
    trimmed = audio[start_sample:end_sample]

    min_samples = int(min_duration_s * sample_rate)
    if trimmed.size < min_samples:
        return np.array([], dtype=np.float32)
    return trimmed.astype(np.float32, copy=False)


def split_audio(
    samples: np.ndarray,
    sample_rate: int,
    max_duration_s: float,
    sensitivity: float = 0.02,
    min_chunk_s: float = 1.0,
    overlap_s: float = 0.0,
) -> list[np.ndarray]:
    """Split audio into chunks no longer than ``max_duration_s``.

    Used for models with a bounded decoder context (currently Qwen3-ASR).
    Cuts are placed inside silence gaps — frames below
    the RMS threshold — so words are not split; when no gap is available
    before the limit, the chunk is hard-cut at the limit. The final chunk may
    be shorter than the rest.

    ``overlap_s`` reuses the tail of each previous chunk at the start of the
    next one, so a word split by a hard cut is still heard whole; the caller
    removes the duplicated seam from the transcriptions. ``overlap_s`` must be
    smaller than ``min_chunk_s`` so every cut still makes progress.
    """
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    max_samples = int(max_duration_s * sample_rate)
    min_samples = int(min_chunk_s * sample_rate)
    overlap_samples = int(overlap_s * sample_rate)
    if overlap_samples >= min_samples:
        raise ValueError("overlap_s must be smaller than min_chunk_s")
    if audio.size <= max_samples or max_samples <= min_samples:
        return [audio]

    energies, frame_size = _energy_frames(audio, sample_rate)
    threshold = max(float(sensitivity), 1e-6)
    silence = energies < threshold

    chunks: list[np.ndarray] = []
    start = 0
    while start < audio.size:
        target = min(start + max_samples, audio.size)
        if target == audio.size:
            chunks.append(audio[start:])
            break
        cut = target
        lo_frame = (start + min_samples) // frame_size
        hi_frame = target // frame_size
        if lo_frame < hi_frame:
            # Prefer the latest silence frame before the limit so the cut
            # lands in a pause rather than mid-word.
            for frame in range(hi_frame - 1, lo_frame - 1, -1):
                if silence[frame]:
                    cut = (frame + 1) * frame_size
                    break
        chunks.append(audio[start:cut])
        start = cut - overlap_samples
    return chunks
