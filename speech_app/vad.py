"""RMS-based voice activity trimming.

For a push-to-talk flow the most useful voice-activity work is trimming silence
from the start and end of the captured buffer. That removes the mechanical
keyboard clicks from pressing/releasing the hotkey and the breath before/after
speech, both of which feed hallucinations in Whisper-family models on near-empty
audio. Internal pauses are intentionally left intact.
"""

from __future__ import annotations

import numpy as np

# ~30 ms frame at 16 kHz is a good balance between responsiveness and a stable
# energy estimate for speech.
_FRAME_MS = 30.0


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
