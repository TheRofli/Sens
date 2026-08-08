from __future__ import annotations

from collections.abc import Callable
from threading import Lock

import numpy as np


class AudioRecorder:
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        level_callback: Callable[[float], None] | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.level_callback = level_callback
        self._stream = None
        self._chunks: list[np.ndarray] = []
        self._lock = Lock()

    @property
    def is_recording(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        if self._stream is not None:
            return

        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError(
                "sounddevice is not installed. Run install.ps1 first."
            ) from exc

        self._chunks = []

        def callback(indata, frames, time_info, status) -> None:
            del frames, time_info, status
            samples = np.asarray(indata, dtype=np.float32).copy()
            with self._lock:
                self._chunks.append(samples.reshape(-1))
            if self.level_callback:
                rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0
                self.level_callback(min(1.0, rms * 18.0))

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            callback=callback,
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        stream = self._stream
        self._stream = None
        if stream is not None:
            stream.stop()
            stream.close()

        with self._lock:
            chunks = self._chunks
            self._chunks = []

        if not chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32, copy=False)

