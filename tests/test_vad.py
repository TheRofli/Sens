import unittest

import numpy as np

from speech_app.vad import trim_silence


def _tone(freq_hz: float, duration_s: float, sample_rate: int = 16000) -> np.ndarray:
    n = int(duration_s * sample_rate)
    t = np.arange(n) / sample_rate
    return (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _silence(duration_s: float, sample_rate: int = 16000) -> np.ndarray:
    return np.zeros(int(duration_s * sample_rate), dtype=np.float32)


class TrimSilenceTests(unittest.TestCase):
    def test_pure_silence_returns_empty(self):
        samples = _silence(2.0)
        out = trim_silence(samples, sample_rate=16000, sensitivity=0.02)
        self.assertEqual(out.size, 0)

    def test_empty_input_returns_empty(self):
        out = trim_silence(np.array([], dtype=np.float32), sample_rate=16000)
        self.assertEqual(out.size, 0)

    def test_trims_leading_and_trailing_silence(self):
        # 0.5s silence + 1.0s tone + 0.5s silence
        samples = np.concatenate([_silence(0.5), _tone(440.0, 1.0), _silence(0.5)])
        out = trim_silence(samples, sample_rate=16000, sensitivity=0.02)
        # Tone must remain and be ~1s long (within one frame tolerance).
        self.assertGreater(out.size, int(0.8 * 16000))
        self.assertLessEqual(out.size, int(1.05 * 16000))
        # The trimmed buffer should contain real energy.
        self.assertGreater(float(np.sqrt(np.mean(np.square(out)))), 0.02)

    def test_short_click_below_min_duration_returns_empty(self):
        # A 50ms burst is below the default 0.3s minimum duration.
        samples = _tone(440.0, 0.05)
        out = trim_silence(samples, sample_rate=16000, sensitivity=0.02)
        self.assertEqual(out.size, 0)

    def test_internal_pause_is_preserved(self):
        # tone - pause - tone: trimming edges only must keep the middle pause.
        samples = np.concatenate(
            [_tone(440.0, 0.4), _silence(0.3), _tone(440.0, 0.4)]
        )
        out = trim_silence(samples, sample_rate=16000, sensitivity=0.02)
        # Total kept should include both tones + the pause (~1.1s).
        self.assertGreater(out.size, int(1.0 * 16000))


if __name__ == "__main__":
    unittest.main()
