import unittest
from unittest import mock

import numpy as np

from speech_app.vad import (
    SpeechSpan,
    split_audio,
    trim_for_recognition,
    trim_silence,
)


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


class SplitAudioTests(unittest.TestCase):
    def test_short_audio_stays_in_one_chunk(self):
        samples = _tone(440.0, 3.0)
        chunks = split_audio(samples, sample_rate=16000, max_duration_s=24.0)
        self.assertEqual(len(chunks), 1)
        np.testing.assert_array_equal(chunks[0], samples)

    def test_empty_audio_returns_single_empty_chunk(self):
        chunks = split_audio(np.array([], dtype=np.float32), sample_rate=16000, max_duration_s=24.0)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].size, 0)

    def test_continuous_tone_is_hard_cut_at_limit(self):
        samples = _tone(440.0, 60.0)
        chunks = split_audio(samples, sample_rate=16000, max_duration_s=24.0)
        self.assertEqual(len(chunks), 3)
        for chunk in chunks[:-1]:
            self.assertEqual(chunk.size, 24 * 16000)
        self.assertEqual(sum(chunk.size for chunk in chunks), samples.size)

    def test_cut_lands_in_silence_gap_before_the_limit(self):
        # Tone 20 s, silence 3 s, then more tone: the 24 s limit falls inside
        # the pause, so the cut must land there and not split a word.
        samples = np.concatenate(
            [_tone(440.0, 20.0), _silence(3.0), _tone(440.0, 40.0)]
        )
        chunks = split_audio(samples, sample_rate=16000, max_duration_s=24.0)
        self.assertGreater(len(chunks), 1)
        first = chunks[0]
        self.assertLess(first.size, 24 * 16000)
        self.assertGreater(first.size, 22 * 16000)
        self.assertEqual(sum(chunk.size for chunk in chunks), samples.size)

    def test_overlap_reuses_tail_of_previous_chunk(self):
        samples = _tone(440.0, 60.0)
        chunks = split_audio(
            samples,
            sample_rate=16000,
            max_duration_s=24.0,
            min_chunk_s=2.0,
            overlap_s=1.5,
        )
        self.assertGreater(len(chunks), 1)
        overlap = int(1.5 * 16000)
        for left, right in zip(chunks, chunks[1:]):
            # The previous chunk's tail is the next chunk's head, so a word
            # split by the cut is still heard whole by the next chunk.
            np.testing.assert_array_equal(left[-overlap:], right[:overlap])

    def test_overlap_cut_still_prefers_silence_gap(self):
        samples = np.concatenate(
            [_tone(440.0, 20.0), _silence(3.0), _tone(440.0, 40.0)]
        )
        chunks = split_audio(
            samples,
            sample_rate=16000,
            max_duration_s=24.0,
            min_chunk_s=2.0,
            overlap_s=1.5,
        )
        self.assertGreater(len(chunks), 1)
        first = chunks[0]
        self.assertLess(first.size, 24 * 16000)
        self.assertGreater(first.size, 22 * 16000)

    def test_overlap_must_be_smaller_than_min_chunk(self):
        samples = _tone(440.0, 60.0)
        with self.assertRaises(ValueError):
            split_audio(
                samples,
                sample_rate=16000,
                max_duration_s=24.0,
                min_chunk_s=1.0,
                overlap_s=1.5,
            )


class NeuralVadTests(unittest.TestCase):
    def test_neural_spans_trim_outer_noise_but_preserve_internal_audio(self):
        audio = np.arange(1000, dtype=np.float32)
        with mock.patch(
            "speech_app.vad.detect_speech_spans",
            return_value=[SpeechSpan(100, 300), SpeechSpan(600, 900)],
        ):
            trimmed = trim_for_recognition(audio, 16000, use_neural=True)
        np.testing.assert_array_equal(trimmed, audio[100:900])

    def test_neural_no_speech_returns_empty(self):
        with mock.patch("speech_app.vad.detect_speech_spans", return_value=[]):
            trimmed = trim_for_recognition(
                np.ones(1600, dtype=np.float32), 16000, use_neural=True
            )
        self.assertEqual(trimmed.size, 0)

    def test_missing_neural_runtime_falls_back_to_rms(self):
        audio = np.concatenate(
            [np.zeros(480), np.ones(1600, dtype=np.float32), np.zeros(480)]
        )
        with mock.patch("speech_app.vad.detect_speech_spans", return_value=None):
            neural = trim_for_recognition(audio, 16000, use_neural=True)
        rms = trim_silence(audio, 16000)
        np.testing.assert_array_equal(neural, rms)


if __name__ == "__main__":
    unittest.main()
