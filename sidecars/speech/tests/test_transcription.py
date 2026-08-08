import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from speech_app.settings import AppSettings
from speech_app.transcription import (
    extract_frames,
    extract_frames_at,
    extract_frames_every,
    load_audio_file,
    settings_for_request,
    transcribe_audio_file,
    validate_audio_path,
)


def write_wav(path: Path, samples: np.ndarray, sample_rate: int = 16000) -> None:
    pcm = np.clip(samples, -1, 1)
    pcm = (pcm * 32767).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm.tobytes())


class FileTranscriptionTests(unittest.TestCase):
    def test_requires_absolute_supported_file(self):
        with self.assertRaises(ValueError):
            validate_audio_path("relative.wav")

    def test_loads_wav_as_mono_float32(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "voice.wav"
            write_wav(path, np.linspace(-0.5, 0.5, 3200, dtype=np.float32))

            samples, sample_rate = load_audio_file(path)

            self.assertEqual(sample_rate, 16000)
            self.assertEqual(samples.dtype, np.float32)
            self.assertEqual(samples.ndim, 1)
            self.assertEqual(samples.size, 3200)

    def test_request_settings_disable_system_output(self):
        snapshot = settings_for_request(AppSettings(), model="qwen")

        self.assertFalse(snapshot.copy_to_clipboard)
        self.assertFalse(snapshot.paste_to_active_input)
        # File transcription must hear sung/vocoded vocals, so the dictation
        # VAD gate is off for agent requests.
        self.assertFalse(snapshot.vad_filter)

    def test_transcription_returns_metadata_without_publishing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "voice.wav"
            # A short tone has enough energy and variation for the real VAD.
            timeline = np.arange(8000, dtype=np.float32) / 16000
            write_wav(path, 0.2 * np.sin(2 * np.pi * 440 * timeline))
            engine = Mock()
            engine.kind = "fake"
            engine.transcribe.return_value = " hello world "

            result = transcribe_audio_file(
                path,
                # This test exercises the ASR/result envelope, not Silero's
                # speech classification. A synthetic 440 Hz tone is not
                # speech and must not make the assertion depend on whether
                # onnxruntime happens to be installed in the test runtime.
                settings=AppSettings(postprocess_text=False, vad_filter=False),
                engine=engine,
            )

            self.assertEqual(result["text"], "hello world")
            self.assertEqual(result["engine"], "fake")
            self.assertEqual(result["container"], "audio")
            self.assertAlmostEqual(result["duration_seconds"], 0.5, places=2)
            self.assertEqual(result["observed"]["source"], "observed")
            self.assertEqual(result["measured"]["inputSamples"], 8000)
            self.assertEqual(result["inferred"]["method"], "fake_asr")
            engine.transcribe.assert_called_once()

    def test_whisper_file_transcription_uses_one_combined_asr_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "voice.wav"
            timeline = np.arange(8000, dtype=np.float32) / 16000
            write_wav(path, 0.2 * np.sin(2 * np.pi * 440 * timeline))
            engine = Mock()
            engine.kind = "whisper"
            engine.model_id = "Systran/faster-whisper-small"
            engine.transcribe_with_segments.return_value = (
                " hello world ",
                [{"start": 0.0, "end": 0.5, "text": "hello world"}],
            )

            result = transcribe_audio_file(
                path,
                settings=AppSettings(model="whisper", postprocess_text=False),
                engine=engine,
            )

            self.assertEqual(result["text"], "hello world")
            self.assertEqual(len(result["segments"]), 1)
            engine.transcribe_with_segments.assert_called_once()
            engine.transcribe.assert_not_called()
            engine.transcribe_segments.assert_not_called()


def write_test_video(path: Path, seconds: float = 1.0, with_audio: bool = True) -> None:
    """Synthesize a tiny MP4 (gradient frames + optional 440 Hz tone)."""
    import av

    container = av.open(str(path), "w", format="mp4")
    video = container.add_stream("mpeg4", rate=10)
    video.width, video.height = 128, 96
    if with_audio:
        audio = container.add_stream("aac", rate=16000)
        sample_rate = 16000
        timeline = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
        tone = (0.2 * np.sin(2 * np.pi * 440 * timeline)).astype(np.float32)
        frame = av.AudioFrame.from_ndarray(
            tone.reshape(1, -1), format="fltp", layout="mono"
        )
        frame.sample_rate = sample_rate
        for packet in audio.encode(frame):
            container.mux(packet)
    for index in range(int(seconds * 10)):
        image = np.full((96, 128, 3), (index * 20) % 256, dtype=np.uint8)
        vframe = av.VideoFrame.from_ndarray(image, format="rgb24")
        for packet in video.encode(vframe):
            container.mux(packet)
    if with_audio:
        for packet in audio.encode(None):
            container.mux(packet)
    for packet in video.encode(None):
        container.mux(packet)
    container.close()


class VideoTranscriptionTests(unittest.TestCase):
    def test_loads_video_audio_track_as_mono_float32(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.mp4"
            write_test_video(path)

            samples, sample_rate = load_audio_file(path)

            self.assertEqual(samples.dtype, np.float32)
            self.assertEqual(samples.ndim, 1)
            self.assertGreater(samples.size, 0)
            self.assertEqual(sample_rate, 16000)

    def test_silent_video_returns_empty_text_without_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "silent.mp4"
            write_test_video(path, seconds=1.0, with_audio=False)
            engine = Mock()
            engine.kind = "fake"
            engine.transcribe.return_value = "should not be called"

            result = transcribe_audio_file(
                path,
                settings=AppSettings(postprocess_text=False),
                engine=engine,
            )

            self.assertEqual(result["text"], "")
            self.assertEqual(result["container"], "video")
            self.assertFalse(result["audioTrack"])
            engine.transcribe.assert_not_called()

    def test_extract_frames_returns_jpegs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.mp4"
            write_test_video(path, seconds=2.0)
            out_dir = Path(tmp) / "frames"

            frame_paths = extract_frames(path, count=4, out_dir=out_dir)

            self.assertEqual(len(frame_paths), 4)
            for frame_path in frame_paths:
                self.assertTrue(Path(frame_path).is_file())
                with open(frame_path, "rb") as stream:
                    self.assertEqual(stream.read(3), b"\xff\xd8\xff")

    def test_extract_frames_ignores_audio_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "voice.wav"
            write_wav(path, np.zeros(3200, dtype=np.float32))

            frame_paths = extract_frames(path, count=4)

            self.assertEqual(frame_paths, [])

    def test_extract_frames_at_exact_seconds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.mp4"
            write_test_video(path, seconds=2.0)
            out_dir = Path(tmp) / "frames"

            frame_paths = extract_frames_at(path, [0.5, 1.5], out_dir=out_dir)

            self.assertEqual(len(frame_paths), 2)
            for frame_path in frame_paths:
                self.assertTrue(Path(frame_path).is_file())
                self.assertIn("-at-", Path(frame_path).name)

    def test_extract_frames_every_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.mp4"
            write_test_video(path, seconds=2.0)
            out_dir = Path(tmp) / "frames"

            frame_paths = extract_frames_every(path, 0.5, out_dir=out_dir)

            self.assertGreaterEqual(len(frame_paths), 3)
            for frame_path in frame_paths:
                self.assertTrue(Path(frame_path).is_file())
                self.assertIn("-at-", Path(frame_path).name)

    def test_extract_frames_every_caps_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.mp4"
            write_test_video(path, seconds=2.0)
            out_dir = Path(tmp) / "frames"

            frame_paths = extract_frames_every(path, 0.2, max_count=3, out_dir=out_dir)

            self.assertEqual(len(frame_paths), 3)

    def test_segments_returned_when_engine_supports_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "voice.wav"
            timeline = np.arange(8000, dtype=np.float32) / 16000
            write_wav(path, 0.2 * np.sin(2 * np.pi * 440 * timeline))
            engine = Mock()
            engine.kind = "fake"
            engine.transcribe.return_value = "hello"
            engine.transcribe_segments.return_value = [
                {"start": 0.0, "end": 0.5, "text": "hello"}
            ]

            result = transcribe_audio_file(
                path,
                settings=AppSettings(postprocess_text=False),
                engine=engine,
            )

            self.assertEqual(result["segments"], [{"start": 0.0, "end": 0.5, "text": "hello"}])

    def test_segments_none_without_engine_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "voice.wav"
            timeline = np.arange(8000, dtype=np.float32) / 16000
            write_wav(path, 0.2 * np.sin(2 * np.pi * 440 * timeline))
            engine = Mock(spec=["transcribe", "kind"])
            engine.kind = "fake"
            engine.transcribe.return_value = "hello"

            result = transcribe_audio_file(
                path,
                settings=AppSettings(postprocess_text=False),
                engine=engine,
            )

            self.assertIsNone(result["segments"])


if __name__ == "__main__":
    unittest.main()
