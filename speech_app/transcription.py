"""Side-effect-free audio-file transcription for agent and API callers.

This module deliberately knows nothing about the clipboard, active window,
Tk, tray, or transcript publishing. Interactive dictation keeps those user
effects in :mod:`speech_app.output`; Sens hearing calls this pure boundary.
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .engine_manager import EngineManager
from .models import get_preset
from .settings import AppSettings
from .textpost import postprocess
from .vad import trim_silence


SUPPORTED_EXTENSIONS = {
    ".wav",
    ".flac",
    ".ogg",
    ".mp3",
    ".m4a",
    ".aac",
    ".opus",
    ".webm",
}
# Containers that carry a video track; the audio track is decoded with PyAV
# (bundled ffmpeg libraries, no external ffmpeg binary needed).
VIDEO_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".m4v",
    ".wmv",
    ".mpg",
    ".mpeg",
    ".ts",
    ".flv",
}
MAX_AUDIO_BYTES = 512 * 1024 * 1024


def validate_audio_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("audio_path must be an absolute local path")
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"audio_path is not a file: {path}")
    if (
        path.suffix.lower() not in SUPPORTED_EXTENSIONS
        and path.suffix.lower() not in VIDEO_EXTENSIONS
    ):
        raise ValueError(
            f"unsupported extension {path.suffix!r}; "
            f"supported: {', '.join(sorted(SUPPORTED_EXTENSIONS | VIDEO_EXTENSIONS))}"
        )
    if path.stat().st_size > MAX_AUDIO_BYTES:
        raise ValueError("audio file exceeds the 512 MiB safety limit")
    return path


def _load_video_audio(path: Path) -> tuple[np.ndarray, int]:
    """Decode the audio track of a video container to mono float32 via PyAV.

    A container without an audio track yields empty samples at 16 kHz instead
    of raising, so silent videos still transcribe to "" and can be analysed
    through stills.
    """
    try:
        import av
    except ImportError as exc:
        raise RuntimeError(
            "Video decoding requires PyAV (av). Run install.ps1 with video support."
        ) from exc
    container = av.open(str(path))
    try:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            return np.zeros(0, dtype=np.float32), 16000
        sample_rate = int(stream.codec_context.sample_rate or 16000)
        chunks: list[np.ndarray] = []
        for frame in container.decode(stream):
            arr = frame.to_ndarray()
            if arr.ndim > 1:
                arr = arr.mean(axis=0)
            if arr.dtype != np.float32:
                if arr.dtype == np.int16:
                    arr = arr.astype(np.float32) / 32768.0
                elif arr.dtype == np.int32:
                    arr = arr.astype(np.float32) / 2147483648.0
                elif arr.dtype == np.uint8:
                    arr = arr.astype(np.float32) / 255.0
                else:
                    arr = arr.astype(np.float32)
            chunks.append(arr)
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    finally:
        container.close()
    audio = np.ascontiguousarray(audio.reshape(-1), dtype=np.float32)
    if not np.isfinite(audio).all():
        raise ValueError("audio contains non-finite samples")
    return audio, sample_rate


def load_audio_file(value: str | Path) -> tuple[np.ndarray, int]:
    """Decode a local audio or video file to mono float32 without changing
    sample rate. Video containers are decoded from their audio track."""
    path = validate_audio_path(value)
    if path.suffix.lower() in VIDEO_EXTENSIONS:
        return _load_video_audio(path)
    try:
        import soundfile as sf

        samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    except (ImportError, RuntimeError):
        try:
            import librosa
        except ImportError as exc:
            raise RuntimeError(
                "Audio decoding requires soundfile or librosa. Run install.ps1."
            ) from exc
        samples, sample_rate = librosa.load(str(path), sr=None, mono=False)

    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim > 1:
        # soundfile uses frames x channels; librosa uses channels x frames.
        channel_axis = 1 if audio.shape[0] >= audio.shape[-1] else 0
        audio = audio.mean(axis=channel_axis)
    audio = np.ascontiguousarray(audio.reshape(-1), dtype=np.float32)
    if not np.isfinite(audio).all():
        raise ValueError("audio contains non-finite samples")
    if int(sample_rate) <= 0:
        raise ValueError("audio sample rate is invalid")
    return audio, int(sample_rate)


def extract_frames(
    video_path: str | Path,
    count: int = 6,
    *,
    out_dir: str | Path | None = None,
    max_side: int = 1280,
) -> list[str]:
    """Extract evenly spaced stills from a video container as JPEG files.

    Returns absolute paths to the saved frames (empty when the container has
    no video track). The caller is responsible for cleaning up ``out_dir``.
    """
    path = Path(video_path).expanduser().resolve()
    if count <= 0 or path.suffix.lower() not in VIDEO_EXTENSIONS:
        return []
    try:
        import av
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Frame extraction requires PyAV and Pillow. Run install.ps1."
        ) from exc

    out_dir = Path(out_dir) if out_dir else path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path))
    frame_paths: list[str] = []
    try:
        stream = next((s for s in container.streams if s.type == "video"), None)
        if stream is None:
            return []
        frames: list[np.ndarray] = []
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
        if not frames:
            return []
        total = len(frames)
        if count == 1:
            picks = [total // 2]
        else:
            picks = [round(i * (total - 1) / (count - 1)) for i in range(count)]
        for index, pick in enumerate(dict.fromkeys(picks)):
            image = Image.fromarray(frames[pick])
            if max_side and max(image.size) > max_side:
                image.thumbnail((max_side, max_side))
            target = out_dir / f"frame-{index:02d}.jpg"
            image.convert("RGB").save(target, "JPEG", quality=85)
            frame_paths.append(str(target))
    finally:
        container.close()
    return frame_paths


def extract_frames_at(
    video_path: str | Path,
    times: Sequence[float],
    *,
    out_dir: str | Path | None = None,
    max_side: int = 1280,
) -> list[str]:
    """Extract stills at exact video seconds (seek + nearest frame).

    Frame file names embed the target time (``frame-at-16.5s.jpg``) so the
    model can correlate a still with the transcript's timestamped segments.
    """
    path = Path(video_path).expanduser().resolve()
    targets = sorted({float(t) for t in times if t >= 0})
    if not targets or path.suffix.lower() not in VIDEO_EXTENSIONS:
        return []
    try:
        import av
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Frame extraction requires PyAV and Pillow. Run install.ps1."
        ) from exc

    out_dir = Path(out_dir) if out_dir else path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path))
    frame_paths: list[str] = []
    try:
        stream = next((s for s in container.streams if s.type == "video"), None)
        if stream is None:
            return []
        time_base = float(stream.time_base)
        for target in targets:
            container.seek(
                max(0, int(target / time_base) - 1),
                stream=stream,
                backward=True,
            )
            best: Any = None
            best_delta = float("inf")
            for frame in container.decode(stream):
                pts = frame.pts
                if pts is None:
                    continue
                stamp = float(pts) * time_base
                delta = abs(stamp - target)
                if delta < best_delta:
                    best, best_delta = frame, delta
                if stamp > target + 0.5:
                    break
            if best is None:
                continue
            image = Image.fromarray(best.to_ndarray(format="rgb24"))
            if max_side and max(image.size) > max_side:
                image.thumbnail((max_side, max_side))
            target_path = out_dir / f"frame-at-{target:.1f}s.jpg"
            image.convert("RGB").save(target_path, "JPEG", quality=85)
            frame_paths.append(str(target_path))
    finally:
        container.close()
    return frame_paths


def extract_frames_every(
    video_path: str | Path,
    interval_s: float,
    *,
    max_count: int = 12,
    out_dir: str | Path | None = None,
    max_side: int = 1280,
) -> list[str]:
    """Extract one still every ``interval_s`` seconds of video.

    Frames land on 0, interval, 2*interval, ... capped at ``max_count``
    stills. File names embed the target time so the model can correlate a
    still with the transcript's timestamped segments.
    """
    path = Path(video_path).expanduser().resolve()
    if interval_s <= 0 or max_count <= 0 or path.suffix.lower() not in VIDEO_EXTENSIONS:
        return []
    try:
        import av
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Frame extraction requires PyAV and Pillow. Run install.ps1."
        ) from exc

    out_dir = Path(out_dir) if out_dir else path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path))
    frame_paths: list[str] = []
    try:
        stream = next((s for s in container.streams if s.type == "video"), None)
        if stream is None:
            return []
        duration_s = 0.0
        if container.duration:
            duration_s = float(container.duration) / av.time_base
        targets = [
            index * interval_s
            for index in range(int(duration_s / interval_s) + 1)
        ][:max_count]
        time_base = float(stream.time_base)
        for target in targets:
            container.seek(
                max(0, int(target / time_base) - 1),
                stream=stream,
                backward=True,
            )
            best: Any = None
            best_delta = float("inf")
            for frame in container.decode(stream):
                pts = frame.pts
                if pts is None:
                    continue
                stamp = float(pts) * time_base
                delta = abs(stamp - target)
                if delta < best_delta:
                    best, best_delta = frame, delta
                if stamp > target + 0.5:
                    break
            if best is None:
                continue
            image = Image.fromarray(best.to_ndarray(format="rgb24"))
            if max_side and max(image.size) > max_side:
                image.thumbnail((max_side, max_side))
            target_path = out_dir / f"frame-at-{target:.1f}s.jpg"
            image.convert("RGB").save(target_path, "JPEG", quality=85)
            frame_paths.append(str(target_path))
    finally:
        container.close()
    return frame_paths


def settings_for_request(
    base: AppSettings,
    *,
    model: str | None = None,
) -> AppSettings:
    snapshot = replace(base)
    if model:
        preset = get_preset(model)
        snapshot.model = preset.key
        snapshot.model_id = preset.model_id
    # Agent transcription cannot trigger publishing side effects even if the
    # user's interactive dictation settings enable them.
    snapshot.copy_to_clipboard = False
    snapshot.paste_to_active_input = False
    # The Silero VAD gate is tuned for clean dictation speech and rejects
    # vocoded/sung vocals over music, so file transcription does not use it.
    snapshot.vad_filter = False
    return snapshot


def transcribe_audio_file(
    audio_path: str | Path,
    *,
    settings: AppSettings,
    engine: EngineManager,
) -> dict[str, Any]:
    started = time.perf_counter()
    path = validate_audio_path(audio_path)
    container_kind = "video" if path.suffix.lower() in VIDEO_EXTENSIONS else "audio"
    samples, sample_rate = load_audio_file(path)
    duration_seconds = float(samples.size / sample_rate) if sample_rate else 0.0
    audio_track = True
    if container_kind == "video" and samples.size == 0:
        # Silent video (no audio track): report it explicitly instead of
        # pretending there was nothing to transcribe.
        audio_track = False
    trimmed = trim_silence(
        samples,
        sample_rate=sample_rate,
        sensitivity=settings.vad_sensitivity,
    )
    if trimmed.size == 0:
        text = ""
    else:
        raw_text = engine.transcribe(trimmed, sample_rate, settings)
        text = (
            postprocess(raw_text)
            if settings.postprocess_text
            else (raw_text or "").strip()
        )
    # Timestamped segments come from the full (untrimmed) audio so the
    # reported times match the original file; only engines that can produce
    # them (whisper) fill this in.
    segments = None
    segment_getter = getattr(engine, "transcribe_segments", None)
    if segment_getter is not None and samples.size > 0:
        segments = segment_getter(samples, sample_rate, settings)
    return {
        "text": text,
        "model": settings.model,
        "engine": engine.kind,
        "sample_rate": sample_rate,
        "duration_seconds": duration_seconds,
        "container": container_kind,
        "audioTrack": audio_track,
        "segments": segments,
        "elapsed_ms": round((time.perf_counter() - started) * 1000),
    }

