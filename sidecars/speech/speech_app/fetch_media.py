"""Fetch media from a video URL (YouTube etc.) for local analysis.

Downloads the audio and video streams separately with yt-dlp (no ffmpeg
needed: PyAV and the vision provider consume them independently) and caches
them by video id so repeat requests do not re-download.

The media is fetched for personal analysis by Sens; it is not a general
purpose downloader.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

MAX_MEDIA_BYTES = 512 * 1024 * 1024

_YOUTUBE_PATTERNS = (
    re.compile(r"(?:youtube\.com|youtu\.be)/(?:watch\?v=|shorts/|embed/|live/)?([\w-]{11})"),
    re.compile(r"youtu\.be/([\w-]{11})"),
)


def extract_video_id(url: str) -> str | None:
    """Best-effort YouTube video id from a URL; None when not a YouTube link."""
    for pattern in _YOUTUBE_PATTERNS:
        match = pattern.search(url)
        if match:
            return match.group(1)
    return None


def _ydl_options() -> dict[str, Any]:
    return {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "socket_timeout": 20,
        "retries": 3,
    }


def _probe(url: str) -> dict[str, Any]:
    """Metadata-only pass; raises when the URL cannot be resolved."""
    import yt_dlp

    options = {**_ydl_options(), "skip_download": True}
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=False)
    if not info:
        raise ValueError("Could not resolve video from URL")
    return info


def _download(url: str, out_dir: Path, format_spec: str) -> str | None:
    import yt_dlp

    options = {
        **_ydl_options(),
        "format": format_spec,
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.download([url])
    for candidate in sorted(out_dir.iterdir()):
        if candidate.is_file() and candidate.suffix.lower() in {
            ".mp4", ".webm", ".mkv", ".mov", ".m4a", ".opus", ".mp3", ".flac",
        }:
            return str(candidate)
    return None


def fetch_video(url: str, cache_dir: str | Path) -> dict[str, Any]:
    """Fetch a video by URL into the cache; returns metadata and local paths.

    The returned dict has: id, title, durationSeconds, channel, audioPath,
    videoPath, cached. ``audioPath``/``videoPath`` may be null when the
    matching stream is unavailable.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    info = _probe(url)
    video_id = str(info.get("id") or extract_video_id(url) or "").strip()
    if not video_id:
        raise ValueError("Could not determine video id from URL")
    title = str(info.get("title") or "").strip()
    duration = int(info.get("duration") or 0)
    channel = str(info.get("channel") or info.get("uploader") or "").strip()

    dest = cache_dir / video_id
    info_path = dest / "info.json"
    if info_path.exists():
        try:
            meta = json.loads(info_path.read_text(encoding="utf-8"))
            if meta.get("id") == video_id:
                return {**meta, "cached": True}
        except (OSError, json.JSONDecodeError):
            pass

    expected = int(info.get("filesize") or info.get("filesize_approx") or 0)
    if expected > MAX_MEDIA_BYTES:
        raise ValueError(
            f"Media is too large ({expected // (1024 * 1024)} MiB > "
            f"{MAX_MEDIA_BYTES // (1024 * 1024)} MiB limit)"
        )

    started = time.perf_counter()
    dest.mkdir(parents=True, exist_ok=True)
    audio_path = _download(url, dest, "bestaudio/best")
    video_path = _download(url, dest, "bestvideo/best")
    meta = {
        "id": video_id,
        "title": title,
        "durationSeconds": duration,
        "channel": channel,
        "audioPath": audio_path,
        "videoPath": video_path,
        "fetchedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsedMs": round((time.perf_counter() - started) * 1000),
    }
    info_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**meta, "cached": False}
