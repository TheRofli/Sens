import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from speech_app.fetch_media import extract_video_id, fetch_video


class VideoIdTests(unittest.TestCase):
    def test_extracts_id_from_watch_url(self):
        self.assertEqual(
            extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ"),
            "dQw4w9WgXcQ",
        )

    def test_extracts_id_from_short_url(self):
        self.assertEqual(extract_video_id("https://youtu.be/dQw4w9WgXcQ"), "dQw4w9WgXcQ")

    def test_extracts_id_from_shorts(self):
        self.assertEqual(
            extract_video_id("https://youtube.com/shorts/dQw4w9WgXcQ?feature=share"),
            "dQw4w9WgXcQ",
        )

    def test_returns_none_for_foreign_url(self):
        self.assertIsNone(extract_video_id("https://example.com/video/123"))


class FetchVideoTests(unittest.TestCase):
    def test_fetch_downloads_then_serves_from_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache"
            probe = {
                "id": "dQw4w9WgXcQ",
                "title": "Test clip",
                "duration": 42,
                "channel": "Some channel",
                "filesize": 10_000_000,
            }
            downloads = []

            def fake_download(url, out_dir, format_spec):
                suffix = ".m4a" if format_spec.startswith("bestaudio") else ".mp4"
                target = out_dir / f"dQw4w9WgXcQ{suffix}"
                target.write_bytes(b"fake media")
                downloads.append(format_spec)
                return str(target)

            with patch("speech_app.fetch_media._probe", return_value=probe), patch(
                "speech_app.fetch_media._download", side_effect=fake_download
            ):
                first = fetch_video("https://youtu.be/dQw4w9WgXcQ", cache)
                second = fetch_video("https://youtu.be/dQw4w9WgXcQ", cache)

            self.assertFalse(first["cached"])
            self.assertTrue(second["cached"])
            self.assertEqual(first["title"], "Test clip")
            self.assertEqual(first["durationSeconds"], 42)
            self.assertEqual(first["channel"], "Some channel")
            self.assertTrue(first["audioPath"].endswith(".m4a"))
            self.assertTrue(first["videoPath"].endswith(".mp4"))
            # Only the first call downloads; the second hits the cache.
            self.assertEqual(len(downloads), 2)

    def test_rejects_oversized_media(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache"
            probe = {"id": "abc123abc12", "title": "Big", "filesize": 2**31}
            with patch("speech_app.fetch_media._probe", return_value=probe):
                with self.assertRaises(ValueError):
                    fetch_video("https://youtu.be/abc123abc12", cache)


if __name__ == "__main__":
    unittest.main()
