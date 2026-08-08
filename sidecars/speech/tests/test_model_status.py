import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from speech_app.model_status import ModelStatus, find_model_status_for_preset
from speech_app.models import get_preset


def _write_complete_model(root: Path, key: str, *, marker: bool = True) -> Path:
    preset = get_preset(key)
    destination = root / key
    for relative in preset.required_files:
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * 1024)
    if marker:
        (destination / "INSTALLED.json").write_text(
            json.dumps(
                {
                    "preset": preset.key,
                    "revision": preset.revision or None,
                    "archive_sha256": preset.download_sha256 or None,
                }
            ),
            encoding="utf-8",
        )
    return destination


class ModelStatusTests(unittest.TestCase):
    def test_reports_missing_model_when_directory_is_absent(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ, {"SPEECH_MODELS_DIR": tmp}
        ):
            status = find_model_status_for_preset(get_preset("qwen"))
        self.assertFalse(status.installed)
        self.assertEqual(status.label, "Not installed")

    def test_reports_verified_complete_model_and_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_complete_model(root, "gigaam")
            with mock.patch.dict(os.environ, {"SPEECH_MODELS_DIR": tmp}):
                status = find_model_status_for_preset(get_preset("gigaam"))
        self.assertTrue(status.installed)
        self.assertEqual(status.snapshot, get_preset("gigaam").download_sha256)
        self.assertGreater(status.size_mb, 0)

    def test_missing_marker_or_required_file_is_not_installed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = _write_complete_model(root, "whisper", marker=False)
            with mock.patch.dict(os.environ, {"SPEECH_MODELS_DIR": tmp}):
                self.assertFalse(
                    find_model_status_for_preset(get_preset("whisper")).installed
                )
                _write_complete_model(root, "whisper")
                (destination / "model.bin").unlink()
                self.assertFalse(
                    find_model_status_for_preset(get_preset("whisper")).installed
                )

    def test_mismatched_revision_or_digest_is_not_installed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = _write_complete_model(root, "whisper")
            marker = destination / "INSTALLED.json"
            metadata = json.loads(marker.read_text(encoding="utf-8"))
            metadata["revision"] = "wrong"
            marker.write_text(json.dumps(metadata), encoding="utf-8")
            with mock.patch.dict(os.environ, {"SPEECH_MODELS_DIR": tmp}):
                self.assertFalse(
                    find_model_status_for_preset(get_preset("whisper")).installed
                )

    def test_remote_is_available_without_local_files(self):
        status = find_model_status_for_preset(get_preset("remote"))
        self.assertTrue(status.installed)
        self.assertEqual(status.snapshot, "remote")

    def test_resource_status_formats_gigabytes(self):
        status = ModelStatus(True, "abc", Path("model"), 1536.0)
        self.assertEqual(status.size_label, "1.50 GB")


if __name__ == "__main__":
    unittest.main()
