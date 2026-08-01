import os
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from speech_app.model_status import (
    ModelStatus,
    find_gigaam_model_status,
    find_model_status,
    find_whisper_model_status,
)
from speech_app.models import get_preset


class ModelStatusTests(unittest.TestCase):
    def test_reports_missing_model_when_snapshot_dir_is_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            status = find_model_status(Path(tmp), "nvidia/parakeet-tdt-0.6b-v3")

        self.assertFalse(status.installed)
        self.assertEqual(status.label, "Not installed")

    def test_reports_installed_model_and_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = (
                root
                / "hub"
                / "models--nvidia--parakeet-tdt-0.6b-v3"
                / "snapshots"
                / "abc"
            )
            snapshot.mkdir(parents=True)
            (snapshot / "weights.bin").write_bytes(b"x" * 1024)

            status = find_model_status(root, "nvidia/parakeet-tdt-0.6b-v3")

        self.assertTrue(status.installed)
        self.assertEqual(status.snapshot, "abc")
        self.assertEqual(status.size_mb, 0.001)
        self.assertIn("Installed", status.label)

    def test_incomplete_snapshot_without_weights_reports_not_installed(self):
        """A snapshot with only config/tokenizer (no weight files) is an
        incomplete download and must NOT be reported as installed."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = (
                root
                / "hub"
                / "models--nvidia--parakeet-tdt-0.6b-v3"
                / "snapshots"
                / "abc"
            )
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}")
            (snapshot / "tokenizer.json").write_text("{}")

            status = find_model_status(root, "nvidia/parakeet-tdt-0.6b-v3")

        self.assertFalse(status.installed)

    def test_resource_status_formats_memory_and_cpu(self):
        status = ModelStatus(
            installed=True,
            snapshot="abc",
            path=Path("D:/Speech/model"),
            size_mb=1536.0,
        )

        self.assertEqual(status.size_label, "1.50 GB")


class WhisperModelStatusTests(unittest.TestCase):
    def test_reports_missing_when_dir_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            with unittest.mock.patch.dict(os.environ, {"SPEECH_HOME": tmp}):
                status = find_whisper_model_status(get_preset("whisper-ru"))
        self.assertFalse(status.installed)

    def test_reports_installed_when_weights_and_marker_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = (
                Path(tmp) / "models" / "whisper" / "whisper-ru"
            )
            model_dir.mkdir(parents=True)
            (model_dir / "model.bin").write_bytes(b"x" * (2 * 1024 * 1024))
            (model_dir / "INSTALLED.json").write_text("{}")
            with unittest.mock.patch.dict(os.environ, {"SPEECH_HOME": tmp}):
                status = find_whisper_model_status(get_preset("whisper-ru"))
        self.assertTrue(status.installed)
        self.assertEqual(status.snapshot, "whisper-ru")
        self.assertAlmostEqual(status.size_mb, 2.0, places=3)

    def test_reports_missing_when_marker_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = (
                Path(tmp) / "models" / "whisper" / "whisper-ru"
            )
            model_dir.mkdir(parents=True)
            (model_dir / "model.bin").write_bytes(b"x" * 1024)
            # No INSTALLED.json marker -> not considered installed.
            with unittest.mock.patch.dict(os.environ, {"SPEECH_HOME": tmp}):
                status = find_whisper_model_status(get_preset("whisper-ru"))
        self.assertFalse(status.installed)


class GigaAMModelStatusTests(unittest.TestCase):
    def test_reports_missing_when_dir_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            with unittest.mock.patch.dict(os.environ, {"SPEECH_HOME": tmp}):
                status = find_gigaam_model_status(get_preset("gigaam"))
        self.assertFalse(status.installed)

    def test_reports_missing_when_weights_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "models" / "gigaam" / "gigaam"
            model_dir.mkdir(parents=True)
            (model_dir / "modeling_gigaam.py").write_text("x")  # code but no weights
            with unittest.mock.patch.dict(os.environ, {"SPEECH_HOME": tmp}):
                status = find_gigaam_model_status(get_preset("gigaam"))
        self.assertFalse(status.installed)

    def test_reports_installed_when_weights_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "models" / "gigaam" / "gigaam"
            model_dir.mkdir(parents=True)
            (model_dir / "pytorch_model.bin").write_bytes(b"x" * (2 * 1024 * 1024))
            (model_dir / "modeling_gigaam.py").write_text("x")
            with unittest.mock.patch.dict(os.environ, {"SPEECH_HOME": tmp}):
                status = find_gigaam_model_status(get_preset("gigaam"))
        self.assertTrue(status.installed)
        self.assertEqual(status.snapshot, "gigaam")
        self.assertAlmostEqual(status.size_mb, 2.0, places=3)


if __name__ == "__main__":
    unittest.main()
