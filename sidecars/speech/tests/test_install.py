import hashlib
import io
import os
import shutil
import sys
import tarfile
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from unittest import mock

from speech_app.engines import install as install_module
from speech_app.engines.install import install_archive_model, install_whisper_model
from speech_app.models import get_preset


def _build_archive(path: Path, preset, *, unsafe: bool = False) -> None:
    with tarfile.open(path, "w:bz2") as bundle:
        for relative in preset.required_files:
            name = "../escape" if unsafe else f"pack/{relative}"
            payload = b"model-data"
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            bundle.addfile(info, io.BytesIO(payload))
            if unsafe:
                break


class ArchiveInstallTests(unittest.TestCase):
    def test_complete_verified_partial_is_reused_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "model.part"
            target.write_bytes(b"complete")
            preset = replace(
                get_preset("gigaam"),
                download_bytes=target.stat().st_size,
                download_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            )
            with mock.patch("urllib.request.urlopen") as open_url:
                install_module._download_archive(preset, target)
            open_url.assert_not_called()

    def test_corrupt_complete_partial_is_deleted_before_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "model.part"
            target.write_bytes(b"corrupt")
            preset = replace(
                get_preset("gigaam"),
                download_bytes=target.stat().st_size,
                download_sha256=hashlib.sha256(b"expected").hexdigest(),
            )
            with mock.patch(
                "urllib.request.urlopen", side_effect=RuntimeError("offline")
            ):
                with self.assertRaisesRegex(RuntimeError, "offline"):
                    install_module._download_archive(preset, target)
            self.assertFalse(target.exists())

    def test_verified_archive_is_staged_marked_and_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            source = temp / "source.tar.bz2"
            base = get_preset("gigaam")
            _build_archive(source, base)
            preset = replace(
                base,
                download_url="https://example.test/model.tar.bz2",
                download_bytes=source.stat().st_size,
                download_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            )
            model_root = temp / "models"

            def provide_archive(_preset, destination):
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)

            with mock.patch.dict(
                os.environ, {"SPEECH_MODELS_DIR": str(model_root)}
            ), mock.patch.object(
                install_module, "_download_archive", side_effect=provide_archive
            ) as download:
                self.assertEqual(install_archive_model(preset), 0)
                self.assertEqual(install_archive_model(preset), 0)

            destination = model_root / "gigaam"
            self.assertTrue((destination / "INSTALLED.json").is_file())
            self.assertTrue((destination / "encoder.int8.onnx").is_file())
            self.assertEqual(download.call_count, 1)
            self.assertFalse(list(model_root.glob(".gigaam-install-*")))

    def test_install_reports_atomic_phase_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            source = temp / "source.tar.bz2"
            base = get_preset("gigaam")
            _build_archive(source, base)
            preset = replace(
                base,
                download_url="https://example.test/model.tar.bz2",
                download_bytes=source.stat().st_size,
                download_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            )
            phases = []

            def provide_archive(_preset, destination, progress):
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                progress({"phase": "downloading", "bytes_present": source.stat().st_size})

            with mock.patch.dict(
                os.environ, {"SPEECH_MODELS_DIR": str(temp / "models")}
            ), mock.patch.object(
                install_module, "_download_archive", side_effect=provide_archive
            ):
                self.assertEqual(
                    install_archive_model(
                        preset, lambda update: phases.append(update["phase"])
                    ),
                    0,
                )
            self.assertEqual(phases, ["downloading", "installing", "ready"])

    def test_bad_digest_never_promotes_partial_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            source = temp / "source.tar.bz2"
            base = get_preset("gigaam")
            _build_archive(source, base)
            preset = replace(
                base,
                download_url="https://example.test/model.tar.bz2",
                download_bytes=source.stat().st_size,
                download_sha256="0" * 64,
            )
            model_root = temp / "models"

            def provide_bad(_preset, destination):
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                raise RuntimeError("digest mismatch")

            with mock.patch.dict(
                os.environ, {"SPEECH_MODELS_DIR": str(model_root)}
            ), mock.patch.object(
                install_module, "_download_archive", side_effect=provide_bad
            ):
                self.assertEqual(install_archive_model(preset), 1)
            self.assertFalse((model_root / "gigaam").exists())

    def test_safe_extract_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            archive = temp / "unsafe.tar.bz2"
            _build_archive(archive, get_preset("gigaam"), unsafe=True)
            destination = temp / "extract"
            destination.mkdir()
            with self.assertRaisesRegex(RuntimeError, "unsafe path"):
                install_module._safe_extract(archive, destination)
            self.assertFalse((temp / "escape").exists())


class WhisperInstallTests(unittest.TestCase):
    def test_pinned_download_is_marked_and_idempotent(self):
        preset = get_preset("whisper")
        calls = []

        def fake_download(model, *, output_dir, revision, use_auth_token):
            calls.append((model, revision, use_auth_token))
            root = Path(output_dir)
            for relative in preset.required_files:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")

        package = ModuleType("faster_whisper")
        utils = ModuleType("faster_whisper.utils")
        utils.download_model = fake_download
        package.utils = utils
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ, {"SPEECH_MODELS_DIR": tmp}
        ), mock.patch.dict(
            sys.modules,
            {"faster_whisper": package, "faster_whisper.utils": utils},
        ):
            self.assertEqual(install_whisper_model(preset), 0)
            self.assertEqual(install_whisper_model(preset), 0)
            marker = Path(tmp) / "whisper" / "INSTALLED.json"
            self.assertTrue(marker.is_file())
        self.assertEqual(calls, [("small", preset.revision, False)])

    def test_install_model_dispatches_supported_local_presets(self):
        with mock.patch.object(
            install_module, "install_archive_model", return_value=0
        ) as archive, mock.patch.object(
            install_module, "install_whisper_model", return_value=0
        ) as whisper, mock.patch.object(
            install_module, "install_vad_model", return_value=0
        ) as vad:
            self.assertEqual(install_module.install_model("qwen"), 0)
            self.assertEqual(install_module.install_model("gigaam"), 0)
            self.assertEqual(install_module.install_model("whisper"), 0)
        self.assertEqual(archive.call_count, 2)
        whisper.assert_called_once()
        self.assertEqual(vad.call_count, 3)


if __name__ == "__main__":
    unittest.main()
