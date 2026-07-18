import json
import os
import tempfile
import unittest
import unittest.mock

from speech_app.engines import install as install_module
from speech_app.engines.install import install_whisper_model
from speech_app.models import get_preset


class WhisperInstallTests(unittest.TestCase):
    def test_writes_marker_and_keeps_weights_after_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            with unittest.mock.patch.dict(os.environ, {"SPEECH_HOME": tmp}):
                preset = get_preset("whisper-ru")

                def fake_convert(model_id, out_dir):
                    # Simulate ct2 output: a weights file lands in out_dir.
                    out_dir.mkdir(parents=True, exist_ok=True)
                    (out_dir / "model.bin").write_bytes(b"x" * 1024)
                    return 0

                with unittest.mock.patch.object(
                    install_module, "_run_ct2_converter", side_effect=fake_convert
                ):
                    rc = install_whisper_model(preset)

                self.assertEqual(rc, 0)
                marker = (
                    os.path.join(tmp, "models", "whisper", "whisper-ru", "INSTALLED.json")
                )
                self.assertTrue(os.path.exists(marker))
                with open(marker, encoding="utf-8") as handle:
                    payload = json.load(handle)
                self.assertEqual(payload["preset"], "whisper-ru")
                # Idempotent: a second run should not reconvert.
                with unittest.mock.patch.object(
                    install_module, "_run_ct2_converter"
                ) as conv:
                    rc2 = install_whisper_model(preset)
                self.assertEqual(rc2, 0)
                conv.assert_not_called()

    def test_install_model_dispatches_to_whisper_for_whisper_preset(self):
        with unittest.mock.patch.object(
            install_module, "install_whisper_model", return_value=0
        ) as whisper_install:
            rc = install_module.install_model("whisper-ru")
        self.assertEqual(rc, 0)
        whisper_install.assert_called_once()

    def test_install_model_dispatches_to_parakeet_for_parakeet_preset(self):
        with unittest.mock.patch.object(
            install_module, "install_parakeet_model", return_value=0
        ) as parakeet_install:
            rc = install_module.install_model("parakeet")
        self.assertEqual(rc, 0)
        parakeet_install.assert_called_once_with("nvidia/parakeet-tdt-0.6b-v3")

    def test_converter_uses_in_process_transformers_converter_not_path_cli(self):
        """Conversion must run via ctranslate2's TransformersConverter inside
        the current process, never via a PATH-resolved ct2-transformers-converter
        binary (which can belong to an unrelated venv). Regression guard."""
        import sys
        import tempfile
        from pathlib import Path

        fake_converter = unittest.mock.MagicMock()
        fake_converter.convert.return_value = str("ok")

        with tempfile.TemporaryDirectory() as out:
            out_dir = Path(out) / "model"
            # Ensure the `import transformers` check inside _run_ct2_converter
            # succeeds even on interpreters without transformers installed.
            sys.modules.setdefault("transformers", unittest.mock.MagicMock())
            try:
                with unittest.mock.patch(
                    "ctranslate2.converters.TransformersConverter",
                    return_value=fake_converter,
                ) as ctor:
                    rc = install_module._run_ct2_converter("some/model", out_dir)
            finally:
                sys.modules.pop("transformers", None)
            self.assertEqual(rc, 0)
            ctor.assert_called_once_with("some/model")
            fake_converter.convert.assert_called_once()
            # Confirm INT8 quantization and force are requested.
            _args, kwargs = fake_converter.convert.call_args
            self.assertEqual(kwargs.get("quantization"), "int8")
            self.assertTrue(kwargs.get("force"))


if __name__ == "__main__":
    unittest.main()
