import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from speech_app.portable import build_portable_env, portable_data_dir
from speech_app.settings import migrate_legacy_user_data


class PortableModeTests(unittest.TestCase):
    def test_data_dir_prefers_explicit_speech_data_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"

            with patch.dict(os.environ, {"SPEECH_DATA_DIR": str(data_dir)}):
                self.assertEqual(portable_data_dir(), data_dir)

    def test_build_portable_env_separates_code_data_and_models(self):
        code_root = Path("C:/Program Files/Sens/sidecars/speech")
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            models = Path(tmp) / "models"
            with patch.dict(
                os.environ,
                {
                    "SPEECH_DATA_DIR": str(data),
                    "SPEECH_MODELS_DIR": str(models),
                },
            ):
                env = build_portable_env(code_root)

        self.assertEqual(env["SPEECH_HOME"], str(code_root))
        self.assertEqual(env["SPEECH_DATA_DIR"], str(data))
        self.assertEqual(env["SPEECH_MODELS_DIR"], str(models))
        self.assertEqual(env["HF_HOME"], str(models / "huggingface"))
        self.assertEqual(env["HF_HUB_CACHE"], str(models / "huggingface" / "hub"))
        self.assertEqual(env["TORCH_HOME"], str(models / "torch"))

    def test_legacy_migration_only_copies_user_files_once(self):
        with tempfile.TemporaryDirectory() as legacy_tmp, tempfile.TemporaryDirectory() as target_tmp:
            legacy_root = Path(legacy_tmp)
            legacy_data = legacy_root / "data"
            legacy_data.mkdir()
            (legacy_data / "settings.json").write_text('{"model":"gigaam"}')
            (legacy_data / "history.jsonl").write_text('{"text":"old"}\n')
            (legacy_data / "sens-managed.token").write_text("secret")
            target = Path(target_tmp)
            with patch.dict(
                os.environ, {"SENS_LEGACY_SPEECH_ROOT": str(legacy_root)}
            ):
                migrated = migrate_legacy_user_data(target)
                second = migrate_legacy_user_data(target)
            self.assertEqual(migrated, ["settings.json", "history.jsonl"])
            self.assertEqual(second, [])
            self.assertEqual(
                (target / "settings.json").read_text(), '{"model":"gigaam"}'
            )
            self.assertFalse((target / "sens-managed.token").exists())


if __name__ == "__main__":
    unittest.main()
