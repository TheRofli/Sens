import unittest

from speech_app.models import (
    MODELS,
    UnknownModel,
    available_presets,
    get_preset,
    normalize_model_key,
    resolve_engine,
    resolve_model_id,
)
from speech_app.settings import AppSettings


class ModelRegistryTests(unittest.TestCase):
    def test_registry_exposes_only_supported_presets(self):
        self.assertEqual(set(MODELS), {"qwen", "gigaam", "whisper", "remote"})

    def test_presets_have_distinct_engines_and_ids(self):
        self.assertEqual(
            {preset.engine for preset in MODELS.values()},
            {"qwen", "gigaam", "whisper", "remote"},
        )
        ids = [preset.model_id for preset in MODELS.values()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_archive_presets_are_pinned_and_verified(self):
        for key in ("qwen", "gigaam"):
            preset = get_preset(key)
            self.assertTrue(preset.download_url.startswith("https://"))
            self.assertEqual(len(preset.download_sha256), 64)
            self.assertGreater(preset.download_bytes, 1_000_000)
            self.assertTrue(preset.required_files)

    def test_get_preset_normalizes_legacy_keys_but_does_not_list_them(self):
        self.assertEqual(get_preset("parakeet").key, "qwen")
        self.assertEqual(get_preset("whisper-ru").key, "whisper")
        self.assertEqual(normalize_model_key("parakeet"), "qwen")
        with self.assertRaises(UnknownModel):
            get_preset("does-not-exist")

    def test_available_presets_is_stable_ordered(self):
        self.assertEqual(
            [preset.key for preset in available_presets()],
            ["qwen", "gigaam", "whisper", "remote"],
        )

    def test_resolve_engine_and_model_id(self):
        self.assertEqual(resolve_engine(AppSettings(model="qwen")), "qwen")
        self.assertEqual(resolve_engine(AppSettings(model="gigaam")), "gigaam")
        self.assertEqual(resolve_engine(AppSettings(model="whisper")), "whisper")
        self.assertEqual(resolve_engine(AppSettings(model="remote")), "remote")
        self.assertEqual(resolve_engine(AppSettings(model="legacy-key")), "qwen")
        self.assertEqual(
            resolve_model_id(AppSettings(model="whisper")),
            "Systran/faster-whisper-small",
        )
        self.assertEqual(
            resolve_model_id(AppSettings(model="legacy-key")),
            "Qwen/Qwen3-ASR-0.6B",
        )


if __name__ == "__main__":
    unittest.main()
