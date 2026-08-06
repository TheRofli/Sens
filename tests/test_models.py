import unittest

from speech_app.models import (
    MODELS,
    UnknownModel,
    available_presets,
    get_preset,
    resolve_engine,
    resolve_model_id,
)
from speech_app.settings import AppSettings


class ModelRegistryTests(unittest.TestCase):
    def test_registry_has_parakeet_and_whisper_presets(self):
        self.assertIn("parakeet", MODELS)
        self.assertIn("whisper-ru", MODELS)
        self.assertIn("gigaam", MODELS)
        self.assertIn("remote", MODELS)

    def test_presets_have_distinct_engines_and_ids(self):
        engines = {preset.engine for preset in MODELS.values()}
        self.assertIn("parakeet", engines)
        self.assertIn("whisper", engines)
        self.assertIn("gigaam", engines)
        self.assertIn("remote", engines)
        ids = [preset.model_id for preset in MODELS.values()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_get_preset_returns_match_and_raises_for_unknown(self):
        self.assertEqual(get_preset("parakeet").key, "parakeet")
        with self.assertRaises(UnknownModel):
            get_preset("does-not-exist")

    def test_available_presets_is_stable_ordered(self):
        presets = available_presets()
        self.assertGreater(len(presets), 1)
        self.assertEqual(presets[0].key, "parakeet")
        # The remote API preset is the last, online option.
        self.assertEqual(presets[-1].key, "remote")

    def test_resolve_engine_for_known_model(self):
        self.assertEqual(resolve_engine(AppSettings(model="whisper-ru")), "whisper")
        self.assertEqual(resolve_engine(AppSettings(model="parakeet")), "parakeet")
        self.assertEqual(resolve_engine(AppSettings(model="gigaam")), "gigaam")
        self.assertEqual(resolve_engine(AppSettings(model="remote")), "remote")

    def test_resolve_engine_falls_back_to_parakeet_for_unknown_model(self):
        self.assertEqual(resolve_engine(AppSettings(model="legacy-key")), "parakeet")

    def test_resolve_model_id_uses_preset_then_falls_back_to_settings(self):
        preset_id = resolve_model_id(AppSettings(model="whisper-ru"))
        self.assertEqual(preset_id, "coriollon/whisper-large-v3-turbo-russian-codeswitch")
        # Unknown model -> fall back to the legacy model_id field.
        legacy = AppSettings(model="legacy-key", model_id="custom/parakeet")
        self.assertEqual(resolve_model_id(legacy), "custom/parakeet")


if __name__ == "__main__":
    unittest.main()
