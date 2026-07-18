import unittest
import unittest.mock
from unittest.mock import patch

from speech_app.app import build_parser, install_parakeet_model


class CliTests(unittest.TestCase):
    def test_parakeet_install_command_is_available(self):
        args = build_parser().parse_args(["parakeet", "install"])

        self.assertEqual(args.command, "parakeet")
        self.assertEqual(args.parakeet_command, "install")

    def test_diagnose_command_is_available(self):
        args = build_parser().parse_args(["diagnose"])

        self.assertEqual(args.command, "diagnose")

    def test_run_can_open_window_on_start(self):
        args = build_parser().parse_args(["run", "--show-window"])

        self.assertEqual(args.command, "run")
        self.assertTrue(args.show_window)

    def test_model_install_command_parses_with_preset(self):
        args = build_parser().parse_args(["model", "install", "whisper-ru"])

        self.assertEqual(args.command, "model")
        self.assertEqual(args.model_command, "install")
        self.assertEqual(args.key, "whisper-ru")

    def test_model_install_command_defaults_key_to_none(self):
        args = build_parser().parse_args(["model", "install"])

        self.assertEqual(args.command, "model")
        self.assertEqual(args.model_command, "install")
        self.assertIsNone(args.key)

    def test_model_list_command_is_available(self):
        args = build_parser().parse_args(["model", "list"])

        self.assertEqual(args.command, "model")
        self.assertEqual(args.model_command, "list")

    def test_parakeet_install_uses_default_preset(self):
        """install_parakeet_model() delegates to install_model for parakeet."""
        with patch("builtins.print"), patch(
            "speech_app.engines.install.install_parakeet_model", return_value=0
        ) as install:
            exit_code = install_parakeet_model()

        self.assertEqual(exit_code, 0)
        install.assert_called_once_with("nvidia/parakeet-tdt-0.6b-v3")


if __name__ == "__main__":
    unittest.main()
