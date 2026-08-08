import unittest

from speech_app.app import build_parser


class CliTests(unittest.TestCase):
    def test_removed_parakeet_command_is_not_available(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["parakeet", "install"])

    def test_diagnose_command_is_available(self):
        self.assertEqual(build_parser().parse_args(["diagnose"]).command, "diagnose")

    def test_run_can_open_window_on_start(self):
        args = build_parser().parse_args(["run", "--show-window"])
        self.assertEqual(args.command, "run")
        self.assertTrue(args.show_window)

    def test_model_install_command_parses_supported_preset(self):
        args = build_parser().parse_args(["model", "install", "whisper"])
        self.assertEqual(args.command, "model")
        self.assertEqual(args.model_command, "install")
        self.assertEqual(args.key, "whisper")

    def test_model_install_defaults_key_to_none(self):
        args = build_parser().parse_args(["model", "install"])
        self.assertIsNone(args.key)

    def test_model_list_command_is_available(self):
        args = build_parser().parse_args(["model", "list"])
        self.assertEqual(args.model_command, "list")


if __name__ == "__main__":
    unittest.main()
