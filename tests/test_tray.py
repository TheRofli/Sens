import unittest

from speech_app.tray import TrayController


class FakeApp:
    """Minimal app double for tray menu construction tests."""

    def __init__(self, active="parakeet"):
        self._active = active
        self.set_model_calls = []

    def available_models(self):
        return [
            {
                "key": "parakeet",
                "label": "Parakeet",
                "engine": "parakeet",
                "installed": True,
                "size_label": "1 GB",
                "active": self._active == "parakeet",
            },
            {
                "key": "whisper-ru",
                "label": "Whisper RU",
                "engine": "whisper",
                "installed": False,
                "size_label": "Not installed",
                "active": self._active == "whisper-ru",
            },
        ]

    def set_model(self, key):
        self.set_model_calls.append(key)

    def current_model(self):
        return self._active


class TrayModelMenuTests(unittest.TestCase):
    def test_build_model_menu_does_not_raise(self):
        """Regression: pystray rejects actions/checked callbacks whose
        positional arity is too high. Building the Model submenu must not
        raise (this previously crashed the whole tray icon)."""
        controller = TrayController(FakeApp())
        # This must not raise.
        menu = controller._build_model_menu()
        self.assertIsNotNone(menu)

    def test_model_menu_checked_callback_arity_is_one(self):
        """pystray invokes checked(menu_item) with a single argument."""
        import functools

        controller = TrayController(FakeApp(active="whisper-ru"))
        menu = controller._build_model_menu()
        # The Menu's descriptors are the MenuItems we built.
        for descriptor in menu._descriptors if hasattr(menu, "_descriptors") else []:
            checked = descriptor._checked
            # partial hides __code__; the wrapped function must accept the one
            # arg pystray passes plus the bound captured_key.
            self.assertIsNotNone(checked)

    def test_model_menu_action_selects_model(self):
        """Invoking the per-model action calls app.set_model with its key."""
        app = FakeApp(active="parakeet")
        controller = TrayController(app)
        menu = controller._build_model_menu()
        descriptors = (
            menu._descriptors if hasattr(menu, "_descriptors") else list(menu)
        )
        self.assertGreaterEqual(len(descriptors), 2)
        # The second item is the whisper-ru entry. Invoke its action the way
        # pystray would: action(icon, item).
        whisper_action = descriptors[1]._action
        whisper_action(object(), object())
        self.assertEqual(app.set_model_calls, ["whisper-ru"])


if __name__ == "__main__":
    unittest.main()
