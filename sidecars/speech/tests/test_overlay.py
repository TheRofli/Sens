import unittest

from speech_app.overlay import VoiceOverlay


class FakeWindow:
    def __init__(self) -> None:
        self.withdraw_count = 0

    def withdraw(self) -> None:
        self.withdraw_count += 1


class OverlayLifecycleTests(unittest.TestCase):
    def test_old_notice_timeout_does_not_hide_new_recording_overlay(self):
        overlay = VoiceOverlay.__new__(VoiceOverlay)
        overlay._show_token = 2
        overlay._running = True
        overlay.window = FakeWindow()

        overlay._hide_if_current(1)

        self.assertTrue(overlay._running)
        self.assertEqual(overlay.window.withdraw_count, 0)

    def test_matching_notice_timeout_hides_overlay(self):
        overlay = VoiceOverlay.__new__(VoiceOverlay)
        overlay._show_token = 2
        overlay._running = True
        overlay.window = FakeWindow()

        overlay._hide_if_current(2)

        self.assertFalse(overlay._running)
        self.assertEqual(overlay.window.withdraw_count, 1)


if __name__ == "__main__":
    unittest.main()
