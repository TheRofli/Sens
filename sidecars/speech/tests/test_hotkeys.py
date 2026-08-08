import unittest

from speech_app.hotkeys import GlobalHotkeyListener, HotkeyState


class FakeWinEvent:
    def __init__(self, vk_code: int, flags: int = 0) -> None:
        self.vkCode = vk_code
        self.flags = flags


class FakePynputListener:
    def __init__(self) -> None:
        self.suppressed = 0

    def suppress_event(self) -> None:
        self.suppressed += 1


class HotkeyStateTests(unittest.TestCase):
    def test_ctrl_win_starts_once_and_stops_on_release(self):
        events = []
        state = HotkeyState(
            required_keys={"ctrl", "win"},
            on_start=lambda: events.append("start"),
            on_stop=lambda: events.append("stop"),
        )

        state.press("ctrl")
        state.press("win")
        state.press("win")
        state.release("ctrl")

        self.assertEqual(events, ["start", "stop"])

    def test_injected_required_key_events_do_not_change_hotkey_state(self):
        events = []
        listener = GlobalHotkeyListener(
            "ctrl+win",
            on_start=lambda: events.append("start"),
            on_stop=lambda: events.append("stop"),
        )

        listener._on_press("ctrl", injected=True)
        listener._on_release("ctrl", injected=True)

        self.assertEqual(listener.state.pressed_keys, set())
        self.assertEqual(events, [])

    def test_win_press_is_suppressed_when_other_hotkey_key_is_down(self):
        events = []
        listener = GlobalHotkeyListener(
            "ctrl+win",
            on_start=lambda: events.append("start"),
            on_stop=lambda: events.append("stop"),
        )
        listener.listener = FakePynputListener()
        listener.state.press("ctrl")

        result = listener._win32_event_filter(0x0100, FakeWinEvent(0x5B))

        self.assertFalse(result)
        self.assertEqual(listener.listener.suppressed, 1)
        self.assertEqual(events, ["start"])

    def test_win_release_stays_suppressed_after_ctrl_releases_first(self):
        events = []
        listener = GlobalHotkeyListener(
            "ctrl+win",
            on_start=lambda: events.append("start"),
            on_stop=lambda: events.append("stop"),
        )
        listener.listener = FakePynputListener()
        listener.state.press("ctrl")
        listener._win32_event_filter(0x0100, FakeWinEvent(0x5B))

        listener.state.release("ctrl")
        result = listener._win32_event_filter(0x0101, FakeWinEvent(0x5B))

        self.assertFalse(result)
        self.assertEqual(listener.listener.suppressed, 2)
        self.assertEqual(events, ["start", "stop"])

    def test_standalone_windows_key_is_not_suppressed(self):
        listener = GlobalHotkeyListener(
            "ctrl+win",
            on_start=lambda: None,
            on_stop=lambda: None,
        )
        listener.listener = FakePynputListener()

        result = listener._win32_event_filter(0x0100, FakeWinEvent(0x5B))

        self.assertTrue(result)
        self.assertEqual(listener.listener.suppressed, 0)

    def test_release_without_active_recording_does_not_stop(self):
        events = []
        state = HotkeyState(
            required_keys={"ctrl", "win"},
            on_start=lambda: events.append("start"),
            on_stop=lambda: events.append("stop"),
        )

        state.press("ctrl")
        state.release("ctrl")

        self.assertEqual(events, [])

    def test_can_ignore_synthetic_required_key_release_briefly(self):
        events = []
        current_time = [10.0]
        state = HotkeyState(
            required_keys={"ctrl", "win"},
            on_start=lambda: events.append("start"),
            on_stop=lambda: events.append("stop"),
            now=lambda: current_time[0],
        )

        state.press("ctrl")
        state.press("win")
        state.ignore_releases_for(0.2)
        state.release("ctrl")
        current_time[0] += 0.3
        state.release("ctrl")

        self.assertEqual(events, ["start", "stop"])


if __name__ == "__main__":
    unittest.main()
