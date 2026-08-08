from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field


_WIN_VK_CODES = {0x5B, 0x5C}
_WIN32_PRESS_MESSAGES = {0x0100, 0x0104}
_WIN32_RELEASE_MESSAGES = {0x0101, 0x0105}
_WIN32_INJECTED_FLAGS = 0x00000010 | 0x00000002


@dataclass
class HotkeyState:
    required_keys: set[str]
    on_start: Callable[[], None]
    on_stop: Callable[[], None]
    pressed_keys: set[str] = field(default_factory=set)
    active: bool = False
    now: Callable[[], float] = time.monotonic
    ignore_releases_until: float = 0.0

    def press(self, key: str) -> None:
        self.pressed_keys.add(key)
        if not self.active and self.required_keys.issubset(self.pressed_keys):
            self.active = True
            self.on_start()

    def release(self, key: str) -> None:
        if key in self.required_keys and self.now() < self.ignore_releases_until:
            return
        self.pressed_keys.discard(key)
        if self.active and not self.required_keys.issubset(self.pressed_keys):
            self.active = False
            self.on_stop()

    def ignore_releases_for(self, seconds: float) -> None:
        self.ignore_releases_until = max(
            self.ignore_releases_until,
            self.now() + seconds,
        )


def parse_hotkey(value: str) -> set[str]:
    aliases = {
        "control": "ctrl",
        "cmd": "win",
        "windows": "win",
        "super": "win",
    }
    keys = set()
    for part in value.lower().replace(" ", "").split("+"):
        if not part:
            continue
        keys.add(aliases.get(part, part))
    return keys


class GlobalHotkeyListener:
    def __init__(
        self,
        hotkey: str,
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
        suppress: bool = False,
    ) -> None:
        self.state = HotkeyState(
            required_keys=parse_hotkey(hotkey),
            on_start=on_start,
            on_stop=on_stop,
        )
        self.suppress = suppress
        self.listener = None
        self._suppress_win_release = False

    def start(self) -> None:
        if self.listener is not None:
            return
        try:
            from pynput import keyboard
        except ImportError as exc:
            raise RuntimeError(
                "pynput is not installed, so global hotkeys are unavailable."
            ) from exc

        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
            suppress=self.suppress,
            win32_event_filter=self._win32_event_filter,
        )
        self.listener.start()

    def stop(self) -> None:
        if self.listener is not None:
            self.listener.stop()
            self.listener = None

    def ignore_releases_for(self, seconds: float) -> None:
        self.state.ignore_releases_for(seconds)

    def _on_press(self, key, injected: bool = False) -> None:
        normalized = _normalize_key(key)
        if injected and normalized in self.state.required_keys:
            return
        self.state.press(normalized)
        if (
            normalized != "win"
            and "win" in self.state.required_keys
            and self.state.active
            and "win" in self.state.pressed_keys
        ):
            self._suppress_win_release = True

    def _on_release(self, key, injected: bool = False) -> None:
        normalized = _normalize_key(key)
        if injected and normalized in self.state.required_keys:
            return
        self.state.release(normalized)

    def _win32_event_filter(self, msg: int, data) -> bool:
        if "win" not in self.state.required_keys:
            return True
        if getattr(data, "vkCode", None) not in _WIN_VK_CODES:
            return True

        injected = bool(getattr(data, "flags", 0) & _WIN32_INJECTED_FLAGS)
        if injected:
            self._suppress_current_event()
            return False

        if msg in _WIN32_PRESS_MESSAGES:
            if self._other_required_keys_are_pressed():
                self.state.press("win")
                self._suppress_win_release = True
                self._suppress_current_event()
                return False
            return True

        if msg in _WIN32_RELEASE_MESSAGES:
            if (
                self._suppress_win_release
                or self.state.active
                or self._other_required_keys_are_pressed()
            ):
                self.state.release("win")
                self._suppress_win_release = False
                self._suppress_current_event()
                return False
            return True

        return True

    def _other_required_keys_are_pressed(self) -> bool:
        return all(
            key == "win" or key in self.state.pressed_keys
            for key in self.state.required_keys
        )

    def _suppress_current_event(self) -> None:
        if self.listener is not None and hasattr(self.listener, "suppress_event"):
            self.listener.suppress_event()


def _normalize_key(key) -> str:
    name = getattr(key, "name", None)
    char = getattr(key, "char", None)
    value = name or char or str(key)
    value = str(value).lower().replace("key.", "")
    if value in {"ctrl_l", "ctrl_r", "ctrl"}:
        return "ctrl"
    if value in {"cmd", "cmd_l", "cmd_r", "win", "win_l", "win_r"}:
        return "win"
    if value in {"shift_l", "shift_r"}:
        return "shift"
    if value in {"alt_l", "alt_r", "alt_gr"}:
        return "alt"
    return value
