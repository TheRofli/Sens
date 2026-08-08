from __future__ import annotations

from collections.abc import Callable

from .history import TranscriptEntry, TranscriptHistory
from .settings import AppSettings


class TranscriptPublisher:
    def __init__(
        self,
        history: TranscriptHistory,
        set_clipboard: Callable[[str], None],
        paste_active_input: Callable[[str], None],
    ) -> None:
        self.history = history
        self.set_clipboard = set_clipboard
        self.paste_active_input = paste_active_input

    def publish(
        self, transcript: str, settings: AppSettings
    ) -> TranscriptEntry | None:
        text = transcript.strip()
        if not text:
            return None

        entry = self.history.add(text)
        if settings.copy_to_clipboard:
            self.set_clipboard(text)
        if settings.paste_to_active_input:
            self.paste_active_input(text)
        return entry
