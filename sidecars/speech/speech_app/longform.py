"""Shared long-form audio helpers for local ASR engines."""

from __future__ import annotations


def _normalized_word(word: str) -> str:
    return word.strip(".,!?;:…\"'()[]«»—-").casefold()


def merge_overlapping_transcripts(parts: list[str]) -> str:
    """Join overlapping chunk transcripts without duplicating their seam.

    At least two normalized words must match. This avoids deleting a genuine
    repeated single word while still tolerating capitalization and punctuation
    differences between adjacent recognizer calls.
    """
    merged = ""
    for part in (value.strip() for value in parts):
        if not part:
            continue
        if not merged:
            merged = part
            continue
        left = merged.split()
        right = part.split()
        match = 0
        for size in range(min(len(left), len(right)), 1, -1):
            if all(
                _normalized_word(a) == _normalized_word(b)
                for a, b in zip(left[-size:], right[:size])
            ):
                match = size
                break
        if match:
            prefix = " ".join(left[:-match])
            merged = f"{prefix} {part}".strip()
        else:
            merged = f"{merged} {part}"
    return merged.strip()
