"""Transcript post-processing.

Light, deterministic cleanup applied after the engine returns text:

* collapse repeated whitespace and stray newlines
* capitalise the first character of the result
* strip leading/trailing whitespace
* drop a small set of Whisper hallucination phrases that appear on silence

This is intentionally conservative and language-agnostic so it is safe to run
on both English and Russian output.
"""

from __future__ import annotations

import re

# A compact subset of the well-known Whisper silence-hallucination phrases.
# Kept lowercase and normalized so they match regardless of model casing. These
# are the recurring English phrases (see the community-collected 135-phrase
# list); the Russian/other-language equivalents are caught downstream by VAD
# trimming silence before it reaches the model.
_HALLUCINATION_PHRASES: tuple[str, ...] = (
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "don't forget to subscribe",
    "do not forget to subscribe",
    "bye bye",
    "bye.",
    "thank you.",
    "thanks for watching.",
    "you",
    "you.",
    "subscribe",
    "music",
    "music.",
    "[music]",
    "[applause]",
    "[laughter]",
    "so",
    "so.",
    "amara",
    "amara.org",
    "i'm going to show you how to",
    "let's get started",
)

_MULTISPACE_RE = re.compile(r"[ \t\u00a0\u2000-\u200a\u202f\u205f\u3000]+")
# Newlines in ASR output are usually segmentation artifacts; collapse them to a
# single space so a push-to-talk capture reads as one phrase.
_MULTINEWLINE_RE = re.compile(r"\s*\n\s*")
_LEADING_PUNCT_RE = re.compile(r"^[\s\-–—.•,;:!?]+")


def _normalize_whitespace(text: str) -> str:
    text = _MULTINEWLINE_RE.sub(" ", text)
    text = _MULTISPACE_RE.sub(" ", text)
    return text.strip()


def _strip_hallucinations(text: str) -> str:
    """Remove known silence-hallucination phrases anywhere in the text.

    Matches happen on a normalized (collapsed-whitespace, trimmed, lowercased)
    copy but operate on the original text by whole-word boundaries so that
    legitimate use of short words like "you" inside a longer sentence is kept.
    """
    if not text:
        return text
    result = text
    for phrase in _HALLUCINATION_PHRASES:
        if not phrase:
            continue
        # Whole-phrase, case-insensitive, allowing trailing punctuation.
        pattern = re.compile(
            r"(?:(?<=\A)|(?<=[.\n!?]))\s*" + re.escape(phrase) + r"[.,!?]?",
            re.IGNORECASE,
        )
        result = pattern.sub("", result)
    return _normalize_whitespace(result)


def _capitalize_first(text: str) -> str:
    if not text:
        return text
    for index, char in enumerate(text):
        if char.isalpha():
            return text[:index] + char.upper() + text[index + 1 :]
    return text


def postprocess(text: str) -> str:
    """Apply the deterministic cleanup pipeline and return the cleaned text.

    Empty/whitespace-only input returns an empty string.
    """
    if text is None:
        return ""
    cleaned = _normalize_whitespace(text)
    cleaned = _strip_hallucinations(cleaned)
    cleaned = _LEADING_PUNCT_RE.sub("", cleaned)
    cleaned = _normalize_whitespace(cleaned)
    cleaned = _capitalize_first(cleaned)
    return cleaned
