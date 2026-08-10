import pytest

from sight.capture import (
    TEXT_NODES_JS,
    _launch_browser,
    capture_request_id,
    normalize_capture_options,
)


def test_capture_identity_includes_reproducible_browser_settings() -> None:
    first = normalize_capture_options(
        {
            "viewport": {"width": 1280, "height": 720},
            "dpr": 1.5,
            "theme": "dark",
            "locale": "ru-RU",
            "waitUntil": "networkidle",
            "fullPage": True,
        }
    )
    second = normalize_capture_options(
        {**first, "viewport": {"width": 1440, "height": 900}}
    )

    assert capture_request_id("https://example.com", first) == capture_request_id(
        "https://example.com", first
    )
    assert capture_request_id("https://example.com", first) != capture_request_id(
        "https://example.com", second
    )
    assert first["viewport"] == {"width": 1280, "height": 720}


def test_capture_rejects_unbounded_or_unknown_settings() -> None:
    with pytest.raises(ValueError, match="viewport width"):
        normalize_capture_options({"viewport": {"width": 99999, "height": 720}})
    with pytest.raises(ValueError, match="waitUntil"):
        normalize_capture_options({"waitUntil": "forever"})


def test_capture_treats_serialized_optional_nulls_as_defaults() -> None:
    settings = normalize_capture_options(
        {
            "viewport": None,
            "dpr": None,
            "theme": None,
            "locale": None,
            "waitUntil": None,
            "fullPage": None,
            "timeoutMs": None,
            "settleMs": None,
            "scrollSteps": None,
        }
    )

    assert settings == {
        "viewport": {"width": 1440, "height": 900},
        "dpr": 1.0,
        "theme": "light",
        "locale": "en-US",
        "waitUntil": "networkidle",
        "fullPage": False,
        "timeoutMs": 30_000,
        "settleMs": 250,
        "scrollSteps": 0,
    }

    with pytest.raises(ValueError, match="viewport must be an object"):
        normalize_capture_options({"viewport": '{"width":1440,"height":900}'})


def test_capture_prefers_system_edge_before_a_playwright_download() -> None:
    calls = []

    class Chromium:
        def launch(self, **kwargs):
            calls.append(kwargs)
            return "browser"

    assert _launch_browser(Chromium()) == "browser"
    assert calls == [{"channel": "msedge", "headless": True}]


def test_capture_uses_opt_in_semantic_text_box_for_transformed_live_glyphs() -> None:
    assert "[data-sens-text-box=\"true\"]" in TEXT_NODES_JS
    assert "semanticParent.getBoundingClientRect()" in TEXT_NODES_JS
    assert "seenSemanticParents.has(semanticParent)" in TEXT_NODES_JS
    assert "semanticParent.textContent" in TEXT_NODES_JS
