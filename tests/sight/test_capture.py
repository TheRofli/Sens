import hashlib
from pathlib import Path

import pytest

from sight.capture import (
    FONT_FACES_JS,
    TEXT_NODES_JS,
    VECTOR_ELEMENTS_JS,
    _best_effort_close,
    _freeze_visual_state,
    _guard_browser_request,
    _launch_browser,
    _navigate_page,
    _persist_source_font_assets,
    _persist_source_raster_assets,
    _persist_source_vector_assets,
    _prepare_network_capture,
    capture_request_id,
    normalize_capture_options,
    validate_network_url,
)


def test_text_node_capture_records_per_word_live_dom_ranges() -> None:
    assert "wordBoxes" in TEXT_NODES_JS
    assert "wordRange.setStart(node, start)" in TEXT_NODES_JS
    assert "wordRange.setEnd(node, start + match[0].length)" in TEXT_NODES_JS
    assert "wordRange.getBoundingClientRect()" in TEXT_NODES_JS


def test_capture_freezes_animated_visual_state_before_observation() -> None:
    class Page:
        def __init__(self) -> None:
            self.scripts = []

        def evaluate(self, script):
            self.scripts.append(script)
            return {
                "animationsObserved": 3,
                "animationsPaused": 3,
                "source": "observed",
                "method": "paused-web-animations-plus-capture-style",
            }

    page = Page()

    evidence = _freeze_visual_state(page)

    assert evidence["animationsPaused"] == 3
    assert len(page.scripts) == 1
    script = page.scripts[0]
    assert "document.getAnimations" in script
    assert "animation.pause()" in script
    assert "animation-play-state:paused" in script
    assert "transition-property:none" in script
    assert "requestAnimationFrame" in script


def test_capture_persists_only_large_visible_sanitized_source_vectors(tmp_path) -> None:
    assert "subpixel(rect.x)" in VECTOR_ELEMENTS_JS
    assert "Math.round(rect.x)" not in VECTOR_ELEMENTS_JS
    vectors = _persist_source_vector_assets(
        [
            {
                "domIndex": 42,
                "box": [100.1254, 100.2496, 500.8754, 600.5004],
                "visible": True,
                "markup": (
                    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10" '
                    'onload="alert(1)"><script>alert(1)</script>'
                    '<path id="letter" d="M0 0H10V10Z" fill="#111"/>'
                    '<use href="https://evil.example/vector.svg#letter"/>'
                    '</svg>'
                ),
            },
            {
                "domIndex": 43,
                "box": [-1000, 100, -600, 600],
                "visible": True,
                "markup": (
                    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
                    '<path d="M0 0H10V10Z"/></svg>'
                ),
            },
        ],
        {"width": 1440, "height": 900},
        tmp_path,
        no_store=False,
    )

    assert len(vectors) == 1
    [asset] = vectors
    assert asset["domIndex"] == 42
    assert asset["box"] == [100.125, 100.25, 500.875, 600.5]
    assert asset["mediaType"] == "image/svg+xml"
    assert asset["source"] == "observed"
    assert asset["method"] == "sanitized-live-dom-svg"
    assert asset["viewportCoverage"] == 1.0
    content = Path(asset["path"]).read_text(encoding="utf-8")
    assert hashlib.sha256(content.encode("utf-8")).hexdigest() == asset["sha256"]
    assert "<script" not in content
    assert "onload" not in content
    assert "evil.example" not in content
    assert "<path" in content


def test_capture_persists_only_referenced_browser_loaded_source_rasters(tmp_path) -> None:
    hero_bytes = b"fake-avif-hero-pixels"

    class Response:
        def __init__(self, url: str, content_type: str, content: bytes) -> None:
            self.url = url
            self.headers = {"content-type": content_type}
            self.content = content
            self.body_calls = 0

        def body(self) -> bytes:
            self.body_calls += 1
            return self.content

    hero = Response(
        "https://cdn.example/hero.avif",
        "image/avif; q=1",
        hero_bytes,
    )
    unreferenced = Response(
        "https://cdn.example/tracker.png",
        "image/png",
        b"not-a-visible-raster",
    )

    assets = _persist_source_raster_assets(
        [
            {
                "domIndex": 158,
                "kind": "img",
                "src": hero.url,
                "visible": True,
                "box": [-95, -274, 1542, 1464],
                "objectFit": "fill",
                "backgroundSize": "auto",
                "backdropColor": "rgb(220, 238, 255)",
            }
        ],
        [{"text": "SLUSH", "box": [345, 184, 1093, 593]}],
        {hero.url: hero, unreferenced.url: unreferenced},
        tmp_path,
        no_store=False,
    )

    assert len(assets) == 1
    [asset] = assets
    assert asset["sha256"] == hashlib.sha256(hero_bytes).hexdigest()
    assert asset["mediaType"] == "image/avif"
    assert asset["sizeBytes"] == len(hero_bytes)
    assert asset["box"] == [-95, -274, 1542, 1464]
    assert asset["overlappingLiveTextCount"] == 1
    assert asset["backdropColor"] == "rgb(220, 238, 255)"
    assert asset["source"] == "observed"
    assert asset["method"] == "playwright-response-body"
    assert Path(asset["path"]).is_file()
    assert Path(asset["path"]).read_bytes() == hero_bytes
    assert hero.body_calls == 1
    assert unreferenced.body_calls == 0


def test_capture_persists_loaded_font_faces_with_observed_css_identity(tmp_path) -> None:
    font_bytes = b"public-whyte-inktrap-medium-woff2"

    class Request:
        resource_type = "font"

    class Response:
        def __init__(self, url: str, content_type: str, content: bytes) -> None:
            self.url = url
            self.headers = {"content-type": content_type}
            self.content = content
            self.request = Request()
            self.body_calls = 0

        def body(self) -> bytes:
            self.body_calls += 1
            return self.content

    font_url = "https://cdn.example/whyte-inktrap-medium.woff2"
    # Several production CDNs label browser-loaded WOFF2 responses as generic
    # bytes even though Playwright reports the request as a font resource.
    response = Response(font_url, "application/octet-stream", font_bytes)
    unrelated = Response(
        "https://cdn.example/unreferenced.woff2",
        "font/woff2",
        b"unused-font",
    )

    assets = _persist_source_font_assets(
        {
            "faces": [
                {
                    "family": "Whyte Inktrap",
                    "style": "normal",
                    "weight": "500",
                    "stretch": "normal",
                    "sources": [font_url],
                    "status": "loaded",
                }
            ]
        },
        {font_url: response, unrelated.url: unrelated},
        tmp_path,
        no_store=False,
    )

    assert "document.fonts" in FONT_FACES_JS
    assert "CSSRule.FONT_FACE_RULE" in FONT_FACES_JS
    assert len(assets) == 1
    [asset] = assets
    assert asset["family"] == "Whyte Inktrap"
    assert asset["weight"] == "500"
    assert asset["style"] == "normal"
    assert asset["mediaType"] == "font/woff2"
    assert asset["format"] == "woff2"
    assert asset["source"] == "observed"
    assert asset["method"] == "playwright-loaded-font-response"
    assert Path(asset["path"]).read_bytes() == font_bytes
    assert asset["sha256"] == hashlib.sha256(font_bytes).hexdigest()
    assert response.body_calls == 1
    assert unrelated.body_calls == 0


def test_capture_cleanup_does_not_mask_the_primary_playwright_failure() -> None:
    class AlreadyStopped:
        def close(self) -> None:
            raise RuntimeError("Event loop is closed! Is Playwright already stopped?")

    _best_effort_close(AlreadyStopped())


def test_networkidle_timeout_falls_back_only_after_dom_is_ready() -> None:
    class NavigationTimeout(Exception):
        pass

    class Page:
        def __init__(self, ready: bool) -> None:
            self.ready = ready
            self.calls = []

        def goto(self, url, **kwargs):
            self.calls.append(("goto", url, kwargs))

        def wait_for_load_state(self, state, **kwargs):
            self.calls.append(("wait_for_load_state", state, kwargs))
            raise NavigationTimeout("networkidle timeout")

        def evaluate(self, _script):
            return self.ready

    settings = normalize_capture_options(
        {"waitUntil": "networkidle", "timeoutMs": 60_000}
    )
    ready_page = Page(True)

    navigation = _navigate_page(
        ready_page,
        "https://example.com",
        settings,
        timeout_error_type=NavigationTimeout,
    )

    assert ready_page.calls[0][2]["wait_until"] == "domcontentloaded"
    assert navigation["requestedWaitUntil"] == "networkidle"
    assert navigation["observedWaitState"] == (
        "dom-ready-after-networkidle-timeout"
    )
    assert navigation["fallbackUsed"] is True

    with pytest.raises(NavigationTimeout, match="networkidle timeout"):
        _navigate_page(
            Page(False),
            "https://example.com",
            settings,
            timeout_error_type=NavigationTimeout,
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


def test_capture_normalizes_and_validates_network_policy() -> None:
    assert normalize_capture_options({})["networkPolicy"] == "explicit"
    assert normalize_capture_options({"networkPolicy": "public"})["networkPolicy"] == "public"
    with pytest.raises(ValueError, match="networkPolicy"):
        normalize_capture_options({"networkPolicy": "private"})


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
        "networkPolicy": "explicit",
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


def test_public_source_network_policy_rejects_private_targets(monkeypatch) -> None:
    def resolve(host, *_args, **_kwargs):
        addresses = {
            "public.example": "93.184.216.34",
            "private.example": "192.168.1.20",
        }
        return [(2, 1, 6, "", (addresses[host], 443))]

    monkeypatch.setattr("sight.capture.socket.getaddrinfo", resolve)

    assert (
        validate_network_url("https://public.example/page", policy="public")
        == "public.example"
    )
    with pytest.raises(ValueError, match="private or local"):
        validate_network_url("https://private.example/secret", policy="public")
    with pytest.raises(ValueError, match="private or local"):
        validate_network_url("http://127.0.0.1:8123", policy="public")


def test_candidate_network_policy_allows_loopback_but_not_lan(monkeypatch) -> None:
    def resolve(host, *_args, **_kwargs):
        addresses = {
            "localhost": "127.0.0.1",
            "router.test": "10.0.0.1",
        }
        return [(2, 1, 6, "", (addresses[host], 80))]

    monkeypatch.setattr("sight.capture.socket.getaddrinfo", resolve)

    assert validate_network_url("http://localhost:8123", policy="candidate") == "localhost"
    with pytest.raises(ValueError, match="private network"):
        validate_network_url("http://router.test", policy="candidate")
    with pytest.raises(ValueError, match="credentials"):
        validate_network_url("https://user:pass@public.example", policy="candidate")


def test_browser_request_guard_aborts_blocked_redirects_and_subresources(monkeypatch) -> None:
    class Route:
        def __init__(self) -> None:
            self.action = None

        def continue_(self) -> None:
            self.action = "continue"

        def abort(self) -> None:
            self.action = "abort"

    class Request:
        def __init__(self, url: str) -> None:
            self.url = url

    def validate(url, *, policy):
        if "private.test" in url:
            raise ValueError("blocked private target")
        return "public.test"

    monkeypatch.setattr("sight.capture.validate_network_url", validate)
    blocked = []

    allowed_route = Route()
    _guard_browser_request(
        allowed_route,
        Request("https://cdn.public.test/app.css"),
        policy="public",
        blocked=blocked,
    )
    assert allowed_route.action == "continue"

    blocked_route = Route()
    _guard_browser_request(
        blocked_route,
        Request("http://private.test/redirect"),
        policy="public",
        blocked=blocked,
    )
    assert blocked_route.action == "abort"
    assert blocked == [
        {
            "url": "http://private.test/redirect",
            "reason": "blocked private target",
        }
    ]

    data_route = Route()
    _guard_browser_request(
        data_route,
        Request("data:image/png;base64,AA=="),
        policy="public",
        blocked=blocked,
    )
    assert data_route.action == "continue"


def test_capture_prepares_source_validation_and_browser_route_guard(monkeypatch) -> None:
    calls = []

    class Context:
        def route(self, pattern, handler) -> None:
            calls.append((pattern, handler))

    def validate(url, *, policy):
        calls.append((url, policy))
        return "public.test"

    monkeypatch.setattr("sight.capture.validate_network_url", validate)
    blocked = _prepare_network_capture(
        Context(),
        "https://public.test/design",
        {"networkPolicy": "public"},
    )

    assert calls[0] == ("https://public.test/design", "public")
    assert calls[1][0] == "**/*"
    assert callable(calls[1][1])
    assert blocked == []
