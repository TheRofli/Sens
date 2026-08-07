"""URL capture: screenshot, DOM/CSS styles, CSS animations, scroll motion events."""
from __future__ import annotations

import base64
import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np

STYLES_JS = """() => {
  const cs = getComputedStyle(document.body);
  const fonts = new Set();
  document.querySelectorAll("h1,h2,h3,p,a,button").forEach(el =>
    fonts.add(getComputedStyle(el).fontFamily));
  return { bodyBackground: cs.backgroundColor, fonts: [...fonts].slice(0, 12) };
}"""

DOM_JS = r"""() => [...document.querySelectorAll('body *')].slice(0, 300).map((el, index) => {
  const box = el.getBoundingClientRect();
  const style = getComputedStyle(el);
  return {
    index, tag: el.tagName.toLowerCase(), id: el.id || null,
    classes: [...el.classList].slice(0, 8), role: el.getAttribute('role'),
    ariaLabel: el.getAttribute('aria-label'),
    text: (el.innerText || '').trim().replace(/\s+/g, ' ').slice(0, 300),
    box: [Math.round(box.x), Math.round(box.y), Math.round(box.right), Math.round(box.bottom)],
    style: { display: style.display, position: style.position, color: style.color,
      background: style.backgroundColor, font: style.font, border: style.border,
      borderRadius: style.borderRadius, gap: style.gap, padding: style.padding }
  };
}).filter(item => item.box[2] > item.box[0] && item.box[3] > item.box[1])"""

ASSETS_JS = """() => ({
  images: [...document.images].slice(0, 100).map(img => ({src: img.currentSrc || img.src, alt: img.alt,
    size: [img.naturalWidth, img.naturalHeight]})),
  stylesheets: [...document.styleSheets].slice(0, 100).map(sheet => sheet.href).filter(Boolean),
  links: [...document.querySelectorAll('a[href]')].slice(0, 100).map(a => a.href)
})"""

VARS_JS = """() => {
  const style = getComputedStyle(document.documentElement); const out = {};
  for (const name of style) if (name.startsWith('--')) out[name] = style.getPropertyValue(name).trim();
  return out;
}"""

ANIM_JS = """() => {
  const out = { keyframes: [], animated: [], live: [] };
  for (const sheet of document.styleSheets) {
    let rules; try { rules = sheet.cssRules; } catch (e) { continue; }
    for (const r of rules) {
      if (r.type === CSSRule.KEYFRAMES_RULE)
        out.keyframes.push({ name: r.name, steps: r.cssRules.length });
      if (r.style && (r.style.animationName || r.style.transitionProperty))
        out.animated.push({
          selector: r.selectorText,
          animation: r.style.animationName,
          duration: r.style.animationDuration || r.style.transitionDuration,
          easing: r.style.animationTimingFunction || r.style.transitionTimingFunction
        });
    }
  }
  out.live = (document.getAnimations ? document.getAnimations() : []).slice(0, 50)
    .map(a => ({ name: a.animationName || "", state: a.playState,
      duration: a.effect && a.effect.getTiming ? a.effect.getTiming().duration : null }));
  return out;
}"""


def normalize_capture_options(options: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = options or {}
    viewport = raw.get("viewport") or {}
    width = int(viewport.get("width", 1440))
    height = int(viewport.get("height", 900))
    if not 320 <= width <= 3840:
        raise ValueError("viewport width must be between 320 and 3840")
    if not 240 <= height <= 2160:
        raise ValueError("viewport height must be between 240 and 2160")
    dpr = float(raw.get("dpr", 1.0))
    if not 0.5 <= dpr <= 3.0:
        raise ValueError("dpr must be between 0.5 and 3.0")
    theme = str(raw.get("theme", "light"))
    if theme not in {"light", "dark", "no-preference"}:
        raise ValueError("theme must be light, dark, or no-preference")
    wait_until = str(raw.get("waitUntil", "networkidle"))
    if wait_until not in {"commit", "domcontentloaded", "load", "networkidle"}:
        raise ValueError("waitUntil must be commit, domcontentloaded, load, or networkidle")
    timeout_ms = int(raw.get("timeoutMs", 30_000))
    if not 1_000 <= timeout_ms <= 60_000:
        raise ValueError("timeoutMs must be between 1000 and 60000")
    settle_ms = int(raw.get("settleMs", 250))
    if not 0 <= settle_ms <= 5_000:
        raise ValueError("settleMs must be between 0 and 5000")
    scroll_steps = int(raw.get("scrollSteps", 0))
    if not 0 <= scroll_steps <= 10:
        raise ValueError("scrollSteps must be between 0 and 10")
    return {
        "viewport": {"width": width, "height": height},
        "dpr": dpr,
        "theme": theme,
        "locale": str(raw.get("locale", "en-US")),
        "waitUntil": wait_until,
        "fullPage": bool(raw.get("fullPage", False)),
        "timeoutMs": timeout_ms,
        "settleMs": settle_ms,
        "scrollSteps": scroll_steps,
    }


def capture_request_id(url: str, options: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(url.encode("utf-8"))
    digest.update(b"\0")
    digest.update(json.dumps(options, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()[:20]


def _changed_boxes(prev: Any, cur: Any, min_area: int = 400) -> list[tuple[int, int, int, int]]:
    diff = cv2.absdiff(prev, cur)
    _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    th = cv2.dilate(th, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area]


def motion_events(frame_paths: list[str], step_seconds: float = 0.7) -> list[dict[str, Any]]:
    """Frame-diff motion events.

    A moving object shows up in a consecutive-frame diff as TWO changed
    regions: where it left and where it arrived. We greedily pair changed
    boxes within each transition (nearest centers) and report the pair's
    center-to-center delta as dx/dy — an approximation of the shift. Boxes
    that pair with nothing are emitted with dx/dy None.
    """
    events: list[dict[str, Any]] = []
    prev_gray = None
    for i, fp in enumerate(frame_paths):
        gray = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        if prev_gray is not None:
            boxes = _changed_boxes(prev_gray, gray)
            centers = [(x + w / 2, y + h / 2) for (x, y, w, h) in boxes]
            order = sorted(range(len(boxes)), key=lambda k: (boxes[k][0], boxes[k][1]))
            used: set[int] = set()
            for a in order:
                if a in used:
                    continue
                best, best_dist = None, None
                for b in order:
                    if b == a or b in used:
                        continue
                    dist = (centers[a][0] - centers[b][0]) ** 2 + (centers[a][1] - centers[b][1]) ** 2
                    if best_dist is None or dist < best_dist:
                        best, best_dist = b, dist
                if best is None:
                    x, y, w, h = boxes[a]
                    events.append(
                        {"frame": i, "box": [x, y, x + w, y + h], "dx": None, "dy": None,
                         "seconds": round(i * step_seconds, 1)}
                    )
                    used.add(a)
                    continue
                used.add(a)
                used.add(best)
                x1, y1, w1, h1 = boxes[a]
                x2, y2, w2, h2 = boxes[best]
                events.append(
                    {
                        "frame": i,
                        "box": [
                            min(x1, x2), min(y1, y2),
                            max(x1 + w1, x2 + w2), max(y1 + h1, y2 + h2),
                        ],
                        "dx": round(centers[best][0] - centers[a][0]),
                        "dy": round(centers[best][1] - centers[a][1]),
                        "seconds": round(i * step_seconds, 1),
                    }
                )
        prev_gray = gray
    return events


def _content_address(path: Path, prefix: str) -> tuple[Path, str]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    target = path.with_name(f"{prefix}-{digest[:20]}{path.suffix}")
    path.replace(target)
    return target, digest


def _launch_browser(chromium: Any) -> Any:
    """Prefer Windows' system Edge so the Sens runtime ships no browser blob."""
    try:
        return chromium.launch(channel="msedge", headless=True)
    except Exception:  # noqa: BLE001 - Playwright has backend-specific errors
        try:
            return chromium.launch(headless=True)
        except Exception as bundled_error:  # noqa: BLE001
            raise RuntimeError(
                "URL capture needs Microsoft Edge or a Playwright Chromium installation"
            ) from bundled_error


def capture_url(
    url: str,
    out_dir: str | Path,
    options: dict[str, Any] | None = None,
    *,
    no_store: bool = False,
) -> dict[str, Any]:
    from playwright.sync_api import sync_playwright

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("capture URL must use http or https")
    settings = normalize_capture_options(options)
    request_id = capture_request_id(url, settings)
    root = Path(out_dir)
    if no_store:
        work = Path(tempfile.mkdtemp(prefix="sens-capture-"))
    else:
        work = root / f"capture-{request_id}-{time.time_ns()}"
        work.mkdir(parents=True, exist_ok=False)

    browser = None
    context = None
    try:
        with sync_playwright() as playwright:
            browser = _launch_browser(playwright.chromium)
            context = browser.new_context(
                viewport=settings["viewport"],
                device_scale_factor=settings["dpr"],
                color_scheme=settings["theme"],
                locale=settings["locale"],
            )
            page = context.new_page()
            page.goto(
                url,
                wait_until=settings["waitUntil"],
                timeout=settings["timeoutMs"],
            )
            page.wait_for_function(
                "() => document.readyState !== 'loading' && document.body && document.body.childElementCount > 0",
                timeout=settings["timeoutMs"],
            )
            page.evaluate("() => document.fonts ? document.fonts.ready : Promise.resolve()")
            if settings["settleMs"]:
                page.wait_for_timeout(settings["settleMs"])

            shot = work / "screenshot.png"
            page.screenshot(path=str(shot), full_page=settings["fullPage"])
            shot, screenshot_hash = _content_address(shot, "screenshot")
            styles = page.evaluate(STYLES_JS)
            animations = page.evaluate(ANIM_JS)
            dom = page.evaluate(DOM_JS)
            assets = page.evaluate(ASSETS_JS)
            css_variables = page.evaluate(VARS_JS)
            try:
                accessibility = page.locator("body").aria_snapshot(timeout=5_000)
            except Exception:  # noqa: BLE001 - browser versions vary
                accessibility = None

            element_paths = []
            regions = page.locator("header, nav, main, section, article, aside, footer")
            for index in range(min(regions.count(), 10)):
                locator = regions.nth(index)
                try:
                    if not locator.is_visible():
                        continue
                    path = work / f"element-{index}.png"
                    locator.screenshot(path=str(path), timeout=5_000)
                    path, digest = _content_address(path, f"element-{index}")
                    element_paths.append(
                        {"path": str(path), "sha256": digest, "index": index}
                    )
                except Exception:  # noqa: BLE001 - one unstable element is non-fatal
                    continue

            frames = [shot]
            for index in range(settings["scrollSteps"]):
                page.evaluate(
                    "amount => window.scrollBy({top: amount, behavior: 'instant'})",
                    round(settings["viewport"]["height"] * 0.75),
                )
                page.wait_for_timeout(max(100, settings["settleMs"]))
                frame = work / f"frame-{index}.png"
                page.screenshot(path=str(frame))
                frame, _ = _content_address(frame, f"frame-{index}")
                frames.append(frame)

            context.close()
            context = None
            browser.close()
            browser = None

        frame_strings = [str(path) for path in frames]
        result = {
            "schemaVersion": "2.0.0",
            "captureId": request_id,
            "source": {
                "url": url,
                "source": "observed",
                "method": "playwright-instrumented-capture",
            },
            "settings": settings,
            "screenshot": None if no_store else str(shot),
            "screenshotSha256": screenshot_hash,
            "styles": styles,
            "cssVariables": css_variables,
            "dom": dom,
            "accessibility": accessibility,
            "assets": assets,
            "animations": animations,
            "elementScreenshots": [] if no_store else element_paths,
            "frames": [] if no_store else frame_strings,
            "motion": motion_events(frame_strings),
            "artifacts": []
            if no_store
            else [
                {
                    "id": f"sha256:{screenshot_hash}",
                    "kind": "web-screenshot",
                    "uri": str(shot),
                }
            ],
        }
        if no_store:
            result["screenshotDataUri"] = (
                "data:image/png;base64," + base64.b64encode(shot.read_bytes()).decode("ascii")
            )
        return result
    finally:
        if context is not None:
            context.close()
        if browser is not None:
            browser.close()
        if no_store:
            shutil.rmtree(work, ignore_errors=True)
