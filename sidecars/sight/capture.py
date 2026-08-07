"""URL capture: screenshot, DOM/CSS styles, CSS animations, scroll motion events."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

STYLES_JS = """() => {
  const cs = getComputedStyle(document.body);
  const fonts = new Set();
  document.querySelectorAll("h1,h2,h3,p,a,button").forEach(el =>
    fonts.add(getComputedStyle(el).fontFamily));
  return { bodyBackground: cs.backgroundColor, fonts: [...fonts].slice(0, 12) };
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


def capture_url(url: str, out_dir: str | Path, scroll_steps: int = 4) -> dict[str, Any]:
    from playwright.sync_api import sync_playwright

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        page.goto(url, wait_until="networkidle")
        shot = out / "shot.png"
        page.screenshot(path=str(shot))
        styles = page.evaluate(STYLES_JS)
        animations = page.evaluate(ANIM_JS)
        frames = [shot]
        for i in range(scroll_steps):
            page.mouse.wheel(0, 800)
            page.wait_for_timeout(700)
            fp = out / f"frame{i}.png"
            page.screenshot(path=str(fp))
            frames.append(fp)
        browser.close()
    frame_strs = [str(f) for f in frames]
    return {
        "screenshot": str(shot),
        "styles": styles,
        "animations": animations,
        "frames": frame_strs,
        "motion": motion_events(frame_strs),
    }
