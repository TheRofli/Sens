"""W3C DTCG design tokens extracted from a deterministic dump."""
from __future__ import annotations

import colorsys
from collections import Counter
from typing import Any


def _saturation(hex_color: str) -> float:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4))
    return colorsys.rgb_to_hsv(r, g, b)[1]


def _contrast(hex_color: str, background: str) -> float:
    def lum(h: str) -> float:
        h = h.lstrip("#")
        rs, gs, bs = (int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))
        return 0.2126 * rs + 0.7152 * gs + 0.0722 * bs

    a, b = lum(hex_color), lum(background)
    return (max(a, b) + 0.05) / (min(a, b) + 0.05)


def build_design_tokens(dump: dict[str, Any]) -> dict[str, Any]:
    colors = dump.get("colors", [])
    canvas_entry = next(
        (entry for entry in colors if entry.get("role") == "canvas-background"),
        colors[0] if colors else None,
    )
    canvas = canvas_entry["hex"] if canvas_entry else "#FFFFFF"
    background_entry = canvas_entry
    if canvas_entry and float(canvas_entry.get("ratio") or 0.0) < 0.35:
        alternatives = [entry for entry in colors if entry is not canvas_entry]
        candidate = max(
            alternatives,
            key=lambda entry: float(entry.get("ratio") or 0.0),
            default=None,
        )
        if candidate and float(candidate.get("ratio") or 0.0) > max(
            0.20, float(canvas_entry.get("ratio") or 0.0) * 1.5
        ):
            background_entry = candidate
    background = background_entry["hex"] if background_entry else canvas
    rest = [c for c in colors[1:] if c["hex"] != background]
    ink = max(rest, key=lambda c: _contrast(c["hex"], background))["hex"] if rest else "#000000"
    saturated = [c for c in rest if _saturation(c["hex"]) > 0.35]
    accent = max(saturated, key=lambda c: _saturation(c["hex"]))["hex"] if saturated else ink
    muted = max(
        (c for c in rest if c["hex"] not in (ink, accent)),
        key=lambda c: c["ratio"],
        default=None,
    )

    sizes = sorted(
        {
            int(e["font"]["fontSize"])
            for e in dump.get("elements", [])
            if e.get("kind") == "text" and e.get("font")
        }
    )
    gaps = [g["px"] for g in dump.get("gaps", [])]
    base_gap = Counter(gaps).most_common(1)[0][0] if gaps else None
    radii = sorted({int(c["cornerRadius"]) for c in dump.get("controls", []) if c.get("cornerRadius")})

    tokens: dict[str, Any] = {
        "$schema": "https://tr.designtokens.org/format/",
        "color": {
            "canvas": {"$type": "color", "$value": canvas},
            "background": {"$type": "color", "$value": background},
            "ink": {"$type": "color", "$value": ink},
            "accent": {"$type": "color", "$value": accent},
        },
        "typography": {"scale": {"$type": "dimension", "$value": sizes}},
        "spacing": {"base": {"$type": "dimension", "$value": f"{base_gap}px"}},
        "borderRadius": {
            ("pill" if r > 16 else "md" if r > 6 else "sm"): {"$type": "dimension", "$value": f"{r}px"}
            for r in radii
        },
        "shadow": {
            f"shadow{i}": {"$type": "shadow", "$value": s}
            for i, s in enumerate(dump.get("shadows", []))
        },
    }
    if muted:
        tokens["color"]["muted"] = {"$type": "color", "$value": muted["hex"]}
    return tokens
