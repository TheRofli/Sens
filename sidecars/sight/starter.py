"""Deterministic semantic starter project for screenshot-to-web work."""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


_COLOR = re.compile(r"^#[0-9a-fA-F]{3,8}$")
_FONT_ROOT = Path(__file__).resolve().parent / "assets" / "fonts"
_BUNDLED_FONT_SPECS = (
    {
        "family": "Sens Inter Tight",
        "source": "InterTight.ttf",
        "filename": "sens-inter-tight.ttf",
        "weight": "100 900",
    },
    {
        "family": "Sens Newsreader",
        "source": "Newsreader.ttf",
        "filename": "sens-newsreader.ttf",
        "weight": "200 800",
    },
)


def _box(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        box = [int(round(float(item))) for item in value]
    except (TypeError, ValueError):
        return None
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def _color(value: Any, fallback: str = "transparent") -> str:
    candidate = str(value or "")
    return candidate if _COLOR.fullmatch(candidate) else fallback


def _number(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _box_style(box: list[int]) -> str:
    x0, y0, x1, y1 = box
    return (
        f"left:{x0}px;top:{y0}px;width:{x1 - x0}px;"
        f"height:{y1 - y0}px"
    )


def _token_background(document: dict[str, Any]) -> str:
    colors = ((document.get("tokens") or {}).get("color") or {})
    token = colors.get("canvas") or colors.get("background") or {}
    return _color(token.get("$value"), "#FFFFFF")


def _text_value(entry: dict[str, Any]) -> str:
    preferred = entry.get("preferredValue")
    return str(preferred if preferred not in (None, "") else entry.get("value") or "")


def _font_family(entry: dict[str, Any]) -> str:
    typography = entry.get("typographyCandidate") or {}
    font = entry.get("fontFeatures") or {}
    kind = str(typography.get("class") or "").casefold()
    family = str(font.get("family") or "").casefold()
    width = str(typography.get("width") or "").casefold()
    contrast = str(typography.get("contrast") or "").casefold()
    size = _number(font.get("fontSize"))
    typography_confidence = _number(typography.get("confidence"))
    typography_reliable = (
        typography.get("confidence") is None or typography_confidence >= 0.55
    )
    family_confidence = _number(font.get("familyConfidence"))
    stroke_ratio = _number(font.get("strokeWidthRatio"))
    ink_coverage = _number(font.get("inkCoverage"))
    render_family = str(font.get("renderFamily") or "").casefold()
    render_confidence = _number(font.get("renderFamilyConfidence"))
    render_family_is_explicit = (
        bool(render_family) and font.get("renderFamilyConfidence") is None
    )
    if render_family_is_explicit or render_confidence >= 0.55:
        if render_family == "newsreader":
            return "'Sens Newsreader', Georgia, 'Times New Roman', serif"
        if render_family == "inter-tight":
            return "'Sens Inter Tight', Arial, 'Segoe UI', sans-serif"
    if "mono" in kind and typography_reliable:
        return "Consolas, 'Courier New', monospace"
    if "script" in kind and typography_reliable:
        return "'Sens Newsreader', Georgia, 'Times New Roman', serif"
    if "serif" in kind and "sans" not in kind and typography_reliable:
        if contrast in {"high", "very-high", "strong"}:
            return "'Sens Newsreader', 'Times New Roman', Times, serif"
        return "Georgia, 'Times New Roman', serif"
    if size >= 120 and stroke_ratio >= 0.10 and ink_coverage >= 0.40:
        return "'Arial Black', Arial, sans-serif"
    if "display" in kind and size >= 32 and typography_confidence >= 0.65:
        if kind == "display" and width in {"normal", "expanded"}:
            return "Arial, 'Segoe UI', sans-serif"
        return "'Sens Inter Tight', Arial, 'Segoe UI', sans-serif"
    if kind and typography_reliable:
        if size >= 24:
            return "'Sens Inter Tight', Arial, 'Segoe UI', sans-serif"
        return "Arial, 'Segoe UI', sans-serif"
    if family in {"bebas", "anton"} and size >= 32 and family_confidence >= 0.7:
        return "Impact, 'Arial Black', sans-serif"
    if (
        ("condens" in width or family == "oswald")
        and size >= 20
        and family_confidence >= 0.7
    ):
        return "'Arial Narrow', 'Roboto Condensed', Arial, sans-serif"
    if size >= 24:
        return "Arial, 'Segoe UI', sans-serif"
    return "Arial, 'Segoe UI', sans-serif"


def _font_weight(entry: dict[str, Any]) -> int:
    font = entry.get("fontFeatures") or {}
    typography = entry.get("typographyCandidate") or {}
    typography_class = str(typography.get("class") or "").casefold()
    typography_weight = str(typography.get("weight") or "").casefold()
    typography_confidence = _number(typography.get("confidence"))
    typography_reliable = (
        typography.get("confidence") is None or typography_confidence >= 0.55
    )
    size = _number(font.get("fontSize"))
    ink_coverage = _number(font.get("inkCoverage"))
    stroke_ratio = _number(font.get("strokeWidthRatio"))
    render_weight = int(_number(font.get("renderWeight")))
    if render_weight and (
        font.get("renderFamilyConfidence") is None
        or _number(font.get("renderFamilyConfidence")) >= 0.55
    ):
        return max(100, min(900, render_weight))
    if "script" in typography_class and typography_confidence >= 0.75:
        if str(font.get("weightCandidate") or "").casefold() in {
            "light",
            "thin",
        }:
            return 300
        if size >= 64 and str(typography.get("contrast") or "").casefold() in {
            "medium",
            "high",
            "very-high",
            "strong",
        }:
            return 300
        return 400
    if (
        "serif" in typography_class
        and "sans" not in typography_class
        and typography_confidence >= 0.75
    ):
        if typography_weight in {"black", "heavy", "extra-bold", "extrabold"}:
            return 900
        if typography_weight in {"bold", "semi-bold", "semibold"}:
            return 700
        if typography_weight == "medium":
            return 500
        if typography_weight in {"light", "thin"}:
            return 300
        return 400
    if size >= 120 and stroke_ratio >= 0.10 and ink_coverage >= 0.40:
        return 900
    if (
        typography_class == "display"
        and str(typography.get("width") or "").casefold() == "expanded"
        and size >= 120
        and stroke_ratio >= 0.10
        and ink_coverage >= 0.40
    ):
        return 900
    value = str(
        font.get("weightCandidate")
        or (typography.get("weight") if typography_reliable else "")
        or ""
    ).casefold()
    if value in {"bold", "semi-bold", "semibold"} and size >= 48:
        # A high fill ratio over photographic artwork can label medium display
        # text as bold.  The measured center-stroke ratio separates it from a
        # genuinely heavy face while preserving the giant black-face rule.
        if 0 < stroke_ratio < 0.075 and ink_coverage < 0.28:
            return 300
        if 0 < stroke_ratio < 0.075:
            return 500
        if stroke_ratio < 0.10 and ink_coverage < 0.50:
            return 600
    if value in {"black", "heavy", "extra-bold", "extrabold"}:
        return 900
    if value in {"bold", "semi-bold", "semibold"}:
        return 700
    if value in {"medium"}:
        return 500
    if value in {"light", "thin"}:
        return 300
    visible_text = _text_value(entry).strip()
    if not font.get("weightCandidate"):
        if size >= 24 and stroke_ratio >= 0.08:
            return 600
        if (
            9 <= size <= 18
            and ink_coverage >= 0.48
            and 0 < len(visible_text) <= 24
        ):
            return 600
    return 400


def _font_style(entry: dict[str, Any]) -> str:
    typography = entry.get("typographyCandidate") or {}
    kind = str(typography.get("class") or "").casefold()
    slant = str(typography.get("slant") or "").casefold()
    confidence = typography.get("confidence")
    reliable = confidence is None or _number(confidence) >= 0.55
    if not reliable:
        return "normal"
    if slant in {"italic", "oblique"}:
        return slant
    return "italic" if "script" in kind else "normal"


def _text_markup(entry: dict[str, Any], value: str) -> str:
    indexed = entry.get("indexedLabel") or {}
    if indexed.get("superscript") is True:
        prefix = str(indexed.get("prefix") or "")
        index = str(indexed.get("index") or "")
        label = str(indexed.get("label") or "")
        if prefix and index and label and value == f"{prefix}{index} {label}":
            return (
                '<span class="sens-text sens-indexed-label" '
                f'data-sens-indexed-label="true" aria-label="{html.escape(value)}">'
                f'{html.escape(prefix)}<sup class="sens-indexed-label-index">{html.escape(index)}</sup> '
                f'{html.escape(label)}</span>'
            )
    runs = entry.get("inlineRuns") or []
    if (
        len(runs) < 2
        or any(not isinstance(run, dict) for run in runs)
        or "".join(str(run.get("text") or "") for run in runs) != value
    ):
        return f'<span class="sens-text">{html.escape(value)}</span>'
    output: list[str] = []
    for run in runs:
        typography = {
            **(entry.get("typographyCandidate") or {}),
            **(run.get("typographyCandidate") or {}),
        }
        font = dict(entry.get("fontFeatures") or {})
        # Keep an extreme measured whole-line stroke class when present. It is
        # more reliable than the small VLM's categorical weight guess; mixed
        # lines in the ambiguous middle still fall through to per-run style.
        run_entry = {
            **entry,
            "fontFeatures": font,
            "typographyCandidate": typography,
        }
        style = (
            f"font-family:{_font_family(run_entry)};"
            f"font-weight:{_font_weight(run_entry)};"
            f"font-style:{_font_style(run_entry)}"
        )
        output.append(
            f'<span class="sens-inline-run" data-sens-run-index="{int(run.get("runIndex") or 0)}" style="{style}">{html.escape(str(run.get("text") or ""))}</span>'
        )
    return (
        '<span class="sens-text" data-sens-inline-runs="true">'
        + "".join(output)
        + "</span>"
    )


def _measured_word_markup(
    entry: dict[str, Any], value: str
) -> str | None:
    """Position verified words independently when their source boxes exist."""
    if (entry.get("indexedLabel") or {}).get("superscript") is True:
        return None
    source_box = _box(entry.get("boxSource"))
    font = entry.get("fontFeatures") or {}
    word_boxes = [
        item
        for item in font.get("wordBoxesSource") or []
        if isinstance(item, dict) and _box(item.get("box")) is not None
    ]
    tokens = list(re.finditer(r"\S+", value))
    if source_box is None or len(tokens) < 2 or len(tokens) != len(word_boxes):
        return None
    if any(
        "".join(character for character in token.group(0).casefold() if character.isalnum())
        != "".join(
            character
            for character in str(item.get("text") or "").casefold()
            if character.isalnum()
        )
        for token, item in zip(tokens, word_boxes, strict=True)
    ):
        return None
    unit_widths = []
    for token, item in zip(tokens, word_boxes, strict=True):
        item_box = _box(item.get("box"))
        if item_box is None:
            return None
        character_count = max(
            1, sum(character.isalnum() for character in token.group(0))
        )
        unit_widths.append((item_box[2] - item_box[0]) / character_count)
    if (
        not unit_widths
        or min(unit_widths) <= 0
        or max(unit_widths) / min(unit_widths) > 3.5
    ):
        return None
    ordered_boxes = [_box(item.get("box")) for item in word_boxes]
    cap_height = max(
        1.0,
        _number(font.get("capHeight"), source_box[3] - source_box[1]),
    )
    minimum_visible_space = max(1.0, cap_height * 0.05)
    if any(
        left is None
        or right is None
        or right[0] - left[2] < minimum_visible_space
        for left, right in zip(ordered_boxes, ordered_boxes[1:])
    ):
        # Touching OCR polygons cannot encode a visible word boundary. Keep
        # natural DOM whitespace instead of stretching words into one token.
        return None
    inline_runs = entry.get("inlineRuns") or []
    nodes: list[str] = []
    for index, (token, item) in enumerate(zip(tokens, word_boxes, strict=True)):
        word_box = _box(item.get("box"))
        if word_box is None:
            return None
        relative = [
            word_box[0] - source_box[0],
            word_box[1] - source_box[1],
            word_box[2] - source_box[0],
            word_box[3] - source_box[1],
        ]
        if (
            relative[0] < -2
            or relative[1] < -2
            or relative[2] > source_box[2] - source_box[0] + 2
            or relative[3] > source_box[3] - source_box[1] + 2
        ):
            return None
        run_typography = (
            {"slant": item.get("slant")}
            if item.get("slant") in {"normal", "italic", "oblique"}
            else {}
        )
        if index < len(inline_runs) and isinstance(inline_runs[index], dict):
            run_typography = {
                **run_typography,
                **(inline_runs[index].get("typographyCandidate") or {}),
            }
        word_font = dict(font)
        for key in (
            "renderFamily",
            "renderFamilyCandidate",
            "renderFamilyConfidence",
            "renderWeight",
            "renderFamilyMethod",
            "renderFamilyScores",
        ):
            if key in item:
                word_font[key] = item[key]
        word_entry = {
            **entry,
            "fontFeatures": word_font,
            "typographyCandidate": {
                **(entry.get("typographyCandidate") or {}),
                **run_typography,
            },
        }
        style = (
            f"{_box_style(relative)};font-size:inherit;line-height:normal;"
            f"font-family:{_font_family(word_entry)};"
            f"font-weight:{_font_weight(word_entry)};"
            f"font-style:{_font_style(word_entry)}"
        )
        nodes.append(
            f'<span class="sens-word-slot sens-fit-slot" data-sens-cap-height="{max(1, relative[3] - relative[1]):g}" data-sens-word-index="{index}" style="{style}"><span class="sens-text">{html.escape(token.group(0))}</span></span>'
        )
    return " ".join(nodes)


def _text_style(entry: dict[str, Any], box: list[int]) -> str:
    font = entry.get("fontFeatures") or {}
    size = max(1.0, _number(font.get("fontSize"), (box[3] - box[1]) * 0.8))
    color = _color(entry.get("color") or font.get("color"), "#111111")
    return (
        f"left:{box[0]}px;top:{box[1]}px;width:{box[2] - box[0]}px;"
        f"height:{box[3] - box[1]}px;font-size:{size:g}px;line-height:normal;"
        f"font-family:{_font_family(entry)};font-weight:{_font_weight(entry)};"
        f"font-style:{_font_style(entry)};color:{color}"
    )


def _text_metrics_attributes(entry: dict[str, Any]) -> str:
    font = entry.get("fontFeatures") or {}
    cap_height = _number(font.get("capHeight"))
    if cap_height <= 0:
        return ""
    return f' data-sens-cap-height="{cap_height:g}"'


_ICON_SVG: dict[str, str] = {
    "home": '<path d="M3 11.5 12 4l9 7.5"/><path d="M5.5 10v10h13V10"/>',
    "wallet": '<rect x="3" y="5" width="18" height="14" rx="3"/><path d="M15 10h6v5h-6a2.5 2.5 0 0 1 0-5Z"/>',
    "message": '<path d="M4 5h16v11H9l-5 4Z"/>',
    "users": '<circle cx="9" cy="9" r="3"/><circle cx="17" cy="10" r="2.5"/><path d="M3.5 20c.5-4 2.5-6 5.5-6s5 2 5.5 6"/><path d="M14 15c3.5-.5 5.5 1.2 6 4"/>',
    "network": '<circle cx="12" cy="5" r="2.5"/><circle cx="5" cy="18" r="2.5"/><circle cx="19" cy="18" r="2.5"/><path d="m10.7 7.2-4.4 8.6M13.3 7.2l4.4 8.6M7.5 18h9"/>',
    "file": '<path d="M6 3h8l4 4v14H6Z"/><path d="M14 3v5h5"/>',
    "file-plus": '<path d="M6 3h8l4 4v14H6Z"/><path d="M14 3v5h5M9 14h6M12 11v6"/>',
    "file-check": '<path d="M6 3h8l4 4v14H6Z"/><path d="M14 3v5h5M9 14l2 2 4-5"/>',
    "chart": '<path d="M4 20V5M4 20h17M8 16l4-5 3 2 5-7"/>',
    "dollar": '<circle cx="12" cy="12" r="9"/><path d="M15.5 8.5c-1-1-2-1.5-3.5-1.5-2 0-3.5 1-3.5 2.5 0 4 7 1.5 7 5.5 0 1.5-1.5 2.5-3.5 2.5-1.5 0-3-.5-4-1.5M12 5v14"/>',
    "shield": '<path d="M12 3 20 6v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6Z"/><path d="m8.5 12 2.2 2.2 4.8-5"/>',
    "trophy": '<path d="M8 4h8v5c0 4-2 6-4 6s-4-2-4-6ZM12 15v4M8 21h8"/><path d="M8 6H4c0 4 1.5 6 5 6M16 6h4c0 4-1.5 6-5 6"/>',
    "send": '<path d="m3 11 18-8-7 18-3-7Z"/><path d="m11 14 4-4"/>',
    "globe": '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c3 3 4 6 4 9s-1 6-4 9c-3-3-4-6-4-9s1-6 4-9Z"/>',
    "calendar": '<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M7 3v4M17 3v4M3 10h18"/>',
    "palette": '<path d="M12 3a9 9 0 1 0 0 18h1.5a2 2 0 0 0 0-4H12a2 2 0 0 1 0-4h4a5 5 0 0 0 5-5c0-3-4-5-9-5Z"/><circle cx="7" cy="10" r="1"/><circle cx="9" cy="6.5" r="1"/><circle cx="14" cy="6" r="1"/>',
    "user-check": '<circle cx="9" cy="8" r="3"/><path d="M3.5 20c.5-4 2.5-6 5.5-6 2 0 3.5.8 4.5 2.2M15 18l2 2 4-5"/>',
    "portal": '<rect x="4" y="3" width="12" height="18" rx="1"/><path d="M10 12h10M17 9l3 3-3 3"/>',
    "cross": '<path d="M12 4v16M4 12h16"/>',
    "arrow-down": '<path d="M12 4v15M6 13l6 6 6-6"/>',
    "arrow-up": '<path d="M12 20V5M6 11l6-6 6 6"/>',
    "arrow-right": '<path d="M4 12h15M13 6l6 6-6 6"/>',
    "arrow-left": '<path d="M20 12H5M11 6l-6 6 6 6"/>',
    "circle": '<circle cx="12" cy="12" r="8"/>',
    "outline": '<rect x="4" y="4" width="16" height="16" rx="4"/>',
    "brand-mark": '<path d="M5 9.5A4.5 4.5 0 1 0 5 14.5"/><path d="M13.5 6.5 18 12l-4.5 5.5"/>',
    "starburst": '<circle cx="12" cy="12" r="11" fill="#F1F1F1" stroke="none"/><path d="M12 4v16M5.1 8l13.8 8M5.1 16l13.8-8"/>',
}


def _icon_svg_markup(name: str, box: list[int], color: str) -> str:
    content = _ICON_SVG.get(name)
    if not content:
        return ""
    rendered_minimum = max(1, min(box[2] - box[0], box[3] - box[1]))
    stroke_width = max(1.8, 24.0 / rendered_minimum)
    return (
        f'<svg class="sens-icon sens-icon-{name}" aria-hidden="true" focusable="false" '
        f'viewBox="0 0 24 24" style="{_box_style(box)};color:{color}" '
        f'fill="none" stroke="currentColor" stroke-width="{stroke_width:g}" '
        f'stroke-linecap="round" stroke-linejoin="round">{content}</svg>'
    )


def _surface_markup(entries: list[dict[str, Any]], class_name: str) -> list[str]:
    output: list[str] = []

    def surface_area(entry: dict[str, Any]) -> int:
        box = _box(entry.get("boxSource"))
        if box is None:
            return 0
        return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

    for entry in sorted(entries, key=surface_area, reverse=True):
        if entry.get("preservedInBackgroundArtwork"):
            continue
        box = _box(entry.get("boxSource"))
        if box is None:
            continue
        background = _color(entry.get("background"))
        border = _color(entry.get("borderColor"))
        border_width = max(0.0, _number(entry.get("borderWidth")))
        radius = max(0.0, _number(entry.get("cornerRadius")))
        style = (
            f"{_box_style(box)};background:{background};"
            f"border:{border_width:g}px solid {border};border-radius:{radius:g}px"
        )
        output.append(f'<div class="{class_name}" aria-hidden="true" style="{style}"></div>')
    return output


def _symbol_visual_markup(
    entry: dict[str, Any], box: list[int], content: str
) -> tuple[str, bool]:
    geometry = entry.get("glyphGeometry") or {}
    dot = geometry.get("dot") if isinstance(geometry, dict) else None
    diamond = geometry.get("diamond") if isinstance(geometry, dict) else None
    if not isinstance(dot, dict) or not isinstance(diamond, dict):
        return "", False
    cell_width = _number(entry.get("cellWidth"))
    row_pitch = _number(entry.get("rowPitch"))
    first_center_x = _number(entry.get("firstCellCenterX"))
    first_baseline_y = _number(entry.get("firstBaselineY"))
    if min(cell_width, row_pitch) <= 0:
        return "", False
    color = _color(entry.get("color"), "#FFFFFF")
    visuals: list[str] = []
    for row, line in enumerate(content.splitlines()):
        baseline = first_baseline_y + row * row_pitch
        for column, character in enumerate(line):
            if character not in {".", "◆"}:
                continue
            kind = "dot" if character == "." else "diamond"
            metrics = dot if kind == "dot" else diamond
            glyph_width = max(1.0, _number(metrics.get("width"), 1.0))
            glyph_height = max(1.0, _number(metrics.get("height"), 1.0))
            center_x = first_center_x + column * cell_width
            left = (
                center_x
                + _number(metrics.get("centerOffsetX"))
                - glyph_width / 2.0
                - box[0]
            )
            top = (
                baseline
                + _number(metrics.get("centerOffsetY"))
                - glyph_height / 2.0
                - box[1]
            )
            visuals.append(
                f'<span class="sens-symbol-visual sens-symbol-{kind}" aria-hidden="true" '
                f'style="left:{left:g}px;top:{top:g}px;width:{glyph_width:g}px;'
                f'height:{glyph_height:g}px;color:{color}"></span>'
            )
    return "".join(visuals), bool(visuals)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _asset_records(spec: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for entry in spec.get("allowedRasterRegions") or []:
        source = Path(str(entry.get("assetPath") or ""))
        box = _box(entry.get("boxSource"))
        if box is None or not source.is_file():
            continue
        content = source.read_bytes()
        records.append(
            {
                "elementId": entry.get("elementId"),
                "artifactId": entry.get("artifactId"),
                "kind": entry.get("kind") or "illustration-or-photo",
                "box": box,
                "content": content,
                "sha256": hashlib.sha256(content).hexdigest(),
                "suffix": source.suffix.lower() if source.suffix else ".png",
            }
        )
    return records


def _font_asset_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for spec in _BUNDLED_FONT_SPECS:
        source = _FONT_ROOT / str(spec["source"])
        if not source.is_file():
            continue
        content = source.read_bytes()
        records.append(
            {
                **spec,
                "content": content,
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return records


def _projection(
    document: dict[str, Any],
    assets: list[dict[str, Any]],
    font_assets: list[dict[str, Any]],
) -> dict[str, Any]:
    spec = document.get("reconstruction") or {}
    return {
        "schemaVersion": "sens-semantic-starter-2",
        "generatorVersion": 22,
        "sourceId": (document.get("source") or {}).get("id"),
        "canvas": spec.get("canvas"),
        "background": _token_background(document),
        "text": spec.get("text") or [],
        "controls": spec.get("visualControlCandidates") or [],
        "surfaces": spec.get("surfaces") or [],
        "decorativeShapes": spec.get("decorativeShapes") or [],
        "structuralLines": spec.get("structuralLines") or [],
        "vectorPaths": spec.get("vectorPaths") or [],
        "icons": spec.get("icons") or [],
        "badges": spec.get("badges") or [],
        "symbolArt": spec.get("symbolArt") or [],
        "assets": [
            {
                "elementId": item["elementId"],
                "artifactId": item.get("artifactId"),
                "kind": item.get("kind"),
                "box": item["box"],
                "sha256": item["sha256"],
                "suffix": item["suffix"],
            }
            for item in assets
        ],
        "bundledFonts": [
            {
                "family": item["family"],
                "filename": item["filename"],
                "sha256": item["sha256"],
            }
            for item in font_assets
        ],
    }


def _render_project(
    document: dict[str, Any],
    assets: list[dict[str, Any]],
    font_assets: list[dict[str, Any]],
    digest: str,
) -> tuple[str, str, str, list[tuple[str, bytes]]]:
    spec = document.get("reconstruction") or {}
    canvas = spec.get("canvas") or {}
    width = max(1, int(canvas.get("width") or 1))
    height = max(1, int(canvas.get("height") or 1))
    background = _token_background(document)
    text_entries = {
        entry.get("elementId"): entry
        for entry in spec.get("text") or []
        if entry.get("elementId") is not None
    }
    consumed_text_ids: set[Any] = set()
    badge_label_ids = {
        badge.get("labelElementId")
        for badge in spec.get("badges") or []
        if badge.get("labelElementId") is not None
    }
    controls: list[str] = []
    for control in spec.get("visualControlCandidates") or []:
        box = _box(control.get("boxSource"))
        if box is None:
            continue
        label_ids = [
            item
            for item in control.get("labelElementIds") or []
            if item not in badge_label_ids
        ]
        labels = [text_entries[item] for item in label_ids if item in text_entries]
        consumed_text_ids.update(item for item in label_ids if item in text_entries)
        label = " ".join(_text_value(item).strip() for item in labels if _text_value(item).strip())
        aria_label = label or str(control.get("ariaLabel") or "")
        background_color = _color(control.get("background"))
        border_color = _color(control.get("borderColor"), "transparent")
        border_width = max(0.0, _number(control.get("borderWidth")))
        radius = max(0.0, _number(control.get("cornerRadius")))
        style = (
            f"{_box_style(box)};background:{background_color};"
            f"border:{border_width:g}px solid {border_color};border-radius:{radius:g}px"
        )
        if control.get("zIndex") is not None:
            style += f";z-index:{int(_number(control.get('zIndex')))}"
        label_nodes: list[str] = []
        for label_entry in labels:
            label_box = _box(label_entry.get("boxSource"))
            if label_box is None:
                continue
            label_font = label_entry.get("fontFeatures") or {}
            measured_font_size = max(
                1.0, _number(label_font.get("fontSize"), 14)
            )
            label_font_cap = max(
                8.0,
                _number(
                    control.get("labelFontSizeMax"),
                    (box[3] - box[1]) * 0.62,
                ),
            )
            label_font_size = min(measured_font_size, label_font_cap)
            measured_cap_height = _number(
                label_font.get("capHeight"),
                label_font_size * 0.76,
            )
            render_cap_height = min(
                measured_cap_height,
                label_font_size * 0.76,
            )
            render_label_entry = {
                **label_entry,
                "fontFeatures": {
                    **label_font,
                    "fontSize": label_font_size,
                    "capHeight": round(render_cap_height, 1),
                    **(
                        {"weightCandidate": "regular"}
                        if not label_font.get("weightCandidate")
                        and label_font_size <= 24
                        else {}
                    ),
                },
            }
            relative_box = [
                label_box[0] - box[0],
                label_box[1] - box[1],
                label_box[2] - box[0],
                label_box[3] - box[1],
            ]
            measured_words = _measured_word_markup(
                render_label_entry, _text_value(label_entry)
            )
            label_classes = (
                "sens-text-slot sens-control-label sens-measured-words"
                if measured_words is not None
                else "sens-text-slot sens-control-label sens-fit-slot"
            )
            label_nodes.append(
                f'<span class="{label_classes}" data-sens-text-box="true"{_text_metrics_attributes(render_label_entry)} data-source-element="{html.escape(str(label_entry.get("elementId") or ""))}" style="{_text_style(render_label_entry, relative_box)}">{measured_words if measured_words is not None else _text_markup(render_label_entry, _text_value(label_entry))}</span>'
            )
        semantic_link = (
            control.get("interaction") == "semantic-link"
            or control.get("semanticRole") in {"nav", "link"}
        )
        if semantic_link:
            controls.append(
                f'<a class="sens-control" href="#" aria-label="{html.escape(aria_label)}" data-source-element="{html.escape(str(control.get("elementId") or ""))}" style="{style}">{"".join(label_nodes)}</a>'
            )
        else:
            controls.append(
                f'<button class="sens-control" type="button" aria-label="{html.escape(aria_label)}" data-source-element="{html.escape(str(control.get("elementId") or ""))}" style="{style}">{"".join(label_nodes)}</button>'
            )

    badges: list[str] = []
    for badge in spec.get("badges") or []:
        box = _box(badge.get("boxSource"))
        label_id = badge.get("labelElementId")
        label_entry = text_entries.get(label_id) or {}
        value = _text_value(label_entry) or str(badge.get("value") or "")
        if box is None or not value:
            continue
        consumed_text_ids.add(label_id)
        font = label_entry.get("fontFeatures") or {}
        font_size = max(
            8.0,
            min(
                _number(font.get("fontSize"), (box[3] - box[1]) * 0.55),
                (box[3] - box[1]) * 0.62,
            ),
        )
        radius = max(
            0.0,
            _number(badge.get("cornerRadius"), (box[3] - box[1]) / 3),
        )
        style = (
            f"{_box_style(box)};background:{_color(badge.get('background'), '#E8EEFF')};"
            f"border-radius:{radius:g}px;color:{_color(badge.get('foreground'), '#2563EB')};"
            f"font-family:{_font_family(label_entry)};font-weight:{_font_weight(label_entry)};"
            f"font-size:{font_size:g}px"
        )
        badges.append(
            f'<span class="sens-badge" data-sens-badge="true" data-source-element="{html.escape(str(label_id or ""))}" style="{style}"><span class="sens-badge-text">{html.escape(value)}</span></span>'
        )

    text_nodes: list[str] = []
    for entry in spec.get("text") or []:
        if entry.get("elementId") in consumed_text_ids:
            continue
        value = _text_value(entry)
        box = _box(entry.get("boxSource"))
        if box is None or not value:
            continue
        measured_words = _measured_word_markup(entry, value)
        classes = (
            "sens-text-slot sens-measured-words"
            if measured_words is not None
            else "sens-text-slot sens-fit-slot"
        )
        text_nodes.append(
            f'<span class="{classes}" data-sens-text-box="true"{_text_metrics_attributes(entry)} data-source-element="{html.escape(str(entry.get("elementId") or ""))}" style="{_text_style(entry, box)}">{measured_words if measured_words is not None else _text_markup(entry, value)}</span>'
        )

    lines: list[str] = []
    for line in spec.get("structuralLines") or []:
        box = _box(line.get("boxSource"))
        if box is None:
            continue
        color = _color(line.get("color"), "#111111")
        if line.get("lineStyle") == "dashed":
            dash_length = max(1.0, _number(line.get("dashLength"), 4.0))
            dash_gap = max(1.0, _number(line.get("dashGap"), 4.0))
            line_background = (
                "repeating-linear-gradient(to right,"
                f"{color} 0 {dash_length:g}px,transparent {dash_length:g}px "
                f"{dash_length + dash_gap:g}px)"
            )
        else:
            line_background = color
        if line.get("preservedInBackgroundArtwork"):
            line_background = "transparent"
        lines.append(
            f'<div class="sens-line" data-sens-line="true" data-sens-line-color="{html.escape(color)}" '
            f'data-sens-line-style="{html.escape(str(line.get("lineStyle") or "solid"))}" '
            f'aria-hidden="true" style="{_box_style(box)};background:{line_background}"></div>'
        )

    symbols: list[str] = []
    for entry in spec.get("symbolArt") or []:
        box = _box(entry.get("boxSource"))
        content = str(entry.get("text") or "")
        if box is None or not content:
            continue
        cell_width = max(1.0, _number(entry.get("cellWidth"), 8.0))
        row_pitch = max(1.0, _number(entry.get("rowPitch"), cell_width * 1.5))
        font_size = cell_width / 0.6
        visual_markup, measured_glyphs = _symbol_visual_markup(entry, box, content)
        text_color = (
            "transparent"
            if measured_glyphs
            else _color(entry.get("color"), "#FFFFFF")
        )
        symbols.append(
            f'<pre class="sens-symbol-art" style="{_box_style(box)};font-size:{font_size:g}px;'
            f'line-height:{row_pitch:g}px;color:{text_color}">{html.escape(content)}'
            f'{visual_markup}</pre>'
        )

    raster_nodes: list[str] = []
    asset_files: list[tuple[str, bytes]] = [
        (str(font["filename"]), font["content"]) for font in font_assets
    ]
    for index, asset in enumerate(assets, start=1):
        name = f"asset-{index}-{asset['sha256'][:8]}{asset['suffix']}"
        asset_files.append((name, asset["content"]))
        background_artwork = asset.get("kind") == "alpha-masked-background-artwork"
        raster_class = "sens-background-artwork" if background_artwork else "sens-raster"
        raster_role = (
            ' data-sens-raster-role="alpha-masked-background-artwork"'
            if background_artwork
            else ""
        )
        raster_nodes.append(
            f'<img class="{raster_class}" src="assets/{html.escape(name)}" alt="" draggable="false"{raster_role} data-sens-artifact-id="{html.escape(str(asset.get("artifactId") or ""))}" data-source-element="{html.escape(str(asset.get("elementId") or ""))}" style="{_box_style(asset["box"])}">'
        )

    vector_nodes: list[str] = []
    for entry in spec.get("vectorPaths") or []:
        if entry.get("preservedInBackgroundArtwork"):
            continue
        points = []
        for point in entry.get("pointsSource") or []:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                continue
            points.append(
                f"{int(round(_number(point[0])))},{int(round(_number(point[1])))}"
            )
        if len(points) < 3:
            continue
        vector_nodes.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{_color(entry.get("strokeColor"), "#111111")}" stroke-width="{max(1.0, _number(entry.get("strokeWidth"), 1.0)):g}" stroke-linecap="round" stroke-linejoin="round" vector-effect="non-scaling-stroke"></polyline>'
        )
    vector_layer = (
        f'<svg class="sens-vector-layer" aria-hidden="true" viewBox="0 0 {width} {height}" width="{width}" height="{height}">{"".join(vector_nodes)}</svg>'
        if vector_nodes
        else ""
    )

    icons: list[str] = []
    for icon in spec.get("icons") or []:
        if icon.get("preservedInBackgroundArtwork"):
            continue
        box = _box(icon.get("boxSource"))
        if box is None:
            continue
        name = re.sub(r"[^a-z0-9_-]", "-", str(icon.get("name") or "unknown").casefold())
        markup = _icon_svg_markup(
            name.replace("_", "-"),
            box,
            _color(icon.get("color"), "#111111"),
        )
        if markup:
            icons.append(markup)

    surfaces = _surface_markup(spec.get("surfaces") or [], "sens-surface")
    shapes = _surface_markup(spec.get("decorativeShapes") or [], "sens-shape")
    body = "\n    ".join(
        [
            *surfaces,
            *shapes,
            *lines,
            vector_layer,
            *raster_nodes,
            *symbols,
            *icons,
            *badges,
            *text_nodes,
            *controls,
        ]
    )
    index = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Sens semantic reconstruction starter</title>
  <link rel="stylesheet" href="styles.css">
  <script src="script.js" defer></script>
</head>
<body>
  <main class="sens-canvas" data-sens-starter="{digest[:16]}">
    {body}
  </main>
</body>
</html>
"""
    font_faces = "\n".join(
        f"@font-face{{font-family:'{font['family']}';src:url('assets/{font['filename']}') format('truetype');font-style:normal;font-weight:{font['weight']};font-display:block}}"
        for font in font_assets
    )
    css = f"""{font_faces}
*{{box-sizing:border-box}}
html,body{{margin:0;width:100%;min-height:100%;background:{background}}}
body{{overflow:auto}}
.sens-canvas{{position:relative;width:{width}px;height:{height}px;overflow:hidden;background:{background};isolation:isolate}}
.sens-surface,.sens-shape,.sens-line,.sens-raster,.sens-background-artwork,.sens-symbol-art,.sens-icon,.sens-badge,.sens-text-slot,.sens-control{{position:absolute;margin:0}}
.sens-background-artwork{{z-index:0;display:block;object-fit:fill;user-select:none;pointer-events:none}}
.sens-surface{{z-index:1}}
.sens-shape{{z-index:2}}
.sens-line{{z-index:3;pointer-events:none}}
.sens-vector-layer{{position:absolute;inset:0;z-index:8;overflow:visible;pointer-events:none}}
.sens-raster{{z-index:10;display:block;object-fit:fill;user-select:none}}
.sens-symbol-art{{z-index:20;white-space:pre;overflow:hidden;font-family:Consolas,'Courier New',monospace;font-weight:400;letter-spacing:0;user-select:text}}
.sens-indexed-label-index{{font-size:.56em;line-height:0;vertical-align:.62em;margin-right:.08em}}
.sens-symbol-visual{{position:absolute;display:block;background:currentColor;pointer-events:none}}
.sens-symbol-diamond{{clip-path:polygon(50% 0,100% 50%,50% 100%,0 50%)}}
.sens-icon{{z-index:25;display:block;pointer-events:none}}
.sens-badge{{z-index:29;display:flex;align-items:center;justify-content:center;line-height:1;white-space:nowrap;user-select:text}}
.sens-badge-text{{display:block;user-select:text}}
.sens-text-slot{{z-index:30;display:block;overflow:visible;white-space:nowrap;letter-spacing:0;text-rendering:geometricPrecision;user-select:text}}
.sens-word-slot{{position:absolute;display:block;overflow:visible;white-space:nowrap;letter-spacing:0;text-rendering:geometricPrecision;user-select:text}}
.sens-text{{display:block;position:absolute;left:0;top:0;line-height:1;white-space:pre;transform-origin:0 0;user-select:text}}
.sens-inline-run{{display:inline;line-height:inherit;white-space:pre;user-select:text}}
.sens-control{{z-index:35;padding:0;display:flex;align-items:center;justify-content:center;appearance:none;text-align:center;text-decoration:none;color:inherit;white-space:nowrap;cursor:pointer}}
.sens-control:hover{{filter:brightness(.98)}}
.sens-control:focus-visible{{outline:2px solid currentColor;outline-offset:2px}}
"""
    script = """const fitSensText = () => {
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  if (!context) return;
  for (const slot of document.querySelectorAll('.sens-fit-slot')) {
    const text = slot.querySelector('.sens-text');
    if (!text || !text.textContent) continue;
    text.style.transform = 'none';
    const target = slot.getBoundingClientRect();
    const style = getComputedStyle(slot);
    const fontSize = Number.parseFloat(style.fontSize) || target.height;
    const runs = [...text.querySelectorAll('.sens-inline-run')];
    let left, right, ascent, descent, fontAscent, fontDescent;
    if (runs.length) {
      let cursor = 0;
      let minInkX = 0;
      let maxInkX = 0;
      ascent = 0;
      descent = 0;
      fontAscent = 0;
      fontDescent = 0;
      for (const run of runs) {
        const runStyle = getComputedStyle(run);
        context.font = `${runStyle.fontStyle} ${runStyle.fontVariant} ${runStyle.fontWeight} ${runStyle.fontSize} ${runStyle.fontFamily}`;
        const runMetrics = context.measureText(run.textContent);
        const runLeft = Number.isFinite(runMetrics.actualBoundingBoxLeft) ? runMetrics.actualBoundingBoxLeft : 0;
        const runRight = Number.isFinite(runMetrics.actualBoundingBoxRight) ? runMetrics.actualBoundingBoxRight : runMetrics.width;
        minInkX = Math.min(minInkX, cursor - runLeft);
        maxInkX = Math.max(maxInkX, cursor + runRight);
        cursor += runMetrics.width;
        ascent = Math.max(ascent, Number.isFinite(runMetrics.actualBoundingBoxAscent) ? runMetrics.actualBoundingBoxAscent : fontSize * .8);
        descent = Math.max(descent, Number.isFinite(runMetrics.actualBoundingBoxDescent) ? runMetrics.actualBoundingBoxDescent : fontSize * .2);
        fontAscent = Math.max(fontAscent, Number.isFinite(runMetrics.fontBoundingBoxAscent) ? runMetrics.fontBoundingBoxAscent : ascent);
        fontDescent = Math.max(fontDescent, Number.isFinite(runMetrics.fontBoundingBoxDescent) ? runMetrics.fontBoundingBoxDescent : descent);
      }
      left = -minInkX;
      right = maxInkX;
    } else {
      context.font = `${style.fontStyle} ${style.fontVariant} ${style.fontWeight} ${style.fontSize} ${style.fontFamily}`;
      const metrics = context.measureText(text.textContent);
      left = Number.isFinite(metrics.actualBoundingBoxLeft) ? metrics.actualBoundingBoxLeft : 0;
      right = Number.isFinite(metrics.actualBoundingBoxRight) ? metrics.actualBoundingBoxRight : metrics.width;
      ascent = Number.isFinite(metrics.actualBoundingBoxAscent) ? metrics.actualBoundingBoxAscent : fontSize * .8;
      descent = Number.isFinite(metrics.actualBoundingBoxDescent) ? metrics.actualBoundingBoxDescent : fontSize * .2;
      fontAscent = Number.isFinite(metrics.fontBoundingBoxAscent) ? metrics.fontBoundingBoxAscent : ascent;
      fontDescent = Number.isFinite(metrics.fontBoundingBoxDescent) ? metrics.fontBoundingBoxDescent : descent;
    }
    const inkWidth = left + right;
    const inkHeight = ascent + descent;
    if (inkWidth <= 0 || inkHeight <= 0) continue;
    const baseline = (fontSize - fontAscent - fontDescent) / 2 + fontAscent;
    const scaleX = target.width / inkWidth;
    const measuredCapHeight = Number.parseFloat(slot.dataset.sensCapHeight || '');
    const desiredInkHeight = Number.isFinite(measuredCapHeight) && measuredCapHeight > 0
      ? Math.min(target.height, measuredCapHeight)
      : Math.min(target.height, fontSize * .73);
    const scaleY = desiredInkHeight / inkHeight;
    const translateX = left * scaleX;
    const verticalPadding = Math.max(0, (target.height - desiredInkHeight) / 2);
    const translateY = verticalPadding - (baseline - ascent) * scaleY;
    text.style.transform = `matrix(${scaleX},0,0,${scaleY},${translateX},${translateY})`;
  }
};
if (document.fonts?.ready) document.fonts.ready.then(fitSensText);
else addEventListener('DOMContentLoaded', fitSensText, { once: true });
"""
    return index, css, script, asset_files


def materialize_starter_project(
    document: dict[str, Any],
    asset_output_dir: str | None,
    *,
    no_store: bool,
) -> dict[str, Any] | None:
    """Write a content-addressed runnable DOM starter without reference pixels."""
    spec = document.get("reconstruction") or {}
    if (
        no_store
        or not asset_output_dir
        or spec.get("targetKind") != "web"
        or not spec.get("canvas")
    ):
        return None
    assets = _asset_records(spec)
    if len(assets) != len(spec.get("allowedRasterRegions") or []):
        return None
    font_assets = _font_asset_records()
    projection = _projection(document, assets, font_assets)
    canonical = json.dumps(
        projection, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    output_root = Path(asset_output_dir).expanduser().resolve()
    destination = output_root / f"sens-starter-{digest[:16]}"
    destination.mkdir(parents=True, exist_ok=True)
    index, css, script, asset_files = _render_project(
        document, assets, font_assets, digest
    )
    _atomic_text(destination / "index.html", index)
    _atomic_text(destination / "styles.css", css)
    _atomic_text(destination / "script.js", script)
    for name, content in asset_files:
        _atomic_bytes(destination / "assets" / name, content)
    result = {
        "schemaVersion": "sens-semantic-starter-2",
        "contentHash": f"sha256:{digest}",
        "directory": str(destination),
        "entryPath": str(destination / "index.html"),
        "stylesheetPath": str(destination / "styles.css"),
        "sourceFiles": ["index.html", "styles.css", "script.js"],
        "rasterAssetCount": len(assets),
        "fontAssetCount": len(font_assets),
        "representation": "live-dom-css-with-allowed-raster-assets-only",
        "rule": "Copy or serve this starter immediately, then repair it only from sens_review repairHints. Do not rewrite the first candidate from scratch.",
    }
    spec["starterProject"] = result
    artifacts = document.setdefault("artifacts", [])
    artifact_id = f"starter:{digest[:16]}"
    if not any(item.get("id") == artifact_id for item in artifacts):
        artifacts.append(
            {
                "id": artifact_id,
                "kind": "semantic-web-starter",
                "uri": str(destination / "index.html"),
                "mediaType": "text/html",
            }
        )
    return result
