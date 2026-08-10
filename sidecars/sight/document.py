"""Visual context document: canonical JSON + markdown renderer."""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from sight.ascii_map import render_ascii
from sight.ascii_text import reconstruct_monospace
from sight.decorative import detect_circular, detect_vertical
from sight.focus import recommend_focus
from sight.tokens import build_design_tokens

SCENE_SCHEMA_VERSION = "2.0.0"

_LABELS = {
    "ru": {
        "screen": "ЭКРАН", "vibe": "вайб", "palette": "палитра", "type": "типографика",
        "elements": "элементы (SoM, координаты 0–1000):", "decorative": "декоративный текст:",
        "graphics": "графика:", "ascii": "ascii 96×48:", "measurements": "измерения (факты, НЕ оценки):",
    },
    "en": {
        "screen": "SCREEN", "vibe": "vibe", "palette": "palette", "type": "typography",
        "elements": "elements (SoM, coords 0–1000):", "decorative": "decorative text:",
        "graphics": "graphics:", "ascii": "ascii 96×48:", "measurements": "measurements (facts, NOT judgments):",
    },
}


def _is_light(hex_color: str) -> bool:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) > 128


def normalize_box(box: list[int], width: int, height: int) -> list[int]:
    return [
        max(0, min(1000, round(box[0] * 1000 / width))),
        max(0, min(1000, round(box[1] * 1000 / height))),
        max(0, min(1000, round(box[2] * 1000 / width))),
        max(0, min(1000, round(box[3] * 1000 / height))),
    ]


def _uncertainty(kind: str, detail: str, value: float | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"kind": kind, "detail": detail}
    if value is not None:
        result["value"] = round(max(0.0, min(1.0, value)), 3)
    return result


def _claim(
    claim_id: str,
    subject: str,
    value: Any,
    epistemic: str,
    method: str,
    evidence: list[str],
    uncertainty: dict[str, Any],
    *,
    region: list[int] | None = None,
    confidence: float | None = None,
) -> dict[str, Any]:
    claim: dict[str, Any] = {
        "id": claim_id,
        "subject": subject,
        "value": value,
        "epistemic": epistemic,
        "method": method,
        "evidence": evidence,
        "uncertainty": uncertainty,
    }
    if region is not None:
        claim["regionNorm"] = region
    if confidence is not None:
        claim["confidence"] = round(float(confidence), 3)
    return claim


def _build_claims(
    dump: dict[str, Any],
    elements: list[dict[str, Any]],
    width: int,
    height: int,
    background: dict[str, Any] | None,
    vibe: str | None,
) -> list[dict[str, Any]]:
    claims = [
        _claim(
            "screen.size",
            "screen",
            [width, height],
            "measured",
            "image-decoder",
            ["source image dimensions"],
            _uncertainty("exact", "Decoded pixel dimensions; maximum error is 0 px."),
        )
    ]
    if background:
        claims.append(
            _claim(
                "palette.background",
                "screen.background",
                background["hex"],
                "measured",
                "opencv-color-quantization",
                ["dominant pixel cluster"],
                _uncertainty(
                    "quantized",
                    "Representative cluster color, not a CSS token or per-pixel constant.",
                    1.0 - float(background.get("ratio", 0.0)),
                ),
                confidence=float(background.get("ratio", 0.0)),
            )
        )
    scene = (dump.get("scene") or [None])[0]
    if scene:
        confidence = float(scene.get("confidence", 0.0))
        claims.append(
            _claim(
                "semantics.scene",
                "screen.scene",
                scene.get("label"),
                "inferred",
                scene.get("method", "clip-vit-b-32"),
                ["whole-image embedding matched against a fixed label set"],
                _uncertainty(
                    "model_confidence",
                    "Closed-set label; alternatives outside the candidate list are not tested.",
                    1.0 - confidence,
                ),
                confidence=confidence,
            )
        )
    if vibe:
        claims.append(
            _claim(
                "semantics.vibe",
                "screen.visual_style",
                vibe,
                "inferred",
                "local-vlm",
                ["whole source image"],
                _uncertainty(
                    "unverified_inference",
                    "Free-form model description; verify against measured tokens and regions.",
                ),
            )
        )
    for element in elements:
        element_id = element["id"]
        region = normalize_box(element["box"], width, height)
        claims.append(
            _claim(
                f"element.{element_id}.geometry",
                f"element:{element_id}",
                {"kind": element["kind"], "boxNorm": region},
                "measured",
                "sight-element-tree",
                [f"connected visual region for SoM element {element_id}"],
                _uncertainty(
                    "pixel_segmentation",
                    "Boundary follows detected pixels and can include antialiasing or merged neighbors.",
                ),
                region=region,
            )
        )
        if element.get("text"):
            claims.append(
                _claim(
                    f"element.{element_id}.text",
                    f"element:{element_id}.text",
                    element["text"],
                    "inferred",
                    element.get("method", "rapidocr"),
                    [
                        f"OCR glyph region for SoM element {element_id}",
                        *(
                            [f"alternative OCR readings: {element['alternatives']}"]
                            if element.get("alternatives")
                            else []
                        ),
                    ],
                    _uncertainty(
                        "model_confidence",
                        "Character recognition can confuse similar glyphs; inspect when exact text matters.",
                    ),
                    region=region,
                    confidence=element.get("confidence"),
                )
            )
    return claims


def _looks_like_ascii_text(items: list[dict[str, Any]]) -> bool:
    if len(items) < 2:
        return False
    text = "".join(str(item.get("text", "")) for item in items)
    visible = [char for char in text if not char.isspace()]
    if len(visible) < 4:
        return False
    ascii_marks = set(r"/\|_-+=*#@%<>()[]{}:;.,'")
    return sum(char in ascii_marks for char in visible) / len(visible) >= 0.35


def _box_in_source(
    box: list[int], coordinates: dict[str, Any]
) -> list[int]:
    transform = coordinates["analysisToSource"]
    source_width, source_height = coordinates["sourceSize"]
    converted = [
        round(box[0] * transform["scaleX"] + transform["offsetX"]),
        round(box[1] * transform["scaleY"] + transform["offsetY"]),
        round(box[2] * transform["scaleX"] + transform["offsetX"]),
        round(box[3] * transform["scaleY"] + transform["offsetY"]),
    ]
    return [
        max(0, min(source_width, converted[0])),
        max(0, min(source_height, converted[1])),
        max(0, min(source_width, converted[2])),
        max(0, min(source_height, converted[3])),
    ]


def _point_in_source(
    point: list[int | float], coordinates: dict[str, Any]
) -> list[int]:
    transform = coordinates["analysisToSource"]
    source_width, source_height = coordinates["sourceSize"]
    return [
        max(
            0,
            min(
                source_width,
                round(point[0] * transform["scaleX"] + transform["offsetX"]),
            ),
        ),
        max(
            0,
            min(
                source_height,
                round(point[1] * transform["scaleY"] + transform["offsetY"]),
            ),
        ),
    ]


def _center_inside(inner: list[int], outer: list[int]) -> bool:
    center_x = (inner[0] + inner[2]) / 2
    center_y = (inner[1] + inner[3]) / 2
    return outer[0] <= center_x <= outer[2] and outer[1] <= center_y <= outer[3]


def _intersection_area(first: list[int], second: list[int]) -> int:
    return max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0, min(first[3], second[3]) - max(first[1], second[1])
    )


def _resolve_target_kind(target_kind: str | None, intent: str | None) -> str:
    if target_kind is not None:
        normalized = target_kind.strip().casefold()
        if normalized not in {"visual", "web"}:
            raise ValueError("target_kind must be 'visual' or 'web'")
        return normalized
    intent_text = (intent or "").casefold()
    web_markers = (
        "website",
        "web page",
        "frontend",
        "html",
        "css",
        "сайт",
        "веб",
        "лендинг",
        "страниц",
        "сверст",
    )
    return "web" if any(marker in intent_text for marker in web_markers) else "visual"


def _reconstruction_element_kind(
    element: dict[str, Any], elements: list[dict[str, Any]]
) -> str:
    if element.get("kind") != "button":
        return str(element.get("kind"))
    labelled = any(
        item.get("kind") == "text"
        and _center_inside(item["box"], element["box"])
        for item in elements
    )
    return "labeled-control-candidate" if labelled else "decorative-shape"


def reconstruction_role(
    element: dict[str, Any], elements: list[dict[str, Any]]
) -> tuple[str, str]:
    kind = str(element.get("kind") or "unknown")
    if kind == "text":
        return (
            "live-text-required",
            "For web output, render this as selectable DOM text; never use a raster crop or glyph paths.",
        )
    if kind == "button":
        labelled = any(
            item.get("kind") == "text"
            and _center_inside(item["box"], element["box"])
            for item in elements
        )
        if labelled:
            return (
                "semantic-control-candidate",
                "For web output, use a semantic HTML control; do not flatten it into an image.",
            )
        return (
            "decorative-shape",
            "No contained text label was measured; preserve the visible shape without inventing interaction.",
        )
    if kind == "image":
        overlaps_text = any(
            item.get("kind") == "text"
            and _intersection_area(element["box"], item["box"]) > 0
            for item in elements
        )
        if overlaps_text:
            return (
                "raster-forbidden-overlaps-text",
                "This image candidate overlaps measured text and must not be used as a web raster asset.",
            )
        return (
            "raster-asset-candidate",
            "This graphic may remain raster only within its measured box and only after web review accepts it.",
        )
    return (
        kind,
        "Use the measured geometry and style; do not infer hidden behavior from appearance alone.",
    )


def _font_in_source(
    font: dict[str, Any], coordinates: dict[str, Any]
) -> dict[str, Any]:
    """Project pixel-valued glyph measurements out of an upscaled zoom."""
    projected = dict(font)
    transform = coordinates["analysisToSource"]
    for key in ("capHeight", "fontSize", "strokeWidthPx", "strokeWidthP75Px"):
        if isinstance(projected.get(key), (int, float)):
            projected[key] = round(projected[key] * transform["scaleY"], 1)
    if isinstance(projected.get("avgGlyphWidth"), (int, float)):
        projected["avgGlyphWidth"] = round(
            projected["avgGlyphWidth"] * transform["scaleX"], 1
        )
    word_boxes = projected.pop("wordBoxes", None) or []
    if word_boxes:
        projected["wordBoxesSource"] = [
            {
                "text": entry.get("text"),
                "box": _box_in_source(entry.get("box") or [], coordinates),
            }
            for entry in word_boxes
            if isinstance(entry, dict) and len(entry.get("box") or []) == 4
        ]
    projected["coordinateSpace"] = "source-pixels"
    return projected


def _build_reconstruction_spec(
    elements: list[dict[str, Any]],
    coordinates: dict[str, Any],
    skeleton: dict[str, Any],
    tree: dict[str, Any],
    surfaces: list[dict[str, Any]],
    symbol_art: list[dict[str, Any]],
    vector_paths: list[dict[str, Any]],
    focus_actions: list[dict[str, Any]],
    semantic_text: str | None,
    semantic_method: str | None,
    semantic_source_box: list[int] | None,
    semantic_typography: dict[str, Any] | None,
    semantic_typography_runs: list[dict[str, Any]],
    monospace_text: dict[str, Any] | None,
    target_kind: str,
) -> dict[str, Any]:
    source_width, source_height = coordinates["sourceSize"]
    transform = coordinates["analysisToSource"]
    regional_analysis = coordinates["regionInSource"] != [
        0,
        0,
        *coordinates["sourceSize"],
    ]
    text_elements = [item for item in elements if item.get("kind") == "text"]
    regional_semantic = bool(semantic_text and semantic_source_box is not None)

    def semantic_value_for(value: Any) -> str | None:
        normalized = "".join(
            character for character in str(value or "").casefold() if character.isalnum()
        )
        if not normalized:
            return None
        candidates: list[str] = []
        for source in [
            *(str(run.get("text") or "") for run in semantic_typography_runs),
            str(semantic_text or ""),
        ]:
            for line in source.splitlines():
                line = line.strip()
                if not line:
                    continue
                candidates.append(line)
                if not any(character.isspace() for character in str(value or "")):
                    candidates.extend(re.findall(r"[\w.-]+", line))
        ranked = []
        for candidate in dict.fromkeys(candidates):
            candidate_normalized = "".join(
                character
                for character in candidate.casefold()
                if character.isalnum()
            )
            if not candidate_normalized:
                continue
            similarity = SequenceMatcher(
                None, normalized, candidate_normalized
            ).ratio()
            length_ratio = min(len(normalized), len(candidate_normalized)) / max(
                len(normalized), len(candidate_normalized)
            )
            ranked.append((0.8 * similarity + 0.2 * length_ratio, similarity, length_ratio, candidate))
        if not ranked:
            return None
        _score, similarity, length_ratio, candidate = max(
            ranked, key=lambda item: item[0]
        )
        if similarity < 0.55 or length_ratio < 0.45:
            return None
        return candidate

    def typography_for(value: Any) -> dict[str, Any] | None:
        normalized = "".join(
            character for character in str(value or "").casefold() if character.isalnum()
        )
        ranked = []
        for run in semantic_typography_runs:
            run_style = {key: item for key, item in run.items() if key != "text"}
            for line in str(run.get("text") or "").splitlines():
                candidate = "".join(
                    character for character in line.casefold() if character.isalnum()
                )
                if not normalized or not candidate:
                    continue
                similarity = SequenceMatcher(None, normalized, candidate).ratio()
                if normalized in candidate or candidate in normalized:
                    similarity = max(similarity, 0.8)
                ranked.append((similarity, run_style))
        if ranked:
            similarity, style = max(ranked, key=lambda item: item[0])
            if similarity >= 0.45:
                return style
        if len(text_elements) == 1:
            return semantic_typography
        return None

    text_entries = []
    for element in sorted(
        text_elements,
        key=lambda item: (item["box"][1], item["box"][0]),
    ):
        confidence = float(element.get("confidence") or 0.0)
        verification = element.get("verified")
        stable = confidence >= 0.90 and verification is not False
        value_normalized = " ".join(
            re.findall(r"[\w.-]+", str(element.get("text") or "").casefold())
        )
        semantic_value = semantic_value_for(element.get("text"))
        semantic_value_normalized = " ".join(
            re.findall(r"[\w.-]+", str(semantic_value or "").casefold())
        )
        semantic_agrees = bool(
            value_normalized
            and semantic_value_normalized
            and value_normalized == semantic_value_normalized
        )
        deterministic_consensus = bool(
            verification is True
            and element.get("method") == "rapidocr-multiscale-consensus"
        )
        confirmed = stable and (semantic_agrees or deterministic_consensus)
        semantic_preferred = bool(
            regional_semantic
            and semantic_value
            and not semantic_agrees
        )
        status = (
            "confirmed"
            if confirmed
            else "stable-candidate"
            if stable
            else "candidate"
        )
        source_box = _box_in_source(element["box"], coordinates)
        font = _font_in_source(element.get("font") or {}, coordinates)
        typography = typography_for(element.get("text"))
        text_entries.append(
            {
                "elementId": element["id"],
                "value": element.get("text"),
                "status": status,
                "confidence": round(confidence, 3),
                "verified": verification,
                "method": element.get("method", "rapidocr"),
                "alternatives": element.get("alternatives", []),
                "confirmedBy": (
                    [
                        method
                        for method in (
                            element.get("method", "rapidocr"),
                            semantic_method if semantic_agrees else None,
                        )
                        if method
                    ]
                    if confirmed
                    else []
                ),
                "preferredValue": semantic_value if semantic_preferred else None,
                "resolutionStatus": (
                    "confirmed"
                    if confirmed
                    else "vlm-preferred-candidate"
                    if semantic_preferred
                    else "unresolved"
                ),
                "boxSource": source_box,
                "boxNormSource": normalize_box(
                    source_box, source_width, source_height
                ),
                "fontFeatures": font,
                "color": font.get("color"),
                "colorSource": font.get("colorSource"),
                "typographyCandidate": (
                    {
                        **typography,
                        "status": "candidate",
                        "epistemic": "inferred",
                        "method": semantic_method,
                    }
                    if typography
                    and regional_semantic
                    else None
                ),
                "fontStrategy": (
                    "measured-stroke-weight-then-class-and-glyph-metrics"
                    if typography
                    else (
                        "match-by-glyph-metrics"
                        if target_kind == "web"
                        else (
                            "preserve-as-asset-or-match-by-glyph-metrics"
                            if font.get("family") == "custom"
                            else "match-by-glyph-metrics"
                        )
                    )
                ),
            }
        )

    graphics = [item for item in elements if item.get("kind") == "image"]
    symbol_art_entries = []
    symbol_art_boxes = []
    for art in symbol_art:
        box = [int(round(value)) for value in art.get("box", [])]
        if len(box) != 4 or not art.get("text"):
            continue
        symbol_art_boxes.append(box)
        symbol_art_entries.append(
            {
                "text": art["text"],
                "boxSource": _box_in_source(box, coordinates),
                "rows": art.get("rows"),
                "columns": art.get("columns"),
                "cellWidth": round(
                    float(art.get("cellWidth") or 0) * transform["scaleX"], 2
                ),
                "rowPitch": round(
                    float(art.get("rowPitch") or 0) * transform["scaleY"], 2
                ),
                **(
                    {
                        "firstCellCenterX": round(
                            float(art["firstCellCenterX"]) * transform["scaleX"]
                            + transform["offsetX"],
                            2,
                        )
                    }
                    if art.get("firstCellCenterX") is not None
                    else {}
                ),
                **(
                    {
                        "firstBaselineY": round(
                            float(art["firstBaselineY"]) * transform["scaleY"]
                            + transform["offsetY"],
                            2,
                        )
                    }
                    if art.get("firstBaselineY") is not None
                    else {}
                ),
                **(
                    {
                        "glyphGeometry": {
                            kind: {
                                "width": round(
                                    float(metrics.get("width") or 0)
                                    * transform["scaleX"],
                                    2,
                                ),
                                "height": round(
                                    float(metrics.get("height") or 0)
                                    * transform["scaleY"],
                                    2,
                                ),
                                "centerOffsetX": round(
                                    float(metrics.get("centerOffsetX") or 0)
                                    * transform["scaleX"],
                                    2,
                                ),
                                "centerOffsetY": round(
                                    float(metrics.get("centerOffsetY") or 0)
                                    * transform["scaleY"],
                                    2,
                                ),
                            }
                            for kind, metrics in art["glyphGeometry"].items()
                            if isinstance(metrics, dict)
                        }
                    }
                    if isinstance(art.get("glyphGeometry"), dict)
                    and art["glyphGeometry"]
                    else {}
                ),
                **(
                    {"color": art["foregroundColor"]}
                    if art.get("foregroundColor")
                    else {}
                ),
                **(
                    {"backgroundColor": art["backgroundColor"]}
                    if art.get("backgroundColor")
                    else {}
                ),
                "alphabet": art.get("alphabet", []),
                "confidence": art.get("confidence"),
                "strategy": "render-as-live-selectable-monospace-text",
                "source": "measured",
                "method": art.get("method", "regular-symbol-grid"),
            }
        )
    allowed_raster_regions = []
    excluded_raster_candidates = []
    for graphic in graphics:
        source_box = _box_in_source(graphic["box"], coordinates)
        overlapping_symbol_art = next(
            (
                box
                for box in symbol_art_boxes
                if _intersection_area(graphic["box"], box) > 0
            ),
            None,
        )
        if target_kind == "web" and overlapping_symbol_art is not None:
            excluded_raster_candidates.append(
                {
                    "elementId": graphic["id"],
                    "boxSource": source_box,
                    "reason": "overlaps-live-symbol-art",
                    "symbolArtBoxSource": _box_in_source(
                        overlapping_symbol_art, coordinates
                    ),
                }
            )
            continue
        overlapping_text = next(
            (
                text
                for text in text_elements
                if _intersection_area(graphic["box"], text["box"]) > 0
            ),
            None,
        )
        if target_kind == "web" and overlapping_text is not None:
            excluded_raster_candidates.append(
                {
                    "elementId": graphic["id"],
                    "boxSource": source_box,
                    "reason": "overlaps-live-text",
                    "overlappingTextElementId": overlapping_text["id"],
                }
            )
            continue
        allowed_raster_regions.append(
            {
                "elementId": graphic["id"],
                "kind": "illustration-or-photo",
                "boxSource": source_box,
                "boxNormSource": normalize_box(
                    source_box, source_width, source_height
                ),
                "strategy": "extract-source-crop-verbatim",
                "implementation": "Crop boxSource once from the original reference into a local raster asset and place that asset at the same source-pixel box. Preserve the pixels verbatim; do not redraw, trace, describe, or semantically reinterpret this region.",
            }
        )
    primary_asset = None
    primary_candidates = (
        [
            item
            for item in graphics
            if any(
                allowed["elementId"] == item["id"]
                for allowed in allowed_raster_regions
            )
        ]
        if target_kind == "web"
        else graphics
    )
    if primary_candidates:
        graphic = max(
            primary_candidates,
            key=lambda item: (item["box"][2] - item["box"][0])
            * (item["box"][3] - item["box"][1]),
        )
        source_box = _box_in_source(graphic["box"], coordinates)
        primary_asset = {
            "elementId": graphic["id"],
            "boxSource": source_box,
            "boxNormSource": normalize_box(
                source_box, source_width, source_height
            ),
            "areaRatio": round(
                ((source_box[2] - source_box[0]) * (source_box[3] - source_box[1]))
                / max(1, source_width * source_height),
                4,
            ),
            "strategy": "extract-source-crop-verbatim",
            "rule": "Extract boxSource verbatim from the original reference and reuse it as one local raster asset. Do not trace, redraw, describe, or re-analyze an allowed principal asset.",
        }

    visual_controls = []
    decorative_shapes = []
    for element in elements:
        if element.get("kind") != "button":
            continue
        labels = [
            item["id"]
            for item in elements
            if item.get("kind") == "text"
            and _center_inside(item["box"], element["box"])
        ]
        entry = {
            "elementId": element["id"],
            "boxSource": _box_in_source(element["box"], coordinates),
            "visibleBoundary": bool(element.get("borderColor")),
            "background": element.get("background"),
            "borderColor": element.get("borderColor"),
            "borderWidth": element.get("borderWidth"),
            "cornerRadius": element.get("cornerRadius"),
            "source": element.get("source", "measured"),
        }
        if labels:
            visual_controls.append(
                {
                    **entry,
                    "labelElementIds": labels,
                    "interaction": (
                        "semantic-control-required"
                        if target_kind == "web"
                        else "unknown"
                    ),
                }
            )
        else:
            decorative_shapes.append(
                {
                    **entry,
                    "reason": "No text label is geometrically contained; do not infer UI behavior.",
                }
            )

    layout_regions: list[dict[str, Any]] = []

    def add_layout_regions(node: dict[str, Any]) -> None:
        if node.get("kind") not in {"screen", "texture"}:
            box = node.get("box") or []
            if len(box) == 4:
                element_ids = []
                for item in node.get("elements") or []:
                    element_id = item.get("id") if isinstance(item, dict) else item
                    if isinstance(element_id, int):
                        element_ids.append(element_id)
                child_ids = [
                    child.get("id")
                    for child in node.get("children") or []
                    if isinstance(child, dict) and isinstance(child.get("id"), int)
                ]
                layout_regions.append(
                    {
                        "regionId": node.get("id"),
                        "role": node.get("role") or "content",
                        "kind": node.get("kind") or "section",
                        "boxSource": _box_in_source(box, coordinates),
                        "elementIds": element_ids,
                        "childRegionIds": child_ids,
                        "source": "measured",
                    }
                )
        for child in node.get("children") or []:
            if isinstance(child, dict):
                add_layout_regions(child)

    if isinstance(tree, dict):
        add_layout_regions(tree)
    layout_regions = layout_regions[:32]

    surface_entries = []
    for surface in surfaces[:24]:
        box = surface.get("box") or []
        if len(box) != 4:
            continue
        surface_entries.append(
            {
                "boxSource": _box_in_source(box, coordinates),
                "background": surface.get("background"),
                "borderColor": surface.get("borderColor"),
                "borderWidth": surface.get("borderWidth"),
                "cornerRadius": surface.get("cornerRadius"),
                "source": surface.get("source", "measured"),
            }
        )

    icon_entries = []
    for element in elements:
        if element.get("kind") != "icon":
            continue
        icon_entries.append(
            {
                "elementId": element["id"],
                "name": element.get("icon") or element.get("name") or "unknown",
                "boxSource": _box_in_source(element["box"], coordinates),
                "color": element.get("color"),
                "strategy": "css-or-inline-svg",
                "source": element.get("source", "measured"),
            }
        )

    blocking_uncertainties = [
        {
            "kind": "text_candidate",
            "elementId": entry["elementId"],
            "value": entry.get("preferredValue") or entry["value"],
            "candidates": list(
                dict.fromkeys(
                    value
                    for value in (entry["value"], entry.get("preferredValue"))
                    if value
                )
            ),
            "action": "Verify with sens_zoom before coding exact copy.",
        }
        for entry in text_entries
        if entry["status"] != "confirmed"
    ]
    verification_candidates = sorted(
        (entry for entry in text_entries if entry["status"] != "confirmed"),
        key=lambda entry: (
            -(
                (1.0 - entry["confidence"])
                + (
                    (entry["boxSource"][2] - entry["boxSource"][0])
                    * (entry["boxSource"][3] - entry["boxSource"][1])
                    / max(1, source_width * source_height)
                )
            ),
            entry["boxSource"][1],
            entry["boxSource"][0],
        ),
    )[:4]
    text_verification_plan = [] if regional_analysis else [
        {
            "tool": "sens_zoom",
            "reason": "Resolve an exact-text candidate before implementation.",
            "evidence": entry["value"],
            "arguments": {
                "region": {
                    "x": entry["boxSource"][0],
                    "y": entry["boxSource"][1],
                    "width": entry["boxSource"][2] - entry["boxSource"][0],
                    "height": entry["boxSource"][3] - entry["boxSource"][1],
                },
                "profile": "reconstruct",
                "response": "compact",
                "targetKind": target_kind,
            },
        }
        for entry in verification_candidates
    ]
    structural_lines = []
    for segment in skeleton.get("segments", []):
        box = [int(round(value)) for value in segment.get("box", [])]
        if len(box) != 4:
            continue
        thickness = float(segment.get("thickness") or 0)
        length = float(segment.get("length") or 0)
        # Long thin rules are layout structure. Thick elongated artwork (for
        # example a road shadow under an illustration) remains decoration.
        if thickness <= 0 or length / thickness < 60:
            continue
        source_box = _box_in_source(box, coordinates)
        start = segment.get("start")
        end = segment.get("end")
        structural_lines.append(
            {
                "orientation": segment.get("orientation"),
                "boxSource": source_box,
                "startSource": (
                    _point_in_source(start, coordinates)
                    if isinstance(start, list) and len(start) == 2
                    else None
                ),
                "endSource": (
                    _point_in_source(end, coordinates)
                    if isinstance(end, list) and len(end) == 2
                    else None
                ),
                "thickness": round(thickness, 1),
                "length": round(length, 1),
                "color": segment.get("color"),
                "source": "measured",
            }
        )
    measured_vector_paths = []
    for path in vector_paths:
        box = [int(round(value)) for value in path.get("box", [])]
        points = path.get("points") or []
        if len(box) != 4 or len(points) < 3:
            continue
        measured_vector_paths.append(
            {
                "boxSource": _box_in_source(box, coordinates),
                "pointsSource": [
                    _point_in_source(point, coordinates)
                    for point in points
                    if isinstance(point, list) and len(point) == 2
                ],
                "strokeColor": path.get("strokeColor"),
                "strokeWidth": round(
                    float(path.get("strokeWidth") or 1)
                    * float(transform.get("scaleY") or 1),
                    2,
                ),
                "fill": "none",
                "source": "measured",
                "method": path.get("method", "saturated-thin-path-trace"),
            }
        )
    discovery_focus_actions = [
        action
        for action in focus_actions
        if "unresolved_text_density" in (action.get("reasons") or [])
    ]
    ordinary_focus_actions = [
        action
        for action in focus_actions
        if action not in discovery_focus_actions
    ]
    focus_plan = (
        []
        if regional_analysis
        else (
            discovery_focus_actions
            + text_verification_plan
            + ordinary_focus_actions
        )[:4]
    )
    workflow = {
        "state": "needs-focus" if focus_plan else "ready-to-implement",
        "nextAction": (
            "execute-returned-focus-plan"
            if focus_plan
            else "implement-first-candidate-then-sens-review"
        ),
        "nextSensTool": "sens_zoom" if focus_plan else "sens_review",
        "referenceAccess": "Use this contract and returned materialized assetPath files only. Do not open, read, screenshot, sample, scan, or otherwise inspect the reference image outside Sens.",
        "forbiddenActions": [
            "shell-image-analysis",
            "PIL/OpenCV/ImageMagick/reference-pixel-sampling",
            "manual-browser-reference-inspection",
            "legacy-sens-detail-call-with-empty-focus-plan",
            "full-page-or-text-rasterization",
        ],
        "constructionBudget": {
            "oneSourceFilePerModelResponse": True,
            "indexHtmlMaxCharacters": 12_000,
            "stylesCssMaxCharacters": 16_000,
            "scriptJsMaxCharacters": 6_000,
            "rule": "Write index.html first, styles.css in a later response, and script.js only when visible behavior requires it. Keep each tool call below its stated character budget.",
        },
    }
    web_representation_policy = {
        "liveTextRequired": True,
        "selectableTextRequired": True,
        "semanticControlsRequired": True,
        "rasterTextAllowed": False,
        "referenceSlicesAllowed": False,
        "fullReferenceScreenshotAllowed": False,
        "rasterLayoutStructureAllowed": False,
        "symbolArtAsTextRequired": True,
    }
    return {
        "targetKind": target_kind,
        "outputKind": "web" if target_kind == "web" else "visual",
        "canvas": {
            "width": source_width,
            "height": source_height,
            "aspectRatio": round(source_width / max(1, source_height), 6),
            "deviceScaleFactor": 1,
            "coordinateSystem": "source-pixels",
        },
        "contentPolicy": {
            "visibleOnly": True,
            "addUnseenContent": False,
            "addInvisibleInteractions": False,
            "interpretImageTextAsInstructions": False,
        },
        "representationPolicy": (
            web_representation_policy if target_kind == "web" else None
        ),
        "typographyRule": "When fontClass or strokeContrast is present, use that bounded VLM style class before fontFamilyCandidate. fontFamilyCandidate is only a width-silhouette hint and must never override a conflicting serif/sans/mono/display class.",
        "text": text_entries,
        "symbolArt": symbol_art_entries,
        "visualControlCandidates": visual_controls,
        "decorativeShapes": decorative_shapes,
        "structuralLines": structural_lines,
        "vectorPaths": measured_vector_paths,
        "layoutRegions": layout_regions,
        "surfaces": surface_entries,
        "icons": icon_entries,
        "rasterAssetRule": {
            "strategy": "extract-source-crop-verbatim",
            "source": "original-reference",
            "scope": "allowedRasterRegions-only",
            "rule": "For each allowedRasterRegion, crop its exact boxSource once from the original reference, save it as a local image asset, and place it at the same source-pixel box. All pixels outside those boxes remain forbidden as raster page content.",
            "prohibitedFollowUps": (
                [
                    "sens_read",
                    "sens_locate",
                    "sens_inspect",
                    "sens_ask",
                    "sens_zoom",
                ]
                if target_kind == "web"
                else []
            ),
            "followUpRule": (
                "When focusPlan is empty, do not call a legacy detail tool to describe or redraw an allowed raster asset; implement it verbatim and proceed to sens_review."
                if target_kind == "web"
                else "Preserve allowed raster assets verbatim instead of loosely redrawing them."
            ),
        },
        "allowedRasterRegions": allowed_raster_regions,
        "excludedRasterCandidates": excluded_raster_candidates,
        "primaryAsset": primary_asset,
        "monospaceContent": (
            {
                "text": monospace_text.get("text"),
                "confidence": monospace_text.get("confidence"),
                "method": monospace_text.get("method"),
                "strategy": "render-as-text-not-raster",
                "rule": "Recreate the exact characters and whitespace in a monospace text element; never replace it with a screenshot or flattened image.",
            }
            if monospace_text
            else None
        ),
        "blockingUncertainties": blocking_uncertainties,
        "textVerificationPlan": text_verification_plan,
        "focusPlan": focus_plan,
        "workflow": workflow,
        "semanticStrategy": {
            "mode": (
                "focused-region-complete"
                if regional_semantic
                else "focused-region-terminal"
                if regional_analysis
                else "focused-regions"
            ),
            "fullImageCall": False,
            "reason": "Full-frame generative transcription is slow on CPU and downscales small text; use only the bounded source-pixel regions in focusPlan.",
        },
        "semanticTextCandidate": (
            {
                "text": semantic_text,
                "status": "candidate",
                "epistemic": "inferred",
                "method": semantic_method,
                "sourceBox": semantic_source_box,
                "typography": semantic_typography,
                "typographyRuns": semantic_typography_runs,
                "instruction": "Reconcile this independent reading with per-region OCR; disagreements require sens_zoom and must not be silently merged.",
            }
            if semantic_text
            else None
        ),
        "completionGate": (
            {
                "tool": "sens_review",
                "requires": ["visual-pass", "web-pass"],
                "rule": "A visual similarity pass alone cannot complete a web reconstruction.",
                "repairPolicy": "Apply only measured repairHints, checkpoint every new champion, and obey iterationPolicy rollback or stop actions.",
            }
            if target_kind == "web"
            else {
                "tool": "sens_compare",
                "requires": ["visual-pass"],
            }
        ),
        "implementationRules": [
            "Use one fixed source-pixel coordinate system for size and position.",
            f"Render at exactly {source_width}x{source_height} CSS pixels with device scale factor 1.",
            "Do not mix viewport-relative sizing with a capped inner positioning canvas.",
            "Do not add sections, copy, controls, or decoration that are not visible in the reference.",
            "Treat control-looking shapes as visual elements; add behavior only with external interaction evidence.",
            *(
                [
                    "Render every visible word as live selectable DOM text; never use a screenshot crop, SVG path, canvas, or raster image for text.",
                    "Implement labeled control candidates as semantic HTML controls with visible focus and pointer behavior.",
                    "Implement structural lines with CSS or vector primitives, never with reference-image slices.",
                    "Render symbolArt exactly as preformatted selectable monospace characters; never use its source pixels as an asset.",
                    "Use raster assets only inside allowedRasterRegions; never use the full reference or sliced reference layout as page content.",
                    "For each allowedRasterRegion, extract its exact boxSource once from the original reference and preserve it verbatim; do not redraw, trace, describe, or semantically reinterpret it.",
                    "When focusPlan is empty, do not call sens_read, sens_locate, sens_inspect, sens_ask, or sens_zoom; implement the returned contract and proceed directly to sens_review.",
                    "Run sens_review after each material repair; apply only its measured repairHints, checkpoint every new champion, and never replace them with manual pixel-scanning scripts.",
                    "If iterationPolicy reports rollback-to-champion or stop-and-return-champion, obey it before any further edit.",
                    "Stop only when visualPass, webPass, and canComplete are all true with no blockingReasons.",
                ]
                if target_kind == "web"
                else [
                    "Run sens_compare with fit=strict after each material repair and stop only when canComplete is true."
                ]
            ),
        ],
    }


def build_document(
    dump: dict[str, Any],
    image: Any,
    vlm: Any | None = None,
    image_path: str | None = None,
    lang: str = "ru",
    intent: str | None = None,
    max_semantic_calls: int = 2,
    profile: str = "analyze",
    target_kind: str | None = None,
) -> dict[str, Any]:
    if profile not in {"analyze", "reconstruct"}:
        raise ValueError("profile must be 'analyze' or 'reconstruct'")
    resolved_target_kind = _resolve_target_kind(target_kind, intent)
    width, height = dump["image"]["width"], dump["image"]["height"]
    coordinates = dump.get("coordinates") or {
        "sourceSize": [width, height],
        "regionInSource": [0, 0, width, height],
        "analysisSize": [width, height],
        "analysisToSource": {
            "scaleX": 1.0,
            "scaleY": 1.0,
            "offsetX": 0.0,
            "offsetY": 0.0,
        },
    }
    elements = dump.get("elements", [])
    texts = [e for e in elements if e.get("kind") == "text"]
    graphics = [e for e in elements if e.get("kind") == "image"]
    colors = dump.get("colors", [])
    background = dump.get("canvasBackground") or (colors[0] if colors else None)
    design = dump.get("design", {})
    facts = design.get("facts", design.get("issues", []))

    semantic_calls = 0
    semantic_budget_exhausted = False
    semantic_warnings: list[dict[str, str]] = []

    def infer(method: str, *args: Any) -> Any | None:
        nonlocal semantic_calls, semantic_budget_exhausted
        if vlm is None or not image_path:
            return None
        if semantic_calls >= max(0, max_semantic_calls):
            semantic_budget_exhausted = True
            return None
        semantic_calls += 1
        try:
            return getattr(vlm, method)(*args)
        except Exception as error:  # noqa: BLE001 - semantics are optional
            semantic_warnings.append(
                {
                    "code": "semantic_call_failed",
                    "message": f"Local VLM {method} failed: {error}",
                    "recovery": "Use deterministic claims or retry the specific region with sens_ask/sens_zoom.",
                }
            )
            return None

    semantic_text = None
    semantic_method = None
    semantic_source_box = None
    semantic_typography = None
    semantic_typography_runs: list[dict[str, Any]] = []
    if profile == "reconstruct":
        region_in_source = [int(value) for value in coordinates["regionInSource"]]
        whole_source = [0, 0, *coordinates["sourceSize"]]
        if region_in_source != whole_source:
            semantic_source_box = region_in_source
            if vlm is not None and hasattr(vlm, "inspect_text"):
                semantic_method = "local-vlm-region-text-inspection"
                inspection = infer("inspect_text", image_path, region_in_source)
                if isinstance(inspection, dict):
                    semantic_text = inspection.get("text")
                    semantic_typography = inspection.get("typography")
                    semantic_typography_runs = inspection.get("runs") or []
            else:
                semantic_method = "local-vlm-region-transcription"
                semantic_text = infer("transcribe", image_path, region_in_source)
        vibe = None
    else:
        vibe = infer("vibe", image_path)

    decorative = []
    for group in detect_circular(texts) + detect_vertical(texts):
        entry = dict(group)
        entry["box_norm"] = normalize_box(group["box"], width, height)
        if profile != "reconstruct" and vlm is not None and image_path:
            entry["transcription"] = infer("transcribe", image_path, group["box"])
        decorative.append(entry)

    graphic_docs = []
    for g in graphics:
        entry: dict[str, Any] = {"id": g["id"], "box_norm": normalize_box(g["box"], width, height)}
        if profile != "reconstruct" and vlm is not None and image_path:
            entry["caption"] = infer("describe", image_path, g["box"])
        graphic_docs.append(entry)

    source = dump.get("source") or {"id": "unknown", "mediaType": "image"}
    semantics_status = "ok" if vlm is not None and image_path else "unavailable"
    if semantic_budget_exhausted or semantic_warnings:
        semantics_status = "partial"
    warnings = list(dump.get("warnings", [])) + semantic_warnings
    if semantics_status == "unavailable":
        warnings.append(
            {
                "code": "semantics_unavailable",
                "message": "No local VLM is loaded; semantic captions and visual-style inference are omitted.",
                "recovery": "Install a supported local vision pack, or continue with measured geometry, color, and OCR.",
            }
        )
    if semantic_budget_exhausted:
        warnings.append(
            {
                "code": "semantic_budget_exhausted",
                "message": f"The bounded semantic pass used {semantic_calls} VLM call(s); remaining regions were not captioned.",
                "recovery": "Use the returned sens_zoom/sens_ask actions only for regions relevant to the task.",
            }
        )
    if coordinates["regionInSource"] != [0, 0, *coordinates["sourceSize"]]:
        warnings.append(
            {
                "code": "regional_analysis",
                "message": "This scene describes a crop, not the complete source image.",
                "recovery": "Use the source transform for original-pixel positions or call sens_see on the full image.",
            }
        )
    artifacts = []
    if dump.get("somPath"):
        artifacts.append(
            {
                "id": f"som:{source['id']}",
                "kind": "set-of-marks",
                "uri": dump["somPath"],
            }
        )

    ocr_items = dump.get("ocr", [])
    monospace_text = None
    if _looks_like_ascii_text(ocr_items):
        monospace_text = reconstruct_monospace(
            ocr_items,
            image_width=width,
            image_height=height,
        )
    claims = _build_claims(dump, elements, width, height, background, vibe)
    if monospace_text and monospace_text.get("text") is not None:
        claims.append(
            _claim(
                "content.monospace_text",
                "screen.monospace_text",
                monospace_text["text"],
                "inferred",
                monospace_text["method"],
                ["OCR spans placed on an estimated fixed-width grid"],
                _uncertainty(
                    "grid_reconstruction",
                    "Whitespace is geometrically reconstructed; question marks identify uncertain glyphs.",
                    1.0 - float(monospace_text.get("confidence") or 0.0),
                ),
                confidence=monospace_text.get("confidence"),
            )
        )
    regional_analysis = coordinates["regionInSource"] != [
        0,
        0,
        *coordinates["sourceSize"],
    ]
    focus_actions = (
        []
        if regional_analysis
        else [
            {
                "tool": focus["tool"],
                "when": "before_using_uncertain_or_small_detail",
                "reason": f"Re-analyze {focus['evidence']!r} at higher effective resolution.",
                "arguments": {
                    "region": focus["region"],
                    **(
                        {
                            "profile": "reconstruct",
                            "response": "compact",
                            "targetKind": resolved_target_kind,
                        }
                        if profile == "reconstruct"
                        else {}
                    ),
                },
                "priority": focus["priority"],
                "reasons": focus["reasons"],
            }
            for focus in recommend_focus(dump, intent=intent)
        ]
    )

    return {
        "schemaVersion": SCENE_SCHEMA_VERSION,
        "profile": profile,
        "reconstruction": (
            _build_reconstruction_spec(
                elements,
                coordinates,
                dump.get("skeleton") or {},
                dump.get("tree") or {},
                dump.get("surfaces") or [],
                dump.get("symbolArt") or [],
                dump.get("vectorPaths") or [],
                focus_actions,
                semantic_text,
                semantic_method,
                semantic_source_box,
                semantic_typography,
                semantic_typography_runs,
                monospace_text,
                resolved_target_kind,
            )
            if profile == "reconstruct"
            else None
        ),
        "source": source,
        "coordinateSpaces": {
            "source": {"unit": "pixel", "size": coordinates["sourceSize"]},
            "analysis": {
                "unit": "pixel",
                "size": coordinates["analysisSize"],
                "regionInSource": coordinates["regionInSource"],
                "sourceTransform": coordinates["analysisToSource"],
            },
            "normalized": {"unit": "permille", "range": [0, 1000]},
        },
        "lang": lang,
        "header": {
            "size": [width, height],
            "theme": "light" if background and _is_light(background["hex"]) else "dark",
            "background": background["hex"] if background else None,
            "scene": (dump.get("scene") or [{}])[0].get("label"),
            "vibe": vibe,
        },
        "tokens": build_design_tokens(dump),
        "elements": [
            {
                "id": e["id"],
                "kind": (
                    _reconstruction_element_kind(e, elements)
                    if profile == "reconstruct"
                    else e["kind"]
                ),
                "text": e.get("text"),
                "box_norm": normalize_box(e["box"], width, height),
                "font": e.get("font"),
            }
            for e in elements
        ],
        "decorative": decorative,
        "graphics": graphic_docs,
        "ascii": render_ascii(image),
        "monospaceText": monospace_text,
        "measurements": [
            {"kind": f.get("kind"), "detail": f.get("detail")}
            for f in facts
            if isinstance(f, dict)
        ],
        "claims": claims,
        "uncertainty": {
            "policy": "Absence of a claim means unknown, not false. Inferred text and semantics require verification when exactness matters."
        },
        "artifacts": artifacts,
        "warnings": warnings,
        "nextActions": focus_actions + [
            {
                "tool": (
                    "sens_review"
                    if profile == "reconstruct" and resolved_target_kind == "web"
                    else "sens_compare"
                ),
                "when": "after_implementation",
                "reason": (
                    "Verify visual convergence plus live DOM text, semantic controls, and raster-use integrity."
                    if profile == "reconstruct" and resolved_target_kind == "web"
                    else "Measure whether the rendered candidate converged toward this reference."
                ),
            }
        ],
        "semantics_status": semantics_status,
    }


def render_markdown(doc: dict[str, Any]) -> str:
    lab = _LABELS.get(doc.get("lang", "ru"), _LABELS["ru"])
    h = doc["header"]
    coordinates = doc["coordinateSpaces"]["analysis"]
    truth_counts = {kind: 0 for kind in ("measured", "inferred", "observed")}
    for claim in doc.get("claims", []):
        if claim.get("epistemic") in truth_counts:
            truth_counts[claim["epistemic"]] += 1
    lines = [
        f"VISUAL SCENE {doc['schemaVersion']} · source {doc['source']['id']}",
        f"COORDS analysis={coordinates['size']} regionInSource={coordinates['regionInSource']} transform={coordinates['sourceTransform']}",
        f"{lab['screen']} {h['size'][0]}×{h['size'][1]} · {h['theme']} · фон {h['background'] or '—'} · сцена: {h['scene'] or '—'}",
        f"{lab['vibe']}: {h['vibe'] or '—'} [inferred]",
        f"{lab['palette']}: " + " · ".join(f"{role} {c['$value']}" for role, c in doc["tokens"]["color"].items()),
        f"{lab['type']}: шкала {doc['tokens']['typography']['scale']['$value']}px · spacing base {doc['tokens']['spacing']['base']['$value']}",
        lab["elements"],
    ]
    for e in doc["elements"]:
        b = e["box_norm"]
        font = e["font"] or {}
        text = f' "{e["text"]}"' if e.get("text") else ""
        lines.append(
            f" [{e['id']}] {e['kind']}{text} @[{b[0]},{b[1]}-{b[2]},{b[3]}] {font.get('family', '?')}~{font.get('fontSize', '?')}px"
        )
    if doc["decorative"]:
        lines.append(lab["decorative"])
        for d in doc["decorative"]:
            lines.append(f" {d['direction']} ids={d['ids']}: {d.get('transcription') or '—'}")
    if doc["graphics"]:
        lines.append(lab["graphics"])
        for g in doc["graphics"]:
            b = g["box_norm"]
            lines.append(f" g{g['id']} @[{b[0]},{b[1]}-{b[2]},{b[3]}] {g.get('caption') or '—'}")
    lines.append(lab["ascii"])
    lines.append(doc["ascii"])
    if doc.get("monospaceText"):
        candidate = doc["monospaceText"]
        lines.append(
            f"ASCII TEXT {candidate['status']} [inferred, {candidate['method']}]:"
        )
        lines.append(candidate["text"] or "—")
        if candidate["ambiguities"]:
            lines.append(f"ASCII AMBIGUITIES: {candidate['ambiguities']}")
    lines.append(f"{lab['measurements']} {len(doc['measurements'])}")
    lines += [f" - {m['kind']}: {m['detail']}" for m in doc["measurements"]]
    lines.append(
        "TRUTH "
        + " ".join(f"{kind}={truth_counts[kind]}" for kind in truth_counts)
        + " · inferred claims are hypotheses; inspect exact text and regions when consequential"
    )
    lines += [
        f"WARNING {warning['code']}: {warning['message']}"
        for warning in doc.get("warnings", [])
    ]
    lines += [
        f"NEXT {action['tool']} when={action['when']}: {action['reason']}"
        for action in doc.get("nextActions", [])
    ]
    lines.append(f"semantics: {doc['semantics_status']}")
    return "\n".join(lines)
