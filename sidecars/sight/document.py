"""Visual context document: canonical JSON + markdown renderer."""
from __future__ import annotations

import re
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


def _center_inside(inner: list[int], outer: list[int]) -> bool:
    center_x = (inner[0] + inner[2]) / 2
    center_y = (inner[1] + inner[3]) / 2
    return outer[0] <= center_x <= outer[2] and outer[1] <= center_y <= outer[3]


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


def _font_in_source(
    font: dict[str, Any], coordinates: dict[str, Any]
) -> dict[str, Any]:
    """Project pixel-valued glyph measurements out of an upscaled zoom."""
    projected = dict(font)
    transform = coordinates["analysisToSource"]
    for key in ("capHeight", "fontSize"):
        if isinstance(projected.get(key), (int, float)):
            projected[key] = round(projected[key] * transform["scaleY"], 1)
    if isinstance(projected.get("avgGlyphWidth"), (int, float)):
        projected["avgGlyphWidth"] = round(
            projected["avgGlyphWidth"] * transform["scaleX"], 1
        )
    projected["coordinateSpace"] = "source-pixels"
    return projected


def _build_reconstruction_spec(
    elements: list[dict[str, Any]],
    coordinates: dict[str, Any],
    focus_actions: list[dict[str, Any]],
    semantic_text: str | None,
    semantic_method: str | None,
    semantic_source_box: list[int] | None,
    monospace_text: dict[str, Any] | None,
) -> dict[str, Any]:
    source_width, source_height = coordinates["sourceSize"]
    semantic_normalized = " ".join(
        re.findall(r"[\w.-]+", (semantic_text or "").casefold())
    )
    text_elements = [item for item in elements if item.get("kind") == "text"]
    regional_semantic = bool(semantic_text and semantic_source_box is not None)
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
        semantic_agrees = bool(
            value_normalized
            and semantic_normalized
            and value_normalized in semantic_normalized
        )
        confirmed = stable and semantic_agrees
        semantic_preferred = bool(
            regional_semantic
            and len(text_elements) == 1
            and not semantic_agrees
            and confidence < 0.80
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
                        element.get("method", "rapidocr"),
                        semantic_method,
                    ]
                    if confirmed
                    else []
                ),
                "preferredValue": semantic_text if semantic_preferred else None,
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
                "fontStrategy": (
                    "preserve-as-asset-or-match-by-glyph-metrics"
                    if font.get("family") == "custom"
                    else "match-by-glyph-metrics"
                ),
            }
        )

    graphics = [item for item in elements if item.get("kind") == "image"]
    primary_asset = None
    if graphics:
        graphic = max(
            graphics,
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
            "strategy": "preserve-or-trace",
            "rule": "Do not loosely redraw the principal illustration; preserve the source pixels when allowed or trace its measured layers and silhouette.",
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
        }
        if labels:
            visual_controls.append(
                {**entry, "labelElementIds": labels, "interaction": "unknown"}
            )
        else:
            decorative_shapes.append(
                {
                    **entry,
                    "reason": "No text label is geometrically contained; do not infer UI behavior.",
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
    text_verification_plan = [] if regional_semantic else [
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
            },
        }
        for entry in verification_candidates
    ]
    return {
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
        "text": text_entries,
        "visualControlCandidates": visual_controls,
        "decorativeShapes": decorative_shapes,
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
        "focusPlan": (
            []
            if regional_semantic
            else (text_verification_plan + focus_actions)[:4]
        ),
        "semanticStrategy": {
            "mode": (
                "focused-region-complete"
                if regional_semantic
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
                "instruction": "Reconcile this independent reading with per-region OCR; disagreements require sens_zoom and must not be silently merged.",
            }
            if semantic_text
            else None
        ),
        "implementationRules": [
            "Use one fixed source-pixel coordinate system for size and position.",
            f"Render at exactly {source_width}x{source_height} CSS pixels with device scale factor 1.",
            "Do not mix viewport-relative sizing with a capped inner positioning canvas.",
            "Do not add sections, copy, controls, or decoration that are not visible in the reference.",
            "Treat control-looking shapes as visual elements; add behavior only with external interaction evidence.",
            "Run sens_compare with fit=strict after each material repair and stop only when canComplete is true.",
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
) -> dict[str, Any]:
    if profile not in {"analyze", "reconstruct"}:
        raise ValueError("profile must be 'analyze' or 'reconstruct'")
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
    background = colors[0] if colors else None
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
    if profile == "reconstruct":
        region_in_source = [int(value) for value in coordinates["regionInSource"]]
        whole_source = [0, 0, *coordinates["sourceSize"]]
        if region_in_source != whole_source:
            semantic_source_box = region_in_source
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
    focus_actions = [
        {
            "tool": focus["tool"],
            "when": "before_using_uncertain_or_small_detail",
            "reason": f"Re-analyze {focus['evidence']!r} at higher effective resolution.",
            "arguments": {
                "region": focus["region"],
                **(
                    {"profile": "reconstruct", "response": "compact"}
                    if profile == "reconstruct"
                    else {}
                ),
            },
            "priority": focus["priority"],
            "reasons": focus["reasons"],
        }
        for focus in recommend_focus(dump, intent=intent)
    ]

    return {
        "schemaVersion": SCENE_SCHEMA_VERSION,
        "profile": profile,
        "reconstruction": (
            _build_reconstruction_spec(
                elements,
                coordinates,
                focus_actions,
                semantic_text,
                semantic_method,
                semantic_source_box,
                monospace_text,
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
                "tool": "sens_compare",
                "when": "after_implementation",
                "reason": "Measure whether the rendered candidate converged toward this reference.",
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
