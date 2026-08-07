"""Visual context document: canonical JSON + markdown renderer."""
from __future__ import annotations

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
                    "rapidocr",
                    [f"OCR glyph region for SoM element {element_id}"],
                    _uncertainty(
                        "model_confidence",
                        "Character recognition can confuse similar glyphs; inspect when exact text matters.",
                    ),
                    region=region,
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


def build_document(
    dump: dict[str, Any],
    image: Any,
    vlm: Any | None = None,
    image_path: str | None = None,
    lang: str = "ru",
    intent: str | None = None,
    max_semantic_calls: int = 2,
) -> dict[str, Any]:
    width, height = dump["image"]["width"], dump["image"]["height"]
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

    vibe = infer("vibe", image_path)

    decorative = []
    for group in detect_circular(texts) + detect_vertical(texts):
        entry = dict(group)
        entry["box_norm"] = normalize_box(group["box"], width, height)
        if vlm is not None and image_path:
            entry["transcription"] = infer("transcribe", image_path, group["box"])
        decorative.append(entry)

    graphic_docs = []
    for g in graphics:
        entry: dict[str, Any] = {"id": g["id"], "box_norm": normalize_box(g["box"], width, height)}
        if vlm is not None and image_path:
            entry["caption"] = infer("describe", image_path, g["box"])
        graphic_docs.append(entry)

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
            "arguments": {"region": focus["region"]},
            "priority": focus["priority"],
            "reasons": focus["reasons"],
        }
        for focus in recommend_focus(dump, intent=intent)
    ]

    return {
        "schemaVersion": SCENE_SCHEMA_VERSION,
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
                "kind": e["kind"],
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
