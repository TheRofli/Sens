"""Visual context document: canonical JSON + markdown renderer."""
from __future__ import annotations

from typing import Any

from sight.ascii_map import render_ascii
from sight.decorative import detect_circular, detect_vertical
from sight.tokens import build_design_tokens

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


def build_document(
    dump: dict[str, Any],
    image: Any,
    vlm: Any | None = None,
    image_path: str | None = None,
    lang: str = "ru",
) -> dict[str, Any]:
    width, height = dump["image"]["width"], dump["image"]["height"]
    elements = dump.get("elements", [])
    texts = [e for e in elements if e.get("kind") == "text"]
    graphics = [e for e in elements if e.get("kind") == "image"]
    colors = dump.get("colors", [])
    background = colors[0] if colors else None
    design = dump.get("design", {})
    facts = design.get("facts", design.get("issues", []))

    decorative = []
    for group in detect_circular(texts) + detect_vertical(texts):
        entry = dict(group)
        entry["box_norm"] = normalize_box(group["box"], width, height)
        if vlm is not None and image_path:
            entry["transcription"] = vlm.transcribe(image_path, group["box"])
        decorative.append(entry)

    graphic_docs = []
    for g in graphics:
        entry: dict[str, Any] = {"id": g["id"], "box_norm": normalize_box(g["box"], width, height)}
        if vlm is not None and image_path:
            entry["caption"] = vlm.describe(image_path, g["box"])
        graphic_docs.append(entry)

    return {
        "lang": lang,
        "header": {
            "size": [width, height],
            "theme": "light" if background and _is_light(background["hex"]) else "dark",
            "background": background["hex"] if background else None,
            "scene": (dump.get("scene") or [{}])[0].get("label"),
            "vibe": vlm.vibe(image_path) if vlm is not None and image_path else None,
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
        "measurements": [
            {"kind": f.get("kind"), "detail": f.get("detail")}
            for f in facts
            if isinstance(f, dict)
        ],
        "semantics_status": "ok" if vlm is not None else "unavailable",
    }


def render_markdown(doc: dict[str, Any]) -> str:
    lab = _LABELS.get(doc.get("lang", "ru"), _LABELS["ru"])
    h = doc["header"]
    lines = [
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
    lines.append(f"{lab['measurements']} {len(doc['measurements'])}")
    lines += [f" - {m['kind']}: {m['detail']}" for m in doc["measurements"]]
    lines.append(f"semantics: {doc['semantics_status']}")
    return "\n".join(lines)
