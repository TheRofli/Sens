"""Deterministic selection of regions that deserve a higher-resolution pass."""

from __future__ import annotations

import re
from typing import Any

from sight.coordinates import box_to_source, identity_coordinates


def _tokens(text: str | None) -> set[str]:
    return {token for token in re.findall(r"[\w.-]+", (text or "").lower()) if len(token) > 2}


def _padded_box(box: list[int], width: int, height: int) -> list[int]:
    x0, y0, x1, y1 = box
    pad_x = max(16, round((x1 - x0) * 0.5))
    pad_y = max(16, round((y1 - y0) * 2.0))
    return [
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(width, x1 + pad_x),
        min(height, y1 + pad_y),
    ]


def _overlap(left: list[int], right: list[int]) -> float:
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    intersection = (x1 - x0) * (y1 - y0)
    smaller = min(
        (left[2] - left[0]) * (left[3] - left[1]),
        (right[2] - right[0]) * (right[3] - right[1]),
    )
    return intersection / smaller if smaller else 0.0


def recommend_focus(
    dump: dict[str, Any],
    *,
    max_regions: int = 4,
    intent: str | None = None,
) -> list[dict[str, Any]]:
    """Return bounded, source-pixel zoom calls for ambiguous or tiny content."""
    width = int(dump["image"]["width"])
    height = int(dump["image"]["height"])
    coordinates = dump.get("coordinates") or identity_coordinates(width, height)
    intent_tokens = _tokens(intent)
    candidates: list[tuple[float, list[int], list[str], str]] = []

    for item in dump.get("ocr", []):
        box = [int(value) for value in item["box"]]
        box_height = max(0, box[3] - box[1])
        confidence = float(item.get("confidence", 0.0))
        reasons = []
        score = 0.0
        if confidence < 0.8:
            reasons.append("low_ocr_confidence")
            score += 1.0 - confidence
        if box_height < 18:
            reasons.append("small_text")
            score += min(0.5, (18 - box_height) / 18)
        if intent_tokens & _tokens(str(item.get("text", ""))):
            reasons.append("intent_match")
            score += 0.75
        if reasons:
            candidates.append(
                (
                    score,
                    _padded_box(box, width, height),
                    reasons,
                    str(item.get("text", "")),
                )
            )

    # Attention regions encode measured text/contrast density even when OCR
    # misses a very large display line. If recognized glyph boxes explain only
    # a minority of such a bounded region, spend one high-resolution semantic
    # call on discovery instead of assuming the missing ink is decoration.
    for attention in dump.get("attention", []):
        why = str(attention.get("why") or "").casefold()
        raw_box = attention.get("box") or []
        if "text" not in why or len(raw_box) != 4:
            continue
        box = [int(round(value)) for value in raw_box]
        box = [
            max(0, min(width, box[0])),
            max(0, min(height, box[1])),
            max(0, min(width, box[2])),
            max(0, min(height, box[3])),
        ]
        pad_x = max(16, round((box[2] - box[0]) * 0.10))
        pad_y = max(16, round((box[3] - box[1]) * 0.08))
        box = [
            max(0, box[0] - pad_x),
            max(0, box[1] - pad_y),
            min(width, box[2] + pad_x),
            min(height, box[3] + pad_y),
        ]
        # Attention tiles often stop midway through a thin navigation bar.
        # Extend that bounded crop to the right edge so an unread CTA at the
        # opposite end of the same row is not silently left in raster artwork.
        if box[1] <= 2 and box[3] - box[1] <= height * 0.16:
            box[2] = width
        area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
        area_ratio = area / max(1, width * height)
        if area <= 0 or not 0.02 <= area_ratio <= 0.50:
            continue
        covered = 0
        recognized_regions = 0
        for item in dump.get("ocr", []):
            other = item.get("box") or []
            if len(other) != 4:
                continue
            x0, y0 = max(box[0], other[0]), max(box[1], other[1])
            x1, y1 = min(box[2], other[2]), min(box[3], other[3])
            if x1 > x0 and y1 > y0:
                covered += (x1 - x0) * (y1 - y0)
                recognized_regions += 1
        coverage = min(1.0, covered / area)
        attention_score = float(attention.get("score") or 0.0)
        if attention_score < 0.32 or coverage >= 0.45:
            continue
        candidates.append(
            (
                1.25
                + attention_score
                + (0.45 - coverage)
                - 0.15 * min(4, recognized_regions),
                box,
                ["unresolved_text_density"],
                "unresolved visible text",
            )
        )

    selected: list[dict[str, Any]] = []
    selected_analysis_boxes: list[list[int]] = []
    for score, analysis_box, reasons, text in sorted(
        candidates, key=lambda candidate: (-candidate[0], candidate[1][1], candidate[1][0])
    ):
        if any(_overlap(analysis_box, prior) > 0.65 for prior in selected_analysis_boxes):
            continue
        source_box = box_to_source(analysis_box, coordinates)
        selected_analysis_boxes.append(analysis_box)
        selected.append(
            {
                "id": f"focus-{len(selected) + 1}",
                "tool": "sens_zoom",
                "priority": round(score, 3),
                "reasons": reasons,
                "evidence": text,
                "region": {
                    "x": source_box[0],
                    "y": source_box[1],
                    "width": source_box[2] - source_box[0],
                    "height": source_box[3] - source_box[1],
                },
            }
        )
        if len(selected) >= max_regions:
            break
    return selected
