"""Reconstruct fixed-grid text separately from the luminance ASCII map."""

from __future__ import annotations

from statistics import median
from typing import Any


def _estimate_cell_width(items: list[dict[str, Any]]) -> float | None:
    metric_widths = [
        float(item["metrics"]["avgGlyphWidth"])
        for item in items
        if item.get("metrics", {}).get("avgGlyphWidth", 0) > 0
    ]
    if metric_widths:
        return float(median(metric_widths))
    widths = []
    for item in items:
        text = str(item.get("text", ""))
        if not text:
            continue
        x0, _, x1, _ = item["box"]
        if x1 > x0:
            widths.append((x1 - x0) / len(text))
    return float(median(widths)) if widths else None


def _estimate_line_height(items: list[dict[str, Any]]) -> float | None:
    heights = [
        float(item["box"][3] - item["box"][1])
        for item in items
        if item["box"][3] > item["box"][1]
    ]
    if not heights:
        return None
    centers = sorted((item["box"][1] + item["box"][3]) / 2 for item in items)
    pitches = [right - left for left, right in zip(centers, centers[1:]) if right - left > 1]
    return float(median(pitches)) if pitches else float(median(heights))


def reconstruct_monospace(
    items: list[dict[str, Any]],
    *,
    image_width: int,
    image_height: int,
    cell_width: float | None = None,
    line_height: float | None = None,
    confidence_threshold: float = 0.75,
) -> dict[str, Any]:
    """Place OCR spans on a fixed grid without collapsing whitespace.

    Low-confidence non-space characters become ``?`` and are listed in
    ``ambiguities``. The result is therefore a reconstruction candidate, not
    a claim that OCR directly observed every character.
    """
    cell_width = cell_width or _estimate_cell_width(items)
    line_height = line_height or _estimate_line_height(items)
    if not cell_width or not line_height or cell_width <= 0 or line_height <= 0:
        return {
            "status": "unavailable",
            "text": None,
            "grid": None,
            "confidence": None,
            "ambiguities": [],
            "method": "rapidocr-grid-reconstruction",
            "source": "inferred",
        }

    columns = max(1, round(image_width / cell_width))
    rows = max(1, round(image_height / line_height))
    grid = [[" " for _ in range(columns)] for _ in range(rows)]
    occupied = [[False for _ in range(columns)] for _ in range(rows)]
    ambiguities: list[dict[str, Any]] = []
    confidences: list[float] = []

    for item in sorted(items, key=lambda value: (value["box"][1], value["box"][0])):
        text = str(item.get("text", ""))
        if not text:
            continue
        x0, y0, _, y1 = item["box"]
        row = max(0, min(rows - 1, round(((y0 + y1) / 2) / line_height - 0.5)))
        start = max(0, round(x0 / cell_width))
        confidence = float(item.get("confidence", 0.0))
        confidences.append(confidence)
        ambiguous_chars = "".join(char for char in text if not char.isspace())
        if confidence < confidence_threshold and ambiguous_chars:
            ambiguities.append(
                {
                    "row": row,
                    "columns": [start, min(columns, start + len(ambiguous_chars))],
                    "observed": ambiguous_chars,
                }
            )
        for offset, char in enumerate(text):
            column = start + offset
            if column >= columns:
                break
            rendered = "?" if confidence < confidence_threshold and not char.isspace() else char
            if occupied[row][column] and grid[row][column] != rendered:
                grid[row][column] = "?"
                ambiguities.append(
                    {
                        "row": row,
                        "columns": [column, column + 1],
                        "observed": char,
                    }
                )
            else:
                grid[row][column] = rendered
                occupied[row][column] = True

    return {
        "status": "candidate" if not ambiguities else "ambiguous",
        "text": "\n".join("".join(row) for row in grid),
        "grid": {
            "columns": columns,
            "rows": rows,
            "cellWidth": round(float(cell_width), 2),
            "lineHeight": round(float(line_height), 2),
        },
        "confidence": round(min(confidences), 3) if confidences else None,
        "ambiguities": ambiguities,
        "method": "rapidocr-grid-reconstruction",
        "source": "inferred",
    }
