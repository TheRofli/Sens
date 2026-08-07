"""Decorative text detectors: circular and vertical OCR groups."""
from __future__ import annotations

import math
import statistics
from typing import Any


def _center(box: list[int]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _union(boxes: list[list[int]]) -> list[int]:
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def detect_vertical(
    items: list[dict[str, Any]], ratio: float = 2.0, x_tol: float = 24.0
) -> list[dict[str, Any]]:
    tall = [
        it
        for it in items
        if (it["box"][3] - it["box"][1]) > ratio * max(1, it["box"][2] - it["box"][0])
    ]
    groups: list[list[dict[str, Any]]] = []
    for it in sorted(tall, key=lambda i: i["box"][1]):
        cx, _ = _center(it["box"])
        for group in groups:
            gx, _ = _center(group[-1]["box"])
            if abs(cx - gx) <= x_tol:
                group.append(it)
                break
        else:
            groups.append([it])
    return [
        {"direction": "vertical", "ids": [it["id"] for it in g], "box": _union([it["box"] for it in g])}
        for g in groups
        if len(g) >= 2
    ]


def _cluster(items: list[dict[str, Any]], gap: float) -> list[list[dict[str, Any]]]:
    clusters: list[list[dict[str, Any]]] = []
    for it in items:
        cx, cy = _center(it["box"])
        for cl in clusters:
            for other in cl:
                ox, oy = _center(other["box"])
                if math.hypot(cx - ox, cy - oy) <= gap:
                    cl.append(it)
                    break
            else:
                continue
            break
        else:
            clusters.append([it])
    return clusters


def detect_circular(
    items: list[dict[str, Any]],
    min_boxes: int = 8,
    cv_max: float = 0.22,
    cover: float = 0.7,
) -> list[dict[str, Any]]:
    widths = [it["box"][2] - it["box"][0] for it in items]
    # 4x median width: adjacent glyph centers on a circle sit ~pi*r/n apart,
    # which for tight rings exceeds 3x glyph width (plan's test ring: 105 vs 90).
    gap = 4.0 * statistics.median(widths) if widths else 120.0
    out: list[dict[str, Any]] = []
    for cl in _cluster(items, gap):
        if len(cl) < min_boxes:
            continue
        cx = sum(_center(b["box"])[0] for b in cl) / len(cl)
        cy = sum(_center(b["box"])[1] for b in cl) / len(cl)
        radii = [math.hypot(_center(b["box"])[0] - cx, _center(b["box"])[1] - cy) for b in cl]
        r_mean = statistics.mean(radii)
        if r_mean <= 0 or statistics.stdev(radii) / r_mean >= cv_max:
            continue
        angles = [math.atan2(_center(b["box"])[1] - cy, _center(b["box"])[0] - cx) for b in cl]
        span = (max(angles) - min(angles)) / (2 * math.pi)
        if span < cover:
            continue
        ordered = sorted(
            cl,
            key=lambda b: math.atan2(_center(b["box"])[1] - cy, _center(b["box"])[0] - cx),
        )
        out.append(
            {
                "direction": "circular",
                "ids": [b["id"] for b in ordered],
                "box": _union([b["box"] for b in cl]),
                "center": [round(cx), round(cy)],
                "radius": round(r_mean),
            }
        )
    return out
