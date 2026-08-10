"""Deterministic reconstruction of repeated dot/diamond character artwork."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


def _cluster(values: list[float], tolerance: float) -> list[list[float]]:
    groups: list[list[float]] = []
    for value in sorted(values):
        if groups and value - float(np.mean(groups[-1])) <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return groups


def _longest_regular_rows(
    row_centers: list[float], pitch: float, tolerance: float
) -> list[float]:
    best: list[float] = []
    for start in row_centers:
        chain = [start]
        current = start
        while True:
            target = current + pitch
            candidates = [
                center
                for center in row_centers
                if center > current and abs(center - target) <= tolerance
            ]
            if not candidates:
                break
            current = min(candidates, key=lambda center: abs(center - target))
            chain.append(current)
        if len(chain) > len(best):
            best = chain
    return best


def _estimate_phase(values: list[float], pitch: float) -> float:
    candidates = np.linspace(0.0, pitch, 201, endpoint=False)

    def error(phase: float) -> float:
        distances = []
        for value in values:
            residual = (value - phase) % pitch
            distances.append(min(residual, pitch - residual))
        return float(np.median(distances)) if distances else pitch

    return float(min(candidates, key=error))


def _refine_periodic_pitch(
    values: list[float], initial_pitch: float, *, radius: float = 0.8
) -> tuple[float, float]:
    """Fit a fractional grid pitch and phase without rounding browser pixels."""
    samples = np.asarray(values, dtype=np.float64)
    if samples.size < 3:
        return initial_pitch, _estimate_phase(values, initial_pitch)
    best = (float("inf"), initial_pitch, 0.0)
    for pitch in np.linspace(
        max(1.0, initial_pitch - radius), initial_pitch + radius, 321
    ):
        angles = samples * (2.0 * np.pi / pitch)
        phase_angle = float(
            np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
        )
        if phase_angle < 0:
            phase_angle += 2.0 * np.pi
        phase = phase_angle * pitch / (2.0 * np.pi)
        residual = (samples - phase) % pitch
        error = float(np.median(np.minimum(residual, pitch - residual)))
        if error < best[0]:
            best = (error, float(pitch), phase)
    return best[1], best[2]


def _linear_grid(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        return (values[0] if values else 0.0), 1.0
    indices = np.arange(len(values), dtype=np.float64)
    pitch, origin = np.polyfit(indices, np.asarray(values, dtype=np.float64), 1)
    return float(origin), float(pitch)


def detect_symbol_art(image: Any, *, min_rows: int = 6) -> list[dict[str, Any]]:
    """Recover artwork laid out on a repeated ``.``/``◆`` text grid.

    The detector intentionally requires many tiny glyphs on a regular baseline
    sequence.  Ordinary headings and isolated icon rows therefore stay out of
    this representation contract.
    """
    import cv2

    height, width = image.shape[:2]
    pixels = image.astype(np.float32)
    border = np.concatenate(
        (pixels[0, :, :], pixels[-1, :, :], pixels[:, 0, :], pixels[:, -1, :]),
        axis=0,
    )
    background = np.median(border, axis=0)
    foreground = (
        np.linalg.norm(pixels - background, axis=2) > 48.0
    ).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        foreground, 8
    )
    dots: list[dict[str, float]] = []
    for component in range(1, count):
        x, y, component_width, component_height, area = (
            int(value) for value in stats[component]
        )
        if not (
            4 <= area <= 30
            and 2 <= component_width <= 7
            and 2 <= component_height <= 7
        ):
            continue
        dots.append(
            {
                "x": float(centroids[component][0]),
                "y": float(centroids[component][1]),
                "area": float(area),
            }
        )
    if len(dots) < min_rows * 3:
        return []

    y_groups = _cluster([dot["y"] for dot in dots], tolerance=2.2)
    row_centers = [float(np.mean(group)) for group in y_groups if len(group) >= 2]
    if len(row_centers) < min_rows:
        return []
    pitch_votes: Counter[int] = Counter()
    for first, second in zip(row_centers, row_centers[1:]):
        difference = second - first
        if 10 <= difference <= 48:
            pitch_votes[round(difference)] += 1
    if not pitch_votes:
        return []
    row_pitch = float(pitch_votes.most_common(1)[0][0])
    regular_rows = _longest_regular_rows(
        row_centers, row_pitch, tolerance=max(2.5, row_pitch * 0.14)
    )
    if len(regular_rows) < min_rows:
        return []
    first_baseline, row_pitch = _linear_grid(regular_rows)
    regular_rows = [
        first_baseline + index * row_pitch for index in range(len(regular_rows))
    ]

    row_dots = [
        dot
        for dot in dots
        if min(abs(dot["y"] - center) for center in regular_rows) <= 2.5
    ]
    horizontal_votes: Counter[int] = Counter()
    for center in regular_rows:
        xs = sorted(
            dot["x"] for dot in row_dots if abs(dot["y"] - center) <= 2.5
        )
        for first, second in zip(xs, xs[1:]):
            difference = second - first
            if 7 <= difference <= 26:
                horizontal_votes[round(difference)] += 1
    if not horizontal_votes:
        return []
    cell_width = float(horizontal_votes.most_common(1)[0][0])
    if cell_width < 7:
        return []
    cell_width, phase = _refine_periodic_pitch(
        [dot["x"] for dot in row_dots], cell_width
    )
    dot_area = float(np.median([dot["area"] for dot in row_dots]))
    minimum_ink = max(2.0, dot_area * 0.35)
    diamond_ink = max(22.0, dot_area * 2.2)

    # Retain only components whose vertical placement matches one of the two
    # observed glyph forms: a baseline dot, or a centered diamond above it.
    # This removes ordinary navigation/button letters that happen to sit near
    # the first art row.
    glyph_foreground = np.zeros_like(foreground)
    for component in range(1, count):
        component_width = int(stats[component][cv2.CC_STAT_WIDTH])
        component_height = int(stats[component][cv2.CC_STAT_HEIGHT])
        area = int(stats[component][cv2.CC_STAT_AREA])
        center_y = float(centroids[component][1])
        baseline = min(regular_rows, key=lambda center: abs(center_y - center))
        offset = center_y - baseline
        dot_shape = (
            4 <= area <= 30
            and 2 <= component_width <= 7
            and 2 <= component_height <= 7
            and abs(offset) <= 2.5
        )
        diamond_shape = (
            area >= 20
            and row_pitch * 0.28 <= component_height <= row_pitch * 0.72
            and -row_pitch * 0.42 <= offset <= -row_pitch * 0.10
        )
        if dot_shape or diamond_shape:
            glyph_foreground[labels == component] = 1

    minimum_column = int(np.floor((0.0 - phase) / cell_width))
    maximum_column = int(np.ceil((width - phase) / cell_width))
    rows: list[list[str]] = []
    occupied_columns: list[int] = []
    glyph_samples: dict[str, list[dict[str, float]]] = {"dot": [], "diamond": []}
    for baseline in regular_rows:
        top = max(0, round(baseline - row_pitch * 0.7))
        bottom = min(height, round(baseline + row_pitch * 0.2) + 1)
        characters: list[str] = []
        for column in range(minimum_column, maximum_column + 1):
            center_x = phase + column * cell_width
            left = max(
                0, min(width, round(center_x - cell_width * 0.48))
            )
            right = max(
                0, min(width, round(center_x + cell_width * 0.48) + 1)
            )
            if right <= left:
                characters.append(" ")
                continue
            cell_mask = glyph_foreground[top:bottom, left:right]
            ink = float(np.count_nonzero(cell_mask))
            if ink < minimum_ink:
                character = " "
            elif ink < diamond_ink:
                character = "."
            else:
                character = "◆"
            characters.append(character)
            if character != " ":
                occupied_columns.append(column)
                ys, xs = np.where(cell_mask > 0)
                if xs.size:
                    glyph_samples[
                        "dot" if character == "." else "diamond"
                    ].append(
                        {
                            "width": float(xs.max() - xs.min() + 1),
                            "height": float(ys.max() - ys.min() + 1),
                            "centerOffsetX": float(
                                left + (xs.min() + xs.max()) / 2.0 - center_x
                            ),
                            "centerOffsetY": float(
                                top + (ys.min() + ys.max()) / 2.0 - baseline
                            ),
                        }
                    )
        rows.append(characters)
    if not occupied_columns:
        return []
    first_column = min(occupied_columns)
    last_column = max(occupied_columns)
    offset = first_column - minimum_column
    span = last_column - first_column + 1
    lines = ["".join(row[offset : offset + span]).rstrip() for row in rows]
    text = "\n".join(lines)
    non_space = sum(character != " " for character in text if character != "\n")
    if non_space < min_rows * 3 or "◆" not in text or "." not in text:
        return []
    left = max(0, round(phase + (first_column - 0.5) * cell_width))
    right = min(width, round(phase + (last_column + 0.5) * cell_width))
    top = max(0, round(regular_rows[0] - row_pitch * 0.7))
    bottom = min(height, round(regular_rows[-1] + row_pitch * 0.2) + 1)
    regularity = 1.0 - min(
        1.0,
        float(
            np.mean(
                [
                    abs((regular_rows[index] - regular_rows[index - 1]) - row_pitch)
                    for index in range(1, len(regular_rows))
                ]
            )
        )
        / max(1.0, row_pitch * 0.2),
    )
    def median_geometry(kind: str) -> dict[str, float]:
        samples = glyph_samples[kind]
        return {
            key: round(float(np.median([sample[key] for sample in samples])), 2)
            if samples
            else 0.0
            for key in ("width", "height", "centerOffsetX", "centerOffsetY")
        }

    first_cell_center = phase + first_column * cell_width
    return [
        {
            "kind": "symbol-art",
            "box": [left, top, right, bottom],
            "text": text,
            "rows": len(lines),
            "columns": span,
            "cellWidth": round(cell_width, 2),
            "rowPitch": round(row_pitch, 2),
            "firstCellCenterX": round(first_cell_center, 2),
            "firstBaselineY": round(first_baseline, 2),
            "glyphGeometry": {
                "dot": median_geometry("dot"),
                "diamond": median_geometry("diamond"),
            },
            "alphabet": [".", "◆"],
            "foregroundColor": "#FFFFFF",
            "backgroundColor": "#{:02X}{:02X}{:02X}".format(
                int(round(background[2])),
                int(round(background[1])),
                int(round(background[0])),
            ),
            "confidence": round(max(0.0, min(1.0, regularity)), 3),
            "strategy": "render-as-live-selectable-monospace-text",
            "rule": "Preserve every character and space in a preformatted live-text element; never use a screenshot, canvas, SVG path, or raster asset.",
            "source": "measured",
            "method": "regular-dot-diamond-grid",
        }
    ]
