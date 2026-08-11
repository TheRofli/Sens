"""Deterministic perception layers L0-L4 (colors, layout, objects, scene, texture)."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from sight.ocr import WORKER_DIR


_DIGIT_FONT_PATH = Path(__file__).resolve().parent / "assets" / "fonts" / "InterTight.ttf"


def _normalized_digit_mask(mask: Any) -> Any | None:
    """Center one glyph in a size-invariant binary comparison canvas."""
    import cv2

    rows, columns = np.where(np.asarray(mask, dtype=np.uint8) > 0)
    if not len(rows) or not len(columns):
        return None
    cropped = np.asarray(mask, dtype=np.uint8)[
        int(rows.min()) : int(rows.max()) + 1,
        int(columns.min()) : int(columns.max()) + 1,
    ]
    height, width = cropped.shape[:2]
    scale = min(24 / max(1, width), 24 / max(1, height))
    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))
    resized = cv2.resize(
        cropped,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )
    resized = (resized > 0.25).astype(np.uint8)
    result = np.zeros((32, 32), np.uint8)
    x = (32 - target_width) // 2
    y = (32 - target_height) // 2
    result[y : y + target_height, x : x + target_width] = resized
    return result


@lru_cache(maxsize=1)
def _digit_templates() -> tuple[tuple[str, Any], ...]:
    """Render an offline digit atlas from bundled and local system fonts."""
    from PIL import Image, ImageDraw, ImageFont

    if not _DIGIT_FONT_PATH.is_file():
        return ()
    templates: list[tuple[str, Any]] = []
    for size in range(8, 20):
        for variation in (b"Regular", b"Medium", b"SemiBold", b"Bold", b"ExtraBold"):
            font = ImageFont.truetype(str(_DIGIT_FONT_PATH), size)
            try:
                font.set_variation_by_name(variation)
            except (AttributeError, OSError, ValueError):
                pass
            for digit in "0123456789":
                canvas = Image.new("L", (50, 50))
                ImageDraw.Draw(canvas).text((5, 2), digit, font=font, fill=255)
                normalized = _normalized_digit_mask(np.asarray(canvas) > 80)
                if normalized is not None:
                    templates.append((digit, normalized))
    windows_fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    for filename in ("arial.ttf", "arialbd.ttf", "segoeui.ttf", "segoeuib.ttf"):
        font_path = windows_fonts / filename
        if not font_path.is_file():
            continue
        for size in range(8, 20):
            font = ImageFont.truetype(str(font_path), size)
            for stroke_width in (0, 1):
                for digit in "0123456789":
                    canvas = Image.new("L", (50, 50))
                    ImageDraw.Draw(canvas).text(
                        (5, 2),
                        digit,
                        font=font,
                        fill=255,
                        stroke_width=stroke_width,
                        stroke_fill=255,
                    )
                    normalized = _normalized_digit_mask(np.asarray(canvas) > 80)
                    if normalized is not None:
                        templates.append((digit, normalized))
    return tuple(templates)


def _digit_mask_distance(left: Any, right: Any) -> float:
    import cv2

    left = np.asarray(left, dtype=np.uint8)
    right = np.asarray(right, dtype=np.uint8)
    left_distance = cv2.distanceTransform(1 - left, cv2.DIST_L2, 3)
    right_distance = cv2.distanceTransform(1 - right, cv2.DIST_L2, 3)
    return float(
        (
            float(left_distance[right > 0].mean())
            + float(right_distance[left > 0].mean())
        )
        / 2
    )


def _classify_digit_mask(mask: Any) -> list[tuple[float, str]]:
    normalized = _normalized_digit_mask(mask)
    if normalized is None:
        return []
    scores: dict[str, float] = {}
    for digit, template in _digit_templates():
        distance = _digit_mask_distance(normalized, template)
        scores[digit] = min(scores.get(digit, float("inf")), distance)
    return sorted((score, digit) for digit, score in scores.items())


def compact_numeric_badges(
    image: Any,
    *,
    max_badges: int = 32,
) -> list[dict[str, Any]]:
    """Recover tiny numeric counts from pale tinted UI pills.

    General OCR commonly drops 8-12 px counter text. Geometry and colors are
    measured directly; the character value is explicitly marked inferred and
    classified against a bundled offline digit atlas. Ambiguous glyphs remain
    unverified so the bounded local VLM focus pass can adjudicate them.
    """
    import cv2

    if image is None or getattr(image, "ndim", 0) != 3:
        return []
    height, width = image.shape[:2]
    pixels = image.astype(np.int16)
    channel_range = pixels.max(axis=2) - pixels.min(axis=2)
    bright_tint = (
        (channel_range >= 12)
        & (pixels.max(axis=2) >= 170)
        & (pixels.min(axis=2) >= 120)
    )
    connected = cv2.morphologyEx(
        bright_tint.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
    )
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        connected, 8
    )
    badges: list[dict[str, Any]] = []
    max_width = max(48, min(96, int(round(width * 0.09))))
    max_height = max(40, min(64, int(round(height * 0.10))))

    for component in range(1, count):
        x, y, badge_width, badge_height, area = (
            int(value) for value in stats[component]
        )
        fill_ratio = area / max(1, badge_width * badge_height)
        aspect = badge_width / max(1, badge_height)
        if (
            badge_width < 10
            or badge_height < 10
            or badge_width > max_width
            or badge_height > max_height
            or not 0.65 <= aspect <= 4.2
            or fill_ratio < 0.42
        ):
            continue

        component_mask = labels[y : y + badge_height, x : x + badge_width] == component
        roi = pixels[y : y + badge_height, x : x + badge_width]
        fill_bgr = np.median(roi[component_mask], axis=0)
        pad = 3
        outer_x0, outer_y0 = max(0, x - pad), max(0, y - pad)
        outer_x1 = min(width, x + badge_width + pad)
        outer_y1 = min(height, y + badge_height + pad)
        outer = pixels[outer_y0:outer_y1, outer_x0:outer_x1]
        outer_mask = np.ones(outer.shape[:2], dtype=bool)
        outer_mask[
            y - outer_y0 : y - outer_y0 + badge_height,
            x - outer_x0 : x - outer_x0 + badge_width,
        ] = False
        if outer_mask.any():
            outside_bgr = np.median(outer[outer_mask], axis=0)
            if float(np.linalg.norm(fill_bgr - outside_bgr)) < 9.0:
                continue

        color_distance = np.linalg.norm(roi - fill_bgr, axis=2)
        luminance = (
            roi[:, :, 2] * 0.2126
            + roi[:, :, 1] * 0.7152
            + roi[:, :, 0] * 0.0722
        )
        fill_luminance = (
            fill_bgr[2] * 0.2126
            + fill_bgr[1] * 0.7152
            + fill_bgr[0] * 0.0722
        )
        ink = (color_distance >= 30.0) & (luminance <= fill_luminance - 20.0)
        ink_rows, ink_columns = np.where(ink)
        if not len(ink_rows) or not len(ink_columns):
            continue
        ink_x0, ink_x1 = int(ink_columns.min()), int(ink_columns.max()) + 1
        ink_y0, ink_y1 = int(ink_rows.min()), int(ink_rows.max()) + 1
        if (
            ink_x1 - ink_x0 > badge_width * 0.75
            or ink_y1 - ink_y0 > badge_height * 0.75
        ):
            continue

        column_sums = ink.sum(axis=0)
        active_columns = np.where(column_sums > 0)[0]
        segments: list[tuple[int, int]] = []
        start = previous = int(active_columns[0])
        for column in active_columns[1:]:
            column = int(column)
            if column > previous + 1:
                segments.append((start, previous + 1))
                start = column
            previous = column
        segments.append((start, previous + 1))
        # Antialiasing can leave a single bridge pixel between adjacent
        # digits. Split a disproportionately wide run at a deep internal
        # projection valley instead of treating the pair as one glyph.
        split_segments: list[tuple[int, int]] = []
        glyph_height = max(1, ink_y1 - ink_y0)
        for start, end in segments:
            if end - start <= glyph_height * 0.9:
                split_segments.append((start, end))
                continue
            interior = column_sums[start + 2 : end - 2]
            if interior.size == 0:
                split_segments.append((start, end))
                continue
            relative = int(np.argmin(interior))
            split_at = start + 2 + relative
            if column_sums[split_at] <= max(1, int(column_sums[start:end].max() * 0.18)):
                split_segments.extend([(start, split_at), (split_at + 1, end)])
            else:
                split_segments.append((start, end))
        segments = split_segments
        if not 1 <= len(segments) <= 3:
            continue

        value_parts: list[str] = []
        glyph_rankings: list[list[tuple[float, str]]] = []
        verified = True
        confidence_parts: list[float] = []
        for start, end in segments:
            ranking = _classify_digit_mask(ink[:, start:end])
            if len(ranking) < 2 or ranking[0][0] > 0.36:
                value_parts = []
                break
            best_score, digit = ranking[0]
            margin = ranking[1][0] - best_score
            value_parts.append(digit)
            glyph_rankings.append(ranking[:3])
            verified = verified and best_score <= 0.28 and margin >= 0.05
            shape_confidence = max(0.0, min(1.0, 1.0 - best_score / 0.45))
            margin_confidence = max(0.0, min(1.0, margin / 0.12))
            confidence_parts.append(
                0.55 * shape_confidence + 0.45 * margin_confidence
            )
        if not value_parts:
            continue

        alternatives = [
            {
                "text": "".join(
                    ranking[min(index, len(ranking) - 1)][1]
                    if glyph_index == alternate_glyph
                    else ranking[0][1]
                    for glyph_index, ranking in enumerate(glyph_rankings)
                ),
                "score": round(
                    glyph_rankings[alternate_glyph][index][0], 3
                ),
            }
            for alternate_glyph in range(len(glyph_rankings))
            for index in range(1, min(3, len(glyph_rankings[alternate_glyph])))
        ]
        ink_distances = color_distance[ink]
        core_cutoff = float(np.percentile(ink_distances, 65))
        core_ink = ink & (color_distance >= core_cutoff)
        foreground_bgr = np.median(
            roi[core_ink] if core_ink.any() else roi[ink], axis=0
        )

        def color_hex(bgr: Any) -> str:
            blue, green, red = (int(round(value)) for value in bgr)
            return f"#{red:02X}{green:02X}{blue:02X}"

        badges.append(
            {
                "box": [x, y, x + badge_width, y + badge_height],
                "textBox": [
                    x + ink_x0,
                    y + ink_y0,
                    x + ink_x1,
                    y + ink_y1,
                ],
                "value": "".join(value_parts),
                "confidence": round(
                    max(0.5, min(0.99, min(confidence_parts))), 3
                ),
                "verified": verified,
                "alternatives": alternatives,
                "background": color_hex(fill_bgr),
                "foreground": color_hex(foreground_bgr),
                "cornerRadius": round(min(badge_width, badge_height) / 3, 1),
                "method": "compact-tinted-badge-local-font-atlas",
                "geometrySource": "measured",
                "epistemic": "inferred",
                "representation": "live-text-on-css-surface",
            }
        )

    badges.sort(key=lambda badge: (badge["box"][1], badge["box"][0]))
    return badges[: max(0, int(max_badges))]


def detect_dashed_structural_lines(
    image: Any,
    excluded_boxes: list[list[int]] | None = None,
    *,
    max_lines: int = 24,
) -> list[dict[str, Any]]:
    """Measure long fragmented rules such as dashboard chart grids."""
    import cv2

    if image is None or getattr(image, "ndim", 0) != 3:
        return []
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 5, 15)
    hough = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=20,
        minLineLength=max(50, int(round(width * 0.06))),
        maxLineGap=12,
    )
    if hough is None:
        return []
    raw: list[list[int]] = []
    for x0, y0, x1, y1 in hough.reshape(-1, 4):
        x0, y0, x1, y1 = (int(value) for value in (x0, y0, x1, y1))
        if x1 < x0:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        if abs(y1 - y0) > 2 or x1 - x0 < width * 0.12:
            continue
        raw.append([x0, int(round((y0 + y1) / 2)), x1])
    raw.sort(key=lambda item: (item[1], item[0]))
    clusters: list[list[list[int]]] = []
    for line in raw:
        destination = None
        for cluster in reversed(clusters):
            cluster_y = float(np.median([item[1] for item in cluster]))
            if line[1] - cluster_y > 3:
                break
            left = min(item[0] for item in cluster)
            right = max(item[2] for item in cluster)
            if abs(line[1] - cluster_y) <= 3 and line[0] <= right + 15 and line[2] >= left - 15:
                destination = cluster
                break
        if destination is None:
            clusters.append([line])
        else:
            destination.append(line)

    excluded = excluded_boxes or []
    output: list[dict[str, Any]] = []
    for cluster in clusters:
        y = int(round(float(np.median([item[1] for item in cluster]))))
        x0 = min(item[0] for item in cluster)
        x1 = max(item[2] for item in cluster)
        box = [x0, max(0, y), x1 + 1, min(height, y + 1)]
        line_width = max(1, x1 - x0 + 1)
        if any(
            max(0, min(x1, item[2]) - max(x0, item[0])) / line_width >= 0.16
            and item[1] <= y <= item[3]
            for item in excluded
            if len(item) == 4
        ):
            continue
        signal_band = gray[
            max(0, y - 1) : min(height, y + 2), x0 : x1 + 1
        ]
        above = gray[max(0, y - 5) : max(0, y - 2), x0 : x1 + 1]
        below = gray[min(height, y + 3) : min(height, y + 6), x0 : x1 + 1]
        if above.size == 0 or below.size == 0:
            continue
        background = (
            np.median(above, axis=0) + np.median(below, axis=0)
        ) / 2
        deviations = np.abs(signal_band.astype(np.float32) - background[None, :])
        strongest_row = np.argmax(deviations, axis=0)
        strongest_deviation = deviations[
            strongest_row, np.arange(deviations.shape[1])
        ]
        active = strongest_deviation > 4
        runs: list[tuple[int, int]] = []
        start: int | None = None
        for index, value in enumerate(active):
            if value and start is None:
                start = index
            elif not value and start is not None:
                runs.append((start, index))
                start = None
        if start is not None:
            runs.append((start, len(active)))
        if len(runs) < 8:
            continue
        run_lengths = [end - start for start, end in runs]
        if float(np.mean(run_lengths)) > 30 or not 0.12 <= float(active.mean()) <= 0.85:
            continue
        gaps = [
            runs[index + 1][0] - runs[index][1]
            for index in range(len(runs) - 1)
            if runs[index + 1][0] > runs[index][1]
        ]
        band = image[max(0, y - 1) : min(height, y + 2), x0 : x1 + 1]
        active_columns = np.flatnonzero(active)
        active_pixels = band[strongest_row[active], active_columns]
        neutral = active_pixels[
            active_pixels.max(axis=1) - active_pixels.min(axis=1) <= 18
        ]
        color_bgr = np.median(
            neutral if len(neutral) else active_pixels, axis=0
        )
        blue, green, red = (int(round(value)) for value in color_bgr)
        output.append(
            {
                "orientation": "horizontal",
                "boxSource": box,
                "thickness": 1,
                "length": line_width,
                "color": f"#{red:02X}{green:02X}{blue:02X}",
                "lineStyle": "dashed",
                "dashLength": max(2, int(round(float(np.median(run_lengths))))),
                "dashGap": max(2, int(round(float(np.median(gaps)))) if gaps else 4),
                "source": "measured",
                "method": "low-contrast-fragmented-hough-rule",
            }
        )
    for line in output:
        box = line["boxSource"]
        line_width = max(1, box[2] - box[0])
        peers = [
            candidate
            for candidate in output
            if abs(candidate["dashLength"] - line["dashLength"]) <= 2
            and abs(candidate["dashGap"] - line["dashGap"]) <= 2
            and abs(candidate["boxSource"][1] - box[1])
            <= max(120, int(round(height * 0.35)))
            and max(
                0,
                min(candidate["boxSource"][2], box[2])
                - max(candidate["boxSource"][0], box[0]),
            )
            / min(
                line_width,
                max(1, candidate["boxSource"][2] - candidate["boxSource"][0]),
            )
            >= 0.4
        ]
        if len(peers) < 4:
            continue
        peer_widths = [
            candidate["boxSource"][2] - candidate["boxSource"][0]
            for candidate in peers
        ]
        median_width = float(np.median(peer_widths))
        anchors = [
            candidate
            for candidate in peers
            if candidate["boxSource"][2] - candidate["boxSource"][0]
            >= median_width * 0.8
        ]
        family_left = int(round(float(np.median([item["boxSource"][0] for item in anchors]))))
        family_right = int(round(float(np.median([item["boxSource"][2] for item in anchors]))))
        if line_width < median_width * 0.75:
            line["boxSource"][0] = min(box[0], family_left)
            line["boxSource"][2] = max(box[2], family_right)
            line["length"] = line["boxSource"][2] - line["boxSource"][0]
            line["method"] += "+measured-grid-family"
    output.sort(key=lambda item: (item["boxSource"][1], item["boxSource"][0]))
    return output[: max(0, int(max_lines))]




def color_zones(image: Any, k: int = 5, sample_side: int = 96) -> list[dict[str, Any]]:
    """Dominant palette + spatial ratio via color k-means on a thumbnail."""
    import cv2

    height, width = image.shape[:2]
    thumb = cv2.resize(image, (sample_side, sample_side), interpolation=cv2.INTER_AREA)
    pixels = thumb.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)
    total = float(counts.sum()) or 1.0
    zones = []
    for index in np.argsort(counts)[::-1]:
        if counts[index] == 0:
            continue
        b, g, r = (int(round(c)) for c in centers[index])
        zones.append({
            "hex": f"#{r:02X}{g:02X}{b:02X}",
            "ratio": round(float(counts[index]) / total, 4),
            "source": "measured",
        })
    strip = max(1, min(height, width) // 100)
    border_pixels = np.concatenate(
        [
            image[:strip, :, :].reshape(-1, 3),
            image[-strip:, :, :].reshape(-1, 3),
            image[strip:-strip or None, :strip, :].reshape(-1, 3),
            image[strip:-strip or None, -strip:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    unique, exact_counts = np.unique(border_pixels, axis=0, return_counts=True)
    canvas_bgr = unique[int(np.argmax(exact_counts))]
    canvas_hex = "#{:02X}{:02X}{:02X}".format(
        int(canvas_bgr[2]), int(canvas_bgr[1]), int(canvas_bgr[0])
    )
    if not zones or zones[0]["hex"] != canvas_hex:
        zones.insert(
            0,
            {
                "hex": canvas_hex,
                "ratio": round(float(exact_counts.max()) / max(1, len(border_pixels)), 4),
                "source": "measured",
                "method": "exact-border-mode",
                "role": "canvas-background",
            },
        )
    else:
        zones[0] = {
            **zones[0],
            "method": "exact-border-mode",
            "role": "canvas-background",
        }
    return {
        "width": width,
        "height": height,
        "dominant": zones[:8],
        "canvasBackground": {
            "hex": canvas_hex,
            "source": "measured",
            "method": "exact-border-mode",
        },
    }




def layout_blocks(image: Any, min_side: int = 24) -> list[dict[str, Any]]:
    """Group text/UI into rectangular blocks via morphology + contours.

    RETR_TREE + an enclosure filter so standalone sections (cards) are
    reported separately instead of being merged into one outer shell.
    """
    import cv2

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Strong gradients -> edges; dilate to fuse nearby strokes into blocks.
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    _, binary = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 7))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    raw = []
    for index, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w < min_side or h < min_side:
            continue
        if w * h > width * height * 0.9:
            continue
        raw.append((index, [x, y, x + w, y + h]))
    # Enclosure filter via the contour tree: a rect that has a child
    # covering > 20% of its area is an outer shell, not a section — unless
    # the shell itself is thick (e.g. a photo whose internal bright zone
    # happens to exceed 20% of the frame).
    kept = []
    for index, rect in raw:
        x0, y0, x1, y1 = rect
        own = (x1 - x0) * (y1 - y0)
        shell = False
        if hierarchy is not None:
            child_index = hierarchy[0][index][2]
            while child_index != -1:
                child = next((r for i, r in raw if i == child_index), None)
                if child is not None:
                    child_area = (child[2] - child[0]) * (child[3] - child[1])
                    if child_area > 0.2 * own:
                        # Thin shells (border-like) are layout containers;
                        # thick ones (content with a large internal zone)
                        # are real sections. Photos (high internal texture)
                        # are never shells.
                        interior = gray[y0 + 6:y1 - 6, x0 + 6:x1 - 6]
                        textured = interior.size > 0 and float(interior.std()) > 45
                        if not textured and (own - child_area) < 0.25 * own:
                            shell = True
                        break
                child_index = hierarchy[0][child_index][0]
        if not shell:
            kept.append(rect)
    blocks = []
    for x0, y0, x1, y1 in kept:
        blocks.append({
            "kind": "block",
            "box": [x0, y0, x1, y1],
            "area": (x1 - x0) * (y1 - y0),
            "source": "measured",
        })
    blocks.sort(key=lambda b: -b["area"])
    return blocks[:40]




def layout_skeleton(image: Any, min_len_ratio: float = 0.35) -> dict[str, Any]:
    """Long horizontal/vertical lines: the frame skeleton of the layout.

    Returns pixel coordinates of lines that span at least `min_len_ratio`
    of the frame along their axis — these are section borders (cards,
    panels), as opposed to short field separators.
    """
    import cv2

    height, width = image.shape[:2]
    pixels = image.astype(np.float32)
    border = np.concatenate(
        (pixels[0, :, :], pixels[-1, :, :], pixels[:, 0, :], pixels[:, -1, :]),
        axis=0,
    )
    background = np.median(border, axis=0)
    binary = (
        np.linalg.norm(pixels - background, axis=2) > 18.0
    ).astype(np.uint8) * 255
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, int(width * min_len_ratio)), 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, int(height * min_len_ratio))))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    horizontal = [
        y
        for y in range(height)
        if int(h_lines[y, :].sum()) > 255 * width * min_len_ratio
    ]
    vertical = [
        x
        for x in range(width)
        if int(v_lines[:, x].sum()) > 255 * height * min_len_ratio
    ]
    segments = _measured_line_segments(
        image,
        h_lines,
        orientation="horizontal",
        min_length=max(15, int(width * min_len_ratio)),
    ) + _measured_line_segments(
        image,
        v_lines,
        orientation="vertical",
        min_length=max(15, int(height * min_len_ratio)),
    )
    segments.sort(
        key=lambda segment: (
            segment["box"][1],
            segment["box"][0],
            segment["orientation"],
        )
    )
    return {
        "horizontal": _line_groups(horizontal),
        "vertical": _line_groups(vertical),
        "segments": segments,
    }




def _line_groups(coords: list[int], tolerance: int = 3) -> list[int]:
    """Collapse thick lines (several adjacent rows/cols) to center rows."""
    groups: list[list[int]] = []
    for coord in coords:
        if groups and coord - groups[-1][-1] <= tolerance:
            groups[-1].append(coord)
        else:
            groups.append([coord])
    return [round(float(np.mean(group))) for group in groups]


def _measured_line_segments(
    image: Any,
    mask: Any,
    *,
    orientation: str,
    min_length: int,
) -> list[dict[str, Any]]:
    """Return source-pixel extents and style for long morphology components."""
    import cv2

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    segments: list[dict[str, Any]] = []
    for component in range(1, count):
        x, y, width, height, _area = (int(value) for value in stats[component])
        length = width if orientation == "horizontal" else height
        thickness = height if orientation == "horizontal" else width
        cross_extent = image.shape[0] if orientation == "horizontal" else image.shape[1]
        maximum_line_thickness = max(12, int(round(cross_extent * 0.02)))
        if (
            length < min_length
            or thickness <= 0
            or thickness > maximum_line_thickness
        ):
            continue
        component_mask = labels[y : y + height, x : x + width] == component
        component_pixels = image[y : y + height, x : x + width][component_mask]
        if component_pixels.size == 0:
            continue
        bgr = np.median(component_pixels.reshape(-1, 3), axis=0)
        edge_contrast = _segment_edge_contrast(
            image,
            [x, y, x + width, y + height],
            orientation,
            bgr,
        )
        if edge_contrast < 12.0:
            continue
        color = "#{:02X}{:02X}{:02X}".format(
            int(round(bgr[2])), int(round(bgr[1])), int(round(bgr[0]))
        )
        if orientation == "horizontal":
            center = round(y + (height - 1) / 2)
            start = [x, center]
            end = [x + width - 1, center]
        else:
            center = round(x + (width - 1) / 2)
            start = [center, y]
            end = [center, y + height - 1]
        segments.append(
            {
                "orientation": orientation,
                "box": [x, y, x + width, y + height],
                "start": start,
                "end": end,
                "thickness": thickness,
                "length": length,
                "color": color,
                "edgeContrast": round(edge_contrast, 2),
                "source": "measured",
                "method": "morphological-line-segment",
            }
        )
    return segments


def _segment_edge_contrast(
    image: Any,
    box: list[int],
    orientation: str,
    line_bgr: Any | None = None,
) -> float:
    """Measure whether a long component is a line, not uniform panel fill."""
    x0, y0, x1, y1 = (int(value) for value in box)
    height, width = image.shape[:2]
    x0, x1 = max(0, x0), min(width, x1)
    y0, y1 = max(0, y0), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    if line_bgr is None:
        line_bgr = np.median(image[y0:y1, x0:x1].reshape(-1, 3), axis=0)
    band = max(2, min(8, (y1 - y0) if orientation == "horizontal" else (x1 - x0)))
    neighbors = (
        [
            image[max(0, y0 - band) : y0, x0:x1],
            image[y1 : min(height, y1 + band), x0:x1],
        ]
        if orientation == "horizontal"
        else [
            image[y0:y1, max(0, x0 - band) : x0],
            image[y0:y1, x1 : min(width, x1 + band)],
        ]
    )
    contrasts = []
    for neighbor in neighbors:
        if neighbor.size == 0:
            continue
        neighbor_bgr = np.median(neighbor.reshape(-1, 3), axis=0)
        contrasts.append(float(np.linalg.norm(line_bgr - neighbor_bgr)))
    return max(contrasts, default=0.0)


def detect_vector_paths(
    image: Any,
    exclusion_boxes: list[list[int]] | None = None,
) -> list[dict[str, Any]]:
    """Measure long thin saturated chart/decorative strokes as polylines."""
    import cv2

    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 100)).astype(np.uint8) * 255
    for raw_box in exclusion_boxes or []:
        if len(raw_box) != 4:
            continue
        x0, y0, x1, y1 = (int(round(value)) for value in raw_box)
        x0, y0 = max(0, x0 - 2), max(0, y0 - 2)
        x1, y1 = min(width, x1 + 2), min(height, y1 + 2)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    paths: list[dict[str, Any]] = []
    for component in range(1, count):
        x, y, box_width, box_height, area = (
            int(value) for value in stats[component]
        )
        box_area = max(1, box_width * box_height)
        fill_ratio = area / box_area
        if (
            box_width < max(80, int(width * 0.08))
            or box_height < max(20, int(height * 0.03))
            or box_width / max(1, box_height) < 1.5
            or fill_ratio > 0.15
            or area < width * height * 0.0005
        ):
            continue
        points: list[list[int]] = []
        column_counts: list[int] = []
        for column in range(x, x + box_width):
            rows = np.where(labels[:, column] == component)[0]
            if not len(rows):
                continue
            points.append([column, int(round(float(np.median(rows))))])
            column_counts.append(int(len(rows)))
        if len(points) < 4:
            continue
        simplified = cv2.approxPolyDP(
            np.asarray(points, dtype=np.int32).reshape(-1, 1, 2),
            2.0,
            False,
        ).reshape(-1, 2)
        if len(simplified) < 3:
            continue
        component_pixels = image[labels == component]
        bgr = np.median(component_pixels.reshape(-1, 3), axis=0)
        color = "#{:02X}{:02X}{:02X}".format(
            int(round(bgr[2])), int(round(bgr[1])), int(round(bgr[0]))
        )
        paths.append(
            {
                "box": [x, y, x + box_width, y + box_height],
                "points": simplified.astype(int).tolist(),
                "strokeColor": color,
                "strokeWidth": max(
                    1,
                    min(6, int(round(float(np.median(column_counts)) / 2.0))),
                ),
                "fill": "none",
                "source": "measured",
                "method": "saturated-thin-path-trace",
            }
        )
    paths.sort(
        key=lambda entry: (
            entry["box"][1],
            entry["box"][0],
            -len(entry["points"]),
        )
    )
    return paths[:8]




def layout_gaps(blocks: list[dict[str, Any]], min_area_ratio: float = 0.03) -> list[dict[str, Any]]:
    """Measure pixel gaps between adjacent sections.

    A pair of blocks is adjacent when they overlap along one axis (>= 30%
    of the smaller extent) and the gap along the other axis is small.
    `touching` means the gap is at most 4px — sections are glued (contour
    erosion eats ~2px per border, so a real 0px gap reads as 0-4px).
    """
    sections = [
        b for b in blocks
        if b.get("kind") != "texture" and b["area"] >= min_area_ratio * max(1, blocks[0]["area"])
    ] if blocks else []
    gaps = []
    for i, first in enumerate(sections):
        for second in sections[i + 1:]:
            fx0, fy0, fx1, fy1 = first["box"]
            sx0, sy0, sx1, sy1 = second["box"]
            overlap_x = min(fx1, sx1) - max(fx0, sx0)
            overlap_y = min(fy1, sy1) - max(fy0, sy0)
            # Near-duplicate contour pairs around the same border: skip.
            if (
                overlap_x > 0.75 * min(fx1 - fx0, sx1 - sx0)
                and overlap_y > 0.75 * min(fy1 - fy0, sy1 - sy0)
            ):
                continue
            if overlap_x >= 0.3 * min(fx1 - fx0, sx1 - sx0):
                # vertical neighbours; order by y
                if sy0 < fy0:
                    top, bottom = second["box"], first["box"]
                else:
                    top, bottom = first["box"], second["box"]
                gap = bottom[1] - top[3]
                axis = "y"
                span = overlap_x
            elif overlap_y >= 0.3 * min(fy1 - fy0, sy1 - sy0):
                # horizontal neighbours; order by x
                if sx0 < fx0:
                    left, right = second["box"], first["box"]
                else:
                    left, right = first["box"], second["box"]
                gap = right[0] - left[2]
                axis = "x"
                span = overlap_y
            else:
                continue
            # Nested pairs (one block fully inside the other) are not
            # adjacent sections — skip them.
            if (
                fx0 >= sx0 + 4 and fy0 >= sy0 + 4 and fx1 <= sx1 - 4 and fy1 <= sy1 - 4
            ) or (
                sx0 >= fx0 + 4 and sy0 >= fy0 + 4 and sx1 <= fx1 - 4 and sy1 <= fy1 - 4
            ):
                continue
            if gap > 64:
                continue
            gaps.append({
                "axis": axis,
                "px": max(0, gap),
                "span": span,
                "touching": gap <= 4,
                "boxes": [top if axis == "y" else left, bottom if axis == "y" else right],
            })
    gaps.sort(key=lambda g: (g["axis"], g["px"]))
    return gaps[:24]




def attention_map(image: Any, ocr_items: list[dict[str, Any]], grid: int = 8) -> list[dict[str, Any]]:
    """8x8 attention grid: text density + local contrast -> hot zones.

    Score per cell: normalized OCR box overlap + normalized luma std.
    Hot zones = cells above the mean, merged by connected groups.
    """
    import cv2

    height, width = image.shape[:2]
    cell_w = width / grid
    cell_h = height / grid
    text_density = np.zeros((grid, grid), dtype=np.float32)
    for item in ocr_items:
        x0, y0, x1, y1 = item["box"]
        cx0 = max(0, int(x0 // cell_w))
        cy0 = max(0, int(y0 // cell_h))
        cx1 = min(grid - 1, int(x1 // cell_w))
        cy1 = min(grid - 1, int(y1 // cell_h))
        for cy in range(cy0, cy1 + 1):
            for cx in range(cx0, cx1 + 1):
                text_density[cy, cx] += 1
    text_density = text_density / max(1.0, float(text_density.max()))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thumb = cv2.resize(gray, (grid, grid), interpolation=cv2.INTER_AREA).astype(np.float32)
    contrast = (thumb - thumb.min()) / max(1e-6, float(thumb.max() - thumb.min()))

    scores = 0.65 * text_density + 0.35 * contrast
    threshold = float(scores.mean()) + 0.5 * float(scores.std())
    mask = scores > threshold
    zones = []
    visited = np.zeros((grid, grid), dtype=bool)
    for cy in range(grid):
        for cx in range(grid):
            if not mask[cy, cx] or visited[cy, cx]:
                continue
            stack = [(cy, cx)]
            visited[cy, cx] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < grid and 0 <= nx < grid and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            min_y = min(c[0] for c in cells)
            max_y = max(c[0] for c in cells)
            min_x = min(c[1] for c in cells)
            max_x = max(c[1] for c in cells)
            score = float(np.mean([scores[y, x] for y, x in cells]))
            zones.append({
                "box": [
                    round(min_x * cell_w), round(min_y * cell_h),
                    round((max_x + 1) * cell_w), round((max_y + 1) * cell_h),
                ],
                "score": round(score, 3),
                "why": "text+contrast density",
            })
    zones.sort(key=lambda z: -z["score"])
    return zones[:8]




# --------------------------------------------------------------------------
# L3: objects (YOLO) + scene (CLIP) — local neural layers, no API
# --------------------------------------------------------------------------

_yolo: Any = None


_clip: Any = None


_clip_preprocess: Any = None


_clip_tokenizer: Any = None


_clip_model_name = "ViT-B-32"



# Zero-shot scene candidates (CLIP is trained on English text).
SCENE_CANDIDATES = [
    "screenshot of a desktop application settings page",
    "screenshot of a mobile app interface",
    "web page layout with navigation",
    "login or sign-up form",
    "dashboard with charts and graphs",
    "spreadsheet or table with numbers",
    "invoice or receipt document",
    "text document or article page",
    "presentation slide",
    "code editor window",
    "chat or messaging interface",
    "error message dialog",
    "e-commerce product page",
    "photograph of people in a park",
    "city street scene",
    "building exterior or interior",
    "nature landscape with trees",
    "beach or sea view",
    "food or meal photograph",
    "product photo on plain background",
    "portrait photo of a person",
    "diagram or flowchart",
    "mathematical formula or equations on a board",
    "map or navigation view",
    "video player interface",
    "macro photography close-up of flowers",
    "butterfly on flowers in a garden",
    "garden with green plants and flowers",
    "wildlife animal photograph",
    "clouds and sky view",
    "abstract geometric shapes and figures",
    "colorful circles and shapes on white background",
    "weather forecast interface",
    "music player interface",
    "photo gallery or image grid",
]




def objects_yolo(image_path: str) -> list[dict[str, Any]]:
    global _yolo
    if _yolo is None:
        from ultralytics import YOLO

        weights = WORKER_DIR.parent / "models" / "yolov8n.pt"
        if not weights.is_file():
            weights = Path(os.environ.get("SENS_SPEECH_ROOT", "")) / "models" / "yolov8n.pt"
        _yolo = YOLO(str(weights))
    results = _yolo(image_path, verbose=False)
    items: list[dict[str, Any]] = []
    for result in results:
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            x0, y0, x1, y1 = (float(v) for v in box)
            items.append({
                "class": result.names[int(cls)],
                "box": [round(x0), round(y0), round(x1), round(y1)],
                "confidence": round(float(conf), 3),
                "source": "inferred",
                "method": "yolov8n",
            })
    return items




def _clip_loaded() -> bool:
    return _clip is not None




def _load_clip() -> None:
    global _clip, _clip_preprocess, _clip_tokenizer
    import open_clip

    _clip, _clip_preprocess, _ = open_clip.create_model_and_transforms(
        _clip_model_name, pretrained="openai"
    )
    _clip_tokenizer = open_clip.get_tokenizer(_clip_model_name)




def scene_clip(image_path: str) -> list[dict[str, Any]]:
    """Zero-shot scene classification over SCENE_CANDIDATES (top 3)."""
    global _clip
    if _clip is None:
        _load_clip()
    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    with torch.no_grad():
        image_input = _clip_preprocess(image).unsqueeze(0)
        image_features = _clip.encode_image(image_input).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_tokens = _clip_tokenizer(SCENE_CANDIDATES)
        text_features = _clip.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)[0]
    ranked = sorted(
        zip(SCENE_CANDIDATES, probs.tolist()),
        key=lambda pair: pair[1],
        reverse=True,
    )[:3]
    return [
        {
            "label": label,
            "confidence": round(float(score), 3),
            "source": "inferred",
            "method": "clip-vit-b-32",
        }
        for label, score in ranked
        if score > 0.01
    ]




def _intersection_ratio(left: list[int], right: list[int]) -> float:
    """Fraction of the smaller box covered by the intersection of both."""
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    intersection = (x1 - x0) * (y1 - y0)
    smaller = min(
        (left[2] - left[0]) * (left[3] - left[1]),
        (right[2] - right[0]) * (right[3] - right[1]),
    )
    return intersection / smaller if smaller > 0 else 0.0




# Known font silhouettes: (family, average glyph width in em, cap height in
# em). Uppercase-heavy UI text; mixed-case reads slightly narrower.
_KNOWN_FONTS: list[tuple[str, float, float]] = [
    ("arial", 0.60, 0.72),
    ("helvetica", 0.60, 0.72),
    ("inter", 0.58, 0.73),
    ("roboto", 0.61, 0.71),
    ("montserrat", 0.66, 0.70),
    ("oswald", 0.55, 0.72),
    ("bebas", 0.44, 0.73),
    ("anton", 0.72, 0.95),
    ("space-grotesk", 0.58, 0.72),
]


def rank_font_candidates(width_em: float, limit: int = 3) -> list[dict[str, Any]]:
    """Rank silhouette matches without presenting a font family as observed."""
    ranked = sorted(_KNOWN_FONTS, key=lambda item: abs(item[1] - width_em))
    return [
        {
            "family": family,
            "widthEm": width,
            "distance": round(abs(width - width_em), 4),
            "status": "candidate",
            "method": "glyph-width-silhouette",
        }
        for family, width, _cap_height in ranked[: max(1, limit)]
    ]




def _glyph_metrics(
    image: Any, box: list[int], text: str | None = None
) -> dict[str, Any] | None:
    """Cap height / average glyph width of an OCR line, plus a best-effort
    family guess. The copier sizes fonts from these numbers instead of
    guessing; `custom` means no known font matches the silhouette (cut the
    line as a graphic asset)."""
    import cv2

    x0, y0, x1, y1 = (int(v) for v in box)
    if x1 - x0 < 8 or y1 - y0 < 4:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    patch = gray[y0:y1, x0:x1].astype(int)
    border = np.concatenate(
        (patch[0, :], patch[-1, :], patch[:, 0], patch[:, -1]), axis=0
    )
    image_border = np.concatenate(
        (gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]), axis=0
    )

    def contrast_amplitude(background: float) -> float:
        deviations = np.abs(patch - background)
        signal = deviations[deviations >= 4]
        if signal.size < 3:
            return 0.0
        # A percentile over the whole crop collapses to zero for thin type
        # occupying under ten percent of a padded OCR polygon.  Measure the
        # non-background signal itself so one-pixel UI strokes remain valid.
        return float(np.percentile(signal, 75))

    def measure(
        background: float,
    ) -> tuple[int, float, float, float, float, int] | None:
        amp = contrast_amplitude(background)
        if amp < 18:
            return None
        mask = np.abs(patch - background) > 0.35 * amp
        rows = np.where(mask.any(axis=1))[0]
        if len(rows) < 3:
            return None
        # Cap rows: rows with a substantial share of the busiest row. This
        # keeps ascender rows of tall caps and drops descender tails (p, y)
        # that only carry a few glyph pixels.
        per_row = mask.sum(axis=1)
        if int(per_row.max()) == 0:
            return None
        strong = np.where(per_row > 0.08 * per_row.max())[0]
        if len(strong) < 2:
            return None
        cap_height = int(strong[-1] - strong[0] + 1)
        # Split strokes into letters; fuse sloppy anti-alias gaps, split glued
        # pairs by the median run width.
        widths: list[int] = []
        run = 0
        for col in range(x1 - x0):
            if mask[:, col].any():
                run += 1
            else:
                if run > 0:
                    widths.append(run)
                run = 0
        if run > 0:
            widths.append(run)
        if not widths:
            return None
        median_w = float(np.median(widths))
        stroke_letters = sum(
            max(1, round(w / max(1.6, median_w))) for w in widths
        )
        recognized_characters = sum(not char.isspace() for char in (text or ""))
        letters = max(stroke_letters, recognized_characters)
        average_glyph = float(np.sum(widths)) / max(1, letters)
        size = cap_height / 0.73
        width_in_em = average_glyph / size if size > 0 else 0.0
        return (
            cap_height,
            average_glyph,
            size,
            width_in_em,
            background,
            stroke_letters,
        )

    measured = measure(float(np.median(border)))
    if measured is None:
        measured = measure(float(np.median(image_border)))
    elif not 0.12 <= measured[3] <= 1.2:
        fallback = measure(float(np.median(image_border)))
        if fallback is not None and 0.12 <= fallback[3] <= 1.2:
            measured = fallback
    if measured is None:
        return None
    (
        cap,
        avg_glyph,
        font_size,
        width_em,
        measured_background,
        measured_character_count,
    ) = measured
    amplitude = contrast_amplitude(measured_background)
    glyph_mask = np.abs(patch - measured_background) > 0.35 * amplitude
    mask_u8 = glyph_mask.astype(np.uint8)
    stroke_distances = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)[glyph_mask]
    stroke_width = (
        float(np.median(stroke_distances)) * 2.0
        if stroke_distances.size
        else 0.0
    )
    stroke_width_p75 = (
        float(np.percentile(stroke_distances, 75)) * 2.0
        if stroke_distances.size
        else 0.0
    )
    stroke_ratio = stroke_width / max(1.0, float(cap))
    ink_coverage = float(glyph_mask.mean())
    # Weight is categorical, so keep it explicitly separate from the measured
    # stroke values. Only emit a candidate at the two well-separated extremes;
    # the local VLM remains useful in the ambiguous middle band and on tiny type.
    weight_candidate = None
    if cap >= 40:
        if stroke_ratio <= 0.09 and ink_coverage <= 0.22:
            weight_candidate = "light"
        elif stroke_ratio >= 0.12 or ink_coverage >= 0.28:
            weight_candidate = "bold"
    glyph_patch = image[y0:y1, x0:x1]
    glyph_distances = np.abs(patch - measured_background)
    glyph_values = glyph_distances[glyph_mask]
    core_mask = glyph_mask
    if glyph_values.size:
        # Anti-aliased edge shades are often repeated more frequently than the
        # actual ink colour on tiny UI type.  Taking the modal glyph pixel made
        # dark labels on light dashboards almost white.  The farthest quartile
        # from the measured local background is the stable ink core for both
        # dark-on-light and light-on-dark text.
        core_threshold = float(np.percentile(glyph_values, 72))
        candidate_core = glyph_mask & (glyph_distances >= core_threshold)
        if int(candidate_core.sum()) >= 3:
            core_mask = candidate_core
    glyph_pixels = glyph_patch[core_mask]
    glyph_color = None
    if glyph_pixels.size:
        bgr = np.median(glyph_pixels.reshape(-1, 3), axis=0)
        glyph_color = "#{:02X}{:02X}{:02X}".format(
            int(bgr[2]), int(bgr[1]), int(bgr[0])
        )
    glyph_boxes: list[dict[str, Any]] = []
    measured_character_count_method = "foreground-column-run-splitting"
    compact_text = str(text or "").strip()
    expected_characters = len(compact_text)
    if (
        cap >= 64
        and re.fullmatch(r"[A-Z0-9]{3,12}", compact_text)
        and (
            measured_character_count < expected_characters
            or ink_coverage >= 0.80
        )
    ):
        minimum_height = max(24, int(round(cap * 0.55)))
        minimum_width = max(8, int(round(cap * 0.08)))
        minimum_area = max(64, int(round(cap * cap * 0.025)))

        # A display OCR crop can begin on a section border.  In that case the
        # border median is not the canvas background: the contrast mask turns
        # most of the artwork into "ink" and its modal colour is wrong.  Search
        # the compact measured palette as well as that first colour, then keep
        # only a horizontally ordered, cap-height-consistent run whose component
        # count agrees with the recognized token.
        color_candidates: list[np.ndarray] = []

        def add_color_candidate(value: Any) -> None:
            candidate = np.asarray(value, dtype=np.float32)
            if candidate.shape != (3,):
                return
            if any(
                float(np.linalg.norm(candidate - existing)) <= 8.0
                for existing in color_candidates
            ):
                return
            color_candidates.append(candidate)

        if glyph_color:
            add_color_candidate(_hex_to_bgr(glyph_color))
        flat_pixels = glyph_patch.reshape(-1, 3)
        quantized = flat_pixels.astype(np.uint16) // 16
        color_codes = (
            quantized[:, 0] * 256
            + quantized[:, 1] * 16
            + quantized[:, 2]
        ).astype(np.int32)
        palette_counts = np.bincount(color_codes, minlength=4096)
        for color_code in np.argsort(palette_counts)[::-1][:32]:
            if int(palette_counts[color_code]) < minimum_area:
                break
            palette_pixels = flat_pixels[color_codes == color_code]
            if palette_pixels.size:
                add_color_candidate(np.median(palette_pixels, axis=0))

        recovered: tuple[
            tuple[float, float, float],
            np.ndarray,
            list[tuple[int, int, int, int, int, int]],
            np.ndarray,
            np.ndarray,
        ] | None = None
        for target_bgr in color_candidates:
            color_distance = np.linalg.norm(
                glyph_patch.astype(np.float32) - target_bgr,
                axis=2,
            )
            component_seed = (color_distance <= 44.0).astype(np.uint8)
            count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
                component_seed,
                8,
            )
            components: list[tuple[int, int, int, int, int, int]] = []
            for label in range(1, count):
                local_x, local_y, component_width, component_height, area = (
                    int(value) for value in stats[label]
                )
                if (
                    component_height >= minimum_height
                    and component_width >= minimum_width
                    and area >= minimum_area
                ):
                    components.append(
                        (
                            local_x,
                            local_y,
                            component_width,
                            component_height,
                            area,
                            label,
                        )
                    )
            components.sort(key=lambda item: item[0])
            if len(components) != expected_characters:
                continue
            heights = np.asarray(
                [component[3] for component in components], dtype=np.float32
            )
            bottoms = np.asarray(
                [component[1] + component[3] for component in components],
                dtype=np.float32,
            )
            median_height = max(1.0, float(np.median(heights)))
            height_spread = float(np.ptp(heights)) / median_height
            baseline_spread = float(np.ptp(bottoms)) / median_height
            if height_spread > 0.35 or baseline_spread > 0.35:
                continue
            quality = (
                height_spread,
                baseline_spread,
                -float(sum(component[4] for component in components)),
            )
            candidate = (quality, target_bgr, components, labels, component_seed)
            if recovered is None or quality < recovered[0]:
                recovered = candidate

        if recovered is not None:
            _quality, target_bgr, components, labels, _component_seed = recovered
            selected_labels = np.asarray(
                [component[5] for component in components], dtype=np.int32
            )
            glyph_mask = np.isin(labels, selected_labels)
            glyph_color = "#{:02X}{:02X}{:02X}".format(
                int(round(target_bgr[2])),
                int(round(target_bgr[1])),
                int(round(target_bgr[0])),
            )
            heights = [component[3] for component in components]
            widths = [component[2] for component in components]
            cap = int(round(float(np.median(heights))))
            avg_glyph = float(np.mean(widths))
            font_size = cap / 0.73
            width_em = avg_glyph / font_size if font_size > 0 else 0.0
            measured_character_count = expected_characters
            measured_character_count_method = (
                "glyph-color-connected-components"
            )
            mask_u8 = glyph_mask.astype(np.uint8)
            stroke_distances = cv2.distanceTransform(
                mask_u8, cv2.DIST_L2, 5
            )[glyph_mask]
            stroke_width = (
                float(np.median(stroke_distances)) * 2.0
                if stroke_distances.size
                else 0.0
            )
            stroke_width_p75 = (
                float(np.percentile(stroke_distances, 75)) * 2.0
                if stroke_distances.size
                else 0.0
            )
            stroke_ratio = stroke_width / max(1.0, float(cap))
            ink_coverage = float(glyph_mask.mean())
            weight_candidate = None
            # OpenCV 5's antialiased Hershey masks occupy more of the crop than
            # older builds. Keep the classes separated by both measured stroke
            # width and coverage instead of treating every portable thin glyph
            # as bold.
            if stroke_ratio <= 0.09 and ink_coverage <= 0.22:
                weight_candidate = "light"
            elif stroke_ratio >= 0.12 or ink_coverage >= 0.28:
                weight_candidate = "bold"
            glyph_boxes = [
                {
                    "text": character,
                    "box": [
                        x0 + component[0],
                        y0 + component[1],
                        x0 + component[0] + component[2],
                        y0 + component[1] + component[3],
                    ],
                }
                for character, component in zip(
                    compact_text, components, strict=True
                )
            ]
    ink_rows, ink_columns = np.where(glyph_mask)
    ink_box = (
        [
            x0 + int(ink_columns.min()),
            y0 + int(ink_rows.min()),
            x0 + int(ink_columns.max()) + 1,
            y0 + int(ink_rows.max()) + 1,
        ]
        if ink_rows.size and ink_columns.size
        else None
    )
    word_boxes: list[dict[str, Any]] = []
    word_tokens = re.findall(r"\S+", str(text or ""))
    active_columns = np.flatnonzero(glyph_mask.any(axis=0))
    column_runs: list[tuple[int, int]] = []
    if active_columns.size:
        start = previous = int(active_columns[0])
        for column in active_columns[1:]:
            column = int(column)
            if column > previous + 1:
                column_runs.append((start, previous + 1))
                start = column
            previous = column
        column_runs.append((start, previous + 1))
    word_box_method = None
    if 2 <= len(word_tokens) <= len(column_runs):
        split_count = len(word_tokens) - 1
        ranked_gaps = sorted(
            (
                (column_runs[index + 1][0] - column_runs[index][1], index)
                for index in range(len(column_runs) - 1)
            ),
            reverse=True,
        )
        split_after = {
            index for gap, index in ranked_gaps[:split_count] if gap >= 2
        }
        if len(split_after) == split_count:
            grouped_runs: list[list[tuple[int, int]]] = [[]]
            for index, run in enumerate(column_runs):
                grouped_runs[-1].append(run)
                if index in split_after:
                    grouped_runs.append([])
            if len(grouped_runs) == len(word_tokens) and all(grouped_runs):
                for token, group in zip(word_tokens, grouped_runs, strict=True):
                    local_x0 = group[0][0]
                    local_x1 = group[-1][1]
                    local_rows = np.flatnonzero(
                        glyph_mask[:, local_x0:local_x1].any(axis=1)
                    )
                    if not local_rows.size:
                        word_boxes = []
                        break
                    word_boxes.append(
                        {
                            "text": token,
                            "box": [
                                x0 + local_x0,
                                y0 + int(local_rows.min()),
                                x0 + local_x1,
                                y0 + int(local_rows.max()) + 1,
                            ],
                        }
                    )
                if word_boxes:
                    word_box_method = "largest-foreground-column-gaps"
    if not word_boxes and 2 <= len(word_tokens) <= 3 and ink_columns.size:
        ink_x0 = int(ink_columns.min())
        ink_x1 = int(ink_columns.max()) + 1
        weights = [
            max(1, sum(character.isalnum() for character in token))
            for token in word_tokens
        ]
        total_weight = max(1, sum(weights))
        cursor = ink_x0
        consumed_weight = 0
        for index, (token, weight) in enumerate(zip(word_tokens, weights, strict=True)):
            consumed_weight += weight
            local_x1 = (
                ink_x1
                if index == len(word_tokens) - 1
                else int(
                    round(
                        ink_x0
                        + (ink_x1 - ink_x0) * consumed_weight / total_weight
                    )
                )
            )
            local_rows = np.flatnonzero(glyph_mask[:, cursor:local_x1].any(axis=1))
            if not local_rows.size or local_x1 <= cursor:
                word_boxes = []
                break
            word_boxes.append(
                {
                    "text": token,
                    "box": [
                        x0 + cursor,
                        y0 + int(local_rows.min()),
                        x0 + local_x1,
                        y0 + int(local_rows.max()) + 1,
                    ],
                }
            )
            cursor = local_x1
        if word_boxes:
            word_box_method = "character-proportional-ink-fallback"

    def measured_slant(local_box: list[int]) -> tuple[str | None, float, float]:
        lx0 = max(0, int(local_box[0]) - x0)
        ly0 = max(0, int(local_box[1]) - y0)
        lx1 = min(glyph_mask.shape[1], int(local_box[2]) - x0)
        ly1 = min(glyph_mask.shape[0], int(local_box[3]) - y0)
        local_gray = patch[ly0:ly1, lx0:lx1]
        if local_gray.size == 0 or min(local_gray.shape) < 8:
            return None, 0.0, 0.0
        local_border = np.concatenate(
            (
                local_gray[0, :],
                local_gray[-1, :],
                local_gray[:, 0],
                local_gray[:, -1],
            ),
            axis=0,
        )
        local_background = float(np.median(local_border))
        local_deviation = np.abs(local_gray - local_background)
        local_signal = local_deviation[local_deviation >= 4]
        if local_signal.size < 3:
            return None, 0.0, 0.0
        local_amplitude = float(np.percentile(local_signal, 75))
        if local_amplitude < 12:
            return None, 0.0, 0.0
        local = (
            local_deviation > max(5.0, 0.35 * local_amplitude)
        ).astype(np.uint8) * 255
        edges = cv2.Canny(local, 50, 150)
        minimum = max(8, int(round(local.shape[0] * 0.25)))
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=max(8, int(round(local.shape[0] * 0.15))),
            minLineLength=minimum,
            maxLineGap=4,
        )
        slopes: list[float] = []
        if lines is not None:
            for sx0, sy0, sx1, sy1 in lines.reshape(-1, 4):
                dx = int(sx1) - int(sx0)
                dy = int(sy1) - int(sy0)
                if abs(dy) >= minimum and abs(dx) <= abs(dy) * 0.6:
                    slopes.append(dx / dy)
        if len(slopes) < 3:
            return None, 0.0, 0.0
        slope = float(np.median(slopes))
        magnitude = abs(slope)
        if magnitude <= 0.04:
            label = "normal"
        elif magnitude >= 0.07:
            label = "italic"
        else:
            label = None
        confidence = min(0.95, 0.55 + 0.04 * len(slopes) + magnitude)
        return label, slope, confidence

    for word_box in word_boxes:
        slant, slope, confidence = measured_slant(word_box["box"])
        if slant is not None:
            word_box["slant"] = slant
            word_box["slantSlope"] = round(slope, 4)
            word_box["slantConfidence"] = round(confidence, 3)
            word_box["slantMethod"] = "vertical-stroke-hough-median"
    candidates = rank_font_candidates(width_em)
    best = candidates[0]
    custom = best["distance"] > 0.07
    return {
        "capHeight": int(cap),
        "avgGlyphWidth": round(avg_glyph, 1),
        "fontSize": int(round(font_size)),
        "widthEm": round(width_em, 2),
        # Compatibility hint only. The candidate list is the canonical result;
        # pixels cannot confirm an installed font family.
        "family": "custom" if custom else best["family"],
        "familyStatus": "unknown" if custom else "candidate",
        "familyCandidates": candidates,
        "familyConfidence": 0.0
        if custom
        else round(max(0.0, 0.6 * (1.0 - best["distance"] / 0.07)), 3),
        "characterCount": sum(not char.isspace() for char in (text or "")) or None,
        "measuredCharacterCount": measured_character_count,
        "measuredCharacterCountMethod": measured_character_count_method,
        "strokeWidthPx": round(stroke_width, 2),
        "strokeWidthP75Px": round(stroke_width_p75, 2),
        "strokeWidthRatio": round(stroke_ratio, 4),
        "inkCoverage": round(ink_coverage, 4),
        "weightCandidate": weight_candidate,
        "weightCandidateStatus": "candidate" if weight_candidate else None,
        "weightCandidateEpistemic": (
            "inferred-from-measurement" if weight_candidate else None
        ),
        "weightCandidateMethod": (
            "glyph-mask-distance-transform" if weight_candidate else None
        ),
        "color": glyph_color,
        "colorSource": "measured-glyph-pixels" if glyph_color else None,
        "inkBox": ink_box,
        "inkBoxSource": "measured-local-background-contrast" if ink_box else None,
        "glyphBoxes": glyph_boxes or None,
        "glyphBoxMethod": (
            measured_character_count_method if glyph_boxes else None
        ),
        "wordBoxes": word_boxes or None,
        "wordBoxMethod": word_box_method,
        "method": "glyph-width-silhouette+ocr-character-count+stroke-geometry",
    }




def texture_blocks(
    image: Any, ocr_items: list[dict[str, Any]], min_area: int = 40000
) -> list[dict[str, Any]]:
    """Large textured graphics (patterns, illustrations) that the contour
    layout pass misses: threshold local variance, close, keep big blobs
    that do not wrap OCR text."""
    import cv2

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = 4  # work at 1/4 resolution
    small = cv2.resize(
        gray, (width // scale, height // scale), interpolation=cv2.INTER_AREA
    )
    mean = cv2.boxFilter(small, ddepth=-1, ksize=(9, 9)).astype(np.float64)
    sq = cv2.boxFilter(small.astype(np.float64) ** 2, ddepth=-1, ksize=(9, 9))
    var = np.clip(sq - mean * mean, 0, None)
    textured = (var > 220).astype(np.uint8) * 255  # local std > ~15
    closed = cv2.morphologyEx(
        textured, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)
    )
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    ocr_boxes = [item["box"] for item in ocr_items]
    result: list[dict[str, Any]] = []
    for index in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[index])
        if area * scale * scale < min_area:
            continue
        box = [x * scale, y * scale, (x + w) * scale, (y + h) * scale]
        area_px = (box[2] - box[0]) * (box[3] - box[1])
        # Drop blobs that are mostly OCR text; text overlapping a large
        # graphic (a pattern behind a headline) must not kill the block.
        text_share = 0.0
        for ocr_box in ocr_boxes:
            ox0 = max(box[0], ocr_box[0])
            oy0 = max(box[1], ocr_box[1])
            ox1 = min(box[2], ocr_box[2])
            oy1 = min(box[3], ocr_box[3])
            if ox1 > ox0 and oy1 > oy0:
                text_share = max(
                    text_share, (ox1 - ox0) * (oy1 - oy0) / area_px
                )
        if text_share > 0.3:
            continue
        result.append({
            "kind": "texture",
            "box": box,
            "area": (box[2] - box[0]) * (box[3] - box[1]),
            "source": "measured",
        })
    result.sort(key=lambda b: -b["area"])
    return result[:8]




def _controls_around_text(
    image: Any,
    ocr_items: list[dict[str, Any]],
    existing: list[dict[str, Any]],
    top: int = 8,
) -> list[dict[str, Any]]:
    """Buttons filled with near-page-background color (dark-on-dark) escape
    the contour pass; find them by the uniform fill ring around an OCR line."""
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    page_bg = float(np.median(gray[:40, :40]))
    out: list[dict[str, Any]] = []
    for item in ocr_items:
        x0, y0, x1, y1 = (int(v) for v in item["box"])
        if x1 - x0 < 12 or y1 - y0 < 4:
            continue
        ring = gray[max(0, y0 - 12):y1 + 12, max(0, x0 - 12):x1 + 12]
        ring_mask = np.ones(ring.shape, dtype=bool)
        py0 = y0 - max(0, y0 - 12)
        px0 = x0 - max(0, x0 - 12)
        ring_mask[py0:py0 + (y1 - y0), px0:px0 + (x1 - x0)] = False
        vals = ring[ring_mask]
        if vals.size < 60:
            continue
        bg = float(np.median(vals))
        if abs(bg - page_bg) < 15 or float(vals.std()) > 34:
            continue
        # Grow bounds while the neighbouring band stays near the fill color.
        # The band is 6px; the fill sits on the box side of it, so sample the
        # half closest to the box to avoid mixing in the page background.
        bx0, by0, bx1, by1 = x0, y0, x1, y1
        for _ in range(200):
            moved = False
            if by0 > 6:
                row = gray[by0 - 6:by0, bx0:bx1]
                if abs(float(np.median(row[-3:])) - bg) < 14:
                    by0 -= 1
                    moved = True
            if by1 < height - 6:
                row = gray[by1:by1 + 6, bx0:bx1]
                if abs(float(np.median(row[:3])) - bg) < 14:
                    by1 += 1
                    moved = True
            if bx0 > 6:
                col = gray[by0:by1, bx0 - 6:bx0]
                if abs(float(np.median(col[:, -3:])) - bg) < 14:
                    bx0 -= 1
                    moved = True
            if bx1 < width - 6:
                col = gray[by0:by1, bx1:bx1 + 6]
                if abs(float(np.median(col[:, :3])) - bg) < 14:
                    bx1 += 1
                    moved = True
            if not moved:
                break
        box = [bx0, by0, bx1, by1]
        if any(_intersection_ratio(box, c["box"]) > 0.5 for c in existing):
            continue
        boundary_deltas: list[float] = []
        boundary_band = 3
        side_pairs = [
            (
                image[by0 : min(by1, by0 + boundary_band), bx0:bx1],
                image[max(0, by0 - boundary_band) : by0, bx0:bx1],
            ),
            (
                image[max(by0, by1 - boundary_band) : by1, bx0:bx1],
                image[by1 : min(height, by1 + boundary_band), bx0:bx1],
            ),
            (
                image[by0:by1, bx0 : min(bx1, bx0 + boundary_band)],
                image[by0:by1, max(0, bx0 - boundary_band) : bx0],
            ),
            (
                image[by0:by1, max(bx0, bx1 - boundary_band) : bx1],
                image[by0:by1, bx1 : min(width, bx1 + boundary_band)],
            ),
        ]
        for inside, outside in side_pairs:
            if inside.size == 0 or outside.size == 0:
                continue
            inside_color = np.median(inside.reshape((-1, 3)), axis=0)
            outside_color = np.median(outside.reshape((-1, 3)), axis=0)
            boundary_deltas.append(
                float(np.linalg.norm(inside_color - outside_color))
            )
        # A uniform section of a photographic card can surround one label but
        # is not a filled control. A real fill has a closed colour transition.
        if len(boundary_deltas) < 3 or min(boundary_deltas) < 12.0:
            continue
        # Fill color: median of the ring around the text (the box center
        # can land on a glyph).
        ring_bgr = image[max(0, y0 - 12):y1 + 12, max(0, x0 - 12):x1 + 12]
        ring_mask = np.ones(ring_bgr.shape[:2], dtype=bool)
        ring_mask[py0:py0 + (y1 - y0), px0:px0 + (x1 - x0)] = False
        pixels = ring_bgr[ring_mask]
        if pixels.size < 30:
            continue
        pixel = np.median(pixels.reshape(-1, 3), axis=0)
        out.append({
            "background": "#{:02X}{:02X}{:02X}".format(
                int(pixel[2]), int(pixel[1]), int(pixel[0])
            ),
            "borderColor": None,
            "borderWidth": None,
            "box": box,
            "cornerRadius": 0,
            "labelText": item.get("text", ""),
            "labelBox": [x0, y0, x1, y1],
            "boundaryEvidence": {
                "closedFill": True,
                "minimumColorDelta": round(min(boundary_deltas), 2),
            },
            "source": "measured",
            "method": "compact-fill-with-closed-color-boundary",
        })
    return out[:top]


def _outline_corner_radius(foreground: Any, band: int) -> int:
    """Estimate a rounded-rectangle radius from its four boundary insets."""
    height, width = foreground.shape[:2]
    if height < 8 or width < 8:
        return 0
    top = np.where(foreground[:band, :].any(axis=0))[0]
    bottom = np.where(foreground[-band:, :].any(axis=0))[0]
    left = np.where(foreground[:, :band].any(axis=1))[0]
    right = np.where(foreground[:, -band:].any(axis=1))[0]
    if any(values.size == 0 for values in (top, bottom, left, right)):
        return 0
    insets = [
        int(top.min()),
        width - 1 - int(top.max()),
        int(bottom.min()),
        width - 1 - int(bottom.max()),
        int(left.min()),
        height - 1 - int(left.max()),
        int(right.min()),
        height - 1 - int(right.max()),
    ]
    inset = float(np.median(insets))
    if inset <= 2.0:
        return 0
    if width / max(1.0, float(height)) >= 2.5 and inset >= band * 1.5:
        return int(round(height / 2.0))
    # For a quarter-circle, the inset at depth b is
    # a = r - sqrt(2rb - b^2), hence r = a+b+sqrt(2ab).
    radius = inset + band + float(np.sqrt(2.0 * inset * band))
    return int(round(min(radius, width / 2.0, height / 2.0)))


def outlined_controls_around_text(
    image: Any,
    ocr_items: list[dict[str, Any]],
    existing: list[dict[str, Any]],
    top: int = 12,
) -> list[dict[str, Any]]:
    """Find outline-only controls whose interior matches the page background.

    A large text label alone is not interaction evidence.  A candidate is kept
    only when one measured contour encloses the OCR box with padding and the
    foreground mask supports all four sides of a closed boundary.
    """
    import cv2

    height, width = image.shape[:2]
    pixels = image.astype(np.float32)
    border_pixels = np.concatenate(
        (pixels[0, :, :], pixels[-1, :, :], pixels[:, 0, :], pixels[:, -1, :]),
        axis=0,
    )
    page_background = np.median(border_pixels, axis=0)
    foreground = (
        np.linalg.norm(pixels - page_background, axis=2) > 22.0
    ).astype(np.uint8) * 255
    contours, _hierarchy = cv2.findContours(
        foreground, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    def side_support(box: list[int], mask: Any = foreground) -> dict[str, float]:
        x0, y0, x1, y1 = box
        band = max(3, min(7, round(min(x1 - x0, y1 - y0) * 0.07)))
        top_band = mask[y0 : min(height, y0 + band), x0:x1]
        bottom_band = mask[max(0, y1 - band) : y1, x0:x1]
        left_band = mask[y0:y1, x0 : min(width, x0 + band)]
        right_band = mask[y0:y1, max(0, x1 - band) : x1]
        return {
            "top": round(float(np.count_nonzero(top_band)) / max(1, top_band.size), 3),
            "right": round(float(np.count_nonzero(right_band)) / max(1, right_band.size), 3),
            "bottom": round(
                float(np.count_nonzero(bottom_band)) / max(1, bottom_band.size), 3
            ),
            "left": round(float(np.count_nonzero(left_band)) / max(1, left_band.size), 3),
        }

    candidates: list[dict[str, Any]] = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_foreground = cv2.Canny(
        cv2.GaussianBlur(gray, (3, 3), 0), 40, 120
    )
    edge_foreground = cv2.morphologyEx(
        edge_foreground,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
    )
    edge_contours, _edge_hierarchy = cv2.findContours(
        edge_foreground, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    def strong_boundary_color(samples: Any) -> Any:
        """Reject pale antialias pixels when a solid outline core exists."""
        flattened = samples.reshape(-1, 3).astype(np.float32)
        distances = np.linalg.norm(flattened - page_background, axis=1)
        if len(flattened) >= 8:
            cutoff = float(np.percentile(distances, 70))
            strong = flattened[distances >= cutoff]
            if len(strong) >= 3:
                flattened = strong
        return np.median(flattened, axis=0)
    for item in ocr_items:
        tx0, ty0, tx1, ty1 = (int(round(value)) for value in item["box"])
        text_width = tx1 - tx0
        text_height = ty1 - ty0
        if text_width < 8 or text_height < 4:
            continue
        eligible: list[tuple[int, Any, list[int], dict[str, float]]] = []
        for contour in edge_contours:
            x, y, candidate_width, candidate_height = cv2.boundingRect(contour)
            x1 = x + candidate_width
            y1 = y + candidate_height
            if candidate_width < text_width + 4 or candidate_height < text_height + 2:
                continue
            if (
                candidate_width > max(500, text_width * 3.0)
                or candidate_height > max(90, text_height * 4.5)
            ):
                continue
            pad_x = max(2, min(8, round(text_height * 0.12)))
            pad_y = max(1, min(4, round(text_height * 0.04)))
            if not (
                x <= tx0 - pad_x
                and y <= ty0 - pad_y
                and x1 >= tx1 + pad_x
                and y1 >= ty1 + pad_y
            ):
                continue
            box = [x, y, x1, y1]
            support = side_support(box, edge_foreground)
            if min(support.values()) < 0.2:
                continue
            perimeter = max(1.0, 2.0 * (candidate_width + candidate_height))
            if cv2.arcLength(contour, True) / perimeter < 0.72:
                continue
            if any(_intersection_ratio(box, control["box"]) > 0.8 for control in existing):
                continue
            eligible.append((candidate_width * candidate_height, contour, box, support))
        if not eligible:
            continue
        _area, contour, box, support = max(eligible, key=lambda value: value[0])
        x, y, x1, y1 = box
        candidate_height = y1 - y
        candidate_width = x1 - x
        band = max(3, min(7, round(min(candidate_width, candidate_height) * 0.07)))
        crop_edges = edge_foreground[y:y1, x:x1] > 0
        band_mask = np.zeros((candidate_height, candidate_width), dtype=bool)
        band_mask[:band, :] = True
        band_mask[-band:, :] = True
        band_mask[:, :band] = True
        band_mask[:, -band:] = True
        crop_foreground = foreground[y:y1, x:x1] > 0
        boundary_pixels = image[y:y1, x:x1][band_mask & crop_foreground]
        if boundary_pixels.size == 0:
            continue
        boundary_bgr = strong_boundary_color(boundary_pixels)
        interior = image[y + band : y1 - band, x + band : x1 - band]
        interior_bgr = (
            np.median(interior.reshape(-1, 3), axis=0)
            if interior.size
            else page_background
        )
        candidates.append(
            {
                "kind": "button",
                "background": "#{:02X}{:02X}{:02X}".format(
                    int(round(interior_bgr[2])),
                    int(round(interior_bgr[1])),
                    int(round(interior_bgr[0])),
                ),
                "borderColor": "#{:02X}{:02X}{:02X}".format(
                    int(round(boundary_bgr[2])),
                    int(round(boundary_bgr[1])),
                    int(round(boundary_bgr[0])),
                ),
                "borderWidth": max(1, round(band * min(support.values()))),
                "box": box,
                "cornerRadius": _outline_corner_radius(crop_edges, band),
                "labelText": item.get("text", ""),
                "labelBox": [tx0, ty0, tx1, ty1],
                "boundaryEvidence": {**support, "closed": True},
                "source": "measured",
                "method": "edge-closed-outline-around-ocr",
            }
        )
    for item in ocr_items:
        tx0, ty0, tx1, ty1 = (int(round(value)) for value in item["box"])
        text_width = tx1 - tx0
        text_height = ty1 - ty0
        if text_width < 8 or text_height < 4:
            continue
        for contour in contours:
            x, y, candidate_width, candidate_height = cv2.boundingRect(contour)
            x1 = x + candidate_width
            y1 = y + candidate_height
            if candidate_width < text_width + 4 or candidate_height < text_height + 2:
                continue
            if candidate_width > width * 0.98 or candidate_height > height * 0.8:
                continue
            # The full OCR box, not merely its center, must sit inside a
            # visibly padded boundary. This rejects individual glyph contours.
            # Small outlined labels are often OCR'd together with one or two
            # antialiased border rows.  A closed measured contour is stronger
            # evidence than generous whitespace, so tolerate tight vertical
            # padding while still requiring the whole OCR box to be enclosed.
            pad_x = max(2, min(8, round(text_height * 0.12)))
            pad_y = max(1, min(4, round(text_height * 0.04)))
            if not (
                x <= tx0 - pad_x
                and y <= ty0 - pad_y
                and x1 >= tx1 + pad_x
                and y1 >= ty1 + pad_y
            ):
                continue
            box = [x, y, x1, y1]
            support = side_support(box)
            closed = min(support.values()) >= 0.2
            if not closed:
                continue
            perimeter = max(1.0, 2.0 * (candidate_width + candidate_height))
            if cv2.arcLength(contour, True) / perimeter < 0.72:
                continue
            if any(_intersection_ratio(box, control["box"]) > 0.8 for control in existing):
                continue

            band_mask = np.zeros((candidate_height, candidate_width), dtype=bool)
            band = max(3, min(7, round(min(candidate_width, candidate_height) * 0.07)))
            band_mask[:band, :] = True
            band_mask[-band:, :] = True
            band_mask[:, :band] = True
            band_mask[:, -band:] = True
            crop_foreground = foreground[y:y1, x:x1] > 0
            corner_radius = _outline_corner_radius(crop_foreground, band)
            boundary_pixels = image[y:y1, x:x1][band_mask & crop_foreground]
            if boundary_pixels.size == 0:
                continue
            boundary_bgr = strong_boundary_color(boundary_pixels)
            interior = image[
                min(y1, y + band) : max(y + band, y1 - band),
                min(x1, x + band) : max(x + band, x1 - band),
            ]
            interior_bgr = (
                np.median(interior.reshape(-1, 3), axis=0)
                if interior.size
                else page_background
            )
            candidate = {
                "kind": "button",
                "background": "#{:02X}{:02X}{:02X}".format(
                    int(round(interior_bgr[2])),
                    int(round(interior_bgr[1])),
                    int(round(interior_bgr[0])),
                ),
                "borderColor": "#{:02X}{:02X}{:02X}".format(
                    int(round(boundary_bgr[2])),
                    int(round(boundary_bgr[1])),
                    int(round(boundary_bgr[0])),
                ),
                "borderWidth": max(1, round(band * min(support.values()))),
                "box": box,
                "cornerRadius": corner_radius,
                "labelText": item.get("text", ""),
                "labelBox": [tx0, ty0, tx1, ty1],
                "boundaryEvidence": {**support, "closed": True},
                "source": "measured",
                "method": "closed-outline-around-ocr",
            }
            if any(
                _intersection_ratio(box, current["box"]) > 0.85
                for current in candidates
            ):
                continue
            candidates.append(candidate)
            break
    candidates.sort(key=lambda control: (control["box"][1], control["box"][0]))
    return candidates[:top]




def _luminance(bgr: Any) -> float:
    return float(0.299 * bgr[2] + 0.587 * bgr[1] + 0.114 * bgr[0])




def _hex_to_bgr(hex_color: str | None) -> tuple[int, int, int]:
    try:
        value = hex_color.lstrip("#")
        return tuple(int(value[i:i + 2], 16) for i in (4, 2, 0))
    except (ValueError, AttributeError):
        return (0, 0, 0)
