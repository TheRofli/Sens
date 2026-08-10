"""Deterministic image comparison."""

from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any

import numpy as np

from sight.ocr import load_cv, run_ocr


PASS_THRESHOLDS = {
    "similarityScore": 0.88,
    "pixelMismatchRatio": 0.12,
    "foregroundMismatchRatio": 0.18,
    "layoutSimilarity": 0.80,
    "largestHotRegionRatio": 0.05,
    "largestMaterialHotRegionBoundingRatio": 0.08,
    "textSimilarity": 0.70,
}


def _material_hot_region_bounding_ratio(region: dict[str, Any]) -> float:
    """Gate broad dense defects without reviving sparse-box false failures."""
    signal = float(
        region.get("signalAreaRatio") or region.get("areaRatio") or 0.0
    )
    bounding = float(region.get("boundingAreaRatio") or 0.0)
    density = signal / max(1e-9, bounding)
    if signal < 0.025 or density < 0.25:
        return 0.0
    return bounding




# --------------------------------------------------------------------------
# Compare: deterministic pixel diff between reference and candidate
# --------------------------------------------------------------------------


def _layout_boxes(image: Any) -> list[list[int]]:
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width * height >= 100:
            boxes.append([x, y, x + width, y + height])
    return sorted(boxes, key=lambda box: -((box[2] - box[0]) * (box[3] - box[1])))[:20]


def _iou(left: list[int], right: list[int]) -> float:
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    intersection = (x1 - x0) * (y1 - y0)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (left_area + right_area - intersection)


def _layout_similarity(reference: Any, candidate: Any) -> tuple[float, list[list[int]], list[list[int]]]:
    import cv2

    reference_boxes = _layout_boxes(reference)
    candidate_boxes = _layout_boxes(candidate)
    height, width = reference.shape[:2]
    grid_width = 32
    grid_height = max(12, round(grid_width * height / max(1, width)))
    reference_occupancy = cv2.resize(
        _foreground_mask(reference).astype(np.float32),
        (grid_width, grid_height),
        interpolation=cv2.INTER_AREA,
    )
    candidate_occupancy = cv2.resize(
        _foreground_mask(candidate).astype(np.float32),
        (grid_width, grid_height),
        interpolation=cv2.INTER_AREA,
    )
    union = float(np.maximum(reference_occupancy, candidate_occupancy).sum())
    if union <= 1e-9:
        similarity = 1.0
    else:
        similarity = 1.0 - float(
            np.abs(reference_occupancy - candidate_occupancy).sum()
        ) / union
    return max(0.0, similarity), reference_boxes, candidate_boxes


def _extract_hot_regions(
    score: Any,
    mask: Any,
    reference_width: int,
    reference_height: int,
) -> list[dict[str, Any]]:
    """Return repair boxes while measuring only pixels that actually differ.

    A bounding rectangle is useful as a repair target, but its area is not a
    mismatch area. Nearby glyph strokes can create a large enclosing rectangle
    whose interior mostly matches. The completion gate therefore uses the
    signal area; the bounding area remains available for repair diagnostics.
    """
    import cv2

    mask_bool = np.asarray(mask, dtype=bool)
    if not bool(mask_bool.any()):
        return []
    mask_u8 = mask_bool.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    canvas_area = max(1, reference_width * reference_height)
    zones: list[dict[str, Any]] = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width < 8 or height < 8:
            continue
        contour_mask = np.zeros(mask_bool.shape, np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 1, thickness=cv2.FILLED)
        signal_mask = np.logical_and(mask_bool, contour_mask.astype(bool))
        signal_area = int(signal_mask.sum())
        if signal_area <= 0:
            continue
        bounding_area = int(width * height)
        contour_area = float(cv2.contourArea(contour))
        signal_ratio = round(signal_area / canvas_area, 4)
        zones.append(
            {
                "box": [x, y, x + width, y + height],
                # Compatibility fields now represent measured mismatch signal,
                # never the enclosing repair rectangle.
                "area": signal_area,
                "areaRatio": signal_ratio,
                "signalArea": signal_area,
                "signalAreaRatio": signal_ratio,
                "boundingArea": bounding_area,
                "boundingAreaRatio": round(bounding_area / canvas_area, 4),
                "contourArea": round(contour_area, 2),
                "contourAreaRatio": round(contour_area / canvas_area, 4),
                "meanDelta": round(float(np.asarray(score)[signal_mask].mean()), 2),
                "signals": ["pixel", "color", "edge"],
            }
        )
    return sorted(zones, key=lambda zone: zone["signalArea"], reverse=True)[:6]


def _normalized_text(path: str) -> str:
    text = " ".join(str(item.get("text", "")) for item in run_ocr(path))
    visual_latin = str.maketrans(
        {
            "а": "a",
            "в": "b",
            "с": "c",
            "е": "e",
            "н": "h",
            "к": "k",
            "м": "m",
            "о": "o",
            "р": "p",
            "т": "t",
            "х": "x",
            "у": "y",
            "і": "i",
            "ј": "j",
            "ѕ": "s",
            "ӏ": "l",
        }
    )
    normalized = text.casefold().translate(visual_latin)
    return " ".join(re.findall(r"[\w.-]+", normalized))


def _text_similarity(reference: str, candidate: str) -> tuple[float, str]:
    """Compare OCR text robustly when recognition fragments or reorders rows."""
    sequence = SequenceMatcher(None, reference, candidate).ratio()
    if not reference and not candidate:
        return 1.0, "empty-text-equality"
    reference_bigrams = Counter(
        reference[index : index + 2] for index in range(max(0, len(reference) - 1))
    )
    candidate_bigrams = Counter(
        candidate[index : index + 2] for index in range(max(0, len(candidate) - 1))
    )
    total = sum(reference_bigrams.values()) + sum(candidate_bigrams.values())
    dice = (
        2.0 * sum((reference_bigrams & candidate_bigrams).values()) / total
        if total
        else sequence
    )
    if dice > sequence:
        return dice, "visual-latin-character-bigram-dice"
    return sequence, "visual-latin-sequence-similarity"


def _foreground_mask(image: Any) -> Any:
    """Measure local visible detail without assuming one page background.

    Dashboards commonly contain a gray rail, white workspace, and several card
    fills.  A single border median labels an entire panel as foreground and can
    hide missing text and controls inside millions of uniform pixels.  Local
    contrast plus stable edges retains glyphs, rules, charts, and object
    boundaries while discarding broad flat surfaces of any colour.
    """
    import cv2

    pixels = image.astype(np.float32)
    local = cv2.GaussianBlur(pixels, (9, 9), 0)
    residual = np.max(np.abs(pixels - local), axis=2) > 8.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120) > 0
    return np.logical_or(residual, edges)


def _quantized_lab_palette(
    image: Any, *, color_step: int = 16, limit: int = 16
) -> tuple[Any, Any]:
    """Return a deterministic dominant palette independent of pixel position."""
    import cv2

    pixels = image.reshape(-1, 3).astype(np.int32)
    bins_per_channel = 256 // color_step
    keys = (
        (pixels[:, 0] // color_step) * bins_per_channel * bins_per_channel
        + (pixels[:, 1] // color_step) * bins_per_channel
        + (pixels[:, 2] // color_step)
    )
    values, counts = np.unique(keys, return_counts=True)
    order = np.argsort(-counts, kind="stable")[:limit]
    values = values[order]
    weights = counts[order].astype(np.float64)
    weights /= max(1.0, float(weights.sum()))
    blue = (values // (bins_per_channel * bins_per_channel)) * color_step
    green = ((values // bins_per_channel) % bins_per_channel) * color_step
    red = (values % bins_per_channel) * color_step
    centers = np.stack([blue, green, red], axis=1) + color_step // 2
    bgr = centers.clip(0, 255).astype(np.uint8).reshape(1, -1, 3)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    return lab, weights


def _palette_lab_distance(reference: Any, candidate: Any) -> float:
    """Symmetric weighted nearest-color distance between dominant palettes."""
    reference_palette, reference_weights = _quantized_lab_palette(reference)
    candidate_palette, candidate_weights = _quantized_lab_palette(candidate)
    distances = np.linalg.norm(
        reference_palette[:, None, :] - candidate_palette[None, :, :], axis=2
    )
    reference_distance = float(
        (distances.min(axis=1) * reference_weights).sum()
    )
    candidate_distance = float(
        (distances.min(axis=0) * candidate_weights).sum()
    )
    return (reference_distance + candidate_distance) / 2.0


def _strict_candidate_canvas(reference: Any, candidate: Any) -> Any:
    """Align a different-sized candidate without resampling it.

    Pixels outside the candidate are deliberately made maximally different
    from the reference so a missing strip remains visible in the diff. Pixels
    beyond the reference are represented by the separate dimension gate.
    """
    aligned = 255 - reference
    overlap_height = min(reference.shape[0], candidate.shape[0])
    overlap_width = min(reference.shape[1], candidate.shape[1])
    aligned[:overlap_height, :overlap_width] = candidate[
        :overlap_height, :overlap_width
    ]
    return aligned


def compare_images(
    reference_path: str, candidate_path: str, *, fit: str = "strict"
) -> dict[str, Any]:
    """Numerical visual difference with explicit reconstruction gates.

    Deterministic and local. Strict mode never resamples the candidate.
    ``fit=resize`` is an explicit compatibility view and cannot prove exact
    completion when it changes the decoded candidate pixels.
    """
    import cv2

    reference = load_cv(reference_path)
    candidate = load_cv(candidate_path)
    if fit not in {"strict", "resize"}:
        raise ValueError("fit must be 'strict' or 'resize'")
    reference_height, reference_width = reference.shape[:2]
    candidate_height, candidate_width = candidate.shape[:2]
    exact_dimensions = candidate.shape[:2] == reference.shape[:2]
    resampled = False
    if not exact_dimensions and fit == "resize":
        candidate = cv2.resize(
            candidate,
            (reference_width, reference_height),
            interpolation=cv2.INTER_AREA,
        )
        resampled = True
    elif not exact_dimensions:
        candidate = _strict_candidate_canvas(reference, candidate)
    absolute_bgr = cv2.absdiff(reference, candidate)
    raw_pixel_mask = np.any(absolute_bgr > 8, axis=2)
    raw_pixel_mismatch = float(raw_pixel_mask.mean())
    blurred_reference = cv2.GaussianBlur(reference, (5, 5), 0)
    blurred_candidate = cv2.GaussianBlur(candidate, (5, 5), 0)
    blur_pixel_mask = np.any(
        cv2.absdiff(blurred_reference, blurred_candidate) > 16,
        axis=2,
    )
    blur_pixel_mismatch = float(blur_pixel_mask.mean())
    pixel_mismatch = min(raw_pixel_mismatch, blur_pixel_mismatch)
    reference_foreground = _foreground_mask(reference)
    candidate_foreground = _foreground_mask(candidate)
    foreground_mask = np.logical_or(reference_foreground, candidate_foreground)
    foreground_coverage = float(foreground_mask.mean())
    tolerance_px = max(
        1, min(6, int(round(min(reference_width, reference_height) / 256.0)))
    )
    foreground_threshold = min(
        0.34,
        PASS_THRESHOLDS["foregroundMismatchRatio"]
        + 0.05 * max(0, tolerance_px - 2),
    )
    tolerance_kernel = np.ones(
        (tolerance_px * 2 + 1, tolerance_px * 2 + 1), np.uint8
    )
    reference_tolerance = cv2.dilate(
        reference_foreground.astype(np.uint8), tolerance_kernel, iterations=1
    ) > 0
    candidate_tolerance = cv2.dilate(
        candidate_foreground.astype(np.uint8), tolerance_kernel, iterations=1
    ) > 0
    unmatched_foreground = np.logical_or(
        np.logical_and(reference_foreground, ~candidate_tolerance),
        np.logical_and(candidate_foreground, ~reference_tolerance),
    )
    foreground_mismatch = (
        float(unmatched_foreground.sum()) / float(foreground_mask.sum())
        if bool(foreground_mask.any())
        else 0.0
    )

    reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
    candidate_lab = cv2.cvtColor(candidate, cv2.COLOR_BGR2LAB).astype(np.float32)
    color_delta = np.linalg.norm(reference_lab - candidate_lab, axis=2)
    color_mean = float(color_delta.mean())
    palette_color_delta = _palette_lab_distance(reference, candidate)
    palette_color_similarity = max(0.0, 1.0 - palette_color_delta / 50.0)

    reference_edges = cv2.Canny(reference, 60, 160) > 0
    candidate_edges = cv2.Canny(candidate, 60, 160) > 0
    edge_mismatch = float(np.logical_xor(reference_edges, candidate_edges).mean())

    reference_text = _normalized_text(reference_path)
    candidate_text = _normalized_text(candidate_path)
    text_similarity, text_similarity_method = _text_similarity(
        reference_text, candidate_text
    )
    layout_similarity, reference_boxes, candidate_boxes = _layout_similarity(
        reference, candidate
    )
    ref_hsv = cv2.cvtColor(reference, cv2.COLOR_BGR2HSV).astype(np.float32)
    cand_hsv = cv2.cvtColor(candidate, cv2.COLOR_BGR2HSV).astype(np.float32)
    delta = np.abs(ref_hsv - cand_hsv)
    # hue is circular: wrap 180 -> 0
    delta[:, :, 0] = np.minimum(delta[:, :, 0], 180.0 - delta[:, :, 0])
    # Hue is meaningless for neutral pixels (JPEG noise on white/gray
    # backgrounds); weight it by how saturated both pixels are.
    neutrality = np.minimum(ref_hsv[:, :, 1], cand_hsv[:, :, 1]) / 255.0
    score = (
        0.4 * delta[:, :, 0] * neutrality
        + 0.3 * delta[:, :, 1]
        + 0.3 * delta[:, :, 2]
    )
    # Adaptive threshold: at least 15 (above noise), at most 30 (so large
    # regions of change are never swallowed by their own percentile).
    threshold = min(max(15.0, float(np.percentile(score, 90))), 30.0)
    mask = score > threshold
    ratio = float(mask.mean())
    zones = []
    if ratio > 0.001:
        zones = _extract_hot_regions(
            score,
            mask,
            reference_width,
            reference_height,
        )
    metrics = {
        "pixel": {
            "mismatchRatio": round(pixel_mismatch, 4),
            "rawMismatchRatio": round(raw_pixel_mismatch, 4),
            "blurNormalizedMismatchRatio": round(blur_pixel_mismatch, 4),
            "meanAbsoluteChannelDelta": round(float(absolute_bgr.mean()), 3),
            "method": "minimum-raw-or-gaussian-5x5-bgr-delta",
        },
        "foreground": {
            "mismatchRatio": round(foreground_mismatch, 4),
            "coverageRatio": round(foreground_coverage, 4),
            "threshold": round(foreground_threshold, 4),
            "tolerancePx": tolerance_px,
            "toleranceRule": "round(min(width,height)/256), clamped 1..6",
            "method": "local-detail-mask-tolerant-xor",
        },
        "color": {
            "meanLabDelta": round(color_mean, 3),
            "p95LabDelta": round(float(np.percentile(color_delta, 95)), 3),
            "paletteMeanLabDelta": round(palette_color_delta, 3),
            "paletteSimilarity": round(palette_color_similarity, 4),
            "scoreMethod": "position-independent-quantized-lab-palette",
            "method": "aligned-opencv-lab-diagnostics-plus-palette-score",
        },
        "edge": {
            "mismatchRatio": round(edge_mismatch, 4),
            "method": "canny-xor",
        },
        "text": {
            "similarity": round(text_similarity, 4),
            "reference": reference_text,
            "candidate": candidate_text,
            "method": f"rapidocr-{text_similarity_method}",
        },
        "layout": {
            "similarity": round(layout_similarity, 4),
            "referenceBlockCount": len(reference_boxes),
            "candidateBlockCount": len(candidate_boxes),
            "grid": [32, max(12, round(32 * reference_height / max(1, reference_width)))],
            "method": "coarse-foreground-occupancy",
        },
    }
    similarity = (
        0.30 * (1.0 - pixel_mismatch)
        + 0.15 * palette_color_similarity
        + 0.20 * (1.0 - edge_mismatch)
        + 0.20 * text_similarity
        + 0.15 * layout_similarity
    )
    next_actions = []
    if zones:
        x0, y0, x1, y1 = zones[0]["box"]
        next_actions.append(
            {
                "tool": "sens_zoom",
                "reason": "Inspect the largest measured visual mismatch before the next repair.",
                "arguments": {
                    "region": {
                        "x": x0,
                        "y": y0,
                        "width": x1 - x0,
                        "height": y1 - y0,
                    }
                },
            }
        )
    similarity_score = round(max(0.0, min(1.0, similarity)), 4)
    largest_hot_region_ratio = zones[0]["areaRatio"] if zones else 0.0
    largest_material_hot_region_bounding_ratio = (
        _material_hot_region_bounding_ratio(zones[0]) if zones else 0.0
    )
    checks = [
        {
            "name": "dimensions_exact",
            "blockingReason": "dimension_mismatch",
            "actual": exact_dimensions,
            "expected": True,
            "passed": exact_dimensions,
        },
        {
            "name": "resampled_candidate",
            "actual": resampled,
            "expected": False,
            "passed": not resampled,
        },
        {
            "name": "similarity_minimum",
            "actual": similarity_score,
            "operator": ">=",
            "threshold": PASS_THRESHOLDS["similarityScore"],
            "passed": similarity_score >= PASS_THRESHOLDS["similarityScore"],
        },
        {
            "name": "pixel_mismatch_maximum",
            "actual": round(pixel_mismatch, 4),
            "operator": "<=",
            "threshold": PASS_THRESHOLDS["pixelMismatchRatio"],
            "passed": pixel_mismatch <= PASS_THRESHOLDS["pixelMismatchRatio"],
        },
        {
            "name": "foreground_mismatch_maximum",
            "actual": round(foreground_mismatch, 4),
            "operator": "<=",
            "threshold": round(foreground_threshold, 4),
            "passed": foreground_mismatch <= foreground_threshold,
        },
        {
            "name": "text_similarity_minimum",
            "actual": round(text_similarity, 4),
            "operator": ">=",
            "threshold": PASS_THRESHOLDS["textSimilarity"],
            "passed": text_similarity >= PASS_THRESHOLDS["textSimilarity"],
        },
        {
            "name": "layout_similarity_minimum",
            "actual": round(layout_similarity, 4),
            "operator": ">=",
            "threshold": PASS_THRESHOLDS["layoutSimilarity"],
            "passed": layout_similarity >= PASS_THRESHOLDS["layoutSimilarity"],
        },
        {
            "name": "largest_hot_region_maximum",
            "actual": largest_hot_region_ratio,
            "operator": "<=",
            "threshold": PASS_THRESHOLDS["largestHotRegionRatio"],
            "passed": largest_hot_region_ratio
            <= PASS_THRESHOLDS["largestHotRegionRatio"],
        },
        {
            "name": "largest_material_hot_region_bounding_maximum",
            "actual": round(largest_material_hot_region_bounding_ratio, 4),
            "operator": "<=",
            "threshold": PASS_THRESHOLDS[
                "largestMaterialHotRegionBoundingRatio"
            ],
            "passed": largest_material_hot_region_bounding_ratio
            <= PASS_THRESHOLDS["largestMaterialHotRegionBoundingRatio"],
        },
    ]
    blocking_reasons = [
        check.get("blockingReason", check["name"])
        for check in checks
        if not check["passed"]
    ]
    if all(check["passed"] for check in checks):
        verdict = "pass"
    elif (
        exact_dimensions
        and similarity_score >= 0.82
        and pixel_mismatch <= 0.15
        and foreground_mismatch <= min(0.40, foreground_threshold + 0.06)
        and text_similarity >= 0.60
        and layout_similarity >= 0.70
        and largest_hot_region_ratio <= 0.08
        and largest_material_hot_region_bounding_ratio <= 0.12
    ):
        verdict = "partial"
    else:
        verdict = "fail"
    required_action = None
    if not exact_dimensions:
        required_action = {
            "kind": "rerender_exact_dimensions",
            "referenceSize": {"width": reference_width, "height": reference_height},
            "candidateSize": {"width": candidate_width, "height": candidate_height},
            "reason": "Candidate dimensions must exactly match the immutable reference before visual completion can be evaluated.",
        }
    elif zones and verdict != "pass":
        required_action = {
            "kind": "repair_largest_hot_region",
            "region": zones[0]["box"],
            "areaRatio": zones[0]["areaRatio"],
            "reason": "Repair the largest measured mismatch, render again at the same viewport, then rerun sens_compare.",
        }
    return {
        "width": reference.shape[1],
        "height": reference.shape[0],
        "completionScope": "visual-only",
        "visualPass": verdict == "pass",
        "webCompletionWarning": "For screenshot-to-web work, visualPass is insufficient; sens_review must also return webPass=true.",
        "dimensions": {
            "reference": {"width": reference_width, "height": reference_height},
            "candidate": {"width": candidate_width, "height": candidate_height},
            "exactMatch": exact_dimensions,
            "aspectRatioDelta": round(
                abs(
                    reference_width / max(1, reference_height)
                    - candidate_width / max(1, candidate_height)
                ),
                6,
            ),
            "fit": fit,
            "resampled": resampled,
        },
        "verdict": verdict,
        "canComplete": verdict == "pass",
        "blockingReasons": blocking_reasons,
        "requiredAction": required_action,
        "acceptance": {
            "policy": "sens-reconstruction-v1",
            "checks": checks,
        },
        "similarityScore": similarity_score,
        "metrics": metrics,
        "hotRegions": zones,
        "nextActions": next_actions,
        "provenance": {
            "source": "measured",
            "method": "sens-multisignal-compare-v6",
        },
        # Compatibility projection retained for 1.x callers.
        "mismatchRatio": round(ratio, 4),
        "meanDelta": round(float(score.mean()), 2),
        "zones": zones,
    }
