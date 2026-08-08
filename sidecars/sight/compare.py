"""Deterministic image comparison."""

from __future__ import annotations

import re
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
}




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
    reference_boxes = _layout_boxes(reference)
    candidate_boxes = _layout_boxes(candidate)
    if not reference_boxes and not candidate_boxes:
        return 1.0, reference_boxes, candidate_boxes
    if not reference_boxes or not candidate_boxes:
        return 0.0, reference_boxes, candidate_boxes
    scores = [max(_iou(box, other) for other in candidate_boxes) for box in reference_boxes]
    unmatched_penalty = min(len(reference_boxes), len(candidate_boxes)) / max(
        len(reference_boxes), len(candidate_boxes)
    )
    return float(np.mean(scores)) * unmatched_penalty, reference_boxes, candidate_boxes


def _normalized_text(path: str) -> str:
    text = " ".join(str(item.get("text", "")) for item in run_ocr(path))
    return " ".join(re.findall(r"[\w.-]+", text.casefold()))


def _foreground_mask(image: Any) -> Any:
    """Estimate visible content against the decoded image-border background."""
    pixels = image.astype(np.float32)
    border = np.concatenate(
        (pixels[0, :, :], pixels[-1, :, :], pixels[:, 0, :], pixels[:, -1, :]),
        axis=0,
    )
    background = np.median(border, axis=0)
    return np.linalg.norm(pixels - background, axis=2) > 18.0


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
    pixel_mask = np.any(absolute_bgr > 8, axis=2)
    pixel_mismatch = float(pixel_mask.mean())
    foreground_mask = np.logical_or(
        _foreground_mask(reference), _foreground_mask(candidate)
    )
    foreground_coverage = float(foreground_mask.mean())
    foreground_mismatch = (
        float(pixel_mask[foreground_mask].mean())
        if bool(foreground_mask.any())
        else 0.0
    )

    reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
    candidate_lab = cv2.cvtColor(candidate, cv2.COLOR_BGR2LAB).astype(np.float32)
    color_delta = np.linalg.norm(reference_lab - candidate_lab, axis=2)
    color_mean = float(color_delta.mean())

    reference_edges = cv2.Canny(reference, 60, 160) > 0
    candidate_edges = cv2.Canny(candidate, 60, 160) > 0
    edge_mismatch = float(np.logical_xor(reference_edges, candidate_edges).mean())

    reference_text = _normalized_text(reference_path)
    candidate_text = _normalized_text(candidate_path)
    text_similarity = SequenceMatcher(None, reference_text, candidate_text).ratio()
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
        mask_u8 = (mask * 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:6]:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 8 or h < 8:
                continue
            zones.append({
                "box": [x, y, x + w, y + h],
                "area": w * h,
                "areaRatio": round((w * h) / (reference_width * reference_height), 4),
                "meanDelta": round(float(score[y:y + h, x:x + w].mean()), 2),
                "signals": ["pixel", "color", "edge"],
            })
    metrics = {
        "pixel": {
            "mismatchRatio": round(pixel_mismatch, 4),
            "meanAbsoluteChannelDelta": round(float(absolute_bgr.mean()), 3),
            "method": "absolute-bgr-delta",
        },
        "foreground": {
            "mismatchRatio": round(foreground_mismatch, 4),
            "coverageRatio": round(foreground_coverage, 4),
            "method": "border-background-union-mask",
        },
        "color": {
            "meanLabDelta": round(color_mean, 3),
            "p95LabDelta": round(float(np.percentile(color_delta, 95)), 3),
            "method": "opencv-lab-distance",
        },
        "edge": {
            "mismatchRatio": round(edge_mismatch, 4),
            "method": "canny-xor",
        },
        "text": {
            "similarity": round(text_similarity, 4),
            "reference": reference_text,
            "candidate": candidate_text,
            "method": "rapidocr-sequence-similarity",
        },
        "layout": {
            "similarity": round(layout_similarity, 4),
            "referenceBlockCount": len(reference_boxes),
            "candidateBlockCount": len(candidate_boxes),
            "method": "contour-box-iou",
        },
    }
    similarity = (
        0.30 * (1.0 - pixel_mismatch)
        + 0.15 * (1.0 - min(1.0, color_mean / 50.0))
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
            "threshold": PASS_THRESHOLDS["foregroundMismatchRatio"],
            "passed": foreground_mismatch
            <= PASS_THRESHOLDS["foregroundMismatchRatio"],
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
        and foreground_mismatch <= 0.24
        and layout_similarity >= 0.70
        and largest_hot_region_ratio <= 0.08
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
            "method": "sens-multisignal-compare-v3",
        },
        # Compatibility projection retained for 1.x callers.
        "mismatchRatio": round(ratio, 4),
        "meanDelta": round(float(score.mean()), 2),
        "zones": zones,
    }
