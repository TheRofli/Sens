"""Deterministic image comparison."""

from __future__ import annotations

from typing import Any

import numpy as np

from sight.ocr import load_cv




# --------------------------------------------------------------------------
# Compare: deterministic pixel diff between reference and candidate
# --------------------------------------------------------------------------


def compare_images(reference_path: str, candidate_path: str) -> dict[str, Any]:
    """Numerical visual difference: HSV delta, mismatch ratio, hot zones.

    Deterministic and local — no provider needed. Candidate is resized to
    the reference size first, so different resolutions still compare.
    """
    import cv2

    reference = load_cv(reference_path)
    candidate = load_cv(candidate_path)
    if candidate.shape[:2] != reference.shape[:2]:
        candidate = cv2.resize(
            candidate, (reference.shape[1], reference.shape[0]),
            interpolation=cv2.INTER_AREA,
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
                "meanDelta": round(float(score[y:y + h, x:x + w].mean()), 2),
            })
    return {
        "width": reference.shape[1],
        "height": reference.shape[0],
        "mismatchRatio": round(ratio, 4),
        "meanDelta": round(float(score.mean()), 2),
        "zones": zones,
    }
