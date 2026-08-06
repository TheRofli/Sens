"""Deterministic perception layers L0-L4 (colors, layout, objects, scene, texture)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from sight.ocr import WORKER_DIR




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
        r, g, b = (int(round(c)) for c in centers[index])
        zones.append({
            "hex": f"#{r:02X}{g:02X}{b:02X}",
            "ratio": round(float(counts[index]) / total, 4),
            "source": "measured",
        })
    return {
        "width": width,
        "height": height,
        "dominant": zones[:8],
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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, int(width * min_len_ratio)), 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, int(height * min_len_ratio))))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    horizontal = []
    for y in range(height):
        if int(h_lines[y, :].sum()) > 255 * width * min_len_ratio:
            horizontal.append(y)
    vertical = []
    for x in range(width):
        if int(v_lines[:, x].sum()) > 255 * height * min_len_ratio:
            vertical.append(x)
    return {
        "horizontal": _line_groups(horizontal),
        "vertical": _line_groups(vertical),
    }




def _line_groups(coords: list[int], tolerance: int = 3) -> list[int]:
    """Collapse thick lines (several adjacent rows/cols) to center rows."""
    groups = []
    for coord in coords:
        if groups and coord - groups[-1] <= tolerance:
            continue
        groups.append(coord)
    return groups




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
                "source": "measured",
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
        {"label": label, "confidence": round(float(score), 3), "source": "measured"}
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




def _glyph_metrics(image: Any, box: list[int]) -> dict[str, Any] | None:
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
    bg = float(np.median(patch))
    amp = float(np.percentile(np.abs(patch - bg), 90))
    if amp < 18:
        return None  # low contrast: not reliably readable text
    mask = np.abs(patch - bg) > 0.35 * amp
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
    cap = int(strong[-1] - strong[0] + 1)
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
    letters = sum(max(1, round(w / max(1.6, median_w))) for w in widths)
    avg_glyph = float(np.sum(widths)) / max(1, letters)
    font_size = cap / 0.73  # cap-based reference em
    width_em = avg_glyph / font_size if font_size > 0 else 0.0
    best = min(_KNOWN_FONTS, key=lambda f: abs(f[1] - width_em))
    custom = abs(best[1] - width_em) > 0.07
    return {
        "capHeight": int(cap),
        "avgGlyphWidth": round(avg_glyph, 1),
        "fontSize": int(round(font_size)),
        "widthEm": round(width_em, 2),
        "family": "custom" if custom else best[0],
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
            "source": "measured",
        })
    return out[:top]




def _luminance(bgr: Any) -> float:
    return float(0.299 * bgr[2] + 0.587 * bgr[1] + 0.114 * bgr[0])




def _hex_to_bgr(hex_color: str | None) -> tuple[int, int, int]:
    try:
        value = hex_color.lstrip("#")
        return tuple(int(value[i:i + 2], 16) for i in (4, 2, 0))
    except (ValueError, AttributeError):
        return (0, 0, 0)
