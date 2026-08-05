"""Persistent NDJSON adapter from the Sens broker to the local vision stack.

Layers implemented here (all local, CPU, no network):
  L0 - global overview: dominant colors (k-means), grayscale stats
  L1 - text: RapidOCR (cyrillic + EN) with boxes and confidence
  L2 - structure: OpenCV layout blocks (morphology + contours)
  L3 - objects (YOLOv8n) + zero-shot scene classification (CLIP ViT-B-32)
  L4 - attention map: 8x8 grid of text density + local contrast -> hot zones
  L5 - cross-layer verification: text blocks, graphic blocks, labeled
       objects, attention coverage and layer conflicts (deterministic)

Operations: see (full dump), read (text only), locate (deterministic text
search over the dump), zoom (region re-analysis), inspect (region or target
-> focused re-analysis).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

WORKER_DIR = Path(__file__).resolve().parent
CYRILLIC_CONFIG_NAME = "rapidocr-cyrillic.yaml"
CACHE_TTL_SECONDS = 7 * 24 * 3600

# --------------------------------------------------------------------------
# RapidOCR setup (cyrillic rec model)
# --------------------------------------------------------------------------


def ensure_ocr_config() -> str:
    target = WORKER_DIR / CYRILLIC_CONFIG_NAME
    if target.exists():
        return str(target)
    try:
        import rapidocr
    except ImportError as exc:
        raise RuntimeError(
            "Local vision requires rapidocr. Run: pip install rapidocr"
        ) from exc
    default = Path(rapidocr.__file__).parent / "config.yaml"
    text = default.read_text(encoding="utf-8")
    text = re.sub(
        r'(Rec:\n    engine_type: "onnxruntime"\n    lang_type: )"ch"',
        r'\1"cyrillic"',
        text,
    )
    text = re.sub(
        r'(Rec:\n    engine_type: "onnxruntime"\n    lang_type: "cyrillic"\n    model_type: )"small"',
        r'\1"mobile"',
        text,
    )
    text = re.sub(
        r'(Rec:\n    engine_type: "onnxruntime"\n    lang_type: "cyrillic"\n    model_type: "mobile"\n    ocr_version: )"PP-OCRv6"',
        r'\1"PP-OCRv4"',
        text,
    )
    target.write_text(text, encoding="utf-8")
    return str(target)


_ocr_engine: Any = None


def ocr_engine() -> Any:
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr import RapidOCR

        _ocr_engine = RapidOCR(config_path=ensure_ocr_config())
    return _ocr_engine


def run_ocr(image_path: str) -> list[dict[str, Any]]:
    out = ocr_engine()(image_path)
    items: list[dict[str, Any]] = []
    boxes = out.boxes if out.boxes is not None else []
    txts = out.txts if out.txts is not None else []
    scores = out.scores if out.scores is not None else []
    for box, txt, score in zip(boxes, txts, scores):
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        items.append({
            "text": str(txt).strip(),
            "box": [round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys))],
            "confidence": round(float(score), 3),
        })
    return [item for item in items if item["text"]]


# --------------------------------------------------------------------------
# OpenCV layers: colors + layout blocks
# --------------------------------------------------------------------------


def load_cv(image_path: str) -> Any:
    import cv2

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"cannot decode image: {image_path}")
    return image


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


# --------------------------------------------------------------------------
# L5: cross-layer verification (deterministic reconciliation)
# --------------------------------------------------------------------------

# Scene labels that imply a text-bearing document/UI surface.
_TEXT_SURFACE_HINTS = (
    "screenshot",
    "interface",
    "web page",
    "dashboard",
    "settings",
    "form",
    "document",
    "invoice",
    "spreadsheet",
    "slide",
    "editor",
    "chat",
    "dialog",
    "player",
)


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


def cross_verify(dump: dict[str, Any], overlap: float = 0.3) -> dict[str, Any]:
    """Reconcile OCR / layout / objects / attention into confirmed structure.

    Every claim is deterministic and cites the layers that support it, so a
    text-only host model can trust layout statements without another vision
    pass.
    """
    ocr_items = dump.get("ocr", [])
    layout = dump.get("layout", [])
    objects = dump.get("objects", [])
    attention = dump.get("attention", [])
    scene = dump.get("scene", [])

    def overlapped_by_ocr(box: list[int]) -> bool:
        return any(
            _intersection_ratio(box, item["box"]) >= overlap for item in ocr_items
        )

    text_blocks = [
        {"box": block["box"], "area": block["area"]}
        for block in layout
        if overlapped_by_ocr(block["box"])
    ]
    graphic_blocks = [
        {"box": block["box"], "area": block["area"]}
        for block in layout
        if not overlapped_by_ocr(block["box"])
    ]

    labeled_objects = []
    for obj in objects:
        labels = [
            item["text"]
            for item in ocr_items
            if _intersection_ratio(obj["box"], item["box"]) >= overlap
        ]
        if labels:
            labeled_objects.append(
                {
                    "class": obj["class"],
                    "box": obj["box"],
                    "confidence": obj["confidence"],
                    "labels": labels[:3],
                }
            )

    attention_with_text = sum(
        1 for zone in attention if overlapped_by_ocr(zone["box"])
    )
    attention_coverage = (
        round(attention_with_text / len(attention), 3) if attention else 0.0
    )

    conflicts: list[dict[str, Any]] = []
    scene_label = scene[0]["label"] if scene else ""
    is_text_surface = any(hint in scene_label for hint in _TEXT_SURFACE_HINTS)
    if is_text_surface and not ocr_items:
        conflicts.append({
            "kind": "scene_says_text_surface_but_no_text",
            "detail": f"scene '{scene_label}' implies readable text, OCR found none",
            "source": "inferred",
        })
    if is_text_surface and not layout:
        conflicts.append({
            "kind": "scene_says_structured_surface_but_no_blocks",
            "detail": f"scene '{scene_label}' implies structure, layout found no blocks",
            "source": "inferred",
        })
    if objects and scene_label and not any(
        hint in scene_label for hint in ("photograph", "photo", "picture", "scene", "view", "park", "street", "landscape", "beach", "food", "portrait", "wildlife", "garden", "macro", "nature", "city")
    ):
        conflicts.append({
            "kind": "objects_in_non_photo_scene",
            "detail": f"scene '{scene_label}' but {len(objects)} physical objects detected",
            "source": "inferred",
        })

    # Visual QA: glued sections and broken vertical rhythm (UI review).
    for gap in dump.get("gaps", []):
        if gap["touching"]:
            conflicts.append({
                "kind": "sections_touch_without_gap",
                "detail": (
                    f"two sections touch along {gap['axis']} with a {gap['px']}px gap "
                    f"(boxes {gap['boxes']})"
                ),
                "source": "measured",
            })
    for issue in dump.get("design", {}).get("issues", []):
        if issue["kind"] in ("text_clipped_at_frame", "text_overflows_section"):
            conflicts.append({
                "kind": "text_overflow",
                "detail": issue["detail"],
                "source": "measured",
            })
        elif issue["kind"] == "low_text_contrast":
            conflicts.append({
                "kind": "low_text_contrast",
                "detail": issue["detail"],
                "source": "measured",
            })
        elif issue["kind"] == "uneven_card_heights":
            conflicts.append({
                "kind": "card_alignment",
                "detail": issue["detail"],
                "source": "measured",
            })
    y_gaps = [g for g in dump.get("gaps", []) if g["axis"] == "y"]
    if len(y_gaps) >= 2:
        values = [g["px"] for g in y_gaps]
        spread = max(values) - min(values)
        if spread > 10 and any(v == 0 for v in values) and any(v > 10 for v in values):
            conflicts.append({
                "kind": "inconsistent_section_rhythm",
                "detail": f"vertical gaps between sections vary: {values}px",
                "source": "measured",
            })

    return {
        "textBlocks": text_blocks[:20],
        "graphicBlocks": graphic_blocks[:20],
        "labeledObjects": labeled_objects[:20],
        "attentionTextCoverage": attention_coverage,
        "conflicts": conflicts,
    }


# --------------------------------------------------------------------------
# Dump cache: content-addressed, deterministic, TTL-bounded
# --------------------------------------------------------------------------

_last_cache_cleanup: float = 0.0
# Bump when the dump schema changes so stale dumps (e.g. without gaps,
# design QA or section style) are not served from cache.
CACHE_SCHEMA_VERSION = "qa7"


def cache_root() -> Path:
    """Cache directory for analysis dumps (overridable via SENS_CACHE_DIR)."""
    if root := os.environ.get("SENS_CACHE_DIR"):
        return Path(root) / "sight"
    if local := os.environ.get("LOCALAPPDATA"):
        return Path(local) / "Sens" / "cache" / "sight"
    return Path.home() / ".cache" / "sens" / "sight"


def cache_key(image_path: str, region: dict[str, int] | None) -> str:
    digest = hashlib.sha256()
    with open(image_path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    region_key = "full"
    if region is not None:
        region_key = "{x}x{y}x{w}x{h}".format(**region)
    return f"{CACHE_SCHEMA_VERSION}-{digest.hexdigest()[:32]}-{region_key}.json"


def read_cache(key: str) -> dict[str, Any] | None:
    path = cache_root() / key
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if time.time() - payload.get("ts", 0) > CACHE_TTL_SECONDS:
        return None
    return payload.get("dump")


def write_cache(key: str, dump: dict[str, Any]) -> None:
    directory = cache_root()
    try:
        directory.mkdir(parents=True, exist_ok=True)
        payload = {"ts": time.time(), "dump": dump}
        fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
        os.close(fd)
        Path(tmp_path).write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(tmp_path, directory / key)
    except OSError:
        # Cache must never break analysis; failures degrade to a miss.
        return
    cleanup_cache(directory)


def cleanup_cache(directory: Path, interval: float = 3600.0) -> None:
    """Remove expired entries at most once per `interval` seconds."""
    global _last_cache_cleanup
    now = time.time()
    if now - _last_cache_cleanup < interval:
        return
    _last_cache_cleanup = now
    try:
        for path in directory.glob("*.json"):
            if now - path.stat().st_mtime > CACHE_TTL_SECONDS:
                path.unlink(missing_ok=True)
    except OSError:
        return


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


# --------------------------------------------------------------------------
# L5b: design QA — overflow, contrast, uneven cards, alignment
# --------------------------------------------------------------------------


def _contrast_of(box: list[int], image: Any) -> float:
    """Text-to-background contrast (0..1) inside an OCR box.

    Uses the 5th/95th luminance percentiles: text pixels vs background.
    """
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    x0, y0, x1, y1 = box
    patch = gray[max(0, y0):y1, max(0, x0):x1]
    if patch.size < 16:
        return 1.0
    lo = float(np.percentile(patch, 5))
    hi = float(np.percentile(patch, 95))
    return round((hi - lo) / 255.0, 3)


def design_qa(image: Any, blocks: list[dict[str, Any]], ocr_items: list[dict[str, Any]]) -> dict[str, Any]:
    """Deterministic design review: overflow, contrast, uneven/misaligned cards."""
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    issues: list[dict[str, Any]] = []
    width = image.shape[1]
    height = image.shape[0]
    sections = [
        b for b in blocks
        if b.get("kind") != "texture" and b["area"] >= 0.02 * max(1, blocks[0]["area"])
    ] if blocks else []

    # 1. Text overflowing its containing section or the frame edge.
    for item in ocr_items:
        box = item["box"]
        for edge, limit in (("left", 0), ("top", 0), ("right", width), ("bottom", height)):
            if (edge in ("left", "top") and box[0 if edge == "left" else 1] < limit - 2) or (
                edge in ("right", "bottom") and box[2 if edge == "right" else 3] > limit + 2
            ):
                issues.append({
                    "kind": "text_clipped_at_frame",
                    "detail": f"text {item['text']!r} is cut at the {edge} edge",
                    "box": box,
                    "source": "measured",
                })
                break
        else:
            for section in sections:
                sx0, sy0, sx1, sy1 = section["box"]
                inside = (
                    box[0] >= sx0 - 2 and box[1] >= sy0 - 2
                    and box[2] <= sx1 + 2 and box[3] <= sy1 + 2
                )
                if inside:
                    break
                overlap_w = min(box[2], sx1) - max(box[0], sx0)
                overlap_h = min(box[3], sy1) - max(box[1], sy0)
                if overlap_w <= 0 or overlap_h <= 0:
                    continue
                box_area = (box[2] - box[0]) * (box[3] - box[1])
                if (overlap_w * overlap_h) > 0.5 * box_area:
                    if box[2] > sx1 + 4 or box[3] > sy1 + 4 or box[0] < sx0 - 4 or box[1] < sy0 - 4:
                        issues.append({
                            "kind": "text_overflows_section",
                            "detail": f"text {item['text']!r} overflows its section (section {sx0},{sy0},{sx1},{sy1})",
                            "box": box,
                            "source": "measured",
                        })
                    break

    # 2. Low text contrast.
    for item in ocr_items:
        contrast = _contrast_of(item["box"], image)
        if contrast < 0.18:
            issues.append({
                "kind": "low_text_contrast",
                "detail": f"text {item['text']!r} has {contrast:.2f} contrast against its background",
                "box": item["box"],
                "source": "measured",
            })

    # 3. Uneven card sizes in a row / misaligned edges.
    rows: list[list[dict[str, Any]]] = []
    for section in sections:
        placed = False
        for row in rows:
            other = row[0]["box"]
            overlap_y = min(section["box"][3], other[3]) - max(section["box"][1], other[1])
            if overlap_y > 0.5 * min(section["box"][3] - section["box"][1], other[3] - other[1]):
                row.append(section)
                placed = True
                break
        if not placed:
            rows.append([section])
    for row in rows:
        if len(row) < 2:
            continue
        heights = {b["box"][3] - b["box"][1] for b in row}
        if len(heights) > 1 and max(heights) - min(heights) > 8:
            issues.append({
                "kind": "uneven_card_heights",
                "detail": f"cards in a row have different heights: {sorted(heights)}px",
                "boxes": [b["box"] for b in row],
                "source": "measured",
            })

    # 4. Texts sharing a visual column should share a left edge. The edge
    # is measured on the glyph core (first dark column), not the OCR box,
    # so OCR padding noise does not trigger false alarms.
    def _glyph_left(box: list[int]) -> int:
        patch = gray[max(0, box[1]):box[3], max(0, box[0]):box[2]]
        if patch.size < 16:
            return box[0]
        dark = patch < min(float(np.percentile(patch, 10)) + 60, 210)
        cols = np.where(dark.any(axis=0))[0]
        return box[0] + int(cols[0]) if len(cols) else box[0]

    ranked = sorted(
        ocr_items,
        key=lambda item: (_glyph_left(item["box"]), item["box"][1]),
    )
    columns: list[list[dict[str, Any]]] = []
    for item in ranked:
        edge = _glyph_left(item["box"])
        placed = False
        for column in columns:
            anchor = column[0]
            if abs(edge - _glyph_left(anchor["box"])) > 10:
                continue
            overlap_y = min(item["box"][3], anchor["box"][3]) - max(item["box"][1], anchor["box"][1])
            span = min(item["box"][3] - item["box"][1], anchor["box"][3] - anchor["box"][1])
            if overlap_y <= 0.3 * span:
                column.append(item)
                placed = True
                break
        if not placed:
            columns.append([item])
    for column in columns:
        if len(column) < 2:
            continue
        edges = [_glyph_left(item["box"]) for item in column]
        spread = max(edges) - min(edges)
        if spread > 5:
            issues.append({
                "kind": "text_left_edges_misaligned",
                "detail": (
                    f"texts sharing a column have left edges {sorted(edges)}px apart "
                    f"(spread {spread}px)"
                ),
                "boxes": [item["box"] for item in column],
                "source": "measured",
            })

    return {"issues": issues[:16]}


# --------------------------------------------------------------------------
# L5c: section style extraction — colors, radius, padding, font size
# --------------------------------------------------------------------------


def _color_dist(left: Any, right: Any) -> float:
    """Mean per-channel absolute distance between two BGR pixels."""
    return float(
        abs(int(left[0]) - int(right[0]))
        + abs(int(left[1]) - int(right[1]))
        + abs(int(left[2]) - int(right[2]))
    ) / 3.0


def _border_of(image: Any, box: list[int], bg_bgr: Any) -> dict[str, Any] | None:
    """Detect a uniform outline (border) around a section.

    Scans a band around each edge (3px either side, since morphology may
    report the block just inside its own outline) for pixels that differ
    from both the section background and the surrounding background;
    reports color and thickness when a clear majority of every edge
    carries the same stroke.
    """
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    if w < 14 or h < 14:
        return None
    # Background outside the box: a thin ring 6..8px away (outside any
    # stroke that sits on the reported edge).
    outer: list[Any] = []
    for px in range(max(0, x0 - 8), min(image.shape[1], x1 + 8)):
        for py in (max(0, y0 - 8), max(0, y0 - 6), min(image.shape[0] - 1, y1 + 5), min(image.shape[0] - 1, y1 + 7)):
            if (px < x0 - 4 or px >= x1 + 4) and (py < y0 - 4 or py >= y1 + 4):
                outer.append(image[py, px])
    for py in range(max(0, y0 - 8), min(image.shape[0], y1 + 8)):
        for px in (max(0, x0 - 8), max(0, x0 - 6), min(image.shape[1] - 1, x1 + 5), min(image.shape[1] - 1, x1 + 7)):
            if (px < x0 - 4 or px >= x1 + 4) and (py < y0 - 4 or py >= y1 + 4):
                outer.append(image[py, px])
    if not outer:
        return None
    outer_bgr = np.median(np.asarray(outer, dtype=np.float32).reshape(-1, 3), axis=0)

    edges = {
        "top": ([(x, y0) for x in range(x0 + w // 5, x0 + 4 * w // 5)], (0, 1)),
        "bottom": ([(x, y1 - 1) for x in range(x0 + w // 5, x0 + 4 * w // 5)], (0, -1)),
        "left": ([(x0, y) for y in range(y0 + h // 5, y0 + 4 * h // 5)], (1, 0)),
        "right": ([(x1 - 1, y) for y in range(y0 + h // 5, y0 + 4 * h // 5)], (-1, 0)),
    }
    stroke_pixels: list[Any] = []
    thicknesses: list[int] = []
    for name, (points, (dx, dy)) in edges.items():
        hits = 0
        for px, py in points:
            found = None
            for offset in range(-3, 4):
                qx, qy = px + offset * dx, py + offset * dy
                if not (0 <= qx < image.shape[1] and 0 <= qy < image.shape[0]):
                    continue
                pixel = image[qy, qx]
                if _color_dist(pixel, bg_bgr) > 30 and _color_dist(pixel, outer_bgr) > 12:
                    found = (qx, qy)
                    break
            if found is None:
                continue
            hits += 1
            stroke_pixels.append(image[found[1], found[0]])
            thickness = 1
            for direction in (-1, 1):
                qx, qy = found
                while True:
                    qx += direction * dx
                    qy += direction * dy
                    if not (0 <= qx < image.shape[1] and 0 <= qy < image.shape[0]):
                        break
                    if _color_dist(image[qy, qx], bg_bgr) > 30 and _color_dist(image[qy, qx], outer_bgr) > 12:
                        thickness += 1
                    else:
                        break
            thicknesses.append(thickness)
        if hits < 0.45 * len(points):
            return None
    if len(stroke_pixels) < 8:
        return None
    color = np.median(np.asarray(stroke_pixels, dtype=np.float32).reshape(-1, 3), axis=0)
    return {
        "color": "#{:02X}{:02X}{:02X}".format(int(color[2]), int(color[1]), int(color[0])),
        "width": round(float(np.median(thicknesses)), 1) if thicknesses else 1,
    }


def _corner_radius(image: Any, x0: int, y0: int, x1: int, y1: int) -> int:
    """Estimate a section's border-radius by scanning its corners.

    For each corner, walk along the edge until a pixel differing from the
    section background appears — that distance approximates the radius.
    """
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inner = gray[y0 + 5:y1 - 5, x0 + 5:x1 - 5]
    if inner.size == 0:
        return 0
    bg = float(np.median(inner))
    radiuses = []
    corners = [
        (x0, y0, 1, 0),   # top-left, scan right
        (x0, y0, 0, 1),   # top-left, scan down
        (x1, y0, -1, 0),  # top-right, scan left
        (x0, y1, 1, 0),   # bottom-left, scan right
    ]
    for cx, cy, dx, dy in corners:
        for step in range(1, 60):
            px = cx + step * dx
            py = cy + step * dy
            if px < 0 or py < 0 or px >= gray.shape[1] or py >= gray.shape[0]:
                break
            if abs(float(gray[py, px]) - bg) > 40:
                radiuses.append(step)
                break
    return round(float(np.median(radiuses))) if radiuses else 0


def _section_padding(box: list[int], ocr_items: list[dict[str, Any]]) -> dict[str, int] | None:
    """Distance from each section edge to the nearest text inside it."""
    x0, y0, x1, y1 = box
    inside = [
        item["box"] for item in ocr_items
        if item["box"][0] > x0 and item["box"][1] > y0
        and item["box"][2] < x1 and item["box"][3] < y1
    ]
    if not inside:
        return None
    return {
        "top": min(b[1] for b in inside) - y0,
        "bottom": y1 - max(b[3] for b in inside),
        "left": min(b[0] for b in inside) - x0,
        "right": x1 - max(b[2] for b in inside),
    }


def section_style(image: Any, blocks: list[dict[str, Any]], ocr_items: list[dict[str, Any]], top: int = 6) -> list[dict[str, Any]]:
    """Per-section design tokens: bg/text color, corner radius, padding, font size."""
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sections = [
        b for b in blocks
        if b.get("kind") != "texture" and b["area"] >= 0.02 * max(1, blocks[0]["area"])
    ][:top] if blocks else []
    result = []
    for section in sections:
        x0, y0, x1, y1 = section["box"]
        if x1 - x0 < 40 or y1 - y0 < 40:
            continue
        inner = image[y0 + 5:y1 - 5, x0 + 5:x1 - 5]
        if inner.size == 0:
            continue
        bg_bgr = np.median(inner.reshape(-1, 3), axis=0)
        texts = [
            item for item in ocr_items
            if item["box"][0] > x0 and item["box"][1] > y0
            and item["box"][2] < x1 and item["box"][3] < y1
        ]
        text_color_bgr = None
        if texts:
            # Text color: median of the darkest pixels inside the OCR box.
            box = texts[0]["box"]
            patch = gray[max(0, box[1]):box[3], max(0, box[0]):box[2]]
            dark_mask = patch < np.percentile(patch, 20)
            pixels = image[max(0, box[1]):box[3], max(0, box[0]):box[2]]
            if int(dark_mask.sum()) > 10:
                text_color_bgr = np.median(pixels[dark_mask], axis=0)
        font_sizes = [item["box"][3] - item["box"][1] for item in texts]
        border = _border_of(image, section["box"], bg_bgr)
        result.append({
            "box": section["box"],
            "background": "#{:02X}{:02X}{:02X}".format(int(bg_bgr[2]), int(bg_bgr[1]), int(bg_bgr[0])),
            "textColor": (
                "#{:02X}{:02X}{:02X}".format(int(text_color_bgr[2]), int(text_color_bgr[1]), int(text_color_bgr[0]))
                if text_color_bgr is not None else None
            ),
            "borderColor": border["color"] if border else None,
            "borderWidth": border["width"] if border else None,
            "cornerRadius": _corner_radius(image, x0, y0, x1, y1),
            "padding": _section_padding(section["box"], ocr_items),
            "fontSize": round(float(np.median(font_sizes))) if font_sizes else None,
        })
    return result


def _split_row_controls(image: Any, block: dict[str, Any]) -> list[list[int]]:
    """Split a wide, low layout block (a row of buttons/pills) into items.

    Colored columns (saturated outlines/accents) delimit the controls: each
    item spans from one outline to the next when the gap between them looks
    like a pill body (20..140px). Falls back to gray-level gaps when the
    row has no saturated columns.
    """
    import cv2

    x0, y0, x1, y1 = block["box"]
    w, h = x1 - x0, y1 - y0
    if w <= 120 or h > 70 or h < 16:
        return []
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Scan a little outside the block: contour blocks often sit just inside
    # their own outline, so the left/right strokes live at x0-1..x0-6.
    ex0 = max(0, x0 - 6)
    ex1 = min(image.shape[1], x1 + 6)
    inner_hsv = hsv[y0 + 2:y1 - 2, ex0:ex1]
    saturated = (inner_hsv[:, :, 1] > 60).mean(axis=0)
    active = saturated > 0.05
    groups: list[tuple[int, int]] = []
    run_start = None
    for col in range(inner_hsv.shape[1]):
        if active[col]:
            if run_start is None:
                run_start = col
        elif run_start is not None:
            groups.append((ex0 + run_start, ex0 + col))
            run_start = None
    if run_start is not None:
        groups.append((ex0 + run_start, ex0 + inner_hsv.shape[1]))

    items: list[list[int]] = []
    if groups:
        index = 0
        while index < len(groups):
            start, end = groups[index]
            # Pair an outline with the next one if the gap between them is
            # a plausible pill body; otherwise the group is a lone accent.
            if index + 1 < len(groups):
                next_start, next_end = groups[index + 1]
                gap = next_start - end
                if 20 <= gap <= 140:
                    items.append([start, y0, next_end, y1])
                    index += 2
                    continue
            index += 1
    if not items:
        # Fallback: gray-level gaps (plain text rows split by background).
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        inner = gray[y0 + 2:y1 - 2, x0:x1]
        if inner.size == 0:
            return []
        bg = float(np.median(inner))
        col_frac = (np.abs(inner - bg) > 25).mean(axis=0)
        bounds = [x0]
        run_start = None
        for col in range(w):
            if col_frac[col] < 0.05:
                if run_start is None:
                    run_start = col
            elif run_start is not None:
                if col - run_start >= 4:
                    bounds.append(x0 + (run_start + col) // 2)
                run_start = None
        if run_start is not None and w - run_start >= 4:
            bounds.append(x0 + (run_start + w) // 2)
        bounds.append(x1)
        items = [
            [start, y0, end, y1]
            for start, end in zip(bounds, bounds[1:])
            if end - start >= 20
        ]
    return [item for item in items if item[2] - item[0] >= 20]


def control_style(image: Any, blocks: list[dict[str, Any]], top: int = 12) -> list[dict[str, Any]]:
    """Design tokens for small controls (buttons, pills): bg, border, radius.

    Layout blocks below the section threshold (14..120px) are typically
    buttons; wide low blocks are split into their row items first. Reports
    fill, outline and corner rounding so a host model can restyle them
    without another vision pass.
    """
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    controls = []
    for block in blocks:
        x0, y0, x1, y1 = block["box"]
        w, h = x1 - x0, y1 - y0
        candidates = [[x0, y0, x1, y1]]
        if w > 120 and h <= 70:
            candidates = _split_row_controls(image, block)
        for x0, y0, x1, y1 in candidates:
            w, h = x1 - x0, y1 - y0
            if w < 14 or h < 14 or w > 160 or h > 160:
                continue
            inner = image[y0 + 3:y1 - 3, x0 + 3:x1 - 3]
            if inner.size == 0:
                continue
            bg_bgr = np.median(inner.reshape(-1, 3), axis=0)
            border = _border_of(image, [x0, y0, x1, y1], bg_bgr)
            controls.append({
                "box": [x0, y0, x1, y1],
                "background": "#{:02X}{:02X}{:02X}".format(int(bg_bgr[2]), int(bg_bgr[1]), int(bg_bgr[0])),
                "borderColor": border["color"] if border else None,
                "borderWidth": border["width"] if border else None,
                "cornerRadius": _corner_radius(image, x0, y0, x1, y1),
                "source": "measured",
            })
    controls.sort(key=lambda c: -(c["box"][2] - c["box"][0]) * (c["box"][3] - c["box"][1]))
    return controls[:top]


def _classify_icon(mask: Any, w: int, h: int, fill: float) -> str | None:
    """Shape classification for a small connected component."""
    if fill < 0.7:
        row_sums = mask.sum(axis=1)
        col_sums = mask.sum(axis=0)
        if (
            int(row_sums.max()) > 0.7 * w
            and int(col_sums.max()) > 0.7 * h
        ):
            return "cross"
    top = mask[:max(1, h // 3), :]
    bottom = mask[min(h - 1, 2 * h // 3):, :]
    left = mask[:, :max(1, w // 3)]
    right = mask[:, min(w - 1, 2 * w // 3):]
    if h >= 0.8 * w:
        if bottom.sum() > 2.2 * top.sum() and bottom.sum() > 0.2 * mask.sum():
            return "arrow_down"
        if top.sum() > 2.2 * bottom.sum() and top.sum() > 0.2 * mask.sum():
            return "arrow_up"
    if w >= 0.8 * h:
        if right.sum() > 2.2 * left.sum() and right.sum() > 0.2 * mask.sum():
            return "arrow_right"
        if left.sum() > 2.2 * right.sum() and left.sum() > 0.2 * mask.sum():
            return "arrow_left"
    if fill > 0.62 and 0.6 <= w / max(1, h) <= 1.6:
        return "circle"
    if fill < 0.2 and (w > 10 or h > 10):
        return "outline"
    return None


def control_icons(
    image: Any,
    blocks: list[dict[str, Any]],
    ocr_items: list[dict[str, Any]],
    top: int = 8,
) -> list[dict[str, Any]]:
    """Small graphic glyphs (cross, arrows, dots) inside control blocks.

    Colored/dark connected components of 4..48px that do not overlap OCR
    text are classified by shape: cross, directional arrow, circle, ring.
    Text strokes are excluded via the OCR overlap check.
    """
    import cv2

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    colored = (hsv[:, :, 1] > 70) | (hsv[:, :, 2] < 120)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(colored.astype(np.uint8), 8)
    ocr_boxes = [item["box"] for item in ocr_items]
    icons = []
    for index in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[index])
        if area < 9 or area > 2304:
            continue
        if w < 4 or h < 4 or w > 48 or h > 48:
            continue
        mask = labels[y:y + h, x:x + w] == index
        fill = area / max(1, w * h)
        kind = _classify_icon(mask, w, h, fill)
        if kind is None:
            continue
        # A straight cross is a glyph (plus icon), not a letter: OCR tends
        # to misread it as "х"/"t". Crosses covered only by a lone OCR box
        # (a misread glyph, <40px wide) stay; crosses inside a real text
        # line (wide OCR box) are letters and are dropped.
        if kind == "cross" and any(
            _intersection_ratio([x, y, x + w, y + h], box) > 0.5
            and (box[2] - box[0]) > 40
            for box in ocr_boxes
        ):
            continue
        if kind != "cross" and any(
            _intersection_ratio([x, y, x + w, y + h], box) > 0.5 for box in ocr_boxes
        ):
            continue
        pixel = image[y + h // 2, x + w // 2]
        icons.append({
            "box": [x, y, x + w, y + h],
            "kind": kind,
            "color": "#{:02X}{:02X}{:02X}".format(int(pixel[2]), int(pixel[1]), int(pixel[0])),
            "size": [w, h],
            "source": "measured",
        })
    icons.sort(key=lambda icon: -(icon["size"][0] * icon["size"][1]))
    return icons[:top]


def shadow_bands(
    image: Any,
    blocks: list[dict[str, Any]],
    objects: list[dict[str, Any]] | None = None,
    top: int = 6,
) -> list[dict[str, Any]]:
    """Soft drop shadows: gradual darkening bands below large blocks.

    A band counts as a shadow when it starts darker than the page
    background, fades towards it monotonically and contains no content
    (low variance) — typical of Figma-style drop shadows under photos.
    Photos often fail contour layout (their texture floods the edge map),
    so large detected objects (>= 60k px) are also shadow candidates.
    """
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bg = float(np.percentile(gray, 90))
    bands = []
    big = [b for b in blocks if b["area"] >= 12000][:12]
    if objects:
        big += [
            {"box": [round(v) for v in obj["box"]]}
            for obj in objects
            if (obj["box"][2] - obj["box"][0]) * (obj["box"][3] - obj["box"][1]) >= 60000
        ]
    seen: set[tuple[int, int, int, int]] = set()
    for block in big:
        x0, y0, x1, y1 = block["box"]
        key = (x0, y0, x1, y1)
        if key in seen:
            continue
        seen.add(key)
        if y1 + 14 >= gray.shape[0]:
            continue
        band = gray[y1:min(gray.shape[0], y1 + 90), max(0, x0 + 2):max(3, x1 - 2)]
        if band.size < 200:
            continue
        row_means = band.mean(axis=1)
        if row_means[0] > bg - 6:
            continue
        if float(row_means.std()) > 10:
            continue
        fading = row_means[-1] > row_means[0] and row_means[-1] > bg - 12
        above_bg = row_means > bg - 6
        depth = int(np.argmax(above_bg)) if above_bg.any() else len(row_means)
        if depth >= 10 and fading:
            bands.append({
                "box": [int(x0), int(y1), int(x1), min(gray.shape[0], y1 + depth)],
                "kind": "drop_shadow",
                "direction": "below",
                "depth": depth,
                "strength": round(bg - float(row_means[0]), 1),
                "source": "measured",
            })
    bands.sort(key=lambda band: -band["depth"])
    return bands[:top]


# --------------------------------------------------------------------------
# Dump construction
# --------------------------------------------------------------------------


def analyze(
    image_path: str,
    region: dict[str, int] | None = None,
    no_store: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    if region is not None:
        import cv2

        image = load_cv(image_path)
        # MCP schemas send f64; slices need ints.
        x, y, w, h = (int(round(v)) for v in (
            region["x"], region["y"], region["width"], region["height"]
        ))
        crop = image[y:y + h, x:x + w]
        # Zoom-Refine: upscale small crops so OCR/layout see readable strokes.
        small_side = min(crop.shape[0], crop.shape[1])
        if 0 < small_side < 512:
            scale = min(4.0, 512.0 / small_side)
            crop = cv2.resize(
                crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
        fd, crop_path = tempfile.mkstemp(suffix=".png", prefix="sight-crop-")
        os.close(fd)
        cv2.imwrite(crop_path, crop)
        try:
            return analyze(crop_path, region=None, no_store=no_store)
        finally:
            try:
                os.unlink(crop_path)
            except OSError:
                pass

    key = cache_key(image_path, None)
    if not no_store:
        cached = read_cache(key)
        if cached is not None:
            cached["elapsedMs"] = round((time.perf_counter() - started) * 1000)
            cached["cached"] = True
            return cached

    dump = analyze_full(image_path)
    dump["cached"] = False
    dump["elapsedMs"] = round((time.perf_counter() - started) * 1000)
    if not no_store:
        write_cache(key, dump)
    return dump


def analyze_full(image_path: str) -> dict[str, Any]:
    image = load_cv(image_path)
    ocr_items = run_ocr(image_path)
    for item in ocr_items:
        metrics = _glyph_metrics(image, item["box"])
        if metrics is not None:
            item["metrics"] = metrics
    colors = color_zones(image)
    blocks = layout_blocks(image)
    blocks = blocks + texture_blocks(image, ocr_items)
    attention = attention_map(image, ocr_items)
    objects = objects_yolo(image_path)
    scene = scene_clip(image_path)
    controls = control_style(image, blocks)
    controls = controls + _controls_around_text(image, ocr_items, controls)
    dump = {
        "image": {"width": colors["width"], "height": colors["height"]},
        "colors": colors["dominant"],
        "ocr": ocr_items,
        "layout": blocks,
        "skeleton": layout_skeleton(image),
        "gaps": layout_gaps(blocks),
        "design": design_qa(image, blocks, ocr_items),
        "sectionStyle": section_style(image, blocks, ocr_items),
        "controls": controls,
        "icons": control_icons(image, blocks, ocr_items),
        "shadows": shadow_bands(image, blocks, objects),
        "attention": attention,
        "scene": scene,
        "objects": objects,
    }
    dump["verification"] = cross_verify(dump)
    return dump


def locate_text(dump: dict[str, Any], query: str) -> dict[str, Any]:
    needle = query.strip().lower()
    if not needle:
        raise ValueError("query is required")
    matches = []
    for item in dump.get("ocr", []):
        text = item["text"].lower()
        if needle == text or needle in text or text in needle:
            matches.append(item)
    if matches:
        best = max(matches, key=lambda m: m["confidence"])
        return {"found": True, "box": best["box"], "text": best["text"], "confidence": best["confidence"], "matches": matches[:5]}
    return {"found": False, "box": None, "text": None, "confidence": None, "matches": []}


def inspect_target(
    image_path: str, target: str, no_store: bool = False, pad_ratio: float = 0.15
) -> dict[str, Any]:
    """Locate `target` text, then re-analyze its crop with padding.

    Deterministic: same image + target always yields the same box and dump.
    """
    import cv2

    dump = analyze(image_path, None, no_store)
    located = locate_text(dump, target)
    if not located["found"]:
        return {
            "found": False,
            "grounding": None,
            "analysis": None,
        }
    image = load_cv(image_path)
    height, width = image.shape[:2]
    x0, y0, x1, y1 = located["box"]
    pad_x = int((x1 - x0) * pad_ratio)
    pad_y = int((y1 - y0) * pad_ratio)
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(width, x1 + pad_x)
    y1 = min(height, y1 + pad_y)
    return {
        "found": True,
        "grounding": located["box"],
        "text": located["text"],
        "analysis": analyze(
            image_path,
            {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0},
            no_store,
        ),
    }


# --------------------------------------------------------------------------
# NDJSON protocol loop (same shape as the other sidecar workers)
# --------------------------------------------------------------------------


def handle(message: dict[str, object]) -> dict[str, object]:
    operation = str(message.get("operation", ""))
    payload = message.get("input") or {}
    no_store = bool(message.get("noStore", False))
    if not isinstance(payload, dict):
        raise ValueError("Sight input must be an object")
    if operation == "see":
        return analyze(str(payload["imagePath"]), payload.get("region"), no_store)
    if operation == "read":
        dump = analyze(str(payload["imagePath"]), payload.get("region"), no_store)
        return {"texts": [item["text"] for item in dump["ocr"]], "ocr": dump["ocr"]}
    if operation == "locate":
        dump = analyze(str(payload["imagePath"]), None, no_store)
        return locate_text(dump, str(payload.get("target", "")))
    if operation == "zoom":
        region = payload.get("region")
        if not isinstance(region, dict):
            raise ValueError("region is required for zoom")
        return analyze(str(payload["imagePath"]), region, no_store)
    if operation == "inspect":
        region = payload.get("region")
        target = payload.get("target")
        if isinstance(region, dict):
            return analyze(str(payload["imagePath"]), region, no_store)
        if target:
            return inspect_target(str(payload["imagePath"]), str(target), no_store)
        raise ValueError("region or target is required for inspect")
    if operation == "compare":
        return compare_images(
            str(payload["referencePath"]), str(payload["candidatePath"])
        )
    raise ValueError(f"Unsupported Sight operation: {operation}")


for line in sys.stdin:
    if not line.strip():
        continue
    request_id = None
    try:
        message = json.loads(line)
        request_id = message.get("requestId")
        result = handle(message)
        response = {"ok": True, "requestId": request_id, "result": result}
    except Exception as error:  # noqa: BLE001 - protocol boundary
        response = {
            "ok": False,
            "requestId": request_id,
            "error": {"message": str(error), "type": type(error).__name__},
        }
    try:
        print(json.dumps(response, ensure_ascii=False), flush=True)
    except Exception as error:  # noqa: BLE001 - stdout is broken; exit for the broker to restart us
        sys.stderr.write(f"could not write response: {error}\n")
        sys.stderr.flush()
        raise SystemExit(3)
