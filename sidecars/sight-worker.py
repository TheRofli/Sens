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
    # covering > 20% of its area is an outer shell, not a section.
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
    sections = [b for b in blocks if b["area"] >= min_area_ratio * max(1, blocks[0]["area"])] if blocks else []
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
# Bump when the dump schema changes so stale dumps (e.g. without gaps)
# are not served from cache.
CACHE_SCHEMA_VERSION = "qa1"


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
    colors = color_zones(image)
    blocks = layout_blocks(image)
    attention = attention_map(image, ocr_items)
    objects = objects_yolo(image_path)
    scene = scene_clip(image_path)
    dump = {
        "image": {"width": colors["width"], "height": colors["height"]},
        "colors": colors["dominant"],
        "ocr": ocr_items,
        "layout": blocks,
        "skeleton": layout_skeleton(image),
        "gaps": layout_gaps(blocks),
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
