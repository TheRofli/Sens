"""Sight operations: analyze, analyze_full, locate, inspect."""

from __future__ import annotations

import os
import tempfile
import time
from typing import Any

from sight.ocr import load_cv, run_ocr
from sight.perception import _controls_around_text, _glyph_metrics, _hex_to_bgr, _intersection_ratio, _luminance, attention_map, color_zones, layout_blocks, layout_gaps, layout_skeleton, objects_yolo, scene_clip, texture_blocks
from sight.qa import _center_in, control_icons, control_style, cross_verify, design_qa, section_style, shadow_bands
from sight.tree import _assign_roles, annotate_som, build_element_tree, build_section_tree, expand_button_subparts, summarize_screen
from sight.cache import cache_key, cache_root, read_cache, write_cache
from sight import document as docmod
from sight.vlm import VISION_PROMPT, VlmHost




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




def _plausible_controls(
    controls: list[dict[str, Any]],
    background_bgr: Any,
    ocr_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop ghost controls and merge duplicates.

    A control whose fill equals the page background, has no border and no
    label text inside is layout noise (a text zone), not a button. Boxes
    overlapping 85%+ are the same widget detected twice."""
    kept: list[dict[str, Any]] = []
    for control in controls:
        keep = True
        if not control.get("borderColor") and control.get("background"):
            bg = _hex_to_bgr(control["background"])
            if abs(_luminance(bg) - _luminance(background_bgr)) < 15:
                # A fill equal to the page background is a button only if a
                # label of plausible size sits inside it (its center within
                # the box and not vastly larger than the box itself).
                cx0, cy0, cx1, cy1 = control["box"]
                control_area = max(1, (cx1 - cx0) * (cy1 - cy0))
                keep = any(
                    _center_in(item["box"], control["box"])
                    and (item["box"][2] - item["box"][0]) * (item["box"][3] - item["box"][1])
                    <= 2 * control_area
                    for item in ocr_items
                )
        if not keep:
            continue
        box = [int(v) for v in control["box"]]
        if any(_intersection_ratio(box, k["box"]) > 0.85 for k in kept):
            continue
        kept.append(control)
    return kept




def analyze_full(image_path: str) -> dict[str, Any]:
    image = load_cv(image_path)
    ocr_items = run_ocr(image_path)
    for item in ocr_items:
        metrics = _glyph_metrics(image, item["box"])
        if metrics is not None:
            item["metrics"] = metrics
    colors = color_zones(image)
    blocks = layout_blocks(image)
    textures = texture_blocks(image, ocr_items)
    blocks = blocks + textures
    attention = attention_map(image, ocr_items)
    objects = objects_yolo(image_path)
    scene = scene_clip(image_path)
    height, width = image.shape[:2]
    background_hex = colors["dominant"][0]["hex"] if colors.get("dominant") else "#000000"
    background_bgr = _hex_to_bgr(background_hex)
    controls = control_style(image, blocks)
    controls = controls + _controls_around_text(image, ocr_items, controls)
    controls = _plausible_controls(controls, background_bgr, ocr_items)
    icons = control_icons(image, blocks, ocr_items)
    # Hierarchical analysis: screen -> sections -> elements -> sub-elements.
    tree = build_section_tree(image, background_bgr, textures, width, height, ocr_items, controls)
    _assign_roles(tree, ocr_items, controls, width, height)
    elements = build_element_tree(tree, ocr_items, controls, icons, textures)
    expand_button_subparts(elements)
    summary = summarize_screen(tree, width, height, background_hex)
    som_path = annotate_som(
        image, elements,
        os.path.join(
            cache_root(),
            f"som-{os.path.splitext(os.path.basename(image_path))[0]}.png",
        ),
    )
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
        "icons": icons,
        "shadows": shadow_bands(image, blocks, objects),
        "attention": attention,
        "scene": scene,
        "objects": objects,
        "tree": tree,
        "elements": elements,
        "somPath": som_path,
        "summary": summary,
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
# Vision 2.0: visual context document + interactive ops (zoom/ask/element)
# --------------------------------------------------------------------------

_lite_host = VlmHost("lite")
_quality_host = VlmHost("quality")


def _host(quality: bool) -> VlmHost | None:
    host = _quality_host if quality else _lite_host
    return host if host.available() else None


def _image_for(image_path: str, region: dict | None):
    image = load_cv(image_path)
    if region:
        x, y = int(region["x"]), int(region["y"])
        w, h = int(region["width"]), int(region["height"])
        image = image[y : y + h, x : x + w]
    return image


def see_document(
    image_path: str,
    region: dict | None = None,
    no_store: bool = False,
    fast: bool = False,
    quality: bool = False,
) -> dict:
    dump = analyze(image_path, region, no_store)
    vlm = None if fast else _host(quality)
    doc = docmod.build_document(dump, _image_for(image_path, region), vlm=vlm, image_path=image_path)
    return {
        "document": docmod.render_markdown(doc),
        "doc": doc,
        "somPath": dump.get("somPath"),
        "legacy": dump,
    }


def zoom(
    image_path: str,
    region: dict | None = None,
    som_id: int | None = None,
    no_store: bool = False,
    quality: bool = False,
) -> dict:
    if region is None and som_id is not None:
        dump = analyze(image_path, None, no_store)
        el = next((e for e in dump["elements"] if e.get("id") == som_id), None)
        if el is None:
            raise ValueError(f"no element with id {som_id}")
        pad = 24
        region = {
            "x": max(0, el["box"][0] - pad),
            "y": max(0, el["box"][1] - pad),
            "width": el["box"][2] - el["box"][0] + 2 * pad,
            "height": el["box"][3] - el["box"][1] + 2 * pad,
        }
    if region is None:
        raise ValueError("region or somId is required for zoom")
    return see_document(image_path, region, no_store, quality=quality)


def ask(image_path: str, question: str, region: dict | None = None, quality: bool = False) -> dict:
    host = _host(quality) or _lite_host
    if not host.available():
        raise RuntimeError("vision models not downloaded; run scripts/download-vision-models.py")
    box = [region["x"], region["y"], region["x"] + region["width"], region["y"] + region["height"]] if region else None
    return {"answer": host.ask(image_path, question, box)}


def element(image_path: str, som_id: int, no_store: bool = False) -> dict:
    dump = analyze(image_path, None, no_store)
    el = next((e for e in dump["elements"] if e.get("id") == som_id), None)
    if el is None:
        raise ValueError(f"no element with id {som_id}")
    w, h = dump["image"]["width"], dump["image"]["height"]
    return {"element": el, "box_norm": docmod.normalize_box(el["box"], w, h)}


def vision_prompt(lang: str = "ru") -> dict:
    return {"prompt": VISION_PROMPT.get(lang, VISION_PROMPT["ru"])}


def warm() -> dict:
    host = _host(False)
    if host is None:
        return {"models": False}
    host._load()  # noqa: SLF001 - intentional warm preload
    return {"models": True}


def capture_op(url: str) -> dict:
    from sight.capture import capture_url

    return capture_url(url, cache_root() / "captures")


def motion_op(url: str) -> dict:
    result = capture_op(url)
    return {
        "animations": result["animations"],
        "motion": result["motion"],
        "screenshot": result["screenshot"],
        "styles": result["styles"],
    }
