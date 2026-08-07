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
from sight.coordinates import crop_coordinates, identity_coordinates
from sight import document as docmod
from sight.vlm import VISION_PROMPT, VlmHost




def _source_from_cache_key(key: str) -> dict[str, str]:
    parts = key.split("-", 2)
    digest = parts[1] if len(parts) > 2 else key.removesuffix(".json")
    return {"id": f"sha256-128:{digest}", "mediaType": "image"}


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

        source_key = cache_key(image_path, None)
        image = load_cv(image_path)
        source_height, source_width = image.shape[:2]
        # MCP schemas send f64; slices need ints.
        x, y, w, h = (int(round(v)) for v in (
            region["x"], region["y"], region["width"], region["height"]
        ))
        if w <= 0 or h <= 0:
            raise ValueError("region width and height must be positive")
        x0 = max(0, min(source_width, x))
        y0 = max(0, min(source_height, y))
        x1 = max(0, min(source_width, x + w))
        y1 = max(0, min(source_height, y + h))
        if x1 <= x0 or y1 <= y0:
            raise ValueError("region does not intersect the source image")
        crop = image[y0:y1, x0:x1]
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
            dump = analyze(crop_path, region=None, no_store=no_store)
            analysis_height, analysis_width = crop.shape[:2]
            dump["coordinates"] = crop_coordinates(
                source_width,
                source_height,
                [x0, y0, x1, y1],
                analysis_width,
                analysis_height,
            )
            dump["source"] = _source_from_cache_key(source_key)
            return dump
        finally:
            try:
                os.unlink(crop_path)
            except OSError:
                pass

    key = cache_key(image_path, None)
    if not no_store:
        cached = read_cache(key)
        if cached is not None:
            cached.setdefault("source", _source_from_cache_key(key))
            cached["elapsedMs"] = round((time.perf_counter() - started) * 1000)
            cached["cached"] = True
            return cached

    artifact_key = key.removesuffix(".json")
    dump = analyze_full(
        image_path,
        store_artifacts=not no_store,
        artifact_key=artifact_key,
    )
    dump["source"] = _source_from_cache_key(key)
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


def _run_optional_layer(
    name: str, operation: Any, image_path: str
) -> tuple[list[dict[str, Any]], dict[str, str] | None]:
    try:
        return operation(image_path), None
    except (ImportError, ModuleNotFoundError, FileNotFoundError, RuntimeError) as error:
        return [], {
            "code": f"optional_{name}_unavailable",
            "message": f"Optional {name} layer is unavailable: {error}",
            "recovery": "Continue with measured layout/OCR and the local VLM, or install the optional detector pack.",
        }




def analyze_full(
    image_path: str,
    *,
    store_artifacts: bool = True,
    artifact_key: str | None = None,
) -> dict[str, Any]:
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
    objects, objects_warning = _run_optional_layer("objects", objects_yolo, image_path)
    scene, scene_warning = _run_optional_layer("scene", scene_clip, image_path)
    optional_warnings = [
        warning for warning in (objects_warning, scene_warning) if warning is not None
    ]
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
    som_path = None
    if store_artifacts:
        name = artifact_key or cache_key(image_path, None).removesuffix(".json")
        som_path = annotate_som(
            image,
            elements,
            os.path.join(cache_root(), f"som-{name}.png"),
        )
    dump = {
        "image": {"width": colors["width"], "height": colors["height"]},
        "coordinates": identity_coordinates(width, height),
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
        "warnings": optional_warnings,
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
_quality_large_host = VlmHost("quality_large")
_HOSTS = {
    "lite": _lite_host,
    "quality": _quality_host,
    "quality_large": _quality_large_host,
}


def _default_pack() -> str:
    env = os.environ.get("SENS_VISION_PACK", "lite")
    return env if env in _HOSTS else "lite"


def _resolve_pack(pack: str | None, quality: bool) -> str:
    # Per-request MCP arguments win over the desktop-settings default,
    # which the broker passes as SENS_VISION_PACK at worker spawn.
    if pack in _HOSTS:
        return pack
    if quality:
        return "quality"
    return _default_pack()


def _host(quality: bool, pack: str | None = None) -> VlmHost | None:
    host = _HOSTS[_resolve_pack(pack, quality)]
    return host if host.available() else None


def _pick(quality: bool, pack: str | None = None) -> tuple[str | None, VlmHost | None]:
    """Resolve the effective pack for a request.

    A pack that is not downloaded degrades to lite (documented behavior);
    returns (pack name or None, host or None)."""
    resolved = _resolve_pack(pack, quality)
    host = _HOSTS[resolved]
    if not host.available():
        if resolved != "lite" and _lite_host.available():
            return "lite", _lite_host
        return None, None
    return resolved, host


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
    pack: str | None = None,
    intent: str | None = None,
    max_semantic_calls: int = 2,
) -> dict:
    dump = analyze(image_path, region, no_store)
    pack_name, vlm = (None, None) if fast else _pick(quality, pack)
    doc = docmod.build_document(
        dump,
        _image_for(image_path, region),
        vlm=vlm,
        image_path=image_path,
        intent=intent,
        max_semantic_calls=max_semantic_calls,
    )
    legacy = dict(dump)
    if "facts" in legacy.get("design", {}):
        legacy["design"] = {"issues": legacy["design"]["facts"]}
    return {
        "document": docmod.render_markdown(doc),
        "doc": doc,
        "somPath": dump.get("somPath"),
        "legacy": legacy,
        "pack": pack_name,
    }


def zoom(
    image_path: str,
    region: dict | None = None,
    som_id: int | None = None,
    no_store: bool = False,
    quality: bool = False,
    pack: str | None = None,
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
    return see_document(image_path, region, no_store, quality=quality, pack=pack)


def ask(
    image_path: str,
    question: str,
    region: dict | None = None,
    quality: bool = False,
    pack: str | None = None,
) -> dict:
    _, host = _pick(quality, pack)
    if host is None:
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
    host = _HOSTS[_default_pack()]
    if not host.available():
        return {"models": False}
    host._load()  # noqa: SLF001 - intentional warm preload
    return {"models": True}


def capture_op(
    url: str,
    options: dict[str, Any] | None = None,
    no_store: bool = False,
) -> dict:
    from sight.capture import capture_url

    return capture_url(url, cache_root() / "captures", options, no_store=no_store)


def motion_op(
    url: str,
    options: dict[str, Any] | None = None,
    no_store: bool = False,
) -> dict:
    capture_options = dict(options or {})
    capture_options.setdefault("scrollSteps", 4)
    result = capture_op(url, capture_options, no_store)
    return {
        "animations": result["animations"],
        "motion": result["motion"],
        "screenshot": result["screenshot"],
        "styles": result["styles"],
    }
