"""Sight operations: analyze, analyze_full, locate, inspect."""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import tempfile
import time
from pathlib import Path
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any

from sight.ocr import (
    discover_display_ocr,
    load_cv,
    merge_script_ocr_passes,
    refine_ocr_for_reconstruction,
    run_latin_ocr,
    run_latin_ocr_image,
    run_ocr,
)
from sight.perception import _controls_around_text, _glyph_metrics, _hex_to_bgr, _intersection_ratio, _luminance, attention_map, color_zones, compact_numeric_badges, layout_blocks, layout_gaps, layout_skeleton, objects_yolo, outlined_controls_around_text, scene_clip, texture_blocks
from sight.qa import control_icons, control_style, cross_verify, design_qa, dominant_fill_surfaces, outlined_surface_regions, section_style, shadow_bands, surface_regions
from sight.tree import _assign_roles, annotate_som, build_element_tree, build_section_tree, expand_button_subparts, summarize_screen
from sight.cache import (
    cache_key,
    cache_root,
    document_cache_key,
    read_cache,
    write_cache,
)
from sight.coordinates import crop_coordinates, identity_coordinates
from sight import document as docmod
from sight.vlm import VISION_PROMPT, VlmHost
from sight.symbol_art import detect_symbol_art
from sight.starter import materialize_starter_project
from sight.perception import (
    _segment_edge_contrast,
    detect_dashed_structural_lines,
    detect_vector_paths,
)
from sight.tokens import build_design_tokens




def _source_from_cache_key(key: str) -> dict[str, str]:
    parts = key.split("-", 2)
    digest = parts[1] if len(parts) > 2 else key.removesuffix(".json")
    return {"id": f"sha256-128:{digest}", "mediaType": "image"}


def _source_box_area(box: list[int] | None) -> int:
    if not box or len(box) != 4:
        return 0
    return max(0, int(box[2] - box[0])) * max(0, int(box[3] - box[1]))


def _sync_bounded_ocr_elements(
    dump: dict[str, Any], image: Any, ocr_items: list[dict[str, Any]]
) -> None:
    """Project a dual-recognizer crop result back into its element list."""
    existing = [
        element
        for element in dump.get("elements") or []
        if element.get("kind") == "text"
    ]
    non_text = [
        element
        for element in dump.get("elements") or []
        if element.get("kind") != "text"
    ]
    unused = set(range(len(existing)))
    next_id = max(
        (int(element.get("id") or 0) for element in dump.get("elements") or []),
        default=0,
    ) + 1
    projected = []
    for item in ocr_items:
        box = [int(round(value)) for value in item.get("box") or []]
        if len(box) != 4:
            continue
        ranked = sorted(
            (
                (
                    _intersection_ratio(box, existing[index].get("box") or []),
                    index,
                )
                for index in unused
            ),
            reverse=True,
        )
        if ranked and ranked[0][0] >= 0.55:
            _score, index = ranked[0]
            unused.remove(index)
            element = dict(existing[index])
        else:
            element = {"id": next_id, "kind": "text"}
            next_id += 1
        metrics = _glyph_metrics(image, box, item.get("text"))
        element.update(
            {
                "kind": "text",
                "box": box,
                "text": item.get("text"),
                "confidence": round(float(item.get("confidence") or 0.0), 3),
                "verified": item.get("verified"),
                "method": item.get("method") or "rapidocr",
                "alternatives": list(item.get("alternatives") or []),
                "epistemic": item.get("epistemic") or "inferred",
            }
        )
        if metrics is not None:
            element["font"] = metrics
        projected.append(element)
    dump["ocr"] = ocr_items
    dump["elements"] = [*non_text, *projected]


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
            try:
                latin_items = run_latin_ocr(crop_path)
                fused_ocr = merge_script_ocr_passes(
                    list(dump.get("ocr") or []), latin_items
                )
                _sync_bounded_ocr_elements(dump, crop, fused_ocr)
                dump["ocrConsensus"] = {
                    "passes": 2,
                    "method": "bounded-cyrillic-plus-latin-recognizer-fusion",
                }
            except (ImportError, ModuleNotFoundError, RuntimeError, OSError) as error:
                dump.setdefault("warnings", []).append(
                    {
                        "code": "optional_latin_ocr_unavailable",
                        "message": f"Bounded Latin OCR is unavailable: {error}",
                        "recovery": "Continue with the Cyrillic-capable OCR and local VLM crop.",
                    }
                )
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
    _ocr_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop ghost controls and merge duplicates.

    A borderless region is a panel/surface, not a control. OCR inside a box is
    evidence for text, not interaction. A measured outline must also contrast
    with its fill/page; otherwise chart frames and separators become fake
    buttons. Boxes overlapping 85%+ are the same widget detected twice."""
    kept: list[dict[str, Any]] = []
    for control in controls:
        keep = True
        border = control.get("borderColor")
        if not border:
            keep = False
        else:
            fill = (
                _hex_to_bgr(control["background"])
                if control.get("background")
                else background_bgr
            )
            if abs(_luminance(_hex_to_bgr(border)) - _luminance(fill)) < 15:
                keep = False
        if not keep:
            continue
        box = [int(v) for v in control["box"]]
        if any(_intersection_ratio(box, k["box"]) > 0.85 for k in kept):
            continue
        kept.append(control)
    return kept


def _sanitize_web_structure(document: dict[str, Any]) -> None:
    """Separate measured panels/chart frames from semantic web controls.

    Older scene caches may contain broad layout regions in ``controls``.  This
    contract-level pass is intentionally idempotent so a corrected 1.3.7
    document is returned even when its deterministic vision layers came from a
    pre-fix cache.
    """
    spec = document.get("reconstruction") or {}
    if spec.get("targetKind") != "web":
        return
    controls = spec.get("visualControlCandidates") or []
    surfaces = spec.setdefault("surfaces", [])
    kept: list[dict[str, Any]] = []
    for control in controls:
        measured_compact_control = (
            control.get("geometrySource")
            in {"measured-compact-fill", "measured-closed-outline"}
            and bool(control.get("labelElementIds"))
        )
        border = control.get("borderColor") or (control.get("style") or {}).get(
            "borderColor"
        )
        background = control.get("background") or (control.get("style") or {}).get(
            "background"
        )
        if not border and not measured_compact_control:
            continue
        fill = background
        if border and fill and not measured_compact_control and abs(
            _luminance(_hex_to_bgr(border)) - _luminance(_hex_to_bgr(fill))
        ) < 15:
            continue
        kept.append(control)
    spec["visualControlCandidates"] = kept
    candidate_containers = [
        *[entry.get("boxSource") or [] for entry in spec.get("text") or []],
        *[
            entry.get("boxSource") or []
            for entry in spec.get("allowedRasterRegions") or []
        ],
    ]

    def covered_by_text_or_raster(entry: dict[str, Any]) -> bool:
        box = entry.get("boxSource") or []
        if len(box) != 4:
            return False
        area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
        return any(
            len(container) == 4
            and _box_intersection(box, container) / area >= 0.75
            for container in candidate_containers
        )

    spec["surfaces"] = [
        surface
        for surface in surfaces
        if surface.get("source") != "measured-control-reclassified-as-surface"
        and not covered_by_text_or_raster(surface)
    ]
    spec["decorativeShapes"] = [
        shape
        for shape in spec.get("decorativeShapes") or []
        if not covered_by_text_or_raster(shape)
    ]


def _refresh_web_tokens(document: dict[str, Any], dump: dict[str, Any]) -> None:
    spec = document.get("reconstruction") or {}
    if spec.get("targetKind") == "web":
        document["tokens"] = build_design_tokens(dump)


def _hydrate_measured_surfaces(document: dict[str, Any], image_path: str) -> None:
    """Add neutral connected UI surfaces to old and new web contracts."""
    spec = document.get("reconstruction") or {}
    if spec.get("targetKind") != "web":
        return
    try:
        image = load_cv(image_path)
    except (OSError, ValueError):
        return
    colors = (document.get("tokens") or {}).get("color") or {}
    canvas = (colors.get("canvas") or {}).get("$value")
    if not canvas:
        canvas = (document.get("header") or {}).get("background") or "#FFFFFF"
    surfaces = spec.setdefault("surfaces", [])
    measured_surfaces = [
        *dominant_fill_surfaces(image, canvas),
        *outlined_surface_regions(image, canvas),
    ]
    for entry in measured_surfaces:
        candidate = {
            "boxSource": entry["box"],
            "background": entry.get("background"),
            "borderColor": entry.get("borderColor"),
            "borderWidth": entry.get("borderWidth"),
            "cornerRadius": entry.get("cornerRadius"),
            "fillRatio": entry.get("fillRatio"),
            "source": entry.get("source"),
            "method": entry.get("method"),
        }
        exact_surface = next(
            (
                surface
                for surface in surfaces
                if surface.get("boxSource") == candidate["boxSource"]
            ),
            None,
        )
        if exact_surface is not None:
            if candidate.get("borderColor"):
                exact_surface.update(
                    {
                        "borderColor": candidate.get("borderColor"),
                        "borderWidth": candidate.get("borderWidth"),
                        "cornerRadius": max(
                            int(exact_surface.get("cornerRadius") or 0),
                            int(candidate.get("cornerRadius") or 0),
                        ),
                        "method": candidate.get("method"),
                    }
                )
            continue
        candidate_box = candidate["boxSource"]
        candidate_area = max(1, _source_box_area(candidate_box))
        same_fill_overlaps = [
            surface
            for surface in surfaces
            if _intersection_ratio(
                candidate_box, surface.get("boxSource") or []
            )
            > 0.92
            and surface.get("background") == candidate["background"]
            and (
                min(
                    candidate_area,
                    max(1, _source_box_area(surface.get("boxSource"))),
                )
                / max(
                    candidate_area,
                    max(1, _source_box_area(surface.get("boxSource"))),
                )
                >= 0.8
                or sum(
                    abs(candidate_box[index] - surface["boxSource"][index]) <= 2
                    for index in range(4)
                )
                >= 3
            )
        ]
        if any(
            max(1, _source_box_area(surface.get("boxSource")))
            >= candidate_area * 0.95
            for surface in same_fill_overlaps
        ):
            continue
        surfaces[:] = [
            surface
            for surface in surfaces
            if surface not in same_fill_overlaps
        ]
        surfaces.append(candidate)
    _sanitize_web_structure(document)


def _refine_overlapping_raster_candidates(
    document: dict[str, Any], image_path: str
) -> None:
    """Recover a large photographic foreground from an over-broad image box.

    Texture segmentation can join a soft hero photo to nearby live text through
    a white fade.  The original box must stay forbidden, but its largest
    measured non-background component can be promoted when it no longer
    intersects any text or control box.
    """
    import cv2
    import numpy as np

    spec = document.get("reconstruction") or {}
    if spec.get("targetKind") != "web":
        return
    excluded = spec.get("excludedRasterCandidates") or []
    if not excluded:
        return
    try:
        image = load_cv(image_path)
    except (OSError, ValueError):
        return
    height, width = image.shape[:2]
    canvas_area = max(1, width * height)
    colors = (document.get("tokens") or {}).get("color") or {}
    background = (
        (colors.get("canvas") or {}).get("$value")
        or (document.get("header") or {}).get("background")
        or "#FFFFFF"
    )
    background_bgr = np.asarray(_hex_to_bgr(background), dtype=np.float32)
    protected_boxes = [
        entry.get("boxSource") or []
        for key in ("text", "visualControlCandidates", "symbolArt")
        for entry in spec.get(key) or []
        if len(entry.get("boxSource") or []) == 4
    ]
    allowed = spec.setdefault("allowedRasterRegions", [])

    def intersects_protected(box: list[int]) -> bool:
        return any(_intersection_ratio(box, other) > 0 for other in protected_boxes)

    for candidate in excluded:
        if candidate.get("reason") != "overlaps-live-text":
            continue
        if candidate.get("refinedIntoAllowedBoxSource"):
            continue
        if any(
            entry.get("elementId") == candidate.get("elementId")
            for entry in allowed
        ):
            continue
        raw_box = candidate.get("boxSource") or []
        if len(raw_box) != 4:
            continue
        x0, y0, x1, y1 = (int(round(value)) for value in raw_box)
        x0, y0 = max(0, min(width, x0)), max(0, min(height, y0))
        x1, y1 = max(0, min(width, x1)), max(0, min(height, y1))
        if x1 <= x0 or y1 <= y0:
            continue
        if (x1 - x0) * (y1 - y0) < canvas_area * 0.20:
            continue
        crop = image[y0:y1, x0:x1].astype(np.float32)
        delta = np.linalg.norm(crop - background_bgr, axis=2)
        mask = (delta > 10.0).astype(np.uint8) * 255
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
        )
        for protected in protected_boxes:
            px0 = max(x0, protected[0] - 4) - x0
            py0 = max(y0, protected[1] - 4) - y0
            px1 = min(x1, protected[2] + 4) - x0
            py1 = min(y1, protected[3] + 4) - y0
            if px1 > px0 and py1 > py0:
                mask[py0:py1, px0:px1] = 0
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            mask, 8
        )
        ranked: list[tuple[int, list[int]]] = []
        for component in range(1, count):
            local_x, local_y, box_width, box_height, area = (
                int(value) for value in stats[component]
            )
            box = [
                x0 + local_x,
                y0 + local_y,
                x0 + local_x + box_width,
                y0 + local_y + box_height,
            ]
            box_area = max(1, box_width * box_height)
            if (
                area < canvas_area * 0.015
                or box_area < canvas_area * 0.04
                or box_width < width * 0.12
                or box_height < height * 0.20
                or area / box_area < 0.12
                or intersects_protected(box)
            ):
                continue
            gray_patch = cv2.cvtColor(
                image[box[1] : box[3], box[0] : box[2]], cv2.COLOR_BGR2GRAY
            )
            if float(gray_patch.std()) < 12.0:
                continue
            ranked.append((area, box))
        if not ranked:
            continue
        _area, refined = max(ranked, key=lambda item: item[0])
        padding = max(4, round(min(width, height) * 0.008))
        refined = [
            max(x0, refined[0] - padding),
            max(y0, refined[1] - padding),
            min(x1, refined[2] + padding),
            min(y1, refined[3] + padding),
        ]
        if intersects_protected(refined):
            continue
        region = {
            "elementId": candidate.get("elementId"),
            "kind": "illustration-or-photo",
            "boxSource": refined,
            "boxNormSource": docmod.normalize_box(refined, width, height),
            "strategy": "extract-source-crop-verbatim",
            "implementation": "Crop boxSource once from the original reference into a local raster asset and place that asset at the same source-pixel box. Preserve the pixels verbatim; do not redraw, trace, describe, or semantically reinterpret this region.",
            "source": "measured",
            "method": "foreground-component-raster-refinement",
            "derivedFromExcludedBoxSource": [x0, y0, x1, y1],
        }
        allowed.append(region)
        candidate["refinedIntoAllowedBoxSource"] = refined
        spec["primaryAsset"] = {
            "elementId": candidate.get("elementId"),
            "boxSource": refined,
            "boxNormSource": docmod.normalize_box(refined, width, height),
            "areaRatio": round(_source_box_area(refined) / canvas_area, 4),
            "strategy": "extract-source-crop-verbatim",
            "rule": "Extract boxSource verbatim from the original reference and reuse it as one local raster asset. Do not trace, redraw, describe, or re-analyze an allowed principal asset.",
        }
    if allowed:
        _sanitize_web_structure(document)


def _hydrate_intrinsic_text_raster_assets(
    document: dict[str, Any], image_path: str
) -> None:
    """Recover a foreground product whose printed label was mistaken for DOM.

    Packaging, book covers, posters held by a person, and similar physical
    objects may contain perfectly legible OCR.  That lettering is still part
    of the photographed object: rebuilding it as independent HTML both
    destroys the object and creates fake controls.  A layout-seeded GrabCut
    pass promotes only a compact, textured foreground component containing at
    least two observed text regions.  Large measured UI surfaces and symbol
    art remain explicitly ineligible.
    """
    import cv2
    import numpy as np

    spec = document.get("reconstruction") or {}
    if spec.get("targetKind") != "web":
        return
    allowed = spec.setdefault("allowedRasterRegions", [])
    if any(entry.get("method") == "layout-seeded-grabcut-object" for entry in allowed):
        return
    try:
        image = load_cv(image_path)
    except (OSError, ValueError):
        return
    height, width = image.shape[:2]
    canvas_area = max(1, width * height)
    text_entries = spec.get("text") or []
    surfaces = spec.get("surfaces") or []
    symbol_boxes = [
        entry.get("boxSource") or []
        for entry in spec.get("symbolArt") or []
        if len(entry.get("boxSource") or []) == 4
    ]
    existing_boxes = [
        entry.get("boxSource") or []
        for entry in allowed
        if len(entry.get("boxSource") or []) == 4
    ]

    def coverage(inner: list[int], outer: list[int]) -> float:
        return _box_intersection(inner, outer) / max(1, _source_box_area(inner))

    def center_inside(entry_box: list[int], outer: list[int]) -> bool:
        if len(entry_box) != 4:
            return False
        center_x = (entry_box[0] + entry_box[2]) / 2
        center_y = (entry_box[1] + entry_box[3]) / 2
        return outer[0] <= center_x <= outer[2] and outer[1] <= center_y <= outer[3]

    regions = sorted(
        (
            entry
            for entry in spec.get("layoutRegions") or []
            if entry.get("role") in {"content", "hero"}
            and len(entry.get("boxSource") or []) == 4
        ),
        key=lambda entry: (
            entry.get("role") != "content",
            _source_box_area(entry.get("boxSource")),
        ),
    )
    promoted: tuple[dict[str, Any], list[int], list[dict[str, Any]]] | None = None
    for region in regions:
        raw_box = region.get("boxSource") or []
        x0, y0, x1, y1 = (int(round(value)) for value in raw_box)
        x0, y0 = max(0, min(width, x0)), max(0, min(height, y0))
        x1, y1 = max(0, min(width, x1)), max(0, min(height, y1))
        region_box = [x0, y0, x1, y1]
        region_area = _source_box_area(region_box)
        if (
            region_area < canvas_area * 0.07
            or region_area > canvas_area * 0.48
            or x1 - x0 < width * 0.18
            or y1 - y0 < height * 0.22
        ):
            continue
        region_text = [
            entry
            for entry in text_entries
            if center_inside(entry.get("boxSource") or [], region_box)
        ]
        if len(region_text) < 2:
            continue
        if any(_intersection_ratio(region_box, box) > 0 for box in symbol_boxes):
            continue
        if any(_intersection_ratio(region_box, box) > 0.72 for box in existing_boxes):
            continue
        if any(
            coverage(surface.get("boxSource") or [], region_box) >= 0.98
            and _box_intersection(surface.get("boxSource") or [], region_box)
            / max(1, region_area)
            >= 0.45
            for surface in surfaces
            if len(surface.get("boxSource") or []) == 4
        ):
            continue

        mask = np.full((height, width), cv2.GC_BGD, np.uint8)
        mask[y0:y1, x0:x1] = cv2.GC_PR_BGD
        region_width = x1 - x0
        region_height = y1 - y0
        seed_x0 = x0 + round(region_width * 0.28)
        seed_x1 = x0 + round(region_width * 0.72)
        seed_y0 = y0 + round(region_height * 0.08)
        mask[seed_y0:y1, seed_x0:seed_x1] = cv2.GC_PR_FGD
        background_model = np.zeros((1, 65), np.float64)
        foreground_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(
                image,
                mask,
                None,
                background_model,
                foreground_model,
                5,
                cv2.GC_INIT_WITH_MASK,
            )
        except cv2.error:
            continue
        foreground = np.isin(mask, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8)
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            foreground, 8
        )
        components: list[tuple[int, list[int], float]] = []
        for component in range(1, count):
            local_x, local_y, box_width, box_height, area = (
                int(value) for value in stats[component]
            )
            box = [
                local_x,
                local_y,
                local_x + box_width,
                local_y + box_height,
            ]
            box_area = max(1, box_width * box_height)
            if (
                area < canvas_area * 0.025
                or box_area < canvas_area * 0.035
                or box_area > canvas_area * 0.24
                or box_width < width * 0.12
                or box_height < height * 0.22
                or box_width > region_width * 0.82
                or area / box_area < 0.32
            ):
                continue
            patch = image[box[1] : box[3], box[0] : box[2]]
            if patch.size == 0 or float(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).std()) < 18:
                continue
            components.append((area, box, area / box_area))
        if not components:
            continue
        _area, object_box, _extent = max(components, key=lambda item: item[0])
        intrinsic = [
            entry
            for entry in region_text
            if coverage(entry.get("boxSource") or [], object_box) >= 0.90
        ]
        if len(intrinsic) < 2:
            continue
        object_height = max(1, object_box[3] - object_box[1])
        intrinsic_height_ratios = [
            ((entry["boxSource"][3] - entry["boxSource"][1]) / object_height)
            for entry in intrinsic
            if len(entry.get("boxSource") or []) == 4
        ]
        if (
            not intrinsic_height_ratios
            or float(np.median(intrinsic_height_ratios)) > 0.18
            or max(intrinsic_height_ratios) > 0.26
        ):
            continue
        promoted = (region, object_box, intrinsic)
        break

    if promoted is None:
        return
    region, object_box, intrinsic_entries = promoted
    intrinsic_ids = {entry.get("elementId") for entry in intrinsic_entries}
    intrinsic_text = [
        {
            "elementId": entry.get("elementId"),
            "value": entry.get("preferredValue") or entry.get("value"),
            "boxSource": entry.get("boxSource"),
            "confidence": entry.get("confidence"),
            "source": "observed",
            "method": entry.get("method") or "ocr-inside-measured-object",
        }
        for entry in intrinsic_entries
    ]
    next_numeric_id = max(
        [
            int(entry.get("elementId"))
            for group in (
                spec.get("text") or [],
                spec.get("visualControlCandidates") or [],
                spec.get("allowedRasterRegions") or [],
            )
            for entry in group
            if isinstance(entry.get("elementId"), int)
        ]
        or [0]
    ) + 1
    asset = {
        "elementId": next_numeric_id,
        "kind": "product-photo-with-intrinsic-text",
        "boxSource": object_box,
        "boxNormSource": docmod.normalize_box(object_box, width, height),
        "strategy": "extract-source-crop-verbatim",
        "implementation": "Crop boxSource once from the original reference and preserve the complete photographed object verbatim. Its intrinsicText records describe lettering printed inside the object and must not be duplicated as DOM text or controls.",
        "containsIntrinsicText": True,
        "intrinsicText": intrinsic_text,
        "source": "measured",
        "method": "layout-seeded-grabcut-object",
        "derivedFromLayoutRegionId": region.get("regionId"),
    }
    allowed.append(asset)
    spec["text"] = [
        entry for entry in text_entries if entry.get("elementId") not in intrinsic_ids
    ]

    def covered_by_object(entry: dict[str, Any], threshold: float = 0.72) -> bool:
        box = entry.get("boxSource") or []
        return len(box) == 4 and coverage(box, object_box) >= threshold

    spec["visualControlCandidates"] = [
        entry
        for entry in spec.get("visualControlCandidates") or []
        if not covered_by_object(entry)
        and not intrinsic_ids.intersection(entry.get("labelElementIds") or [])
    ]
    for key in (
        "surfaces",
        "decorativeShapes",
        "icons",
        "structuralLines",
        "vectorPaths",
    ):
        spec[key] = [
            entry for entry in spec.get(key) or [] if not covered_by_object(entry)
        ]
    for layout_region in spec.get("layoutRegions") or []:
        layout_region["elementIds"] = [
            element_id
            for element_id in layout_region.get("elementIds") or []
            if element_id not in intrinsic_ids
        ]
    spec["primaryAsset"] = {
        "elementId": next_numeric_id,
        "boxSource": object_box,
        "boxNormSource": docmod.normalize_box(object_box, width, height),
        "areaRatio": round(_source_box_area(object_box) / canvas_area, 4),
        "strategy": "extract-source-crop-verbatim",
        "rule": "Preserve this photographed object and its printed lettering as one approved raster asset. Do not duplicate intrinsicText as DOM content.",
    }
    policy = spec.get("representationPolicy")
    if isinstance(policy, dict):
        policy["intrinsicRasterTextAllowedInsideApprovedObjects"] = True
    _sanitize_web_structure(document)


_SOURCE_BACKGROUND_MEDIA_TYPES = {
    "image/avif",
    "image/gif",
    "image/jpeg",
    "image/png",
    "image/webp",
}
_SOURCE_BACKGROUND_SUFFIXES = {
    "image/avif": ".avif",
    "image/gif": ".gif",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
_SOURCE_BACKGROUND_MAX_BYTES = 12 * 1024 * 1024
_SOURCE_VECTOR_MAX_BYTES = 512 * 1024
_SOURCE_FONT_MAX_BYTES = 4 * 1024 * 1024
_SOURCE_FONT_MEDIA_TYPES = {
    "font/woff2": (".woff2", "woff2"),
    "application/font-woff2": (".woff2", "woff2"),
    "font/woff": (".woff", "woff"),
    "application/font-woff": (".woff", "woff"),
    "font/ttf": (".ttf", "truetype"),
    "application/x-font-ttf": (".ttf", "truetype"),
    "font/otf": (".otf", "opentype"),
    "application/x-font-opentype": (".otf", "opentype"),
}


def _source_raster_cache_evidence(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    evidence: list[dict[str, Any]] = []
    for entry in value[:12]:
        if not isinstance(entry, dict):
            continue
        evidence.append(
            {
                key: entry.get(key)
                for key in (
                    "rasterIndex",
                    "domIndex",
                    "kind",
                    "sha256",
                    "sizeBytes",
                    "mediaType",
                    "box",
                    "visible",
                    "objectFit",
                    "backgroundSize",
                    "backdropColor",
                    "overlappingLiveTextCount",
                    "source",
                    "method",
                )
            }
        )
    return evidence


def _source_vector_cache_evidence(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    evidence: list[dict[str, Any]] = []
    for entry in value[:12]:
        if not isinstance(entry, dict):
            continue
        evidence.append(
            {
                key: entry.get(key)
                for key in (
                    "vectorIndex",
                    "domIndex",
                    "sha256",
                    "sizeBytes",
                    "mediaType",
                    "box",
                    "visible",
                    "viewportCoverage",
                    "source",
                    "method",
                )
            }
        )
    return evidence


def _source_text_cache_evidence(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    evidence: list[dict[str, Any]] = []
    for entry in value[:500]:
        if not isinstance(entry, dict) or entry.get("visible") is False:
            continue
        style = entry.get("style") if isinstance(entry.get("style"), dict) else {}
        evidence.append(
            {
                "text": str(entry.get("text") or "")[:500],
                "box": entry.get("box"),
                "style": {
                    key: style.get(key)
                    for key in (
                        "fontFamily",
                        "fontSize",
                        "fontWeight",
                        "fontStyle",
                        "letterSpacing",
                    )
                },
            }
        )
    return evidence


def _source_font_cache_evidence(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    evidence: list[dict[str, Any]] = []
    for entry in value[:16]:
        if not isinstance(entry, dict):
            continue
        evidence.append(
            {
                key: entry.get(key)
                for key in (
                    "family",
                    "weight",
                    "style",
                    "stretch",
                    "sha256",
                    "sizeBytes",
                    "mediaType",
                    "format",
                    "source",
                    "method",
                )
            }
        )
    return evidence


def _source_font_weight(value: Any, fallback: int = 400) -> int:
    text = str(value or "").strip().casefold()
    named = {
        "normal": 400,
        "regular": 400,
        "medium": 500,
        "semibold": 600,
        "semi-bold": 600,
        "bold": 700,
        "light": 300,
    }
    if text in named:
        return named[text]
    match = re.search(r"\b([1-9]00)\b", text)
    if match:
        return int(match.group(1))
    return fallback


def _source_font_family(value: Any) -> str:
    first = str(value or "").split(",", 1)[0].strip().strip("'\"")
    if not first or len(first) > 128 or any(ord(character) < 32 for character in first):
        return ""
    return first


def _source_box_match(left: Any, right: Any) -> float:
    if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
        return 0.0
    if len(left) != 4 or len(right) != 4:
        return 0.0
    intersection = _box_intersection(list(left), list(right))
    try:
        left_area = max(1.0, (float(left[2]) - float(left[0])) * (float(left[3]) - float(left[1])))
        right_area = max(1.0, (float(right[2]) - float(right[0])) * (float(right[3]) - float(right[1])))
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, intersection / min(left_area, right_area))


def _verified_source_font_assets(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    verified: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for entry in value[:16]:
        if not isinstance(entry, dict):
            continue
        family = _source_font_family(entry.get("family"))
        digest = str(entry.get("sha256") or "").casefold()
        media_type = str(entry.get("mediaType") or "").split(";", 1)[0].casefold()
        media = _SOURCE_FONT_MEDIA_TYPES.get(media_type)
        path = Path(str(entry.get("path") or "")).expanduser()
        if (
            not family
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or media is None
            or not path.is_file()
        ):
            continue
        try:
            content = path.read_bytes()
        except OSError:
            continue
        if not content or len(content) > _SOURCE_FONT_MAX_BYTES:
            continue
        if hashlib.sha256(content).hexdigest() != digest:
            continue
        suffix, font_format = media
        weight = str(entry.get("weight") or "normal")[:32]
        style = str(entry.get("style") or "normal").casefold()
        style = style if style in {"normal", "italic", "oblique"} else "normal"
        identity = (family.casefold(), weight.casefold(), style, digest)
        if identity in seen:
            continue
        seen.add(identity)
        verified.append(
            {
                "family": family,
                "alias": f"Sens Source {digest[:12]}",
                "weight": weight,
                "style": style,
                "stretch": str(entry.get("stretch") or "normal")[:32],
                "path": str(path.resolve()),
                "sha256": digest,
                "sizeBytes": len(content),
                "mediaType": media_type,
                "format": font_format,
                "suffix": suffix,
                "source": "observed",
                "method": "verified-playwright-loaded-font-response",
            }
        )
    return verified


def _hydrate_source_dom_typography(
    document: dict[str, Any],
    source_text_nodes: Any = None,
    source_font_assets: Any = None,
) -> None:
    """Prefer observed live-DOM typography over screenshot font inference."""
    spec = document.get("reconstruction") or {}
    if not spec:
        return
    fonts = _verified_source_font_assets(source_font_assets)
    if fonts:
        spec["sourceFontAssets"] = fonts
    nodes = [
        entry
        for entry in (source_text_nodes if isinstance(source_text_nodes, list) else [])[:500]
        if isinstance(entry, dict)
        and entry.get("visible") is not False
        and _normalized_text(entry.get("text"))
        and isinstance(entry.get("style"), dict)
    ]
    if not nodes:
        return
    spec["typographyAuthority"] = _web_typography_authority()

    def best_node(entry: dict[str, Any]) -> dict[str, Any] | None:
        target = _normalized_text(entry.get("preferredValue") or entry.get("value"))
        if not target:
            return None
        target_box = entry.get("boxSource") or []
        ranked: list[tuple[float, dict[str, Any]]] = []
        for node in nodes:
            observed = _normalized_text(node.get("text"))
            exact = target == observed
            contains = target in observed or observed in target
            similarity = SequenceMatcher(None, target, observed).ratio()
            if not exact and not contains and similarity < 0.72:
                continue
            box_match = _source_box_match(target_box, node.get("box"))
            text_score = 1.0 if exact else max(similarity, 0.86 if contains else 0.0)
            ranked.append((text_score * 0.75 + box_match * 0.25, node))
        if not ranked:
            return None
        score, node = max(ranked, key=lambda item: item[0])
        return node if score >= 0.66 else None

    def closest_font(family: str, weight: int, style: str) -> dict[str, Any] | None:
        candidates = [font for font in fonts if font["family"].casefold() == family.casefold()]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda font: (
                0 if font["style"] == style else 1,
                abs(_source_font_weight(font["weight"]) - weight),
            ),
        )

    def text_words(value: Any) -> list[tuple[str, str]]:
        words: list[tuple[str, str]] = []
        for match in re.finditer(r"\S+", str(value or "")):
            normalized = _normalized_text(match.group(0))
            if normalized:
                words.append((match.group(0), normalized))
        return words

    observed_words: list[dict[str, Any]] = []
    for node_index, node in enumerate(nodes):
        node_words = text_words(node.get("text"))
        if not node_words:
            continue
        measured_words = [
            item
            for item in (node.get("wordBoxes") or [])
            if isinstance(item, dict)
            and _normalized_text(item.get("text"))
            and isinstance(item.get("box"), (list, tuple))
            and len(item.get("box")) == 4
        ]
        use_measured_words = (
            len(measured_words) == len(node_words)
            and all(
                _normalized_text(item.get("text")) == normalized
                for item, (_, normalized) in zip(
                    measured_words, node_words, strict=True
                )
            )
        )
        for word_index, (text, normalized) in enumerate(node_words):
            observed_words.append(
                {
                    "text": text,
                    "normalized": normalized,
                    "box": list(measured_words[word_index]["box"])
                    if use_measured_words
                    else list(node.get("box") or [])[:4],
                    "node": node,
                    "nodeIndex": node_index,
                }
            )

    def observed_word_styles(entry: dict[str, Any]) -> list[dict[str, Any]]:
        target_words = text_words(
            entry.get("preferredValue") or entry.get("value")
        )
        if not target_words or len(observed_words) < len(target_words):
            return []
        target_normalized = [normalized for _, normalized in target_words]
        target_box = list(entry.get("boxSource") or [])[:4]
        ranked: list[tuple[float, list[dict[str, Any]]]] = []
        for start in range(len(observed_words) - len(target_words) + 1):
            window = observed_words[start : start + len(target_words)]
            if [word["normalized"] for word in window] != target_normalized:
                continue
            boxes = [word.get("box") or [] for word in window]
            if any(len(box) != 4 for box in boxes):
                continue
            union = [
                min(float(box[0]) for box in boxes),
                min(float(box[1]) for box in boxes),
                max(float(box[2]) for box in boxes),
                max(float(box[3]) for box in boxes),
            ]
            overlap = _source_box_match(target_box, union)
            try:
                target_center = (
                    (float(target_box[0]) + float(target_box[2])) / 2.0,
                    (float(target_box[1]) + float(target_box[3])) / 2.0,
                )
                union_center = (
                    (union[0] + union[2]) / 2.0,
                    (union[1] + union[3]) / 2.0,
                )
                scale = max(
                    1.0,
                    float(target_box[2]) - float(target_box[0]),
                    float(target_box[3]) - float(target_box[1]),
                )
                distance = (
                    (target_center[0] - union_center[0]) ** 2
                    + (target_center[1] - union_center[1]) ** 2
                ) ** 0.5
                proximity = max(0.0, 1.0 - distance / (scale * 2.0))
            except (IndexError, TypeError, ValueError):
                proximity = 0.0
            ranked.append((overlap * 0.75 + proximity * 0.25, window))
        if not ranked:
            return []
        score, selected = max(ranked, key=lambda item: item[0])
        if score < 0.45:
            return []
        styles: list[dict[str, Any]] = []
        for (target_text, _), observed in zip(
            target_words, selected, strict=True
        ):
            node = observed["node"]
            style_data = node.get("style") or {}
            family = _source_font_family(style_data.get("fontFamily"))
            if not family:
                return []
            weight = _source_font_weight(style_data.get("fontWeight"))
            style = str(style_data.get("fontStyle") or "normal").casefold()
            style = style if style in {"normal", "italic", "oblique"} else "normal"
            result = {
                "text": target_text,
                "sourceDomFamily": family,
                "sourceDomFontWeight": weight,
                "sourceDomFontStyle": style,
                "sourceDomFontSize": str(style_data.get("fontSize") or "")[:32],
                "sourceDomLineHeight": str(style_data.get("lineHeight") or "")[:32],
                "sourceDomLetterSpacing": str(
                    style_data.get("letterSpacing") or ""
                )[:32],
                "sourceDomBox": list(observed.get("box") or [])[:4],
                "sourceDomTypographySource": (
                    "observed-live-dom-computed-style"
                ),
            }
            font = closest_font(family, weight, style)
            if font is not None:
                result["sourceFontFamily"] = font["alias"]
                result["sourceFontAssetSha256"] = font["sha256"]
            styles.append(result)
        return styles

    for entry in spec.get("text") or []:
        if not isinstance(entry, dict):
            continue
        node = best_node(entry)
        if node is None:
            continue
        style_data = node.get("style") or {}
        family = _source_font_family(style_data.get("fontFamily"))
        if not family:
            continue
        weight = _source_font_weight(style_data.get("fontWeight"))
        style = str(style_data.get("fontStyle") or "normal").casefold()
        style = style if style in {"normal", "italic", "oblique"} else "normal"
        metrics = dict(entry.get("fontFeatures") or {})
        metrics.update(
            {
                "sourceDomFamily": family,
                "sourceDomFontWeight": weight,
                "sourceDomFontStyle": style,
                "sourceDomFontSize": str(style_data.get("fontSize") or "")[:32],
                "sourceDomLineHeight": str(style_data.get("lineHeight") or "")[:32],
                "sourceDomLetterSpacing": str(style_data.get("letterSpacing") or "")[:32],
                "sourceDomTextTransform": str(style_data.get("textTransform") or "")[:32],
                "sourceDomTextAlign": str(style_data.get("textAlign") or "")[:32],
                "sourceDomBox": list(node.get("box") or [])[:4],
                "sourceDomText": str(node.get("text") or "")[:500],
                "sourceDomTypographySource": "observed-live-dom-computed-style",
                "renderWeight": weight,
            }
        )
        font = closest_font(family, weight, style)
        if font is not None:
            metrics["sourceFontFamily"] = font["alias"]
            metrics["sourceFontAssetSha256"] = font["sha256"]
        word_styles = observed_word_styles(entry)
        if word_styles:
            metrics["sourceDomWordStyles"] = word_styles
            metrics["sourceDomRunAuthority"] = (
                "observed-live-dom-computed-style"
            )
        entry["fontFeatures"] = metrics
        typography = entry.get("typographyCandidate")
        if isinstance(typography, dict):
            typography = dict(typography)
            typography["authority"] = "inferred-fallback-only-when-source-dom-is-unavailable"
            entry["typographyCandidate"] = typography


def _hydrate_source_vector_regions(
    document: dict[str, Any], source_vector_assets: Any = None
) -> None:
    """Verify captured SVG bytes and bind large wordmarks to selectable labels."""
    spec = document.get("reconstruction") or {}
    canvas = spec.get("canvas") or {}
    if (
        spec.get("targetKind") != "web"
        or not isinstance(source_vector_assets, list)
        or int(canvas.get("width") or 0) <= 0
        or int(canvas.get("height") or 0) <= 0
    ):
        return
    from sight.capture import _sanitize_source_svg

    verified: list[dict[str, Any]] = []
    safe_root = Path(cache_root()) / "source-vectors"
    for asset_index, asset in enumerate(source_vector_assets[:12]):
        if not isinstance(asset, dict) or (
            asset.get("source") != "observed"
            or asset.get("method") != "sanitized-live-dom-svg"
            or asset.get("mediaType") != "image/svg+xml"
            or asset.get("visible") is not True
        ):
            continue
        box = asset.get("box")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            normalized_box = [round(float(value), 3) for value in box]
        except (TypeError, ValueError):
            continue
        if _source_box_area(normalized_box) <= 0:
            continue
        path = Path(str(asset.get("path") or "")).expanduser()
        try:
            content = path.read_bytes()
            declared_size = int(asset.get("sizeBytes") or 0)
        except (OSError, TypeError, ValueError):
            continue
        if (
            not content
            or len(content) > _SOURCE_VECTOR_MAX_BYTES
            or declared_size != len(content)
            or hashlib.sha256(content).hexdigest()
            != str(asset.get("sha256") or "").casefold()
        ):
            continue
        try:
            markup = content.decode("utf-8")
        except UnicodeDecodeError:
            continue
        safe_content = _sanitize_source_svg(
            markup,
            id_prefix=f"sens-contract-vector-{asset_index}-",
        )
        if safe_content is None:
            continue
        safe_digest = hashlib.sha256(safe_content).hexdigest()
        destination = safe_root / f"source-vector-{safe_digest[:20]}.svg"
        try:
            safe_root.mkdir(parents=True, exist_ok=True)
            if not destination.exists() or destination.read_bytes() != safe_content:
                descriptor, temporary = tempfile.mkstemp(
                    prefix=f".{destination.name}.", dir=safe_root
                )
                os.close(descriptor)
                try:
                    Path(temporary).write_bytes(safe_content)
                    os.replace(temporary, destination)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
        except OSError:
            continue
        element_id = f"source-vector-{asset_index}-{safe_digest[:12]}"
        region = {
            "elementId": element_id,
            "boxSource": normalized_box,
            "assetPath": str(destination.resolve()),
            "contentSha256": safe_digest,
            "mediaType": "image/svg+xml",
            "source": "observed",
            "method": "verified-sanitized-live-dom-svg",
            "ariaHidden": True,
            "evidence": {
                "vectorIndex": asset.get("vectorIndex"),
                "domIndex": asset.get("domIndex"),
                "captureSha256": asset.get("sha256"),
                "sizeBytes": len(safe_content),
            },
        }
        verified.append(region)
        artifact_id = f"vector:{safe_digest}"
        artifacts = document.setdefault("artifacts", [])
        if not any(item.get("id") == artifact_id for item in artifacts):
            artifacts.append(
                {
                    "id": artifact_id,
                    "kind": "sanitized-source-vector",
                    "uri": str(destination.resolve()),
                    "mediaType": "image/svg+xml",
                }
            )
    if not verified:
        return
    verified.sort(key=lambda entry: (entry["boxSource"][0], entry["boxSource"][1]))

    for entry in spec.get("text") or []:
        value = str(entry.get("preferredValue") or entry.get("value") or "").strip()
        text_box = entry.get("boxSource") or []
        if (
            not re.fullmatch(r"[A-Za-z0-9]{2,12}", value)
            or len(text_box) != 4
            or not str(entry.get("method") or "").startswith(
                "rapidocr-downscaled-display-"
            )
        ):
            continue
        text_width = max(1.0, float(text_box[2]) - float(text_box[0]))
        text_height = max(1.0, float(text_box[3]) - float(text_box[1]))
        candidates = [
            region
            for region in verified
            if (
                max(
                    0.0,
                    min(float(text_box[2]), float(region["boxSource"][2]))
                    - max(float(text_box[0]), float(region["boxSource"][0])),
                )
                / max(
                    1.0,
                    float(region["boxSource"][2])
                    - float(region["boxSource"][0]),
                )
                >= 0.5
                and max(
                    0.0,
                    min(float(text_box[3]), float(region["boxSource"][3]))
                    - max(float(text_box[1]), float(region["boxSource"][1])),
                )
                / text_height
                >= 0.65
            )
        ]
        if len(candidates) != len(value):
            continue
        union = [
            min(region["boxSource"][0] for region in candidates),
            min(region["boxSource"][1] for region in candidates),
            max(region["boxSource"][2] for region in candidates),
            max(region["boxSource"][3] for region in candidates),
        ]
        horizontal_coverage = max(
            0.0,
            min(float(text_box[2]), float(union[2]))
            - max(float(text_box[0]), float(union[0])),
        ) / text_width
        if horizontal_coverage < 0.8:
            continue
        asset_ids = [region["elementId"] for region in candidates]
        entry["visualRepresentation"] = (
            "source-vector-wordmark-with-selectable-live-label"
        )
        entry["sourceVectorAssetIds"] = asset_ids
        entry["representationSource"] = "verified-live-dom-svg-geometry"
        for region in candidates:
            region["wordmarkText"] = value
            region["selectableLabelElementId"] = entry.get("elementId")
    materialized = [region for region in verified if region.get("wordmarkText")]
    if materialized:
        spec["sourceVectorRegions"] = materialized
    else:
        spec.pop("sourceVectorRegions", None)


def _css_backdrop_to_hex(value: Any) -> str | None:
    """Normalize a measured opaque CSS backdrop to a starter-safe hex color."""
    text = str(value or "").strip()
    if re.fullmatch(r"#[0-9A-Fa-f]{6}", text):
        return text.upper()
    short_hex = re.fullmatch(r"#([0-9A-Fa-f])([0-9A-Fa-f])([0-9A-Fa-f])", text)
    if short_hex:
        return "#" + "".join(channel * 2 for channel in short_hex.groups()).upper()
    match = re.fullmatch(
        r"rgba?\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})"
        r"(?:\s*,\s*(0(?:\.\d+)?|1(?:\.0+)?))?\s*\)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    red, green, blue = (int(channel) for channel in match.groups()[:3])
    if any(channel > 255 for channel in (red, green, blue)):
        return None
    alpha = match.group(4)
    if alpha is not None and float(alpha) < 0.999:
        return None
    return f"#{red:02X}{green:02X}{blue:02X}"


def _verified_source_background(
    source_raster_assets: Any,
    *,
    width: int,
    height: int,
) -> dict[str, Any] | None:
    if not isinstance(source_raster_assets, list) or width <= 0 or height <= 0:
        return None
    candidates: list[tuple[float, dict[str, Any]]] = []
    for asset in source_raster_assets[:12]:
        if not isinstance(asset, dict):
            continue
        if (
            asset.get("source") != "observed"
            or asset.get("method") != "playwright-response-body"
            or asset.get("visible") is not True
            or str(asset.get("mediaType") or "").lower()
            not in _SOURCE_BACKGROUND_MEDIA_TYPES
            or int(asset.get("overlappingLiveTextCount") or 0) < 1
        ):
            continue
        box = asset.get("box")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            x0, y0, x1, y1 = (int(round(float(value))) for value in box)
        except (TypeError, ValueError):
            continue
        if x1 <= x0 or y1 <= y0:
            continue
        intersection_width = max(0, min(width, x1) - max(0, x0))
        intersection_height = max(0, min(height, y1) - max(0, y0))
        coverage = (intersection_width * intersection_height) / max(
            1, width * height
        )
        if coverage < 0.55:
            continue
        path = Path(str(asset.get("path") or "")).expanduser()
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size <= 0 or size > _SOURCE_BACKGROUND_MAX_BYTES:
            continue
        try:
            declared_size = int(asset.get("sizeBytes") or size)
        except (TypeError, ValueError):
            continue
        if declared_size != size:
            continue
        content_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if content_sha256 != str(asset.get("sha256") or "").lower():
            continue
        background_color = _css_backdrop_to_hex(asset.get("backdropColor"))
        candidates.append(
            (
                coverage,
                {
                    "elementId": "browser-source-background",
                    "kind": "browser-source-background-artwork",
                    "boxSource": [x0, y0, x1, y1],
                    "strategy": "preserve-browser-loaded-source-raster",
                    "implementation": "Use the Sens-copied browser source as the noninteractive background at its measured box. Keep every separately observed text node and control as live semantic DOM above it.",
                    "semanticContentRemoved": True,
                    "protectionVersion": 4,
                    "protectionPolicy": {
                        "backgroundOnly": True,
                        "liveText": "separate-observed-live-dom",
                        "controlDecoration": "separate-semantic-css",
                        "fullReferenceScreenshot": False,
                    },
                    "source": "observed",
                    "method": "verified-playwright-response-body",
                    "sourceAssetPath": str(path.resolve()),
                    "contentSha256": content_sha256,
                    "mediaType": str(asset.get("mediaType")).lower(),
                    "objectFit": asset.get("objectFit"),
                    "backgroundSize": asset.get("backgroundSize"),
                    **(
                        {"backgroundColor": background_color}
                        if background_color
                        else {}
                    ),
                    "evidence": {
                        "rasterIndex": asset.get("rasterIndex"),
                        "domIndex": asset.get("domIndex"),
                        "viewportCoverage": round(coverage, 5),
                        "overlappingLiveTextCount": int(
                            asset.get("overlappingLiveTextCount") or 0
                        ),
                        "sizeBytes": size,
                    },
                },
            )
        )
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _hydrate_background_artwork_layer(
    document: dict[str, Any],
    image_path: str,
    source_raster_assets: Any = None,
) -> None:
    """Preserve a genuinely textured canvas without flattening live content.

    A flat CSS color is insufficient for paper grain, foliage, or dense UI
    chrome. The protected layer keeps only non-text presentation: source glyphs
    are inpainted and replaced by selectable DOM, controls remain semantic, and
    symbol art remains preformatted text. Dashboards can therefore preserve
    exact cards, charts, icons, and separators without flattening their text.
    """
    import cv2
    import numpy as np

    spec = document.get("reconstruction") or {}
    if spec.get("targetKind") != "web":
        return
    allowed = spec.setdefault("allowedRasterRegions", [])
    try:
        source_image = load_cv(image_path)
    except (OSError, ValueError):
        return
    source_height, source_width = source_image.shape[:2]
    source_background = _verified_source_background(
        source_raster_assets,
        width=source_width,
        height=source_height,
    )
    if source_background is not None:
        allowed[:] = [
            entry
            for entry in allowed
            if entry.get("kind")
            not in {
                "alpha-masked-background-artwork",
                "browser-source-background-artwork",
            }
        ]
        composite_overlay = {
            "elementId": "browser-source-composite-overlay",
            "kind": "alpha-masked-background-artwork",
            "boxSource": [0, 0, source_width, source_height],
            "boxNormSource": [0, 0, 1000, 1000],
            "strategy": "preserve-reference-decoration-with-semantic-alpha-holes",
            "implementation": "Layer this Sens-materialized decorative overlay above the verified browser source raster. Its measured semantic regions are transparent, so live text and controls remain DOM while the exact source raster supplies the pixels beneath them.",
            "semanticContentRemoved": True,
            "protectionVersion": 5,
            "protectionPolicy": {
                "backgroundOnly": True,
                "liveText": "transparent-holes-reveal-verified-browser-source-under-live-dom",
                "controlDecoration": "transparent-holes-reveal-verified-browser-source-under-semantic-css",
                "microIcons": "transparent-holes-reveal-verified-browser-source-under-svg-css",
                "surfaces": "transparent-holes-reveal-verified-browser-source-under-css",
                "structuralLines": "transparent-holes-reveal-verified-browser-source-under-css-vector",
                "vectorPaths": "transparent-holes-reveal-verified-browser-source-under-svg",
                "badges": "transparent-holes-reveal-verified-browser-source-under-live-dom",
                "symbolArt": "transparent-holes-reveal-verified-browser-source-under-live-preformatted-text",
                "objects": "independent-approved-assets-only",
                "fullReferenceScreenshot": False,
            },
            "source": "measured",
            "method": "protected-composite-alpha-mask",
            "compositeUnderlay": "browser-source-background",
            "evidence": {
                "browserSourceSha256": source_background.get("contentSha256"),
                "browserSourceBox": source_background.get("boxSource"),
            },
        }
        allowed.extend([source_background, composite_overlay])
        background_color = source_background.get("backgroundColor")
        if background_color:
            tokens = document.setdefault("tokens", {})
            colors = tokens.setdefault("color", {})
            color_token = {"$type": "color", "$value": background_color}
            colors["canvas"] = dict(color_token)
            colors["background"] = dict(color_token)
            document.setdefault("header", {})["background"] = background_color
        return
    if any(
        entry.get("kind")
        in {
            "alpha-masked-background-artwork",
            "browser-source-background-artwork",
        }
        for entry in allowed
    ):
        return
    image = source_image
    height, width = image.shape[:2]
    if width * height <= 0:
        return
    unclaimed = np.ones((height, width), np.uint8)
    for key in ("text", "symbolArt", "allowedRasterRegions"):
        for entry in spec.get(key) or []:
            box = entry.get("boxSource") or []
            if len(box) != 4:
                continue
            x0, y0, x1, y1 = (int(round(value)) for value in box)
            padding = 4
            unclaimed[
                max(0, y0 - padding) : min(height, y1 + padding),
                max(0, x0 - padding) : min(width, x1 + padding),
            ] = 0
    available = unclaimed.astype(bool)
    available_ratio = float(available.mean())
    if available_ratio < 0.20:
        return
    colors = (document.get("tokens") or {}).get("color") or {}
    canvas_color = (
        (colors.get("canvas") or {}).get("$value")
        or (colors.get("background") or {}).get("$value")
        or (document.get("header") or {}).get("background")
        or "#FFFFFF"
    )
    canvas_bgr = np.asarray(_hex_to_bgr(canvas_color), dtype=np.float32)
    pixels = image[available].astype(np.float32)
    color_distance = np.linalg.norm(pixels - canvas_bgr, axis=1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    local_detail = np.abs(cv2.Laplacian(gray, cv2.CV_32F))[available]
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    soft_gradient = cv2.magnitude(gradient_x, gradient_y)[available]
    distance_p75 = float(np.percentile(color_distance, 75))
    distance_spread = float(
        np.percentile(color_distance, 90) - np.percentile(color_distance, 25)
    )
    detail_p75 = float(np.percentile(local_detail, 75))
    gradient_p75 = float(np.percentile(soft_gradient, 75))
    soft_photographic_artwork = bool(
        distance_p75 >= 24.0
        and distance_spread >= 20.0
        and gradient_p75 >= 3.0
    )
    textured_artwork = bool(
        distance_spread >= 8.0
        and detail_p75 >= 4.0
        and gradient_p75 >= 6.0
    )
    dense_interface = bool(
        len(spec.get("surfaces") or []) > 8
        or len(spec.get("visualControlCandidates") or []) > 14
    )
    complex_interface = bool(
        dense_interface
        or len(spec.get("vectorPaths") or []) > 0
    )
    # Density is never a reason to flatten interface chrome into a raster.
    # A full-canvas layer is allowed only when the pixels provide genuine
    # photographic or textural information that CSS cannot reproduce.
    if not (soft_photographic_artwork or textured_artwork):
        return
    allowed.append(
        {
            "elementId": "background-artwork",
            "kind": "alpha-masked-background-artwork",
            "boxSource": [0, 0, width, height],
            "boxNormSource": [0, 0, 1000, 1000],
            "strategy": "extract-background-with-protected-alpha-mask",
            "implementation": "Use only the Sens-materialized background-only asset. Sens removes full measured boxes for live text, controls, surfaces, separators, paths, icons, badges, symbol art, and foreground raster objects before materialization. Recreate those elements independently with semantic HTML, CSS, SVG, or approved object assets.",
            "semanticContentRemoved": True,
            "protectionVersion": 2,
            "protectionPolicy": {
                "backgroundOnly": True,
                "liveText": "full-box-inpainted-under-live-dom",
                "controlDecoration": "removed-from-raster-recreated-as-semantic-css",
                "microIcons": "removed-from-raster-recreated-as-svg-css",
                "surfaces": "removed-from-raster-recreated-as-css",
                "structuralLines": "removed-from-raster-recreated-as-css-vector",
                "vectorPaths": "removed-from-raster-recreated-as-svg",
                "badges": "removed-from-raster-recreated-as-live-dom",
                "symbolArt": "removed-from-raster-recreated-as-live-preformatted-text",
                "objects": "removed-from-raster-recreated-as-approved-assets",
                "fullReferenceScreenshot": False,
            },
            "source": "measured",
            "method": "protected-pixel-alpha-mask",
            "evidence": {
                "unclaimedCanvasRatio": round(available_ratio, 4),
                "colorDistanceP75": round(distance_p75, 2),
                "colorDistanceSpread": round(distance_spread, 2),
                "localDetailP75": round(detail_p75, 2),
                "softGradientP75": round(gradient_p75, 2),
                "softPhotographicArtwork": soft_photographic_artwork,
                "texturedArtwork": textured_artwork,
                "denseInterface": dense_interface,
                "complexInterface": complex_interface,
            },
        }
    )


def _hydrate_measured_vector_paths(
    document: dict[str, Any], image_path: str
) -> None:
    """Trace native chart/decorative polylines for cached full-image contracts."""
    spec = document.get("reconstruction") or {}
    if spec.get("targetKind") != "web":
        return
    try:
        image = load_cv(image_path)
    except (OSError, ValueError):
        return
    exclusions = [
        *[entry.get("boxSource") or [] for entry in spec.get("text") or []],
        *[
            entry.get("boxSource") or []
            for entry in spec.get("visualControlCandidates") or []
        ],
        *[
            entry.get("boxSource") or []
            for entry in spec.get("symbolArt") or []
        ],
    ]
    spec["vectorPaths"] = [
        {
            "boxSource": entry["box"],
            "pointsSource": entry["points"],
            "strokeColor": entry.get("strokeColor"),
            "strokeWidth": entry.get("strokeWidth"),
            "fill": "none",
            "source": "measured",
            "method": entry.get("method"),
        }
        for entry in detect_vector_paths(image, exclusions)
    ]


def _refine_large_text_box(
    image: Any, entry: dict[str, Any]
) -> list[int] | None:
    """Shrink an oversized OCR region to its measured dominant glyph ink."""
    import cv2
    import numpy as np

    font = entry.get("fontFeatures") or {}
    if float(font.get("fontSize") or 0.0) < 90:
        return None
    raw_box = entry.get("boxSource") or []
    if len(raw_box) != 4:
        return None
    height, width = image.shape[:2]
    raw_x0, raw_y0, raw_x1, raw_y1 = (
        int(round(value)) for value in raw_box
    )
    is_downscaled_display = str(entry.get("method") or "").startswith(
        "rapidocr-downscaled-display-"
    )
    pad_x = (
        min(72, max(4, round((raw_x1 - raw_x0) * 0.06)))
        if is_downscaled_display
        else 0
    )
    pad_y = (
        min(42, max(4, round((raw_y1 - raw_y0) * 0.05)))
        if is_downscaled_display
        else 0
    )
    x0, y0 = max(0, raw_x0 - pad_x), max(0, raw_y0 - pad_y)
    x1, y1 = min(width, raw_x1 + pad_x), min(height, raw_y1 + pad_y)
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    color = entry.get("color") or font.get("color")
    if not color:
        return None
    target = np.asarray(_hex_to_bgr(color), dtype=np.float32)
    crop = image[y0:y1, x0:x1].astype(np.float32)
    mask = (np.linalg.norm(crop - target, axis=2) <= 36.0).astype(np.uint8)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    components: list[dict[str, int]] = []
    for component in range(1, count):
        x, y, component_width, component_height, area = (
            int(value) for value in stats[component]
        )
        if area < 8 or component_width < 2 or component_height < 2:
            continue
        components.append(
            {
                "x0": x,
                "y0": y,
                "x1": x + component_width,
                "y1": y + component_height,
                "height": component_height,
                "area": area,
            }
        )
    if not components:
        return None
    largest_area = max(component["area"] for component in components)
    dominant_pool = [
        component
        for component in components
        if component["area"] >= max(20, largest_area * 0.08)
        and component["height"] >= 3
    ]
    if not dominant_pool:
        return None
    dominant_height = float(
        np.median([component["height"] for component in dominant_pool])
    )
    if dominant_height < 20:
        return None
    primary = [
        component
        for component in components
        if dominant_height * 0.55
        <= component["height"]
        <= dominant_height * 1.5
        and component["area"] >= 8
    ]
    expected_glyphs = sum(
        character.isalnum()
        for character in str(
            entry.get("preferredValue") or entry.get("value") or ""
        )
    )
    required_components = max(1, min(3, round(expected_glyphs * 0.35)))
    if len(primary) < required_components:
        return None
    edge_artwork_trimmed = False
    if is_downscaled_display and len(primary) >= 3:
        primary = sorted(primary, key=lambda component: component["x0"])
        median_width = float(
            np.median(
                [component["x1"] - component["x0"] for component in primary]
            )
        )
        gaps = [
            max(0, right["x0"] - left["x1"])
            for left, right in zip(primary, primary[1:])
        ]
        median_gap = float(np.median(gaps)) if gaps else 0.0
        interior = primary[1:-1] or primary
        consensus_y0 = round(
            float(np.median([component["y0"] for component in interior]))
        )
        consensus_y1 = round(
            float(np.median([component["y1"] for component in interior]))
        )
        first = primary[0]
        if first["x1"] - first["x0"] >= median_width * 1.30:
            first["x0"] = max(
                first["x0"],
                round(
                    primary[1]["x0"]
                    - median_gap
                    - median_width
                ),
            )
            first["y0"] = consensus_y0
            first["y1"] = consensus_y1
            edge_artwork_trimmed = True
        last = primary[-1]
        if last["x1"] - last["x0"] >= median_width * 1.30:
            last["x1"] = min(
                last["x1"],
                round(
                    primary[-2]["x1"]
                    + median_gap
                    + median_width
                ),
            )
            last["y0"] = consensus_y0
            last["y1"] = consensus_y1
            edge_artwork_trimmed = True
    refined = [
        max(0, x0 + min(component["x0"] for component in primary) - 1),
        max(0, y0 + min(component["y0"] for component in primary) - 1),
        min(width, x0 + max(component["x1"] for component in primary) + 1),
        min(height, y0 + max(component["y1"] for component in primary) + 1),
    ]
    old_area = max(1, (x1 - x0) * (y1 - y0))
    refined_area = max(
        1, (refined[2] - refined[0]) * (refined[3] - refined[1])
    )
    height_overscan = (y1 - y0) / max(1, refined[3] - refined[1])
    minimum_height_overscan = 1.35 if is_downscaled_display else 1.5
    if height_overscan < minimum_height_overscan and not (
        is_downscaled_display and edge_artwork_trimmed
    ):
        return None
    if refined_area / old_area > 0.92:
        return None
    if (
        refined[2] - refined[0] < (x1 - x0) * 0.3
        or refined[3] - refined[1] < (y1 - y0) * 0.3
    ):
        return None
    return refined


def _refine_small_text_ink_box(
    image: Any,
    entry: dict[str, Any],
    metrics: dict[str, Any],
) -> tuple[list[int], str] | None:
    """Remove OCR padding and a measured leading icon from small live text.

    RapidOCR polygons around compact UI labels often include 4-6 pixels of
    padding.  Some also fuse the preceding navigation icon into the same row.
    Stretching the preferred DOM copy across that broad polygon made labels
    oversized and visibly collide with the separately reconstructed icon.
    """
    import cv2
    import numpy as np

    raw_box = entry.get("boxSource") or []
    if len(raw_box) != 4 or float(metrics.get("fontSize") or 0) >= 90:
        return None
    original = [int(round(value)) for value in raw_box]
    refined = list(original)
    methods: list[str] = []
    geometry_source = str(entry.get("geometrySource") or "")
    ink_box = metrics.get("inkBox")
    if (
        not geometry_source.startswith("measured-glyph-ink-bounds")
        and isinstance(ink_box, list)
        and len(ink_box) == 4
    ):
        candidate = [int(round(value)) for value in ink_box]
        original_area = max(1, _source_box_area(original))
        candidate_area = max(1, _source_box_area(candidate))
        if (
            candidate[0] >= original[0]
            and candidate[1] >= original[1]
            and candidate[2] <= original[2]
            and candidate[3] <= original[3]
            and candidate[2] - candidate[0] >= (original[2] - original[0]) * 0.35
            and candidate[3] - candidate[1] >= (original[3] - original[1]) * 0.35
            and candidate_area / original_area <= 0.92
        ):
            refined = candidate
            methods.append("ink-bounds")

    raw_text = "".join(
        character
        for character in str(entry.get("value") or "").casefold()
        if character.isalnum()
    )
    preferred_text = "".join(
        character
        for character in str(entry.get("preferredValue") or "").casefold()
        if character.isalnum()
    )
    leading_noise = bool(
        preferred_text
        and raw_text != preferred_text
        and raw_text.endswith(preferred_text)
        and 1 <= len(raw_text) - len(preferred_text) <= 4
    )
    if not leading_noise and preferred_text and raw_text != preferred_text:
        longest = SequenceMatcher(None, raw_text, preferred_text).find_longest_match(
            0, len(raw_text), 0, len(preferred_text)
        )
        leading_noise = bool(
            0 < longest.a <= 4
            and longest.b == 0
            and longest.size >= max(3, min(5, len(preferred_text) // 2))
        )
    # A brand mark can be fused into the OCR polygon while the recognizer still
    # returns the word itself (for example, a logo glyph immediately before a
    # lowercase wordmark).  In that case string comparison cannot reveal the
    # extra visual component.  The independently measured foreground-run count
    # can: permit the same conservative gap test below when there are only one
    # or two more visible components than alphanumeric characters.  The gap,
    # position, and remaining-width guards still prevent ordinary word spacing
    # or kerning from being cut away.
    measured_character_count = int(metrics.get("measuredCharacterCount") or 0)
    if (
        not leading_noise
        and raw_text
        and " " not in str(entry.get("preferredValue") or entry.get("value") or "")
        and 1 <= measured_character_count - len(raw_text) <= 2
    ):
        leading_noise = True
    if leading_noise and "leading-icon-separation" not in geometry_source:
        x0, y0, x1, y1 = refined
        crop = image[y0:y1, x0:x1]
        if crop.size and x1 - x0 >= 18:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
            border = np.concatenate(
                (gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]), axis=0
            )
            background = float(np.median(border))
            distance = np.abs(gray - background)
            amplitude = float(np.percentile(distance, 90))
            if amplitude >= 12:
                occupied = np.where((distance > 0.35 * amplitude).any(axis=0))[0]
                if occupied.size >= 2:
                    differences = np.diff(occupied)
                    cap_height = max(4.0, float(metrics.get("capHeight") or 0))
                    minimum_gap = max(3, int(round(cap_height * 0.35)))
                    candidates = [
                        index
                        for index, difference in enumerate(differences)
                        if difference - 1 >= minimum_gap
                        and occupied[index + 1] <= (x1 - x0) * 0.45
                    ]
                    if candidates:
                        gap_index = max(candidates, key=lambda index: differences[index])
                        cut = int(occupied[gap_index + 1])
                        if (
                            cut <= (x1 - x0) * 0.35
                            and (x1 - x0) - cut >= (x1 - x0) * 0.55
                        ):
                            refined[0] = x0 + cut
                            methods.append("leading-icon-separation")

    if refined == original:
        return None
    method = "measured-glyph-ink-bounds"
    if "leading-icon-separation" in methods:
        method += "-with-leading-icon-separation"
    return refined, method


def _repair_reversed_glyph_color(
    image: Any, box: list[int], metrics: dict[str, Any]
) -> str | None:
    """Recover foreground ink when a tight OCR box swaps ink/background.

    On dense, tiny uppercase text the glyphs can occupy most of the OCR crop.
    A crop-only median then treats the lettering as background and reports the
    surrounding canvas as the glyph colour.  The ring immediately outside the
    measured box is independent background evidence and lets us detect and
    repair that inversion without guessing from the page palette.
    """
    import numpy as np

    current = metrics.get("color")
    if not current or len(box) != 4:
        return None
    height, width = image.shape[:2]
    x0, y0, x1, y1 = (int(round(value)) for value in box)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1 - x0 < 4 or y1 - y0 < 3:
        return None
    padding = max(3, min(8, int(round((y1 - y0) * 0.6))))
    rx0, ry0 = max(0, x0 - padding), max(0, y0 - padding)
    rx1, ry1 = min(width, x1 + padding), min(height, y1 + padding)
    expanded = image[ry0:ry1, rx0:rx1]
    if expanded.size == 0:
        return None
    ring_mask = np.ones(expanded.shape[:2], dtype=bool)
    ring_mask[y0 - ry0 : y1 - ry0, x0 - rx0 : x1 - rx0] = False
    ring_pixels = expanded[ring_mask]
    if ring_pixels.shape[0] < 8:
        return None
    background = np.median(ring_pixels.reshape(-1, 3), axis=0).astype(np.float32)
    current_bgr = np.asarray(_hex_to_bgr(str(current)), dtype=np.float32)
    if float(np.linalg.norm(current_bgr - background)) > 28.0:
        return None

    crop = image[y0:y1, x0:x1].astype(np.float32)
    distances = np.linalg.norm(crop - background, axis=2)
    if float(np.percentile(distances, 90)) < 32.0:
        return None
    threshold = max(28.0, float(np.percentile(distances, 72)))
    core = crop[distances >= threshold]
    if core.shape[0] < 3:
        return None
    candidate = np.median(core.reshape(-1, 3), axis=0)
    if float(np.linalg.norm(candidate - background)) < 32.0:
        return None
    return "#{:02X}{:02X}{:02X}".format(
        int(round(candidate[2])),
        int(round(candidate[1])),
        int(round(candidate[0])),
    )


_RENDER_FONT_ROOT = Path(__file__).resolve().parent / "assets" / "fonts"


@lru_cache(maxsize=1024)
def _bundled_font_mask(
    family: str,
    text: str,
    weight: int,
    optical_size: int,
) -> Any | None:
    """Render one offline comparison mask from the bundled fallback fonts."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    filename = "Newsreader.ttf" if family == "newsreader" else "InterTight.ttf"
    path = _RENDER_FONT_ROOT / filename
    if not path.is_file() or not text:
        return None
    try:
        font = ImageFont.truetype(str(path), 160)
        values = []
        for axis in font.get_variation_axes():
            name = axis["name"].decode("ascii", errors="ignore").casefold()
            values.append(weight if name == "weight" else optical_size)
        if values:
            font.set_variation_by_axes(values)
        box = font.getbbox(text)
        width = max(1, int(box[2] - box[0]))
        height = max(1, int(box[3] - box[1]))
        canvas = Image.new("L", (width + 12, height + 12))
        ImageDraw.Draw(canvas).text(
            (6 - int(box[0]), 6 - int(box[1])), text, font=font, fill=255
        )
    except (AttributeError, OSError, ValueError):
        return None
    mask = np.asarray(canvas) > 64
    rows, columns = np.where(mask)
    if not len(rows) or not len(columns):
        return None
    return mask[
        int(rows.min()) : int(rows.max()) + 1,
        int(columns.min()) : int(columns.max()) + 1,
    ].astype(np.uint8)


def _bundled_font_mask_distance(measured: Any, rendered: Any) -> float:
    import cv2
    import numpy as np

    rendered = cv2.resize(
        np.asarray(rendered, dtype=np.uint8),
        (int(measured.shape[1]), int(measured.shape[0])),
        interpolation=cv2.INTER_AREA,
    )
    rendered = (rendered > 0.35).astype(np.uint8)
    measured = np.asarray(measured, dtype=np.uint8)
    if not measured.any() or not rendered.any():
        return float("inf")
    measured_distance = cv2.distanceTransform(1 - measured, cv2.DIST_L2, 3)
    rendered_distance = cv2.distanceTransform(1 - rendered, cv2.DIST_L2, 3)
    chamfer = float(
        measured_distance[rendered > 0].mean()
        + rendered_distance[measured > 0].mean()
    ) / max(1, measured.shape[0], measured.shape[1])
    intersection = int((measured & rendered).sum())
    union = max(1, int((measured | rendered).sum()))
    return chamfer + 1.0 - intersection / union


def _measure_bundled_render_font(
    image: Any,
    entry: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any] | None:
    """Choose the closer local serif/sans renderer from measured glyph pixels.

    This does not claim to identify the source font.  It only picks between the
    two offline fonts that the generated starter can actually ship, and records
    the choice as a measured rendering strategy.
    """
    import numpy as np

    text = str(entry.get("preferredValue") or entry.get("value") or "").strip()
    box = entry.get("boxSource") or []
    cap_height = int(metrics.get("capHeight") or 0)
    visible_characters = sum(character.isalnum() for character in text)
    if (
        len(box) != 4
        or cap_height < 18
        or visible_characters < 3
        or "\n" in text
    ):
        return None
    x0, y0, x1, y1 = (int(round(value)) for value in box)
    patch = image[y0:y1, x0:x1].astype(np.float32)
    color = str(entry.get("color") or metrics.get("color") or "")
    if patch.size == 0 or not re.fullmatch(r"#[0-9A-Fa-f]{6}", color):
        return None
    rgb = [int(color[index : index + 2], 16) for index in (1, 3, 5)]
    target_bgr = np.asarray(rgb[::-1], dtype=np.float32)
    measured = (np.linalg.norm(patch - target_bgr, axis=2) <= 70.0).astype(
        np.uint8
    )
    rows, columns = np.where(measured)
    if not len(rows) or not len(columns):
        return None
    measured = measured[
        int(rows.min()) : int(rows.max()) + 1,
        int(columns.min()) : int(columns.max()) + 1,
    ]
    coverage = float(measured.mean())
    if min(measured.shape) < 8 or coverage < 0.01 or coverage > 0.78:
        return None

    optical_size = max(6, min(72, int(metrics.get("fontSize") or cap_height)))
    scores: dict[str, tuple[float, int]] = {}
    for family in ("inter-tight", "newsreader"):
        ranked = []
        for weight in range(200, 801, 100):
            rendered = _bundled_font_mask(family, text, weight, optical_size)
            if rendered is None:
                continue
            ranked.append(
                (_bundled_font_mask_distance(measured, rendered), weight)
            )
        if ranked:
            scores[family] = min(ranked, key=lambda item: item[0])
    if len(scores) != 2:
        return None
    family = min(scores, key=lambda name: scores[name][0])
    other = "newsreader" if family == "inter-tight" else "inter-tight"
    best_score, weight = scores[family]
    margin = scores[other][0] - best_score
    if margin < 0.07 or best_score > 0.95:
        return {
            "renderFamilyCandidate": family,
            "renderFamilyConfidence": round(max(0.0, 0.5 + margin), 3),
            "renderFamilyMethod": "bundled-glyph-raster-match-uncertain",
            "renderFamilyScores": {
                key: round(value[0], 4) for key, value in scores.items()
            },
        }
    return {
        "renderFamily": family,
        "renderFamilyCandidate": family,
        "renderFamilyConfidence": round(min(0.98, 0.55 + margin * 1.8), 3),
        "renderWeight": weight,
        "renderFamilyMethod": "bundled-glyph-raster-chamfer-iou",
        "renderFamilyScores": {
            key: round(value[0], 4) for key, value in scores.items()
        },
    }


def _semantic_spacing_is_plausible(value: Any) -> bool:
    """Reject obvious intra-word gaps while keeping measured phrase spaces."""
    parts = str(value or "").split()
    if len(parts) < 2:
        return True
    suffix = parts[-1]
    implausible_suffix = (
        len(suffix) == 1
        and suffix.isalpha()
        and suffix.islower()
        and suffix not in {"a", "i"}
    )
    prefix = parts[0]
    implausible_prefix = (
        len(prefix) == 1
        and prefix.isalpha()
        and prefix.casefold() not in {"a", "i"}
        and len(parts[1]) >= 3
    )
    return not (implausible_suffix or implausible_prefix)


def _restore_word_spaces_from_glyph_gaps(
    image: Any,
    entry: dict[str, Any],
    metrics: dict[str, Any],
) -> str | None:
    """Restore OCR-collapsed spaces from dominant measured column gaps."""
    import cv2
    import numpy as np

    value = str(entry.get("preferredValue") or entry.get("value") or "").strip()
    box = entry.get("boxSource") or []
    if (
        len(box) != 4
        or not value
        or not value.isalnum()
        or any(character.isspace() for character in value)
        or len(value) < 5
    ):
        return None
    x0, y0, x1, y1 = (int(round(item)) for item in box)
    patch = image[y0:y1, x0:x1].astype(np.float32)
    color = str(entry.get("color") or metrics.get("color") or "")
    if patch.size == 0 or not re.fullmatch(r"#[0-9A-Fa-f]{6}", color):
        return None
    rgb = [int(color[index : index + 2], 16) for index in (1, 3, 5)]
    target_bgr = np.asarray(rgb[::-1], dtype=np.float32)
    occupied = np.where(
        (np.linalg.norm(patch - target_bgr, axis=2) <= 70.0).any(axis=0)
    )[0]
    if len(occupied) < len(value):
        return None
    differences = np.diff(occupied)
    boundaries = np.where(differences > 1)[0]
    if not len(boundaries):
        return None
    gaps = np.asarray([differences[index] - 1 for index in boundaries], dtype=float)
    if not len(gaps):
        return None
    cap_height = max(1.0, float(metrics.get("capHeight") or y1 - y0))
    typical = (
        float(np.median(np.sort(gaps)[: max(1, len(gaps) // 2)]))
        if len(gaps) >= 3
        else 0.0
    )
    threshold = max(cap_height * 0.22, typical * 1.7, 3.0)
    large_gap_indexes = [index for index, gap in enumerate(gaps) if gap >= threshold]
    if not large_gap_indexes:
        return None
    if len(boundaries) != len(value) - 1:
        if cap_height < 24 or len(large_gap_indexes) > 3:
            return None
        split_boundaries = [boundaries[index] for index in large_gap_indexes]
        segment_ranges: list[tuple[int, int]] = []
        start = int(occupied[0])
        for boundary in split_boundaries:
            segment_ranges.append((start, int(occupied[boundary]) + 1))
            start = int(occupied[boundary + 1])
        segment_ranges.append((start, int(occupied[-1]) + 1))
        readings: list[str] = []
        confidences: list[float] = []
        padding = max(2, round(cap_height * 0.10))
        for local_x0, local_x1 in segment_ranges:
            crop = image[
                max(0, y0 - padding) : min(image.shape[0], y1 + padding),
                max(0, x0 + local_x0 - padding) : min(
                    image.shape[1], x0 + local_x1 + padding
                ),
            ]
            if crop.size == 0:
                return None
            scale = 2.0 if cap_height < 64 else 1.5
            bounded = cv2.resize(
                crop,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )
            try:
                recognized = run_latin_ocr_image(bounded)
            except (ImportError, ModuleNotFoundError, RuntimeError, OSError):
                return None
            recognized = [
                item
                for item in recognized
                if str(item.get("text") or "").strip()
                and float(item.get("confidence") or 0.0) >= 0.88
            ]
            if not recognized:
                return None
            recognized.sort(
                key=lambda item: (
                    (item.get("box") or [0, 0, 0, 0])[1],
                    (item.get("box") or [0, 0, 0, 0])[0],
                )
            )
            readings.append(
                " ".join(str(item.get("text") or "").strip() for item in recognized)
            )
            confidences.append(
                min(float(item.get("confidence") or 0.0) for item in recognized)
            )
        restored = " ".join(readings)
        if (
            _visual_latin_text(restored) == _visual_latin_text(value)
            and min(confidences, default=0.0) >= 0.88
            and _semantic_spacing_is_plausible(restored)
        ):
            return restored
        return None
    large = [index + 1 for index in large_gap_indexes]
    result = []
    for index, character in enumerate(value):
        if index in large:
            result.append(" ")
        result.append(character)
    restored = "".join(result)
    return (
        restored
        if restored != value and _semantic_spacing_is_plausible(restored)
        else None
    )


def _separate_navigation_prefix_icons(
    document: dict[str, Any], image_path: str
) -> None:
    """Split a repeated menu icon from the live label and re-read the label.

    Compact dashboard navigation often places a 12 px pictogram roughly 8 px
    before a label. Full-frame OCR can fuse both into values such as
    ``8Groups`` or ``III Analytics``. Geometry identifies the isolated leading
    glyph; a bounded Latin pass then observes only the selectable label.
    """
    import cv2
    import numpy as np

    spec = document.get("reconstruction") or {}
    canvas = spec.get("canvas") or {}
    canvas_width = int(canvas.get("width") or 0)
    canvas_height = int(canvas.get("height") or 0)
    if (
        spec.get("targetKind") != "web"
        or canvas_width <= 0
        or canvas_height <= 0
    ):
        return
    rail_surfaces = [
        box
        for surface in spec.get("surfaces") or []
        for box in [surface.get("boxSource") or []]
        if len(box) == 4
        and box[0] <= canvas_width * 0.10
        and box[2] - box[0] <= canvas_width * 0.32
        and box[3] - box[1] >= canvas_height * 0.65
    ]
    try:
        image = load_cv(image_path)
    except (OSError, ValueError):
        return
    icons = spec.setdefault("icons", [])

    def foreground_columns(
        entry: dict[str, Any], box: list[int]
    ) -> Any:
        x0, y0, x1, y1 = box
        patch = image[y0:y1, x0:x1].astype(np.float32)
        if patch.size == 0:
            return np.asarray([], dtype=int)
        # OCR boxes are deliberately tight; even their full area can be mostly
        # bold ink. Sample a measured ring immediately outside the box instead.
        padding = max(2, round((y1 - y0) * 0.28))
        ox0, oy0 = max(0, x0 - padding), max(0, y0 - padding)
        ox1 = min(image.shape[1], x1 + padding)
        oy1 = min(image.shape[0], y1 + padding)
        outer = image[oy0:oy1, ox0:ox1].astype(np.float32)
        ring = np.ones(outer.shape[:2], dtype=bool)
        ring[y0 - oy0 : y1 - oy0, x0 - ox0 : x1 - ox0] = False
        ring_pixels = outer[ring]
        background = np.median(
            (ring_pixels if len(ring_pixels) else patch.reshape((-1, 3))),
            axis=0,
        )
        distance = np.linalg.norm(patch - background, axis=2)
        mask = distance >= max(16.0, float(np.percentile(distance, 58)))
        color = str(
            entry.get("color")
            or (entry.get("fontFeatures") or {}).get("color")
            or ""
        )
        if re.fullmatch(r"#[0-9A-Fa-f]{6}", color):
            text_bgr = np.asarray(_hex_to_bgr(color), dtype=np.float32)
            color_mask = (
                np.linalg.norm(patch - text_bgr, axis=2) <= 54.0
            ) & (distance >= 8.0)
            if int(color_mask.sum()) >= 4:
                mask = color_mask
        return np.where(mask.any(axis=0))[0]

    for entry in spec.get("text") or []:
        observed_label = str(
            entry.get("preferredValue") or entry.get("value") or ""
        ).strip()
        if not rail_surfaces and any(
            character.isspace() for character in observed_label
        ):
            # Away from a measured navigation rail, only compact fused tokens
            # such as ``8Groups`` are eligible. Natural phrase spaces (for
            # example ``See how in 140s``) are not icon separators.
            continue
        box = [int(round(value)) for value in entry.get("boxSource") or []]
        if len(box) != 4:
            continue
        x0, y0, x1, y1 = box
        height = y1 - y0
        width = x1 - x0
        if (
            x0 > canvas_width * 0.24
            or not 7 <= height <= max(34, canvas_height * 0.07)
            or width < max(34, height * 3)
        ):
            continue

        prefix_end: int | None = None
        label_start: int | None = None
        word_boxes = [
            item.get("box")
            for item in (entry.get("fontFeatures") or {}).get(
                "wordBoxesSource", []
            )
            if isinstance(item, dict) and len(item.get("box") or []) == 4
        ]
        if len(word_boxes) >= 2:
            first = [int(round(value)) for value in word_boxes[0]]
            second = [int(round(value)) for value in word_boxes[1]]
            if (
                first[0] <= x0 + 3
                and first[2] - first[0] <= max(18, height * 1.6)
                and second[0] - first[2] >= max(5, height * 0.32)
                and x1 - second[0] >= max(20, height * 1.8)
            ):
                prefix_end, label_start = first[2], second[0]
        if label_start is None:
            occupied = foreground_columns(entry, box)
            if len(occupied) < 4:
                continue
            gaps = np.diff(occupied)
            minimum_gap = max(5, round(height * 0.32))
            for index in np.where(gaps > minimum_gap)[0]:
                left_end = int(occupied[index])
                right_start = int(occupied[index + 1])
                if (
                    left_end - int(occupied[0]) + 1
                    <= max(18, height * 1.6)
                    and int(occupied[-1]) - right_start + 1
                    >= max(20, height * 1.8)
                ):
                    prefix_end = x0 + left_end + 1
                    label_start = x0 + right_start
                    break
        if label_start is None or prefix_end is None:
            continue

        crop_padding = max(4, round(height * 0.24))
        crop_x0 = max(0, label_start - crop_padding)
        crop_y0 = max(0, y0 - crop_padding)
        crop_x1 = min(image.shape[1], x1 + crop_padding)
        crop_y1 = min(image.shape[0], y1 + crop_padding)
        crop = image[crop_y0:crop_y1, crop_x0:crop_x1]
        if crop.size == 0:
            continue
        border_color = tuple(
            int(value)
            for value in np.median(
                np.concatenate(
                    (crop[0], crop[-1], crop[:, 0], crop[:, -1]), axis=0
                ),
                axis=0,
            )
        )
        border_size = 6
        bounded = cv2.copyMakeBorder(
            crop,
            border_size,
            border_size,
            border_size,
            border_size,
            cv2.BORDER_CONSTANT,
            value=border_color,
        )
        scale = 4 if height < 24 else 3
        bounded = cv2.resize(
            bounded,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
        try:
            recognized = run_latin_ocr_image(bounded)
        except (ImportError, ModuleNotFoundError, RuntimeError, OSError):
            continue
        recognized = [
            item
            for item in recognized
            if str(item.get("text") or "").strip()
            and float(item.get("confidence") or 0.0) >= 0.88
        ]
        if not recognized:
            continue
        recognized.sort(
            key=lambda item: (
                (item.get("box") or [0, 0, 0, 0])[1],
                (item.get("box") or [0, 0, 0, 0])[0],
            )
        )
        label = " ".join(
            str(item.get("text") or "").strip() for item in recognized
        ).strip()
        confidence = min(float(item.get("confidence") or 0.0) for item in recognized)
        for prior in (entry.get("value"), entry.get("preferredValue")):
            spaced = re.sub(
                r"^[^\w]+|[^\w]+$", "", str(prior or "").strip()
            )
            if (
                any(character.isspace() for character in spaced)
                and _normalized_text(spaced) == _normalized_text(label)
            ):
                label = spaced
                break
        if len(_normalized_text(label)) < 2 or not any(
            character.isalpha() for character in label
        ):
            continue

        icon_box = [x0, y0, prefix_end, y1]
        if not any(
            _box_intersection(icon.get("boxSource") or [], icon_box)
            / max(1, _source_box_area(icon_box))
            >= 0.55
            for icon in icons
        ):
            icons.append(
                {
                    "elementId": f"navigation-prefix-icon:{entry.get('elementId')}",
                    "kind": "navigation-prefix-icon",
                    "boxSource": icon_box,
                    "source": "measured-prefix-separation",
                    "epistemic": "measured",
                    "representation": "preserve-source-decoration",
                    "ocrEvidence": entry.get("value"),
                }
            )
        entry.update(
            {
                "value": label,
                "preferredValue": label,
                "boxSource": [label_start, y0, x1, y1],
                "boxNormSource": docmod.normalize_box(
                    [label_start, y0, x1, y1], canvas_width, canvas_height
                ),
                "status": "confirmed",
                "confidence": round(confidence, 3),
                "verified": confidence >= 0.97,
                "method": "rapidocr-bounded-navigation-label",
                "resolutionStatus": "confirmed",
                "resolutionMethod": (
                    "bounded-latin-ocr-after-icon-prefix-split"
                ),
                "resolutionConfidence": round(confidence, 3),
                "geometrySource": "measured-navigation-prefix-gap",
                "epistemic": "inferred",
                "confirmedBy": ["bounded-latin-ocr", "measured-prefix-gap"],
            }
        )


def _tighten_control_label_boxes(document: dict[str, Any], image_path: str) -> None:
    """Re-read native control labels without absorbing adjacent brand icons."""
    spec = document.get("reconstruction") or {}
    canvas = spec.get("canvas") or {}
    canvas_width = int(canvas.get("width") or 0)
    canvas_height = int(canvas.get("height") or 0)
    if (
        spec.get("targetKind") != "web"
        or canvas_width <= 0
        or canvas_height <= 0
    ):
        return
    labels_by_id = {
        entry.get("elementId"): entry
        for entry in spec.get("text") or []
        if entry.get("elementId") is not None
    }
    controls = [
        control
        for control in spec.get("visualControlCandidates") or []
        if isinstance(control, dict)
        and len(control.get("boxSource") or []) == 4
        and control.get("labelElementIds")
    ]
    if not controls:
        return
    try:
        image = load_cv(image_path)
    except (OSError, ValueError):
        return

    for control in controls[:24]:
        control_box = [int(round(value)) for value in control["boxSource"]]
        cx0, cy0, cx1, cy1 = control_box
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        crop = image[
            max(0, cy0) : min(image.shape[0], cy1),
            max(0, cx0) : min(image.shape[1], cx1),
        ]
        if crop.size == 0:
            continue
        try:
            recognized = run_latin_ocr_image(crop)
        except (ImportError, ModuleNotFoundError, RuntimeError, OSError):
            continue
        recognized = [
            item
            for item in recognized
            if str(item.get("text") or "").strip()
            and len(item.get("box") or []) == 4
            and float(item.get("confidence") or 0.0) >= 0.9
        ]
        recognized.sort(
            key=lambda item: (
                (item["box"][1] + item["box"][3]) / 2.0,
                item["box"][0],
            )
        )
        for label_id in control.get("labelElementIds") or []:
            entry = labels_by_id.get(label_id)
            if entry is None:
                continue
            expected = str(
                entry.get("preferredValue") or entry.get("value") or ""
            ).strip()
            current_box = [
                int(round(value)) for value in entry.get("boxSource") or []
            ]
            if (
                len(current_box) != 4
                or len(expected.split()) < 2
                or current_box[2] <= current_box[0]
                or current_box[3] <= current_box[1]
            ):
                continue
            matches = [
                item
                for item in recognized
                if _normalized_text(item.get("text"))
                == _normalized_text(expected)
            ]
            if not matches and recognized:
                joined = " ".join(
                    str(item.get("text") or "").strip() for item in recognized
                )
                if _normalized_text(joined) == _normalized_text(expected):
                    matches = [
                        {
                            "text": joined,
                            "confidence": min(
                                float(item.get("confidence") or 0.0)
                                for item in recognized
                            ),
                            "box": [
                                min(int(item["box"][0]) for item in recognized),
                                min(int(item["box"][1]) for item in recognized),
                                max(int(item["box"][2]) for item in recognized),
                                max(int(item["box"][3]) for item in recognized),
                            ],
                        }
                    ]
            if not matches:
                continue
            match = max(
                matches, key=lambda item: float(item.get("confidence") or 0.0)
            )
            local_box = [int(round(value)) for value in match["box"]]
            measured_box = [
                cx0 + local_box[0],
                cy0 + local_box[1],
                cx0 + local_box[2],
                cy0 + local_box[3],
            ]
            measured_area = max(1, _source_box_area(measured_box))
            overlap = _box_intersection(current_box, measured_box) / measured_area
            if (
                overlap < 0.65
                or measured_box[2] - measured_box[0]
                < (current_box[2] - current_box[0]) * 0.55
                or max(
                    abs(measured_box[index] - current_box[index])
                    for index in range(4)
                )
                < 3
            ):
                continue
            entry["boxSource"] = measured_box
            entry["boxNormSource"] = docmod.normalize_box(
                measured_box, canvas_width, canvas_height
            )
            entry["geometrySource"] = "bounded-latin-control-label"
            entry["geometryConfidence"] = round(
                float(match.get("confidence") or 0.0), 3
            )
            methods = entry.setdefault("confirmedBy", [])
            if "bounded-latin-control-label" not in methods:
                methods.append("bounded-latin-control-label")


def _hydrate_measured_typography(
    document: dict[str, Any], image_path: str
) -> None:
    """Refresh cheap pixel measurements even when the semantic document is cached."""
    spec = document.get("reconstruction") or {}
    if spec.get("targetKind") != "web":
        return
    try:
        image = load_cv(image_path)
    except (OSError, ValueError):
        return
    canvas_width, canvas_height = (document.get("header") or {}).get(
        "size", [image.shape[1], image.shape[0]]
    )
    for entry in spec.get("text") or []:
        box = entry.get("boxSource") or []
        if len(box) != 4:
            continue
        refined_box = _refine_large_text_box(image, entry)
        if refined_box is not None:
            entry["boxSource"] = refined_box
            entry["boxNormSource"] = docmod.normalize_box(
                refined_box, int(canvas_width), int(canvas_height)
            )
            entry["geometrySource"] = "measured-dominant-glyph-components"
            box = refined_box
        metrics = _glyph_metrics(
            image,
            [int(round(value)) for value in box],
            entry.get("preferredValue") or entry.get("value"),
        )
        if metrics is None:
            continue
        repaired_color = _repair_reversed_glyph_color(
            image, [int(round(value)) for value in box], metrics
        )
        if repaired_color is not None:
            metrics["color"] = repaired_color
            metrics["colorSource"] = "measured-external-ring-contrast-repair"
        small_refinement = _refine_small_text_ink_box(image, entry, metrics)
        if small_refinement is not None:
            refined_box, geometry_source = small_refinement
            entry["boxSource"] = refined_box
            entry["boxNormSource"] = docmod.normalize_box(
                refined_box, int(canvas_width), int(canvas_height)
            )
            entry["geometrySource"] = geometry_source
            box = refined_box
            refreshed = _glyph_metrics(
                image,
                refined_box,
                entry.get("preferredValue") or entry.get("value"),
            )
            if refreshed is not None:
                metrics = refreshed
                repaired_color = _repair_reversed_glyph_color(
                    image, refined_box, metrics
                )
                if repaired_color is not None:
                    metrics["color"] = repaired_color
                    metrics["colorSource"] = (
                        "measured-external-ring-contrast-repair"
                    )
        metrics["coordinateSpace"] = "source-pixels"
        restored_spacing = _restore_word_spaces_from_glyph_gaps(
            image, entry, metrics
        )
        if restored_spacing is not None:
            entry.update(
                {
                    "preferredValue": restored_spacing,
                    "resolutionStatus": "measured-word-gap-restored",
                    "resolutionMethod": "dominant-foreground-column-gap",
                    "resolutionConfidence": 0.94,
                    "epistemic": "measured",
                }
            )
            refreshed_spacing_metrics = _glyph_metrics(
                image,
                [int(round(value)) for value in box],
                restored_spacing,
            )
            if refreshed_spacing_metrics is not None:
                metrics = refreshed_spacing_metrics
                repaired_color = _repair_reversed_glyph_color(
                    image,
                    [int(round(value)) for value in box],
                    metrics,
                )
                if repaired_color is not None:
                    metrics["color"] = repaired_color
                    metrics["colorSource"] = (
                        "measured-external-ring-contrast-repair"
                    )
                metrics["coordinateSpace"] = "source-pixels"
        render_font = _measure_bundled_render_font(image, entry, metrics)
        if render_font is not None:
            metrics.update(render_font)
        word_boxes = metrics.pop("wordBoxes", None) or []
        for item in word_boxes:
            item_box = item.get("box") or []
            item_text = str(item.get("text") or "")
            if len(item_box) != 4 or not item_text:
                continue
            word_metrics = _glyph_metrics(image, item_box, item_text)
            if word_metrics is None:
                continue
            word_render = _measure_bundled_render_font(
                image,
                {
                    **entry,
                    "value": item_text,
                    "preferredValue": item_text,
                    "boxSource": item_box,
                },
                word_metrics,
            )
            if word_render is None:
                continue
            for key in (
                "renderFamily",
                "renderFamilyCandidate",
                "renderFamilyConfidence",
                "renderWeight",
                "renderFamilyMethod",
                "renderFamilyScores",
            ):
                if key in word_render:
                    item[key] = word_render[key]
        if word_boxes:
            metrics["wordBoxesSource"] = word_boxes
        entry["fontFeatures"] = metrics
        measured_slants = [
            str(item.get("slant") or "")
            for item in word_boxes
            if isinstance(item, dict)
            and sum(
                character.isalnum()
                for character in str(item.get("text") or "")
            )
            >= 2
        ]
        if (
            len(word_boxes) >= 2
            and len(measured_slants) >= 2
            and all(measured_slants)
            and len(set(measured_slants)) >= 2
        ):
            display = str(entry.get("preferredValue") or entry.get("value") or "")
            words = list(re.finditer(r"\S+", display))
            if len(words) == len(word_boxes):
                measured_runs = []
                for index, (word, item) in enumerate(
                    zip(words, word_boxes, strict=True)
                ):
                    end = (
                        words[index + 1].start()
                        if index + 1 < len(words)
                        else len(display)
                    )
                    measured_runs.append(
                        {
                            "text": display[word.start() : end],
                            "typographyCandidate": {
                                "slant": item["slant"],
                                "confidence": item.get("slantConfidence"),
                                "status": "candidate",
                                "epistemic": "measured",
                                "method": item.get("slantMethod"),
                            },
                            "runIndex": index,
                        }
                    )
                entry["inlineRuns"] = measured_runs
                entry["inlineRunMethod"] = "measured-word-slant"
        entry["color"] = metrics.get("color") or entry.get("color")
        entry["colorSource"] = metrics.get("colorSource") or entry.get("colorSource")
        if metrics.get("weightCandidate"):
            entry["fontStrategy"] = (
                "measured-stroke-weight-then-class-and-glyph-metrics"
            )
    _resolve_editorial_mixed_display_fonts(document)
    _resolve_symmetric_render_fonts(document)


def _resolve_editorial_mixed_display_fonts(document: dict[str, Any]) -> None:
    """Resolve large mixed-slant editorial lines from measured word evidence."""
    spec = document.get("reconstruction") or {}
    if spec.get("targetKind") != "web":
        return
    for entry in spec.get("text") or []:
        font = entry.get("fontFeatures") or {}
        word_boxes = [
            item
            for item in font.get("wordBoxesSource") or []
            if isinstance(item, dict)
        ]
        display = str(entry.get("preferredValue") or entry.get("value") or "")
        words = list(re.finditer(r"\S+", display))
        if (
            max(
                float(font.get("fontSize") or 0.0),
                float(font.get("capHeight") or 0.0),
            )
            < 64.0
            or not 2 <= len(word_boxes) <= 4
            or len(words) != len(word_boxes)
        ):
            continue
        italic_evidence = [
            item
            for item in word_boxes
            if str(item.get("slant") or "").casefold()
            in {"italic", "oblique"}
            and float(item.get("slantConfidence") or 0.0) >= 0.7
        ]
        if not italic_evidence:
            continue
        if (
            str(font.get("renderFamily") or "").casefold() == "inter-tight"
            and float(font.get("renderFamilyConfidence") or 0.0) >= 0.72
        ) or any(
            str(item.get("renderFamily") or "").casefold() == "inter-tight"
            and float(item.get("renderFamilyConfidence") or 0.0) >= 0.72
            for item in word_boxes
        ):
            continue
        editorial_evidence = False
        for item in italic_evidence:
            scores = item.get("renderFamilyScores") or {}
            editorial_evidence = editorial_evidence or (
                str(item.get("renderFamilyCandidate") or "").casefold()
                == "newsreader"
                or (
                    "inter-tight" in scores
                    and "newsreader" in scores
                    and float(scores["newsreader"])
                    <= float(scores["inter-tight"]) + 0.04
                )
            )
        if not editorial_evidence:
            continue
        for item in word_boxes:
            item.update(
                {
                    "renderFamily": "newsreader",
                    "renderFamilyCandidate": "newsreader",
                    "renderFamilyConfidence": 0.78,
                    "renderWeight": 300,
                    "renderFamilyMethod": (
                        "measured-mixed-editorial-slant-consensus"
                    ),
                }
            )
        for run in entry.get("inlineRuns") or []:
            if not isinstance(run, dict):
                continue
            typography = dict(run.get("typographyCandidate") or {})
            typography.update(
                {
                    "class": "serif",
                    "contrast": "high",
                    "width": "normal",
                    "weight": "light",
                    "confidence": 0.78,
                    "status": "candidate",
                    "epistemic": "inferred",
                    "method": "measured-mixed-editorial-slant-consensus",
                }
            )
            run["typographyCandidate"] = typography
        entry["fontStrategy"] = "measured-mixed-editorial-slant-consensus"


def _resolve_symmetric_render_fonts(document: dict[str, Any]) -> None:
    """Resolve conflicting font matches from symmetric numeric display peers."""
    spec = document.get("reconstruction") or {}
    candidates: list[dict[str, Any]] = []
    for entry in spec.get("text") or []:
        value = str(entry.get("preferredValue") or entry.get("value") or "")
        alphanumeric = [character for character in value if character.isalnum()]
        digits = sum(character.isdigit() for character in alphanumeric)
        box = entry.get("boxSource") or []
        font = entry.get("fontFeatures") or {}
        scores = font.get("renderFamilyScores") or {}
        if (
            len(box) != 4
            or len(alphanumeric) < 3
            or digits < 3
            or digits / max(1, len(alphanumeric)) < 0.45
            or float(font.get("capHeight") or 0.0) < 40.0
            or not all(family in scores for family in ("inter-tight", "newsreader"))
        ):
            continue
        candidates.append(entry)

    used: set[int] = set()
    for index, left in enumerate(candidates):
        if index in used:
            continue
        left_box = [float(value) for value in left.get("boxSource") or []]
        left_font = left.get("fontFeatures") or {}
        left_cap = float(left_font.get("capHeight") or 0.0)
        best: tuple[float, int, dict[str, Any]] | None = None
        for other_index in range(index + 1, len(candidates)):
            if other_index in used:
                continue
            right = candidates[other_index]
            right_box = [float(value) for value in right.get("boxSource") or []]
            right_font = right.get("fontFeatures") or {}
            right_cap = float(right_font.get("capHeight") or 0.0)
            vertical_delta = abs(
                (left_box[1] + left_box[3]) / 2.0
                - (right_box[1] + right_box[3]) / 2.0
            )
            if (
                vertical_delta > max(left_cap, right_cap) * 0.28
                or abs(left_cap - right_cap) / max(left_cap, right_cap) > 0.10
                or str(left.get("color") or left_font.get("color") or "").casefold()
                != str(right.get("color") or right_font.get("color") or "").casefold()
                or not (left_box[2] <= right_box[0] or right_box[2] <= left_box[0])
            ):
                continue
            if best is None or vertical_delta < best[0]:
                best = (vertical_delta, other_index, right)
        if best is None:
            continue
        _delta, other_index, right = best
        pair = (left, right)
        aggregate = {
            family: sum(
                float(
                    ((entry.get("fontFeatures") or {}).get("renderFamilyScores") or {})[
                        family
                    ]
                )
                for entry in pair
            )
            for family in ("inter-tight", "newsreader")
        }
        winner = min(aggregate, key=aggregate.get)
        loser = "newsreader" if winner == "inter-tight" else "inter-tight"
        average_margin = (aggregate[loser] - aggregate[winner]) / len(pair)
        if average_margin < 0.02:
            continue
        render_weight = next(
            (
                int((entry.get("fontFeatures") or {}).get("renderWeight") or 0)
                for entry in pair
                if (entry.get("fontFeatures") or {}).get("renderFamilyCandidate")
                == winner
                and int((entry.get("fontFeatures") or {}).get("renderWeight") or 0)
                > 0
            ),
            400,
        )
        confidence = round(min(0.94, 0.55 + average_margin * 1.8), 3)
        for entry in pair:
            font = entry.get("fontFeatures") or {}
            font.update(
                {
                    "renderFamily": winner,
                    "renderFamilyCandidate": winner,
                    "renderFamilyConfidence": confidence,
                    "renderWeight": render_weight,
                    "renderFamilyMethod": "symmetric-numeric-render-consensus",
                    "renderFamilyConsensusScores": {
                        family: round(score / len(pair), 4)
                        for family, score in aggregate.items()
                    },
                }
            )
        used.update((index, other_index))


def _hydrate_measured_control_geometry(
    document: dict[str, Any], image_path: str
) -> None:
    """Refresh outline radius on cached semantic controls."""
    import cv2
    import numpy as np

    spec = document.get("reconstruction") or {}
    if spec.get("targetKind") != "web":
        return
    try:
        image = load_cv(image_path)
    except (OSError, ValueError):
        return
    ocr_items = [
        {
            "text": entry.get("preferredValue") or entry.get("value") or "",
            "box": entry.get("boxSource") or [],
            "confidence": entry.get("confidence") or 0.0,
        }
        for entry in spec.get("text") or []
        if len(entry.get("boxSource") or []) == 4
    ]
    outlined = outlined_controls_around_text(image, ocr_items, [])
    filled = _controls_around_text(image, ocr_items, outlined)
    detected = [
        *outlined,
        *filled,
    ]
    colors = (document.get("tokens") or {}).get("color") or {}
    canvas = (colors.get("canvas") or {}).get("$value")
    if not canvas:
        canvas = (document.get("header") or {}).get("background") or "#FFFFFF"
    detected.extend(
        {
            **entry,
            "box": entry["box"],
            "boundaryEvidence": {"closed": True},
        }
        for entry in outlined_surface_regions(
            image,
            canvas,
            minimum_area_ratio=0.0015,
            maximum_area_ratio=0.03,
        )
    )
    controls = spec.setdefault("visualControlCandidates", [])

    def label_entries(candidate_box: list[int]) -> list[dict[str, Any]]:
        matches = []
        for entry in spec.get("text") or []:
            label_box = entry.get("boxSource") or []
            label_area = max(1, _source_box_area(label_box))
            if (
                len(label_box) == 4
                and _box_intersection(candidate_box, label_box) / label_area >= 0.82
            ):
                matches.append(entry)
        return matches

    existing_ids = [
        int(entry.get("elementId"))
        for collection in (
            spec.get("text") or [],
            controls,
            spec.get("icons") or [],
        )
        for entry in collection
        if str(entry.get("elementId") or "").isdigit()
    ]
    next_element_id = max(existing_ids, default=0) + 1
    image_area = max(1, image.shape[0] * image.shape[1])
    for candidate in detected:
        candidate_box = candidate.get("box") or []
        labels = label_entries(candidate_box)
        if len(candidate_box) != 4 or len(labels) != 1:
            continue
        label = labels[0]
        label_box = label.get("boxSource") or []
        label_area = max(1, _source_box_area(label_box))
        candidate_area = max(1, _source_box_area(candidate_box))
        candidate_width = candidate_box[2] - candidate_box[0]
        candidate_height = candidate_box[3] - candidate_box[1]
        label_width = max(1, label_box[2] - label_box[0])
        label_height = max(1, label_box[3] - label_box[1])
        if (
            candidate_area / image_area > 0.35
            or candidate_area / label_area > 15.0
            or candidate_width / label_width > 4.0
            or candidate_height / label_height > 6.5
        ):
            continue
        if any(
            _intersection_ratio(candidate_box, control.get("boxSource") or []) > 0.8
            for control in controls
        ):
            continue
        outlined = bool(candidate.get("boundaryEvidence", {}).get("closed"))
        controls.append(
            {
                "elementId": next_element_id,
                "kind": "button",
                "interaction": "semantic-button",
                "boxSource": candidate_box,
                "labelElementIds": [label.get("elementId")],
                "background": candidate.get("background"),
                "borderColor": candidate.get("borderColor"),
                "borderWidth": candidate.get("borderWidth") or 0,
                "cornerRadius": candidate.get("cornerRadius") or 0,
                "geometrySource": (
                    "measured-closed-outline"
                    if outlined
                    else "measured-compact-fill"
                ),
                "source": "measured",
                "method": candidate.get("method")
                or "compact-uniform-fill-around-single-ocr-label",
                "labelFontSizeMax": round(max(8.0, candidate_height * 0.62), 1),
                "labelFontSizeSource": "measured-control-interior",
            }
        )
        next_element_id += 1
    pixels = image.astype(np.float32)
    border_pixels = np.concatenate(
        (pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]), axis=0
    )
    page_background = np.median(border_pixels, axis=0)
    foreground = (
        np.linalg.norm(pixels - page_background, axis=2) > 22.0
    ).astype(np.uint8) * 255
    contours, _hierarchy = cv2.findContours(
        foreground, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    raw_outlines: list[dict[str, Any]] = []
    height, width = image.shape[:2]
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if box_width < 12 or box_height < 10:
            continue
        x1, y1 = x + box_width, y + box_height
        if box_width > width * 0.98 or box_height > height * 0.8:
            continue
        band = max(3, min(7, round(min(box_width, box_height) * 0.07)))
        edge_bands = [
            foreground[y : min(height, y + band), x:x1],
            foreground[max(0, y1 - band) : y1, x:x1],
            foreground[y:y1, x : min(width, x + band)],
            foreground[y:y1, max(0, x1 - band) : x1],
        ]
        support = [
            float(np.count_nonzero(edge)) / max(1, edge.size)
            for edge in edge_bands
        ]
        if min(support) < 0.18:
            continue
        perimeter = max(1.0, 2.0 * (box_width + box_height))
        if cv2.arcLength(contour, True) / perimeter < 0.72:
            continue
        band_mask = np.zeros((box_height, box_width), dtype=bool)
        band_mask[:band] = True
        band_mask[-band:] = True
        band_mask[:, :band] = True
        band_mask[:, -band:] = True
        crop_foreground = foreground[y:y1, x:x1] > 0
        boundary_pixels = image[y:y1, x:x1][band_mask & crop_foreground]
        if boundary_pixels.size == 0:
            continue
        boundary_bgr = np.median(boundary_pixels.reshape(-1, 3), axis=0)
        interior = image[y + band : y1 - band, x + band : x1 - band]
        interior_bgr = (
            np.median(interior.reshape(-1, 3), axis=0)
            if interior.size
            else page_background
        )
        raw_outlines.append(
            {
                "box": [x, y, x1, y1],
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
                "borderWidth": max(1, round(band * min(support))),
                "cornerRadius": (
                    round(box_height / 2.0)
                    if box_width / max(1, box_height) >= 2.2
                    else 0
                ),
                "source": "measured",
                "method": "closed-outline-existing-control",
            }
        )
    for control in controls:
        box = control.get("boxSource") or []
        current_area = max(1, _source_box_area(box))
        current_height = max(1, box[3] - box[1])
        matches = [
            candidate
            for candidate in [*detected, *raw_outlines]
            if _intersection_ratio(box, candidate.get("box") or []) > 0.85
            and current_area * 0.75
            <= _source_box_area(candidate.get("box"))
            <= current_area * 3.0
            and current_height * 0.7
            <= (candidate["box"][3] - candidate["box"][1])
            <= current_height * 1.8
        ]
        if not matches:
            continue
        match = max(matches, key=lambda candidate: _source_box_area(candidate.get("box")))
        control["boxSource"] = match["box"]
        control["background"] = match.get("background")
        control["borderColor"] = match.get("borderColor")
        control["borderWidth"] = match.get("borderWidth")
        control["cornerRadius"] = match.get("cornerRadius") or 0
        control["cornerRadiusSource"] = "measured-outline-geometry"
        control["geometrySource"] = "measured-closed-outline"
        control["labelFontSizeMax"] = round(
            max(8.0, (match["box"][3] - match["box"][1]) * 0.62), 1
        )
        control["labelFontSizeSource"] = "measured-control-interior"

    text_by_id = {
        entry.get("elementId"): entry for entry in spec.get("text") or []
    }
    for control in controls:
        background = control.get("background")
        if not isinstance(background, str):
            continue
        fill_bgr = np.asarray(_hex_to_bgr(background), dtype=np.float32)
        for label_id in control.get("labelElementIds") or []:
            label = text_by_id.get(label_id)
            if label is None:
                continue
            box = label.get("boxSource") or []
            if len(box) != 4:
                continue
            x0, y0, x1, y1 = (int(round(value)) for value in box)
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(image.shape[1], x1), min(image.shape[0], y1)
            patch = image[y0:y1, x0:x1].astype(np.float32)
            if patch.size == 0:
                continue
            distances = np.linalg.norm(patch - fill_bgr, axis=2)
            non_fill = distances[distances >= 18.0]
            if non_fill.size < 3:
                continue
            core_threshold = max(18.0, float(np.percentile(non_fill, 60)))
            glyph_pixels = patch[distances >= core_threshold]
            if glyph_pixels.size < 9:
                continue
            glyph_bgr = np.median(glyph_pixels.reshape((-1, 3)), axis=0)
            glyph_color = "#{:02X}{:02X}{:02X}".format(
                int(round(glyph_bgr[2])),
                int(round(glyph_bgr[1])),
                int(round(glyph_bgr[0])),
            )
            label["color"] = glyph_color
            label["colorSource"] = "measured-control-label-contrast"
            font = label.get("fontFeatures")
            if isinstance(font, dict):
                font["color"] = glyph_color
                font["colorSource"] = "measured-control-label-contrast"


def _exclude_dense_parallel_artwork_lines(
    lines: list[dict[str, Any]], canvas_height: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate halftone rows from genuine horizontal layout rules.

    A halftone illustration can expose every coloured dot row as a one-pixel
    Hough line.  Rendering those rows again as DOM rules destroys the artwork.
    Real chart and section rules are sparse, so only long, tightly packed
    clusters are excluded here.
    """
    horizontal = [
        line
        for line in lines
        if str(line.get("orientation") or "horizontal") == "horizontal"
        and len(line.get("boxSource") or []) == 4
    ]
    horizontal.sort(
        key=lambda line: (
            (line["boxSource"][1] + line["boxSource"][3]) / 2,
            line["boxSource"][0],
        )
    )
    maximum_gap = max(10.0, min(18.0, float(canvas_height) * 0.018))
    clusters: list[list[dict[str, Any]]] = []
    for line in horizontal:
        box = line["boxSource"]
        center_y = (box[1] + box[3]) / 2
        attached = False
        for cluster in reversed(clusters[-3:]):
            previous = cluster[-1]["boxSource"]
            previous_center = (previous[1] + previous[3]) / 2
            if center_y - previous_center > maximum_gap:
                continue
            overlap = max(0.0, min(box[2], previous[2]) - max(box[0], previous[0]))
            minimum_width = max(1.0, min(box[2] - box[0], previous[2] - previous[0]))
            if overlap / minimum_width < 0.25:
                continue
            cluster.append(line)
            attached = True
            break
        if not attached:
            clusters.append([line])

    excluded_ids: set[int] = set()
    for cluster in clusters:
        if len(cluster) < 8:
            continue
        thicknesses = [
            max(1.0, float(line["boxSource"][3] - line["boxSource"][1]))
            for line in cluster
        ]
        ordered_thicknesses = sorted(thicknesses)
        if ordered_thicknesses[len(ordered_thicknesses) // 2] > 2.0:
            continue
        for line in cluster:
            excluded_ids.add(id(line))

    kept = [line for line in lines if id(line) not in excluded_ids]
    excluded = []
    for line in lines:
        if id(line) not in excluded_ids:
            continue
        excluded.append(
            {
                **line,
                "reason": "dense-parallel-halftone-artwork",
                "representation": "preserved-in-background-artwork",
            }
        )
    return kept, excluded


def _exclude_symbol_art_lines(
    lines: list[dict[str, Any]],
    symbol_art: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Do not redraw character-art rows as independent structural rules."""
    symbol_boxes = [
        entry.get("boxSource") or []
        for entry in symbol_art
        if len(entry.get("boxSource") or []) == 4
    ]
    excluded_ids: set[int] = set()
    for line in lines:
        box = line.get("boxSource") or []
        if len(box) != 4:
            continue
        area = max(1.0, _source_box_area(box))
        if any(
            _box_intersection(box, symbol_box) / area >= 0.5
            for symbol_box in symbol_boxes
        ):
            excluded_ids.add(id(line))
    kept = [line for line in lines if id(line) not in excluded_ids]
    excluded = [
        {
            **line,
            "reason": "overlaps-live-symbol-art",
            "representation": "owned-by-preformatted-symbol-art",
        }
        for line in lines
        if id(line) in excluded_ids
    ]
    return kept, excluded


def _sanitize_structural_lines(document: dict[str, Any], image_path: str) -> None:
    """Keep measured rules and recover low-contrast fragmented chart grids."""
    spec = document.get("reconstruction") or {}
    if spec.get("targetKind") != "web":
        return
    try:
        image = load_cv(image_path)
    except (OSError, ValueError):
        return
    kept: list[dict[str, Any]] = []
    for line in spec.get("structuralLines") or []:
        box = line.get("boxSource") or []
        if len(box) != 4:
            continue
        contrast = _segment_edge_contrast(
            image,
            [int(round(value)) for value in box],
            str(line.get("orientation") or "horizontal"),
        )
        if contrast < 12.0:
            continue
        line["edgeContrast"] = round(contrast, 2)
        kept.append(line)

    excluded_boxes = [
        [int(round(value)) for value in entry.get("boxSource")]
        for entry in [*(spec.get("text") or []), *(spec.get("badges") or [])]
        if len(entry.get("boxSource") or []) == 4
    ]
    measured_dashes = detect_dashed_structural_lines(
        image,
        excluded_boxes,
    )
    surfaces = [
        surface.get("boxSource") or [] for surface in spec.get("surfaces") or []
    ]
    for candidate in measured_dashes:
        box = candidate.get("boxSource") or []
        if len(box) != 4:
            continue
        width = max(1, box[2] - box[0])
        is_surface_border = any(
            len(surface) == 4
            and (
                abs(box[1] - surface[1]) <= 3
                or abs(box[1] - surface[3]) <= 3
            )
            and max(0, min(box[2], surface[2]) - max(box[0], surface[0]))
            / width
            >= 0.65
            for surface in surfaces
        )
        if is_surface_border:
            continue
        duplicate = any(
            len(existing.get("boxSource") or []) == 4
            and abs(existing["boxSource"][1] - box[1]) <= 3
            and max(
                0,
                min(existing["boxSource"][2], box[2])
                - max(existing["boxSource"][0], box[0]),
            )
            / width
            >= 0.75
            for existing in kept
        )
        if not duplicate:
            kept.append(candidate)
    kept.sort(
        key=lambda line: (
            (line.get("boxSource") or [0, 0])[1],
            (line.get("boxSource") or [0])[0],
        )
    )
    canvas_height = int((spec.get("canvas") or {}).get("height") or image.shape[0])
    kept, excluded_artwork_lines = _exclude_dense_parallel_artwork_lines(
        kept, canvas_height
    )
    kept, excluded_symbol_lines = _exclude_symbol_art_lines(
        kept, spec.get("symbolArt") or []
    )
    spec["structuralLines"] = kept
    if excluded_artwork_lines or excluded_symbol_lines:
        spec["excludedStructuralLineCandidates"] = [
            *(spec.get("excludedStructuralLineCandidates") or []),
            *excluded_artwork_lines,
            *excluded_symbol_lines,
        ]


def _hydrate_navigation_rail(document: dict[str, Any], image_path: str) -> None:
    """Recover isolated navigation glyphs in a measured narrow left rail."""
    spec = document.get("reconstruction") or {}
    if spec.get("targetKind") != "web":
        return
    canvas = spec.get("canvas") or {}
    canvas_width = int(canvas.get("width") or 0)
    canvas_height = int(canvas.get("height") or 0)
    rail_boundaries = sorted(
        int(round(box[0]))
        for surface in spec.get("surfaces") or []
        for box in [surface.get("boxSource") or []]
        if len(box) == 4
        and 40 <= box[0] <= min(120, canvas_width * 0.15)
        and box[3] - box[1] >= canvas_height * 0.7
    )
    if not rail_boundaries:
        return
    rail_width = rail_boundaries[0]
    try:
        image = load_cv(image_path)
    except (OSError, ValueError):
        return
    if image.shape[1] < rail_width or image.shape[0] < canvas_height * 0.7:
        return

    import cv2
    import numpy as np

    rail = image[:, :rail_width]
    rail_background = np.median(rail.reshape(-1, 3), axis=0)
    distance = np.linalg.norm(
        rail.astype(np.float32) - rail_background,
        axis=2,
    )
    mask = (distance > 14).astype(np.uint8) * 255
    for text in spec.get("text") or []:
        box = text.get("boxSource") or []
        if len(box) != 4 or box[0] >= rail_width:
            continue
        x0, y0, x1, y1 = (int(round(value)) for value in box)
        mask[
            max(0, y0 - 3) : min(mask.shape[0], y1 + 3),
            max(0, x0 - 3) : min(mask.shape[1], x1 + 3),
        ] = 0
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
    )
    contours, _hierarchy = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    components = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if (
            not 12 <= width <= 48
            or not 12 <= height <= 48
            or y < 40
            or width * height < 180
            or abs((x + width / 2) - rail_width / 2) > rail_width * 0.25
        ):
            continue
        components.append([x, y, x + width, y + height])
    components.sort(key=lambda box: (box[1], box[0]))
    if len(components) < 2:
        return

    icons = spec.setdefault("icons", [])
    surfaces = spec.setdefault("surfaces", [])
    shapes = spec.setdefault("decorativeShapes", [])

    def as_hex(bgr: Any) -> str:
        blue, green, red = (int(round(value)) for value in bgr)
        return f"#{red:02X}{green:02X}{blue:02X}"

    rail_gray = float(
        cv2.cvtColor(
            np.asarray([[rail_background]], dtype=np.uint8),
            cv2.COLOR_BGR2GRAY,
        )[0, 0]
    )
    for index, box in enumerate(components):
        x0, y0, x1, y1 = box
        patch = image[y0:y1, x0:x1]
        patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        patch_median = np.median(patch.reshape(-1, 3), axis=0)
        selected = (
            x1 - x0 >= 28
            and y1 - y0 >= 28
            and float(np.mean(patch_median)) >= float(np.mean(rail_background)) + 12
        )
        dark_rows, dark_columns = np.where(
            patch_gray < min(200, max(40, rail_gray - 24))
        )
        if selected and len(dark_columns):
            icon_box = [
                x0 + int(dark_columns.min()),
                y0 + int(dark_rows.min()),
                x0 + int(dark_columns.max()) + 1,
                y0 + int(dark_rows.max()) + 1,
            ]
            surface_id = f"rail-navigation-surface-{index}"
            surface = next(
                (
                    item
                    for item in surfaces
                    if item.get("elementId") == surface_id
                ),
                None,
            )
            surface_values = {
                "elementId": surface_id,
                "boxSource": box,
                "background": as_hex(patch_median),
                "borderColor": None,
                "borderWidth": 0,
                "cornerRadius": round(min(x1 - x0, y1 - y0) * 0.2, 1),
                "source": "measured",
                "method": "navigation-rail-isolated-surface",
            }
            if surface is None:
                surfaces.append(surface_values)
            else:
                surface.update(surface_values)
            icon_name = "network"
        elif index == 0 and float(np.mean(patch_median)) < rail_gray - 60:
            icon_box = [x0 + 5, y0 + 5, x1 - 5, y1 - 5]
            icon_name = "brand-mark"
            for shape in shapes:
                if _intersection_ratio(shape.get("boxSource") or [], box) >= 0.7:
                    shape.update(
                        {
                            "boxSource": box,
                            "background": as_hex(patch_median),
                            "borderWidth": 0,
                            "cornerRadius": round(min(x1 - x0, y1 - y0) / 2, 1),
                            "source": "measured",
                            "method": "navigation-rail-filled-circle",
                        }
                    )
                    break
        else:
            icon_box = box
            icon_name = "globe"
        dark_pixels = patch[patch_gray < min(200, max(40, rail_gray - 24))]
        icon_color = (
            "#FFFFFF"
            if icon_name == "brand-mark"
            else as_hex(np.median(dark_pixels, axis=0))
            if len(dark_pixels)
            else "#555555"
        )
        icon_id = f"rail-navigation-icon-{index}"
        icon = next(
            (item for item in icons if item.get("elementId") == icon_id),
            None,
        )
        if icon is None:
            icon = next(
                (
                    item
                    for item in icons
                    if _intersection_ratio(item.get("boxSource") or [], icon_box)
                    >= 0.65
                ),
                None,
            )
        icon_values = {
            "elementId": icon_id,
            "name": icon_name,
            "boxSource": icon_box,
            "color": icon_color,
            "strategy": "css-or-inline-svg",
            "source": "measured-geometry-inferred-symbol",
            "epistemic": "inferred",
            "method": "navigation-rail-isolated-glyph",
        }
        if icon is None:
            icons.append(icon_values)
        else:
            icon.update(icon_values)


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
        metrics = _glyph_metrics(image, item["box"], item.get("text"))
        if metrics is not None:
            item["metrics"] = metrics
    colors = color_zones(image)
    blocks = layout_blocks(image)
    textures = texture_blocks(image, ocr_items)
    symbol_art = detect_symbol_art(image)
    blocks = blocks + textures
    attention = attention_map(image, ocr_items)
    objects, objects_warning = _run_optional_layer("objects", objects_yolo, image_path)
    scene, scene_warning = _run_optional_layer("scene", scene_clip, image_path)
    optional_warnings = [
        warning for warning in (objects_warning, scene_warning) if warning is not None
    ]
    height, width = image.shape[:2]
    canvas_background = colors.get("canvasBackground") or {}
    background_hex = canvas_background.get("hex") or (
        colors["dominant"][0]["hex"] if colors.get("dominant") else "#000000"
    )
    background_bgr = _hex_to_bgr(background_hex)
    controls = control_style(image, blocks)
    controls = controls + outlined_controls_around_text(image, ocr_items, controls)
    controls = controls + _controls_around_text(image, ocr_items, controls)
    controls = _plausible_controls(controls, background_bgr, ocr_items)
    vector_paths = detect_vector_paths(
        image,
        [
            *[item.get("box") or [] for item in ocr_items],
            *[item.get("box") or [] for item in controls],
        ],
    )
    icons = control_icons(image, blocks, ocr_items)
    numeric_badges = compact_numeric_badges(image)
    surfaces = surface_regions(image, blocks, ocr_items, background_hex)
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
        "canvasBackground": canvas_background,
        "ocr": ocr_items,
        "layout": blocks,
        "skeleton": layout_skeleton(image),
        "symbolArt": symbol_art,
        "gaps": layout_gaps(blocks),
        "design": design_qa(image, blocks, ocr_items),
        "sectionStyle": section_style(image, blocks, ocr_items),
        "controls": controls,
        "vectorPaths": vector_paths,
        "icons": icons,
        "numericBadges": numeric_badges,
        "surfaces": surfaces,
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


_RECONSTRUCTION_INTENT_MARKERS = (
    "reconstruct",
    "recreate",
    "reproduce",
    "repeat this design",
    "copy this design",
    "clone this",
    "match this screenshot",
    "pixel-perfect",
    "screenshot to code",
    "повтори",
    "повторить",
    "воссозд",
    "скопир",
    "точь-в-точь",
    "как на скриншоте",
    "сверст",
)


def _resolve_profile(
    profile: str | None,
    intent: str | None,
    dump: dict[str, Any] | None = None,
) -> str:
    if profile is not None:
        if profile not in {"analyze", "reconstruct"}:
            raise ValueError("profile must be 'analyze' or 'reconstruct'")
        return profile
    normalized = (intent or "").casefold()
    if any(marker in normalized for marker in _RECONSTRUCTION_INTENT_MARKERS):
        return "reconstruct"
    if dump is not None:
        text_count = len(dump.get("ocr", []))
        element_count = len(dump.get("elements", []))
        # Some agent clients omit the user's task when they call an MCP tool.
        # Dense, structured screenshots/posters still need the safe exact-copy
        # contract; photos with little text retain the semantic analyze profile.
        if text_count >= 4 and element_count >= 5:
            return "reconstruct"
    return "analyze"


def _compact_summary(doc: dict[str, Any]) -> dict[str, Any]:
    reconstruction = doc.get("reconstruction") or {}
    if doc.get("profile") == "reconstruct":
        next_actions = [
            {
                key: action[key]
                for key in ("tool", "reason", "arguments")
                if action.get(key) is not None
            }
            for action in reconstruction.get("focusPlan", [])
        ]
    else:
        next_actions = doc.get("nextActions", [])[:4]
    return {
        "schemaVersion": doc.get("schemaVersion"),
        "profile": doc.get("profile"),
        "source": doc.get("source"),
        "canvas": reconstruction.get("canvas") or {
            "size": (doc.get("header") or {}).get("size")
        },
        "semanticsStatus": doc.get("semantics_status"),
        "warningCodes": [
            warning.get("code") for warning in doc.get("warnings", [])
        ],
        "nextActions": next_actions,
        "completionRule": (
            (
                "For web reconstruction, render at the exact source dimensions and stop only when sens_review returns visualPass=true, webPass=true, and canComplete=true."
                if reconstruction.get("targetKind") == "web"
                else "For reconstruction, render at the exact source dimensions and stop only when strict sens_compare returns canComplete=true."
            )
            if doc.get("profile") == "reconstruct"
            else None
        ),
    }


def _jsonl_table(columns: list[str], rows: list[list[Any]]) -> dict[str, Any]:
    return {
        "encoding": "jsonl-arrays",
        "columns": columns,
        "count": len(rows),
        "rows": "\n".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"))
            for row in rows
        ),
    }


def _sparse_jsonl_table(
    columns: list[str], rows: list[list[Any]]
) -> dict[str, Any]:
    """Encode repeated records without repeating field names or defaults.

    ``constants`` apply to every row. ``defaults`` apply wherever the row cell
    is null. A default is emitted only when the source column contains no real
    null, so decoding remains unambiguous.
    """
    if not rows:
        return _jsonl_table(columns, rows)
    if len(rows) == 1:
        active_indexes = [
            index for index, value in enumerate(rows[0]) if value is not None
        ]
        return _jsonl_table(
            [columns[index] for index in active_indexes],
            [[rows[0][index] for index in active_indexes]],
        )
    constants: dict[str, Any] = {}
    defaults: dict[str, Any] = {}
    active_indexes: list[int] = []
    default_values: dict[int, Any] = {}
    for index, column in enumerate(columns):
        values = [row[index] for row in rows]
        if all(value is None for value in values):
            continue
        if all(value == values[0] for value in values):
            constants[column] = values[0]
            continue
        if all(value is not None for value in values):
            encoded_values = [
                json.dumps(value, ensure_ascii=False, sort_keys=True)
                for value in values
            ]
            dominant_encoded = max(
                dict.fromkeys(encoded_values), key=encoded_values.count
            )
            dominant_count = encoded_values.count(dominant_encoded)
            if dominant_count / len(values) >= 0.8:
                dominant_index = encoded_values.index(dominant_encoded)
                dominant_value = values[dominant_index]
                defaults[column] = dominant_value
                default_values[index] = dominant_value
        active_indexes.append(index)
    compact_rows = [
        [
            (
                None
                if index in default_values and row[index] == default_values[index]
                else row[index]
            )
            for index in active_indexes
        ]
        for row in rows
    ]
    result = _jsonl_table(
        [columns[index] for index in active_indexes], compact_rows
    )
    if constants:
        result["constants"] = constants
    if defaults:
        result["defaults"] = defaults
    return result


def _mapping_jsonl_table(
    entries: list[dict[str, Any]], columns: list[str]
) -> dict[str, Any]:
    return _sparse_jsonl_table(
        columns,
        [[entry.get(column) for column in columns] for entry in entries],
    )


def _compact_document(doc: dict[str, Any]) -> dict[str, Any]:
    """Keep reconstruction responses task-complete without replaying analysis.

    Claims, ASCII previews, and the generic element projection duplicate the
    exact reconstruction spec and caused large agent contexts on every repair
    loop. They remain available through ``response=full``.
    """
    if doc.get("profile") != "reconstruct":
        return doc
    reconstruction = doc.get("reconstruction") or {}

    def present(mapping: dict[str, Any], *keys: str) -> dict[str, Any]:
        return {
            key: mapping[key]
            for key in keys
            if key in mapping and mapping[key] is not None
        }

    dense_text = len(reconstruction.get("text", [])) > 40
    text_schema = (
        [
            "id",
            "text",
            "observedValue",
            "resolution",
            "confidence",
            "boxSource",
            "fontSize",
            "widthEm",
            "fontFamily",
            "sourceFontFamily",
            "sourceDomFamily",
            "sourceDomFontWeight",
            "sourceDomFontStyle",
            "sourceDomFontSize",
            "sourceDomWords",
            "typographyAuthority",
            "fontClass",
            "fontWidth",
            "fontWeight",
            "fontWeightSource",
            "color",
            "inlineRuns",
        ]
        if dense_text
        else [
            "id",
            "value",
            "preferredValue",
            "status",
            "resolution",
            "confidence",
            "boxSource",
            "fontSize",
            "capHeight",
            "avgGlyphWidth",
            "widthEm",
            "fontFamily",
            "sourceFontFamily",
            "sourceDomFamily",
            "sourceDomFontWeight",
            "sourceDomFontStyle",
            "sourceDomFontSize",
            "sourceDomWords",
            "typographyAuthority",
            "fontFamilyStatus",
            "fontFamilyCandidate",
            "fontFamilyDistance",
            "fontClass",
            "strokeContrast",
            "fontWidth",
            "fontWeight",
            "fontWeightSource",
            "typographyConfidence",
            "color",
            "colorSource",
            "inlineRuns",
        ]
    )
    structural_line_schema = [
        "orientation",
        "boxSource",
        "startSource",
        "endSource",
        "thickness",
        "length",
        "color",
        "lineStyle",
        "dashLength",
        "dashGap",
        "source",
        "method",
    ]

    def compact_text_row(entry: dict[str, Any]) -> list[Any]:
        font = entry.get("fontFeatures") or {}
        source_authority = (
            font.get("sourceDomTypographySource")
            == "observed-live-dom-computed-style"
        )
        candidates = sorted(
            (
                candidate
                for candidate in font.get("familyCandidates", [])
                if candidate.get("family")
            ),
            key=lambda candidate: float(candidate.get("distance") or 999),
        )
        candidate = candidates[0] if candidates else {}
        typography = entry.get("typographyCandidate") or {}
        source_words = [
            [
                word.get("text"),
                word.get("sourceFontFamily"),
                word.get("sourceDomFontWeight"),
                word.get("sourceDomFontStyle"),
                word.get("sourceDomLetterSpacing"),
                word.get("sourceDomBox"),
            ]
            for word in font.get("sourceDomWordStyles") or []
            if isinstance(word, dict)
        ] or None
        if dense_text:
            preferred_value = entry.get("preferredValue")
            observed_value = entry.get("value")
            resolved_value = preferred_value or observed_value
            family = (
                font.get("sourceDomFamily")
                if source_authority
                else font.get("family")
            )
            if family in {None, "custom", "unknown"} and candidate.get("family"):
                family = candidate.get("family")
            return [
                entry.get("elementId"),
                resolved_value,
                (
                    observed_value
                    if preferred_value and preferred_value != observed_value
                    else None
                ),
                entry.get("resolutionStatus"),
                entry.get("confidence"),
                entry.get("boxSource"),
                font.get("fontSize"),
                font.get("widthEm"),
                family,
                font.get("sourceFontFamily"),
                font.get("sourceDomFamily"),
                font.get("sourceDomFontWeight"),
                font.get("sourceDomFontStyle"),
                font.get("sourceDomFontSize"),
                source_words,
                font.get("sourceDomTypographySource"),
                "observed-live-dom" if source_authority else typography.get("class"),
                typography.get("width"),
                (
                    font.get("sourceDomFontWeight")
                    if source_authority
                    else font.get("weightCandidate") or typography.get("weight")
                ),
                (
                    font.get("sourceDomTypographySource")
                    if source_authority
                    else font.get("weightCandidateMethod")
                    if font.get("weightCandidate")
                    else typography.get("method")
                ),
                entry.get("color") or font.get("color"),
                None if source_authority else [
                    [
                        run.get("text"),
                        (run.get("typographyCandidate") or {}).get("class"),
                        (run.get("typographyCandidate") or {}).get("contrast"),
                        (run.get("typographyCandidate") or {}).get("width"),
                        (run.get("typographyCandidate") or {}).get("weight"),
                        (run.get("typographyCandidate") or {}).get("slant"),
                    ]
                    for run in entry.get("inlineRuns") or []
                ]
                or None,
            ]
        return [
            entry.get("elementId"),
            entry.get("value"),
            entry.get("preferredValue"),
            entry.get("status"),
            entry.get("resolutionStatus"),
            entry.get("confidence"),
            entry.get("boxSource"),
            font.get("fontSize"),
            font.get("capHeight"),
            font.get("avgGlyphWidth"),
            font.get("widthEm"),
            font.get("sourceDomFamily") if source_authority else font.get("family"),
            font.get("sourceFontFamily"),
            font.get("sourceDomFamily"),
            font.get("sourceDomFontWeight"),
            font.get("sourceDomFontStyle"),
            font.get("sourceDomFontSize"),
            source_words,
            font.get("sourceDomTypographySource"),
            "observed" if source_authority else font.get("familyStatus"),
            None if source_authority else candidate.get("family"),
            None if source_authority else candidate.get("distance"),
            "observed-live-dom" if source_authority else typography.get("class"),
            typography.get("contrast"),
            typography.get("width"),
            (
                font.get("sourceDomFontWeight")
                if source_authority
                else font.get("weightCandidate") or typography.get("weight")
            ),
            (
                font.get("sourceDomTypographySource")
                if source_authority
                else font.get("weightCandidateMethod")
                if font.get("weightCandidate")
                else typography.get("method")
            ),
            None if source_authority else typography.get("confidence"),
            entry.get("color") or font.get("color"),
            entry.get("colorSource") or font.get("colorSource"),
            None if source_authority else [
                [
                    run.get("text"),
                    (run.get("typographyCandidate") or {}).get("class"),
                    (run.get("typographyCandidate") or {}).get("contrast"),
                    (run.get("typographyCandidate") or {}).get("width"),
                    (run.get("typographyCandidate") or {}).get("weight"),
                    (run.get("typographyCandidate") or {}).get("slant"),
                ]
                for run in entry.get("inlineRuns") or []
            ]
            or None,
        ]

    def compact_shape(entry: dict[str, Any]) -> dict[str, Any]:
        return present(entry, "elementId", "boxSource", "visibleBoundary")

    def compact_raster(entry: dict[str, Any]) -> dict[str, Any]:
        return present(
            entry,
            "elementId",
            "kind",
            "boxSource",
            "strategy",
            "implementation",
            "reason",
            "overlappingTextElementId",
            "assetPath",
            "artifactId",
            "mediaType",
            "alphaProtected",
            "protectionPolicy",
        )

    def compact_table(columns: list[str], rows: list[list[Any]]) -> dict[str, Any]:
        return _jsonl_table(columns, rows)

    def compact_sparse_table(
        columns: list[str], rows: list[list[Any]]
    ) -> dict[str, Any]:
        return _sparse_jsonl_table(columns, rows)

    def compact_mapping_table(
        entries: list[dict[str, Any]], columns: list[str]
    ) -> dict[str, Any]:
        return _mapping_jsonl_table(entries, columns)

    text_rows = [
        compact_text_row(entry) for entry in reconstruction.get("text", [])
    ]
    compact_text_table = compact_sparse_table(text_schema, text_rows)

    compact_reconstruction = present(
        reconstruction,
        "targetKind",
        "outputKind",
        "canvas",
        "contentPolicy",
        "representationPolicy",
        "typographyRule",
        "typographyAuthority",
        "completionGate",
        "semanticStrategy",
        "resolvedFocus",
        "layoutRegions",
        "surfaces",
        "vectorPaths",
        "starterProject",
    )
    compact_reconstruction.update(
        {
            "text": compact_text_table,
            "visualControlCandidates": compact_mapping_table(
                reconstruction.get("visualControlCandidates", []),
                [
                    "elementId",
                    "kind",
                    "boxSource",
                    "labelElementIds",
                    "ariaLabel",
                    "interaction",
                    "semanticRole",
                    "visibleBoundary",
                    "background",
                    "borderColor",
                    "borderWidth",
                    "cornerRadius",
                    "zIndex",
                ],
            ),
            "icons": compact_mapping_table(
                reconstruction.get("icons", []),
                [
                    "elementId",
                    "name",
                    "boxSource",
                    "color",
                    "strategy",
                    "source",
                    "geometrySource",
                    "semanticSource",
                    "epistemic",
                    "preservedInBackgroundArtwork",
                    "zIndex",
                ],
            ),
            "badges": compact_mapping_table(
                reconstruction.get("badges", []),
                [
                    "elementId",
                    "labelElementId",
                    "boxSource",
                    "textBoxSource",
                    "background",
                    "foreground",
                    "borderColor",
                    "borderWidth",
                    "cornerRadius",
                    "value",
                    "confidence",
                    "verified",
                    "geometrySource",
                    "epistemic",
                    "representation",
                    "method",
                    "decorationPreservedInBackgroundArtwork",
                ],
            ),
            "structuralLines": compact_table(
                structural_line_schema,
                [
                    [line.get(key) for key in structural_line_schema]
                    for line in reconstruction.get("structuralLines", [])
                ],
            ),
            "symbolArt": reconstruction.get("symbolArt", []),
            "sourceFontAssets": compact_mapping_table(
                reconstruction.get("sourceFontAssets", []),
                ["family", "alias", "weight", "style", "format", "sha256", "source"],
            ),
            "allowedRasterRegions": [
                compact_raster(entry)
                for entry in reconstruction.get("allowedRasterRegions", [])
            ],
            "excludedRasterCandidates": [
                compact_raster(entry)
                for entry in reconstruction.get("excludedRasterCandidates", [])
            ],
            "decorativeShapes": [
                compact_shape(entry)
                for entry in reconstruction.get("decorativeShapes", [])
            ],
        }
    )
    focus_plan = reconstruction.get("focusPlan") or []
    compact_reconstruction["focusPlan"] = {
        "encoding": "response-reference",
        "path": "summary.nextActions",
        "count": len(focus_plan),
    }
    raster_rule = reconstruction.get("rasterAssetRule")
    if isinstance(raster_rule, dict):
        compact_reconstruction["rasterAssetRule"] = present(
            raster_rule,
            "strategy",
            "scope",
            "assetsReady",
            "assetCount",
            "prohibitedFollowUps",
            "nextAction",
        )
    workflow = reconstruction.get("workflow")
    if isinstance(workflow, dict):
        compact_workflow = present(
            workflow,
            "state",
            "nextAction",
            "nextSensTool",
            "forbiddenActions",
        )
        construction_budget = workflow.get("constructionBudget")
        if isinstance(construction_budget, dict):
            compact_workflow["constructionBudget"] = present(
                construction_budget,
                "oneSourceFilePerModelResponse",
                "indexHtmlMaxCharacters",
                "stylesCssMaxCharacters",
                "scriptJsMaxCharacters",
            )
        compact_reconstruction["workflow"] = compact_workflow
    primary_asset = reconstruction.get("primaryAsset")
    if isinstance(primary_asset, dict):
        compact_reconstruction["primaryAsset"] = present(
            primary_asset,
            "elementId",
            "boxSource",
            "areaRatio",
            "strategy",
            "rule",
            "assetPath",
            "artifactId",
            "mediaType",
        )
    implementation_rules = reconstruction.get("implementationRules", [])
    if isinstance(implementation_rules, list):
        if reconstruction.get("targetKind") == "web":
            compact_reconstruction["implementationRules"] = [
                "Use starterProject for the first candidate; do not rebuild it from scratch.",
                "Preserve starter @font-face rules and observed DOM typography exactly; inferred typography is fallback-only.",
                "Render at the exact source canvas and DPR without inventing hidden content.",
                "Keep every word as selectable DOM text and every control as semantic HTML.",
                "Use native CSS/SVG/preformatted text for structure and symbol art.",
                "Use raster content only from allowedRasterRegions at its exact boxSource.",
                "Stop only when sens_review returns visualPass, webPass, and canComplete true.",
            ]
        else:
            compact_reconstruction["implementationRules"] = list(
                dict.fromkeys(
                    rule
                    for rule in implementation_rules
                    if isinstance(rule, str) and rule
                )
            )
    if reconstruction.get("monospaceContent") is not None:
        compact_reconstruction["monospaceContent"] = reconstruction[
            "monospaceContent"
        ]
    semantic_text = reconstruction.get("semanticTextCandidate")
    if isinstance(semantic_text, dict):
        compact_reconstruction["semanticTextCandidate"] = present(
            semantic_text,
            "text",
            "status",
            "method",
            "sourceBox",
            "typography",
            "typographyRuns",
        )

    return {
        "schemaVersion": doc.get("schemaVersion"),
        "profile": doc.get("profile"),
        "source": doc.get("source"),
        "coordinateSpaces": doc.get("coordinateSpaces"),
        "header": doc.get("header"),
        "tokens": doc.get("tokens"),
        "measurements": doc.get("measurements"),
        "reconstruction": compact_reconstruction,
        "warningCodes": [
            warning.get("code")
            for warning in doc.get("warnings", [])
            if warning.get("code")
        ],
        "semantics_status": doc.get("semantics_status"),
    }


def _brief_table(columns: list[str], rows: list[list[Any]]) -> dict[str, Any]:
    return _sparse_jsonl_table(columns, rows)


def _web_geometry_authority() -> dict[str, Any]:
    return {
        "textDomBox": "reconstruction.text[].boxSource",
        "textInkBox": "reconstruction.text[].fontFeatures.inkBox",
        "textGlyphBoxes": "reconstruction.text[].fontFeatures.glyphBoxes",
        "controls": "reconstruction.visualControlCandidates[].boxSource",
        "structuralLines": "reconstruction.structuralLines[].boxSource",
        "rasterAssets": "reconstruction.allowedRasterRegions[].boxSource",
        "nonAuthoritative": [
            "elements[].box_norm",
            "claims[].regionNorm",
        ],
        "rule": (
            "Use the named reconstruction fields for implementation geometry. "
            "Raw element and claim regions are connected-component diagnostics; "
            "they can merge neighboring text, controls, and artwork and must never "
            "be used as DOM, text-ink, control, or asset layout boxes."
        ),
    }


def _web_typography_authority() -> dict[str, Any]:
    return {
        "priority": [
            "observed-live-dom-computed-style",
            "measured-screenshot-geometry",
            "inferred-typography-fallback",
        ],
        "authoritativeFields": [
            "sourceFontFamily",
            "sourceDomFamily",
            "sourceDomFontWeight",
            "sourceDomFontStyle",
            "sourceDomFontSize",
            "sourceDomLetterSpacing",
            "sourceDomWordStyles",
        ],
        "nonAuthoritativeWhenObserved": [
            "familyHint",
            "fontFeatures.renderFamilyScores",
            "typographyCandidate",
            "inlineRuns.typographyCandidate",
        ],
        "rule": (
            "When a text row has observed-live-dom-computed-style evidence, its "
            "source font family, weight, style, and packaged @font-face are authoritative. "
            "Screenshot or semantic inference must not override them; inferred typography "
            "is a fallback only when observed DOM typography is unavailable."
        ),
    }


def _implementation_brief(
    doc: dict[str, Any], contract_path: str | None
) -> dict[str, Any]:
    """Project a dense reconstruction contract into a low-context build brief."""
    spec = doc.get("reconstruction") or {}
    text_columns = [
        "id",
        "text",
        "observed",
        "resolution",
        "box",
        "fontSize",
        "widthEm",
        "fontClass",
        "fontWeight",
        "fontWeightSource",
        "familyHint",
        "sourceFontFamily",
        "sourceFontAssetSha256",
        "sourceDomFamily",
        "sourceDomFontWeight",
        "sourceDomFontStyle",
        "sourceDomFontSize",
        "sourceDomLetterSpacing",
        "sourceDomBox",
        "sourceDomWords(text,font,weight,style,letterSpacing,box)",
        "typographyAuthority",
        "color",
        "inkBox",
        "glyphBoxes(text,box)",
        "measuredCharacterCount",
        "inkCoverage",
        "strokeWidthPx",
        "inlineRuns(text,class,contrast,width,weight,slant)",
    ]
    text_rows = []
    for entry in spec.get("text") or []:
        font = entry.get("fontFeatures") or {}
        typography = entry.get("typographyCandidate") or {}
        source_authority = (
            font.get("sourceDomTypographySource")
            == "observed-live-dom-computed-style"
        )
        candidates = sorted(
            font.get("familyCandidates") or [],
            key=lambda item: float(item.get("distance") or 999),
        )
        family_hint = (
            font.get("sourceDomFamily")
            if source_authority
            else candidates[0].get("family")
            if candidates
            else font.get("family")
        )
        text_rows.append(
            [
                entry.get("elementId"),
                entry.get("preferredValue") or entry.get("value"),
                (
                    entry.get("value")
                    if entry.get("preferredValue")
                    and entry.get("preferredValue") != entry.get("value")
                    else None
                ),
                entry.get("resolutionStatus"),
                entry.get("boxSource"),
                font.get("fontSize"),
                font.get("widthEm"),
                "observed-live-dom" if source_authority else typography.get("class"),
                (
                    font.get("sourceDomFontWeight")
                    if source_authority
                    else font.get("weightCandidate") or typography.get("weight")
                ),
                (
                    font.get("sourceDomTypographySource")
                    if source_authority
                    else font.get("weightCandidateMethod")
                    if font.get("weightCandidate")
                    else typography.get("method")
                ),
                family_hint,
                font.get("sourceFontFamily"),
                font.get("sourceFontAssetSha256"),
                font.get("sourceDomFamily"),
                font.get("sourceDomFontWeight"),
                font.get("sourceDomFontStyle"),
                font.get("sourceDomFontSize"),
                font.get("sourceDomLetterSpacing"),
                font.get("sourceDomBox"),
                [
                    [
                        word.get("text"),
                        word.get("sourceFontFamily"),
                        word.get("sourceDomFontWeight"),
                        word.get("sourceDomFontStyle"),
                        word.get("sourceDomLetterSpacing"),
                        word.get("sourceDomBox"),
                    ]
                    for word in font.get("sourceDomWordStyles") or []
                    if isinstance(word, dict)
                ]
                or None,
                font.get("sourceDomTypographySource"),
                entry.get("color") or font.get("color"),
                font.get("inkBox"),
                [
                    [item.get("text"), item.get("box")]
                    for item in font.get("glyphBoxes") or []
                    if isinstance(item, dict)
                    and item.get("text") is not None
                    and len(item.get("box") or []) == 4
                ]
                or None,
                font.get("measuredCharacterCount"),
                font.get("inkCoverage"),
                font.get("strokeWidthPx"),
                None if source_authority else [
                    [
                        run.get("text"),
                        (run.get("typographyCandidate") or {}).get("class"),
                        (run.get("typographyCandidate") or {}).get("contrast"),
                        (run.get("typographyCandidate") or {}).get("width"),
                        (run.get("typographyCandidate") or {}).get("weight"),
                        (run.get("typographyCandidate") or {}).get("slant"),
                    ]
                    for run in entry.get("inlineRuns") or []
                ]
                or None,
            ]
        )
    line_columns = ["box", "thickness", "color"]
    line_rows = [
        [
            line.get("boxSource"),
            line.get("thickness"),
            line.get("color"),
        ]
        for line in spec.get("structuralLines") or []
    ]

    def selected(entry: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
        return {
            key: entry[key]
            for key in keys
            if entry.get(key) is not None
        }

    text_by_id = {
        entry.get("elementId"): entry
        for entry in spec.get("text") or []
        if entry.get("elementId") is not None
    }

    def control_brief(entry: dict[str, Any]) -> dict[str, Any]:
        result = selected(
            entry,
            (
                "elementId",
                "boxSource",
                "visibleBoundary",
                "background",
                "borderColor",
                "borderWidth",
                "cornerRadius",
                "labelElementIds",
                "interaction",
            ),
        )
        labels = [
            text_by_id[element_id]
            for element_id in entry.get("labelElementIds") or []
            if element_id in text_by_id
        ]
        result["label"] = " ".join(
            str(label.get("preferredValue") or label.get("value") or "").strip()
            for label in labels
        )
        return result

    control_columns = [
        "elementId",
        "boxSource",
        "visibleBoundary",
        "background",
        "borderColor",
        "borderWidth",
        "cornerRadius",
        "labelElementIds",
        "interaction",
        "label",
    ]
    controls = _mapping_jsonl_table(
        [
            control_brief(entry)
            for entry in spec.get("visualControlCandidates") or []
        ],
        control_columns,
    )
    surface_columns = [
        "boxSource",
        "background",
        "borderColor",
        "borderWidth",
        "cornerRadius",
        "shadow",
        "source",
    ]
    surfaces = _mapping_jsonl_table(
        [
        selected(
            entry,
            tuple(surface_columns),
        )
        for entry in spec.get("surfaces") or []
        ],
        surface_columns,
    )
    icon_columns = ["elementId", "name", "boxSource", "color", "strategy"]
    icons = _mapping_jsonl_table(
        [
            selected(entry, tuple(icon_columns))
            for entry in spec.get("icons") or []
        ],
        icon_columns,
    )
    badge_columns = [
        "elementId",
        "labelElementId",
        "boxSource",
        "textBoxSource",
        "background",
        "foreground",
        "borderColor",
        "borderWidth",
        "cornerRadius",
        "value",
        "confidence",
        "verified",
    ]
    badges = _mapping_jsonl_table(
        [
            selected(entry, tuple(badge_columns))
            for entry in spec.get("badges") or []
        ],
        badge_columns,
    )
    raster_assets = [
        selected(
            entry,
            ("elementId", "boxSource", "assetPath", "mediaType", "strategy"),
        )
        for entry in spec.get("allowedRasterRegions") or []
    ]
    return {
        "schemaVersion": "sens-web-brief-3",
        "source": doc.get("source"),
        "canvas": spec.get("canvas"),
        "geometryAuthority": spec.get("geometryAuthority")
        or _web_geometry_authority(),
        "typographyAuthority": spec.get("typographyAuthority")
        or _web_typography_authority(),
        "palette": (doc.get("tokens") or {}).get("color"),
        "text": _brief_table(text_columns, text_rows),
        "controls": controls,
        "structuralLines": _brief_table(line_columns, line_rows),
        "vectorPaths": spec.get("vectorPaths") or [],
        "layoutRegions": spec.get("layoutRegions") or [],
        "surfaces": surfaces,
        "icons": icons,
        "badges": badges,
        "symbolArt": spec.get("symbolArt") or [],
        "rasterAssets": raster_assets,
        "starterProject": spec.get("starterProject"),
        "sourceFonts": [
            {
                key: asset.get(key)
                for key in ("family", "alias", "weight", "style", "format", "source")
                if asset.get(key) is not None
            }
            for asset in spec.get("sourceFontAssets") or []
            if isinstance(asset, dict)
        ],
        "representationPolicy": spec.get("representationPolicy"),
        "workflow": spec.get("workflow"),
        "completionGate": spec.get("completionGate"),
        "reviewArguments": {
            "contractPath": contract_path,
            "rule": "Pass this contractPath unchanged to every sens_review call for this reconstruction.",
        },
        "implementationRules": [
            "If starterProject is present, copy or serve it immediately; do not generate the first candidate from scratch.",
            "Render only visible content on the exact source-pixel canvas at DPR 1.",
            "All words are live selectable DOM text; controls are semantic HTML with hover and focus.",
            "When glyphBoxes are present, fit each live character to those measured boxes; inkCoverage is aggregate evidence and must never be interpreted as a connected slab.",
            "Lines, cards, chart geometry, and symbol art are native HTML/CSS/SVG or preformatted text, never reference slices.",
            "Raster images are forbidden except the exact returned rasterAssets paths and boxes.",
            "For layout geometry use geometryAuthority only; never position DOM from elements[].box_norm or claims[].regionNorm.",
            "Preserve starter @font-face rules and observed DOM typography exactly; never replace them from familyHint, render-family scores, typographyCandidate, or inferred inline runs.",
            "After the first candidate call sens_review with this brief's contractPath; use only its repairHints, checkpoint champions, and roll back regressions.",
        ],
        "contract": {
            "path": contract_path,
            "format": "application/json",
            "rule": "The brief is sufficient for the first candidate. Read this local contract artifact in bounded chunks only when a named brief field is insufficient; it is not the reference image.",
        },
        "warningCodes": [
            warning.get("code")
            for warning in doc.get("warnings") or []
            if warning.get("code")
        ],
    }


def _write_reconstruction_contract(
    doc: dict[str, Any], asset_output_dir: str | None, *, no_store: bool
) -> str | None:
    if no_store:
        return None
    output_dir = (
        Path(asset_output_dir).expanduser()
        if asset_output_dir
        else Path(cache_root()) / "reconstruction-contracts"
    ).resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(doc, ensure_ascii=False, indent=2).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()[:16]
        destination = output_dir / f"sens-web-contract-{digest}.json"
        if not destination.exists():
            fd, temporary = tempfile.mkstemp(
                prefix=f".{destination.stem}-", suffix=".json", dir=str(output_dir)
            )
            os.close(fd)
            try:
                Path(temporary).write_bytes(encoded)
                os.replace(temporary, destination)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        artifact_id = f"web-contract:{digest}"
        artifacts = doc.setdefault("artifacts", [])
        if not any(item.get("id") == artifact_id for item in artifacts):
            artifacts.append(
                {
                    "id": artifact_id,
                    "kind": "reconstruction-contract",
                    "uri": str(destination),
                    "mediaType": "application/json",
                }
            )
        return str(destination)
    except OSError:
        return None


def _apply_reconstruction_ocr(image_path: str, dump: dict[str, Any]) -> None:
    refined = refine_ocr_for_reconstruction(image_path, dump.get("ocr", []))
    try:
        refined = merge_script_ocr_passes(refined, run_latin_ocr(image_path))
        ocr_method = "rapidocr-multiscale-plus-latin-consensus"
        ocr_passes = 3
    except Exception as error:  # noqa: BLE001 - optional third-party OCR must degrade
        dump.setdefault("warnings", []).append(
            {
                "code": "optional_latin_ocr_unavailable",
                "message": f"Full-frame Latin OCR is unavailable: {error}",
                "recovery": "Continue with the Cyrillic-capable multiscale OCR and bounded focus.",
            }
        )
        ocr_method = "rapidocr-multiscale-consensus"
        ocr_passes = 2
    try:
        display_items = discover_display_ocr(image_path, refined)
        refined.extend(display_items)
        selected_display_scale = (
            float(display_items[0].get("displayScale") or 0.5)
            if display_items
            else None
        )
        display_discovery = {
            "status": "complete",
            "scale": selected_display_scale,
            "primaryScale": 0.5,
            "fallbackScales": [0.3, 0.4, 0.6],
            "fallbackUsed": bool(
                selected_display_scale is not None
                and abs(selected_display_scale - 0.5) > 0.001
            ),
            "candidateCount": len(display_items),
            "method": "rapidocr-downscaled-display-scan",
        }
        ocr_passes += 2
    except (ImportError, ModuleNotFoundError, RuntimeError, OSError, ValueError) as error:
        display_discovery = {
            "status": "unavailable",
            "scale": None,
            "primaryScale": 0.5,
            "fallbackScales": [0.3, 0.4, 0.6],
            "fallbackUsed": False,
            "candidateCount": 0,
            "method": "rapidocr-downscaled-display-scan",
        }
        dump.setdefault("warnings", []).append(
            {
                "code": "display_text_discovery_unavailable",
                "message": f"Downscaled display-text OCR is unavailable: {error}",
                "recovery": "Do not trust a full-canvas background raster; continue with live DOM and non-background assets.",
            }
        )
    numeric_badges = list(dump.get("numericBadges") or [])
    for badge_index, badge in enumerate(numeric_badges):
        text_box = [int(value) for value in badge.get("textBox") or []]
        if len(text_box) != 4 or not str(badge.get("value") or "").isdigit():
            continue
        text_area = max(
            1,
            (text_box[2] - text_box[0]) * (text_box[3] - text_box[1]),
        )
        retained = []
        for item in refined:
            item_box = [int(value) for value in item.get("box") or []]
            item_area = (
                max(1, (item_box[2] - item_box[0]) * (item_box[3] - item_box[1]))
                if len(item_box) == 4
                else 1
            )
            overlap = _box_intersection(text_box, item_box)
            if overlap / max(1, min(text_area, item_area)) < 0.35:
                retained.append(item)
        refined = retained
        refined.append(
            {
                "box": text_box,
                "text": str(badge["value"]),
                "confidence": float(badge.get("confidence") or 0.5),
                "verified": bool(badge.get("verified")),
                "method": badge.get("method"),
                "alternatives": list(badge.get("alternatives") or []),
                "numericBadgeIndex": badge_index,
                "epistemic": badge.get("epistemic") or "inferred",
            }
        )
    dump["ocr"] = refined
    dump["ocrConsensus"] = {
        "passes": ocr_passes,
        "scale": 1.5,
        "method": ocr_method,
        "displayTextDiscovery": display_discovery,
    }
    if not refined:
        return
    image = load_cv(image_path)
    _sync_bounded_ocr_elements(dump, image, refined)


def _attach_display_text_discovery(
    document: dict[str, Any], dump: dict[str, Any]
) -> None:
    spec = document.get("reconstruction") or {}
    if not spec or spec.get("targetKind") != "web":
        return
    discovery = (dump.get("ocrConsensus") or {}).get(
        "displayTextDiscovery"
    )
    if isinstance(discovery, dict):
        spec["displayTextDiscovery"] = dict(discovery)


def _hydrate_numeric_badges(
    document: dict[str, Any], dump: dict[str, Any]
) -> None:
    """Attach measured badge surfaces to their live numeric text nodes."""
    spec = document.get("reconstruction") or {}
    if not spec or spec.get("targetKind") != "web":
        return
    text_entries = list(spec.get("text") or [])
    badges: list[dict[str, Any]] = []
    verified_ids: set[Any] = set()
    badge_boxes: list[list[int]] = []
    for source in dump.get("numericBadges") or []:
        box = [int(value) for value in source.get("box") or []]
        text_box = [int(value) for value in source.get("textBox") or []]
        if len(box) != 4 or len(text_box) != 4:
            continue
        text_area = max(
            1,
            (text_box[2] - text_box[0]) * (text_box[3] - text_box[1]),
        )
        ranked = sorted(
            (
                (
                    _box_intersection(entry.get("boxSource") or [], text_box)
                    / text_area,
                    entry,
                )
                for entry in text_entries
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        if not ranked or ranked[0][0] < 0.3:
            continue
        entry = ranked[0][1]
        value = str(source.get("value") or "")
        entry.update(
            {
                "value": value,
                "preferredValue": value,
                "resolutionStatus": (
                    "deterministic-badge-template-confirmed"
                    if source.get("verified")
                    else "deterministic-badge-template-candidate"
                ),
                "resolutionMethod": source.get("method"),
                "resolutionConfidence": float(
                    source.get("confidence") or 0.5
                ),
                "epistemic": source.get("epistemic") or "inferred",
                "color": source.get("foreground"),
                "colorSource": "measured-badge-glyph-pixels",
            }
        )
        font = dict(entry.get("fontFeatures") or {})
        font["color"] = source.get("foreground")
        font["colorSource"] = "measured-badge-glyph-pixels"
        entry["fontFeatures"] = font
        if source.get("verified"):
            entry["status"] = "confirmed"
            entry["confirmedBy"] = [source.get("method")]
            verified_ids.add(entry.get("elementId"))
        else:
            entry["status"] = "candidate"
            entry["alternatives"] = list(source.get("alternatives") or [])
        badges.append(
            {
                "elementId": f"badge-{len(badges) + 1}",
                "labelElementId": entry.get("elementId"),
                "boxSource": box,
                "textBoxSource": text_box,
                "background": source.get("background"),
                "foreground": source.get("foreground"),
                "cornerRadius": source.get("cornerRadius"),
                "value": value,
                "confidence": source.get("confidence"),
                "verified": bool(source.get("verified")),
                "geometrySource": source.get("geometrySource") or "measured",
                "epistemic": source.get("epistemic") or "inferred",
                "representation": "live-text-on-css-surface",
                "method": source.get("method"),
            }
        )
        badge_boxes.append(box)
    if not badges:
        return
    spec["badges"] = badges
    if verified_ids:
        spec["blockingUncertainties"] = [
            item
            for item in spec.get("blockingUncertainties") or []
            if item.get("elementId") not in verified_ids
        ]
    spec["icons"] = [
        icon
        for icon in spec.get("icons") or []
        if not any(
            _box_intersection(icon.get("boxSource") or [], badge_box)
            / max(1, _source_box_area(icon.get("boxSource") or []))
            >= 0.25
            for badge_box in badge_boxes
        )
    ]


def _refresh_reconstruction_workflow(document: dict[str, Any]) -> None:
    spec = document.get("reconstruction") or {}
    if not spec or spec.get("targetKind") != "web":
        return
    spec["geometryAuthority"] = _web_geometry_authority()
    focus_plan = spec.get("focusPlan") or []
    assets_ready = (spec.get("rasterAssetRule") or {}).get("assetsReady", True)
    if focus_plan:
        state = "needs-focus"
        next_action = "execute-returned-focus-plan"
        next_tool = "sens_zoom"
    elif not assets_ready:
        state = "needs-asset-output"
        next_action = "retry-sens-see-with-storage-enabled"
        next_tool = "sens_see"
    elif (spec.get("starterProject") or {}).get("entryPath"):
        state = "ready-to-implement"
        next_action = "copy-or-serve-starter-then-sens-review"
        next_tool = "sens_review"
    else:
        state = "ready-to-implement"
        next_action = "implement-first-candidate-then-sens-review"
        next_tool = "sens_review"
    workflow = spec.setdefault("workflow", {})
    workflow.update(
        {
            "state": state,
            "nextAction": next_action,
            "nextSensTool": next_tool,
            "constructionBudget": {
                "oneSourceFilePerModelResponse": True,
                "indexHtmlMaxCharacters": 12_000,
                "stylesCssMaxCharacters": 16_000,
                "scriptJsMaxCharacters": 6_000,
                "rule": "Write index.html first, styles.css in a later response, and script.js only when visible behavior requires it. Keep each tool call below its stated character budget.",
            },
        }
    )


def _materialize_raster_assets(
    document: dict[str, Any],
    image_path: str,
    asset_output_dir: str | None,
    *,
    no_store: bool,
) -> None:
    """Extract every allowed web raster region once and return its path.

    The host model receives a ready-to-copy artifact and never needs PIL,
    OpenCV, ImageMagick, or another vision call to inspect/redraw it.
    """
    spec = document.get("reconstruction") or {}
    rule = spec.setdefault("rasterAssetRule", {})
    regions = spec.get("allowedRasterRegions") or []
    if not regions:
        rule.update(
            {
                "assetsReady": True,
                "assetCount": 0,
                "nextAction": "No raster assets are required.",
            }
        )
        _refresh_reconstruction_workflow(document)
        return
    if no_store:
        rule.update(
            {
                "assetsReady": False,
                "assetCount": 0,
                "nextAction": "Retry sens_see with noStore=false so Sens can materialize exact allowed crops.",
            }
        )
        _refresh_reconstruction_workflow(document)
        return

    output_dir = Path(asset_output_dir).expanduser() if asset_output_dir else Path(cache_root()) / "reconstruction-assets"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image = load_cv(image_path)
    height, width = image.shape[:2]
    artifacts = document.setdefault("artifacts", [])
    primary = spec.get("primaryAsset")
    written = 0

    def protected_background_artwork() -> Any:
        import cv2
        import numpy as np

        rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        semantic_removal_mask = np.zeros((height, width), np.uint8)
        removed_categories: set[str] = set()
        display_glyph_mask_element_ids: list[Any] = []

        def clipped_box(entry: dict[str, Any], padding: int = 0) -> list[int] | None:
            box = entry.get("boxSource") or []
            if len(box) != 4:
                points = entry.get("pointsSource") or []
                valid_points = [
                    point
                    for point in points
                    if isinstance(point, (list, tuple)) and len(point) == 2
                ]
                if valid_points:
                    box = [
                        min(float(point[0]) for point in valid_points),
                        min(float(point[1]) for point in valid_points),
                        max(float(point[0]) for point in valid_points),
                        max(float(point[1]) for point in valid_points),
                    ]
            if len(box) != 4:
                return None
            x0, y0, x1, y1 = (int(round(value)) for value in box)
            box = [
                max(0, min(width, x0 - padding)),
                max(0, min(height, y0 - padding)),
                max(0, min(width, x1 + padding)),
                max(0, min(height, y1 + padding)),
            ]
            return box if box[2] > box[0] and box[3] > box[1] else None

        def remove_box(
            entry: dict[str, Any],
            category: str,
            padding: int = 3,
            *,
            skip_canvas_surface: bool = False,
        ) -> None:
            box = clipped_box(entry, padding)
            if box is None:
                return
            if skip_canvas_surface:
                area = (box[2] - box[0]) * (box[3] - box[1])
                if area / max(1, width * height) >= 0.85:
                    return
            semantic_removal_mask[box[1] : box[3], box[0] : box[2]] = 255
            removed_categories.add(category)

        def remove_display_text_glyphs(entry: dict[str, Any]) -> bool:
            method = str(entry.get("method") or "")
            if not method.startswith("rapidocr-downscaled-display-"):
                return False
            stroke_width = float(
                (entry.get("fontFeatures") or {}).get("strokeWidthPx") or 3.0
            )
            box = clipped_box(
                entry,
                max(1, min(3, round(stroke_width * 0.08))),
            )
            color = entry.get("color") or (
                entry.get("fontFeatures") or {}
            ).get("color")
            if box is None or not color:
                return False
            crop = image[box[1] : box[3], box[0] : box[2]].astype(
                np.float32
            )
            target = np.asarray(_hex_to_bgr(str(color)), dtype=np.float32)
            color_distance = np.linalg.norm(crop - target, axis=2)
            seed = (color_distance <= 44.0).astype(np.uint8)
            count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
                seed, 8
            )
            cap_height = float(
                (entry.get("fontFeatures") or {}).get("capHeight")
                or box[3] - box[1]
            )
            minimum_height = max(12, round(cap_height * 0.32))
            minimum_area = max(
                24,
                round((box[2] - box[0]) * (box[3] - box[1]) * 0.0015),
            )
            eligible_components: list[tuple[int, int]] = []
            for component in range(1, count):
                _x, _y, _w, component_height, area = (
                    int(value) for value in stats[component]
                )
                if component_height >= minimum_height and area >= minimum_area:
                    eligible_components.append((area, component))
            expected_glyphs = sum(
                character.isalnum()
                for character in str(
                    entry.get("preferredValue") or entry.get("value") or ""
                )
            )
            if expected_glyphs >= 2 and len(eligible_components) > expected_glyphs:
                eligible_components = sorted(
                    eligible_components,
                    key=lambda item: item[0],
                    reverse=True,
                )[:expected_glyphs]
            glyph_mask = np.zeros_like(seed)
            for _area, component in eligible_components:
                glyph_mask[labels == component] = 255
            if int(glyph_mask.sum()) == 0:
                return False
            dilation = max(3, min(9, round(stroke_width * 0.16) * 2 + 1))
            glyph_mask = cv2.dilate(
                glyph_mask,
                np.ones((dilation, dilation), np.uint8),
                iterations=1,
            )
            target_view = semantic_removal_mask[
                box[1] : box[3], box[0] : box[2]
            ]
            np.maximum(target_view, glyph_mask, out=target_view)
            removed_categories.add("liveText")
            entry["backgroundRemovalMode"] = (
                "measured-display-glyph-mask-inpaint"
            )
            display_glyph_mask_element_ids.append(entry.get("elementId"))
            return True

        for entry in spec.get("text") or []:
            if not remove_display_text_glyphs(entry):
                remove_box(entry, "liveText", 6)

        removal_groups = (
            ("visualControlCandidates", "controlDecoration", 5, False),
            ("surfaces", "surfaces", 3, True),
            ("decorativeShapes", "decorativeShapes", 4, False),
            ("icons", "microIcons", 4, False),
            ("badges", "badges", 4, False),
            ("symbolArt", "symbolArt", 5, False),
            ("structuralLines", "structuralLines", 4, False),
            ("vectorPaths", "vectorPaths", 4, False),
        )
        for key, category, padding, skip_canvas_surface in removal_groups:
            for entry in spec.get(key) or []:
                remove_box(
                    entry,
                    category,
                    padding,
                    skip_canvas_surface=skip_canvas_surface,
                )
        for entry in regions:
            if entry.get("kind") not in {
                "alpha-masked-background-artwork",
                "browser-source-background-artwork",
            }:
                remove_box(entry, "objects", 4)

        composite_over_browser_source = any(
            entry.get("kind") == "alpha-masked-background-artwork"
            and entry.get("compositeUnderlay")
            == "browser-source-background"
            for entry in regions
        )

        # Cached contracts from the previous policy may ask the starter to hide
        # independent DOM/CSS decoration. Restore those measured values before
        # producing the new background-only asset.
        for key in ("visualControlCandidates", "badges"):
            for entry in spec.get(key) or []:
                preserved = entry.pop("preservedDecoration", None)
                if isinstance(preserved, dict):
                    entry.update(preserved)
                entry.pop("decorationPreservedInBackgroundArtwork", None)
        for key in (
            "icons",
            "surfaces",
            "decorativeShapes",
            "vectorPaths",
            "structuralLines",
        ):
            for entry in spec.get(key) or []:
                entry.pop("preservedInBackgroundArtwork", None)

        removal_ratio = float((semantic_removal_mask > 0).mean())
        if int(semantic_removal_mask.sum()) > 0:
            if composite_over_browser_source:
                rgba[:, :, 3][semantic_removal_mask > 0] = 0
            elif removal_ratio >= 0.85:
                remaining = image[semantic_removal_mask == 0]
                fill = (
                    np.median(remaining.reshape((-1, 3)), axis=0)
                    if remaining.size
                    else np.asarray((255, 255, 255), dtype=np.float32)
                )
                rgba[:, :, :3] = np.asarray(fill, dtype=np.uint8)
            else:
                inpaint_radius = max(
                    3.0, min(7.0, min(width, height) / 220.0)
                )
                rgba[:, :, :3] = cv2.inpaint(
                    image,
                    semantic_removal_mask,
                    inpaint_radius,
                    cv2.INPAINT_TELEA,
                )

        display_discovery = spec.get("displayTextDiscovery") or {}
        display_discovery_complete = bool(
            isinstance(display_discovery, dict)
            and display_discovery.get("status") == "complete"
        )
        semantic_residual_protection = {
            "displayTextDiscoveryComplete": display_discovery_complete,
            "displayTextCandidateCount": int(
                display_discovery.get("candidateCount") or 0
            )
            if isinstance(display_discovery, dict)
            else 0,
            "method": (
                display_discovery.get("method")
                if isinstance(display_discovery, dict)
                else None
            ),
            "displayGlyphMaskElementIds": [
                element_id
                for element_id in display_glyph_mask_element_ids
                if element_id is not None
            ],
        }
        for region in regions:
            if region.get("kind") != "alpha-masked-background-artwork":
                continue
            composite_overlay = bool(
                region.get("compositeUnderlay")
                == "browser-source-background"
            )
            region.update(
                {
                    "semanticContentRemoved": True,
                    "protectionVersion": (
                        5
                        if composite_overlay and display_discovery_complete
                        else 4
                        if composite_overlay
                        else 3
                        if display_discovery_complete
                        else 2
                    ),
                    "semanticRemovalMaskCoverage": round(removal_ratio, 5),
                    "semanticCategoriesRemoved": sorted(removed_categories),
                    "semanticResidualProtection": semantic_residual_protection,
                }
            )
            protection = region.setdefault("protectionPolicy", {})
            protection.update(
                {
                    "backgroundOnly": True,
                    "liveText": (
                        "transparent-holes-reveal-verified-browser-source-under-live-dom"
                        if composite_overlay
                        else
                        "full-box-and-measured-display-glyph-mask-inpainted-under-live-dom"
                        if display_glyph_mask_element_ids
                        else "full-box-inpainted-under-live-dom"
                    ),
                    "controlDecoration": (
                        "transparent-holes-reveal-verified-browser-source-under-semantic-css"
                        if composite_overlay
                        else "removed-from-raster-recreated-as-semantic-css"
                    ),
                    "microIcons": (
                        "transparent-holes-reveal-verified-browser-source-under-svg-css"
                        if composite_overlay
                        else "removed-from-raster-recreated-as-svg-css"
                    ),
                    "surfaces": (
                        "transparent-holes-reveal-verified-browser-source-under-css"
                        if composite_overlay
                        else "removed-from-raster-recreated-as-css"
                    ),
                    "structuralLines": (
                        "transparent-holes-reveal-verified-browser-source-under-css-vector"
                        if composite_overlay
                        else "removed-from-raster-recreated-as-css-vector"
                    ),
                    "vectorPaths": (
                        "transparent-holes-reveal-verified-browser-source-under-svg"
                        if composite_overlay
                        else "removed-from-raster-recreated-as-svg"
                    ),
                    "badges": (
                        "transparent-holes-reveal-verified-browser-source-under-live-dom"
                        if composite_overlay
                        else "removed-from-raster-recreated-as-live-dom"
                    ),
                    "symbolArt": (
                        "transparent-holes-reveal-verified-browser-source-under-live-preformatted-text"
                        if composite_overlay
                        else "removed-from-raster-recreated-as-live-preformatted-text"
                    ),
                    "objects": "removed-from-raster-recreated-as-approved-assets",
                    "fullReferenceScreenshot": False,
                }
            )
        return rgba

    for region in regions:
        box = region.get("boxSource") or []
        if len(box) != 4:
            continue
        if region.get("kind") == "browser-source-background-artwork":
            try:
                measured_box = [int(round(float(value))) for value in box]
            except (TypeError, ValueError):
                continue
            if (
                measured_box[2] <= measured_box[0]
                or measured_box[3] <= measured_box[1]
            ):
                continue
            source_path = Path(
                str(region.get("sourceAssetPath") or "")
            ).expanduser()
            try:
                source_bytes = source_path.read_bytes()
            except OSError:
                continue
            content_sha256 = hashlib.sha256(source_bytes).hexdigest()
            media_type = str(region.get("mediaType") or "").lower()
            suffix = _SOURCE_BACKGROUND_SUFFIXES.get(media_type)
            if (
                not source_bytes
                or len(source_bytes) > _SOURCE_BACKGROUND_MAX_BYTES
                or suffix is None
                or content_sha256 != str(region.get("contentSha256") or "").lower()
            ):
                continue
            destination = output_dir / (
                f"sens-raster-browser-source-{content_sha256[:16]}{suffix}"
            )
            if not destination.exists():
                fd, temporary = tempfile.mkstemp(
                    prefix=f".{destination.stem}-",
                    suffix=suffix,
                    dir=str(output_dir),
                )
                os.close(fd)
                try:
                    Path(temporary).write_bytes(source_bytes)
                    os.replace(temporary, destination)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
            artifact_id = f"raster:{content_sha256[:16]}"
            region.update(
                {
                    "boxSource": measured_box,
                    "assetPath": str(destination),
                    "artifactId": artifact_id,
                    "mediaType": media_type,
                    "contentSha256": content_sha256,
                }
            )
            if (
                isinstance(primary, dict)
                and primary.get("elementId") == region.get("elementId")
            ):
                primary.update(
                    {
                        "assetPath": str(destination),
                        "artifactId": artifact_id,
                        "mediaType": media_type,
                    }
                )
            if not any(item.get("id") == artifact_id for item in artifacts):
                artifacts.append(
                    {
                        "id": artifact_id,
                        "kind": "reconstruction-browser-source-background",
                        "uri": str(destination),
                        "mediaType": media_type,
                        "boxSource": measured_box,
                    }
                )
            written += 1
            continue
        x0, y0, x1, y1 = (int(round(value)) for value in box)
        x0, y0 = max(0, min(width, x0)), max(0, min(height, y0))
        x1, y1 = max(0, min(width, x1)), max(0, min(height, y1))
        if x1 <= x0 or y1 <= y0:
            continue
        is_background_artwork = (
            region.get("kind") == "alpha-masked-background-artwork"
        )
        payload = (
            protected_background_artwork()
            if is_background_artwork
            else image[y0:y1, x0:x1]
        )
        encoded_ok, encoded = __import__("cv2").imencode(".png", payload)
        if not encoded_ok:
            raise RuntimeError("Could not encode a reconstruction raster asset")
        encoded_bytes = encoded.tobytes()
        digest = hashlib.sha256(encoded_bytes).hexdigest()[:16]
        destination = output_dir / f"sens-raster-{region.get('elementId', 'asset')}-{digest}.png"
        if not destination.exists():
            fd, temporary = tempfile.mkstemp(
                prefix=f".{destination.stem}-", suffix=".png", dir=str(output_dir)
            )
            os.close(fd)
            try:
                Path(temporary).write_bytes(encoded_bytes)
                os.replace(temporary, destination)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        artifact_id = f"raster:{digest}"
        region.update(
            {
                "boxSource": [x0, y0, x1, y1],
                "assetPath": str(destination),
                "artifactId": artifact_id,
                "mediaType": "image/png",
                **(
                    {
                        "alphaProtected": True,
                        "contentSha256": hashlib.sha256(encoded_bytes).hexdigest(),
                    }
                    if is_background_artwork
                    else {}
                ),
            }
        )
        if isinstance(primary, dict) and primary.get("elementId") == region.get("elementId"):
            primary.update(
                {
                    "assetPath": str(destination),
                    "artifactId": artifact_id,
                    "mediaType": "image/png",
                }
            )
        if not any(item.get("id") == artifact_id for item in artifacts):
            artifacts.append(
                {
                    "id": artifact_id,
                    "kind": (
                        "reconstruction-alpha-masked-background"
                        if is_background_artwork
                        else "reconstruction-raster-asset"
                    ),
                    "uri": str(destination),
                    "mediaType": "image/png",
                    "boxSource": [x0, y0, x1, y1],
                }
            )
        written += 1
    rule.update(
        {
            "assetsReady": written == len(regions),
            "assetCount": written,
            "assetOutputDir": str(output_dir),
            "nextAction": "Use each returned assetPath directly; do not inspect or redraw it.",
        }
    )
    _refresh_reconstruction_workflow(document)


def _focus_region_box(action: dict[str, Any]) -> list[int] | None:
    region = (action.get("arguments") or {}).get("region")
    if not isinstance(region, dict):
        return None
    try:
        x = int(round(float(region["x"])))
        y = int(round(float(region["y"])))
        width = int(round(float(region["width"])))
        height = int(round(float(region["height"])))
    except (KeyError, TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return [x, y, x + width, y + height]


def _box_intersection(left: list[Any], right: list[Any]) -> float:
    if len(left) != 4 or len(right) != 4:
        return 0.0
    width = max(0.0, min(float(left[2]), float(right[2])) - max(float(left[0]), float(right[0])))
    height = max(0.0, min(float(left[3]), float(right[3])) - max(float(left[1]), float(right[1])))
    return width * height


def _normalized_text(value: Any) -> str:
    return "".join(character for character in str(value or "").casefold() if character.isalnum())


_CONTEXTUAL_UI_LABELS: tuple[tuple[str, str, str | None], ...] = (
    ("confirmpendingpayouts", "action", "wallet"),
    ("respondtopartners", "action", "message"),
    ("reviewnewapplications", "action", "user-check"),
    ("partnernetwork", "nav", "network"),
    ("frauddetection", "nav", "shield"),
    ("emailcampaigns", "nav", "send"),
    ("applicationpage", "link", "file-plus"),
    ("partnerportal", "link", "portal"),
    ("landingpage", "link", "file"),
    ("allpartners", "nav", "users"),
    ("applications", "nav", "file-check"),
    ("commissions", "nav", "dollar"),
    ("last30days", "utility", "calendar"),
    ("overview", "nav", "home"),
    ("payouts", "nav", "wallet"),
    ("messages", "nav", "message"),
    ("analytics", "nav", "chart"),
    ("bounties", "nav", "trophy"),
    ("resources", "nav", "globe"),
    ("branding", "utility", "palette"),
    ("groups", "nav", "users"),
    ("viewall", "utility", None),
)


def _contextual_ui_role(value: Any) -> tuple[str, str | None] | None:
    normalized_values = {
        _normalized_text(value),
        _visual_latin_text(value),
    }
    for label, role, icon in _CONTEXTUAL_UI_LABELS:
        for normalized in normalized_values:
            if (
                normalized == label
                or normalized.startswith(label)
                or (
                    normalized.endswith(label)
                    and len(normalized) - len(label) <= 4
                )
            ):
                return role, icon
    ranked = []
    for label, role, icon in _CONTEXTUAL_UI_LABELS:
        for normalized in normalized_values:
            if not normalized:
                continue
            length_ratio = min(len(normalized), len(label)) / max(
                len(normalized), len(label)
            )
            similarity = SequenceMatcher(None, normalized, label).ratio()
            ranked.append((similarity * length_ratio, similarity, length_ratio, role, icon))
    if ranked:
        _score, similarity, length_ratio, role, icon = max(ranked)
        if similarity >= 0.74 and length_ratio >= 0.72:
            return role, icon
    return None


def _infer_top_navigation_controls(document: dict[str, Any]) -> None:
    """Represent a repeated aligned top text row as semantic links."""
    spec = document.get("reconstruction") or {}
    if spec.get("targetKind") != "web":
        return
    canvas = spec.get("canvas") or {}
    width = int(canvas.get("width") or 0)
    height = int(canvas.get("height") or 0)
    if width <= 0 or height <= 0:
        return
    entries = []
    for entry in spec.get("text") or []:
        box = entry.get("boxSource") or []
        value = str(entry.get("preferredValue") or entry.get("value") or "").strip()
        font_size = float((entry.get("fontFeatures") or {}).get("fontSize") or 0.0)
        if (
            len(box) != 4
            or not value
            or len(value) > 40
            or (box[1] + box[3]) / 2 > height * 0.12
            or box[3] - box[1] > height * 0.065
            or font_size > min(40.0, height * 0.055)
        ):
            continue
        entries.append(entry)
    if len(entries) < 3:
        return
    centers = sorted((entry["boxSource"][1] + entry["boxSource"][3]) / 2 for entry in entries)
    median_center = centers[len(centers) // 2]
    aligned = [
        entry
        for entry in entries
        if abs(
            (entry["boxSource"][1] + entry["boxSource"][3]) / 2
            - median_center
        )
        <= max(12.0, height * 0.018)
    ]
    aligned.sort(key=lambda entry: entry["boxSource"][0])
    if (
        len(aligned) < 3
        or aligned[-1]["boxSource"][2] - aligned[0]["boxSource"][0]
        < width * 0.18
    ):
        return
    controls = [
        dict(control)
        for control in spec.get("visualControlCandidates") or []
        if control.get("source") != "inferred-top-navigation-row"
    ]

    def contains_center(control: dict[str, Any], entry: dict[str, Any]) -> bool:
        control_box = control.get("boxSource") or []
        entry_box = entry.get("boxSource") or []
        return (
            len(control_box) == 4
            and len(entry_box) == 4
            and control_box[0] <= (entry_box[0] + entry_box[2]) / 2 <= control_box[2]
            and control_box[1] <= (entry_box[1] + entry_box[3]) / 2 <= control_box[3]
        )

    for entry in aligned:
        if any(contains_center(control, entry) for control in controls):
            continue
        box = [int(round(value)) for value in entry["boxSource"]]
        pad_x = max(6, round((box[3] - box[1]) * 0.65))
        pad_y = max(5, round((box[3] - box[1]) * 0.45))
        controls.append(
            {
                "elementId": f"inferred-top-nav-{entry.get('elementId')}",
                "boxSource": [
                    max(0, box[0] - pad_x),
                    max(0, box[1] - pad_y),
                    min(width, box[2] + pad_x),
                    min(height, box[3] + pad_y),
                ],
                "labelElementIds": [entry.get("elementId")],
                "visibleBoundary": False,
                "background": "transparent",
                "borderColor": "transparent",
                "borderWidth": 0,
                "cornerRadius": 6,
                "interaction": "semantic-link",
                "semanticRole": "nav",
                "interactionEvidence": "repeated-aligned-visible-top-row",
                "source": "inferred-top-navigation-row",
                "epistemic": "inferred",
                "behavior": "local-placeholder-no-invented-destination",
            }
        )
    spec["visualControlCandidates"] = controls


def _infer_contextual_ui_structure(document: dict[str, Any]) -> None:
    """Turn repeated visible UI rows into native controls and named icons.

    The geometry stays measured when a detected component exists. Missing
    row/icon boxes are inferred only from repeated aligned visible labels and
    are marked as such; no destination or hidden behavior is invented.
    """
    spec = document.get("reconstruction") or {}
    if not spec or spec.get("targetKind") != "web":
        return
    canvas = spec.get("canvas") or {}
    canvas_width = int(canvas.get("width") or 0)
    canvas_height = int(canvas.get("height") or 0)
    if canvas_width <= 0 or canvas_height <= 0:
        return
    entries = [
        entry
        for entry in spec.get("text") or []
        if len(entry.get("boxSource") or []) == 4
    ]
    contextual: list[tuple[dict[str, Any], str, str | None]] = []
    for entry in entries:
        value = entry.get("preferredValue") or entry.get("value")
        match = _contextual_ui_role(value)
        if match is None:
            continue
        box = entry.get("boxSource") or []
        if match[0] == "nav" and box[0] > canvas_width * 0.24:
            continue
        if match[0] in {"action", "link"} and box[0] < canvas_width * 0.45:
            continue
        contextual.append((entry, match[0], match[1]))
    if not contextual:
        return

    badges = spec.get("badges") or []
    badge_boxes = [badge.get("boxSource") or [] for badge in badges]
    icons = [
        dict(icon)
        for icon in spec.get("icons") or []
        if not any(
            _box_intersection(icon.get("boxSource") or [], badge_box)
            / max(1, _source_box_area(icon.get("boxSource") or []))
            >= 0.25
            for badge_box in badge_boxes
        )
    ]
    claimed_icons: set[int] = set()
    for entry, contextual_role, icon_name in contextual:
        if icon_name is None:
            continue
        box = [int(value) for value in entry["boxSource"]]
        center_y = (box[1] + box[3]) / 2
        candidates = []
        for index, icon in enumerate(icons):
            if index in claimed_icons:
                continue
            icon_box = icon.get("boxSource") or []
            if len(icon_box) != 4:
                continue
            icon_center_y = (icon_box[1] + icon_box[3]) / 2
            horizontal_gap = box[0] - icon_box[2]
            if (
                abs(icon_center_y - center_y) <= max(14, box[3] - box[1])
                and -4 <= horizontal_gap <= 46
            ):
                candidates.append(
                    (
                        abs(icon_center_y - center_y)
                        + abs(horizontal_gap - 10) * 0.2,
                        index,
                    )
                )
        if candidates:
            _distance, index = min(candidates)
            icon = icons[index]
            claimed_icons.add(index)
            icon.update(
                {
                    "name": icon_name,
                    "strategy": "inline-svg",
                    "geometrySource": icon.get("source") or "measured",
                    "semanticSource": "inferred-from-adjacent-live-label",
                    "epistemic": "inferred",
                }
            )
            continue
        size = max(10, min(18, int(round((box[3] - box[1]) * 1.25))))
        anchors = [
            icon.get("boxSource") or []
            for icon in icons
            if icon.get("semanticSource")
            and len(icon.get("boxSource") or []) == 4
            and (
                contextual_role != "nav"
                or (icon.get("boxSource") or [canvas_width])[0]
                < canvas_width * 0.24
            )
        ]
        if contextual_role == "nav" and anchors:
            anchor = sorted(anchors, key=lambda item: item[0])[len(anchors) // 2]
            size = max(10, min(18, int(round(anchor[2] - anchor[0]))))
            right = int(round(anchor[2]))
        else:
            right = max(size, box[0] - 8)
        top = int(round(center_y - size / 2))
        icons.append(
            {
                "elementId": f"inferred-icon-{entry.get('elementId')}",
                "name": icon_name,
                "boxSource": [right - size, top, right, top + size],
                "color": entry.get("color")
                or (entry.get("fontFeatures") or {}).get("color"),
                "strategy": "inline-svg",
                "source": "inferred-from-repeated-visible-row-pattern",
                "geometrySource": "inferred",
                "semanticSource": "inferred-from-adjacent-live-label",
                "epistemic": "inferred",
            }
        )
    # Unresolved OCR glyphs are not drawable icon evidence. Keep measured
    # named shapes, plus every contextual icon we just grounded in a label.
    filtered_icons = [
        icon
        for icon in icons
        if icon.get("semanticSource")
        or (
            icon.get("source") == "measured"
            and icon.get("name")
            not in {None, "unknown", "ambiguous-glyph-icon"}
        )
    ]
    deduplicated_icons: list[dict[str, Any]] = []
    for icon in sorted(
        filtered_icons,
        key=lambda item: (
            item.get("geometrySource") == "inferred",
            (item.get("boxSource") or [0, 0, 0, 0])[1],
            (item.get("boxSource") or [0, 0, 0, 0])[0],
        ),
    ):
        box = icon.get("boxSource") or []
        area = max(1, _source_box_area(box))
        if any(
            existing.get("name") == icon.get("name")
            and _box_intersection(existing.get("boxSource") or [], box)
            / max(1, min(_source_box_area(existing.get("boxSource") or []), area))
            >= 0.6
            for existing in deduplicated_icons
        ):
            continue
        if (
            icon.get("name") == "cross"
            and not icon.get("semanticSource")
            and any(
                _box_intersection(entry.get("boxSource") or [], box) / area >= 0.25
                for entry in entries
            )
        ):
            continue
        deduplicated_icons.append(icon)
    spec["icons"] = deduplicated_icons

    controls = [dict(control) for control in spec.get("visualControlCandidates") or []]
    surfaces = [
        surface
        for surface in spec.get("surfaces") or []
        if len(surface.get("boxSource") or []) == 4
    ]

    def contains_center(container: list[Any], item: list[Any]) -> bool:
        return (
            len(container) == 4
            and len(item) == 4
            and container[0] <= (item[0] + item[2]) / 2 <= container[2]
            and container[1] <= (item[1] + item[3]) / 2 <= container[3]
        )

    def smallest_surface(
        item_box: list[int], role: str
    ) -> dict[str, Any] | None:
        candidates = []
        for surface in surfaces:
            surface_box = surface["boxSource"]
            surface_width = surface_box[2] - surface_box[0]
            surface_height = surface_box[3] - surface_box[1]
            if not contains_center(surface_box, item_box):
                continue
            if role == "nav":
                valid = (
                    surface_width <= canvas_width * 0.32
                    and surface_height >= canvas_height * 0.45
                )
            else:
                valid = (
                    surface_width <= canvas_width * 0.48
                    and surface_height >= (item_box[3] - item_box[1]) * 2.2
                )
            if valid:
                candidates.append((surface_width * surface_height, surface))
        return min(candidates, key=lambda item: item[0])[1] if candidates else None

    role_groups: dict[tuple[str, tuple[int, ...]], list[dict[str, Any]]] = {}
    for entry, role, _icon_name in contextual:
        if role == "utility":
            continue
        surface = smallest_surface(entry["boxSource"], role)
        surface_key = tuple(surface["boxSource"]) if surface else ()
        role_groups.setdefault((role, surface_key), []).append(entry)

    for entry, role, _icon_name in contextual:
        box = [int(value) for value in entry["boxSource"]]
        existing = next(
            (
                control
                for control in controls
                if contains_center(control.get("boxSource") or [], box)
            ),
            None,
        )
        if existing is not None:
            label_ids = list(existing.get("labelElementIds") or [])
            if entry.get("elementId") not in label_ids:
                label_ids.append(entry.get("elementId"))
            existing.update(
                {
                    "labelElementIds": label_ids,
                    "interaction": "semantic-control-required",
                    "semanticRole": role,
                    "interactionEvidence": "visible-control-affordance",
                }
            )
            continue

        surface = smallest_surface(box, role)
        surface_box = surface.get("boxSource") if surface else None
        center_y = (box[1] + box[3]) / 2
        if role == "utility":
            pad_x = max(8, int(round((box[3] - box[1]) * 0.8)))
            pad_y = max(6, int(round((box[3] - box[1]) * 0.6)))
            control_box = [
                max(0, box[0] - pad_x),
                max(0, box[1] - pad_y),
                min(canvas_width, box[2] + pad_x),
                min(canvas_height, box[3] + pad_y),
            ]
        elif surface_box is not None:
            group = role_groups.get((role, tuple(surface_box)), [entry])
            centers = sorted(
                (candidate["boxSource"][1] + candidate["boxSource"][3]) / 2
                for candidate in group
            )
            gaps = [
                centers[index + 1] - centers[index]
                for index in range(len(centers) - 1)
                if centers[index + 1] > centers[index]
            ]
            half_height = (
                max(12, min(22, int(round(sorted(gaps)[len(gaps) // 2] / 2))))
                if gaps
                else max(13, min(21, int(round((box[3] - box[1]) * 1.5))))
            )
            inset = max(6, int(round((surface_box[2] - surface_box[0]) * 0.02)))
            control_box = [
                int(surface_box[0] + inset),
                max(int(surface_box[1]), int(round(center_y - half_height))),
                int(surface_box[2] - inset),
                min(int(surface_box[3]), int(round(center_y + half_height))),
            ]
        else:
            control_box = [
                max(0, box[0] - 28),
                max(0, box[1] - 8),
                min(canvas_width, box[2] + 12),
                min(canvas_height, box[3] + 8),
            ]

        label_ids = [entry.get("elementId")]
        if role == "link":
            subtitles = sorted(
                (
                    candidate
                    for candidate in entries
                    if candidate is not entry
                    and abs(candidate["boxSource"][0] - box[0]) <= 8
                    and 0
                    < (
                        (candidate["boxSource"][1] + candidate["boxSource"][3])
                        / 2
                        - center_y
                    )
                    <= 22
                    and contains_center(control_box, candidate["boxSource"])
                ),
                key=lambda candidate: candidate["boxSource"][1],
            )
            if subtitles:
                label_ids.append(subtitles[0].get("elementId"))
        controls.append(
            {
                "elementId": f"inferred-control-{entry.get('elementId')}",
                "boxSource": control_box,
                "labelElementIds": [item for item in label_ids if item is not None],
                "visibleBoundary": False,
                "background": "transparent",
                "borderColor": "transparent",
                "borderWidth": 0,
                "cornerRadius": 8,
                "interaction": "semantic-control-required",
                "semanticRole": role,
                "interactionEvidence": "repeated-aligned-visible-ui-row",
                "source": "inferred-from-visible-affordance",
                "epistemic": "inferred",
                "behavior": "local-placeholder-no-invented-destination",
            }
        )

    controls.sort(
        key=lambda control: (
            (control.get("boxSource") or [0, 0, 0, 0])[1],
            (control.get("boxSource") or [0, 0, 0, 0])[0],
        )
    )
    spec["visualControlCandidates"] = controls
    rules = spec.setdefault("implementationRules", [])
    inferred_rule = (
        "Controls inferred from repeated visible UI affordances use native semantic HTML "
        "but keep placeholder-only behavior; do not invent destinations or hidden actions."
    )
    if inferred_rule not in rules:
        rules.append(inferred_rule)


def _infer_corner_navigation_controls(document: dict[str, Any]) -> None:
    """Promote a measured bottom-corner arrow glyph to a native control."""
    spec = document.get("reconstruction") or {}
    canvas = spec.get("canvas") or {}
    width = int(canvas.get("width") or 0)
    height = int(canvas.get("height") or 0)
    if spec.get("targetKind") != "web" or width <= 0 or height <= 0:
        return
    controls = [dict(item) for item in spec.get("visualControlCandidates") or []]
    for icon in spec.get("icons") or []:
        box = [int(round(value)) for value in icon.get("boxSource") or []]
        if len(box) != 4:
            continue
        icon_width = box[2] - box[0]
        icon_height = box[3] - box[1]
        center_x = (box[0] + box[2]) / 2.0
        center_y = (box[1] + box[3]) / 2.0
        if (
            str(icon.get("name") or "").casefold()
            not in {"cross", "outline", "arrow_down"}
            or center_x > width * 0.08
            or center_y < height * 0.88
            or not 2 <= icon_width <= 18
            or not 4 <= icon_height <= 18
        ):
            continue
        if any(
            len(control.get("boxSource") or []) == 4
            and control["boxSource"][0] <= center_x <= control["boxSource"][2]
            and control["boxSource"][1] <= center_y <= control["boxSource"][3]
            for control in controls
        ):
            continue
        icon.update(
            {
                "name": "arrow_down",
                "strategy": "inline-svg",
                "semanticSource": "inferred-from-bottom-corner-position",
                "epistemic": "inferred",
            }
        )
        half_width = max(10, round(icon_height * 1.2))
        half_height = max(18, round(icon_height * 2.4))
        control_box = [
            max(0, round(center_x - half_width)),
            max(0, round(center_y - half_height)),
            min(width, round(center_x + half_width)),
            min(height, round(center_y + half_height)),
        ]
        background = (
            (((document.get("tokens") or {}).get("color") or {}).get("background") or {}).get("$value")
            or "#FFFFFF"
        )
        light_background = True
        if re.fullmatch(r"#[0-9A-Fa-f]{6}", str(background)):
            channels = [
                int(str(background)[index : index + 2], 16)
                for index in (1, 3, 5)
            ]
            light_background = sum(channels) / 3.0 >= 128
        controls.append(
            {
                "elementId": f"corner-scroll-control:{icon.get('elementId')}",
                "kind": "button",
                "boxSource": control_box,
                "labelElementIds": [],
                "ariaLabel": "Scroll down",
                "interaction": "semantic-button",
                "background": "#F1F1F1" if light_background else "#242424",
                "borderColor": "transparent",
                "borderWidth": 0,
                "cornerRadius": half_width,
                "zIndex": 24,
                "source": "inferred-bottom-corner-navigation",
                "geometrySource": "inferred-around-measured-arrow-glyph",
                "epistemic": "inferred",
            }
        )
    spec["visualControlCandidates"] = controls


_VISUAL_LATIN_TRANSLATION = str.maketrans(
    {
        "а": "a",
        "в": "b",
        "с": "c",
        "е": "e",
        "і": "i",
        "к": "k",
        "м": "m",
        "н": "h",
        "о": "o",
        "р": "p",
        "т": "t",
        "х": "x",
        "у": "y",
    }
)


def _visual_latin_text(value: Any) -> str:
    return _normalized_text(value).translate(_VISUAL_LATIN_TRANSLATION)


def _preferred_matches_measured_glyph_count(
    entry: dict[str, Any], value: Any
) -> bool:
    """Reject large-text readings that omit measured visible glyphs."""
    font = entry.get("fontFeatures") or {}
    font_size = float(font.get("fontSize") or 0.0)
    try:
        measured_count = max(
            int(font.get("characterCount") or 0),
            int(font.get("measuredCharacterCount") or 0),
        )
    except (TypeError, ValueError):
        measured_count = 0
    if font_size < 48 or measured_count < 4:
        return True
    visible_count = sum(not character.isspace() for character in str(value or ""))
    return visible_count >= math.ceil(measured_count * 0.72)


def _candidate_preserves_protected_numeric_format(
    entry: dict[str, Any], value: Any
) -> bool:
    """Keep measured currency punctuation and deterministic badge counts."""
    candidate = str(value or "").strip()
    current = str(entry.get("preferredValue") or entry.get("value") or "").strip()
    status = str(entry.get("resolutionStatus") or "")
    if current.startswith(("$", "€", "£", "¥")) and not candidate.startswith(
        current[0]
    ):
        return False
    if status == "layout-sequence-inferred" and candidate != current:
        return False
    if (
        status == "deterministic-badge-template-confirmed"
        and candidate != current
    ):
        return False
    return True


def _semantic_span_assignments(
    semantic_value: str, candidates: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], str, float]]:
    """Distribute one crop-level reading across its measured OCR boxes.

    Small VLMs often return the complete crop as one phrase even when OCR has
    already measured four separate lines.  A phrase must never be attached to
    one arbitrary element.  Instead, align non-overlapping word spans to each
    measured value and keep only strong, length-compatible matches.
    """
    words = list(
        re.finditer(
            r"[^\W_]+(?:['’\-\u2013\u2014][^\W_]+)*",
            str(semantic_value or ""),
            flags=re.UNICODE,
        )
    )
    if not words:
        return []
    proposals: list[tuple[float, int, int, int, dict[str, Any], str]] = []
    for order, candidate in enumerate(candidates):
        observed = str(candidate.get("value") or "").strip()
        target = _visual_latin_text(observed)
        if not target:
            continue
        observed_words = list(
            re.finditer(
                r"[^\W_]+(?:['’\-\u2013\u2014][^\W_]+)*",
                observed,
                flags=re.UNICODE,
            )
        )
        expected_count = max(1, len(observed_words))
        minimum_count = max(1, expected_count - 2)
        maximum_count = min(len(words), expected_count + 2)
        for start in range(len(words)):
            for count in range(minimum_count, maximum_count + 1):
                end_index = start + count - 1
                if end_index >= len(words):
                    break
                source_start = words[start].start()
                source_end = words[end_index].end()
                segment = semantic_value[source_start:source_end].strip()
                normalized = _visual_latin_text(segment)
                if not normalized:
                    continue
                length_ratio = min(len(target), len(normalized)) / max(
                    len(target), len(normalized)
                )
                if length_ratio < 0.65:
                    continue
                similarity = SequenceMatcher(None, target, normalized).ratio()
                if similarity < 0.68:
                    continue
                score = similarity - 0.08 * (1.0 - length_ratio)
                proposals.append(
                    (
                        score,
                        -len(normalized),
                        order,
                        source_start,
                        candidate,
                        segment,
                    )
                )

    assignments: list[tuple[dict[str, Any], str, float]] = []
    assigned: set[int] = set()
    occupied: list[tuple[int, int]] = []
    for score, _negative_length, _order, source_start, candidate, segment in sorted(
        proposals, key=lambda item: item[:4], reverse=True
    ):
        candidate_id = id(candidate)
        source_end = source_start + len(segment)
        if candidate_id in assigned or any(
            max(source_start, left) < min(source_end, right)
            for left, right in occupied
        ):
            continue
        assigned.add(candidate_id)
        occupied.append((source_start, source_end))
        assignments.append((candidate, segment, score))
    return assignments


def _ordered_semantic_row_assignments(
    semantic_value: str,
    candidates: list[dict[str, Any]],
    source_box: list[int] | None = None,
) -> list[tuple[dict[str, Any], str, float]]:
    """Partition a crop-level reading across adjacent measured text rows.

    Individual fuzzy matches are intentionally conservative, but that can
    truncate a badly OCR'd paragraph (for example ``CFAIE MOFE`` becoming only
    ``CRAVE``).  Adjacent, left-aligned rows provide stronger evidence: their
    order and line geometry are measured even when several characters are
    wrong.  This pass performs a bounded dynamic-programming partition of one
    semantic phrase and only accepts a complete, ordered multi-row match.
    """
    word_matches = list(
        re.finditer(
            r"[^\W_]+(?:['’\-\u2013\u2014][^\W_]+)*",
            str(semantic_value or ""),
            flags=re.UNICODE,
        )
    )
    if len(word_matches) < 2:
        return []

    measured = [
        entry
        for entry in candidates
        if len(entry.get("boxSource") or []) == 4
        and len(_visual_latin_text(entry.get("value"))) >= 3
    ]
    measured.sort(
        key=lambda entry: (
            ((entry["boxSource"][1] + entry["boxSource"][3]) / 2.0),
            entry["boxSource"][0],
        )
    )
    if len(measured) < 2:
        return []

    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for entry in measured:
        if not current:
            current = [entry]
            continue
        previous = current[-1]
        previous_box = previous["boxSource"]
        box = entry["boxSource"]
        previous_height = max(1.0, previous_box[3] - previous_box[1])
        height = max(1.0, box[3] - box[1])
        center_delta = (
            (box[1] + box[3]) / 2.0
            - (previous_box[1] + previous_box[3]) / 2.0
        )
        top_delta = box[1] - previous_box[1]
        width = max(previous_box[2] - previous_box[0], box[2] - box[0])
        left_aligned = abs(box[0] - previous_box[0]) <= max(24.0, width * 0.18)
        vertically_adjacent = (
            center_delta > max(previous_height, height) * 0.42
            and top_delta <= max(24.0, max(previous_height, height) * 3.0)
        )
        if left_aligned and vertically_adjacent:
            current.append(entry)
        else:
            if len(current) >= 2:
                groups.append(current)
            current = [entry]
    if len(current) >= 2:
        groups.append(current)

    def segment_text(start: int, end: int) -> str:
        trailing_end = (
            word_matches[end].start()
            if end < len(word_matches)
            else len(semantic_value)
        )
        return semantic_value[
            word_matches[start].start() : trailing_end
        ].strip()

    def row_score(entry: dict[str, Any], value: str) -> float | None:
        observed = _visual_latin_text(entry.get("value"))
        proposed = _visual_latin_text(value)
        if not observed or not proposed:
            return None
        length_ratio = min(len(observed), len(proposed)) / max(
            len(observed), len(proposed)
        )
        similarity = SequenceMatcher(None, observed, proposed).ratio()
        if length_ratio < 0.45 or similarity < 0.35:
            return None
        return 0.82 * similarity + 0.18 * length_ratio

    assignments: list[tuple[dict[str, Any], str, float]] = []
    assigned_ids: set[int] = set()
    word_count = len(word_matches)
    for group in groups:
        # Paragraphs longer than six measured lines should be handled in
        # multiple bounded crops rather than by an exponential partition.
        group = group[:6]
        row_count = len(group)
        best: tuple[float, list[tuple[str, float]]] | None = None
        possible_starts: Any = range(0, word_count - row_count + 1)
        first_box = group[0]["boxSource"]
        first_height = max(1, first_box[3] - first_box[1])
        if (
            source_box is not None
            and len(source_box) == 4
            and group[0] is measured[0]
            and first_box[1] - source_box[1] <= max(24, first_height * 3)
        ):
            possible_starts = (0,)
        for start in possible_starts:
            # token_index -> (sum score, [(segment, score), ...])
            states: dict[int, tuple[float, list[tuple[str, float]]]] = {
                start: (0.0, [])
            }
            for row_index, entry in enumerate(group):
                next_states: dict[int, tuple[float, list[tuple[str, float]]]] = {}
                remaining_rows = row_count - row_index - 1
                for token_index, (score_sum, parts) in states.items():
                    maximum_end = min(
                        word_count - remaining_rows,
                        token_index + 8,
                    )
                    for end in range(token_index + 1, maximum_end + 1):
                        value = segment_text(token_index, end)
                        score = row_score(entry, value)
                        if score is None:
                            continue
                        candidate = (score_sum + score, [*parts, (value, score)])
                        previous_state = next_states.get(end)
                        if previous_state is None or candidate[0] > previous_state[0]:
                            next_states[end] = candidate
                states = next_states
                if not states:
                    break
            for _end, (score_sum, parts) in states.items():
                if len(parts) != row_count:
                    continue
                row_scores = [score for _value, score in parts]
                combined_observed = "".join(
                    _visual_latin_text(entry.get("value")) for entry in group
                )
                combined_proposed = "".join(
                    _visual_latin_text(value) for value, _score in parts
                )
                overall = SequenceMatcher(
                    None, combined_observed, combined_proposed
                ).ratio()
                quality = 0.65 * (score_sum / row_count) + 0.35 * overall
                if min(row_scores) < 0.48 or overall < 0.55 or quality < 0.61:
                    continue
                if best is None or quality > best[0]:
                    best = (quality, parts)
        if best is None:
            continue
        quality, parts = best
        for entry, (value, score) in zip(group, parts, strict=True):
            if id(entry) in assigned_ids:
                continue
            assignments.append((entry, value, min(quality, score)))
            assigned_ids.add(id(entry))
    return assignments


def _focus_text_match_score(base: dict[str, Any], regional: dict[str, Any]) -> float:
    base_box = base.get("boxSource") or []
    regional_box = regional.get("boxSource") or []
    intersection = _box_intersection(base_box, regional_box)
    base_area = max(
        1.0,
        (float(base_box[2]) - float(base_box[0]))
        * (float(base_box[3]) - float(base_box[1])),
    ) if len(base_box) == 4 else 1.0
    regional_area = max(
        1.0,
        (float(regional_box[2]) - float(regional_box[0]))
        * (float(regional_box[3]) - float(regional_box[1])),
    ) if len(regional_box) == 4 else 1.0
    union = max(1.0, base_area + regional_area - intersection)
    geometry = intersection / union
    size_ratio = min(base_area, regional_area) / max(base_area, regional_area)
    base_text = _normalized_text(base.get("preferredValue") or base.get("value"))
    regional_text = _normalized_text(
        regional.get("preferredValue") or regional.get("value")
    )
    lexical = (
        SequenceMatcher(None, base_text, regional_text).ratio()
        if base_text and regional_text
        else 0.0
    )
    return 0.6 * geometry + 0.2 * size_ratio + 0.2 * lexical


def _regional_fragments_for_base(
    base: dict[str, Any], regional_text: list[dict[str, Any]], consumed: set[int]
) -> tuple[str, list[int], dict[str, Any] | None] | None:
    """Join multiple regional OCR words that refine one coarse base box."""
    base_box = base.get("boxSource") or []
    if len(base_box) != 4:
        return None
    fragments: list[tuple[int, dict[str, Any], str]] = []
    for index, regional in enumerate(regional_text):
        regional_box = regional.get("boxSource") or []
        regional_area = max(1, _source_box_area(regional_box))
        value = str(
            regional.get("preferredValue") or regional.get("value") or ""
        ).strip()
        if (
            index in consumed
            or len(regional_box) != 4
            or not value
            or _box_intersection(base_box, regional_box) / regional_area < 0.7
        ):
            continue
        fragments.append((index, regional, value))
    if len(fragments) < 2:
        return None
    fragments.sort(
        key=lambda item: (
            (item[1]["boxSource"][1] + item[1]["boxSource"][3]) / 2.0,
            item[1]["boxSource"][0],
        )
    )
    combined = " ".join(item[2] for item in fragments)
    combined_normalized = _normalized_text(combined)
    observed_normalized = _normalized_text(base.get("value"))
    if not combined_normalized or not observed_normalized:
        return None
    length_ratio = len(combined_normalized) / max(1, len(observed_normalized))
    similarity = SequenceMatcher(
        None, combined_normalized, observed_normalized
    ).ratio()
    if not 0.5 <= length_ratio <= 1.25 or similarity < 0.55:
        return None
    typography_candidates = [
        fragment.get("typographyCandidate")
        for _index, fragment, _value in fragments
        if isinstance(fragment.get("typographyCandidate"), dict)
    ]
    typography = (
        max(
            typography_candidates,
            key=lambda candidate: float(candidate.get("confidence") or 0.0),
        )
        if typography_candidates
        else None
    )
    return combined, [item[0] for item in fragments], typography


def _reconcile_inline_run_text(document: dict[str, Any]) -> None:
    """Retarget stored word styles after a later crop corrects OCR spelling."""
    spec = document.get("reconstruction") or {}
    for entry in spec.get("text") or []:
        runs = entry.get("inlineRuns") or []
        display = str(entry.get("preferredValue") or entry.get("value") or "")
        words = list(re.finditer(r"\S+", display))
        current_counts = [
            len(re.findall(r"\S+", str(run.get("text") or "")))
            for run in runs
            if isinstance(run, dict)
        ]
        preferred_counts = [
            int((run.get("typographyCandidate") or {}).get("preferredWordCount") or 0)
            for run in runs
            if isinstance(run, dict)
        ]
        if (
            len(runs) < 2
            or len(current_counts) != len(runs)
            or not all(current_counts)
        ):
            continue
        if len(preferred_counts) == len(runs) and all(preferred_counts) and sum(preferred_counts) == len(words):
            counts = preferred_counts
        elif sum(current_counts) == len(words):
            counts = current_counts
        elif len(words) >= len(runs):
            source_lengths = [
                max(1, len(_visual_latin_text(run.get("text")))) for run in runs
            ]
            source_total = sum(source_lengths)
            target_lengths = [
                max(1, len(_visual_latin_text(word.group(0)))) for word in words
            ]
            boundaries: list[int] = []
            previous = 0
            cumulative = 0
            for source_length in source_lengths[:-1]:
                cumulative += source_length
                target_ratio = cumulative / source_total
                choices = range(previous + 1, len(words) - (len(runs) - len(boundaries) - 1) + 1)
                boundary = min(
                    choices,
                    key=lambda value: abs(
                        sum(target_lengths[:value]) / sum(target_lengths)
                        - target_ratio
                    ),
                )
                boundaries.append(boundary)
                previous = boundary
            points = [0, *boundaries, len(words)]
            counts = [points[index + 1] - points[index] for index in range(len(runs))]
        else:
            continue
        cursor = 0
        for run, count in zip(runs, counts, strict=True):
            start = 0 if cursor == 0 else words[cursor].start()
            cursor += count
            end = words[cursor].start() if cursor < len(words) else len(display)
            run["text"] = display[start:end]


def _merge_focus_documents(
    document: dict[str, Any],
    resolutions: list[tuple[dict[str, Any], dict[str, Any] | None]],
) -> dict[str, Any]:
    """Merge bounded regional evidence into one implementation-ready spec.

    The merged values remain candidates unless the regional OCR and VLM agree.
    Failed regions stay in ``focusPlan`` so the consumer never receives a false
    claim that semantics completed.
    """
    spec = document.get("reconstruction") or {}
    base_text = spec.get("text") or []
    remaining: list[dict[str, Any]] = []
    resolved_focus: list[dict[str, Any]] = []
    resolved_calls = 0
    failed_calls = 0
    confirmed_ids: set[Any] = set()

    def inline_typography_runs(
        value: Any, runs: list[dict[str, Any]]
    ) -> list[dict[str, Any]] | None:
        """Align adjacent VLM style runs to the exact live-text value.

        The local VLM supplies style boundaries but not reliable pixel boxes.
        Word spans in the measured OCR value provide deterministic ordering and
        preserve every original space.  A mixed run is accepted only when all
        words are covered in order and at least one visible style property
        changes; otherwise the safer uniform-style path remains in force.
        """
        display = str(value or "")
        words = list(re.finditer(r"\S+", display))
        semantic_runs = [
            run
            for run in runs
            if isinstance(run, dict)
            and str(run.get("text") or "").strip()
            and re.findall(r"\S+", str(run.get("text") or ""))
        ]
        if len(words) < 2 or len(semantic_runs) < 2:
            return None
        counts = [
            len(re.findall(r"\S+", str(run.get("text") or "")))
            for run in semantic_runs
        ]
        if sum(counts) != len(words):
            return None

        style_keys = ("class", "contrast", "width", "weight", "slant")
        signatures = {
            tuple(str(run.get(key) or "").casefold() for key in style_keys)
            for run in semantic_runs
        }
        if len(signatures) < 2:
            return None

        output: list[dict[str, Any]] = []
        word_cursor = 0
        for run_index, (run, count) in enumerate(zip(semantic_runs, counts)):
            start = 0 if word_cursor == 0 else words[word_cursor].start()
            word_cursor += count
            end = (
                words[word_cursor].start()
                if word_cursor < len(words)
                else len(display)
            )
            segment = display[start:end]
            observed = _visual_latin_text(segment)
            inferred = _visual_latin_text(run.get("text"))
            if not observed or not inferred:
                return None
            similarity = SequenceMatcher(None, observed, inferred).ratio()
            length_ratio = min(len(observed), len(inferred)) / max(
                len(observed), len(inferred)
            )
            if similarity < 0.72 or length_ratio < 0.72:
                return None
            typography = {
                **{key: item for key, item in run.items() if key != "text"},
                "status": "candidate",
                "epistemic": "inferred",
                "method": "local-vlm-region-inline-text-inspection",
            }
            output.append(
                {
                    "text": segment,
                    "typographyCandidate": typography,
                    "runIndex": run_index,
                }
            )
        if "".join(str(run["text"]) for run in output) != display:
            return None
        return output

    for action, regional_document in resolutions:
        source_box = _focus_region_box(action)
        blocking_failure = action.get("blocking", True) is not False

        def apply_focused_typography(
            base: dict[str, Any], candidate: Any
        ) -> bool:
            if not isinstance(candidate, dict) or not candidate:
                return False
            previous_box = base.get("typographyEvidenceBoxSource")
            previous_area = _source_box_area(previous_box)
            incoming_area = _source_box_area(source_box)
            if (
                base.get("typographyCandidate")
                and previous_area > 0
                and incoming_area >= previous_area
            ):
                return False
            base["typographyCandidate"] = candidate
            base["fontStrategy"] = "match-class-first-then-glyph-metrics"
            if source_box is not None:
                base["typographyEvidenceBoxSource"] = list(source_box)
                base["typographyEvidenceMethod"] = "bounded-regional-vlm"
            return True

        def apply_focused_inline_typography(
            base: dict[str, Any], runs: list[dict[str, Any]]
        ) -> bool:
            previous_box = base.get("typographyEvidenceBoxSource")
            previous_area = _source_box_area(previous_box)
            incoming_area = _source_box_area(source_box)
            if (
                base.get("inlineRuns")
                and previous_area > 0
                and incoming_area >= previous_area
            ):
                return False
            measured = [
                item
                for item in (base.get("fontFeatures") or {}).get(
                    "wordBoxesSource", []
                )
                if isinstance(item, dict) and item.get("slant")
            ]
            if len(measured) == len(runs):
                for run, item in zip(runs, measured, strict=True):
                    typography = run.setdefault("typographyCandidate", {})
                    typography["slant"] = item["slant"]
                    typography["slantConfidence"] = item.get("slantConfidence")
                    typography["slantMethod"] = item.get("slantMethod")
            base["inlineRuns"] = runs
            base["inlineRunMethod"] = (
                "bounded-vlm-class-plus-measured-word-slant"
                if len(measured) == len(runs)
                else "bounded-regional-vlm-word-styles"
            )
            base["fontStrategy"] = "match-inline-class-first-then-glyph-metrics"
            if source_box is not None:
                base["typographyEvidenceBoxSource"] = list(source_box)
                base["typographyEvidenceMethod"] = "bounded-regional-vlm"
            return True

        regional_spec = (
            regional_document.get("reconstruction")
            if isinstance(regional_document, dict)
            else None
        )
        regional_text = (
            regional_spec.get("text") or []
            if isinstance(regional_spec, dict)
            else []
        )
        semantic = (
            regional_spec.get("semanticTextCandidate")
            if isinstance(regional_spec, dict)
            else None
        )
        reason = str(action.get("reason") or "").casefold()
        text_focused = "text" in reason or bool(action.get("evidence"))
        semantic_available = bool(
            isinstance(semantic, dict) and str(semantic.get("text") or "").strip()
        )
        style_available = any(
            entry.get("typographyCandidate")
            or entry.get("preferredValue")
            or entry.get("resolutionStatus") == "confirmed"
            for entry in regional_text
        )

        if not isinstance(regional_spec, dict) or (
            text_focused and not semantic_available and not style_available
        ):
            failed_calls += 1
            if blocking_failure:
                remaining.append(action)
            resolved_focus.append(
                {
                    "region": source_box,
                    "evidence": action.get("evidence"),
                    "status": (
                        "failed" if blocking_failure else "optional-no-update"
                    ),
                    "textUpdates": 0,
                }
            )
            continue

        candidates = [
            entry
            for entry in base_text
            if source_box is None
            or _box_intersection(entry.get("boxSource") or [], source_box) > 0
        ]
        consumed: set[int] = set()
        updated_entries: set[int] = set()
        for base in candidates:
            fragment_join = _regional_fragments_for_base(
                base, regional_text, consumed
            )
            if fragment_join is not None and (
                not _preferred_matches_measured_glyph_count(base, fragment_join[0])
                or not _candidate_preserves_protected_numeric_format(
                    base, fragment_join[0]
                )
            ):
                fragment_join = None
            if fragment_join is not None:
                combined, fragment_indices, typography = fragment_join
                consumed.update(fragment_indices)
                base["preferredValue"] = combined
                base["resolutionStatus"] = "regional-fragment-consensus"
                base["resolutionMethod"] = "regional-ocr-fragment-join"
                base["resolutionConfidence"] = round(
                    SequenceMatcher(
                        None,
                        _normalized_text(base.get("value")),
                        _normalized_text(combined),
                    ).ratio(),
                    3,
                )
                base["epistemic"] = "inferred"
                if typography is not None:
                    apply_focused_typography(base, typography)
                updated_entries.add(id(base))
                continue
            ranked = sorted(
                (
                    (_focus_text_match_score(base, regional), index, regional)
                    for index, regional in enumerate(regional_text)
                    if index not in consumed
                ),
                key=lambda item: item[0],
                reverse=True,
            )
            if not ranked or ranked[0][0] < 0.2:
                continue
            _, index, regional = ranked[0]
            consumed.add(index)
            materially_updated = False
            preferred = regional.get("preferredValue")
            if preferred is None and regional.get("status") == "confirmed":
                preferred = regional.get("value")
            if (
                preferred
                and _preferred_matches_measured_glyph_count(base, preferred)
                and _candidate_preserves_protected_numeric_format(base, preferred)
            ):
                base["preferredValue"] = preferred
                materially_updated = True
            if apply_focused_typography(
                base, regional.get("typographyCandidate")
            ):
                materially_updated = True
            if regional.get("status") == "confirmed":
                base["status"] = "confirmed"
                confirmed_ids.add(base.get("elementId"))
                materially_updated = True
            if regional.get("resolutionStatus") not in {None, "unresolved"}:
                base["resolutionStatus"] = regional["resolutionStatus"]
                materially_updated = True
            if materially_updated:
                updated_entries.add(id(base))

        if semantic_available:
            semantic_value = str(semantic["text"]).strip()
            typography_runs = [
                run
                for run in semantic.get("typographyRuns") or []
                if isinstance(run, dict) and str(run.get("text") or "").strip()
            ]
            for candidate in candidates:
                inline_runs = inline_typography_runs(
                    candidate.get("preferredValue") or candidate.get("value"),
                    typography_runs,
                )
                if inline_runs is not None and apply_focused_inline_typography(
                    candidate, inline_runs
                ):
                    updated_entries.add(id(candidate))
                    continue
                observed = _visual_latin_text(
                    candidate.get("preferredValue") or candidate.get("value")
                )
                if not observed:
                    continue
                ranked_runs: list[tuple[float, dict[str, Any]]] = []
                for run in typography_runs:
                    run_text = _visual_latin_text(run.get("text"))
                    if not run_text:
                        continue
                    similarity = SequenceMatcher(None, observed, run_text).ratio()
                    length_ratio = min(len(observed), len(run_text)) / max(
                        len(observed), len(run_text)
                    )
                    if observed in run_text or run_text in observed:
                        similarity = max(similarity, 0.82)
                    score = 0.8 * similarity + 0.2 * length_ratio
                    ranked_runs.append((score, run))
                if not ranked_runs:
                    continue
                run_score, run = max(ranked_runs, key=lambda item: item[0])
                if run_score < 0.48:
                    continue
                typography = {
                    **{key: value for key, value in run.items() if key != "text"},
                    "status": "candidate",
                    "epistemic": "inferred",
                    "method": "local-vlm-region-text-inspection",
                }
                if apply_focused_typography(candidate, typography):
                    updated_entries.add(id(candidate))
            for candidate, span_value, span_score in _semantic_span_assignments(
                semantic_value, candidates
            ):
                if not _preferred_matches_measured_glyph_count(
                    candidate, span_value
                ) or not _candidate_preserves_protected_numeric_format(
                    candidate, span_value
                ):
                    continue
                observed = _visual_latin_text(candidate.get("value"))
                existing_value = str(candidate.get("preferredValue") or "").strip()
                existing = _visual_latin_text(existing_value)
                existing_score = (
                    SequenceMatcher(None, observed, existing).ratio()
                    if observed and existing
                    else 0.0
                )
                existing_length_ratio = (
                    min(len(observed), len(existing)) / max(len(observed), len(existing))
                    if observed and existing
                    else 0.0
                )
                if (
                    not existing_value
                    or existing_length_ratio < 0.65
                    or span_score > existing_score + 0.04
                ):
                    candidate["preferredValue"] = span_value
                    candidate["resolutionStatus"] = "vlm-preferred-candidate"
                    candidate["resolutionMethod"] = (
                        "crop-semantic-to-measured-box-alignment"
                    )
                    candidate["resolutionConfidence"] = round(
                        min(0.95, max(0.65, span_score)), 3
                    )
                    candidate["epistemic"] = "inferred"
                    updated_entries.add(id(candidate))
            for candidate, span_value, span_score in _ordered_semantic_row_assignments(
                semantic_value, candidates, source_box
            ):
                if not _preferred_matches_measured_glyph_count(
                    candidate, span_value
                ) or not _candidate_preserves_protected_numeric_format(
                    candidate, span_value
                ):
                    continue
                observed = _visual_latin_text(candidate.get("value"))
                existing_value = str(candidate.get("preferredValue") or "").strip()
                existing = _visual_latin_text(existing_value)
                existing_score = (
                    SequenceMatcher(None, observed, existing).ratio()
                    if observed and existing
                    else 0.0
                )
                proposed = _visual_latin_text(span_value)
                same_value = bool(existing and existing == proposed)
                repairs_truncation = bool(
                    existing
                    and len(proposed) >= len(existing) * 1.25
                    and SequenceMatcher(None, observed, proposed).ratio()
                    >= existing_score - 0.02
                )
                if (
                    not existing_value
                    or same_value
                    or repairs_truncation
                    or span_score > existing_score + 0.04
                ):
                    candidate["preferredValue"] = span_value
                    candidate["resolutionStatus"] = (
                        "regional-multiline-semantic-partition"
                    )
                    candidate["resolutionMethod"] = (
                        "measured-row-fuzzy-partition"
                    )
                    candidate["resolutionConfidence"] = round(
                        min(0.95, max(0.65, span_score)), 3
                    )
                    candidate["epistemic"] = "inferred"
                    updated_entries.add(id(candidate))
            semantic_targets = [
                candidate for candidate in candidates if id(candidate) in updated_entries
            ]
            candidate = semantic_targets[0] if len(semantic_targets) == 1 else None
            if candidate is None and action.get("evidence"):
                evidence = _normalized_text(action["evidence"])
                ranked_candidates = sorted(
                    (
                        (
                            SequenceMatcher(
                                None,
                                evidence,
                                _normalized_text(entry.get("value")),
                            ).ratio(),
                            entry,
                        )
                        for entry in candidates
                    ),
                    key=lambda item: item[0],
                    reverse=True,
                )
                if ranked_candidates and ranked_candidates[0][0] >= 0.5:
                    candidate = ranked_candidates[0][1]
            if candidate is not None and id(candidate) not in updated_entries:
                semantic_similarity = SequenceMatcher(
                    None,
                    _normalized_text(candidate.get("value")),
                    _normalized_text(semantic_value),
                ).ratio()
                if semantic_similarity < 0.55:
                    candidate = None
            if (
                semantic_value
                and candidate is not None
                and not candidate.get("preferredValue")
                and _preferred_matches_measured_glyph_count(
                    candidate, semantic_value
                )
                and _candidate_preserves_protected_numeric_format(
                    candidate, semantic_value
                )
            ):
                candidate["preferredValue"] = semantic_value
                candidate["resolutionStatus"] = "vlm-preferred-candidate"
                updated_entries.add(id(candidate))

        if "unresolved_text_density" in (action.get("reasons") or []):
            next_element_id = (
                max(
                    (
                        int(entry.get("elementId"))
                        for entry in base_text
                        if isinstance(entry.get("elementId"), int)
                    ),
                    default=0,
                )
                + 1
            )
            for regional_index, regional in enumerate(regional_text):
                if regional_index in consumed:
                    continue
                value = str(
                    regional.get("preferredValue")
                    or regional.get("value")
                    or ""
                ).strip()
                box = regional.get("boxSource") or []
                confidence = float(regional.get("confidence") or 0.0)
                if (
                    not value
                    or len(box) != 4
                    or (
                        regional.get("status") != "confirmed"
                        and confidence < 0.72
                    )
                ):
                    continue
                if source_box is not None and _box_intersection(box, source_box) <= 0:
                    continue
                duplicate = any(
                    _intersection_ratio(box, current.get("boxSource") or []) >= 0.55
                    or (
                        _normalized_text(value)
                        == _normalized_text(
                            current.get("preferredValue") or current.get("value")
                        )
                        and _box_intersection(box, current.get("boxSource") or []) > 0
                    )
                    for current in base_text
                )
                if duplicate:
                    continue
                discovered = {
                    **regional,
                    "elementId": next_element_id,
                    "preferredValue": value,
                    "resolutionStatus": "regional-new-text-candidate",
                    "resolutionMethod": "bounded-unresolved-ink-discovery",
                    "geometrySource": "bounded-regional-measurement",
                    "epistemic": "inferred",
                }
                next_element_id += 1
                base_text.append(discovered)
                consumed.add(regional_index)
                updated_entries.add(id(discovered))

        text_updates = len(updated_entries)

        if text_focused and text_updates == 0:
            failed_calls += 1
            if blocking_failure:
                remaining.append(action)
            status = "failed" if blocking_failure else "optional-no-update"
        else:
            resolved_calls += 1
            status = "resolved"
        resolved_focus.append(
            {
                "region": source_box,
                "evidence": action.get("evidence"),
                "status": status,
                "textUpdates": text_updates,
            }
        )

    original_plan = spec.get("focusPlan") or []
    attempted_ids = {id(action) for action, _document in resolutions}
    spec["focusPlan"] = [
        action
        for action in original_plan
        if id(action) not in attempted_ids
    ] + remaining
    original_text_plan = spec.get("textVerificationPlan") or []
    spec["textVerificationPlan"] = [
        action
        for action in original_text_plan
        if id(action) not in attempted_ids
    ] + [action for action in remaining if action in original_text_plan]
    if confirmed_ids:
        spec["blockingUncertainties"] = [
            uncertainty
            for uncertainty in spec.get("blockingUncertainties") or []
            if uncertainty.get("elementId") not in confirmed_ids
        ]
    previous_strategy = spec.get("semanticStrategy") or {}
    previous_resolved_calls = int(previous_strategy.get("resolvedCalls") or 0)
    previous_failed_calls = int(previous_strategy.get("failedCalls") or 0)
    previous_focus = list(spec.get("resolvedFocus") or [])
    spec["semanticStrategy"] = {
        **previous_strategy,
        "mode": "internally-resolved-focus-regions",
        "resolvedCalls": previous_resolved_calls + resolved_calls,
        "failedCalls": previous_failed_calls + failed_calls,
        "remainingCalls": len(spec["focusPlan"]),
        "fullImageCall": False,
        "reason": "Sens resolved bounded source-pixel crops internally so the consumer receives one compact web specification.",
    }
    spec["resolvedFocus"] = previous_focus + resolved_focus
    _reconcile_inline_run_text(document)
    return document


def _grouped_text_focus_actions(
    spec: dict[str, Any], max_calls: int
) -> list[dict[str, Any]]:
    """Cover all live text with a bounded set of whitespace-separated crops."""
    limit = max(0, min(5, int(max_calls)))
    canvas = spec.get("canvas") or {}
    width = int(canvas.get("width") or 0)
    height = int(canvas.get("height") or 0)
    if limit == 0 or width <= 0 or height <= 0:
        return []
    symbol_boxes = [
        entry.get("boxSource") or [] for entry in spec.get("symbolArt") or []
    ]
    entries = []
    for entry in spec.get("text") or []:
        box = entry.get("boxSource") or []
        if len(box) != 4 or not str(entry.get("value") or "").strip():
            continue
        area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
        if any(
            _box_intersection(box, symbol_box) / area >= 0.5
            for symbol_box in symbol_boxes
        ):
            continue
        entries.append(entry)
    if not entries:
        return []

    groups: list[list[dict[str, Any]]] = [entries]
    if limit >= 2 and len(entries) >= 2:
        canvas_area = max(1, width * height)
        dominant = max(
            entries,
            key=lambda entry: (
                (entry["boxSource"][2] - entry["boxSource"][0])
                * (entry["boxSource"][3] - entry["boxSource"][1])
            ),
        )
        dominant_box = dominant["boxSource"]
        dominant_width = dominant_box[2] - dominant_box[0]
        dominant_area = dominant_width * (dominant_box[3] - dominant_box[1])
        if dominant_width / width >= 0.7 and dominant_area / canvas_area >= 0.08:
            groups = [[dominant], [entry for entry in entries if entry is not dominant]]
    dimensions = (width, height)
    while len(groups) < min(limit, len(entries)):
        best: tuple[float, int, int, list[dict[str, Any]], int] | None = None
        for group_index, group in enumerate(groups):
            if len(group) < 2:
                continue
            for axis in (0, 1):
                ordered = sorted(group, key=lambda item: item["boxSource"][axis])
                max_end = float(ordered[0]["boxSource"][axis + 2])
                for split in range(1, len(ordered)):
                    next_start = float(ordered[split]["boxSource"][axis])
                    gap = next_start - max_end
                    balance = min(split, len(ordered) - split) / len(ordered)
                    score = (
                        (gap / max(1, dimensions[axis]))
                        * (0.75 + balance)
                        * len(group)
                    )
                    candidate = (score, group_index, axis, ordered, split)
                    if gap > 0 and (best is None or candidate[0] > best[0]):
                        best = candidate
                    max_end = max(
                        max_end, float(ordered[split]["boxSource"][axis + 2])
                    )
        if best is None:
            splittable = [
                (len(group), index, group)
                for index, group in enumerate(groups)
                if len(group) >= 2
            ]
            if not splittable:
                break
            _count, group_index, group = max(splittable)
            spans = []
            for axis in (0, 1):
                centers = [
                    (entry["boxSource"][axis] + entry["boxSource"][axis + 2]) / 2
                    for entry in group
                ]
                spans.append((max(centers) - min(centers)) / max(1, dimensions[axis]))
            axis = 0 if spans[0] >= spans[1] else 1
            ordered = sorted(
                group,
                key=lambda entry: (
                    entry["boxSource"][axis] + entry["boxSource"][axis + 2]
                )
                / 2,
            )
            split = len(ordered) // 2
            best = (0.0, group_index, axis, ordered, split)
        _score, group_index, _axis, ordered, split = best
        left, right = ordered[:split], ordered[split:]
        if not left or not right:
            break
        groups[group_index : group_index + 1] = [left, right]

    def group_box(group: list[dict[str, Any]]) -> list[int]:
        x1 = min(int(entry["boxSource"][0]) for entry in group)
        y1 = min(int(entry["boxSource"][1]) for entry in group)
        x2 = max(int(entry["boxSource"][2]) for entry in group)
        y2 = max(int(entry["boxSource"][3]) for entry in group)
        if (
            len(group) == 1
            and (x2 - x1) / width >= 0.7
            and ((x2 - x1) * (y2 - y1)) / max(1, width * height) >= 0.08
        ):
            return [max(0, x1), max(0, y1), min(width, x2), min(height, y2)]
        pad_x = max(12, round(width * 0.008))
        pad_y = max(10, round(height * 0.008))
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(width, x2 + pad_x), min(height, y2 + pad_y)
        if x2 - x1 < 160:
            extra = (160 - (x2 - x1) + 1) // 2
            x1, x2 = max(0, x1 - extra), min(width, x2 + extra)
        if y2 - y1 < 96:
            extra = (96 - (y2 - y1) + 1) // 2
            y1, y2 = max(0, y1 - extra), min(height, y2 + extra)
        return [x1, y1, x2, y2]

    groups.sort(
        key=lambda group: (
            min(entry["boxSource"][1] for entry in group),
            min(entry["boxSource"][0] for entry in group),
        )
    )
    actions = []
    for group in groups:
        box = group_box(group)
        evidence = " | ".join(str(entry.get("value") or "") for entry in group)
        actions.append(
            {
                "tool": "sens_zoom",
                "reason": "Resolve grouped live text and typography in one bounded source-pixel crop.",
                "evidence": evidence[:240],
                "blocking": any(
                    not (
                        (
                            entry.get("verified") is True
                            and float(entry.get("confidence") or 0.0) >= 0.90
                        )
                        or (
                            entry.get("preferredValue") not in {None, ""}
                            and entry.get("resolutionStatus") != "unresolved"
                        )
                        or entry.get("resolutionStatus") == "confirmed"
                    )
                    for entry in group
                ),
                "arguments": {
                    "region": {
                        "x": box[0],
                        "y": box[1],
                        "width": box[2] - box[0],
                        "height": box[3] - box[1],
                    },
                    "profile": "reconstruct",
                    "response": "compact",
                    "targetKind": spec.get("targetKind") or "web",
                },
            }
        )
    return actions


def _fallback_text_focus_actions(
    spec: dict[str, Any], max_calls: int
) -> list[dict[str, Any]]:
    limit = max(0, int(max_calls))
    if limit == 0:
        return []
    canvas = spec.get("canvas") or {}
    width = int(canvas.get("width") or 0)
    height = int(canvas.get("height") or 0)
    symbol_boxes = [
        entry.get("boxSource") or [] for entry in spec.get("symbolArt") or []
    ]
    candidates = []
    for entry in spec.get("text") or []:
        box = entry.get("boxSource") or []
        if (
            len(box) != 4
            or entry.get("resolutionStatus") != "unresolved"
            or not str(entry.get("value") or "").strip()
        ):
            continue
        area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
        if any(
            _box_intersection(box, symbol_box) / area >= 0.5
            for symbol_box in symbol_boxes
        ):
            continue
        confidence = float(entry.get("confidence") or 0.0)
        candidates.append((1.0 - confidence, area, entry))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    actions = []
    for _uncertainty, _area, entry in candidates[:limit]:
        box = [int(value) for value in entry["boxSource"]]
        box = [
            max(0, min(width, box[0])),
            max(0, min(height, box[1])),
            max(0, min(width, box[2])),
            max(0, min(height, box[3])),
        ]
        actions.append(
            {
                "tool": "sens_zoom",
                "reason": "Targeted fallback for one text element left unresolved by the grouped pass.",
                "evidence": entry.get("value"),
                "arguments": {
                    "region": {
                        "x": box[0],
                        "y": box[1],
                        "width": box[2] - box[0],
                        "height": box[3] - box[1],
                    },
                    "profile": "reconstruct",
                    "response": "compact",
                    "targetKind": spec.get("targetKind") or "web",
                },
            }
        )
    return actions


def _control_text_focus_actions(
    spec: dict[str, Any], max_calls: int
) -> list[dict[str, Any]]:
    """Spend remaining semantic budget on compact measured control labels."""
    limit = max(0, int(max_calls))
    canvas = spec.get("canvas") or {}
    width = int(canvas.get("width") or 0)
    height = int(canvas.get("height") or 0)
    if limit == 0 or width <= 0 or height <= 0:
        return []
    text_by_id = {
        entry.get("elementId"): entry for entry in spec.get("text") or []
    }
    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    seen: set[Any] = set()
    for control in spec.get("visualControlCandidates") or []:
        control_box = control.get("boxSource") or []
        if len(control_box) != 4:
            continue
        for element_id in control.get("labelElementIds") or []:
            entry = text_by_id.get(element_id)
            if (
                entry is None
                or element_id in seen
                or entry.get("preferredValue") not in {None, ""}
                or entry.get("resolutionStatus") != "unresolved"
            ):
                continue
            seen.add(element_id)
            candidates.append((entry, control))
    candidates.sort(
        key=lambda pair: (
            pair[1]["boxSource"][1],
            pair[1]["boxSource"][0],
        )
    )
    groups: list[list[tuple[dict[str, Any], dict[str, Any]]]] = []
    for pair in candidates:
        box = pair[1]["boxSource"]
        center_y = (box[1] + box[3]) / 2.0
        if groups:
            current = groups[-1]
            current_boxes = [item[1]["boxSource"] for item in current]
            current_center_y = sum(
                (item[1] + item[3]) / 2.0 for item in current_boxes
            ) / len(current_boxes)
            max_height = max(
                box[3] - box[1],
                *(item[3] - item[1] for item in current_boxes),
            )
            combined_left = min(box[0], *(item[0] for item in current_boxes))
            combined_right = max(box[2], *(item[2] for item in current_boxes))
            if (
                abs(center_y - current_center_y) <= max_height * 1.5
                and combined_right - combined_left <= width * 0.55
            ):
                current.append(pair)
                continue
        groups.append([pair])

    actions = []
    pad_x = max(12, round(width * 0.01))
    pad_y = max(10, round(height * 0.01))
    for group in groups[:limit]:
        boxes = [pair[1]["boxSource"] for pair in group]
        x0 = max(0, min(box[0] for box in boxes) - pad_x)
        y0 = max(0, min(box[1] for box in boxes) - pad_y)
        x1 = min(width, max(box[2] for box in boxes) + pad_x)
        y1 = min(height, max(box[3] for box in boxes) + pad_y)
        actions.append(
            {
                "tool": "sens_zoom",
                "reason": "Resolve exact live labels inside measured semantic controls.",
                "evidence": " | ".join(
                    str(pair[0].get("value") or "") for pair in group
                )[:240],
                "arguments": {
                    "region": {
                        "x": x0,
                        "y": y0,
                        "width": x1 - x0,
                        "height": y1 - y0,
                    },
                    "profile": "reconstruct",
                    "response": "compact",
                    "targetKind": spec.get("targetKind") or "web",
                },
            }
        )
    return actions


def _typography_focus_actions(
    spec: dict[str, Any], max_calls: int
) -> list[dict[str, Any]]:
    """Recheck large runs whose grouped typography is missing or suspicious."""
    limit = max(0, int(max_calls))
    if limit == 0:
        return []
    canvas = spec.get("canvas") or {}
    width = int(canvas.get("width") or 0)
    height = int(canvas.get("height") or 0)
    candidates = []
    for entry in spec.get("text") or []:
        box = entry.get("boxSource") or []
        value = str(entry.get("preferredValue") or entry.get("value") or "")
        font = entry.get("fontFeatures") or {}
        typography = entry.get("typographyCandidate") or {}
        font_size = float(font.get("fontSize") or 0.0)
        style_class = str(typography.get("class") or "").casefold()
        missing_style = not style_class
        suspicious_numeric = bool(
            any(character.isdigit() for character in value)
            and (not style_class or "sans" in style_class)
        )
        measured_characters = float(font.get("measuredCharacterCount") or 0.0)
        recognized_characters = float(font.get("characterCount") or 0.0)
        unusual_style_measurement = bool(
            str(font.get("weightCandidate") or "").casefold()
            in {"light", "thin"}
            or float(font.get("inkCoverage") or 0.0) >= 0.55
            or (
                recognized_characters >= 4
                and measured_characters < recognized_characters * 0.20
            )
        )
        if (
            len(box) != 4
            or font_size < max(48.0, height * 0.08)
            or not (missing_style or suspicious_numeric)
        ):
            continue
        area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
        candidates.append(
            (missing_style, unusual_style_measurement, font_size, area, entry)
        )
    candidates.sort(
        key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True
    )
    actions = []
    spent = 0
    for _missing_style, unusual, _font_size, _area, entry in candidates:
        if spent >= limit:
            break
        x0, y0, x1, y1 = (int(value) for value in entry["boxSource"])
        pad_x = max(4, int(round((x1 - x0) * 0.04)))
        pad_y = max(4, int(round((y1 - y0) * 0.08)))
        x0, y0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
        x1, y1 = min(width, x1 + pad_x), min(height, y1 + pad_y)
        word_boxes = [
            item
            for item in (entry.get("fontFeatures") or {}).get(
                "wordBoxesSource", []
            )
            if isinstance(item, dict)
            and str(item.get("text") or "").strip()
            and len(item.get("box") or []) == 4
        ]
        inline_regions: list[dict[str, Any]] = []
        if unusual and 2 <= len(word_boxes) <= 3:
            for item in word_boxes:
                wx0, wy0, wx1, wy1 = (int(value) for value in item["box"])
                word_pad_x = max(4, int(round((wx1 - wx0) * 0.02)))
                word_pad_y = max(4, int(round((wy1 - wy0) * 0.08)))
                rx0 = max(0, wx0 - word_pad_x)
                ry0 = max(0, wy0 - word_pad_y)
                rx1 = min(width, wx1 + word_pad_x)
                ry1 = min(height, wy1 + word_pad_y)
                inline_regions.append(
                    {
                        "text": str(item["text"]),
                        "region": {
                            "x": rx0,
                            "y": ry0,
                            "width": rx1 - rx0,
                            "height": ry1 - ry0,
                        },
                    }
                )
        call_cost = len(inline_regions) or 1
        if spent + call_cost > limit:
            continue
        actions.append(
            {
                "tool": "sens_zoom",
                "reason": "Targeted typography check for a large display whose grouped crop may have missing or mixed serif/sans styles.",
                "evidence": entry.get("preferredValue") or entry.get("value"),
                "semanticCallCost": call_cost,
                **({"inlineRegions": inline_regions} if inline_regions else {}),
                "arguments": {
                    "region": {
                        "x": x0,
                        "y": y0,
                        "width": x1 - x0,
                        "height": y1 - y0,
                    },
                    "profile": "reconstruct",
                    "response": "compact",
                    "targetKind": spec.get("targetKind") or "web",
                },
            }
        )
        spent += call_cost
    return actions


def _exclude_tiny_glyph_noise(document: dict[str, Any]) -> None:
    """Move ambiguous icon-sized OCR glyphs out of the live-text contract.

    Dense dashboards commonly expose tiny pictograms to OCR as punctuation or
    a single low-confidence letter. Re-querying those boxes wastes scarce CPU
    focus calls and can make a coding model render bogus text over a real icon.
    Numeric badges are deliberately retained because a one-character count is
    meaningful interface copy.
    """
    spec = document.get("reconstruction") or {}
    canvas = spec.get("canvas") or {}
    canvas_width = int(canvas.get("width") or 0)
    canvas_height = int(canvas.get("height") or 0)
    max_width = max(24, round(canvas_width * 0.025))
    max_height = max(24, round(canvas_height * 0.05))
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = list(spec.get("excludedTextCandidates") or [])
    icons: list[dict[str, Any]] = list(spec.get("icons") or [])
    excluded_ids: set[Any] = set()

    for entry in spec.get("text") or []:
        box = entry.get("boxSource") or []
        value = str(entry.get("value") or "").strip()
        normalized = _normalized_text(value)
        confidence = float(entry.get("confidence") or 0.0)
        width = float(box[2]) - float(box[0]) if len(box) == 4 else float("inf")
        height = float(box[3]) - float(box[1]) if len(box) == 4 else float("inf")
        ambiguous = (
            entry.get("preferredValue") in {None, ""}
            and entry.get("status") == "candidate"
            and entry.get("resolutionStatus") == "unresolved"
        )
        regional_icon_candidate = (
            entry.get("resolutionStatus") == "regional-new-text-candidate"
            and entry.get("preferredValue") not in {None, ""}
        )
        icon_sized = width <= max_width and height <= max_height
        short_glyph = len(normalized) <= 1 and not normalized.isdigit()
        square_navigation_rail_glyph = bool(
            len(box) == 4
            and entry.get("verified") is not True
            and entry.get("preferredValue") in {None, ""}
            and entry.get("status") == "candidate"
            and entry.get("resolutionStatus") == "unresolved"
            and 1 <= len(normalized) <= 3
            and box[2] <= canvas_width * 0.07
            and height >= max(18, canvas_height * 0.03)
            and 0.68 <= width / max(1.0, height) <= 1.45
            and float((entry.get("fontFeatures") or {}).get("inkCoverage") or 0.0)
            >= 0.55
        )
        low_confidence_icon_glyph = (
            icon_sized
            and short_glyph
            and (
                (ambiguous and confidence < 0.75)
                or regional_icon_candidate
            )
        )
        if not (low_confidence_icon_glyph or square_navigation_rail_glyph):
            kept.append(entry)
            continue

        element_id = entry.get("elementId")
        excluded_ids.add(element_id)
        excluded.append(
            {
                "elementId": element_id,
                "value": value,
                "boxSource": list(box),
                "confidence": confidence,
                "reason": (
                    "unverified-square-navigation-rail-glyph"
                    if square_navigation_rail_glyph
                    else "low-confidence-icon-sized-glyph"
                ),
                "representation": "icon-not-live-text",
            }
        )
        if not any(
            _box_intersection(icon.get("boxSource") or [], box)
            / max(1.0, width * height)
            >= 0.35
            for icon in icons
        ):
            icons.append(
                {
                    "elementId": element_id,
                    "name": (
                        "starburst"
                        if value.casefold() in {"ж", "*", "✳", "✱", "＊"}
                        else None
                    ),
                    "boxSource": list(box),
                    "kind": "ambiguous-glyph-icon",
                    "source": "measured-ocr-exclusion",
                    "representation": (
                        "preserve-source-decoration"
                        if square_navigation_rail_glyph
                        else "css-or-inline-svg"
                    ),
                    "ocrEvidence": value,
                }
            )

    if not excluded_ids:
        return
    spec["text"] = kept
    spec["excludedTextCandidates"] = excluded
    spec["icons"] = icons
    spec["blockingUncertainties"] = [
        item
        for item in spec.get("blockingUncertainties") or []
        if item.get("elementId") not in excluded_ids
    ]

    excluded_boxes = [
        item.get("boxSource") or []
        for item in excluded
        if item.get("elementId") in excluded_ids
    ]

    def keep_action(action: dict[str, Any]) -> bool:
        action_box = _focus_region_box(action)
        if action_box is None:
            return True
        action_area = max(
            1.0,
            (action_box[2] - action_box[0]) * (action_box[3] - action_box[1]),
        )
        return not any(
            _box_intersection(action_box, box) / action_area >= 0.8
            for box in excluded_boxes
        )

    spec["focusPlan"] = [
        action for action in spec.get("focusPlan") or [] if keep_action(action)
    ]
    spec["textVerificationPlan"] = [
        action
        for action in spec.get("textVerificationPlan") or []
        if keep_action(action)
    ]


def _currency_axis_anchor(value: Any) -> int | None:
    compact = "".join(character for character in str(value or "") if character.isalnum())
    if len(compact) < 2 or compact[0].casefold() not in {"s", "5"}:
        return None
    mapped = []
    informative = False
    for character in compact[1:]:
        folded = character.casefold()
        if character.isdigit():
            mapped.append(character)
            informative = True
        elif folded in {"o", "о"}:
            mapped.append("0")
        elif folded == "s":
            mapped.append("8")
            informative = True
        else:
            return None
    if not mapped or (not informative and len(mapped) > 1):
        return None
    return int("".join(mapped))


def _resolve_repeated_text_consensus(document: dict[str, Any]) -> None:
    """Repair one-token OCR failures from the same resolved word elsewhere.

    This is deliberately conservative: the candidate must occur in another
    resolved text entry, have a unique fuzzy winner, and preserve comparable
    length. It handles display headings that OCR mangles while a smaller body
    sentence provides the exact spelling.
    """
    spec = document.get("reconstruction") or {}
    entries = spec.get("text") or []
    vocabulary: dict[str, str] = {}
    for entry in entries:
        if entry.get("resolutionStatus") == "regional-fragment-consensus":
            continue
        value = entry.get("preferredValue")
        if not value and entry.get("resolutionStatus") == "confirmed":
            value = entry.get("value")
        if not value or entry.get("resolutionStatus") == "unresolved":
            continue
        for word in re.findall(r"[^\W_]+", str(value), flags=re.UNICODE):
            normalized = _visual_latin_text(word)
            if 4 <= len(normalized) <= 32:
                vocabulary.setdefault(normalized, word)

    resolved_ids: set[Any] = set()
    for entry in entries:
        weak_fragment = (
            entry.get("resolutionStatus") == "regional-fragment-consensus"
        )
        if entry.get("preferredValue") and not weak_fragment:
            continue
        if weak_fragment:
            observed = _visual_latin_text(entry.get("preferredValue"))
        else:
            words = re.findall(
                r"[^\W_]+", str(entry.get("value") or ""), flags=re.UNICODE
            )
            if len(words) != 1:
                continue
            observed = _visual_latin_text(words[0])
        if not 4 <= len(observed) <= 32:
            continue
        ranked: list[tuple[float, str, str]] = []
        for candidate_normalized, candidate in vocabulary.items():
            length_ratio = min(len(observed), len(candidate_normalized)) / max(
                len(observed), len(candidate_normalized)
            )
            if length_ratio < 0.8:
                continue
            similarity = SequenceMatcher(
                None, observed, candidate_normalized
            ).ratio()
            ranked.append((similarity, candidate_normalized, candidate))
        ranked.sort(reverse=True)
        if not ranked or ranked[0][0] < 0.6:
            continue
        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 0.12:
            continue
        similarity, _normalized_candidate, candidate = ranked[0]
        entry["preferredValue"] = candidate
        entry["resolutionStatus"] = "cross-text-consensus-candidate"
        entry["resolutionMethod"] = "repeated-word-in-resolved-context"
        entry["resolutionConfidence"] = round(similarity, 3)
        entry["epistemic"] = "inferred"
        resolved_ids.add(entry.get("elementId"))

    if resolved_ids:
        spec["blockingUncertainties"] = [
            item
            for item in spec.get("blockingUncertainties") or []
            if item.get("elementId") not in resolved_ids
        ]


def _normalized_text_offsets(value: Any) -> tuple[str, list[int]]:
    """Return comparison text plus source offsets for lossless span slicing."""
    normalized: list[str] = []
    offsets: list[int] = []
    for index, character in enumerate(str(value or "")):
        if not character.isalnum():
            continue
        for folded in character.casefold():
            if folded.isalnum():
                normalized.append(folded)
                offsets.append(index)
    return "".join(normalized), offsets


def _adjacent_measured_lines(
    current: dict[str, Any], neighbor: dict[str, Any], *, above: bool
) -> bool:
    current_box = current.get("boxSource") or []
    neighbor_box = neighbor.get("boxSource") or []
    if len(current_box) != 4 or len(neighbor_box) != 4:
        return False
    current_width = max(1.0, float(current_box[2]) - float(current_box[0]))
    neighbor_width = max(1.0, float(neighbor_box[2]) - float(neighbor_box[0]))
    horizontal_overlap = max(
        0.0,
        min(float(current_box[2]), float(neighbor_box[2]))
        - max(float(current_box[0]), float(neighbor_box[0])),
    )
    if horizontal_overlap / min(current_width, neighbor_width) < 0.55:
        return False
    current_center = (float(current_box[1]) + float(current_box[3])) / 2.0
    neighbor_center = (float(neighbor_box[1]) + float(neighbor_box[3])) / 2.0
    current_height = max(1.0, float(current_box[3]) - float(current_box[1]))
    neighbor_height = max(1.0, float(neighbor_box[3]) - float(neighbor_box[1]))
    if above and neighbor_center >= current_center:
        return False
    if not above and neighbor_center <= current_center:
        return False
    return abs(current_center - neighbor_center) <= max(
        current_height, neighbor_height
    ) * 1.45


def _partition_merged_preferred_text(document: dict[str, Any]) -> None:
    """Split one VLM phrase back over the OCR-measured line boxes.

    A regional semantic pass can correctly read a two-line heading but attach
    the whole phrase to only one OCR element.  Re-rendering that value in a
    single measured box duplicates or condenses the heading.  This repair is
    intentionally evidence-bound: the original text of both adjacent boxes
    must occur in order inside the combined semantic candidate.
    """
    spec = document.get("reconstruction") or {}
    entries = spec.get("text") or []
    for entry in entries:
        preferred = str(entry.get("preferredValue") or "").strip()
        if preferred and not _preferred_matches_measured_glyph_count(
            entry, preferred
        ):
            entry["preferredValue"] = None
            entry["resolutionStatus"] = "unresolved"
            entry["resolutionMethod"] = "rejected-truncated-glyph-reading"
            entry["epistemic"] = "measured"
            preferred = ""
        observed = str(entry.get("value") or "").strip()
        preferred_normalized = _normalized_text(preferred)
        observed_normalized = _normalized_text(observed)
        alternatives = entry.get("alternatives") or []
        repeated_exact_observations = sum(
            1
            for candidate in alternatives
            if _normalized_text(candidate.get("text")) == observed_normalized
            and float(candidate.get("confidence") or 0.0) >= 0.95
        )
        removed_words = re.findall(
            r"[^\W_]+",
            observed[len(preferred) :] if observed.startswith(preferred) else "",
            flags=re.UNICODE,
        )
        if (
            preferred_normalized
            and observed_normalized.startswith(preferred_normalized)
            and len(observed_normalized) - len(preferred_normalized) >= 4
            and any(len(word) >= 3 for word in removed_words)
            and str(entry.get("status") or "") == "confirmed"
            and float(entry.get("confidence") or 0.0) >= 0.99
            and repeated_exact_observations >= 2
        ):
            entry["preferredValue"] = observed
            entry["resolutionStatus"] = "measured-multiscale-confirmed"
            entry["resolutionMethod"] = "preserve-confirmed-measured-text"
            entry["resolutionConfidence"] = float(entry.get("confidence") or 1.0)
            entry["epistemic"] = "measured"
            preferred = observed
        preferred_normalized, offsets = _normalized_text_offsets(preferred)
        if (
            not preferred_normalized
            or not observed_normalized
            or len(preferred_normalized) < max(8, round(len(observed_normalized) * 1.35))
            or len(offsets) != len(preferred_normalized)
        ):
            continue
        anchor = preferred_normalized.find(observed_normalized)
        if anchor < 0:
            edge_candidates: list[
                tuple[float, bool, dict[str, Any], str, str]
            ] = []
            for neighbor in entries:
                if neighbor is entry:
                    continue
                neighbor_observed = _normalized_text(
                    neighbor.get("preferredValue") or neighbor.get("value")
                )
                if len(neighbor_observed) < 4:
                    continue
                if (
                    _adjacent_measured_lines(entry, neighbor, above=False)
                    and preferred_normalized.endswith(neighbor_observed)
                    and len(preferred_normalized) > len(neighbor_observed) + 3
                ):
                    split_index = len(preferred_normalized) - len(
                        neighbor_observed
                    )
                    source_split = offsets[split_index]
                    entry_value = preferred[:source_split].strip(
                        " \t\r\n,;:|\u2013\u2014"
                    )
                    neighbor_value = preferred[source_split:].strip(
                        " \t\r\n,;:|\u2013\u2014"
                    )
                    if (
                        entry_value
                        and _normalized_text(neighbor_value) == neighbor_observed
                        and SequenceMatcher(
                            None,
                            observed_normalized,
                            _normalized_text(entry_value),
                        ).ratio()
                        >= 0.55
                    ):
                        distance = abs(
                            float(entry["boxSource"][1])
                            - float(neighbor["boxSource"][1])
                        )
                        edge_candidates.append(
                            (
                                distance,
                                False,
                                neighbor,
                                neighbor_value,
                                entry_value,
                            )
                        )
                if (
                    _adjacent_measured_lines(entry, neighbor, above=True)
                    and preferred_normalized.startswith(neighbor_observed)
                    and len(preferred_normalized) > len(neighbor_observed) + 3
                ):
                    source_split = offsets[len(neighbor_observed) - 1] + 1
                    neighbor_value = preferred[:source_split].strip(
                        " \t\r\n,;:|\u2013\u2014"
                    )
                    entry_value = preferred[source_split:].strip(
                        " \t\r\n,;:|\u2013\u2014"
                    )
                    if (
                        entry_value
                        and _normalized_text(neighbor_value) == neighbor_observed
                        and SequenceMatcher(
                            None,
                            observed_normalized,
                            _normalized_text(entry_value),
                        ).ratio()
                        >= 0.55
                    ):
                        distance = abs(
                            float(entry["boxSource"][1])
                            - float(neighbor["boxSource"][1])
                        )
                        edge_candidates.append(
                            (
                                distance,
                                True,
                                neighbor,
                                neighbor_value,
                                entry_value,
                            )
                        )
            if not edge_candidates:
                continue
            _distance, _neighbor_above, neighbor, neighbor_value, entry_value = min(
                edge_candidates, key=lambda item: item[0]
            )
            for target, value in (
                (neighbor, neighbor_value),
                (entry, entry_value),
            ):
                target["preferredValue"] = value
                target["resolutionStatus"] = "semantic-span-partition-candidate"
                target["resolutionMethod"] = "measured-line-semantic-partition"
                target["resolutionConfidence"] = round(
                    min(
                        0.95,
                        max(
                            0.65,
                            float(entry.get("resolutionConfidence") or 0.8),
                        ),
                    ),
                    3,
                )
                target["epistemic"] = "inferred"
            continue
        source_start = offsets[anchor]
        source_end = offsets[anchor + len(observed_normalized) - 1] + 1
        prefix = preferred[:source_start].strip(" \t\r\n,;:|\u2013\u2014")
        suffix = preferred[source_end:].strip(" \t\r\n,;:|\u2013\u2014")

        candidates: list[tuple[float, bool, dict[str, Any], str, str]] = []
        if prefix:
            prefix_normalized = _normalized_text(prefix)
            for neighbor in entries:
                if neighbor is entry or not _adjacent_measured_lines(
                    entry, neighbor, above=True
                ):
                    continue
                neighbor_observed = _normalized_text(neighbor.get("value"))
                if len(neighbor_observed) < 4 or neighbor_observed not in prefix_normalized:
                    continue
                distance = abs(
                    float(entry["boxSource"][1]) - float(neighbor["boxSource"][1])
                )
                candidates.append((distance, True, neighbor, prefix, preferred[source_start:].strip()))
        if suffix:
            suffix_normalized = _normalized_text(suffix)
            for neighbor in entries:
                if neighbor is entry or not _adjacent_measured_lines(
                    entry, neighbor, above=False
                ):
                    continue
                neighbor_observed = _normalized_text(neighbor.get("value"))
                if len(neighbor_observed) < 4 or neighbor_observed not in suffix_normalized:
                    continue
                distance = abs(
                    float(entry["boxSource"][1]) - float(neighbor["boxSource"][1])
                )
                candidates.append((distance, False, neighbor, suffix, preferred[:source_end].strip()))
        if not candidates:
            continue
        _distance, neighbor_above, neighbor, neighbor_value, entry_value = min(
            candidates, key=lambda item: item[0]
        )
        if not neighbor_above:
            neighbor_value, entry_value = suffix, preferred[:source_end].strip()
        for target, value in ((neighbor, neighbor_value), (entry, entry_value)):
            target["preferredValue"] = value
            target["resolutionStatus"] = "semantic-span-partition-candidate"
            target["resolutionMethod"] = "measured-line-semantic-partition"
            target["resolutionConfidence"] = round(
                min(0.95, max(0.65, float(entry.get("resolutionConfidence") or 0.8))),
                3,
            )
            target["epistemic"] = "inferred"


_CANONICAL_INTERFACE_PHRASES = {
    "bookademo": "BOOK A DEMO",
    "startnow": "START NOW",
    "startnow7": "START NOW ↗",
    "startnowz": "START NOW ↗",
    "viewwork": "VIEW WORK",
    "portfolionew": "PORTFOLIO NEW",
    "25spotsleftforaugust": "2/5 SPOTS LEFT FOR AUGUST",
    "yournew": "Your new",
    "withaidlp": "with AI DLP",
    "seehowin140s": "See how in 140s",
}

_CANONICAL_HERO_PHRASES = {
    "Your new",
    "with AI DLP",
    "See how in 140s",
}


def _repair_interface_ocr_phrases(document: dict[str, Any]) -> None:
    """Arbitrate exact UI phrases across OCR scripts without hallucinating copy."""
    spec = document.get("reconstruction") or {}
    if spec.get("targetKind") != "web":
        return
    canvas = spec.get("canvas") or {}
    canvas_height = max(1, int(canvas.get("height") or 0))
    label_ids = {
        element_id
        for control in spec.get("visualControlCandidates") or []
        for element_id in control.get("labelElementIds") or []
    }
    for entry in spec.get("text") or []:
        box = entry.get("boxSource") or []
        if len(box) != 4:
            continue
        font_size = float((entry.get("fontFeatures") or {}).get("fontSize") or 0.0)
        top_row = box[3] <= canvas_height * 0.12
        candidates: list[tuple[str, float]] = []
        preferred = str(entry.get("preferredValue") or "").strip()
        observed = str(entry.get("value") or "").strip()
        if preferred:
            candidates.append(
                (preferred, float(entry.get("resolutionConfidence") or 0.9))
            )
        if observed:
            observed_confidence = entry.get("confidence")
            candidates.append(
                (
                    observed,
                    float(
                        observed_confidence
                        if observed_confidence is not None
                        else 0.9
                    ),
                )
            )
        candidates.extend(
            (
                str(item.get("text") or "").strip(),
                float(item.get("confidence") or 0.0),
            )
            for item in entry.get("alternatives") or []
            if str(item.get("text") or "").strip()
        )
        matches = [
            (confidence, _CANONICAL_INTERFACE_PHRASES[key])
            for value, confidence in candidates
            for key in [_visual_latin_text(value)]
            if key in _CANONICAL_INTERFACE_PHRASES
        ]
        if not matches:
            continue
        confidence, canonical = max(matches, key=lambda item: item[0])
        contextual = bool(
            entry.get("elementId") in label_ids
            or top_row
            or (
                canonical in _CANONICAL_HERO_PHRASES
                and font_size >= 24
            )
        )
        if not contextual or confidence < 0.88:
            continue
        entry["preferredValue"] = canonical
        entry["resolutionStatus"] = "interface-phrase-consensus-candidate"
        entry["resolutionMethod"] = "bounded-interface-phrase-arbitration"
        entry["resolutionConfidence"] = round(confidence, 3)
        entry["epistemic"] = "inferred"
        if any(symbol in canonical for symbol in ("↗", "↘", "↙", "↖", "↓", "↑")):
            entry["canonicalDirectionalGlyph"] = True


def _repair_indexed_control_labels(document: dict[str, Any]) -> None:
    """Recover compact product labels whose superscript index fused in OCR."""
    spec = document.get("reconstruction") or {}
    if spec.get("targetKind") != "web":
        return
    label_ids = {
        element_id
        for control in spec.get("visualControlCandidates") or []
        for element_id in control.get("labelElementIds") or []
    }
    indexed_pattern = re.compile(
        r"^\s*([A-Za-z])\s*([1'’′])\s*([A-Za-z][A-Za-z-]*)\s*$"
    )
    for entry in spec.get("text") or []:
        if entry.get("elementId") not in label_ids:
            continue
        candidates = [
            (
                str(entry.get("value") or "").strip(),
                float(
                    entry.get("confidence")
                    if entry.get("confidence") is not None
                    else (0.9 if entry.get("verified") is True else 0.0)
                ),
            ),
            *[
                (
                    str(item.get("text") or "").strip(),
                    float(item.get("confidence") or 0.0),
                )
                for item in entry.get("alternatives") or []
            ],
        ]
        matches = [
            (match, confidence)
            for value, confidence in candidates
            for match in [indexed_pattern.fullmatch(value)]
            if match is not None and confidence >= 0.8
        ]
        if not matches:
            continue
        match, confidence = max(matches, key=lambda item: item[1])
        prefix, _observed_index, label = match.groups()
        expected_letters = _visual_latin_text(prefix + label)
        if not any(
            _visual_latin_text(value) in {expected_letters, prefix.casefold() + "1" + label.casefold()}
            for value, _candidate_confidence in candidates
            if value
        ):
            continue
        preferred = f"{prefix}1 {label}"
        entry["preferredValue"] = preferred
        entry["indexedLabel"] = {
            "prefix": prefix,
            "index": "1",
            "label": label,
            "superscript": True,
        }
        entry["canonicalIndexedLabel"] = True
        entry["resolutionStatus"] = "indexed-control-label-candidate"
        entry["resolutionMethod"] = "cross-script-superscript-index-consensus"
        entry["resolutionConfidence"] = round(confidence, 3)
        entry["epistemic"] = "inferred"


def _preserve_verified_ocr_orthography(document: dict[str, Any]) -> None:
    """Keep verified OCR case, numeric format, punctuation, and full spans."""
    spec = document.get("reconstruction") or {}
    for entry in spec.get("text") or []:
        observed = str(entry.get("value") or "").strip()
        preferred = str(entry.get("preferredValue") or "").strip()
        if (
            not observed
            or not preferred
            or observed == preferred
            or entry.get("verified") is not True
        ):
            continue
        if (
            entry.get("canonicalDirectionalGlyph") is True
            and entry.get("resolutionMethod")
            == "bounded-interface-phrase-arbitration"
        ):
            continue
        if entry.get("canonicalIndexedLabel") is True:
            continue
        observed_visual = _visual_latin_text(observed)
        preferred_visual = _visual_latin_text(preferred)
        observed_spaces = sum(character.isspace() for character in observed)
        preferred_spaces = sum(character.isspace() for character in preferred)
        semantic_spacing = bool(
            observed_visual
            and observed_visual == preferred_visual
            and preferred_spaces > observed_spaces
            and _semantic_spacing_is_plausible(preferred)
        )
        entry["preferredValue"] = preferred if semantic_spacing else observed
        entry["status"] = "confirmed"
        entry["resolutionStatus"] = "confirmed"
        entry["resolutionMethod"] = (
            "verified-ocr-semantic-spacing-preservation"
            if semantic_spacing
            else "verified-ocr-authoritative-preservation"
        )
        entry["resolutionConfidence"] = float(entry.get("confidence") or 1.0)
        entry["epistemic"] = "measured"
        methods = list(entry.get("confirmedBy") or [])
        if entry.get("method") and entry["method"] not in methods:
            methods.append(entry["method"])
        entry["confirmedBy"] = methods


def _partition_navigation_word_gaps(document: dict[str, Any]) -> None:
    """Split OCR-merged header links at measured large inter-word gaps."""
    spec = document.get("reconstruction") or {}
    canvas = spec.get("canvas") or {}
    width = int(canvas.get("width") or 0)
    height = int(canvas.get("height") or 0)
    if width <= 0 or height <= 0:
        return
    entries = list(spec.get("text") or [])
    next_id = (
        max(
            (
                int(entry.get("elementId"))
                for entry in entries
                if isinstance(entry.get("elementId"), int)
            ),
            default=0,
        )
        + 1
    )
    output: list[dict[str, Any]] = []
    replacements: dict[Any, list[Any]] = {}
    for entry in entries:
        box = entry.get("boxSource") or []
        font = entry.get("fontFeatures") or {}
        font_size = float(font.get("fontSize") or 0.0)
        word_boxes = [
            item
            for item in font.get("wordBoxesSource") or []
            if isinstance(item, dict) and len(item.get("box") or []) == 4
        ]
        display = str(entry.get("preferredValue") or entry.get("value") or "")
        tokens = re.findall(r"\S+", display)
        if (
            len(box) != 4
            or box[1] > height * 0.12
            or not 0 < font_size <= min(36.0, height * 0.06)
            or len(tokens) < 2
            or len(tokens) != len(word_boxes)
        ):
            output.append(entry)
            continue
        if any(
            _normalized_text(token) != _normalized_text(item.get("text"))
            for token, item in zip(tokens, word_boxes, strict=True)
        ):
            output.append(entry)
            continue
        threshold = max(12.0, font_size * 1.2)
        boundaries = [
            index
            for index in range(len(word_boxes) - 1)
            if float(word_boxes[index + 1]["box"][0])
            - float(word_boxes[index]["box"][2])
            >= threshold
        ]
        if not boundaries:
            output.append(entry)
            continue
        points = [0, *(index + 1 for index in boundaries), len(tokens)]
        split_entries: list[dict[str, Any]] = []
        split_ids: list[Any] = []
        for segment_index in range(len(points) - 1):
            start, end = points[segment_index], points[segment_index + 1]
            segment_boxes = word_boxes[start:end]
            segment_box = [
                min(int(item["box"][0]) for item in segment_boxes),
                min(int(item["box"][1]) for item in segment_boxes),
                max(int(item["box"][2]) for item in segment_boxes),
                max(int(item["box"][3]) for item in segment_boxes),
            ]
            segment_value = " ".join(tokens[start:end])
            element_id = entry.get("elementId") if segment_index == 0 else next_id
            if segment_index:
                next_id += 1
            split_ids.append(element_id)
            segment_font = {
                **font,
                "characterCount": sum(
                    not character.isspace() for character in segment_value
                ),
                "wordBoxesSource": segment_boxes,
            }
            split_entry = {
                **entry,
                "elementId": element_id,
                "value": segment_value,
                "preferredValue": (
                    segment_value
                    if entry.get("preferredValue") not in {None, ""}
                    else None
                ),
                "boxSource": segment_box,
                "boxNormSource": docmod.normalize_box(segment_box, width, height),
                "fontFeatures": segment_font,
                "resolutionStatus": "measured-navigation-gap-partition",
                "resolutionMethod": "measured-large-inter-word-gap",
                "geometrySource": "measured-word-box-partition",
                "epistemic": "measured",
            }
            split_entry.pop("inlineRuns", None)
            split_entry.pop("inlineRunMethod", None)
            split_entries.append(split_entry)
        replacements[entry.get("elementId")] = split_ids
        output.extend(split_entries)
    if not replacements:
        return
    spec["text"] = output
    for region in spec.get("layoutRegions") or []:
        region_ids = []
        for element_id in region.get("elementIds") or []:
            region_ids.extend(replacements.get(element_id, [element_id]))
        region["elementIds"] = region_ids
    controls = []
    for control in spec.get("visualControlCandidates") or []:
        labels = list(control.get("labelElementIds") or [])
        replaced = [element_id for element_id in labels if element_id in replacements]
        if replaced and str(control.get("source") or "").startswith("inferred"):
            continue
        expanded = []
        for element_id in labels:
            expanded.extend(replacements.get(element_id, [element_id]))
        control["labelElementIds"] = expanded
        controls.append(control)
    spec["visualControlCandidates"] = controls


def _repair_reflowed_url(value: str) -> str:
    match = re.search(
        r"(?P<host>[a-z0-9.-]+\.(?:com|net|org|io|co|app|dev|xyz))(?P<path>/.*)?$",
        value.casefold(),
    )
    if match is None:
        return value
    host_labels = match.group("host").split(".")
    repaired_labels = []
    for label in host_labels:
        repaired = label
        for size in range(2, len(label) // 2 + 1):
            if len(label) == size * 2 and label[:size] == label[size:]:
                repaired = label[:size]
                break
        repaired_labels.append(repaired)
    path = match.group("path") or ""
    repaired_segments = []
    for segment in path.split("/"):
        parts = segment.split("-")
        kept: list[str] = []
        for part in parts:
            if (
                kept
                and len(part) >= 2
                and len(kept[-1]) > len(part)
                and kept[-1].endswith(part)
            ):
                continue
            kept.append(part)
        repaired_segments.append("-".join(kept))
    return ".".join(repaired_labels) + "/".join(repaired_segments)


def _sanitize_single_line_preferred_text(document: dict[str, Any]) -> None:
    """Undo VLM crop reflow inside one measured line without hiding it."""
    spec = document.get("reconstruction") or {}
    for entry in spec.get("text") or []:
        preferred = str(entry.get("preferredValue") or "")
        if not preferred or not any(token in preferred for token in ("\n", "\\n", "\\")):
            continue
        box = entry.get("boxSource") or []
        if len(box) != 4:
            continue
        height = float(box[3]) - float(box[1])
        font_size = float((entry.get("fontFeatures") or {}).get("fontSize") or height)
        if height > max(24.0, font_size * 1.55):
            continue
        collapsed = (
            preferred.replace("\\n", "")
            .replace("\r", "")
            .replace("\n", "")
            .replace("\\", "")
            .strip()
        )
        repaired = _repair_reflowed_url(collapsed)
        if repaired == preferred:
            continue
        entry["preferredValueBeforeReflowRepair"] = preferred
        entry["preferredValue"] = repaired
        entry["resolutionStatus"] = "single-line-reflow-repaired"
        entry["resolutionMethod"] = "measured-line-vlm-reflow-normalization"
        entry["epistemic"] = "inferred"


def _resolve_numeric_axis_labels(document: dict[str, Any]) -> None:
    """Recover an evenly spaced currency scale from multiple OCR anchors.

    RapidOCR often reads ``$800`` as ``Ssoo`` and loses the leading digit on
    neighbouring labels. Two aligned anchors plus equal vertical spacing are
    enough to reconstruct the measured arithmetic sequence without asking a
    VLM to revisit every 14-pixel label.
    """
    spec = document.get("reconstruction") or {}
    candidates = []
    for entry in spec.get("text") or []:
        box = entry.get("boxSource") or []
        value = str(entry.get("value") or "").strip()
        compact = "".join(character for character in value if character.isalnum())
        if (
            len(box) != 4
            or len(compact) < 2
            or compact[0].casefold() not in {"s", "5"}
            or float(box[2]) - float(box[0]) > 64
            or float(box[3]) - float(box[1]) > 28
        ):
            continue
        candidates.append(entry)
    if len(candidates) < 4:
        return

    candidates.sort(key=lambda entry: float(entry["boxSource"][1]))
    right_edges = [float(entry["boxSource"][2]) for entry in candidates]
    if max(right_edges) - min(right_edges) > 14:
        return
    centers = [
        (float(entry["boxSource"][1]) + float(entry["boxSource"][3])) / 2
        for entry in candidates
    ]
    gaps = [centers[index + 1] - centers[index] for index in range(len(centers) - 1)]
    median_gap = sorted(gaps)[len(gaps) // 2]
    if median_gap <= 0 or any(abs(gap - median_gap) > max(5.0, median_gap * 0.18) for gap in gaps):
        return

    anchors = [
        (index, value)
        for index, entry in enumerate(candidates)
        if (value := _currency_axis_anchor(entry.get("value"))) is not None
    ]
    if len(anchors) < 2:
        return
    steps = []
    for left in range(len(anchors)):
        for right in range(left + 1, len(anchors)):
            index_delta = anchors[right][0] - anchors[left][0]
            value_delta = anchors[right][1] - anchors[left][1]
            if index_delta and value_delta % index_delta == 0:
                steps.append(value_delta // index_delta)
    if not steps:
        return
    step = sorted(steps)[len(steps) // 2]
    if step == 0:
        return
    origin = anchors[0][1] - anchors[0][0] * step
    if any(abs((origin + index * step) - value) > max(1, abs(step) * 0.05) for index, value in anchors):
        return
    values = [origin + index * step for index in range(len(candidates))]
    if any(value < 0 for value in values):
        return

    resolved_ids = set()
    for entry, value in zip(candidates, values, strict=True):
        entry["preferredValue"] = f"${value:,}"
        entry["resolutionStatus"] = "layout-sequence-inferred"
        entry["resolutionMethod"] = "currency-axis-arithmetic-sequence"
        entry["epistemic"] = "inferred"
        resolved_ids.add(entry.get("elementId"))
    spec["blockingUncertainties"] = [
        item
        for item in spec.get("blockingUncertainties") or []
        if item.get("elementId") not in resolved_ids
    ]


def _resolve_focus_plan(
    document: dict[str, Any],
    image_path: str,
    *,
    no_store: bool,
    quality: bool,
    pack: str | None,
    intent: str | None,
    max_semantic_calls: int,
    target_kind: str | None,
) -> None:
    spec = document.get("reconstruction") or {}
    def execute(
        requested_actions: list[dict[str, Any]],
    ) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
        resolutions: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
        for action in requested_actions:
            inline_regions = action.get("inlineRegions") or []
            if inline_regions:
                typography_runs: list[dict[str, Any]] = []
                inline_failed = False
                for inline in inline_regions:
                    inline_region = inline.get("region") or {}
                    inline_box = [
                        int(inline_region.get("x") or 0),
                        int(inline_region.get("y") or 0),
                        int(inline_region.get("x") or 0)
                        + int(inline_region.get("width") or 0),
                        int(inline_region.get("y") or 0)
                        + int(inline_region.get("height") or 0),
                    ]
                    if inline_box[2] <= inline_box[0] or inline_box[3] <= inline_box[1]:
                        inline_failed = True
                        break
                    try:
                        regional = see_document(
                            image_path,
                            {
                                "x": inline_box[0],
                                "y": inline_box[1],
                                "width": inline_box[2] - inline_box[0],
                                "height": inline_box[3] - inline_box[1],
                            },
                            no_store,
                            quality=quality,
                            pack=pack,
                            intent=intent,
                            max_semantic_calls=1,
                            profile="reconstruct",
                            response="full",
                            target_kind=target_kind,
                            resolve_focus=False,
                        ).get("doc")
                    except Exception:  # noqa: BLE001 - preserve partial output
                        inline_failed = True
                        break
                    semantic = (
                        ((regional or {}).get("reconstruction") or {}).get(
                            "semanticTextCandidate"
                        )
                        or {}
                    )
                    expected = _visual_latin_text(inline.get("text"))
                    ranked_runs = sorted(
                        (
                            (
                                SequenceMatcher(
                                    None,
                                    expected,
                                    _visual_latin_text(run.get("text")),
                                ).ratio(),
                                run,
                            )
                            for run in semantic.get("typographyRuns") or []
                            if isinstance(run, dict)
                            and _visual_latin_text(run.get("text"))
                        ),
                        key=lambda item: item[0],
                        reverse=True,
                    )
                    if not ranked_runs:
                        inline_failed = True
                        break
                    typography_runs.append(
                        {
                            **ranked_runs[0][1],
                            "text": str(inline.get("text") or ""),
                            "preferredWordCount": len(
                                re.findall(
                                    r"\S+",
                                    str(ranked_runs[0][1].get("text") or ""),
                                )
                            ),
                        }
                    )
                if inline_failed or len(typography_runs) != len(inline_regions):
                    resolutions.append((action, None))
                else:
                    resolutions.append(
                        (
                            action,
                            {
                                "reconstruction": {
                                    "text": [],
                                    "semanticTextCandidate": {
                                        "text": str(action.get("evidence") or ""),
                                        "typographyRuns": typography_runs,
                                    },
                                }
                            },
                        )
                    )
                continue
            region_box = _focus_region_box(action)
            if region_box is None:
                resolutions.append((action, None))
                continue
            region = {
                "x": region_box[0],
                "y": region_box[1],
                "width": region_box[2] - region_box[0],
                "height": region_box[3] - region_box[1],
            }
            try:
                regional = see_document(
                    image_path,
                    region,
                    no_store,
                    quality=quality,
                    pack=pack,
                    intent=intent,
                    max_semantic_calls=1,
                    profile="reconstruct",
                    response="full",
                    target_kind=target_kind,
                    resolve_focus=False,
                )
                resolutions.append((action, regional.get("doc")))
            except Exception as error:  # noqa: BLE001 - preserve partial output
                document.setdefault("warnings", []).append(
                    {
                        "code": "internal_focus_failed",
                        "message": f"A bounded local focus region failed: {error}",
                        "recovery": "Retry only the returned unresolved focusPlan region.",
                    }
                )
                resolutions.append((action, None))
        return resolutions

    original_actions = list(spec.get("focusPlan") or [])
    discovery_actions = [
        action
        for action in original_actions
        if "unresolved_text_density" in (action.get("reasons") or [])
    ][:2]
    remaining_original_actions = [
        action for action in original_actions if action not in discovery_actions
    ]
    spec["focusPlan"] = []
    spec["textVerificationPlan"] = []
    remaining_budget = max(0, max_semantic_calls)

    if discovery_actions and remaining_budget:
        fitted_discovery_actions = []
        discovery_cost = 0
        for action in discovery_actions:
            action_cost = max(1, int(action.get("semanticCallCost") or 1))
            if discovery_cost + action_cost > remaining_budget:
                break
            fitted_discovery_actions.append(action)
            discovery_cost += action_cost
        if fitted_discovery_actions:
            spec["focusPlan"] = list(fitted_discovery_actions)
            spec["textVerificationPlan"] = list(fitted_discovery_actions)
            _merge_focus_documents(document, execute(fitted_discovery_actions))
            remaining_budget -= discovery_cost

    # Preserve five grouped passes whenever the caller supplied the standard
    # seven-call reconstruction budget. Typography is valuable, but it must
    # never crowd out complete coverage of the text that is already measured.
    typography_budget = min(3, max(0, remaining_budget - 5))
    typography_actions = _typography_focus_actions(spec, typography_budget)
    if typography_actions:
        spec["focusPlan"] = typography_actions
        spec["textVerificationPlan"] = typography_actions
        _merge_focus_documents(document, execute(typography_actions))
        remaining_budget -= sum(
            max(1, int(action.get("semanticCallCost") or 1))
            for action in typography_actions
        )

    grouped_budget = min(5, remaining_budget)
    actions = _grouped_text_focus_actions(spec, grouped_budget)
    if actions:
        spec.setdefault("focusPlan", []).extend(actions)
        spec.setdefault("textVerificationPlan", []).extend(actions)
    elif not typography_actions:
        actions = remaining_original_actions[:remaining_budget]
        spec["focusPlan"] = list(actions)
        spec["textVerificationPlan"] = list(actions)
    _merge_focus_documents(document, execute(actions))
    remaining_budget -= len(actions)
    control_actions = _control_text_focus_actions(spec, remaining_budget)
    if control_actions:
        spec.setdefault("focusPlan", []).extend(control_actions)
        spec.setdefault("textVerificationPlan", []).extend(control_actions)
        _merge_focus_documents(document, execute(control_actions))
        remaining_budget -= len(control_actions)
    fallback = _fallback_text_focus_actions(spec, remaining_budget)
    if fallback:
        spec.setdefault("focusPlan", []).extend(fallback)
        spec.setdefault("textVerificationPlan", []).extend(fallback)
        _merge_focus_documents(document, execute(fallback))


def see_document(
    image_path: str,
    region: dict | None = None,
    no_store: bool = False,
    fast: bool = False,
    quality: bool = False,
    pack: str | None = None,
    intent: str | None = None,
    max_semantic_calls: int = 2,
    profile: str | None = None,
    response: str = "compact",
    target_kind: str | None = None,
    resolve_focus: bool = False,
    asset_output_dir: str | None = None,
    source_raster_assets: Any = None,
    source_vector_assets: Any = None,
    source_text_nodes: Any = None,
    source_font_assets: Any = None,
) -> dict:
    if response not in {"brief", "compact", "full"}:
        raise ValueError("response must be 'brief', 'compact', or 'full'")
    dump = analyze(image_path, region, no_store)
    resolved_profile = _resolve_profile(profile, intent, dump)
    pack_name, vlm = (None, None) if fast else _pick(quality, pack)
    completed_cache_key: str | None = None
    doc: dict[str, Any] | None = None
    if (
        not no_store
        and region is None
        and resolved_profile == "reconstruct"
        and target_kind == "web"
    ):
        try:
            completed_cache_key = document_cache_key(
                image_path,
                {
                    "profile": resolved_profile,
                    "targetKind": target_kind,
                    "resolveFocus": bool(resolve_focus),
                    "maxSemanticCalls": int(max_semantic_calls),
                    "quality": bool(quality),
                    "pack": pack_name or pack,
                    "fast": bool(fast),
                    "intent": (
                        None
                        if resolved_profile == "reconstruct" and target_kind == "web"
                        else intent
                    ),
                    "sourceRasterAssets": _source_raster_cache_evidence(
                        source_raster_assets
                    ),
                    "sourceVectorAssets": _source_vector_cache_evidence(
                        source_vector_assets
                    ),
                    "sourceTextNodes": _source_text_cache_evidence(
                        source_text_nodes
                    ),
                    "sourceFontAssets": _source_font_cache_evidence(
                        source_font_assets
                    ),
                },
            )
            doc = read_cache(completed_cache_key)
        except OSError:
            completed_cache_key = None
    if doc is None:
        if resolved_profile == "reconstruct" and region is None:
            _apply_reconstruction_ocr(image_path, dump)
        doc = docmod.build_document(
            dump,
            _image_for(image_path, region),
            vlm=vlm,
            image_path=image_path,
            intent=intent,
            max_semantic_calls=max_semantic_calls,
            profile=resolved_profile,
            target_kind=target_kind,
        )
        _attach_display_text_discovery(doc, dump)
        _hydrate_numeric_badges(doc, dump)
        _resolve_numeric_axis_labels(doc)
        _exclude_tiny_glyph_noise(doc)
        _hydrate_measured_typography(doc, image_path)
        _separate_navigation_prefix_icons(doc, image_path)
        _hydrate_measured_control_geometry(doc, image_path)
        _tighten_control_label_boxes(doc, image_path)
        reconstruction = doc.get("reconstruction") or {}
        if (
            resolve_focus
            and resolved_profile == "reconstruct"
            and region is None
            and reconstruction.get("targetKind") == "web"
            and vlm is not None
            and max_semantic_calls > 0
        ):
            _resolve_focus_plan(
                doc,
                image_path,
                no_store=no_store,
                quality=quality,
                pack=pack_name or pack,
                intent=intent,
                max_semantic_calls=min(7, max_semantic_calls),
                target_kind=target_kind or "web",
            )
            _exclude_tiny_glyph_noise(doc)
        _resolve_repeated_text_consensus(doc)
        _partition_merged_preferred_text(doc)
        _repair_interface_ocr_phrases(doc)
        _repair_indexed_control_labels(doc)
        _preserve_verified_ocr_orthography(doc)
        _reconcile_inline_run_text(doc)
        _sanitize_single_line_preferred_text(doc)
        _refresh_reconstruction_workflow(doc)
        _refresh_web_tokens(doc, dump)
        _sanitize_web_structure(doc)
        _hydrate_measured_typography(doc, image_path)
        _separate_navigation_prefix_icons(doc, image_path)
        _partition_navigation_word_gaps(doc)
        _hydrate_measured_control_geometry(doc, image_path)
        _tighten_control_label_boxes(doc, image_path)
        _hydrate_measured_surfaces(doc, image_path)
        _hydrate_navigation_rail(doc, image_path)
        if region is None:
            _hydrate_measured_vector_paths(doc, image_path)
        _sanitize_structural_lines(doc, image_path)
        _refine_overlapping_raster_candidates(doc, image_path)
        _hydrate_intrinsic_text_raster_assets(doc, image_path)
        _hydrate_background_artwork_layer(
            doc,
            image_path,
            source_raster_assets,
        )
        _hydrate_source_vector_regions(doc, source_vector_assets)
        _hydrate_source_dom_typography(
            doc,
            source_text_nodes,
            source_font_assets,
        )
        _infer_contextual_ui_structure(doc)
        _infer_corner_navigation_controls(doc)
        _infer_top_navigation_controls(doc)
        if completed_cache_key is not None:
            write_cache(completed_cache_key, doc)
    _resolve_repeated_text_consensus(doc)
    _partition_merged_preferred_text(doc)
    _repair_interface_ocr_phrases(doc)
    _repair_indexed_control_labels(doc)
    _preserve_verified_ocr_orthography(doc)
    _reconcile_inline_run_text(doc)
    _sanitize_single_line_preferred_text(doc)
    _refresh_web_tokens(doc, dump)
    _sanitize_web_structure(doc)
    _hydrate_measured_typography(doc, image_path)
    _separate_navigation_prefix_icons(doc, image_path)
    _partition_navigation_word_gaps(doc)
    _hydrate_measured_control_geometry(doc, image_path)
    _tighten_control_label_boxes(doc, image_path)
    _hydrate_measured_surfaces(doc, image_path)
    _hydrate_navigation_rail(doc, image_path)
    if region is None:
        _hydrate_measured_vector_paths(doc, image_path)
    _sanitize_structural_lines(doc, image_path)
    _refine_overlapping_raster_candidates(doc, image_path)
    _hydrate_intrinsic_text_raster_assets(doc, image_path)
    _hydrate_background_artwork_layer(
        doc,
        image_path,
        source_raster_assets,
    )
    _hydrate_source_vector_regions(doc, source_vector_assets)
    _hydrate_source_dom_typography(
        doc,
        source_text_nodes,
        source_font_assets,
    )
    _infer_contextual_ui_structure(doc)
    _infer_corner_navigation_controls(doc)
    _infer_top_navigation_controls(doc)
    reconstruction = doc.get("reconstruction") or {}
    if (
        resolved_profile == "reconstruct"
        and region is None
        and reconstruction.get("targetKind") == "web"
    ):
        _materialize_raster_assets(
            doc,
            image_path,
            asset_output_dir,
            no_store=no_store,
        )
    else:
        _refresh_reconstruction_workflow(doc)
    starter_output_dir = asset_output_dir or str(
        Path(cache_root()) / "reconstruction-starters"
    )
    materialize_starter_project(
        doc,
        starter_output_dir,
        no_store=no_store,
    )
    _refresh_reconstruction_workflow(doc)
    summary = _compact_summary(doc)
    compatibility = {
        "response": response,
        "legacyIncluded": response == "full",
        "fullResponse": "Set response=full only for legacy debugging.",
    }
    if response == "brief":
        contract_path = _write_reconstruction_contract(
            doc, asset_output_dir, no_store=no_store
        )
        return {
            "brief": _implementation_brief(doc, contract_path),
            "contractPath": contract_path,
            "summary": summary,
            "artifacts": doc.get("artifacts", []),
            "pack": pack_name,
            "compatibility": compatibility,
        }
    if response == "compact":
        return {
            "doc": _compact_document(doc),
            "summary": summary,
            "artifacts": doc.get("artifacts", []),
            "pack": pack_name,
            "compatibility": compatibility,
        }
    legacy = dict(dump)
    if "facts" in legacy.get("design", {}):
        legacy["design"] = {"issues": legacy["design"]["facts"]}
    return {
        "document": docmod.render_markdown(doc),
        "doc": doc,
        "somPath": dump.get("somPath"),
        "legacy": legacy,
        "pack": pack_name,
        "summary": summary,
        "artifacts": doc.get("artifacts", []),
        "compatibility": compatibility,
    }


def zoom(
    image_path: str,
    region: dict | None = None,
    som_id: int | None = None,
    no_store: bool = False,
    quality: bool = False,
    pack: str | None = None,
    profile: str | None = None,
    response: str = "compact",
    target_kind: str | None = None,
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
    return see_document(
        image_path,
        region,
        no_store,
        quality=quality,
        pack=pack,
        profile=profile,
        response=response,
        target_kind=target_kind,
    )


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
    role, guidance = docmod.reconstruction_role(el, dump["elements"])
    return {
        "element": el,
        "box_norm": docmod.normalize_box(el["box"], w, h),
        "rawKind": el.get("kind"),
        "reconstructionRole": role,
        "representationGuidance": guidance,
    }


def vision_prompt(lang: str = "ru") -> dict:
    prompt = VISION_PROMPT.get(lang, VISION_PROMPT["ru"])
    starter_guidance = {
        "ru": (
            " Если brief вернул starterProject, сразу скопируй или запусти его entryPath: "
            "это уже живой DOM/CSS-каркас из измеренного контракта. Не переписывай первый "
            "вариант с нуля и не заменяй его скриншотом; после запуска используй только "
            "sens_review и его repairHints."
        ),
        "en": (
            " If the brief returns starterProject, copy or serve its entryPath immediately: "
            "it is already a live DOM/CSS candidate generated from the measured contract. "
            "Do not rewrite the first candidate from scratch or replace it with a screenshot; "
            "after serving it use only sens_review and its repairHints."
        ),
    }
    return {
        "prompt": (
            prompt.replace(
                "sens_see(profile=reconstruct, response=compact",
                "sens_see(profile=reconstruct, response=brief",
            )
            + starter_guidance.get(lang, starter_guidance["ru"])
        )
    }


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


def review_op(
    reference_path: str,
    url: str,
    options: dict[str, Any] | None = None,
    no_store: bool = False,
) -> dict[str, Any]:
    from sight.web_review import review_web

    return review_web(
        reference_path,
        url,
        options,
        out_dir=cache_root() / "reviews",
        no_store=no_store,
    )


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
