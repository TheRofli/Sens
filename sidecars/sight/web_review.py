"""Combined visual and DOM-integrity review for screenshot-to-web work."""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from sight.capture import capture_url
from sight.compare import compare_images
from sight.ocr import load_cv


def _box_area(box: list[int | float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(
        0.0, float(box[3]) - float(box[1])
    )


def _intersection_area(
    first: list[int | float], second: list[int | float]
) -> float:
    return max(
        0.0, min(float(first[2]), float(second[2])) - max(float(first[0]), float(second[0]))
    ) * max(
        0.0, min(float(first[3]), float(second[3])) - max(float(first[1]), float(second[1]))
    )


def _coverage(subject: list[int | float], covering: list[int | float]) -> float:
    return _intersection_area(subject, covering) / max(1.0, _box_area(subject))


def _normalized_text(value: Any) -> str:
    return "".join(character for character in str(value or "").casefold() if character.isalnum())


def _element_identifier(value: Any) -> int | str:
    """Preserve stable string IDs while keeping legacy numeric IDs numeric."""
    if value is None or value == "":
        return -1
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return str(value)


def _canonical_symbol_text(value: Any) -> str:
    return "\n".join(
        line.rstrip()
        for line in str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip("\n").split("\n")
    )


def _source_box(
    box: list[int | float],
    *,
    canvas_width: int,
    canvas_height: int,
    viewport_width: int,
    viewport_height: int,
) -> list[int]:
    return [
        round(float(box[0]) * canvas_width / max(1, viewport_width)),
        round(float(box[1]) * canvas_height / max(1, viewport_height)),
        round(float(box[2]) * canvas_width / max(1, viewport_width)),
        round(float(box[3]) * canvas_height / max(1, viewport_height)),
    ]


def _blocking_reason(code: str, detail: str, **evidence: Any) -> dict[str, Any]:
    return {
        "code": code,
        "detail": detail,
        "epistemic": "measured",
        "evidence": evidence,
    }


def _geometry_delta(
    reference_box: list[int | float], candidate_box: list[int | float]
) -> dict[str, int]:
    """Return candidate-minus-reference geometry in source pixels."""
    reference_width = float(reference_box[2]) - float(reference_box[0])
    reference_height = float(reference_box[3]) - float(reference_box[1])
    candidate_width = float(candidate_box[2]) - float(candidate_box[0])
    candidate_height = float(candidate_box[3]) - float(candidate_box[1])
    return {
        "x": round(float(candidate_box[0]) - float(reference_box[0])),
        "y": round(float(candidate_box[1]) - float(reference_box[1])),
        "width": round(candidate_width - reference_width),
        "height": round(candidate_height - reference_height),
    }


def _geometry_error(delta: dict[str, int]) -> int:
    return sum(abs(int(delta[key])) for key in ("x", "y", "width", "height"))


def _compact_style(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    keys = (
        "fontFamily",
        "fontSize",
        "fontWeight",
        "fontStyle",
        "lineHeight",
        "letterSpacing",
        "textTransform",
        "textAlign",
        "color",
        "background",
        "border",
        "borderRadius",
        "padding",
    )
    return {key: value[key] for key in keys if value.get(key) not in (None, "")}


def _reference_typography(reference: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "fontSize",
        "capHeight",
        "avgGlyphWidth",
        "widthEm",
        "fontFamily",
        "fontFamilyStatus",
        "fontFamilyCandidate",
        "fontFamilyDistance",
    )
    return {key: reference[key] for key in keys if reference.get(key) is not None}


def _ranked_hints(hints: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in hint.items() if key != "_priority"}
        for hint in sorted(
            hints,
            key=lambda hint: (-float(hint.get("_priority") or 0), str(hint.get("kind") or "")),
        )[:limit]
    ]


def _css_pixels(value: Any) -> float | None:
    text = str(value or "").strip().casefold()
    if not text.endswith("px"):
        return None
    try:
        return float(text[:-2])
    except ValueError:
        return None


def evaluate_web_integrity(
    reconstruction: dict[str, Any], capture: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate whether a visual candidate is a real, usable web document.

    Pixel similarity is intentionally not accepted here.  This function checks
    the representation chosen by the implementation: live/selectable text,
    semantic controls, and raster use limited to measured graphic regions.
    """
    canvas = reconstruction.get("canvas") or {}
    canvas_width = int(canvas.get("width") or 1)
    canvas_height = int(canvas.get("height") or 1)
    viewport = (capture.get("settings") or {}).get("viewport") or {}
    viewport_width = int(viewport.get("width") or canvas_width)
    viewport_height = int(viewport.get("height") or canvas_height)

    def project(box: list[int | float]) -> list[int]:
        return _source_box(
            box,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
        )

    live_nodes = [
        {**node, "boxSource": project(node["box"])}
        for node in capture.get("textNodes", [])
        if node.get("visible", True)
        and node.get("text")
        and isinstance(node.get("box"), list)
        and len(node["box"]) == 4
    ]
    missing_text: list[int | str] = []
    content_mismatch_text: list[int | str] = []
    unselectable_text: list[int | str] = []
    text_matches: list[dict[str, Any]] = []
    text_repair_hints: list[dict[str, Any]] = []
    consumed_text_nodes: set[int] = set()
    for reference in reconstruction.get("text", []):
        reference_box = reference.get("boxSource")
        if not isinstance(reference_box, list) or len(reference_box) != 4:
            continue
        reference_text = _normalized_text(reference.get("preferredValue") or reference.get("value"))
        ranked = []
        for index, node in enumerate(live_nodes):
            if index in consumed_text_nodes:
                continue
            overlap = _coverage(reference_box, node["boxSource"])
            if overlap < 0.12:
                continue
            candidate_coverage = _coverage(node["boxSource"], reference_box)
            mutual_overlap = min(overlap, candidate_coverage)
            node_text = _normalized_text(node.get("text"))
            text_similarity = (
                SequenceMatcher(None, reference_text, node_text).ratio()
                if reference_text and node_text
                else 0.0
            )
            match_score = 0.7 * text_similarity + 0.3 * mutual_overlap
            ranked.append(
                (match_score, overlap, text_similarity, index, node)
            )
        if not ranked:
            element_id = _element_identifier(reference.get("elementId"))
            missing_text.append(element_id)
            lexical = []
            for index, node in enumerate(live_nodes):
                node_text = _normalized_text(node.get("text"))
                similarity = (
                    SequenceMatcher(None, reference_text, node_text).ratio()
                    if reference_text and node_text
                    else 0.0
                )
                if similarity >= 0.72:
                    lexical.append((similarity, index, node))
            hint: dict[str, Any] = {
                "kind": "live-text-missing-or-displaced",
                "referenceElementId": element_id,
                "referenceText": reference.get("preferredValue") or reference.get("value"),
                "referenceBoxSource": reference_box,
                "referenceTypography": _reference_typography(reference),
                "action": "Render this value as live selectable DOM text at referenceBoxSource.",
                "_priority": 1_000_000,
            }
            if lexical:
                similarity, node_index, node = max(lexical, key=lambda item: item[0])
                delta = _geometry_delta(reference_box, node["boxSource"])
                hint.update(
                    {
                        "candidateTextNodeIndex": node_index,
                        "candidateText": node.get("text"),
                        "candidateBoxSource": node["boxSource"],
                        "geometryDelta": delta,
                        "candidateStyle": _compact_style(node.get("style")),
                        "textSimilarity": round(similarity, 4),
                        "_priority": 1_000_000 + _geometry_error(delta),
                    }
                )
            text_repair_hints.append(hint)
            continue
        _score, overlap, text_similarity, node_index, node = max(
            ranked, key=lambda item: item[0]
        )
        element_id = _element_identifier(reference.get("elementId"))
        if reference_text and text_similarity < 0.72:
            content_mismatch_text.append(element_id)
            text_repair_hints.append(
                {
                    "kind": "live-text-content-mismatch",
                    "referenceElementId": element_id,
                    "referenceText": reference.get("preferredValue")
                    or reference.get("value"),
                    "candidateText": node.get("text"),
                    "referenceBoxSource": reference_box,
                    "candidateBoxSource": node["boxSource"],
                    "textSimilarity": round(text_similarity, 4),
                    "action": "Replace the overlapping DOM copy with the exact reference value.",
                    "_priority": 1_000_000 + (1.0 - text_similarity) * 1000,
                }
            )
            continue
        consumed_text_nodes.add(node_index)
        selectable = str(node.get("userSelect") or "auto").casefold() != "none"
        if not selectable:
            unselectable_text.append(element_id)
        delta = _geometry_delta(reference_box, node["boxSource"])
        reference_typography = _reference_typography(reference)
        candidate_style = _compact_style(node.get("style"))
        text_matches.append(
            {
                "referenceElementId": element_id,
                "textNodeIndex": node_index,
                "coverage": round(overlap, 4),
                "textSimilarity": round(text_similarity, 4),
                "selectable": selectable,
                "geometryDelta": delta,
                "source": "observed-dom",
            }
        )
        reference_font_size = reference_typography.get("fontSize")
        candidate_font_size = _css_pixels(candidate_style.get("fontSize"))
        font_size_delta = (
            round(candidate_font_size - float(reference_font_size), 1)
            if isinstance(reference_font_size, (int, float))
            and candidate_font_size is not None
            else None
        )
        error = _geometry_error(delta)
        if error >= 2 or (font_size_delta is not None and abs(font_size_delta) >= 1):
            text_repair_hints.append(
                {
                    "kind": "text-geometry-or-typography",
                    "referenceElementId": element_id,
                    "referenceText": reference.get("preferredValue") or reference.get("value"),
                    "candidateText": node.get("text"),
                    "referenceBoxSource": reference_box,
                    "candidateBoxSource": node["boxSource"],
                    "geometryDelta": delta,
                    "fontSizeDeltaPx": font_size_delta,
                    "referenceTypography": reference_typography,
                    "candidateStyle": candidate_style,
                    "action": "Adjust this DOM node's CSS so candidateBoxSource and typography match the reference.",
                    "_priority": error + abs(font_size_delta or 0),
                }
            )

    missing_symbol_art: list[int] = []
    unselectable_symbol_art: list[int] = []
    non_preformatted_symbol_art: list[int] = []
    symbol_art_matches: list[dict[str, Any]] = []
    for index, reference in enumerate(reconstruction.get("symbolArt", [])):
        reference_box = reference.get("boxSource")
        if not isinstance(reference_box, list) or len(reference_box) != 4:
            continue
        expected = _canonical_symbol_text(reference.get("text"))
        candidates = []
        for node_index, node in enumerate(live_nodes):
            overlap = _coverage(reference_box, node["boxSource"])
            actual = _canonical_symbol_text(node.get("rawText") or node.get("text"))
            if overlap >= 0.12 and actual == expected:
                candidates.append((overlap, node_index, node))
        if not candidates:
            missing_symbol_art.append(index)
            continue
        overlap, node_index, node = max(candidates, key=lambda item: item[0])
        selectable = str(node.get("userSelect") or "auto").casefold() != "none"
        white_space = str(node.get("whiteSpace") or "normal").casefold()
        preformatted = white_space in {"pre", "pre-wrap", "break-spaces"}
        if not selectable:
            unselectable_symbol_art.append(index)
        if not preformatted:
            non_preformatted_symbol_art.append(index)
        symbol_art_matches.append(
            {
                "referenceIndex": index,
                "textNodeIndex": node_index,
                "coverage": round(overlap, 4),
                "exact": True,
                "selectable": selectable,
                "preformatted": preformatted,
                "source": "observed-dom",
            }
        )

    semantic_controls = [
        {**control, "boxSource": project(control["box"])}
        for control in capture.get("semanticControls", [])
        if control.get("visible", True)
        and isinstance(control.get("box"), list)
        and len(control["box"]) == 4
    ]
    missing_controls: list[int | str] = []
    label_mismatch_controls: list[int | str] = []
    control_matches: list[dict[str, Any]] = []
    control_repair_hints: list[dict[str, Any]] = []
    consumed_controls: set[int] = set()
    reference_text_by_id = {
        entry.get("elementId"): entry
        for entry in reconstruction.get("text", [])
        if entry.get("elementId") is not None
    }
    for reference in reconstruction.get("visualControlCandidates", []):
        reference_box = reference.get("boxSource")
        if not isinstance(reference_box, list) or len(reference_box) != 4:
            continue
        expected_label = " ".join(
            str(
                reference_text_by_id.get(label_id, {}).get("preferredValue")
                or reference_text_by_id.get(label_id, {}).get("value")
                or ""
            ).strip()
            for label_id in reference.get("labelElementIds") or []
        ).strip()
        expected_label_normalized = _normalized_text(expected_label)
        candidates = []
        label_mismatches = []
        for index, control in enumerate(semantic_controls):
            if index in consumed_controls:
                continue
            overlap = _coverage(reference_box, control["boxSource"])
            usable = (
                str(control.get("pointerEvents") or "auto").casefold() != "none"
                and not bool(control.get("disabled", False))
            )
            if overlap >= 0.2 and usable:
                control_name = _normalized_text(control.get("name"))
                label_similarity = (
                    SequenceMatcher(
                        None, expected_label_normalized, control_name
                    ).ratio()
                    if expected_label_normalized and control_name
                    else 1.0
                    if not expected_label_normalized
                    else 0.0
                )
                if expected_label_normalized and label_similarity < 0.65:
                    label_mismatches.append(
                        (overlap, label_similarity, index, control)
                    )
                    continue
                candidates.append(
                    (0.8 * overlap + 0.2 * label_similarity, overlap, label_similarity, index, control)
                )
        element_id = _element_identifier(reference.get("elementId"))
        if not candidates:
            if label_mismatches:
                label_mismatch_controls.append(element_id)
                overlap, similarity, _index, control = max(
                    label_mismatches, key=lambda item: item[0]
                )
                control_repair_hints.append(
                    {
                        "kind": "semantic-control-label-mismatch",
                        "referenceElementId": element_id,
                        "referenceLabel": expected_label,
                        "candidateLabel": control.get("name"),
                        "referenceBoxSource": reference_box,
                        "candidateBoxSource": control["boxSource"],
                        "textSimilarity": round(similarity, 4),
                        "action": "Use the exact visible label as the semantic control name.",
                        "_priority": 1_000_000 + (1.0 - similarity) * 1000,
                    }
                )
                continue
            missing_controls.append(element_id)
            control_repair_hints.append(
                {
                    "kind": "semantic-control-missing-or-displaced",
                    "referenceElementId": element_id,
                    "referenceBoxSource": reference_box,
                    "action": "Create a usable semantic HTML control at referenceBoxSource; do not use a raster crop.",
                    "_priority": 1_000_000,
                }
            )
            continue
        _score, overlap, label_similarity, control_index, control = max(
            candidates, key=lambda item: item[0]
        )
        consumed_controls.add(control_index)
        delta = _geometry_delta(reference_box, control["boxSource"])
        control_matches.append(
            {
                "referenceElementId": element_id,
                "controlIndex": control_index,
                "coverage": round(overlap, 4),
                "labelSimilarity": round(label_similarity, 4),
                "geometryDelta": delta,
                "source": "observed-dom",
            }
        )
        error = _geometry_error(delta)
        if error >= 2:
            control_repair_hints.append(
                {
                    "kind": "semantic-control-geometry",
                    "referenceElementId": element_id,
                    "referenceBoxSource": reference_box,
                    "candidateBoxSource": control["boxSource"],
                    "geometryDelta": delta,
                    "candidateStyle": _compact_style(control.get("style")),
                    "action": "Adjust the semantic control's CSS box to match referenceBoxSource.",
                    "_priority": error,
                }
            )

    candidate_lines = [
        {**line, "boxSource": project(line["box"])}
        for line in capture.get("structuralLines", [])
        if line.get("orientation") in {"horizontal", "vertical"}
        and isinstance(line.get("box"), list)
        and len(line["box"]) == 4
    ]
    line_matches: list[dict[str, Any]] = []
    missing_lines: list[int] = []
    structure_repair_hints: list[dict[str, Any]] = []
    consumed_lines: set[int] = set()
    for reference_index, reference in enumerate(reconstruction.get("structuralLines", [])):
        reference_box = reference.get("boxSource")
        orientation = reference.get("orientation")
        if (
            orientation not in {"horizontal", "vertical"}
            or not isinstance(reference_box, list)
            or len(reference_box) != 4
        ):
            continue
        ref_length = (
            float(reference_box[2]) - float(reference_box[0])
            if orientation == "horizontal"
            else float(reference_box[3]) - float(reference_box[1])
        )
        ranked_lines = []
        for candidate_index, candidate in enumerate(candidate_lines):
            if candidate_index in consumed_lines or candidate.get("orientation") != orientation:
                continue
            candidate_box = candidate["boxSource"]
            candidate_length = (
                float(candidate_box[2]) - float(candidate_box[0])
                if orientation == "horizontal"
                else float(candidate_box[3]) - float(candidate_box[1])
            )
            orthogonal_distance = (
                abs(float(candidate_box[1]) - float(reference_box[1]))
                if orientation == "horizontal"
                else abs(float(candidate_box[0]) - float(reference_box[0]))
            )
            length_distance = abs(candidate_length - ref_length)
            start_distance = (
                abs(float(candidate_box[0]) - float(reference_box[0]))
                if orientation == "horizontal"
                else abs(float(candidate_box[1]) - float(reference_box[1]))
            )
            score = orthogonal_distance + 0.25 * length_distance + 0.1 * start_distance
            ranked_lines.append((score, candidate_index, candidate, orthogonal_distance))
        if not ranked_lines:
            missing_lines.append(reference_index)
            structure_repair_hints.append(
                {
                    "kind": "structural-line-missing",
                    "referenceIndex": reference_index,
                    "orientation": orientation,
                    "referenceBoxSource": reference_box,
                    "referenceColor": reference.get("color"),
                    "action": "Create this divider as independent CSS or vector structure.",
                    "_priority": 1_000_000,
                }
            )
            continue
        _score, candidate_index, candidate, orthogonal_distance = min(
            ranked_lines, key=lambda item: item[0]
        )
        candidate_box = candidate["boxSource"]
        delta = _geometry_delta(reference_box, candidate_box)
        candidate_length = (
            float(candidate_box[2]) - float(candidate_box[0])
            if orientation == "horizontal"
            else float(candidate_box[3]) - float(candidate_box[1])
        )
        plausible = orthogonal_distance <= max(12.0, float(reference.get("thickness") or 1) * 4)
        plausible = plausible and candidate_length >= max(8.0, ref_length * 0.5)
        if plausible:
            consumed_lines.add(candidate_index)
            line_matches.append(
                {
                    "referenceIndex": reference_index,
                    "candidateIndex": candidate_index,
                    "geometryDelta": delta,
                    "source": "measured-dom-css",
                }
            )
        else:
            missing_lines.append(reference_index)
        error = _geometry_error(delta)
        if not plausible or error >= 2:
            structure_repair_hints.append(
                {
                    "kind": "structural-line-geometry" if plausible else "structural-line-missing-or-displaced",
                    "referenceIndex": reference_index,
                    "orientation": orientation,
                    "referenceBoxSource": reference_box,
                    "candidateBoxSource": candidate_box,
                    "geometryDelta": delta,
                    "referenceColor": reference.get("color"),
                    "candidateColor": candidate.get("color"),
                    "candidateSource": candidate.get("source"),
                    "action": "Adjust or create an independent CSS/vector divider to match referenceBoxSource.",
                    "_priority": (1_000_000 if not plausible else 0) + error,
                }
            )

    reference_text_boxes = [
        entry["boxSource"]
        for entry in reconstruction.get("text", [])
        if isinstance(entry.get("boxSource"), list) and len(entry["boxSource"]) == 4
    ]
    reference_symbol_art_boxes = [
        entry["boxSource"]
        for entry in reconstruction.get("symbolArt", [])
        if isinstance(entry.get("boxSource"), list) and len(entry["boxSource"]) == 4
    ]
    allowed_entries = [
        entry
        for entry in reconstruction.get("allowedRasterRegions", [])
        if isinstance(entry.get("boxSource"), list) and len(entry["boxSource"]) == 4
    ]
    structural_lines = [
        entry["boxSource"]
        for entry in reconstruction.get("structuralLines", [])
        if isinstance(entry.get("boxSource"), list) and len(entry["boxSource"]) == 4
    ]
    raster_kinds = {"img", "image", "canvas", "video", "picture", "background-image"}
    raster_entries = []
    raster_text_indexes: list[int] = []
    raster_symbol_art_indexes: list[int] = []
    raster_structure_indexes: list[int] = []
    outside_allowed_indexes: list[int] = []
    full_reference_indexes: list[int] = []
    untrusted_background_indexes: list[int] = []
    allowed_count = 0
    canvas_area = max(1.0, float(canvas_width * canvas_height))
    for index, raster in enumerate(capture.get("rasterElements", [])):
        if not raster.get("visible", True) or raster.get("kind") not in raster_kinds:
            continue
        raster_source = raster.get("src")
        if (
            raster.get("kind") == "background-image"
            and isinstance(raster_source, str)
            and "url(" not in raster_source.lower()
            and "image-set(" not in raster_source.lower()
        ):
            continue
        box = raster.get("box")
        if not isinstance(box, list) or len(box) != 4:
            continue
        source_box = project(box)
        area_ratio = _box_area(source_box) / canvas_area
        matching_allowed = [
            entry
            for entry in allowed_entries
            if _coverage(source_box, entry["boxSource"]) >= 0.72
        ]
        declared_role = raster.get("sensRasterRole")
        role_matching_allowed = [
            entry
            for entry in matching_allowed
            if declared_role and entry.get("kind") == declared_role
        ]
        matched_allowed = next(
            iter(role_matching_allowed or matching_allowed),
            None,
        )
        background_policy = (
            matched_allowed.get("protectionPolicy")
            if isinstance(matched_allowed, dict)
            else None
        )
        preserves_interface_pixels = bool(
            isinstance(background_policy, dict)
            and any(
                "preserved-in-background" in str(value).casefold()
                for value in background_policy.values()
            )
        )
        alpha_background_candidate = bool(
            matched_allowed
            and matched_allowed.get("kind") == "alpha-masked-background-artwork"
            and raster.get("sensRasterRole") == "alpha-masked-background-artwork"
        )
        browser_source_candidate = bool(
            matched_allowed
            and matched_allowed.get("kind")
            == "browser-source-background-artwork"
            and raster.get("sensRasterRole")
            == "browser-source-background-artwork"
        )
        background_candidate = bool(
            alpha_background_candidate or browser_source_candidate
        )
        semantic_residual_protection = (
            matched_allowed.get("semanticResidualProtection")
            if isinstance(matched_allowed, dict)
            else None
        )
        trusted_alpha_background = bool(
            alpha_background_candidate
            and matched_allowed.get("alphaProtected") is True
            and matched_allowed.get("semanticContentRemoved") is True
            and int(matched_allowed.get("protectionVersion") or 0) >= 3
            and isinstance(semantic_residual_protection, dict)
            and semantic_residual_protection.get(
                "displayTextDiscoveryComplete"
            )
            is True
            and isinstance(background_policy, dict)
            and background_policy.get("backgroundOnly") is True
            and not preserves_interface_pixels
            and raster.get("sensArtifactId") == matched_allowed.get("artifactId")
        )
        candidate_source_asset = next(
            (
                asset
                for asset in capture.get("sourceRasterAssets", [])
                if isinstance(asset, dict)
                and asset.get("rasterIndex") == index
            ),
            None,
        )
        contract_content_sha256 = str(
            matched_allowed.get("contentSha256")
            if isinstance(matched_allowed, dict)
            else ""
        ).lower()
        valid_contract_hash = bool(
            len(contract_content_sha256) == 64
            and all(
                character in "0123456789abcdef"
                for character in contract_content_sha256
            )
        )
        trusted_browser_source_background = bool(
            browser_source_candidate
            and matched_allowed.get("semanticContentRemoved") is True
            and int(matched_allowed.get("protectionVersion") or 0) >= 4
            and matched_allowed.get("source") == "observed"
            and matched_allowed.get("method")
            == "verified-playwright-response-body"
            and isinstance(background_policy, dict)
            and background_policy.get("backgroundOnly") is True
            and background_policy.get("liveText")
            == "separate-observed-live-dom"
            and background_policy.get("controlDecoration")
            == "separate-semantic-css"
            and background_policy.get("fullReferenceScreenshot") is False
            and not preserves_interface_pixels
            and valid_contract_hash
            and raster.get("sensArtifactId")
            == f"raster:{contract_content_sha256[:16]}"
            and raster.get("sensArtifactId") == matched_allowed.get("artifactId")
            and isinstance(candidate_source_asset, dict)
            and str(candidate_source_asset.get("sha256") or "").lower()
            == contract_content_sha256
            and str(candidate_source_asset.get("mediaType") or "").lower()
            == str(matched_allowed.get("mediaType") or "").lower()
            and candidate_source_asset.get("source") == "observed"
            and candidate_source_asset.get("method")
            == "playwright-response-body"
        )
        trusted_background = bool(
            trusted_alpha_background or trusted_browser_source_background
        )
        if background_candidate and not trusted_background:
            untrusted_background_indexes.append(index)
        allowed = matched_allowed is not None
        overlaps_text = not trusted_background and any(
            _coverage(text_box, source_box) >= 0.12 for text_box in reference_text_boxes
        )
        overlaps_symbol_art = not trusted_background and any(
            _coverage(symbol_box, source_box) >= 0.12
            for symbol_box in reference_symbol_art_boxes
        )
        overlaps_structure = not trusted_background and any(
            _coverage(line_box, source_box) >= 0.5 for line_box in structural_lines
        )
        if allowed and not overlaps_text and not overlaps_symbol_art and not overlaps_structure:
            allowed_count += 1
        if overlaps_text:
            raster_text_indexes.append(index)
        if overlaps_symbol_art:
            raster_symbol_art_indexes.append(index)
        if overlaps_structure:
            raster_structure_indexes.append(index)
        if area_ratio >= 0.45 and not trusted_background:
            full_reference_indexes.append(index)
        if not allowed and area_ratio >= 0.0005:
            outside_allowed_indexes.append(index)
        raster_entries.append(
            {
                "index": index,
                "kind": raster.get("kind"),
                "boxSource": source_box,
                "areaRatio": round(area_ratio, 5),
                "allowed": allowed
                and not overlaps_text
                and not overlaps_symbol_art
                and not overlaps_structure,
                "overlapsText": overlaps_text,
                "overlapsSymbolArt": overlaps_symbol_art,
                "overlapsStructure": overlaps_structure,
                "alphaMaskedBackground": trusted_alpha_background,
                "browserSourceBackground": trusted_browser_source_background,
                "backgroundTrustFailure": background_candidate
                and not trusted_background,
                "source": "observed-dom",
            }
        )

    blocking_reasons: list[dict[str, Any]] = []
    if missing_text:
        blocking_reasons.append(
            _blocking_reason(
                "missing-live-text",
                "Reference text regions are not covered by visible DOM text nodes.",
                elementIds=missing_text,
            )
        )
    if unselectable_text:
        blocking_reasons.append(
            _blocking_reason(
                "unselectable-live-text",
                "Visible DOM text is disabled from normal text selection.",
                elementIds=unselectable_text,
            )
        )
    if content_mismatch_text:
        blocking_reasons.append(
            _blocking_reason(
                "live-text-content-mismatch",
                "Visible DOM text overlaps the reference region but contains different copy.",
                elementIds=content_mismatch_text,
            )
        )
    if missing_symbol_art:
        blocking_reasons.append(
            _blocking_reason(
                "missing-live-symbol-art",
                "Detected character artwork is not present as exact live preformatted text.",
                symbolArtIndexes=missing_symbol_art,
            )
        )
    if unselectable_symbol_art:
        blocking_reasons.append(
            _blocking_reason(
                "unselectable-symbol-art",
                "Character artwork exists in the DOM but normal text selection is disabled.",
                symbolArtIndexes=unselectable_symbol_art,
            )
        )
    if non_preformatted_symbol_art:
        blocking_reasons.append(
            _blocking_reason(
                "non-preformatted-symbol-art",
                "Character artwork does not preserve spaces and line breaks with preformatted CSS.",
                symbolArtIndexes=non_preformatted_symbol_art,
            )
        )
    if missing_controls:
        blocking_reasons.append(
            _blocking_reason(
                "missing-semantic-control",
                "Measured labeled controls are not represented by usable semantic HTML controls.",
                elementIds=missing_controls,
            )
        )
    if label_mismatch_controls:
        blocking_reasons.append(
            _blocking_reason(
                "semantic-control-label-mismatch",
                "A semantic control occupies the right region but its accessible name does not match the visible label.",
                elementIds=label_mismatch_controls,
            )
        )
    if raster_text_indexes:
        blocking_reasons.append(
            _blocking_reason(
                "raster-overlaps-text",
                "Raster elements cover regions that must be live text.",
                rasterIndexes=raster_text_indexes,
            )
        )
    if raster_symbol_art_indexes:
        blocking_reasons.append(
            _blocking_reason(
                "raster-overlaps-symbol-art",
                "Raster elements cover character artwork that must remain live text.",
                rasterIndexes=raster_symbol_art_indexes,
            )
        )
    if raster_structure_indexes:
        blocking_reasons.append(
            _blocking_reason(
                "raster-layout-structure",
                "Raster elements cover measured layout structure.",
                rasterIndexes=raster_structure_indexes,
            )
        )
    if outside_allowed_indexes:
        blocking_reasons.append(
            _blocking_reason(
                "raster-outside-allowed-region",
                "Raster elements extend beyond measured illustration/photo/logo regions.",
                rasterIndexes=outside_allowed_indexes,
            )
        )
    if full_reference_indexes:
        blocking_reasons.append(
            _blocking_reason(
                "full-reference-raster",
                "A raster element covers most of the reconstruction canvas.",
                rasterIndexes=full_reference_indexes,
            )
        )
    if untrusted_background_indexes:
        blocking_reasons.append(
            _blocking_reason(
                "untrusted-background-raster",
                "A full-canvas background raster has not proved that live text, controls, surfaces, lines, paths, icons, badges, symbol art, and foreground objects were removed before materialization.",
                rasterIndexes=untrusted_background_indexes,
            )
        )

    text_count = len(reconstruction.get("text", []))
    selectable_count = len(text_matches) - len(unselectable_text)
    result = {
        "webPass": not blocking_reasons,
        "textCoverage": {
            "referenceCount": text_count,
            "liveCount": len(text_matches),
            "selectableCount": selectable_count,
            "missingElementIds": missing_text,
            "contentMismatchElementIds": content_mismatch_text,
            "unselectableElementIds": unselectable_text,
        },
        "textMatches": text_matches,
        "symbolArtCoverage": {
            "referenceCount": len(reconstruction.get("symbolArt", [])),
            "exactSelectableCount": sum(
                match["exact"] and match["selectable"] and match["preformatted"]
                for match in symbol_art_matches
            ),
            "missingIndexes": missing_symbol_art,
            "unselectableIndexes": unselectable_symbol_art,
            "nonPreformattedIndexes": non_preformatted_symbol_art,
        },
        "symbolArtMatches": symbol_art_matches,
        "controlCoverage": {
            "referenceCount": len(reconstruction.get("visualControlCandidates", [])),
            "semanticCount": len(control_matches),
            "missingElementIds": missing_controls,
            "labelMismatchElementIds": label_mismatch_controls,
        },
        "controlMatches": control_matches,
        "structuralLineCoverage": {
            "referenceCount": len(reconstruction.get("structuralLines", [])),
            "matchedCount": len(line_matches),
            "missingIndexes": missing_lines,
        },
        "structuralLineMatches": line_matches,
        "repairHints": {
            "source": "measured-reference-vs-observed-dom-css",
            "coordinateSpace": "source-pixels",
            "text": _ranked_hints(text_repair_hints, 8),
            "controls": _ranked_hints(control_repair_hints, 4),
            "structure": _ranked_hints(structure_repair_hints, 4),
        },
        "rasterAudit": {
            "observedCount": len(raster_entries),
            "allowedCount": allowed_count,
            "elements": raster_entries,
        },
        "blockingReasons": blocking_reasons,
        "observed": {
            "domTextNodeCount": len(live_nodes),
            "semanticControlCount": len(semantic_controls),
            "rasterElementCount": len(raster_entries),
        },
        "measured": {
            "canvas": [canvas_width, canvas_height],
            "viewport": [viewport_width, viewport_height],
        },
        "inferred": [],
    }
    return result


def _ocr_text_repair_hint(
    visual_result: dict[str, Any],
    web_result: dict[str, Any],
    reconstruction: dict[str, Any] | None,
) -> dict[str, Any] | None:
    checks = (visual_result.get("acceptance") or {}).get("checks") or []
    failed = next(
        (
            check
            for check in checks
            if check.get("name") == "text_similarity_minimum"
            and not check.get("passed")
        ),
        None,
    )
    if not failed:
        return None

    text_rows = _decode_compact_table((reconstruction or {}).get("text"))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in text_rows:
        key = _normalized_text(row.get("preferredValue") or row.get("value"))
        if key:
            grouped.setdefault(key, []).append(row)
    repeated_groups = []
    for rows in sorted(grouped.values(), key=len, reverse=True):
        if len(rows) < 2:
            continue
        repeated_groups.append(
            {
                "text": str(rows[0].get("preferredValue") or rows[0].get("value") or ""),
                "elementIds": [
                    _element_identifier(row.get("elementId")) for row in rows[:12]
                ],
                "boxesSource": [row.get("boxSource") for row in rows[:12]],
                "count": len(rows),
            }
        )
        if len(repeated_groups) >= 4:
            break

    text_matches = {
        _element_identifier(match.get("referenceElementId")): match
        for match in web_result.get("textMatches") or []
    }
    verified_vectors = []
    for row in text_rows:
        if row.get("visualRepresentation") != (
            "source-vector-wordmark-with-selectable-live-label"
        ):
            continue
        element_id = _element_identifier(row.get("elementId"))
        match = text_matches.get(element_id) or {}
        if not match.get("exact") or not match.get("selectable"):
            continue
        verified_vectors.append(
            {
                "elementId": element_id,
                "text": str(row.get("preferredValue") or row.get("value") or ""),
                "boxSource": row.get("boxSource"),
            }
        )

    text_metrics = (visual_result.get("metrics") or {}).get("text") or {}
    return {
        "kind": "ocr-text-similarity",
        "actual": failed.get("actual", text_metrics.get("similarity")),
        "threshold": failed.get("threshold"),
        "referenceOcr": str(text_metrics.get("reference") or "")[:1200],
        "candidateOcr": str(text_metrics.get("candidate") or "")[:1200],
        "repeatedReferenceTextGroups": repeated_groups,
        "verifiedVectorWordmarks": verified_vectors,
        "action": (
            "Repair the visible rendering and spacing of repeated/live text groups first, "
            "using their contract boxes and DOM text hints. Compare the returned referenceOcr "
            "and candidateOcr strings to identify missing or garbled repetitions. Do not "
            "replace verified source-vector wordmarks solely to improve OCR; their exact "
            "selectable label and source vectors are already verified. Then call sens_review again."
        ),
        "source": "measured-rapidocr-plus-live-dom-contract",
    }


def combine_review(
    visual_result: dict[str, Any],
    web_result: dict[str, Any],
    reconstruction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    visual_pass = bool(
        visual_result.get("canComplete")
        or visual_result.get("verdict") == "pass"
    )
    web_pass = bool(web_result.get("webPass"))
    blocking_reasons = list(web_result.get("blockingReasons", []))
    if not visual_pass:
        blocking_reasons.insert(
            0,
            _blocking_reason(
                "visual-threshold-failed",
                "Strict image comparison did not reach the visual completion threshold.",
                verdict=visual_result.get("verdict"),
                similarityScore=visual_result.get("similarityScore"),
            ),
        )
    if not visual_pass and not web_pass:
        required_action = "repair-visual-and-web"
    elif not visual_pass:
        required_action = "repair-visual"
    elif not web_pass:
        required_action = "repair-web-representation"
    else:
        required_action = "complete"
    can_complete = visual_pass and web_pass
    visual_projection = {
        key: value
        for key, value in visual_result.items()
        if key not in {"nextActions", "requiredAction"}
    }
    visual_hints = []
    if not visual_pass:
        text_hint = _ocr_text_repair_hint(
            visual_result,
            web_result,
            reconstruction,
        )
        if text_hint:
            visual_hints.append(text_hint)
            visual_projection["requiredAction"] = {
                "kind": "repair-ocr-text-rendering-from-existing-contract",
                "reason": "Use the OCR strings, repeated text groups, and existing DOM hints returned by this review; preserve verified source vectors.",
            }
        hot_regions = visual_result.get("hotRegions") or []
        if hot_regions:
            largest = hot_regions[0]
            visual_hints.append(
                {
                    "kind": "largest-visual-hot-region",
                    "referenceBoxSource": largest.get("box"),
                    "areaRatio": largest.get("areaRatio"),
                    "action": "Repair HTML/CSS geometry or styling in this source-pixel region using the existing reconstruction contract and DOM repair hints, then call sens_review again. Do not inspect the reference or call another vision tool.",
                    "source": "measured-strict-diff",
                }
            )
            if not text_hint:
                visual_projection["requiredAction"] = {
                    "kind": "repair-largest-hot-region-from-existing-contract",
                    "region": largest.get("box"),
                    "reason": "Use the existing reconstruction contract plus measured DOM/CSS deltas; do not request another visual description.",
                }
    web_hints = web_result.get("repairHints") or {}
    return {
        "schemaVersion": "2.0.0",
        "completionScope": "visual+web",
        "visualPass": visual_pass,
        "webPass": web_pass,
        "canComplete": can_complete,
        "verdict": "pass" if can_complete else "fail",
        "requiredAction": required_action,
        "blockingReasons": blocking_reasons,
        "repairHints": {
            "source": "measured-reference-vs-candidate",
            "coordinateSpace": "source-pixels",
            "visual": visual_hints,
            "text": web_hints.get("text", []),
            "controls": web_hints.get("controls", []),
            "structure": web_hints.get("structure", []),
        },
        "workflow": {
            "state": "complete" if can_complete else "repair-from-returned-hints",
            "nextAction": "complete" if can_complete else "apply-one-bounded-repair-then-sens-review",
            "nextTool": None if can_complete else "sens_review",
            "allowedNextTools": [] if can_complete else ["sens_review"],
            "prohibitedNextTools": [
                "sens_see",
                "sens_read",
                "sens_locate",
                "sens_inspect",
                "sens_ask",
                "sens_zoom",
                "sens_compare",
            ],
            "rule": "A sens_review hotRegion is a measured repair target, never a request for another vision call. Apply only repairHints to source, then repeat sens_review.",
        },
        "visual": visual_projection,
        "web": web_result,
    }


def _decode_compact_table(table: Any) -> list[dict[str, Any]]:
    """Decode compact table sections (columns + JSONL array rows) into rows.

    Compact responses encode text/structuralLines/symbolArt as
    {"columns": [...], "count": N, "encoding": "jsonl-arrays", "rows": "..."}.
    Legacy list-of-dict form is returned unchanged.
    """
    if isinstance(table, list):
        return table
    if not isinstance(table, dict):
        return []
    columns = table.get("columns")
    rows = table.get("rows")
    if not isinstance(columns, list) or not columns or not isinstance(rows, str):
        return []
    decoded: list[dict[str, Any]] = []
    for line in rows.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(values, list) or len(values) != len(columns):
            continue
        entry = dict(zip(columns, values))
        if "id" in entry and "elementId" not in entry:
            entry["elementId"] = entry["id"]
        decoded.append(entry)
    return decoded


def _reference_reconstruction(
    reference_path: str,
    contract_path: str | None = None,
) -> dict[str, Any]:
    if contract_path:
        contract_file = Path(contract_path).expanduser().resolve(strict=True)
        payload = json.loads(contract_file.read_text(encoding="utf-8"))
        reconstruction = payload.get("reconstruction")
        if not isinstance(reconstruction, dict):
            raise ValueError("contractPath must contain a reconstruction object")
        if reconstruction.get("targetKind") != "web":
            raise ValueError("contractPath must contain a web reconstruction")
        reference = load_cv(reference_path)
        reference_height, reference_width = reference.shape[:2]
        canvas = reconstruction.get("canvas") or {}
        if (
            int(canvas.get("width") or 0) != reference_width
            or int(canvas.get("height") or 0) != reference_height
        ):
            raise ValueError(
                "contractPath canvas dimensions must match the reference screenshot"
            )
        return reconstruction

    from sight.ops import see_document

    result = see_document(
        reference_path,
        no_store=True,
        fast=True,
        intent="Reconstruct this reference as a real website.",
        profile="reconstruct",
        response="compact",
        target_kind="web",
    )
    reconstruction = (result.get("doc") or {}).get("reconstruction")
    if not isinstance(reconstruction, dict):
        raise RuntimeError("reference analysis did not return a web reconstruction spec")
    for key in ("text", "structuralLines", "symbolArt"):
        reconstruction[key] = _decode_compact_table(reconstruction.get(key))
    return reconstruction


def review_web(
    reference_path: str,
    url: str,
    options: dict[str, Any] | None = None,
    *,
    out_dir: str | Path | None = None,
    no_store: bool = False,
) -> dict[str, Any]:
    """Capture a URL and require visual plus web-representation convergence."""
    reference = load_cv(reference_path)
    reference_height, reference_width = reference.shape[:2]
    capture_options = dict(options or {})
    contract_path_value = capture_options.pop("contractPath", None)
    contract_path = (
        str(contract_path_value) if contract_path_value not in (None, "") else None
    )
    capture_options.setdefault(
        "viewport", {"width": reference_width, "height": reference_height}
    )
    capture_options.setdefault("dpr", 1.0)
    capture_options.setdefault("fullPage", False)
    capture_options["networkPolicy"] = "candidate"

    owned_root: Path | None = None
    if no_store:
        owned_root = Path(tempfile.mkdtemp(prefix="sens-web-review-"))
        capture_root = owned_root
    else:
        base = Path(out_dir) if out_dir is not None else Path(tempfile.gettempdir()) / "sens-reviews"
        capture_root = base / f"review-{time.time_ns()}"
        capture_root.mkdir(parents=True, exist_ok=False)

    try:
        capture = capture_url(
            url,
            capture_root,
            capture_options,
            no_store=False,
        )
        screenshot_path = capture.get("screenshot")
        if not screenshot_path:
            raise RuntimeError("browser capture did not produce a candidate screenshot")
        reconstruction = _reference_reconstruction(reference_path, contract_path)
        visual = compare_images(reference_path, str(screenshot_path), fit="strict")
        web = evaluate_web_integrity(reconstruction, capture)
        result = combine_review(visual, web, reconstruction)
        capture_summary = {
            "captureId": capture.get("captureId"),
            "source": capture.get("source"),
            "settings": capture.get("settings"),
            "screenshotSha256": capture.get("screenshotSha256"),
            "textNodeCount": len(capture.get("textNodes", [])),
            "semanticControlCount": len(capture.get("semanticControls", [])),
            "structuralLineCount": len(capture.get("structuralLines", [])),
            "rasterElementCount": len(capture.get("rasterElements", [])),
            "accessibilityAvailable": capture.get("accessibility") is not None,
            "visualFreeze": capture.get("visualFreeze"),
        }
        artifacts = [] if no_store else list(capture.get("artifacts", []))
        if not no_store:
            capture_summary["screenshot"] = screenshot_path
        result.update(
            {
                "reference": {
                    "path": reference_path,
                    "size": {"width": reference_width, "height": reference_height},
                    "source": "observed",
                    "method": (
                        "persisted-web-contract-plus-decoded-reference"
                        if contract_path
                        else "decoded-reference-image"
                    ),
                    **({"contractPath": contract_path} if contract_path else {}),
                },
                "candidate": {
                    "url": url,
                    "source": "observed",
                    "method": "playwright-instrumented-capture",
                },
                "capture": capture_summary,
                "artifacts": artifacts,
                "noStore": no_store,
            }
        )
        return result
    finally:
        if owned_root is not None:
            shutil.rmtree(owned_root, ignore_errors=True)
