"""Hierarchical section/element tree, SoM annotation, screen summary."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from sight.perception import _hex_to_bgr, _luminance
from sight.qa import _center_in, _containment_ratio, _find_gaps, _inside




# --------------------------------------------------------------------------
# Hierarchical analysis: screen -> sections -> elements -> sub-elements.
# AnyRes-style (global pass + per-section detail), ScreenAI-style roles,
# OmniParser-style element typing, Set-of-Marks output for the host LLM.
# --------------------------------------------------------------------------

_SCREEN_ROLES = ("nav", "header", "hero", "badge", "cta", "content", "footer", "graphic")




def xycut_sections(
    mask: Any,
    min_gap: int = 12,
    min_size: int = 40,
    depth: int = 0,
    max_depth: int = 10,
    box: tuple[int, int, int, int] | None = None,
) -> list[list[int]]:
    """Recursive XY-cut: split a rectangle along the widest empty band.

    Produces the container hierarchy of the screen (top bar, hero, badge,
    footer...) the way document pipelines (PP-Structure, Docling) split
    pages into blocks. Horizontal cuts first (reading order)."""
    if box is None:
        box = (0, 0, mask.shape[1], mask.shape[0])
    x0, y0, x1, y1 = (int(v) for v in box)
    if x1 - x0 < min_size or y1 - y0 < min_size or depth >= max_depth:
        return [[x0, y0, x1, y1]]
    rows = mask[y0:y1, x0:x1].sum(axis=1)
    gaps = _find_gaps(rows, min_gap)
    if gaps:
        # widest horizontal band wins
        start, end = max(gaps, key=lambda g: g[1] - g[0])
        gy = y0 + (start + end) // 2
        if gy - y0 < min_size or y1 - gy < min_size:
            return [[x0, y0, x1, y1]]
        top = xycut_sections(mask, min_gap, min_size, depth + 1, max_depth, (x0, y0, x1, gy))
        bottom = xycut_sections(mask, min_gap, min_size, depth + 1, max_depth, (x0, gy, x1, y1))
        return top + bottom
    cols = mask[y0:y1, x0:x1].sum(axis=0)
    gaps = _find_gaps(cols, min_gap)
    if gaps:
        start, end = max(gaps, key=lambda g: g[1] - g[0])
        gx = x0 + (start + end) // 2
        if gx - x0 < min_size or x1 - gx < min_size:
            return [[x0, y0, x1, y1]]
        left = xycut_sections(mask, min_gap, min_size, depth + 1, max_depth, (x0, y0, gx, y1))
        right = xycut_sections(mask, min_gap, min_size, depth + 1, max_depth, (gx, y0, x1, y1))
        return left + right
    return [[x0, y0, x1, y1]]




def content_mask(image: Any, background_bgr: Any, threshold: int = 14) -> Any:
    """Pixels that differ from the page background (morphologically closed)."""
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bg = float(0.299 * background_bgr[2] + 0.587 * background_bgr[1] + 0.114 * background_bgr[0])
    mask = (np.abs(gray.astype(int) - int(bg)) > threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return mask




def build_section_tree(
    image: Any,
    background_bgr: Any,
    textures: list[dict[str, Any]],
    image_w: int,
    image_h: int,
    ocr_items: list[dict[str, Any]] | None = None,
    controls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Section tree from XY-cut containers + texture blocks as graphic
    sections (AnyRes-style: containers first, detail inside)."""
    mask = content_mask(image, background_bgr)
    # Carve texture zones (padded) out of the split mask so background art
    # cannot glue distinct content bands (badge/hero/cta) into one section.
    # The pad also covers sub-pixel edges of the art block that lie outside
    # its detected box.
    for texture in textures:
        tx0, ty0, tx1, ty1 = (int(v) for v in texture["box"])
        mask[max(0, ty0 - 4):min(image_h, ty1 + 4),
             max(0, tx0 - 4):min(image_w, tx1 + 4)] = 0
    # Restore controls and texts that sit on top of the art (buttons over a
    # background pattern) so their sections keep their full height.
    for zone in (controls or []) + (ocr_items or []):
        zx0, zy0, zx1, zy1 = (int(v) for v in zone["box"])
        mask[zy0:zy1, zx0:zx1] = 1
    boxes = xycut_sections(mask)
    nodes: list[dict[str, Any]] = []
    for index, box in enumerate(boxes):
        if (box[2] - box[0]) * (box[3] - box[1]) < 400:
            continue
        # Drop cut bands that contain no content (pure whitespace frames).
        sub = mask[box[1]:box[3], box[0]:box[2]]
        if sub.sum() == 0:
            continue
        # Shrink the container to its actual content bounds so sections do
        # not inherit the whitespace of the bands around them.
        ys, xs = np.nonzero(sub)
        box = [box[0] + int(xs.min()), box[1] + int(ys.min()),
               box[0] + int(xs.max()) + 1, box[1] + int(ys.max()) + 1]
        if box[2] - box[0] < 24 or box[3] - box[1] < 24:
            continue
        nodes.append({
            "id": index + 1,
            "kind": "section",
            "role": "content",
            "box": [int(v) for v in box],
            "area": (box[2] - box[0]) * (box[3] - box[1]),
            "style": {},
            "elements": [],
            "children": [],
        })
    nodes = _merge_band_sections(nodes, image_h)
    for texture in textures:
        box = [int(v) for v in texture["box"]]
        nodes.append({
            "id": len(nodes) + 1,
            "kind": "texture",
            "role": "graphic",
            "box": box,
            "area": (box[2] - box[0]) * (box[3] - box[1]),
            "style": {},
            "elements": [],
            "children": [],
        })
    # Containment: child = smallest-area section that contains >= 80% of it.
    for node in nodes:
        if node["kind"] == "texture":
            node["parent"] = 0
            continue
        best_parent: dict[str, Any] | None = None
        for candidate in nodes:
            if candidate is node or candidate["kind"] == "texture":
                continue
            if _containment_ratio(node["box"], candidate["box"]) >= 0.8:
                if best_parent is None or candidate["area"] < best_parent["area"]:
                    best_parent = candidate
        node["parent"] = best_parent["id"] if best_parent is not None else 0
    root: dict[str, Any] = {
        "id": 0, "kind": "screen", "role": "screen",
        "box": [0, 0, image_w, image_h],
        "area": image_w * image_h, "style": {}, "elements": [], "children": [],
    }
    by_id = {node["id"]: node for node in nodes}
    by_id[0] = root
    for node in nodes:
        by_id[node["parent"]]["children"].append(node)
    for node in nodes:
        node.pop("parent", None)
    return root




def _merge_band_sections(
    nodes: list[dict[str, Any]], image_h: int
) -> list[dict[str, Any]]:
    """Merge sibling XY-cut chips that sit on the same thin horizontal band.

    A top bar split into logo / menu / cta chunks by vertical gaps is one
    navigation strip, not three sections. Chips whose y-ranges overlap by
    >= 60% of the shorter one and whose combined band height is below 12%
    of the screen merge into a single container."""
    band_limit = max(40, int(image_h * 0.12))
    bands: list[dict[str, Any]] = []
    for node in sorted(nodes, key=lambda n: (n["box"][1], n["box"][0])):
        x0, y0, x1, y1 = node["box"]
        for band in bands:
            bx0, by0, bx1, by1 = band["box"]
            overlap = min(y1, by1) - max(y0, by0)
            if overlap >= 0.6 * min(y1 - y0, by1 - by0):
                band["box"] = [min(x0, bx0), min(y0, by0),
                               max(x1, bx1), max(y1, by1)]
                band["members"].append(node)
                break
        else:
            bands.append({"box": [x0, y0, x1, y1], "members": [node]})
    merged: list[dict[str, Any]] = []
    next_id = 1
    for band in bands:
        if len(band["members"]) == 1 or band["box"][3] - band["box"][1] > band_limit:
            for member in band["members"]:
                member["id"] = next_id
                next_id += 1
                merged.append(member)
            continue
        x0, y0, x1, y1 = band["box"]
        merged.append({
            "id": next_id,
            "kind": "section",
            "role": "content",
            "box": [x0, y0, x1, y1],
            "area": (x1 - x0) * (y1 - y0),
            "style": {},
            "elements": [],
            "children": [],
        })
        next_id += 1
    return merged




def _section_role(
    node: dict[str, Any],
    texts: list[dict[str, Any]],
    buttons: list[dict[str, Any]],
    image_w: int,
    image_h: int,
    largest_cap: float,
) -> str:
    """Heuristic role for a section (ScreenAI-style), from geometry + content."""
    box = node["box"]
    w = box[2] - box[0]
    h = box[3] - box[1]
    if node["kind"] == "texture":
        return "graphic"
    if h < image_h * 0.18 and box[1] < image_h * 0.15 and len(texts) >= 2:
        return "nav"
    if len(buttons) >= 2:
        return "cta"
    if h < image_h * 0.14 and w < image_w * 0.55 and texts:
        return "badge"
    if box[1] < image_h * 0.2 and h < image_h * 0.3:
        return "header"
    if box[1] > image_h * 0.85:
        return "footer"
    cap = max((t.get("metrics", {}).get("capHeight") or 0) for t in texts) if texts else 0
    if cap > 0 and cap >= largest_cap * 0.8:
        return "hero"
    return "content"




def _assign_roles(
    root: dict[str, Any],
    ocr_items: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    image_w: int,
    image_h: int,
) -> None:
    """Two passes: geometry/text/button based roles, then promote the section
    holding the largest glyphs to hero (badge/header keep their role)."""
    largest_cap = 0.0
    for item in ocr_items:
        cap = (item.get("metrics") or {}).get("capHeight") or 0
        largest_cap = max(largest_cap, float(cap))
    largest_cap = max(largest_cap, 1.0)

    def walk(node: dict[str, Any]) -> None:
        if node["kind"] != "screen":
            texts = [o for o in ocr_items if _center_in(o["box"], node["box"])]
            buttons = [c for c in controls if _center_in(c["box"], node["box"])]
            node["role"] = _section_role(node, texts, buttons, image_w, image_h, largest_cap)
        for child in node["children"]:
            walk(child)

    walk(root)
    # Promote the section with the single largest text to hero unless it is
    # a nav/header/badge/footer/graphic.
    if ocr_items:
        biggest = max(ocr_items, key=lambda o: (o.get("metrics") or {}).get("capHeight") or 0)
        hero = None
        def find_hero(node: dict[str, Any]) -> None:
            nonlocal hero
            if hero is not None:
                return
            if _center_in(biggest["box"], node["box"]) and node["kind"] == "section":
                if node["role"] not in ("nav", "badge", "footer", "graphic"):
                    hero = node
                    return
            for child in node["children"]:
                find_hero(child)
        find_hero(root)
        if hero is not None:
            hero["role"] = "hero"




def build_element_tree(
    root: dict[str, Any],
    ocr_items: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    icons: list[dict[str, Any]],
    textures: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach every detected element to the smallest section containing it.

    Returns a flat element list (each with a stable id for Set-of-Marks)."""
    elements: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []

    def collect(node: dict[str, Any]) -> None:
        nodes.append(node)
        for child in node["children"]:
            collect(child)

    collect(root)
    nodes_by_area = sorted(nodes, key=lambda n: n["area"])

    def attach(element: dict[str, Any]) -> None:
        element = dict(element)
        for node in nodes_by_area:
            if node["kind"] == "texture":
                continue
            if _center_in(element["box"], node["box"]):
                node["elements"].append(element)
                elements.append(element)
                return
        root["elements"].append(element)
        elements.append(element)

    for item in ocr_items:
        element = {
            "kind": "text",
            "box": [int(v) for v in item["box"]],
            "text": item["text"],
            "confidence": round(float(item.get("confidence", 0)), 3),
            "font": item.get("metrics"),
        }
        attach(element)
    for control in controls:
        element = {
            "kind": "button",
            "box": [int(v) for v in control["box"]],
            "background": control.get("background"),
            "borderColor": control.get("borderColor"),
            "borderWidth": control.get("borderWidth"),
            "cornerRadius": control.get("cornerRadius"),
        }
        attach(element)
    for icon in icons:
        element = {
            "kind": "icon",
            "box": [int(v) for v in icon["box"]],
            "icon": icon.get("kind"),
            "color": icon.get("color"),
        }
        attach(element)
    for texture in textures:
        element = {
            "kind": "image",
            "box": [int(v) for v in texture["box"]],
            "texture": True,
        }
        attach(element)
    for index, element in enumerate(elements):
        element["id"] = index + 1
    # Prune empty shell sections: XY-cut whitespace frames that carry no
    # elements and no nested content are layout noise, not containers.
    def prune(node: dict[str, Any]) -> bool:
        node["children"] = [child for child in node["children"] if prune(child)]
        if (
            node["kind"] not in ("screen", "texture")
            and not node["elements"]
            and not node["children"]
        ):
            return False
        return True

    root["children"] = [child for child in root["children"] if prune(child)]
    return elements




def _button_subparts(
    element: dict[str, Any],
    all_elements: list[dict[str, Any]],
) -> dict[str, Any]:
    """Sub-elements of a button: text and icon inside it (level 3)."""
    sub = {"text": [], "icon": []}
    for other in all_elements:
        if other is element:
            continue
        if not _inside(other["box"], element["box"], pad=4):
            continue
        if other["kind"] == "text":
            sub["text"].append(other)
        elif other["kind"] == "icon":
            sub["icon"].append(other)
    return sub




def expand_button_subparts(elements: list[dict[str, Any]]) -> None:
    """Populate button subparts in-place (level 3 of the hierarchy)."""
    for element in elements:
        if element["kind"] == "button":
            element["sub"] = _button_subparts(element, elements)




def annotate_som(
    image: Any,
    elements: list[dict[str, Any]],
    out_path: str,
) -> str:
    """Set-of-Marks: draw numbered boxes over detected elements.

    The host LLM refers to elements by their ids ("the button 7") exactly
    like GPT-4V + SoM prompts."""
    import cv2

    canvas = image.copy()
    palette = [
        (0, 0, 255), (0, 200, 0), (255, 128, 0), (0, 200, 200),
        (200, 0, 200), (200, 200, 0), (255, 0, 0), (0, 160, 255),
    ]
    for element in elements:
        x0, y0, x1, y1 = element["box"]
        color = palette[element["id"] % len(palette)]
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
        label = str(element["id"])
        tag_w = 14 + 7 * len(label)
        cv2.rectangle(canvas, (x0, max(0, y0 - 16)), (x0 + tag_w, y0), color, -1)
        cv2.putText(
            canvas, label, (x0 + 3, max(12, y0 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA,
        )
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if not cv2.imwrite(out_path, canvas):
            return ""
    except OSError:
        return ""
    return out_path




def summarize_screen(
    root: dict[str, Any],
    image_w: int,
    image_h: int,
    background: str,
) -> str:
    """Human/LLM readable walk of the section tree with reading order."""
    theme = "dark" if _luminance(_hex_to_bgr(background)) < 128 else "light"
    lines = [f"Screen {image_w}x{image_h}, {theme} theme, background {background}."]

    def describe_element(element: dict[str, Any], indent: str) -> str:
        kind = element["kind"]
        box = element["box"]
        if kind == "text":
            font = element.get("font") or {}
            family = font.get("family", "?")
            size = font.get("fontSize", "?")
            return f"{indent}- text \"{element['text']}\" at [{box[0]},{box[1]},{box[2]},{box[3]}] font {family}~{size}px"
        if kind == "button":
            parts = []
            if element.get("background"):
                parts.append(f"bg {element['background']}")
            if element.get("borderColor"):
                parts.append(f"border {element['borderColor']}")
            texts = [t["text"] for t in element.get("sub", {}).get("text", [])]
            if texts:
                parts.append("label " + "/".join(texts))
            return f"{indent}- button [{box[0]},{box[1]},{box[2]},{box[3]}] " + ", ".join(parts)
        if kind == "icon":
            return f"{indent}- icon {element.get('icon')} at [{box[0]},{box[1]},{box[2]},{box[3]}]"
        if kind == "image":
            return f"{indent}- graphic [{box[0]},{box[1]},{box[2]},{box[3]}]"
        return f"{indent}- {kind} [{box[0]},{box[1]},{box[2]},{box[3]}]"

    def walk(node: dict[str, Any], depth: int) -> None:
        indent = "  " * (depth + 1)
        box = node["box"]
        lines.append(
            f"{indent}- section role={node['role']} "
            f"[{box[0]},{box[1]},{box[2]},{box[3]}]"
        )
        for element in node["elements"]:
            lines.append(describe_element(element, indent + "  "))
        for child in node["children"]:
            walk(child, depth + 1)

    walk(root, 0)
    return "\n".join(lines)
