"""Design QA, styles and cross-layer verification (L5)."""

from __future__ import annotations

from typing import Any

import numpy as np

from sight.perception import _intersection_ratio




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




def _inside(box: list[int], container: list[int], pad: int = 2) -> bool:
    """True when the whole `box` lies inside `container` (with a small pad)."""
    return (
        box[0] >= container[0] - pad and box[1] >= container[1] - pad
        and box[2] <= container[2] + pad and box[3] <= container[3] + pad
    )




def _center_in(box: list[int], container: list[int]) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return (
        container[0] <= cx <= container[2] and container[1] <= cy <= container[3]
    )




def _containment_ratio(child: list[int], parent: list[int]) -> float:
    """Fraction of the child box covered by the parent box."""
    x0, y0 = max(child[0], parent[0]), max(child[1], parent[1])
    x1, y1 = min(child[2], parent[2]), min(child[3], parent[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    child_area = (child[2] - child[0]) * (child[3] - child[1])
    if child_area <= 0:
        return 0.0
    return (x1 - x0) * (y1 - y0) / child_area




def _find_gaps(profile: Any, min_gap: int) -> list[tuple[int, int]]:
    """Runs of near-empty rows/cols: (start, end) pairs, at least min_gap.

    Classic XY-cut semantics: only gaps *between* content segments count;
    leading/trailing whitespace of the box is not a cut candidate (the
    caller shrinks the box around content instead)."""
    empty = profile <= max(1, int(0.02 * profile.max()))
    nonempty = np.nonzero(~empty)[0]
    if nonempty.size == 0:
        return []
    first, last = int(nonempty[0]), int(nonempty[-1])
    gaps: list[tuple[int, int]] = []
    run_start = None
    for index in range(first, last + 1):
        if empty[index] and run_start is None:
            run_start = index
        elif not empty[index] and run_start is not None:
            if index - run_start >= min_gap:
                gaps.append((run_start, index))
            run_start = None
    if run_start is not None and last + 1 - run_start >= min_gap:
        gaps.append((run_start, last + 1))
    return gaps




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
