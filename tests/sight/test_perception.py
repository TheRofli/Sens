import cv2
import numpy as np

from sight.ops import _run_optional_layer
from sight.perception import (
    _glyph_metrics,
    color_zones,
    compact_numeric_badges,
    detect_dashed_structural_lines,
    detect_vector_paths,
    layout_skeleton,
    rank_font_candidates,
)
from sight.qa import outlined_surface_regions, surface_regions


def test_color_zones_converts_opencv_bgr_to_rgb_hex() -> None:
    red_bgr = np.full((8, 8, 3), (0, 0, 255), dtype=np.uint8)
    blue_bgr = np.full((8, 8, 3), (255, 0, 0), dtype=np.uint8)

    red = color_zones(red_bgr, k=1, sample_side=4)["dominant"][0]["hex"]
    blue = color_zones(blue_bgr, k=1, sample_side=4)["dominant"][0]["hex"]

    assert red == "#FF0000"
    assert blue == "#0000FF"


def test_color_zones_reports_exact_canvas_background_from_border_mode() -> None:
    image = np.full((120, 200, 3), (241, 248, 255), np.uint8)
    image[20:100, 30:170] = (239, 247, 253)

    colors = color_zones(image, k=2, sample_side=24)

    assert colors["canvasBackground"] == {
        "hex": "#FFF8F1",
        "source": "measured",
        "method": "exact-border-mode",
    }
    assert colors["dominant"][0]["hex"] == "#FFF8F1"


def test_optional_heavy_layer_degrades_with_an_explicit_warning() -> None:
    def unavailable(_image_path: str):
        raise ModuleNotFoundError("No module named 'ultralytics'")

    values, warning = _run_optional_layer("objects", unavailable, "fixture.png")

    assert values == []
    assert warning == {
        "code": "optional_objects_unavailable",
        "message": "Optional objects layer is unavailable: No module named 'ultralytics'",
        "recovery": "Continue with measured layout/OCR and the local VLM, or install the optional detector pack.",
    }


def test_font_matching_returns_candidates_not_a_confirmed_family() -> None:
    candidates = rank_font_candidates(0.58)

    assert [item["family"] for item in candidates[:3]] == [
        "inter",
        "space-grotesk",
        "arial",
    ]
    assert all(item["status"] == "candidate" for item in candidates)
    assert candidates[0]["distance"] <= candidates[1]["distance"]


def test_glyph_metrics_use_border_background_for_heavy_display_type() -> None:
    image = np.full((110, 200, 3), (239, 247, 252), np.uint8)
    for x in (10, 45, 80, 115, 150):
        cv2.rectangle(image, (x, 10), (x + 29, 99), (255, 120, 0), -1)

    metrics = _glyph_metrics(image, [10, 10, 180, 100], "ABCDE")

    assert metrics is not None
    assert metrics["capHeight"] == 90
    assert metrics["avgGlyphWidth"] == 30.0
    assert metrics["widthEm"] == 0.24
    assert metrics["characterCount"] == 5
    assert metrics["color"] == "#0078FF"
    assert metrics["colorSource"] == "measured-glyph-pixels"


def test_glyph_metrics_uses_the_ink_core_not_repeated_antialias_gray() -> None:
    image = np.full((80, 320, 3), (246, 246, 246), np.uint8)
    cv2.putText(
        image,
        "Partner Program",
        (24, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (30, 28, 26),
        1,
        cv2.LINE_AA,
    )

    metrics = _glyph_metrics(image, [18, 26, 190, 56], "Partner Program")

    assert metrics is not None
    measured = tuple(
        int(metrics["color"][index : index + 2], 16) for index in (1, 3, 5)
    )
    assert max(measured) < 90
    assert metrics["inkBox"][0] > 18
    assert metrics["inkBox"][1] > 26
    assert metrics["inkBox"][2] < 190
    assert metrics["inkBox"][3] < 56


def test_glyph_metrics_recovers_measured_word_boxes_from_the_largest_ink_gap() -> None:
    image = np.full((120, 520, 3), (18, 22, 48), np.uint8)
    cv2.putText(
        image,
        "Your",
        (20, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.2,
        (250, 250, 250),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "new",
        (285, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.2,
        (250, 250, 250),
        3,
        cv2.LINE_AA,
    )

    metrics = _glyph_metrics(image, [10, 10, 500, 105], "Your new")

    assert metrics is not None
    assert [entry["text"] for entry in metrics["wordBoxes"]] == ["Your", "new"]
    first, second = metrics["wordBoxes"]
    assert first["box"][2] < second["box"][0]
    assert first["box"][0] >= 10
    assert second["box"][2] <= 500
    assert metrics["wordBoxMethod"] == "largest-foreground-column-gaps"


def test_glyph_metrics_distinguishes_upright_and_slanted_word_strokes() -> None:
    image = np.full((130, 440, 3), (20, 24, 44), np.uint8)
    for x in (25, 62, 99):
        cv2.rectangle(image, (x, 22), (x + 12, 108), (248, 248, 248), -1)
    for x in (245, 282, 319, 356):
        cv2.fillConvexPoly(
            image,
            np.asarray(
                [[x + 18, 22], [x + 30, 22], [x + 12, 108], [x, 108]],
                dtype=np.int32,
            ),
            (248, 248, 248),
        )

    metrics = _glyph_metrics(image, [10, 10, 410, 118], "up tilt")

    assert metrics is not None
    assert metrics["wordBoxMethod"] == "largest-foreground-column-gaps"
    assert [entry.get("slant") for entry in metrics["wordBoxes"]] == [
        "normal",
        "italic",
    ]
    assert all(
        entry.get("slantMethod") == "vertical-stroke-hough-median"
        for entry in metrics["wordBoxes"]
    )


def test_compact_numeric_badges_recover_live_counts_from_tinted_pills() -> None:
    from pathlib import Path

    from PIL import Image, ImageDraw, ImageFont

    image = np.full((90, 230, 3), 255, np.uint8)
    font_path = (
        Path(__file__).parents[2]
        / "sidecars"
        / "sight"
        / "assets"
        / "fonts"
        / "InterTight.ttf"
    )
    rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(rgb)
    expected = [
        ((16, 16, 36, 34), "2"),
        ((72, 16, 92, 34), "4"),
        ((132, 16, 161, 45), "12"),
    ]
    for box, value in expected:
        draw.rounded_rectangle(box, radius=6, fill=(228, 235, 255))
        font = ImageFont.truetype(str(font_path), 12 if value != "12" else 13)
        font.set_variation_by_name("SemiBold")
        text_box = draw.textbbox((0, 0), value, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        x = box[0] + ((box[2] - box[0]) - text_width) / 2
        y = box[1] + ((box[3] - box[1]) - text_height) / 2 - text_box[1]
        draw.text((x, y), value, font=font, fill=(53, 111, 232))
    image = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)

    badges = compact_numeric_badges(image)

    assert [badge["value"] for badge in badges] == ["2", "4", "12"]
    assert all(badge["epistemic"] == "inferred" for badge in badges)
    assert all(badge["geometrySource"] == "measured" for badge in badges)
    assert all(badge["representation"] == "live-text-on-css-surface" for badge in badges)


def test_dashed_chart_rules_are_measured_as_structural_lines() -> None:
    image = np.full((180, 420, 3), 255, np.uint8)
    for x in range(50, 350, 8):
        cv2.line(image, (x, 92), (min(x + 3, 350), 92), (226, 226, 226), 1)
    cv2.line(image, (50, 130), (350, 130), (226, 226, 226), 1)

    lines = detect_dashed_structural_lines(image)

    assert len(lines) == 1
    [line] = lines
    assert line["lineStyle"] == "dashed"
    assert line["boxSource"][0] <= 50
    assert line["boxSource"][2] >= 350
    assert line["dashLength"] >= 2
    assert line["dashGap"] >= 2
    assert line["source"] == "measured"


def test_glyph_metrics_distinguish_thin_and_heavy_strokes() -> None:
    def render(thickness: int) -> tuple[np.ndarray, list[int]]:
        image = np.full((240, 900, 3), 255, np.uint8)
        (width, height), baseline = cv2.getTextSize(
            "06.24.21", cv2.FONT_HERSHEY_COMPLEX, 4, thickness
        )
        cv2.putText(
            image,
            "06.24.21",
            (20, 180),
            cv2.FONT_HERSHEY_COMPLEX,
            4,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )
        return image, [10, 170 - height, 30 + width, 190 + baseline]

    # OpenCV 5 clamps Hershey thickness values above 2, so use its two
    # portable stroke classes instead of relying on the older 2-vs-16 range.
    thin_image, thin_box = render(1)
    heavy_image, heavy_box = render(2)

    thin = _glyph_metrics(thin_image, thin_box, "06.24.21")
    heavy = _glyph_metrics(heavy_image, heavy_box, "06.24.21")

    assert thin is not None
    assert heavy is not None
    assert thin["strokeWidthRatio"] < heavy["strokeWidthRatio"]
    assert thin["inkCoverage"] < heavy["inkCoverage"]
    assert thin["weightCandidate"] == "light"
    assert heavy["weightCandidate"] == "bold"


def test_glyph_metrics_recover_display_glyphs_over_contaminated_artwork() -> None:
    image = np.full((500, 900, 3), (235, 225, 205), np.uint8)
    cv2.line(
        image,
        (0, 80),
        (900, 430),
        (215, 95, 10),
        150,
        cv2.LINE_AA,
    )
    box = [75, 45, 825, 455]
    glyphs = [
        ("S", 75, 140),
        ("L", 230, 105),
        ("U", 350, 140),
        ("S", 505, 140),
        ("H", 660, 165),
    ]
    stroke = 45
    for character, left, width in glyphs:
        right = left + width
        if character == "S":
            for top in (50, 228, 405):
                cv2.rectangle(
                    image,
                    (left, top),
                    (right - 1, min(449, top + stroke - 1)),
                    (0, 0, 0),
                    -1,
                )
            cv2.rectangle(image, (left, 50), (left + stroke - 1, 250), (0, 0, 0), -1)
            cv2.rectangle(image, (right - stroke, 228), (right - 1, 449), (0, 0, 0), -1)
        elif character == "L":
            cv2.rectangle(image, (left, 50), (left + stroke - 1, 449), (0, 0, 0), -1)
            cv2.rectangle(image, (left, 405), (right - 1, 449), (0, 0, 0), -1)
        elif character == "U":
            cv2.rectangle(image, (left, 50), (left + stroke - 1, 449), (0, 0, 0), -1)
            cv2.rectangle(image, (right - stroke, 50), (right - 1, 449), (0, 0, 0), -1)
            cv2.rectangle(image, (left, 405), (right - 1, 449), (0, 0, 0), -1)
        else:
            cv2.rectangle(image, (left, 50), (left + stroke - 1, 449), (0, 0, 0), -1)
            cv2.rectangle(image, (right - stroke, 50), (right - 1, 449), (0, 0, 0), -1)
            cv2.rectangle(image, (left, 228), (right - 1, 272), (0, 0, 0), -1)
    cv2.circle(image, (575, 160), 35, (20, 200, 250), -1)
    cv2.rectangle(image, (740, 250), (815, 320), (180, 80, 220), -1)
    # A full-width section border can touch the OCR crop.  It must not become
    # the assumed background and turn the pale canvas into the measured ink.
    cv2.line(image, (box[0], box[1]), (box[2] - 1, box[1]), (0, 0, 0), 1)

    metrics = _glyph_metrics(image, box, "SLUSH")

    assert metrics is not None
    assert metrics["measuredCharacterCount"] == 5
    assert metrics["measuredCharacterCountMethod"] == (
        "glyph-color-connected-components"
    )
    assert [item["text"] for item in metrics["glyphBoxes"]] == list("SLUSH")
    assert [item["box"] for item in metrics["glyphBoxes"]] == [
        [75, 50, 215, 450],
        [230, 50, 335, 450],
        [350, 50, 490, 450],
        [505, 50, 645, 450],
        [660, 50, 825, 450],
    ]
    assert 0.45 < metrics["inkCoverage"] < 0.80


def test_surface_regions_keep_real_card_and_drop_plain_text_block() -> None:
    image = np.full((500, 800, 3), (241, 248, 255), np.uint8)
    cv2.rectangle(image, (100, 100), (699, 399), (220, 220, 220), 2)
    cv2.rectangle(image, (102, 102), (697, 397), (255, 255, 255), -1)
    cv2.rectangle(image, (160, 180), (520, 230), (255, 120, 0), -1)
    blocks = [
        {"kind": "block", "box": [100, 100, 700, 400], "area": 180_000},
        {"kind": "block", "box": [160, 180, 521, 231], "area": 18_411},
    ]

    surfaces = surface_regions(image, blocks, [], "#FFF8F1")

    assert len(surfaces) == 1
    assert surfaces[0]["box"] == [100, 100, 700, 400]
    assert surfaces[0]["background"] == "#FFFFFF"
    assert surfaces[0]["borderColor"] == "#DCDCDC"
    assert surfaces[0]["source"] == "measured"


def test_surface_regions_do_not_promote_heavy_display_glyph_groups() -> None:
    image = cv2.imread(
        "qa/fixtures/reconstruction-matrix/references/caldera-3b581b8f2b4632fd.png"
    )
    assert image is not None
    glyph_blocks = [
        {"kind": "block", "box": [57, 137, 212, 207], "area": 10_850},
        {"kind": "block", "box": [236, 137, 580, 207], "area": 24_080},
        {"kind": "block", "box": [57, 227, 153, 298], "area": 6_816},
        {"kind": "block", "box": [175, 227, 462, 298], "area": 20_377},
    ]

    surfaces = surface_regions(image, glyph_blocks, [], "#E2E2DF")

    assert not any(
        tuple(surface["box"]) in {tuple(block["box"]) for block in glyph_blocks}
        for surface in surfaces
    )


def test_layout_skeleton_exposes_measured_line_segments() -> None:
    image = np.full((500, 1000, 3), (241, 248, 255), np.uint8)
    cv2.line(image, (25, 200), (974, 200), (255, 120, 0), 3)

    skeleton = layout_skeleton(image)

    assert skeleton["horizontal"] == [200]
    assert skeleton["vertical"] == []
    [segment] = skeleton["segments"]
    assert segment["orientation"] == "horizontal"
    assert segment["start"][1] == segment["end"][1] == 200
    assert segment["start"][0] <= 25
    assert segment["end"][0] >= 974
    assert segment["thickness"] == 5
    assert segment["length"] >= 950
    assert segment["color"] == "#0078FF"
    assert segment["edgeContrast"] > 100
    assert segment["source"] == "measured"
    assert segment["method"] == "morphological-line-segment"


def test_layout_skeleton_does_not_turn_uniform_panel_fill_into_lines() -> None:
    image = np.full((500, 1000, 3), (229, 229, 229), np.uint8)
    image[20:480, 300:800] = (255, 255, 255)

    skeleton = layout_skeleton(image)

    assert skeleton["segments"] == []


def test_surface_regions_recovers_large_uniform_dashboard_cards() -> None:
    image = np.full((500, 1000, 3), (247, 246, 245), np.uint8)
    image[:, :60] = (229, 229, 229)
    image[100:400, 300:700] = (254, 254, 254)
    image[100:260, 740:960] = (254, 254, 254)
    cv2.putText(
        image,
        "Revenue",
        (330, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )

    surfaces = surface_regions(image, [], [], "#E5E5E5")
    boxes = [surface["box"] for surface in surfaces]

    assert [300, 100, 700, 400] in boxes
    assert [740, 100, 960, 260] in boxes
    assert all(
        surface["method"] == "dominant-color-connected-surface"
        for surface in surfaces
    )


def test_surface_regions_recovers_broad_neutral_panels_with_subtle_gradient() -> None:
    image = np.full((500, 1000, 3), (229, 229, 229), np.uint8)
    for y in range(image.shape[0]):
        sidebar = 240 if (y // 6) % 2 == 0 else 247
        main = 248 if (y // 6) % 2 == 0 else 255
        image[y, 60:270] = (sidebar, sidebar, sidebar)
        image[y, 270:] = (main, main, main)

    surfaces = surface_regions(image, [], [], "#E5E5E5")

    assert any(
        surface["box"][0] <= 62
        and surface["box"][2] >= 268
        and surface["box"][3] >= 495
        and surface["background"] == "#F0F0F0"
        for surface in surfaces
    )
    assert any(
        surface["box"][0] <= 272
        and surface["box"][2] >= 995
        and surface["box"][3] >= 495
        and surface["background"] == "#F8F8F8"
        for surface in surfaces
    )


def test_outlined_surface_regions_keeps_quiet_nested_dashboard_cards() -> None:
    image = np.full((500, 1000, 3), (249, 249, 249), np.uint8)
    cv2.rectangle(image, (250, 10), (990, 490), (255, 255, 255), -1)
    cv2.rectangle(image, (250, 10), (990, 490), (244, 244, 244), 1)
    cv2.rectangle(image, (300, 100), (700, 430), (255, 255, 255), -1)
    cv2.rectangle(image, (300, 100), (700, 430), (242, 242, 242), 1)
    cv2.putText(
        image,
        "Revenue",
        (330, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )

    surfaces = outlined_surface_regions(image, "#F9F9F9")
    boxes = [surface["box"] for surface in surfaces]

    assert any(box[0] <= 251 and box[2] >= 990 for box in boxes)
    assert any(box[0] <= 301 and box[2] >= 700 for box in boxes)
    assert all(surface["borderWidth"] in {None, 1} for surface in surfaces)
    assert any(surface["borderWidth"] == 1 for surface in surfaces)
    assert all(
        surface["method"] == "low-contrast-closed-surface"
        for surface in surfaces
    )


def test_outlined_surface_regions_rejects_complex_halftone_artwork_container() -> None:
    image = np.full((500, 1000, 3), (226, 226, 223), np.uint8)
    cv2.rectangle(image, (50, 50), (950, 450), (218, 218, 215), 2)
    for y in range(54, 448, 12):
        for x in range(54, 948, 12):
            color = (36, 80, 255) if (x // 12 + y // 12) % 2 else (255, 74, 112)
            cv2.circle(image, (x, y), 4, color, -1, cv2.LINE_AA)

    surfaces = outlined_surface_regions(image, "#E2E2DF")

    assert not any(
        surface["box"][0] <= 52
        and surface["box"][1] <= 52
        and surface["box"][2] >= 948
        and surface["box"][3] >= 448
        for surface in surfaces
    )


def test_detect_vector_paths_traces_chart_line_and_ignores_control_box() -> None:
    image = np.full((500, 1000, 3), (247, 246, 245), np.uint8)
    chart_points = np.asarray(
        [[300, 350], [380, 250], [470, 390], [560, 230], [680, 330]],
        dtype=np.int32,
    )
    cv2.polylines(image, [chart_points], False, (241, 112, 176), 3, cv2.LINE_AA)
    cv2.rectangle(image, (40, 40), (220, 110), (255, 110, 0), 3)

    paths = detect_vector_paths(image, [[35, 35, 225, 115]])

    assert len(paths) == 1
    assert paths[0]["box"][0] <= 300
    assert paths[0]["box"][2] >= 680
    assert len(paths[0]["points"]) >= 5
    assert paths[0]["fill"] == "none"
    assert paths[0]["source"] == "measured"
