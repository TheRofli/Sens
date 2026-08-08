import cv2
import numpy as np

from sight.ops import _run_optional_layer
from sight.perception import _glyph_metrics, color_zones, rank_font_candidates


def test_color_zones_converts_opencv_bgr_to_rgb_hex() -> None:
    red_bgr = np.full((8, 8, 3), (0, 0, 255), dtype=np.uint8)
    blue_bgr = np.full((8, 8, 3), (255, 0, 0), dtype=np.uint8)

    red = color_zones(red_bgr, k=1, sample_side=4)["dominant"][0]["hex"]
    blue = color_zones(blue_bgr, k=1, sample_side=4)["dominant"][0]["hex"]

    assert red == "#FF0000"
    assert blue == "#0000FF"


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
