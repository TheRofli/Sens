import numpy as np

from sight.ops import _run_optional_layer
from sight.perception import color_zones


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
