import numpy as np

from sight.document import build_document, normalize_box, render_markdown

DUMP = {
    "image": {"width": 1000, "height": 500},
    "colors": [{"hex": "#FDFDFD", "ratio": 0.6}, {"hex": "#262525", "ratio": 0.1}],
    "scene": [{"label": "web page layout with navigation"}],
    "elements": [
        {"id": 1, "kind": "text", "text": "HELLO", "box": [10, 20, 110, 40],
         "font": {"family": "arial", "fontSize": 20}},
        {"id": 2, "kind": "image", "box": [200, 100, 600, 400]},
    ],
    "design": {"issues": [{"kind": "card_alignment", "detail": "cards differ in height"}]},
    "gaps": [], "controls": [], "shadows": [],
}


def test_normalize_box_range() -> None:
    assert normalize_box([0, 0, 500, 250], 1000, 500) == [0, 0, 500, 500]
    # y2=500 при h=500 — полная высота -> 1000 (пер-осевая нормализация)
    assert normalize_box([-5, 0, 2000, 500], 1000, 500) == [0, 0, 1000, 1000]


def test_document_sections_without_vlm() -> None:
    img = np.full((500, 1000, 3), 200, np.uint8)
    doc = build_document(DUMP, img)
    assert doc["header"]["theme"] == "light"
    assert doc["semantics_status"] == "unavailable"
    assert doc["measurements"][0]["kind"] == "card_alignment"
    assert "issues" not in doc
    md = render_markdown(doc)
    assert "ЭКРАН 1000×500" in md
    assert "[1]" in md and "HELLO" in md
    assert "измерения" in md


class _FakeVlm:
    def vibe(self, path): return "retro poster, blue on cream"
    def describe(self, path, box): return "flat illustration of a car"
    def transcribe(self, path, box): return "circular text"


def test_document_with_vlm() -> None:
    img = np.full((500, 1000, 3), 200, np.uint8)
    doc = build_document(DUMP, img, vlm=_FakeVlm(), image_path="x.png")
    assert doc["semantics_status"] == "ok"
    assert doc["header"]["vibe"] == "retro poster, blue on cream"
    assert doc["graphics"][0]["caption"] == "flat illustration of a car"
