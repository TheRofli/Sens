import numpy as np

from sight.document import build_document, normalize_box, render_markdown

DUMP = {
    "image": {"width": 1000, "height": 500},
    "source": {"id": "sha256:fixture", "mediaType": "image/png"},
    "coordinates": {
        "sourceSize": [1000, 500],
        "regionInSource": [0, 0, 1000, 500],
        "analysisSize": [1000, 500],
        "analysisToSource": {
            "scaleX": 1.0, "scaleY": 1.0, "offsetX": 0.0, "offsetY": 0.0,
        },
    },
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


def test_visual_scene_v2_is_versioned_and_auditable() -> None:
    img = np.full((500, 1000, 3), 200, np.uint8)

    doc = build_document(DUMP, img)

    assert doc["schemaVersion"] == "2.0.0"
    assert doc["source"]["id"] == "sha256:fixture"
    assert doc["coordinateSpaces"]["analysis"]["sourceTransform"]["scaleX"] == 1.0
    assert doc["claims"]
    for claim in doc["claims"]:
        assert claim["epistemic"] in {"observed", "measured", "inferred"}
        assert claim["method"]
        assert claim["evidence"]
        assert "uncertainty" in claim
    assert any(warning["code"] == "semantics_unavailable" for warning in doc["warnings"])
    assert any(action["tool"] == "sens_compare" for action in doc["nextActions"])
    markdown = render_markdown(doc)
    assert "VISUAL SCENE 2.0.0" in markdown
    assert "sha256:fixture" in markdown
    assert "TRUTH measured=" in markdown
    assert "WARNING semantics_unavailable" in markdown


class _FakeVlm:
    def vibe(self, path): return "retro poster, blue on cream"
    def describe(self, path, box): return "flat illustration of a car"
    def transcribe(self, path, box): return "circular text"


def test_document_reads_facts_key() -> None:
    dump = {**DUMP, "design": {"facts": [{"kind": "contrast", "detail": "min 4.6:1"}]}}
    img = np.full((500, 1000, 3), 200, np.uint8)
    doc = build_document(dump, img)
    assert doc["measurements"][0]["kind"] == "contrast"


def test_document_with_vlm() -> None:
    img = np.full((500, 1000, 3), 200, np.uint8)
    doc = build_document(DUMP, img, vlm=_FakeVlm(), image_path="x.png")
    assert doc["semantics_status"] == "ok"
    assert doc["header"]["vibe"] == "retro poster, blue on cream"
    assert doc["graphics"][0]["caption"] == "flat illustration of a car"


def test_document_separates_composition_map_from_ascii_text() -> None:
    dump = {
        **DUMP,
        "image": {"width": 10, "height": 4},
        "source": {"id": "sha256:ascii", "mediaType": "image/png"},
        "coordinates": {
            "sourceSize": [10, 4],
            "regionInSource": [0, 0, 10, 4],
            "analysisSize": [10, 4],
            "analysisToSource": {
                "scaleX": 1.0, "scaleY": 1.0, "offsetX": 0.0, "offsetY": 0.0,
            },
        },
        "ocr": [
            {"text": "/\\", "box": [2, 0, 6, 2], "confidence": 0.99},
            {"text": "||", "box": [2, 2, 6, 4], "confidence": 0.98},
        ],
        "elements": [],
    }
    image = np.full((4, 10, 3), 200, np.uint8)

    doc = build_document(dump, image)

    assert isinstance(doc["ascii"], str)
    assert doc["monospaceText"]["text"] == " /\\  \n ||  "
    assert doc["monospaceText"]["source"] == "inferred"
    assert "ASCII TEXT candidate" in render_markdown(doc)


def test_document_turns_user_intent_into_source_pixel_zoom_action() -> None:
    dump = {
        **DUMP,
        "ocr": [
            {"text": "TOTAL 18.37", "box": [100, 100, 180, 110], "confidence": 0.45}
        ],
    }
    image = np.full((500, 1000, 3), 200, np.uint8)

    doc = build_document(dump, image, intent="verify the total")

    zoom = next(action for action in doc["nextActions"] if action["tool"] == "sens_zoom")
    assert zoom["arguments"]["region"]["x"] >= 0
    assert "intent_match" in zoom["reasons"]


def test_semantic_calls_are_bounded_and_report_partial_state() -> None:
    dump = {
        **DUMP,
        "elements": [
            {"id": 1, "kind": "image", "box": [0, 0, 100, 100]},
            {"id": 2, "kind": "image", "box": [100, 0, 200, 100]},
            {"id": 3, "kind": "image", "box": [200, 0, 300, 100]},
        ],
    }
    image = np.full((500, 1000, 3), 200, np.uint8)
    model = _FakeVlm()

    doc = build_document(
        dump,
        image,
        vlm=model,
        image_path="x.png",
        max_semantic_calls=2,
    )

    assert doc["header"]["vibe"]
    assert sum(graphic.get("caption") is not None for graphic in doc["graphics"]) == 1
    assert doc["semantics_status"] == "partial"
    assert any(warning["code"] == "semantic_budget_exhausted" for warning in doc["warnings"])
