from sight.focus import recommend_focus


def test_focus_prioritizes_uncertain_small_text_in_source_coordinates() -> None:
    dump = {
        "image": {"width": 400, "height": 200},
        "coordinates": {
            "sourceSize": [1000, 500],
            "regionInSource": [100, 50, 900, 450],
            "analysisSize": [400, 200],
            "analysisToSource": {
                "scaleX": 2.0,
                "scaleY": 2.0,
                "offsetX": 100.0,
                "offsetY": 50.0,
            },
        },
        "ocr": [
            {"text": "TOTAL 18.37", "box": [40, 30, 110, 39], "confidence": 0.48},
            {"text": "Dashboard", "box": [20, 80, 120, 106], "confidence": 0.98},
        ],
        "layout": [],
    }

    [focus] = recommend_focus(dump, max_regions=1, intent="verify the total")

    assert focus["tool"] == "sens_zoom"
    assert "low_ocr_confidence" in focus["reasons"]
    assert "small_text" in focus["reasons"]
    assert "intent_match" in focus["reasons"]
    assert focus["region"]["x"] >= 100
    assert focus["region"]["y"] >= 50
    assert focus["region"]["x"] + focus["region"]["width"] <= 900
    assert focus["region"]["y"] + focus["region"]["height"] <= 450
