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


def test_focus_discovers_text_dense_region_not_covered_by_ocr() -> None:
    dump = {
        "image": {"width": 1440, "height": 900},
        "ocr": [
            {
                "text": "Of Chains",
                "box": [47, 212, 478, 314],
                "confidence": 0.96,
            }
        ],
        "attention": [
            {
                "box": [0, 0, 540, 338],
                "score": 0.49,
                "why": "text+contrast density",
            }
        ],
    }

    [focus] = recommend_focus(dump, max_regions=1)

    assert focus["region"] == {"x": 0, "y": 0, "width": 594, "height": 365}
    assert "unresolved_text_density" in focus["reasons"]
    assert focus["evidence"] == "unresolved visible text"


def test_focus_ranks_unread_navigation_before_ocr_explained_body() -> None:
    dump = {
        "image": {"width": 1440, "height": 900},
        "ocr": [
            {"text": "Of Chains", "box": [47, 212, 478, 314], "confidence": 0.96},
            {"text": "Body one", "box": [886, 736, 1341, 753], "confidence": 0.98},
            {"text": "Body two", "box": [886, 763, 1174, 780], "confidence": 0.98},
            {"text": "Explore", "box": [944, 842, 1068, 859], "confidence": 0.98},
            {"text": "Book", "box": [1217, 842, 1310, 855], "confidence": 0.98},
        ],
        "attention": [
            {
                "box": [720, 675, 1440, 900],
                "score": 0.731,
                "why": "text+contrast density",
            },
            {
                "box": [0, 0, 540, 338],
                "score": 0.492,
                "why": "text+contrast density",
            },
            {
                "box": [540, 0, 1080, 112],
                "score": 0.348,
                "why": "text+contrast density",
            },
        ],
    }

    focus = recommend_focus(dump, max_regions=2)

    assert [item["region"] for item in focus] == [
        {"x": 486, "y": 0, "width": 954, "height": 128},
        {"x": 0, "y": 0, "width": 594, "height": 365},
    ]
