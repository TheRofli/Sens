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

    reconstruction = build_document(dump, image, profile="reconstruct")
    assert reconstruction["reconstruction"]["monospaceContent"] == {
        "text": " /\\  \n ||  ",
        "confidence": reconstruction["monospaceText"]["confidence"],
        "method": reconstruction["monospaceText"]["method"],
        "strategy": "render-as-text-not-raster",
        "rule": "Recreate the exact characters and whitespace in a monospace text element; never replace it with a screenshot or flattened image.",
    }


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


def test_reconstruction_profile_returns_an_implementation_contract() -> None:
    dump = {
        **DUMP,
        "elements": [
            {
                "id": 1,
                "kind": "text",
                "text": "THE SUMMER",
                "box": [40, 30, 800, 150],
                "confidence": 0.97,
                "font": {"family": "custom", "fontSize": 120, "capHeight": 88},
            },
            {
                "id": 2,
                "kind": "image",
                "box": [300, 180, 700, 460],
                "texture": True,
            },
        ],
    }
    image = np.full((500, 1000, 3), 200, np.uint8)

    doc = build_document(dump, image, profile="reconstruct")

    assert doc["profile"] == "reconstruct"
    spec = doc["reconstruction"]
    assert spec["canvas"] == {
        "width": 1000,
        "height": 500,
        "aspectRatio": 2.0,
        "deviceScaleFactor": 1,
        "coordinateSystem": "source-pixels",
    }
    assert spec["contentPolicy"]["visibleOnly"] is True
    assert spec["contentPolicy"]["addInvisibleInteractions"] is False
    assert spec["text"][0]["status"] == "stable-candidate"
    assert spec["text"][0]["boxSource"] == [40, 30, 800, 150]
    assert spec["primaryAsset"]["strategy"] == "extract-source-crop-verbatim"
    assert spec["rasterAssetRule"]["strategy"] == "extract-source-crop-verbatim"
    assert spec["implementationRules"][0] == "Use one fixed source-pixel coordinate system for size and position."


def test_web_reconstruction_requires_live_dom_and_preserves_only_graphic_assets() -> None:
    dump = {
        **DUMP,
        "skeleton": {
            "segments": [
                {
                    "orientation": "horizontal",
                    "box": [20, 160, 980, 162],
                    "start": [20, 161],
                    "end": [979, 161],
                    "thickness": 2,
                    "length": 960,
                    "color": "#0078FF",
                    "source": "measured",
                },
                {
                    "orientation": "horizontal",
                    "box": [300, 300, 760, 325],
                    "start": [300, 312],
                    "end": [759, 312],
                    "thickness": 25,
                    "length": 460,
                    "color": "#93C4FF",
                    "source": "measured",
                }
            ]
        },
        "surfaces": [
            {
                "box": [20, 170, 980, 490],
                "background": "#FFFFFF",
                "borderColor": "#E0E0E0",
                "borderWidth": 1,
                "cornerRadius": 16,
                "source": "measured",
            }
        ],
        "tree": {
            "id": 0,
            "kind": "screen",
            "role": "screen",
            "box": [0, 0, 1000, 500],
            "elements": [],
            "children": [
                {
                    "id": 10,
                    "kind": "section",
                    "role": "hero",
                    "box": [20, 10, 980, 160],
                    "elements": [{"id": 1}],
                    "children": [],
                }
            ],
        },
        "elements": [
            {
                "id": 1,
                "kind": "text",
                "text": "THE SUMMER DRIVE",
                "box": [40, 20, 900, 140],
                "confidence": 0.98,
                "font": {
                    "family": "custom",
                    "fontSize": 120,
                    "color": "#0078FF",
                },
            },
            {
                "id": 2,
                "kind": "image",
                "box": [300, 180, 700, 460],
                "texture": True,
            },
            {
                "id": 3,
                "kind": "image",
                "box": [30, 10, 920, 150],
                "texture": True,
            },
            {
                "id": 4,
                "kind": "button",
                "box": [40, 400, 220, 470],
                "background": "#FFF8F1",
                "borderColor": "#0078FF",
                "borderWidth": 2,
                "cornerRadius": 24,
            },
            {
                "id": 5,
                "kind": "text",
                "text": "TICKETS",
                "box": [80, 420, 180, 450],
                "confidence": 0.99,
                "font": {"family": "custom", "fontSize": 30},
            },
            {
                "id": 6,
                "kind": "icon",
                "icon": "arrow-right",
                "box": [900, 420, 930, 450],
                "color": "#0078FF",
            },
        ],
    }
    image = np.full((500, 1000, 3), 200, np.uint8)

    doc = build_document(dump, image, profile="reconstruct", target_kind="web")

    spec = doc["reconstruction"]
    assert spec["targetKind"] == "web"
    assert spec["representationPolicy"] == {
        "liveTextRequired": True,
        "selectableTextRequired": True,
        "semanticControlsRequired": True,
        "rasterTextAllowed": False,
        "referenceSlicesAllowed": False,
        "fullReferenceScreenshotAllowed": False,
        "rasterLayoutStructureAllowed": False,
        "symbolArtAsTextRequired": True,
    }
    assert all(
        text["fontStrategy"] == "match-by-glyph-metrics"
        for text in spec["text"]
    )
    assert [asset["elementId"] for asset in spec["allowedRasterRegions"]] == [2]
    assert spec["allowedRasterRegions"][0]["strategy"] == (
        "extract-source-crop-verbatim"
    )
    assert "sens_ask" in spec["rasterAssetRule"]["prohibitedFollowUps"]
    assert "original reference" in spec["allowedRasterRegions"][0]["implementation"]
    assert spec["primaryAsset"]["elementId"] == 2
    assert spec["excludedRasterCandidates"][0]["elementId"] == 3
    assert spec["excludedRasterCandidates"][0]["reason"] == "overlaps-live-text"
    assert [line["boxSource"] for line in spec["structuralLines"]] == [
        [20, 160, 980, 162]
    ]
    assert spec["visualControlCandidates"][0]["interaction"] == (
        "semantic-control-required"
    )
    assert spec["visualControlCandidates"][0]["borderColor"] == "#0078FF"
    assert spec["visualControlCandidates"][0]["cornerRadius"] == 24
    assert spec["text"][0]["color"] == "#0078FF"
    assert spec["layoutRegions"][0]["role"] == "hero"
    assert spec["layoutRegions"][0]["elementIds"] == [1]
    assert spec["surfaces"][0]["background"] == "#FFFFFF"
    assert spec["icons"] == [
        {
            "elementId": 6,
            "name": "arrow-right",
            "boxSource": [900, 420, 930, 450],
            "color": "#0078FF",
            "strategy": "css-or-inline-svg",
            "source": "measured",
        }
    ]
    assert spec["workflow"]["state"] == "needs-focus"
    assert spec["workflow"]["nextAction"] == "execute-returned-focus-plan"
    assert "shell-image-analysis" in spec["workflow"]["forbiddenActions"]
    assert spec["completionGate"]["tool"] == "sens_review"
    assert spec["completionGate"]["requires"] == ["visual-pass", "web-pass"]
    assert "repairHints" in spec["completionGate"]["repairPolicy"]
    assert any("iterationPolicy" in rule for rule in spec["implementationRules"])
    assert any("focusPlan is empty" in rule for rule in spec["implementationRules"])


def test_web_reconstruction_forbids_rasterizing_detected_symbol_art() -> None:
    symbol_text = "..◆◆..\n.◆◆◆◆.\n◆◆..◆◆"
    dump = {
        **DUMP,
        "symbolArt": [
            {
                "kind": "symbol-art",
                "box": [0, 180, 1000, 500],
                "text": symbol_text,
                "rows": 3,
                "columns": 6,
                "cellWidth": 12,
                "rowPitch": 24,
                "alphabet": [".", "◆"],
                "confidence": 0.97,
                "source": "measured",
                "method": "regular-dot-diamond-grid",
            }
        ],
        "elements": [
            {
                "id": 9,
                "kind": "image",
                "box": [0, 180, 1000, 500],
                "texture": True,
            }
        ],
    }
    image = np.zeros((500, 1000, 3), np.uint8)

    doc = build_document(dump, image, profile="reconstruct", target_kind="web")

    spec = doc["reconstruction"]
    assert spec["symbolArt"] == [
        {
            "text": symbol_text,
            "boxSource": [0, 180, 1000, 500],
            "rows": 3,
            "columns": 6,
            "cellWidth": 12,
            "rowPitch": 24,
            "alphabet": [".", "◆"],
            "confidence": 0.97,
            "strategy": "render-as-live-selectable-monospace-text",
            "source": "measured",
            "method": "regular-dot-diamond-grid",
        }
    ]
    assert spec["allowedRasterRegions"] == []
    assert spec["primaryAsset"] is None
    assert spec["excludedRasterCandidates"][0]["reason"] == (
        "overlaps-live-symbol-art"
    )


def test_reconstruction_focus_actions_keep_profile_and_compact_response() -> None:
    dump = {
        **DUMP,
        "ocr": [
            {"text": "06.24.21", "box": [20, 30, 100, 42], "confidence": 0.45}
        ],
    }
    image = np.full((500, 1000, 3), 200, np.uint8)

    doc = build_document(dump, image, profile="reconstruct")

    zoom = next(action for action in doc["nextActions"] if action["tool"] == "sens_zoom")
    assert zoom["arguments"]["profile"] == "reconstruct"
    assert zoom["arguments"]["response"] == "compact"


def test_missing_text_discovery_precedes_four_known_text_verifications() -> None:
    elements = [
        {
            "id": index + 1,
            "kind": "text",
            "text": f"Known line {index + 1}",
            "box": [600, 300 + index * 30, 850, 320 + index * 30],
            "confidence": 0.6,
            "verified": False,
            "font": {"fontSize": 20},
        }
        for index in range(4)
    ]
    dump = {
        **DUMP,
        "elements": elements,
        "ocr": [
            {
                "text": element["text"],
                "box": element["box"],
                "confidence": element["confidence"],
            }
            for element in elements
        ],
        "attention": [
            {
                "box": [0, 0, 400, 220],
                "score": 0.8,
                "why": "text+contrast density",
            }
        ],
    }
    image = np.full((500, 1000, 3), 220, np.uint8)

    doc = build_document(dump, image, profile="reconstruct")

    focus_plan = doc["reconstruction"]["focusPlan"]
    assert len(focus_plan) == 4
    assert focus_plan[0]["reasons"] == ["unresolved_text_density"]
    assert "unresolved visible text" in focus_plan[0]["reason"]


def test_reconstruction_keeps_high_confidence_ocr_disagreement_as_candidate() -> None:
    dump = {
        **DUMP,
        "elements": [
            {
                "id": 1,
                "kind": "text",
                "text": "THE SUMMER",
                "box": [40, 30, 800, 150],
                "confidence": 0.94,
                "verified": False,
                "method": "rapidocr-multiscale-disagreement",
                "alternatives": [
                    {"text": "HЕMЕR", "confidence": 0.71, "scale": 1.0},
                    {"text": "THE SUMMER", "confidence": 0.94, "scale": 1.5},
                ],
                "font": {"family": "custom", "fontSize": 120},
            }
        ],
    }
    image = np.full((500, 1000, 3), 200, np.uint8)

    doc = build_document(dump, image, profile="reconstruct")

    [text] = doc["reconstruction"]["text"]
    assert text["status"] == "candidate"
    assert text["method"] == "rapidocr-multiscale-disagreement"
    assert len(text["alternatives"]) == 2
    text_claim = next(claim for claim in doc["claims"] if claim["id"] == "element.1.text")
    assert text_claim["method"] == "rapidocr-multiscale-disagreement"
    assert text_claim["confidence"] == 0.94
    [verification] = doc["reconstruction"]["textVerificationPlan"]
    assert verification["tool"] == "sens_zoom"
    assert verification["arguments"]["profile"] == "reconstruct"
    assert verification["arguments"]["region"] == {
        "x": 40,
        "y": 30,
        "width": 760,
        "height": 120,
    }


def test_verified_multiscale_ocr_consensus_is_confirmed_without_vlm() -> None:
    dump = {
        **DUMP,
        "elements": [
            {
                "id": 1,
                "kind": "text",
                "text": "Explore Chains",
                "box": [40, 30, 220, 60],
                "confidence": 0.998,
                "verified": True,
                "method": "rapidocr-multiscale-consensus",
                "alternatives": [
                    {"text": "Explore Chains", "confidence": 0.99, "scale": 1.0},
                    {"text": "Explore Chains", "confidence": 0.998, "scale": 1.5},
                ],
                "font": {"fontSize": 20},
            }
        ],
    }
    image = np.full((500, 1000, 3), 220, np.uint8)

    doc = build_document(dump, image, profile="reconstruct")

    [text] = doc["reconstruction"]["text"]
    assert text["status"] == "confirmed"
    assert text["resolutionStatus"] == "confirmed"
    assert text["preferredValue"] is None
    assert text["confirmedBy"] == ["rapidocr-multiscale-consensus"]


def test_reconstruction_treats_unlabelled_control_detection_as_decoration() -> None:
    dump = {
        **DUMP,
        "elements": [
            {
                "id": 1,
                "kind": "text",
                "text": "TICKETS",
                "box": [20, 20, 100, 40],
                "confidence": 0.99,
                "font": {},
            },
            {
                "id": 2,
                "kind": "button",
                "box": [400, 200, 520, 300],
                "background": "#63AF81",
                "borderColor": None,
            },
        ],
    }
    image = np.full((500, 1000, 3), 200, np.uint8)

    doc = build_document(dump, image, profile="reconstruct")

    button = next(element for element in doc["elements"] if element["id"] == 2)
    assert button["kind"] == "decorative-shape"
    assert doc["reconstruction"]["visualControlCandidates"] == []
    assert doc["reconstruction"]["decorativeShapes"][0]["elementId"] == 2


def test_reconstruction_defers_semantics_to_bounded_focus_regions() -> None:
    image = np.full((500, 1000, 3), 200, np.uint8)

    class UnexpectedVlm(_FakeVlm):
        def vibe(self, path):
            raise AssertionError("reconstruction must not invoke full-image vibe")

        def describe(self, path, box):
            raise AssertionError("reconstruction must not caption graphics on the critical path")

        def transcribe(self, path, box):
            raise AssertionError("full-image reconstruction must not invoke the VLM")

    doc = build_document(
        DUMP,
        image,
        vlm=UnexpectedVlm(),
        image_path="poster.png",
        profile="reconstruct",
        max_semantic_calls=1,
    )

    assert doc["reconstruction"]["semanticTextCandidate"] is None
    assert doc["reconstruction"]["semanticStrategy"]["mode"] == "focused-regions"
    assert doc["reconstruction"]["semanticStrategy"]["fullImageCall"] is False
    assert doc["header"]["vibe"] is None


def test_independent_vlm_text_agreement_can_confirm_ocr() -> None:
    class MatchingVlm(_FakeVlm):
        def transcribe(self, path, box):
            return "THE SUMMER\nDRIVE"

    dump = {
        **DUMP,
        "coordinates": {
            "sourceSize": [1000, 500],
            "regionInSource": [20, 20, 820, 180],
            "analysisSize": [1000, 500],
            "analysisToSource": {
                "scaleX": 0.8,
                "scaleY": 0.32,
                "offsetX": 20.0,
                "offsetY": 20.0,
            },
        },
        "elements": [
            {
                "id": 1,
                "kind": "text",
                "text": "THE SUMMER",
                "box": [40, 30, 800, 150],
                "confidence": 0.97,
                "verified": True,
                "method": "rapidocr-multiscale-consensus",
                "font": {},
            }
        ],
    }
    image = np.full((500, 1000, 3), 200, np.uint8)

    doc = build_document(
        dump,
        image,
        vlm=MatchingVlm(),
        image_path="poster.png",
        profile="reconstruct",
        max_semantic_calls=1,
    )

    [text] = doc["reconstruction"]["text"]
    assert text["status"] == "confirmed"
    assert text["confirmedBy"] == [
        "rapidocr-multiscale-consensus",
        "local-vlm-region-transcription",
    ]
    assert doc["reconstruction"]["semanticTextCandidate"]["sourceBox"] == [
        20,
        20,
        820,
        180,
    ]


def test_regional_vlm_can_prefer_a_low_confidence_text_without_zoom_loop() -> None:
    class DateVlm(_FakeVlm):
        def inspect_text(self, path, box):
            assert box == [102, 339, 527, 490]
            return {
                "text": "06.24.21",
                "typography": {
                    "class": "serif",
                    "contrast": "high",
                    "width": "condensed",
                    "weight": "regular",
                    "case": "numeric",
                    "confidence": 0.94,
                },
            }

    dump = {
        **DUMP,
        "coordinates": {
            "sourceSize": [2557, 1273],
            "regionInSource": [102, 339, 527, 490],
            "analysisSize": [1441, 512],
            "analysisToSource": {
                "scaleX": 425 / 1441,
                "scaleY": 151 / 512,
                "offsetX": 102.0,
                "offsetY": 339.0,
            },
        },
        "image": {"width": 1441, "height": 512},
        "elements": [
            {
                "id": 1,
                "kind": "text",
                "text": "O642",
                "box": [0, 0, 1441, 512],
                "confidence": 0.66,
                "font": {
                    "capHeight": 391,
                    "avgGlyphWidth": 151.4,
                    "fontSize": 536,
                },
            }
        ],
    }
    image = np.full((512, 1441, 3), 200, np.uint8)

    doc = build_document(
        dump,
        image,
        vlm=DateVlm(),
        image_path="poster.png",
        profile="reconstruct",
        max_semantic_calls=1,
    )

    [text] = doc["reconstruction"]["text"]
    assert text["value"] == "O642"
    assert text["preferredValue"] == "06.24.21"
    assert text["resolutionStatus"] == "vlm-preferred-candidate"
    assert text["fontFeatures"]["capHeight"] == 115.3
    assert text["fontFeatures"]["coordinateSpace"] == "source-pixels"
    assert text["typographyCandidate"] == {
        "class": "serif",
        "contrast": "high",
        "width": "condensed",
        "weight": "regular",
        "case": "numeric",
        "confidence": 0.94,
        "status": "candidate",
        "epistemic": "inferred",
        "method": "local-vlm-region-text-inspection",
    }
    assert doc["reconstruction"]["semanticTextCandidate"]["typography"]["class"] == "serif"
    assert doc["reconstruction"]["focusPlan"] == []
    assert doc["reconstruction"]["semanticStrategy"]["mode"] == (
        "focused-region-complete"
    )
    assert not any(action["tool"] == "sens_zoom" for action in doc["nextActions"])


def test_regional_typography_runs_map_mixed_styles_to_their_ocr_lines() -> None:
    class MixedVlm(_FakeVlm):
        def inspect_text(self, path, box):
            return {
                "text": "WE RENTED IT OUT\nSTANDARD\nHALL",
                "typography": None,
                "runs": [
                    {
                        "text": "WE RENTED IT OUT",
                        "class": "sans-serif",
                        "contrast": "low",
                        "width": "normal",
                        "weight": "regular",
                        "case": "uppercase",
                        "confidence": 0.91,
                    },
                    {
                        "text": "STANDARD\nHALL",
                        "class": "serif",
                        "contrast": "high",
                        "width": "condensed",
                        "weight": "regular",
                        "case": "uppercase",
                        "confidence": 0.96,
                    },
                ],
            }

    dump = {
        **DUMP,
        "coordinates": {
            "sourceSize": [1000, 500],
            "regionInSource": [100, 100, 800, 450],
            "analysisSize": [700, 350],
            "analysisToSource": {
                "scaleX": 1.0,
                "scaleY": 1.0,
                "offsetX": 100.0,
                "offsetY": 100.0,
            },
        },
        "image": {"width": 700, "height": 350},
        "elements": [
            {
                "id": 1,
                "kind": "text",
                "text": "WE RENTED IT OUT",
                "box": [10, 10, 250, 50],
                "confidence": 0.98,
                "font": {},
            },
            {
                "id": 2,
                "kind": "text",
                "text": "STIANDARD",
                "box": [10, 80, 400, 160],
                "confidence": 0.98,
                "font": {},
            },
            {
                "id": 3,
                "kind": "text",
                "text": "HALL",
                "box": [10, 170, 250, 250],
                "confidence": 0.98,
                "font": {},
            },
        ],
    }

    doc = build_document(
        dump,
        np.full((350, 700, 3), 200, np.uint8),
        vlm=MixedVlm(),
        image_path="poster.png",
        profile="reconstruct",
        max_semantic_calls=1,
    )

    entries = {entry["value"]: entry for entry in doc["reconstruction"]["text"]}
    assert entries["WE RENTED IT OUT"]["typographyCandidate"]["class"] == "sans-serif"
    assert entries["STIANDARD"]["typographyCandidate"]["class"] == "serif"
    assert entries["STIANDARD"]["preferredValue"] == "STANDARD"
    assert entries["STIANDARD"]["resolutionStatus"] == "vlm-preferred-candidate"
    assert entries["HALL"]["typographyCandidate"]["contrast"] == "high"


def test_regional_analysis_is_terminal_even_when_semantics_are_unavailable() -> None:
    dump = {
        **DUMP,
        "coordinates": {
            "sourceSize": [2557, 1273],
            "regionInSource": [102, 339, 527, 490],
            "analysisSize": [1441, 512],
            "analysisToSource": {
                "scaleX": 425 / 1441,
                "scaleY": 151 / 512,
                "offsetX": 102.0,
                "offsetY": 339.0,
            },
        },
        "image": {"width": 1441, "height": 512},
        "elements": [
            {
                "id": 1,
                "kind": "text",
                "text": "O642",
                "box": [0, 0, 1441, 512],
                "confidence": 0.66,
                "font": {"capHeight": 391, "fontSize": 536},
            }
        ],
    }
    image = np.full((512, 1441, 3), 200, np.uint8)

    doc = build_document(
        dump,
        image,
        vlm=None,
        image_path="poster.png",
        profile="reconstruct",
        target_kind="web",
    )

    assert doc["reconstruction"]["focusPlan"] == []
    assert doc["reconstruction"]["textVerificationPlan"] == []
    assert doc["reconstruction"]["semanticStrategy"]["mode"] == (
        "focused-region-terminal"
    )
    assert [action["tool"] for action in doc["nextActions"]] == ["sens_review"]
