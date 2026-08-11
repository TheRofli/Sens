import json
from pathlib import Path

import cv2
import numpy as np

from sight import web_review
from sight.web_review import combine_review, evaluate_web_integrity, review_web


SPEC = {
    "targetKind": "web",
    "canvas": {"width": 1000, "height": 500},
    "text": [
        {
            "elementId": 1,
            "value": "THE SUMMER DRIVE",
            "boxSource": [40, 20, 900, 140],
            "confidence": 0.98,
            "fontSize": 96,
            "capHeight": 70,
            "avgGlyphWidth": 54.0,
            "fontFamilyCandidate": "archivo-black",
        },
        {
            "elementId": 5,
            "value": "TICKETS",
            "boxSource": [80, 420, 180, 450],
            "confidence": 0.99,
        },
    ],
    "visualControlCandidates": [
        {
            "elementId": 4,
            "boxSource": [40, 400, 220, 470],
            "labelElementIds": [5],
            "interaction": "semantic-control-required",
        }
    ],
    "structuralLines": [
        {
            "orientation": "horizontal",
            "boxSource": [20, 160, 980, 162],
        }
    ],
    "allowedRasterRegions": [
        {
            "elementId": 2,
            "kind": "illustration-or-photo",
            "boxSource": [300, 180, 700, 390],
        }
    ],
}


GOOD_CAPTURE = {
    "settings": {"viewport": {"width": 1000, "height": 500}, "dpr": 1.0},
    "textNodes": [
        {
            "text": "THE SUMMER DRIVE",
            "box": [40, 20, 900, 140],
            "userSelect": "text",
            "style": {
                "fontFamily": "Arial, sans-serif",
                "fontSize": "92px",
                "fontWeight": "900",
                "lineHeight": "100px",
                "color": "rgb(0, 111, 255)",
            },
            "visible": True,
        },
        {
            "text": "TICKETS",
            "box": [80, 420, 180, 450],
            "userSelect": "auto",
            "style": {
                "fontFamily": "Arial, sans-serif",
                "fontSize": "24px",
                "fontWeight": "400",
                "color": "rgb(0, 111, 255)",
            },
            "visible": True,
        },
    ],
    "semanticControls": [
        {
            "tag": "button",
            "role": "button",
            "name": "TICKETS",
            "box": [40, 400, 220, 470],
            "pointerEvents": "auto",
            "disabled": False,
            "style": {
                "background": "rgba(0, 0, 0, 0)",
                "border": "2px solid rgb(0, 111, 255)",
                "borderRadius": "35px",
            },
            "visible": True,
        }
    ],
    "structuralLines": [
        {
            "orientation": "horizontal",
            "box": [20, 160, 980, 162],
            "thickness": 2,
            "color": "rgb(0, 111, 255)",
            "source": "thin-fill",
        }
    ],
    "rasterElements": [
        {
            "kind": "img",
            "box": [300, 180, 700, 390],
            "src": "car.png",
            "visible": True,
        }
    ],
}


def test_real_dom_text_controls_and_allowed_illustration_pass() -> None:
    result = evaluate_web_integrity(SPEC, GOOD_CAPTURE)

    assert result["webPass"] is True
    assert result["blockingReasons"] == []
    assert result["textCoverage"] == {
        "referenceCount": 2,
        "liveCount": 2,
        "selectableCount": 2,
        "missingElementIds": [],
        "contentMismatchElementIds": [],
        "unselectableElementIds": [],
    }
    assert result["controlCoverage"]["semanticCount"] == 1
    assert result["rasterAudit"]["allowedCount"] == 1


def test_approved_alpha_masked_background_does_not_count_as_full_screenshot() -> None:
    spec = {
        **SPEC,
        "allowedRasterRegions": [
            *SPEC["allowedRasterRegions"],
            {
                "elementId": "background-artwork",
                "artifactId": "raster:bg123",
                "kind": "alpha-masked-background-artwork",
                "boxSource": [0, 0, 1000, 500],
                "alphaProtected": True,
                "semanticContentRemoved": True,
                "protectionVersion": 3,
                "semanticResidualProtection": {
                    "displayTextDiscoveryComplete": True,
                    "displayTextCandidateCount": 0,
                    "method": "rapidocr-downscaled-display-scan",
                },
                "protectionPolicy": {
                    "backgroundOnly": True,
                    "liveText": "full-box-inpainted-under-live-dom",
                    "controlDecoration": "removed-from-raster-recreated-as-semantic-css",
                    "surfaces": "removed-from-raster-recreated-as-css",
                    "structuralLines": "removed-from-raster-recreated-as-css-vector",
                },
            },
        ],
    }
    capture = {
        **GOOD_CAPTURE,
        "rasterElements": [
            *GOOD_CAPTURE["rasterElements"],
            {
                "kind": "img",
                "box": [0, 0, 1000, 500],
                "src": "background.png",
                "sensRasterRole": "alpha-masked-background-artwork",
                "sensArtifactId": "raster:bg123",
                "visible": True,
            },
        ],
    }

    result = evaluate_web_integrity(spec, capture)

    assert result["webPass"] is True
    assert result["rasterAudit"]["allowedCount"] == 2
    background = result["rasterAudit"]["elements"][1]
    assert background["alphaMaskedBackground"] is True
    assert background["overlapsText"] is False


def test_verified_browser_source_background_can_cover_live_dom_text() -> None:
    content_sha256 = "a" * 64
    artifact_id = f"raster:{content_sha256[:16]}"
    spec = {
        **SPEC,
        "allowedRasterRegions": [
            *SPEC["allowedRasterRegions"],
            {
                "elementId": "browser-source-background",
                "artifactId": artifact_id,
                "kind": "browser-source-background-artwork",
                "boxSource": [-50, -40, 1050, 540],
                "semanticContentRemoved": True,
                "protectionVersion": 4,
                "contentSha256": content_sha256,
                "mediaType": "image/avif",
                "source": "observed",
                "method": "verified-playwright-response-body",
                "protectionPolicy": {
                    "backgroundOnly": True,
                    "liveText": "separate-observed-live-dom",
                    "controlDecoration": "separate-semantic-css",
                    "fullReferenceScreenshot": False,
                },
            },
        ],
    }
    capture = {
        **GOOD_CAPTURE,
        "rasterElements": [
            *GOOD_CAPTURE["rasterElements"],
            {
                "kind": "img",
                "box": [-50, -40, 1050, 540],
                "src": "assets/hero.avif",
                "sensRasterRole": "browser-source-background-artwork",
                "sensArtifactId": artifact_id,
                "visible": True,
            },
        ],
        "sourceRasterAssets": [
            {
                "rasterIndex": 1,
                "sha256": content_sha256,
                "mediaType": "image/avif",
                "source": "observed",
                "method": "playwright-response-body",
            }
        ],
    }

    result = evaluate_web_integrity(spec, capture)

    assert result["webPass"] is True
    assert result["blockingReasons"] == []
    background = result["rasterAudit"]["elements"][1]
    assert background["browserSourceBackground"] is True
    assert background["overlapsText"] is False


def test_overlapping_background_layers_match_their_declared_raster_roles() -> None:
    content_sha256 = "a" * 64
    source_artifact_id = f"raster:{content_sha256[:16]}"
    overlay_artifact_id = "raster:composite-overlay"
    spec = {
        **SPEC,
        "allowedRasterRegions": [
            *SPEC["allowedRasterRegions"],
            {
                "elementId": "browser-source-background",
                "artifactId": source_artifact_id,
                "kind": "browser-source-background-artwork",
                "boxSource": [-50, -40, 1050, 540],
                "semanticContentRemoved": True,
                "protectionVersion": 4,
                "contentSha256": content_sha256,
                "mediaType": "image/avif",
                "source": "observed",
                "method": "verified-playwright-response-body",
                "protectionPolicy": {
                    "backgroundOnly": True,
                    "liveText": "separate-observed-live-dom",
                    "controlDecoration": "separate-semantic-css",
                    "fullReferenceScreenshot": False,
                },
            },
            {
                "elementId": "browser-source-composite-overlay",
                "artifactId": overlay_artifact_id,
                "kind": "alpha-masked-background-artwork",
                "boxSource": [0, 0, 1000, 500],
                "alphaProtected": True,
                "semanticContentRemoved": True,
                "protectionVersion": 5,
                "semanticResidualProtection": {
                    "displayTextDiscoveryComplete": True,
                    "displayTextCandidateCount": 1,
                    "method": "rapidocr-downscaled-display-scan",
                },
                "protectionPolicy": {
                    "backgroundOnly": True,
                    "liveText": "transparent-holes-reveal-verified-browser-source-under-live-dom",
                    "controlDecoration": "transparent-holes-reveal-verified-browser-source-under-semantic-css",
                    "fullReferenceScreenshot": False,
                },
            },
        ],
    }
    capture = {
        **GOOD_CAPTURE,
        "rasterElements": [
            *GOOD_CAPTURE["rasterElements"],
            {
                "kind": "img",
                "box": [-50, -40, 1050, 540],
                "src": "assets/hero.avif",
                "sensRasterRole": "browser-source-background-artwork",
                "sensArtifactId": source_artifact_id,
                "visible": True,
            },
            {
                "kind": "img",
                "box": [0, 0, 1000, 500],
                "src": "assets/composite-overlay.png",
                "sensRasterRole": "alpha-masked-background-artwork",
                "sensArtifactId": overlay_artifact_id,
                "visible": True,
            },
        ],
        "sourceRasterAssets": [
            {
                "rasterIndex": 1,
                "sha256": content_sha256,
                "mediaType": "image/avif",
                "source": "observed",
                "method": "playwright-response-body",
            }
        ],
    }

    result = evaluate_web_integrity(spec, capture)

    assert result["webPass"] is True
    source_audit, overlay_audit = result["rasterAudit"]["elements"][1:]
    assert source_audit["browserSourceBackground"] is True
    assert overlay_audit["alphaMaskedBackground"] is True


def test_browser_source_background_with_changed_bytes_is_blocked() -> None:
    content_sha256 = "a" * 64
    artifact_id = f"raster:{content_sha256[:16]}"
    spec = {
        **SPEC,
        "allowedRasterRegions": [
            *SPEC["allowedRasterRegions"],
            {
                "elementId": "browser-source-background",
                "artifactId": artifact_id,
                "kind": "browser-source-background-artwork",
                "boxSource": [0, 0, 1000, 500],
                "semanticContentRemoved": True,
                "protectionVersion": 4,
                "contentSha256": content_sha256,
                "mediaType": "image/avif",
                "source": "observed",
                "method": "verified-playwright-response-body",
                "protectionPolicy": {
                    "backgroundOnly": True,
                    "liveText": "separate-observed-live-dom",
                    "fullReferenceScreenshot": False,
                },
            },
        ],
    }
    capture = {
        **GOOD_CAPTURE,
        "rasterElements": [
            *GOOD_CAPTURE["rasterElements"],
            {
                "kind": "img",
                "box": [0, 0, 1000, 500],
                "src": "assets/hero.avif",
                "sensRasterRole": "browser-source-background-artwork",
                "sensArtifactId": artifact_id,
                "visible": True,
            },
        ],
        "sourceRasterAssets": [
            {
                "rasterIndex": 1,
                "sha256": "b" * 64,
                "mediaType": "image/avif",
                "source": "observed",
                "method": "playwright-response-body",
            }
        ],
    }

    result = evaluate_web_integrity(spec, capture)

    codes = {reason["code"] for reason in result["blockingReasons"]}
    assert "untrusted-background-raster" in codes
    assert "full-reference-raster" in codes
    assert "raster-overlaps-text" in codes


def test_v2_background_without_display_text_scan_is_blocked() -> None:
    spec = {
        **SPEC,
        "allowedRasterRegions": [
            *SPEC["allowedRasterRegions"],
            {
                "elementId": "background-artwork",
                "artifactId": "raster:bg-v2",
                "kind": "alpha-masked-background-artwork",
                "boxSource": [0, 0, 1000, 500],
                "alphaProtected": True,
                "semanticContentRemoved": True,
                "protectionVersion": 2,
                "protectionPolicy": {
                    "backgroundOnly": True,
                    "liveText": "full-box-inpainted-under-live-dom",
                    "controlDecoration": "removed-from-raster-recreated-as-semantic-css",
                },
            },
        ],
    }
    capture = {
        **GOOD_CAPTURE,
        "rasterElements": [
            *GOOD_CAPTURE["rasterElements"],
            {
                "kind": "img",
                "box": [0, 0, 1000, 500],
                "src": "background-v2.png",
                "sensRasterRole": "alpha-masked-background-artwork",
                "sensArtifactId": "raster:bg-v2",
                "visible": True,
            },
        ],
    }

    result = evaluate_web_integrity(spec, capture)

    assert result["webPass"] is False
    assert "untrusted-background-raster" in {
        reason["code"] for reason in result["blockingReasons"]
    }


def test_legacy_full_canvas_background_that_preserves_ui_is_blocked() -> None:
    spec = {
        **SPEC,
        "allowedRasterRegions": [
            *SPEC["allowedRasterRegions"],
            {
                "elementId": "background-artwork",
                "artifactId": "raster:legacy",
                "kind": "alpha-masked-background-artwork",
                "boxSource": [0, 0, 1000, 500],
                "alphaProtected": True,
                "protectionPolicy": {
                    "controlDecoration": "preserved-in-background-behind-semantic-dom",
                    "surfaces": "preserved-in-background-no-duplicate-css-surface",
                },
            },
        ],
    }
    capture = {
        **GOOD_CAPTURE,
        "rasterElements": [
            *GOOD_CAPTURE["rasterElements"],
            {
                "kind": "img",
                "box": [0, 0, 1000, 500],
                "src": "legacy-background.png",
                "sensRasterRole": "alpha-masked-background-artwork",
                "sensArtifactId": "raster:legacy",
                "visible": True,
            },
        ],
    }

    result = evaluate_web_integrity(spec, capture)

    assert result["webPass"] is False
    codes = {entry["code"] for entry in result["blockingReasons"]}
    assert "untrusted-background-raster" in codes
    assert "full-reference-raster" in codes


def test_string_element_ids_remain_stable_in_web_review() -> None:
    spec = {
        **SPEC,
        "text": [
            {**SPEC["text"][0], "elementId": "heading-main"},
            {**SPEC["text"][1], "elementId": "button-label"},
        ],
        "visualControlCandidates": [
            {
                **SPEC["visualControlCandidates"][0],
                "elementId": "inferred-control-4",
                "labelElementIds": ["button-label"],
            }
        ],
    }

    result = evaluate_web_integrity(spec, GOOD_CAPTURE)

    assert result["webPass"] is True
    assert [
        match["referenceElementId"] for match in result["textMatches"]
    ] == ["heading-main", "button-label"]
    assert result["controlMatches"][0]["referenceElementId"] == "inferred-control-4"


def test_review_loads_the_resolved_web_contract_without_reanalysis(tmp_path) -> None:
    reference = tmp_path / "reference.png"
    cv2.imwrite(str(reference), np.full((500, 1000, 3), 255, np.uint8))
    contract = tmp_path / "contract.json"
    resolved = {
        **SPEC,
        "text": [
            {**SPEC["text"][0], "preferredValue": "Resolved heading"},
            SPEC["text"][1],
        ],
    }
    contract.write_text(
        json.dumps({"reconstruction": resolved}),
        encoding="utf-8",
    )

    reconstruction = web_review._reference_reconstruction(
        str(reference), str(contract)
    )

    assert reconstruction["text"][0]["preferredValue"] == "Resolved heading"


def test_review_returns_measured_dom_geometry_and_css_repair_hints() -> None:
    capture = {
        **GOOD_CAPTURE,
        "textNodes": [
            {
                **GOOD_CAPTURE["textNodes"][0],
                "box": [52, 28, 872, 138],
            },
            GOOD_CAPTURE["textNodes"][1],
        ],
        "semanticControls": [
            {
                **GOOD_CAPTURE["semanticControls"][0],
                "box": [48, 406, 218, 466],
            }
        ],
        "structuralLines": [
            {
                **GOOD_CAPTURE["structuralLines"][0],
                "box": [20, 172, 930, 175],
            }
        ],
    }

    result = evaluate_web_integrity(SPEC, capture)

    heading_match = next(
        match
        for match in result["textMatches"]
        if match["referenceElementId"] == 1
    )
    assert heading_match["geometryDelta"] == {
        "x": 12,
        "y": 8,
        "width": -40,
        "height": -10,
    }
    heading_hint = next(
        hint
        for hint in result["repairHints"]["text"]
        if hint["referenceElementId"] == 1
    )
    assert heading_hint["referenceBoxSource"] == [40, 20, 900, 140]
    assert heading_hint["candidateBoxSource"] == [52, 28, 872, 138]
    assert heading_hint["referenceTypography"]["fontFamilyCandidate"] == "archivo-black"
    assert heading_hint["candidateStyle"]["fontWeight"] == "900"

    control_hint = result["repairHints"]["controls"][0]
    assert control_hint["kind"] == "semantic-control-geometry"
    assert control_hint["geometryDelta"] == {
        "x": 8,
        "y": 6,
        "width": -10,
        "height": -10,
    }
    line_hint = result["repairHints"]["structure"][0]
    assert line_hint["kind"] == "structural-line-geometry"
    assert line_hint["geometryDelta"] == {
        "x": 0,
        "y": 12,
        "width": -50,
        "height": 1,
    }


def test_displaced_live_text_and_missing_divider_get_actionable_hints() -> None:
    capture = {
        **GOOD_CAPTURE,
        "textNodes": [
            {
                **GOOD_CAPTURE["textNodes"][0],
                "box": [40, 250, 900, 370],
            },
            GOOD_CAPTURE["textNodes"][1],
        ],
        "structuralLines": [],
    }

    result = evaluate_web_integrity(SPEC, capture)

    heading_hint = next(
        hint
        for hint in result["repairHints"]["text"]
        if hint["referenceElementId"] == 1
    )
    assert heading_hint["kind"] == "live-text-missing-or-displaced"
    assert heading_hint["candidateBoxSource"] == [40, 250, 900, 370]
    assert heading_hint["geometryDelta"]["y"] == 230
    assert result["structuralLineCoverage"]["missingIndexes"] == [0]
    assert result["repairHints"]["structure"][0]["kind"] == "structural-line-missing"


def test_raster_slices_fail_even_when_they_visually_cover_reference_content() -> None:
    capture = {
        **GOOD_CAPTURE,
        "textNodes": [],
        "semanticControls": [],
        "rasterElements": [
            {
                "kind": "img",
                "box": [30, 10, 920, 150],
                "src": "heading-crop.png",
                "visible": True,
            },
            {
                "kind": "img",
                "box": [40, 400, 220, 470],
                "src": "tickets-crop.png",
                "visible": True,
            },
        ],
    }

    result = evaluate_web_integrity(SPEC, capture)

    assert result["webPass"] is False
    codes = {reason["code"] for reason in result["blockingReasons"]}
    assert {
        "missing-live-text",
        "missing-semantic-control",
        "raster-overlaps-text",
        "raster-outside-allowed-region",
    } <= codes
    assert result["rasterAudit"]["allowedCount"] == 0


def test_live_but_unselectable_text_is_a_blocker() -> None:
    capture = {
        **GOOD_CAPTURE,
        "textNodes": [
            {**node, "userSelect": "none"} for node in GOOD_CAPTURE["textNodes"]
        ],
    }

    result = evaluate_web_integrity(SPEC, capture)

    assert result["webPass"] is False
    assert result["textCoverage"]["liveCount"] == 2
    assert result["textCoverage"]["selectableCount"] == 0
    assert any(
        reason["code"] == "unselectable-live-text"
        for reason in result["blockingReasons"]
    )


def test_overlapping_wrong_dom_copy_does_not_satisfy_live_text_contract() -> None:
    capture = {
        **GOOD_CAPTURE,
        "textNodes": [
            {**GOOD_CAPTURE["textNodes"][0], "text": "UNRELATED HEADING"},
            GOOD_CAPTURE["textNodes"][1],
        ],
    }

    result = evaluate_web_integrity(SPEC, capture)

    assert result["webPass"] is False
    assert result["textCoverage"]["contentMismatchElementIds"] == [1]
    assert any(
        reason["code"] == "live-text-content-mismatch"
        for reason in result["blockingReasons"]
    )


def test_control_name_must_match_the_measured_visible_label() -> None:
    capture = {
        **GOOD_CAPTURE,
        "semanticControls": [
            {**GOOD_CAPTURE["semanticControls"][0], "name": "Delete account"}
        ],
    }

    result = evaluate_web_integrity(SPEC, capture)

    assert result["webPass"] is False
    assert result["controlCoverage"]["labelMismatchElementIds"] == [4]
    assert any(
        reason["code"] == "semantic-control-label-mismatch"
        for reason in result["blockingReasons"]
    )


def test_css_gradients_are_not_treated_as_raster_cheats() -> None:
    capture = {
        **GOOD_CAPTURE,
        "rasterElements": [
            *GOOD_CAPTURE["rasterElements"],
            {
                "kind": "background-image",
                "box": [0, 0, 1000, 500],
                "src": "linear-gradient(90deg, rgb(0, 0, 0), rgb(255, 255, 255))",
                "visible": True,
            },
        ],
    }

    result = evaluate_web_integrity(SPEC, capture)

    assert result["webPass"] is True
    assert result["rasterAudit"]["observedCount"] == 1


def test_symbol_art_requires_exact_selectable_preformatted_text() -> None:
    symbol_text = "..◆◆..\n.◆◆◆◆.\n◆◆..◆◆"
    spec = {
        **SPEC,
        "text": [],
        "visualControlCandidates": [],
        "allowedRasterRegions": [],
        "symbolArt": [
            {
                "text": symbol_text,
                "boxSource": [0, 180, 1000, 500],
            }
        ],
    }
    good_capture = {
        **GOOD_CAPTURE,
        "textNodes": [
            {
                "text": "..◆◆.. .◆◆◆◆. ◆◆..◆◆",
                "rawText": symbol_text,
                "whiteSpace": "pre",
                "box": [0, 180, 1000, 500],
                "userSelect": "text",
                "visible": True,
            }
        ],
        "semanticControls": [],
        "rasterElements": [],
    }

    passed = evaluate_web_integrity(spec, good_capture)
    failed = evaluate_web_integrity(
        spec,
        {
            **good_capture,
            "textNodes": [],
            "rasterElements": [
                {
                    "kind": "img",
                    "box": [0, 180, 1000, 500],
                    "src": "hands.png",
                    "visible": True,
                }
            ],
        },
    )

    assert passed["webPass"] is True
    assert passed["symbolArtCoverage"]["exactSelectableCount"] == 1
    assert failed["webPass"] is False
    codes = {reason["code"] for reason in failed["blockingReasons"]}
    assert "missing-live-symbol-art" in codes
    assert "raster-overlaps-symbol-art" in codes


def test_combined_completion_requires_visual_and_web_passes() -> None:
    web = evaluate_web_integrity(SPEC, GOOD_CAPTURE)
    passed = combine_review(
        {"verdict": "pass", "canComplete": True, "similarityScore": 0.99},
        web,
    )
    failed = combine_review(
        {
            "verdict": "fail",
            "canComplete": False,
            "similarityScore": 0.8,
            "requiredAction": {
                "kind": "repair_largest_hot_region",
                "region": [10, 20, 900, 220],
                "reason": "Repair and rerun sens_compare.",
            },
            "hotRegions": [{"box": [10, 20, 900, 220], "areaRatio": 0.12}],
            "nextActions": [
                {
                    "tool": "sens_zoom",
                    "arguments": {"region": {"x": 10, "y": 20, "width": 890, "height": 200}},
                }
            ],
        },
        web,
    )

    assert passed["visualPass"] is True
    assert passed["webPass"] is True
    assert passed["canComplete"] is True
    assert passed["verdict"] == "pass"
    assert failed["visualPass"] is False
    assert failed["webPass"] is True
    assert failed["canComplete"] is False
    assert failed["requiredAction"] == "repair-visual"
    assert "nextActions" not in failed["visual"]
    assert failed["visual"]["requiredAction"]["kind"] == (
        "repair-largest-hot-region-from-existing-contract"
    )
    assert failed["repairHints"]["visual"][0]["referenceBoxSource"] == [
        10,
        20,
        900,
        220,
    ]
    assert failed["workflow"]["nextTool"] == "sens_review"
    assert failed["workflow"]["allowedNextTools"] == ["sens_review"]
    assert "sens_zoom" in failed["workflow"]["prohibitedNextTools"]
    assert "sens_zoom" not in str(failed["visual"])


def test_text_similarity_failure_prioritizes_repeated_text_not_verified_vector_wordmark() -> None:
    reconstruction = {
        "text": [
            {
                "elementId": 1,
                "value": "SLUSH CARD WAITLIST IS LIVE",
                "boxSource": [0, 8, 182, 27],
            },
            {
                "elementId": 2,
                "value": "SLUSH CARD WAITLIST IS LIVE",
                "boxSource": [192, 8, 374, 27],
            },
            {
                "elementId": 32,
                "value": "SLUSH",
                "boxSource": [345, 184, 1103, 593],
                "visualRepresentation": (
                    "source-vector-wordmark-with-selectable-live-label"
                ),
            },
        ]
    }
    web = {
        "webPass": True,
        "blockingReasons": [],
        "repairHints": {"text": [], "controls": [], "structure": []},
        "textMatches": [
            {
                "referenceElementId": 1,
                "referenceText": "SLUSH CARD WAITLIST IS LIVE",
                "candidateText": "SLUSH CARD WAITLIST IS LIVE",
                "exact": True,
                "selectable": True,
            },
            {
                "referenceElementId": 2,
                "referenceText": "SLUSH CARD WAITLIST IS LIVE",
                "candidateText": "SLUSH CARD WAITLIST IS LIVE",
                "exact": True,
                "selectable": True,
            },
            {
                "referenceElementId": 32,
                "referenceText": "SLUSH",
                "candidateText": "SLUSH",
                "exact": True,
                "selectable": True,
            },
        ],
    }
    visual = {
        "verdict": "partial",
        "canComplete": False,
        "similarityScore": 0.8804,
        "acceptance": {
            "checks": [
                {"name": "similarity_minimum", "passed": True},
                {
                    "name": "text_similarity_minimum",
                    "passed": False,
                    "actual": 0.641,
                    "threshold": 0.7,
                },
                {"name": "largest_hot_region_maximum", "passed": True},
            ]
        },
        "metrics": {
            "text": {
                "similarity": 0.641,
                "reference": "slush card waitlist is live slush card waitlist is live slush",
                "candidate": "slueh card waitliet is lile slush",
            }
        },
        "hotRegions": [{"box": [343, 182, 853, 610], "areaRatio": 0.013}],
    }

    result = combine_review(visual, web, reconstruction)

    hint = result["repairHints"]["visual"][0]
    assert hint["kind"] == "ocr-text-similarity"
    assert hint["actual"] == 0.641
    assert hint["threshold"] == 0.7
    assert hint["repeatedReferenceTextGroups"][0]["text"] == (
        "SLUSH CARD WAITLIST IS LIVE"
    )
    assert hint["repeatedReferenceTextGroups"][0]["elementIds"] == [1, 2]
    assert hint["verifiedVectorWordmarks"] == [
        {
            "elementId": 32,
            "text": "SLUSH",
            "boxSource": [345, 184, 1103, 593],
        }
    ]
    assert "Do not replace verified source-vector wordmarks" in hint["action"]
    assert result["visual"]["requiredAction"]["kind"] == (
        "repair-ocr-text-rendering-from-existing-contract"
    )


def test_review_no_store_cleans_owned_browser_artifacts(tmp_path, monkeypatch) -> None:
    reference = tmp_path / "reference.png"
    cv2.imwrite(str(reference), np.full((500, 1000, 3), 255, np.uint8))
    owned = tmp_path / "owned-review"

    monkeypatch.setattr(
        web_review.tempfile,
        "mkdtemp",
        lambda **_kwargs: str(owned),
    )
    monkeypatch.setattr(
        web_review,
        "_reference_reconstruction",
        lambda _path, _contract_path=None: SPEC,
    )

    def fake_capture(_url, out_dir, options, *, no_store=False):
        assert no_store is False
        assert options["networkPolicy"] == "candidate"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        screenshot = Path(out_dir) / "candidate.png"
        cv2.imwrite(str(screenshot), np.full((500, 1000, 3), 255, np.uint8))
        return {**GOOD_CAPTURE, "screenshot": str(screenshot), "artifacts": []}

    monkeypatch.setattr(web_review, "capture_url", fake_capture)
    monkeypatch.setattr(
        web_review,
        "compare_images",
        lambda *_args, **_kwargs: {
            "verdict": "pass",
            "canComplete": True,
            "similarityScore": 1.0,
        },
    )

    result = review_web(
        str(reference),
        "http://localhost:8123/index.html",
        no_store=True,
    )

    assert result["canComplete"] is True
    assert result["artifacts"] == []
    assert "screenshot" not in result["capture"]
    assert not owned.exists()
