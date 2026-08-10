from pathlib import Path

import cv2
import numpy as np

from sight.ops import (
    _hydrate_measured_control_geometry,
    _hydrate_background_artwork_layer,
    _hydrate_intrinsic_text_raster_assets,
    _hydrate_navigation_rail,
    _hydrate_measured_surfaces,
    _infer_contextual_ui_structure,
    _plausible_controls,
    _refine_overlapping_raster_candidates,
    _sanitize_web_structure,
)
from sight.perception import _controls_around_text, outlined_controls_around_text
from sight.qa import control_style


CREAM_BGR = np.array((239, 247, 252), dtype=np.uint8)
BLUE_BGR = (255, 120, 0)


def test_navigation_rail_recovers_isolated_icons_and_selected_surface(
    tmp_path,
) -> None:
    image = np.full((240, 400, 3), (230, 228, 229), np.uint8)
    image[:, 60:] = (250, 250, 250)
    cv2.circle(image, (26, 70), 11, (10, 10, 10), -1)
    cv2.circle(image, (26, 118), 8, (90, 90, 90), 2)
    cv2.rectangle(image, (9, 148), (43, 182), (255, 255, 255), -1)
    cv2.circle(image, (26, 156), 2, (70, 70, 70), -1)
    cv2.circle(image, (19, 173), 2, (70, 70, 70), -1)
    cv2.circle(image, (34, 173), 2, (70, 70, 70), -1)
    cv2.line(image, (25, 158), (20, 171), (70, 70, 70), 1)
    cv2.line(image, (27, 158), (33, 171), (70, 70, 70), 1)
    image_path = tmp_path / "rail.png"
    cv2.imwrite(str(image_path), image)
    document = {
        "reconstruction": {
            "targetKind": "web",
            "canvas": {"width": 400, "height": 240},
            "text": [
                {"elementId": 1, "value": "dub", "boxSource": [10, 10, 40, 30]}
            ],
            "surfaces": [
                {"boxSource": [60, 0, 400, 240], "background": "#FAFAFA"}
            ],
            "decorativeShapes": [
                {
                    "elementId": 2,
                    "boxSource": [14, 58, 39, 83],
                    "background": "#050505",
                    "cornerRadius": 1,
                }
            ],
            "icons": [
                {
                    "elementId": 3,
                    "name": "cross",
                    "boxSource": [15, 60, 37, 82],
                }
            ],
        }
    }

    _hydrate_navigation_rail(document, str(image_path))

    reconstruction = document["reconstruction"]
    assert [icon["name"] for icon in reconstruction["icons"]] == [
        "brand-mark",
        "globe",
        "network",
    ]
    assert reconstruction["decorativeShapes"][0]["cornerRadius"] >= 10
    selected = next(
        surface
        for surface in reconstruction["surfaces"]
        if str(surface.get("elementId", "")).startswith("rail-navigation-surface")
    )
    assert selected["boxSource"][0] <= 9
    assert selected["background"] == "#FFFFFF"


def test_repeated_visible_ui_rows_become_provenance_marked_semantic_controls() -> None:
    document = {
        "reconstruction": {
            "targetKind": "web",
            "canvas": {"width": 1000, "height": 600},
            "text": [
                {"elementId": 1, "value": "Overview", "boxSource": [100, 54, 170, 68]},
                {"elementId": 2, "value": "Payouts", "boxSource": [100, 86, 155, 100]},
                {"elementId": 3, "value": "Messages", "boxSource": [100, 118, 170, 132]},
                {
                    "elementId": 4,
                    "preferredValue": "Confirm pending payouts",
                    "boxSource": [748, 142, 890, 156],
                },
                {
                    "elementId": 5,
                    "preferredValue": "Respond to partners",
                    "boxSource": [748, 184, 875, 198],
                },
                {
                    "elementId": 6,
                    "preferredValue": "Review new applications",
                    "boxSource": [748, 226, 895, 240],
                },
            ],
            "surfaces": [
                {"boxSource": [58, 0, 260, 600], "background": "#F6F6F6"},
                {"boxSource": [700, 100, 980, 280], "background": "#FFFFFF"},
            ],
            "icons": [
                {
                    "elementId": 20,
                    "name": "cross",
                    "boxSource": [78, 54, 90, 68],
                    "source": "measured",
                },
                {
                    "elementId": 21,
                    "name": None,
                    "boxSource": [10, 10, 20, 20],
                    "source": "measured-ocr-exclusion",
                },
            ],
            "badges": [],
            "visualControlCandidates": [
                {
                    "elementId": 30,
                    "boxSource": [68, 46, 252, 76],
                    "labelElementIds": [1],
                }
            ],
            "implementationRules": [],
        }
    }

    _infer_contextual_ui_structure(document)
    _infer_contextual_ui_structure(document)

    spec = document["reconstruction"]
    controls = spec["visualControlCandidates"]
    assert len(controls) == 6
    assert {control["semanticRole"] for control in controls} == {"nav", "action"}
    inferred = [control for control in controls if str(control["elementId"]).startswith("inferred-")]
    assert len(inferred) == 5
    assert all(control["source"] == "inferred-from-visible-affordance" for control in inferred)
    assert all(
        control["behavior"] == "local-placeholder-no-invented-destination"
        for control in inferred
    )
    names = {icon["name"] for icon in spec["icons"]}
    assert {"home", "wallet", "message", "user-check"} <= names
    assert all(icon.get("name") is not None for icon in spec["icons"])
    icon_keys = [
        (icon["name"], tuple(icon["boxSource"])) for icon in spec["icons"]
    ]
    assert len(icon_keys) == len(set(icon_keys))
    assert any("do not invent destinations" in rule for rule in spec["implementationRules"])


def test_plain_poster_words_are_not_classified_as_pills() -> None:
    image = np.full((160, 800, 3), CREAM_BGR, np.uint8)
    cv2.putText(
        image,
        "A NO-WORK WORK-EVENT FOR TEAMS",
        (30, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.25,
        BLUE_BGR,
        2,
        cv2.LINE_AA,
    )
    blocks = [{"box": [20, 50, 760, 115], "area": 740 * 65, "kind": "block"}]
    candidates = control_style(image, blocks)
    ocr_items = [
        {
            "text": f"word-{index}",
            "box": [box[0] + 5, box[1] + 12, box[2] - 5, box[3] - 12],
            "confidence": 0.99,
        }
        for index, control in enumerate(candidates)
        for box in [control["box"]]
    ]

    assert candidates, "fixture must exercise the row-splitting heuristic"
    assert _plausible_controls(candidates, CREAM_BGR, ocr_items) == []


def test_measured_closed_outline_remains_a_control() -> None:
    image = np.full((120, 180, 3), CREAM_BGR, np.uint8)
    cv2.rectangle(image, (30, 30), (140, 90), BLUE_BGR, 2)
    cv2.putText(
        image,
        "GO",
        (70, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        BLUE_BGR,
        2,
        cv2.LINE_AA,
    )
    blocks = [{"box": [30, 30, 141, 91], "area": 111 * 61, "kind": "block"}]
    candidates = control_style(image, blocks)

    kept = _plausible_controls(
        candidates,
        CREAM_BGR,
        [{"text": "GO", "box": [70, 50, 100, 75], "confidence": 0.99}],
    )

    assert len(kept) == 1
    assert kept[0]["borderColor"] == "#0078FF"


def test_same_background_wide_outline_around_text_is_a_control() -> None:
    image = np.full((160, 440, 3), CREAM_BGR, np.uint8)
    cv2.rectangle(image, (40, 35), (400, 125), BLUE_BGR, 3, cv2.LINE_AA)
    cv2.putText(
        image,
        "TICKETS",
        (140, 93),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        BLUE_BGR,
        2,
        cv2.LINE_AA,
    )
    ocr = [
        {
            "text": "TICKETS",
            "box": [138, 60, 302, 100],
            "confidence": 0.99,
        }
    ]

    controls = outlined_controls_around_text(image, ocr, [])

    assert len(controls) == 1
    assert controls[0]["box"][0] <= 40
    assert controls[0]["box"][1] <= 35
    assert controls[0]["box"][2] >= 401
    assert controls[0]["box"][3] >= 126
    assert controls[0]["borderColor"] == "#0078FF"
    assert controls[0]["cornerRadius"] == 0
    assert controls[0]["labelText"] == "TICKETS"
    assert controls[0]["boundaryEvidence"]["closed"] is True


def test_measured_pill_outline_reports_its_corner_radius() -> None:
    image = np.full((160, 440, 3), CREAM_BGR, np.uint8)
    cv2.line(image, (85, 35), (355, 35), BLUE_BGR, 3, cv2.LINE_AA)
    cv2.line(image, (85, 125), (355, 125), BLUE_BGR, 3, cv2.LINE_AA)
    cv2.ellipse(image, (85, 80), (45, 45), 0, 90, 270, BLUE_BGR, 3, cv2.LINE_AA)
    cv2.ellipse(image, (355, 80), (45, 45), 0, -90, 90, BLUE_BGR, 3, cv2.LINE_AA)
    cv2.putText(
        image,
        "TICKETS",
        (140, 93),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        BLUE_BGR,
        2,
        cv2.LINE_AA,
    )

    [control] = outlined_controls_around_text(
        image,
        [{"text": "TICKETS", "box": [138, 60, 302, 100], "confidence": 0.99}],
        [],
    )

    assert 40 <= control["cornerRadius"] <= 47


def test_tight_ocr_box_still_recovers_the_complete_closed_outline() -> None:
    image = np.full((120, 280, 3), CREAM_BGR, np.uint8)
    cv2.rectangle(image, (40, 35), (240, 76), BLUE_BGR, 2, cv2.LINE_AA)
    cv2.putText(
        image,
        "A1 SENSE",
        (52, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        BLUE_BGR,
        2,
        cv2.LINE_AA,
    )
    ocr = [{"text": "A1 SENSE", "box": [50, 36, 230, 75], "confidence": 0.99}]

    [control] = outlined_controls_around_text(image, ocr, [])

    assert control["box"][0] <= 40
    assert control["box"][1] <= 35
    assert control["box"][2] >= 241
    assert control["box"][3] >= 77


def test_cached_control_geometry_expands_partial_box_to_measured_outline(
    tmp_path,
) -> None:
    image = np.full((120, 280, 3), CREAM_BGR, np.uint8)
    cv2.rectangle(image, (40, 35), (240, 76), BLUE_BGR, 2, cv2.LINE_AA)
    cv2.putText(
        image,
        "A1 SENSE",
        (52, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        BLUE_BGR,
        2,
        cv2.LINE_AA,
    )
    image_path = tmp_path / "tight-control.png"
    cv2.imwrite(str(image_path), image)
    document = {
        "reconstruction": {
            "targetKind": "web",
            "text": [
                    {
                        "elementId": 1,
                        "value": "A1 SENSE",
                        "boxSource": [40, 35, 241, 77],
                        "confidence": 0.99,
                    }
            ],
            "visualControlCandidates": [
                {
                    "elementId": 2,
                    "boxSource": [110, 34, 241, 77],
                    "labelElementIds": [1],
                }
            ],
        }
    }

    _hydrate_measured_control_geometry(document, str(image_path))

    [control] = document["reconstruction"]["visualControlCandidates"]
    assert control["boxSource"][0] <= 40
    assert control["boxSource"][2] >= 241
    assert control["geometrySource"] == "measured-closed-outline"


def test_compact_filled_region_around_one_label_is_promoted_to_a_button(
    tmp_path,
) -> None:
    image = np.zeros((130, 300, 3), np.uint8)
    cv2.rectangle(image, (40, 35), (240, 94), (205, 205, 205), -1)
    cv2.putText(
        image,
        "START NOW",
        (70, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    image_path = tmp_path / "filled-button.png"
    cv2.imwrite(str(image_path), image)
    document = {
        "reconstruction": {
            "targetKind": "web",
            "text": [
                {
                    "elementId": 1,
                    "value": "START NOW",
                    "boxSource": [68, 52, 207, 80],
                    "confidence": 0.99,
                }
            ],
            "visualControlCandidates": [],
        }
    }

    _hydrate_measured_control_geometry(document, str(image_path))

    [control] = document["reconstruction"]["visualControlCandidates"]
    assert control["labelElementIds"] == [1]
    assert control["boxSource"][0] <= 40
    assert control["boxSource"][2] >= 240
    assert control["background"] == "#CDCDCD"
    assert control["geometrySource"] in {
        "measured-compact-fill",
        "measured-closed-outline",
    }


def test_filled_panel_with_multiple_labels_is_not_promoted_to_a_button(
    tmp_path,
) -> None:
    image = np.zeros((220, 360, 3), np.uint8)
    cv2.rectangle(image, (30, 25), (330, 195), (205, 205, 205), -1)
    cv2.putText(image, "TITLE", (60, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(image, "VALUE", (60, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    image_path = tmp_path / "filled-panel.png"
    cv2.imwrite(str(image_path), image)
    document = {
        "reconstruction": {
            "targetKind": "web",
            "text": [
                {"elementId": 1, "value": "TITLE", "boxSource": [58, 60, 130, 90]},
                {"elementId": 2, "value": "VALUE", "boxSource": [58, 120, 140, 150]},
            ],
            "visualControlCandidates": [],
        }
    }

    _hydrate_measured_control_geometry(document, str(image_path))

    assert document["reconstruction"]["visualControlCandidates"] == []


def test_plain_text_without_an_outline_is_not_a_control_candidate() -> None:
    image = np.full((120, 440, 3), CREAM_BGR, np.uint8)
    cv2.putText(
        image,
        "TICKETS",
        (140, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        BLUE_BGR,
        2,
        cv2.LINE_AA,
    )

    assert outlined_controls_around_text(
        image,
        [{"text": "TICKETS", "box": [138, 47, 302, 87], "confidence": 0.99}],
        [],
    ) == []


def test_borderless_panel_is_not_promoted_to_a_control() -> None:
    controls = [
        {
            "box": [62, 6, 265, 240],
            "background": "#F6F6F6",
            "borderColor": None,
            "cornerRadius": 0,
            "source": "measured",
        }
    ]

    assert _plausible_controls(controls, np.array((229, 229, 229)), []) == []


def test_low_contrast_chart_outline_is_not_a_button() -> None:
    controls = [
        {
            "kind": "button",
            "box": [302, 445, 722, 492],
            "background": "#F7F8F8",
            "borderColor": "#F8F9F9",
            "borderWidth": 2,
            "labelText": "Tue, Nov 11",
            "boundaryEvidence": {
                "top": 0.998,
                "right": 0.986,
                "bottom": 0.671,
                "left": 0.702,
                "closed": True,
            },
            "method": "closed-outline-around-ocr",
        }
    ]

    assert _plausible_controls(controls, np.array((229, 229, 229)), []) == []


def test_high_contrast_rounded_outline_remains_a_button() -> None:
    controls = [
        {
            "kind": "button",
            "box": [245, 1103, 586, 1209],
            "background": "#FCF7EF",
            "borderColor": "#006EFF",
            "borderWidth": 2,
            "labelText": "TICKETS",
            "boundaryEvidence": {
                "top": 0.37,
                "right": 0.24,
                "bottom": 0.37,
                "left": 0.24,
                "closed": True,
            },
            "method": "closed-outline-around-ocr",
        }
    ]

    assert _plausible_controls(controls, CREAM_BGR, [])[0]["labelText"] == "TICKETS"


def test_local_edges_find_outline_buttons_over_a_photographic_gradient() -> None:
    reference = (
        Path(__file__).resolve().parents[2]
        / "qa"
        / "fixtures"
        / "reconstruction-matrix"
        / "references"
        / "dope-security-29b0c7897d361444.png"
    )
    image = cv2.imread(str(reference))
    ocr = [
        {"text": "Free Instant Trial", "box": [889, 446, 1207, 483]},
        {"text": "Try now with Google", "box": [1004, 529, 1160, 549]},
        {"text": "Try now with Microsoft", "box": [967, 605, 1170, 625]},
        {"text": "LOG IN", "box": [1354, 21, 1394, 31]},
    ]

    controls = outlined_controls_around_text(image, ocr, [])

    labels = {control["labelText"]: control for control in controls}
    assert "Free Instant Trial" not in labels
    assert labels["Try now with Google"]["box"] == [891, 509, 1246, 569]
    assert labels["Try now with Microsoft"]["box"] == [891, 583, 1246, 646]
    assert labels["LOG IN"]["box"] == [1333, 8, 1415, 44]
    assert all(
        control["method"] == "edge-closed-outline-around-ocr"
        for control in labels.values()
    )


def test_uniform_photo_section_is_not_a_filled_button() -> None:
    reference = (
        Path(__file__).resolve().parents[2]
        / "qa"
        / "fixtures"
        / "reconstruction-matrix"
        / "references"
        / "dope-security-29b0c7897d361444.png"
    )
    image = cv2.imread(str(reference))
    ocr = [
        {"text": "BOOK A DEMO", "box": [1218, 21, 1304, 31]},
        {"text": "Free Instant Trial", "box": [889, 446, 1207, 483]},
        {"text": "Try now with Google", "box": [1004, 529, 1160, 549]},
    ]
    outlined = outlined_controls_around_text(image, ocr, [])

    controls = _controls_around_text(image, ocr, outlined)

    labels = {control.get("labelText") for control in controls}
    assert "BOOK A DEMO" in labels
    assert "Free Instant Trial" not in labels
    assert "Try now with Google" not in labels


def test_cached_web_contract_drops_panels_and_chart_frames_from_controls() -> None:
    document = {
        "reconstruction": {
            "targetKind": "web",
            "surfaces": [],
            "visualControlCandidates": [
                {
                    "boxSource": [62, 6, 265, 240],
                    "background": "#F6F6F6",
                    "borderColor": None,
                    "cornerRadius": 0,
                },
                {
                    "boxSource": [302, 445, 722, 492],
                    "background": "#F7F8F8",
                    "borderColor": "#F8F9F9",
                    "borderWidth": 2,
                },
                {
                    "boxSource": [245, 1103, 586, 1209],
                    "background": "#FCF7EF",
                    "borderColor": "#006EFF",
                    "borderWidth": 2,
                },
            ],
        }
    }

    _sanitize_web_structure(document)
    _sanitize_web_structure(document)

    spec = document["reconstruction"]
    assert [control["boxSource"] for control in spec["visualControlCandidates"]] == [
        [245, 1103, 586, 1209]
    ]
    assert spec["surfaces"] == []


def test_cached_web_contract_drops_fragments_inside_text_or_raster() -> None:
    document = {
        "reconstruction": {
            "targetKind": "web",
            "text": [{"boxSource": [50, 50, 250, 200]}],
            "allowedRasterRegions": [{"boxSource": [400, 100, 700, 400]}],
            "visualControlCandidates": [],
            "surfaces": [
                {"boxSource": [80, 80, 180, 150], "background": "#006EFF"},
                {"boxSource": [20, 20, 300, 240], "background": "#FFFFFF"},
                {"boxSource": [450, 150, 600, 300], "background": "#000000"},
            ],
            "decorativeShapes": [
                {"boxSource": [90, 90, 160, 140]},
                {"boxSource": [480, 180, 620, 320]},
                {"boxSource": [720, 50, 760, 90]},
            ],
        }
    }

    _sanitize_web_structure(document)

    spec = document["reconstruction"]
    assert spec["surfaces"] == [
        {"boxSource": [20, 20, 300, 240], "background": "#FFFFFF"}
    ]
    assert spec["decorativeShapes"] == [{"boxSource": [720, 50, 760, 90]}]


def test_cached_web_contract_replaces_surface_fragment_with_broad_panel(
    tmp_path,
) -> None:
    image = np.full((300, 600, 3), (229, 229, 229), np.uint8)
    image[:, 60:240] = (246, 246, 246)
    image[:, 240:] = (255, 255, 255)
    image_path = tmp_path / "dashboard.png"
    cv2.imwrite(str(image_path), image)
    document = {
        "tokens": {
            "color": {"canvas": {"$type": "color", "$value": "#E5E5E5"}}
        },
        "reconstruction": {
            "targetKind": "web",
            "text": [],
            "allowedRasterRegions": [],
            "visualControlCandidates": [],
            "decorativeShapes": [],
            "surfaces": [
                {
                    "boxSource": [60, 0, 240, 120],
                    "background": "#F6F6F6",
                }
            ],
        },
    }

    _hydrate_measured_surfaces(document, str(image_path))

    surfaces = document["reconstruction"]["surfaces"]
    assert not any(surface["boxSource"] == [60, 0, 240, 120] for surface in surfaces)
    assert any(surface["boxSource"] == [60, 0, 240, 300] for surface in surfaces)
    assert any(surface["boxSource"] == [240, 0, 600, 300] for surface in surfaces)


def test_low_contrast_single_label_rectangle_becomes_semantic_control(
    tmp_path,
) -> None:
    image = np.full((240, 500, 3), (246, 246, 246), np.uint8)
    cv2.rectangle(image, (60, 50), (250, 84), (255, 246, 238), -1)
    cv2.rectangle(image, (60, 50), (250, 84), (239, 232, 225), 1)
    cv2.putText(
        image,
        "Overview",
        (92, 74),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    image_path = tmp_path / "quiet-control.png"
    cv2.imwrite(str(image_path), image)
    document = {
        "header": {"background": "#F6F6F6"},
        "tokens": {
            "color": {"canvas": {"$type": "color", "$value": "#F6F6F6"}}
        },
        "reconstruction": {
            "targetKind": "web",
            "text": [
                {
                    "elementId": 1,
                    "value": "Overview",
                    "preferredValue": "Overview",
                    "boxSource": [92, 58, 170, 78],
                    "confidence": 0.99,
                }
            ],
            "visualControlCandidates": [],
            "surfaces": [],
            "decorativeShapes": [],
            "allowedRasterRegions": [],
            "icons": [],
        },
    }

    _hydrate_measured_control_geometry(document, str(image_path))

    [control] = document["reconstruction"]["visualControlCandidates"]
    assert control["kind"] == "button"
    assert control["labelElementIds"] == [1]
    assert control["geometrySource"] == "measured-closed-outline"


def test_large_photo_candidate_is_refined_away_from_live_text(tmp_path) -> None:
    image = np.full((500, 800, 3), (255, 255, 255), np.uint8)
    cv2.ellipse(image, (520, 280), (145, 205), 0, 0, 360, (55, 75, 95), -1)
    cv2.ellipse(image, (480, 245), (70, 105), 0, 0, 360, (90, 120, 155), -1)
    cv2.putText(
        image,
        "Beyond Humanware",
        (40, 430),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    image_path = tmp_path / "photo-hero.png"
    cv2.imwrite(str(image_path), image)
    document = {
        "header": {"size": [800, 500], "background": "#FFFFFF"},
        "tokens": {
            "color": {"canvas": {"$type": "color", "$value": "#FFFFFF"}}
        },
        "reconstruction": {
            "targetKind": "web",
            "canvas": {"width": 800, "height": 500},
            "text": [{"elementId": 1, "boxSource": [35, 395, 310, 445]}],
            "visualControlCandidates": [],
            "symbolArt": [],
            "allowedRasterRegions": [],
            "excludedRasterCandidates": [
                {
                    "elementId": 20,
                    "boxSource": [0, 70, 720, 500],
                    "reason": "overlaps-live-text",
                    "overlappingTextElementId": 1,
                }
            ],
        },
    }

    _refine_overlapping_raster_candidates(document, str(image_path))

    [asset] = document["reconstruction"]["allowedRasterRegions"]
    assert asset["elementId"] == 20
    assert asset["method"] == "foreground-component-raster-refinement"
    x0, y0, x1, y1 = asset["boxSource"]
    assert 350 <= x0 < 430
    assert 60 <= y0 < 100
    assert 650 < x1 <= 690
    assert 470 < y1 <= 500
    assert x0 >= 310
    assert document["reconstruction"]["primaryAsset"]["boxSource"] == asset["boxSource"]
    assert document["reconstruction"]["excludedRasterCandidates"][0]["refinedIntoAllowedBoxSource"] == asset["boxSource"]


def test_product_packaging_keeps_intrinsic_label_text_inside_one_raster(
    tmp_path,
) -> None:
    rng = np.random.default_rng(7)
    image = np.full((500, 800, 3), (28, 63, 126), np.uint8)
    texture = rng.normal(0, 5, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + texture, 0, 255).astype(np.uint8)
    cv2.rectangle(image, (302, 190), (498, 500), (16, 82, 224), -1)
    cv2.ellipse(image, (400, 190), (98, 24), 0, 0, 360, (8, 8, 8), -1)
    cv2.rectangle(image, (312, 310), (488, 500), (9, 21, 73), -1)
    cv2.putText(
        image,
        "TIGER",
        (346, 360),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (30, 180, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "MASALA",
        (328, 405),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (30, 180, 245),
        2,
        cv2.LINE_AA,
    )
    image_path = tmp_path / "packaging.png"
    cv2.imwrite(str(image_path), image)
    document = {
        "header": {"size": [800, 500], "background": "#7E3F1C"},
        "tokens": {
            "color": {"canvas": {"$type": "color", "$value": "#7E3F1C"}}
        },
        "reconstruction": {
            "targetKind": "web",
            "canvas": {"width": 800, "height": 500},
            "text": [
                {"elementId": 1, "value": "BUY NOW", "boxSource": [20, 430, 105, 455]},
                {"elementId": 2, "value": "TIGER", "boxSource": [342, 335, 458, 365]},
                {"elementId": 3, "value": "MASALA", "boxSource": [325, 380, 475, 410]},
            ],
            "layoutRegions": [
                {
                    "regionId": 10,
                    "role": "content",
                    "kind": "section",
                    "boxSource": [220, 160, 580, 500],
                    "elementIds": [2, 3],
                }
            ],
            "visualControlCandidates": [
                {
                    "elementId": 20,
                    "boxSource": [330, 320, 470, 375],
                    "labelElementIds": [2],
                }
            ],
            "surfaces": [
                {"boxSource": [312, 310, 488, 500], "background": "#491509"}
            ],
            "structuralLines": [
                {"boxSource": [398, 190, 402, 500], "orientation": "vertical"}
            ],
            "vectorPaths": [],
            "decorativeShapes": [],
            "icons": [],
            "symbolArt": [],
            "allowedRasterRegions": [],
            "excludedRasterCandidates": [],
            "primaryAsset": None,
        },
    }

    _hydrate_intrinsic_text_raster_assets(document, str(image_path))

    reconstruction = document["reconstruction"]
    [asset] = reconstruction["allowedRasterRegions"]
    assert asset["kind"] == "product-photo-with-intrinsic-text"
    assert asset["method"] == "layout-seeded-grabcut-object"
    x0, y0, x1, y1 = asset["boxSource"]
    assert 285 <= x0 <= 310
    assert 160 <= y0 <= 200
    assert 490 <= x1 <= 510
    assert y1 == 500
    assert [entry["elementId"] for entry in asset["intrinsicText"]] == [2, 3]
    assert [entry["elementId"] for entry in reconstruction["text"]] == [1]
    assert reconstruction["visualControlCandidates"] == []
    assert reconstruction["structuralLines"] == []
    assert reconstruction["primaryAsset"]["boxSource"] == asset["boxSource"]


def test_intrinsic_text_raster_does_not_flatten_a_measured_ui_surface(
    tmp_path,
) -> None:
    image = np.full((400, 700, 3), (245, 245, 245), np.uint8)
    cv2.rectangle(image, (100, 80), (600, 360), (255, 255, 255), -1)
    cv2.rectangle(image, (100, 80), (600, 360), (215, 215, 215), 2)
    image_path = tmp_path / "dashboard-card.png"
    cv2.imwrite(str(image_path), image)
    document = {
        "header": {"size": [700, 400], "background": "#F5F5F5"},
        "reconstruction": {
            "targetKind": "web",
            "canvas": {"width": 700, "height": 400},
            "text": [
                {"elementId": 1, "value": "Revenue", "boxSource": [130, 110, 220, 135]},
                {"elementId": 2, "value": "$42,000", "boxSource": [130, 150, 260, 185]},
            ],
            "layoutRegions": [
                {
                    "regionId": 1,
                    "role": "content",
                    "kind": "section",
                    "boxSource": [80, 60, 620, 380],
                    "elementIds": [1, 2],
                }
            ],
            "surfaces": [
                {"boxSource": [100, 80, 600, 360], "background": "#FFFFFF"}
            ],
            "visualControlCandidates": [],
            "structuralLines": [],
            "vectorPaths": [],
            "decorativeShapes": [],
            "icons": [],
            "symbolArt": [],
            "allowedRasterRegions": [],
            "excludedRasterCandidates": [],
            "primaryAsset": None,
        },
    }

    _hydrate_intrinsic_text_raster_assets(document, str(image_path))

    assert document["reconstruction"]["allowedRasterRegions"] == []


def test_intrinsic_text_raster_does_not_flatten_a_large_hero_heading() -> None:
    reference = (
        Path(__file__).resolve().parents[2]
        / "qa"
        / "fixtures"
        / "reconstruction-matrix"
        / "references"
        / "dope-security-29b0c7897d361444.png"
    )
    document = {
        "header": {"size": [1440, 900], "background": "#070A35"},
        "reconstruction": {
            "targetKind": "web",
            "canvas": {"width": 1440, "height": 900},
            "text": [
                {
                    "elementId": 10,
                    "value": "Your new",
                    "boxSource": [60, 186, 465, 272],
                },
                {
                    "elementId": 12,
                    "value": "Secure Web",
                    "boxSource": [74, 268, 548, 334],
                },
                {
                    "elementId": 16,
                    "value": "Gateway",
                    "boxSource": [62, 327, 448, 426],
                },
                {
                    "elementId": 17,
                    "value": "with AI DLP",
                    "boxSource": [60, 400, 531, 492],
                },
            ],
            "layoutRegions": [
                {
                    "regionId": 2,
                    "role": "hero",
                    "kind": "section",
                    "boxSource": [39, 186, 562, 492],
                    "elementIds": [10, 12, 16, 17],
                }
            ],
            "surfaces": [],
            "visualControlCandidates": [],
            "structuralLines": [],
            "vectorPaths": [],
            "decorativeShapes": [],
            "icons": [],
            "symbolArt": [],
            "allowedRasterRegions": [],
            "excludedRasterCandidates": [],
            "primaryAsset": None,
        },
    }

    _hydrate_intrinsic_text_raster_assets(document, str(reference))

    reconstruction = document["reconstruction"]
    assert reconstruction["allowedRasterRegions"] == []
    assert [entry["elementId"] for entry in reconstruction["text"]] == [
        10,
        12,
        16,
        17,
    ]


def test_textured_canvas_gets_an_alpha_masked_background_artwork_layer(
    tmp_path,
) -> None:
    rng = np.random.default_rng(11)
    base = np.full((400, 700, 3), (24, 56, 132), np.int16)
    image = np.clip(base + rng.normal(0, 9, base.shape), 0, 255).astype(np.uint8)
    cv2.putText(
        image,
        "BOLD FLAVOR",
        (25, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.5,
        (40, 180, 250),
        9,
        cv2.LINE_AA,
    )
    image_path = tmp_path / "textured-canvas.png"
    cv2.imwrite(str(image_path), image)
    document = {
        "tokens": {
            "color": {"canvas": {"$type": "color", "$value": "#843818"}}
        },
        "reconstruction": {
            "targetKind": "web",
            "canvas": {"width": 700, "height": 400},
            "text": [
                {
                    "elementId": 1,
                    "value": "BOLD FLAVOR",
                    "boxSource": [25, 125, 675, 195],
                    "color": "#FAB428",
                }
            ],
            "visualControlCandidates": [],
            "surfaces": [],
            "structuralLines": [],
            "vectorPaths": [],
            "decorativeShapes": [],
            "icons": [],
            "symbolArt": [],
            "allowedRasterRegions": [],
        },
    }

    _hydrate_background_artwork_layer(document, str(image_path))

    [layer] = document["reconstruction"]["allowedRasterRegions"]
    assert layer["kind"] == "alpha-masked-background-artwork"
    assert layer["boxSource"] == [0, 0, 700, 400]
    assert layer["method"] == "protected-pixel-alpha-mask"
    assert (
        layer["protectionPolicy"]["liveText"]
        == "source-glyphs-inpainted-under-live-dom"
    )


def test_soft_photographic_canvas_gets_a_protected_background_layer(
    tmp_path,
) -> None:
    height, width = 360, 640
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    image = np.zeros((height, width, 3), np.float32)
    image[..., 0] = 55 + 85 * x
    image[..., 1] = 18 + 45 * y
    image[..., 2] = 12 + 65 * (1.0 - x)
    cv2.circle(image, (470, 190), 115, (180, 95, 210), -1)
    image = cv2.GaussianBlur(image, (0, 0), 28)
    image = np.clip(image, 0, 255).astype(np.uint8)
    image_path = tmp_path / "soft-photo.png"
    cv2.imwrite(str(image_path), image)
    document = {
        "header": {"size": [width, height], "background": "#0C1351"},
        "tokens": {
            "color": {"canvas": {"$type": "color", "$value": "#0C1351"}}
        },
        "reconstruction": {
            "targetKind": "web",
            "canvas": {"width": width, "height": height},
            "text": [],
            "visualControlCandidates": [],
            "surfaces": [],
            "structuralLines": [],
            "vectorPaths": [],
            "decorativeShapes": [],
            "icons": [],
            "symbolArt": [],
            "allowedRasterRegions": [],
        },
    }

    _hydrate_background_artwork_layer(document, str(image_path))

    [layer] = document["reconstruction"]["allowedRasterRegions"]
    assert layer["kind"] == "alpha-masked-background-artwork"
    assert layer["evidence"]["softPhotographicArtwork"] is True
    assert (
        layer["protectionPolicy"]["liveText"]
        == "source-glyphs-inpainted-under-live-dom"
    )


def test_flat_canvas_does_not_get_a_background_raster_layer(tmp_path) -> None:
    image = np.full((400, 700, 3), (241, 248, 255), np.uint8)
    image_path = tmp_path / "flat-canvas.png"
    cv2.imwrite(str(image_path), image)
    document = {
        "tokens": {
            "color": {"canvas": {"$type": "color", "$value": "#FFF8F1"}}
        },
        "reconstruction": {
            "targetKind": "web",
            "canvas": {"width": 700, "height": 400},
            "text": [],
            "visualControlCandidates": [],
            "surfaces": [],
            "structuralLines": [],
            "vectorPaths": [],
            "decorativeShapes": [],
            "icons": [],
            "symbolArt": [],
            "allowedRasterRegions": [],
        },
    }

    _hydrate_background_artwork_layer(document, str(image_path))

    assert document["reconstruction"]["allowedRasterRegions"] == []


def test_dense_interface_gets_protected_non_text_chrome_layer(tmp_path) -> None:
    image = np.full((400, 700, 3), (248, 248, 248), np.uint8)
    for index in range(10):
        x = 20 + (index % 5) * 130
        y = 20 + (index // 5) * 170
        cv2.rectangle(image, (x, y), (x + 110, y + 140), (255, 255, 255), -1)
        cv2.rectangle(image, (x, y), (x + 110, y + 140), (220, 220, 220), 1)
    image_path = tmp_path / "dashboard.png"
    cv2.imwrite(str(image_path), image)
    document = {
        "tokens": {
            "color": {"canvas": {"$type": "color", "$value": "#F8F8F8"}}
        },
        "reconstruction": {
            "targetKind": "web",
            "canvas": {"width": 700, "height": 400},
            "text": [],
            "visualControlCandidates": [],
            "surfaces": [
                {
                    "boxSource": [
                        20 + (index % 5) * 130,
                        20 + (index // 5) * 170,
                        130 + (index % 5) * 130,
                        160 + (index // 5) * 170,
                    ]
                }
                for index in range(10)
            ],
            "structuralLines": [],
            "vectorPaths": [],
            "decorativeShapes": [],
            "icons": [],
            "symbolArt": [],
            "allowedRasterRegions": [],
        },
    }

    _hydrate_background_artwork_layer(document, str(image_path))

    [layer] = document["reconstruction"]["allowedRasterRegions"]
    assert layer["kind"] == "alpha-masked-background-artwork"
    assert layer["evidence"]["complexInterface"] is True


def test_short_label_inside_tall_filled_button_stays_a_semantic_control(
    tmp_path,
) -> None:
    image = np.full((180, 320, 3), (30, 70, 130), np.uint8)
    cv2.rectangle(image, (40, 100), (160, 150), (40, 180, 250), -1)
    cv2.putText(
        image,
        "BUY NOW",
        (68, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (20, 35, 60),
        1,
        cv2.LINE_AA,
    )
    image_path = tmp_path / "tall-button.png"
    cv2.imwrite(str(image_path), image)
    document = {
        "header": {"size": [320, 180], "background": "#82461E"},
        "tokens": {
            "color": {"canvas": {"$type": "color", "$value": "#82461E"}}
        },
        "reconstruction": {
            "targetKind": "web",
            "canvas": {"width": 320, "height": 180},
            "text": [
                {
                    "elementId": 1,
                    "value": "BUY NOW",
                    "boxSource": [68, 119, 132, 132],
                    "color": "#FAB428",
                    "fontFeatures": {"color": "#FAB428"},
                }
            ],
            "visualControlCandidates": [],
            "surfaces": [],
            "decorativeShapes": [],
            "allowedRasterRegions": [],
            "icons": [],
        },
    }

    _hydrate_measured_control_geometry(document, str(image_path))

    [control] = document["reconstruction"]["visualControlCandidates"]
    assert control["labelElementIds"] == [1]
    assert control["boxSource"][3] - control["boxSource"][1] >= 45
    assert document["reconstruction"]["text"][0]["color"] != "#FAB428"
