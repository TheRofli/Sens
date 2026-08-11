import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from sight import ops


def test_source_dom_typography_overrides_inferred_font_with_loaded_observation(
    tmp_path,
) -> None:
    content = b"whyte-medium-woff2"
    font_path = tmp_path / "whyte-medium.woff2"
    font_path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    document = {
        "profile": "reconstruct",
        "reconstruction": {
            "text": [
                {
                    "elementId": 16,
                    "value": "Gateway",
                    "preferredValue": "Gateway",
                    "boxSource": [62, 327, 448, 426],
                    "fontFeatures": {"fontSize": 136, "renderFamily": "inter-tight"},
                }
            ]
        }
    }
    source_nodes = [
        {
            "text": "Gateway",
            "box": [62, 327, 448, 426],
            "visible": True,
            "style": {
                "fontFamily": '"Whyte Inktrap", sans-serif',
                "fontWeight": "500",
                "fontStyle": "normal",
                "fontSize": "136px",
                "letterSpacing": "0px",
            },
        }
    ]
    source_fonts = [
        {
            "family": "Whyte Inktrap",
            "weight": "500",
            "style": "normal",
            "stretch": "normal",
            "path": str(font_path),
            "sha256": digest,
            "sizeBytes": len(content),
            "mediaType": "font/woff2",
            "format": "woff2",
            "source": "observed",
            "method": "playwright-loaded-font-response",
        }
    ]

    ops._hydrate_source_dom_typography(document, source_nodes, source_fonts)

    entry = document["reconstruction"]["text"][0]
    font = entry["fontFeatures"]
    assert font["sourceDomFamily"] == "Whyte Inktrap"
    assert font["sourceDomFontWeight"] == 500
    assert font["sourceDomFontStyle"] == "normal"
    assert font["sourceFontFamily"].startswith("Sens Source ")
    assert font["sourceFontAssetSha256"] == digest
    assert font["sourceDomTypographySource"] == "observed-live-dom-computed-style"
    [asset] = document["reconstruction"]["sourceFontAssets"]
    assert asset["sha256"] == digest
    assert asset["family"] == "Whyte Inktrap"
    authority = document["reconstruction"]["typographyAuthority"]
    assert authority["priority"][0] == "observed-live-dom-computed-style"
    assert "must not override" in authority["rule"]


def test_source_dom_typography_preserves_mixed_word_styles_from_live_dom(
    tmp_path,
) -> None:
    normal_content = b"grandslang-normal-woff2"
    italic_content = b"grandslang-italic-woff2"
    normal_path = tmp_path / "grandslang-normal.woff2"
    italic_path = tmp_path / "grandslang-italic.woff2"
    normal_path.write_bytes(normal_content)
    italic_path.write_bytes(italic_content)
    normal_digest = hashlib.sha256(normal_content).hexdigest()
    italic_digest = hashlib.sha256(italic_content).hexdigest()
    document = {
        "profile": "reconstruct",
        "reconstruction": {
            "text": [
                {
                    "elementId": 10,
                    "value": "Yournew",
                    "preferredValue": "Your new",
                    "boxSource": [60, 185, 463, 274],
                    "fontFeatures": {
                        "fontSize": 105,
                        "wordBoxesSource": [
                            {"text": "Your", "box": [68, 197, 258, 274]},
                            {"text": "new", "box": [268, 212, 455, 274]},
                        ],
                    },
                }
            ]
        },
    }
    source_nodes = [
        {
            "text": "Your",
            "box": [72, 179, 269, 284],
            "visible": True,
            "style": {
                "fontFamily": '"GrandSlang", serif',
                "fontWeight": "400",
                "fontStyle": "normal",
                "fontSize": "88px",
                "lineHeight": "70.4px",
                "letterSpacing": "-2.64px",
            },
        },
        {
            "text": "new",
            "box": [269, 179, 456, 284],
            "visible": True,
            "style": {
                "fontFamily": '"GrandSlang", serif',
                "fontWeight": "400",
                "fontStyle": "italic",
                "fontSize": "88px",
                "lineHeight": "70.4px",
                "letterSpacing": "-2.64px",
            },
        },
    ]
    source_fonts = [
        {
            "family": "GrandSlang",
            "weight": "400",
            "style": "normal",
            "path": str(normal_path),
            "sha256": normal_digest,
            "mediaType": "font/woff2",
        },
        {
            "family": "GrandSlang",
            "weight": "400",
            "style": "italic",
            "path": str(italic_path),
            "sha256": italic_digest,
            "mediaType": "font/woff2",
        },
    ]

    ops._hydrate_source_dom_typography(document, source_nodes, source_fonts)

    font = document["reconstruction"]["text"][0]["fontFeatures"]
    words = font["sourceDomWordStyles"]
    assert [word["text"] for word in words] == ["Your", "new"]
    assert [word["sourceDomFontStyle"] for word in words] == [
        "normal",
        "italic",
    ]
    assert [word["sourceDomLetterSpacing"] for word in words] == [
        "-2.64px",
        "-2.64px",
    ]
    assert words[0]["sourceFontAssetSha256"] == normal_digest
    assert words[1]["sourceFontAssetSha256"] == italic_digest
    assert words[0]["sourceFontFamily"] != words[1]["sourceFontFamily"]
    assert font["sourceDomRunAuthority"] == "observed-live-dom-computed-style"


def test_web_brief_makes_observed_dom_typography_authoritative_over_inference() -> None:
    document = {
        "profile": "reconstruct",
        "reconstruction": {
            "targetKind": "web",
            "canvas": {"width": 1440, "height": 900},
            "typographyAuthority": {
                "priority": [
                    "observed-live-dom-computed-style",
                    "measured-screenshot-geometry",
                    "inferred-typography-fallback",
                ],
                "rule": "Observed source DOM typography must not be overridden by screenshot inference.",
            },
            "sourceFontAssets": [
                {
                    "family": "Whyte Inktrap",
                    "alias": "Sens Source abcdef123456",
                    "weight": "500",
                    "style": "normal",
                    "format": "woff2",
                }
            ],
            "text": [
                {
                    "elementId": 17,
                    "value": "Gateway",
                    "preferredValue": "Gateway",
                    "boxSource": [61, 327, 449, 426],
                    "typographyCandidate": {
                        "class": "serif",
                        "weight": "light",
                        "method": "semantic-inference",
                    },
                    "inlineRuns": [
                        {
                            "text": "Gateway",
                            "typographyCandidate": {
                                "class": "serif",
                                "slant": "italic",
                            },
                        }
                    ],
                    "fontFeatures": {
                        "fontSize": 136,
                        "family": "newsreader",
                        "weightCandidate": "light",
                        "sourceDomFamily": "Whyte Inktrap",
                        "sourceFontFamily": "Sens Source abcdef123456",
                        "sourceFontAssetSha256": "a" * 64,
                        "sourceDomFontWeight": 500,
                        "sourceDomFontStyle": "normal",
                        "sourceDomFontSize": "88px",
                        "sourceDomWordStyles": [
                            {
                                "text": "Gateway",
                                "sourceFontFamily": "Sens Source abcdef123456",
                                "sourceDomFontWeight": 500,
                                "sourceDomFontStyle": "normal",
                                "sourceDomLetterSpacing": "-2.64px",
                                "sourceDomBox": [72, 327, 449, 426],
                            }
                        ],
                        "sourceDomTypographySource": "observed-live-dom-computed-style",
                    },
                }
            ],
        }
    }

    brief = ops._implementation_brief(document, "contract.json")

    table = brief["text"]
    row = json.loads(table["rows"])
    values = {**table.get("constants", {}), **table.get("defaults", {})}
    values.update(
        {
            column: value
            for column, value in zip(table["columns"], row, strict=True)
            if value is not None
        }
    )
    assert brief["typographyAuthority"]["priority"][0] == (
        "observed-live-dom-computed-style"
    )
    assert values["familyHint"] == "Whyte Inktrap"
    assert values["fontWeight"] == 500
    assert values["fontWeightSource"] == "observed-live-dom-computed-style"
    assert values["sourceFontFamily"] == "Sens Source abcdef123456"
    assert values["sourceDomFontStyle"] == "normal"
    assert values["sourceDomFontSize"] == "88px"
    assert values[
        "sourceDomWords(text,font,weight,style,letterSpacing,box)"
    ][0][3] == "normal"
    assert values.get("inlineRuns(text,class,contrast,width,weight,slant)") is None
    assert brief["sourceFonts"][0]["alias"] == "Sens Source abcdef123456"
    assert any(
        "observed DOM typography" in rule
        for rule in brief["implementationRules"]
    )

    compact = ops._compact_document(document)
    assert compact["reconstruction"]["typographyAuthority"]["priority"][0] == (
        "observed-live-dom-computed-style"
    )
    compact_table = compact["reconstruction"]["text"]
    compact_row = json.loads(compact_table["rows"])
    compact_values = {
        **compact_table.get("constants", {}),
        **compact_table.get("defaults", {}),
    }
    compact_values.update(
        {
            column: value
            for column, value in zip(
                compact_table["columns"], compact_row, strict=True
            )
            if value is not None
        }
    )
    assert compact_values["sourceDomFamily"] == "Whyte Inktrap"
    assert compact_values["fontFamily"] == "Whyte Inktrap"
    assert compact_values["fontWeight"] == 500
    assert compact_values["sourceDomWords"][0][3] == "normal"
    assert compact_values.get("inlineRuns") is None
    assert any(
        "observed DOM typography" in rule
        for rule in compact["reconstruction"]["implementationRules"]
    )


def test_materializer_copies_browser_source_background_without_clipping_or_inpaint(
    tmp_path,
) -> None:
    reference = np.full((120, 200, 3), (180, 120, 40), np.uint8)
    reference_path = tmp_path / "reference.png"
    cv2.imwrite(str(reference_path), reference)
    source_bytes = b"original-browser-avif"
    source_path = tmp_path / "hero.avif"
    source_path.write_bytes(source_bytes)
    digest = hashlib.sha256(source_bytes).hexdigest()
    document = {
        "artifacts": [],
        "reconstruction": {
            "allowedRasterRegions": [
                {
                    "elementId": "browser-source-background",
                    "kind": "browser-source-background-artwork",
                    "boxSource": [-20, -30, 240, 180],
                    "sourceAssetPath": str(source_path),
                    "contentSha256": digest,
                    "mediaType": "image/avif",
                }
            ],
            "rasterAssetRule": {},
        },
    }

    ops._materialize_raster_assets(
        document,
        str(reference_path),
        str(tmp_path / "assets"),
        no_store=False,
    )

    [layer] = document["reconstruction"]["allowedRasterRegions"]
    assert layer["boxSource"] == [-20, -30, 240, 180]
    assert Path(layer["assetPath"]).suffix == ".avif"
    assert Path(layer["assetPath"]).read_bytes() == source_bytes
    assert layer["mediaType"] == "image/avif"
    assert layer["contentSha256"] == digest
    assert document["artifacts"][0]["kind"] == (
        "reconstruction-browser-source-background"
    )


def test_materializer_layers_semantic_holes_over_browser_source_background(
    tmp_path,
) -> None:
    reference = np.full((80, 120, 3), (220, 238, 255), np.uint8)
    reference[25:50, 35:85] = (0, 0, 0)
    reference_path = tmp_path / "reference.png"
    assert cv2.imwrite(str(reference_path), reference)
    source_bytes = b"original-transparent-browser-source"
    source_path = tmp_path / "hero.avif"
    source_path.write_bytes(source_bytes)
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    document = {
        "artifacts": [],
        "reconstruction": {
            "text": [
                {
                    "elementId": 1,
                    "value": "SLUSH",
                    "boxSource": [35, 25, 85, 50],
                    "method": "rapidocr-multiscale-consensus",
                }
            ],
            "visualControlCandidates": [],
            "surfaces": [],
            "decorativeShapes": [],
            "icons": [],
            "badges": [],
            "symbolArt": [],
            "structuralLines": [],
            "vectorPaths": [],
            "displayTextDiscovery": {
                "status": "complete",
                "candidateCount": 0,
                "method": "rapidocr-downscaled-display-scan",
            },
            "allowedRasterRegions": [
                {
                    "elementId": "browser-source-background",
                    "kind": "browser-source-background-artwork",
                    "boxSource": [-10, -10, 130, 90],
                    "sourceAssetPath": str(source_path),
                    "contentSha256": source_digest,
                    "mediaType": "image/avif",
                },
                {
                    "elementId": "browser-source-composite-overlay",
                    "kind": "alpha-masked-background-artwork",
                    "boxSource": [0, 0, 120, 80],
                    "compositeUnderlay": "browser-source-background",
                },
            ],
            "rasterAssetRule": {},
        },
    }

    ops._materialize_raster_assets(
        document,
        str(reference_path),
        str(tmp_path / "assets"),
        no_store=False,
    )

    source_layer, overlay = document["reconstruction"]["allowedRasterRegions"]
    assert Path(source_layer["assetPath"]).read_bytes() == source_bytes
    rgba = cv2.imread(overlay["assetPath"], cv2.IMREAD_UNCHANGED)
    assert rgba.shape == (80, 120, 4)
    assert rgba[5, 5, 3] == 255
    assert rgba[35, 60, 3] == 0
    assert np.array_equal(rgba[5, 5, :3], reference[5, 5])
    assert overlay["semanticContentRemoved"] is True
    assert overlay["semanticRemovalMaskCoverage"] > 0
    assert overlay["semanticResidualProtection"][
        "displayTextDiscoveryComplete"
    ] is True
    assert document["reconstruction"]["rasterAssetRule"]["assetCount"] == 2


def _stub_document(profile: str) -> dict:
    return {
        "schemaVersion": "2.0.0",
        "profile": profile,
        "source": {"id": "sha256:fixture", "mediaType": "image"},
        "coordinateSpaces": {"source": {"size": [1000, 500]}},
        "header": {"size": [1000, 500]},
        "artifacts": [{"id": "som:fixture", "kind": "set-of-marks"}],
        "warnings": [],
        "nextActions": [],
        "claims": [{"id": "duplicate"}],
        "ascii": "duplicate composition map",
        "elements": [{"id": 1}],
        "semantics_status": "unavailable",
        "reconstruction": {"canvas": {"width": 1000, "height": 500}}
        if profile == "reconstruct"
        else None,
    }


def _install_stubs(monkeypatch, captured) -> None:
    monkeypatch.setattr(
        ops,
        "analyze",
        lambda *_args, **_kwargs: {
            "somPath": "som.png",
            "design": {"facts": [{"kind": "alignment", "detail": "ok"}]},
            "warnings": [],
        },
    )
    monkeypatch.setattr(
        ops,
        "_image_for",
        lambda *_args, **_kwargs: np.zeros((10, 10, 3), np.uint8),
    )

    def build_document(*_args, **kwargs):
        captured.update(kwargs)
        return _stub_document(kwargs["profile"])

    monkeypatch.setattr(ops.docmod, "build_document", build_document)
    monkeypatch.setattr(ops.docmod, "render_markdown", lambda _doc: "FULL MARKDOWN")


def test_materialize_raster_assets_writes_exact_allowed_crop(tmp_path) -> None:
    source = np.zeros((60, 100, 3), np.uint8)
    source[10:40, 20:80] = (12, 34, 56)
    reference = tmp_path / "reference.png"
    assert cv2.imwrite(str(reference), source)
    document = {
        "source": {"id": "sha256:fixture"},
        "artifacts": [],
        "reconstruction": {
            "allowedRasterRegions": [
                {
                    "elementId": 7,
                    "boxSource": [20, 10, 80, 40],
                    "strategy": "extract-source-crop-verbatim",
                }
            ],
            "primaryAsset": {"elementId": 7},
            "rasterAssetRule": {},
        },
    }

    ops._materialize_raster_assets(
        document,
        str(reference),
        str(tmp_path / "assets"),
        no_store=False,
    )

    region = document["reconstruction"]["allowedRasterRegions"][0]
    asset_path = region["assetPath"]
    crop = cv2.imread(asset_path)
    assert crop.shape[:2] == (30, 60)
    assert np.all(crop == (12, 34, 56))
    assert document["reconstruction"]["primaryAsset"]["assetPath"] == asset_path
    assert document["reconstruction"]["rasterAssetRule"]["assetsReady"] is True
    assert document["artifacts"][0]["kind"] == "reconstruction-raster-asset"


def test_materialized_background_artwork_removes_semantic_chrome_and_keeps_live_dom(
    tmp_path,
) -> None:
    source = np.full((120, 200, 3), (25, 55, 130), np.uint8)
    cv2.putText(
        source,
        "BOLD",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (40, 180, 250),
        4,
        cv2.LINE_AA,
    )
    cv2.rectangle(source, (20, 80), (90, 110), (40, 180, 250), 2)
    cv2.circle(source, (105, 95), 6, (40, 180, 250), 2)
    reference = tmp_path / "textured-reference.png"
    assert cv2.imwrite(str(reference), source)
    document = {
        "source": {"id": "sha256:texture"},
        "tokens": {
            "color": {"canvas": {"$type": "color", "$value": "#823719"}}
        },
        "artifacts": [],
        "reconstruction": {
            "targetKind": "web",
            "displayTextDiscovery": {
                "status": "complete",
                "scale": 0.5,
                "candidateCount": 0,
                "method": "rapidocr-downscaled-display-scan",
            },
            "text": [
                {
                    "elementId": 1,
                    "value": "BOLD",
                    "boxSource": [20, 30, 120, 65],
                    "color": "#FAB428",
                }
            ],
            "visualControlCandidates": [
                {
                    "elementId": 2,
                    "boxSource": [20, 80, 91, 111],
                    "background": "#823719",
                    "borderColor": "#FAB428",
                    "borderWidth": 2,
                }
            ],
            "surfaces": [],
            "decorativeShapes": [],
            "icons": [
                {"elementId": 3, "name": "unknown", "boxSource": [99, 89, 112, 102]}
            ],
            "badges": [],
            "symbolArt": [],
            "structuralLines": [],
            "vectorPaths": [],
            "allowedRasterRegions": [
                {
                    "elementId": "background-artwork",
                    "kind": "alpha-masked-background-artwork",
                    "boxSource": [0, 0, 200, 120],
                }
            ],
            "rasterAssetRule": {},
        },
    }

    ops._materialize_raster_assets(
        document,
        str(reference),
        str(tmp_path / "assets"),
        no_store=False,
    )

    layer = document["reconstruction"]["allowedRasterRegions"][0]
    rgba = cv2.imread(layer["assetPath"], cv2.IMREAD_UNCHANGED)
    assert rgba.shape == (120, 200, 4)
    assert rgba[10, 10, 3] == 255
    assert np.count_nonzero(rgba[78:113, 18:93, 3]) == 35 * 75
    assert np.count_nonzero(rgba[87:104, 97:114, 3]) == 17 * 17
    assert np.count_nonzero(rgba[25:70, 15:130, 3] == 0) == 0
    original_text_bgr = np.asarray((40, 180, 250), dtype=np.int16)
    text_patch = rgba[25:70, 15:130, :3].astype(np.int16)
    assert np.count_nonzero(
        np.linalg.norm(text_patch - original_text_bgr, axis=2) < 8
    ) < 20
    control_patch = rgba[76:115, 16:95, :3].astype(np.int16)
    icon_patch = rgba[85:106, 95:116, :3].astype(np.int16)
    assert np.count_nonzero(
        np.linalg.norm(control_patch - original_text_bgr, axis=2) < 8
    ) < 20
    assert np.count_nonzero(
        np.linalg.norm(icon_patch - original_text_bgr, axis=2) < 8
    ) < 12
    control = document["reconstruction"]["visualControlCandidates"][0]
    assert "decorationPreservedInBackgroundArtwork" not in control
    assert control["background"] == "#823719"
    assert control["borderColor"] == "#FAB428"
    assert control["borderWidth"] == 2
    assert "preservedInBackgroundArtwork" not in document["reconstruction"][
        "icons"
    ][0]
    assert layer["protectionPolicy"]["controlDecoration"] == (
        "removed-from-raster-recreated-as-semantic-css"
    )
    assert layer["protectionPolicy"]["liveText"] == (
        "full-box-inpainted-under-live-dom"
    )
    assert layer["alphaProtected"] is True
    assert layer["semanticContentRemoved"] is True
    assert layer["protectionVersion"] == 3
    assert layer["semanticRemovalMaskCoverage"] > 0
    assert layer["semanticResidualProtection"] == {
        "displayTextDiscoveryComplete": True,
        "displayTextCandidateCount": 0,
        "method": "rapidocr-downscaled-display-scan",
        "displayGlyphMaskElementIds": [],
    }
    assert document["artifacts"][0]["kind"] == (
        "reconstruction-alpha-masked-background"
    )


def test_dense_interface_background_removes_the_entire_source_text_footprint(
    tmp_path,
) -> None:
    background = np.asarray((246, 246, 246), dtype=np.uint8)
    source = np.full((90, 260, 3), background, np.uint8)
    cv2.putText(
        source,
        "Partner Network",
        (36, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (68, 68, 68),
        2,
        cv2.LINE_AA,
    )
    reference = tmp_path / "dense-interface.png"
    assert cv2.imwrite(str(reference), source)
    document = {
        "source": {"id": "sha256:dense-interface"},
        "artifacts": [],
        "reconstruction": {
            "targetKind": "web",
            "text": [
                {
                    "elementId": 1,
                    "value": "Partner Network",
                    "boxSource": [34, 34, 210, 57],
                    "color": "#444444",
                }
            ],
            "visualControlCandidates": [],
            "surfaces": [],
            "decorativeShapes": [],
            "icons": [],
            "badges": [],
            "symbolArt": [],
            "structuralLines": [],
            "vectorPaths": [],
            "allowedRasterRegions": [
                {
                    "elementId": "background-artwork",
                    "kind": "alpha-masked-background-artwork",
                    "boxSource": [0, 0, 260, 90],
                    "evidence": {
                        "denseInterface": True,
                        "complexInterface": True,
                    },
                }
            ],
            "rasterAssetRule": {},
        },
    }

    ops._materialize_raster_assets(
        document,
        str(reference),
        str(tmp_path / "dense-assets"),
        no_store=False,
    )

    layer = document["reconstruction"]["allowedRasterRegions"][0]
    rgba = cv2.imread(layer["assetPath"], cv2.IMREAD_UNCHANGED)
    text_footprint = rgba[32:59, 32:212, :3].astype(np.int16)
    distance = np.linalg.norm(text_footprint - background.astype(np.int16), axis=2)
    assert float(np.percentile(distance, 99)) < 8.0


def test_see_defaults_to_compact_reconstruction_for_copy_intent(monkeypatch) -> None:
    captured = {}
    _install_stubs(monkeypatch, captured)
    refined = []
    monkeypatch.setattr(
        ops,
        "refine_ocr_for_reconstruction",
        lambda path, items: refined.append((path, items)) or items,
        raising=False,
    )

    result = ops.see_document(
        "fixture.png",
        fast=True,
        intent="Repeat this design exactly from the screenshot",
    )

    assert captured["profile"] == "reconstruct"
    assert refined == [("fixture.png", [])]
    assert set(result) == {"doc", "summary", "artifacts", "pack", "compatibility"}
    assert result["compatibility"] == {
        "response": "compact",
        "legacyIncluded": False,
        "fullResponse": "Set response=full only for legacy debugging.",
    }
    assert "document" not in result
    assert "legacy" not in result
    assert "claims" not in result["doc"]
    assert "ascii" not in result["doc"]
    assert "elements" not in result["doc"]
    assert result["summary"]["nextActions"] == []


def test_compact_reconstruction_is_bounded_without_losing_agent_contract() -> None:
    verbose_text = []
    for index in range(60):
        verbose_text.append(
            {
                "elementId": index + 1,
                "value": f"Dashboard label {index + 1}",
                "status": "candidate",
                "confidence": 0.78,
                "verified": False,
                "method": "rapidocr-multiscale-disagreement",
                "alternatives": [
                    {
                        "text": f"Dashboard label {index + 1}",
                        "confidence": 0.78,
                        "scale": 1.0,
                    },
                    {
                        "text": f"Dashboard labeI {index + 1}",
                        "confidence": 0.71,
                        "scale": 1.5,
                    },
                ],
                "confirmedBy": [],
                "preferredValue": None,
                "resolutionStatus": "unresolved",
                "boxSource": [20, 30 + index * 10, 220, 40 + index * 10],
                "boxNormSource": [20, 60, 220, 80],
                "fontFeatures": {
                    "capHeight": 12.0,
                    "avgGlyphWidth": 7.0,
                    "fontSize": 16.0,
                    "widthEm": 0.44,
                    "characterCount": 20,
                    "coordinateSpace": "source-pixels",
                    "family": "custom",
                    "familyStatus": "unknown",
                    "familyConfidence": 0.0,
                    "color": "#112233",
                    "colorSource": "measured-glyph-pixels",
                    "familyCandidates": [
                        {
                            "family": family,
                            "widthEm": width,
                            "distance": distance,
                            "status": "candidate",
                            "method": "glyph-width-silhouette",
                        }
                        for family, width, distance in (
                            ("inter", 0.5, 0.06),
                            ("roboto", 0.51, 0.07),
                            ("montserrat", 0.55, 0.11),
                            ("anton", 0.72, 0.28),
                        )
                    ],
                    "method": "measured-glyph-geometry",
                },
                "fontStrategy": "match-by-glyph-metrics",
            }
        )

    focus_plan = [
        {
            "tool": "sens_zoom",
            "reason": "Resolve exact text.",
            "arguments": {
                "region": {"x": 20, "y": 30, "width": 200, "height": 20},
                "profile": "reconstruct",
                "response": "compact",
                "targetKind": "web",
            },
        }
    ]
    dense_controls = [
        {
            "elementId": f"inferred-control-{index}",
            "kind": "button" if index % 4 == 0 else None,
            "interaction": "semantic-control-required",
            "boxSource": [20, 30 + index * 12, 220, 40 + index * 12],
            "labelElementIds": [(index % 60) + 1],
            "visibleBoundary": index % 4 == 0,
            "background": "#00000000",
            "borderColor": "#00000000",
            "borderWidth": 0,
            "cornerRadius": 8,
            "semanticRole": "nav",
            "source": "inferred-from-visible-affordance",
            "epistemic": "inferred",
            "interactionEvidence": "repeated-aligned-visible-ui-row",
            "behavior": "local-placeholder-no-invented-destination",
            "decorationPreservedInBackgroundArtwork": True,
        }
        for index in range(30)
    ]
    dense_icons = [
        {
            "elementId": f"inferred-icon-{index}",
            "name": "navigation-item",
            "boxSource": [10, 20 + index * 12, 18, 28 + index * 12],
            "color": "#434141",
            "strategy": "inline-svg",
            "source": "inferred-from-repeated-visible-row-pattern",
            "geometrySource": "inferred-from-repeated-visible-row-pattern",
            "semanticSource": "inferred-from-adjacent-live-label",
            "epistemic": "inferred",
            "preservedInBackgroundArtwork": True,
        }
        for index in range(30)
    ]
    dense_badges = [
        {
            "elementId": f"badge-{index}",
            "labelElementId": index + 1,
            "boxSource": [220, 30 + index * 12, 240, 42 + index * 12],
            "textBoxSource": [226, 32 + index * 12, 234, 40 + index * 12],
            "background": "#00000000",
            "foreground": "#2D6CFB",
            "cornerRadius": 5.3,
            "value": str(index),
            "confidence": 0.9,
            "verified": True,
            "geometrySource": "measured",
            "epistemic": "inferred",
            "representation": "live-text-on-css-surface",
            "method": "compact-tinted-badge-local-font-atlas",
            "decorationPreservedInBackgroundArtwork": True,
            "borderColor": "#00000000",
            "borderWidth": 0,
        }
        for index in range(12)
    ]
    doc = {
        **_stub_document("reconstruct"),
        "tokens": {"colors": ["#FFFFFF", "#111111"]},
        "measurements": [{"kind": "alignment", "detail": "left"}],
        "reconstruction": {
            "targetKind": "web",
            "canvas": {"width": 1440, "height": 900, "dpr": 1},
            "contentPolicy": "visible-content-only",
            "representationPolicy": {
                "text": "live-selectable-dom",
                "controls": "semantic-html",
            },
            "text": verbose_text,
            "symbolArt": [],
            "visualControlCandidates": dense_controls,
            "decorativeShapes": [],
            "structuralLines": [],
            "layoutRegions": [
                {
                    "regionId": 1,
                    "role": "content",
                    "boxSource": [10, 10, 990, 490],
                    "elementIds": [1, 2],
                }
            ],
            "surfaces": [
                {
                    "boxSource": [10, 10, 990, 490],
                    "background": "#FFFFFF",
                    "borderColor": "#DDDDDD",
                    "borderWidth": 1,
                    "cornerRadius": 12,
                    "source": "measured",
                }
            ],
            "icons": dense_icons,
            "badges": dense_badges,
            "allowedRasterRegions": [],
            "excludedRasterCandidates": [],
            "primaryAsset": None,
            "monospaceContent": None,
            "blockingUncertainties": [{"kind": "text"}] * 60,
            "textVerificationPlan": focus_plan,
            "focusPlan": focus_plan,
            "semanticStrategy": {"mode": "focused-regions"},
            "semanticTextCandidate": None,
            "completionGate": {
                "tool": "sens_review",
                "requires": ["visual-pass", "web-pass"],
            },
            "rasterAssetRule": {
                "strategy": "extract-source-crop-verbatim",
                "prohibitedFollowUps": ["sens_inspect", "sens_ask"],
            },
            "workflow": {
                "state": "needs-focus",
                "nextAction": "execute-returned-focus-plan",
                "forbiddenActions": ["shell-image-analysis"],
            },
            "implementationRules": ["A deliberately verbose repeated rule."] * 12,
        },
    }

    compact = ops._compact_document(doc)
    reconstruction = compact["reconstruction"]
    payload = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))

    assert len(payload.encode("utf-8")) < 15_000
    assert reconstruction["focusPlan"] == {
        "encoding": "response-reference",
        "path": "summary.nextActions",
        "count": 1,
    }
    assert reconstruction["completionGate"]["tool"] == "sens_review"
    text_table = reconstruction["text"]
    assert text_table["encoding"] == "jsonl-arrays"
    assert text_table["count"] == 60
    first_text = dict(
        zip(text_table["columns"], json.loads(text_table["rows"].splitlines()[0]))
    )
    assert first_text["boxSource"] == [20, 30, 220, 40]
    assert first_text["text"] == "Dashboard label 1"
    assert text_table["constants"]["fontFamily"] == "inter"
    assert text_table["constants"]["color"] == "#112233"
    assert "boxNormSource" not in text_table["columns"]
    assert "fontClass" not in text_table["columns"]
    assert "fontWeight" not in text_table["columns"]
    assert "inlineRuns" not in text_table["columns"]
    assert reconstruction["visualControlCandidates"]["encoding"] == "jsonl-arrays"
    assert reconstruction["visualControlCandidates"]["count"] == 30
    assert reconstruction["icons"]["encoding"] == "jsonl-arrays"
    assert reconstruction["icons"]["count"] == 30
    assert reconstruction["badges"]["encoding"] == "jsonl-arrays"
    assert reconstruction["badges"]["count"] == 12
    assert "blockingUncertainties" not in reconstruction
    assert "textVerificationPlan" not in reconstruction
    assert len(reconstruction["implementationRules"]) == 7
    assert any(
        "selectable DOM text" in rule
        for rule in reconstruction["implementationRules"]
    )
    assert reconstruction["rasterAssetRule"]["strategy"] == (
        "extract-source-crop-verbatim"
    )
    assert reconstruction["layoutRegions"][0]["role"] == "content"
    assert reconstruction["surfaces"][0]["background"] == "#FFFFFF"
    assert reconstruction["workflow"]["nextAction"] == (
        "execute-returned-focus-plan"
    )

    brief = ops._implementation_brief(doc, "D:/project/assets/sens-contract.json")
    brief_payload = json.dumps(brief, ensure_ascii=False, separators=(",", ":"))
    assert len(brief_payload.encode("utf-8")) < 12_000
    assert brief["contract"]["path"].endswith("sens-contract.json")
    assert brief["text"]["count"] == 60
    assert brief["text"]["columns"] == ["id", "text", "box"]
    assert brief["text"]["constants"]["familyHint"] == "inter"
    assert brief["controls"]["count"] == 30
    assert brief["icons"]["count"] == 30
    assert brief["badges"]["count"] == 12


def test_brief_exposes_flat_control_style_palette_and_starter() -> None:
    doc = {
        **_stub_document("reconstruct"),
        "tokens": {
            "color": {
                "background": {"$type": "color", "$value": "#FCF7EF"},
                "accent": {"$type": "color", "$value": "#006EFF"},
            }
        },
        "reconstruction": {
            "targetKind": "web",
            "canvas": {"width": 800, "height": 500},
            "text": [
                {
                    "elementId": 2,
                    "value": "TICKETS",
                    "preferredValue": None,
                    "boxSource": [80, 420, 180, 450],
                }
            ],
            "visualControlCandidates": [
                {
                    "elementId": 3,
                    "boxSource": [40, 400, 220, 470],
                    "visibleBoundary": True,
                    "background": "#FCF7EF",
                    "borderColor": "#006EFF",
                    "borderWidth": 2,
                    "cornerRadius": 35,
                    "labelElementIds": [2],
                    "interaction": "semantic-control-required",
                }
            ],
            "structuralLines": [],
            "surfaces": [],
            "icons": [
                {
                    "elementId": 4,
                    "name": "cross",
                    "boxSource": [10, 10, 30, 30],
                    "color": "#111111",
                    "strategy": "css-or-inline-svg",
                }
            ],
            "allowedRasterRegions": [],
            "layoutRegions": [],
            "symbolArt": [],
            "starterProject": {"entryPath": "D:/assets/starter/index.html"},
        },
    }

    brief = ops._implementation_brief(doc, "D:/assets/contract.json")

    assert brief["schemaVersion"] == "sens-web-brief-3"
    assert brief["palette"]["background"]["$value"] == "#FCF7EF"
    controls = brief["controls"]
    control = dict(zip(controls["columns"], json.loads(controls["rows"])))
    assert control == {
        "elementId": 3,
        "boxSource": [40, 400, 220, 470],
        "visibleBoundary": True,
        "background": "#FCF7EF",
        "borderColor": "#006EFF",
        "borderWidth": 2,
        "cornerRadius": 35,
        "labelElementIds": [2],
        "interaction": "semantic-control-required",
        "label": "TICKETS",
    }
    icons = brief["icons"]
    icon = dict(zip(icons["columns"], json.loads(icons["rows"])))
    assert icon["name"] == "cross"
    assert brief["starterProject"]["entryPath"].endswith("index.html")


def test_internal_focus_resolution_merges_text_style_and_clears_completed_calls() -> None:
    action = {
        "tool": "sens_zoom",
        "reason": "Resolve an exact-text candidate before implementation.",
        "evidence": "STANDARO",
        "arguments": {
            "region": {"x": 120, "y": 300, "width": 360, "height": 90},
            "profile": "reconstruct",
            "response": "compact",
            "targetKind": "web",
        },
    }
    document = {
        **_stub_document("reconstruct"),
        "reconstruction": {
            "targetKind": "web",
            "text": [
                {
                    "elementId": 7,
                    "value": "STANDARO",
                    "preferredValue": None,
                    "status": "candidate",
                    "resolutionStatus": "unresolved",
                    "boxSource": [120, 300, 480, 390],
                    "typographyCandidate": None,
                }
            ],
            "focusPlan": [action],
            "textVerificationPlan": [action],
            "blockingUncertainties": [{"elementId": 7}],
            "semanticStrategy": {"mode": "focused-regions"},
        },
    }
    regional_document = {
        **_stub_document("reconstruct"),
        "semantics_status": "ok",
        "reconstruction": {
            "text": [
                {
                    "elementId": 1,
                    "value": "STANDARD",
                    "preferredValue": "STANDARD",
                    "status": "confirmed",
                    "resolutionStatus": "confirmed",
                    "boxSource": [122, 302, 476, 388],
                    "typographyCandidate": {
                        "class": "slab-serif",
                        "contrast": "high",
                        "confidence": 0.94,
                    },
                }
            ],
            "semanticTextCandidate": {
                "text": "STANDARD",
                "sourceBox": [120, 300, 480, 390],
            },
        },
    }

    ops._merge_focus_documents(document, [(action, regional_document)])

    spec = document["reconstruction"]
    text = spec["text"][0]
    assert text["preferredValue"] == "STANDARD"
    assert text["status"] == "confirmed"
    assert text["resolutionStatus"] == "confirmed"
    assert text["typographyCandidate"]["class"] == "slab-serif"
    assert spec["focusPlan"] == []
    assert spec["textVerificationPlan"] == []
    assert spec["semanticStrategy"]["mode"] == "internally-resolved-focus-regions"
    assert spec["semanticStrategy"]["resolvedCalls"] == 1
    assert spec["resolvedFocus"][0]["status"] == "resolved"


def test_internal_focus_resolution_keeps_failed_region_as_followup() -> None:
    action = {
        "tool": "sens_zoom",
        "reason": "Resolve exact text.",
        "arguments": {
            "region": {"x": 20, "y": 30, "width": 200, "height": 20}
        },
    }
    document = {
        **_stub_document("reconstruct"),
        "reconstruction": {
            "targetKind": "web",
            "text": [],
            "focusPlan": [action],
            "textVerificationPlan": [action],
            "semanticStrategy": {"mode": "focused-regions"},
        },
    }

    ops._merge_focus_documents(document, [(action, None)])

    spec = document["reconstruction"]
    assert spec["focusPlan"] == [action]
    assert spec["semanticStrategy"]["failedCalls"] == 1
    assert spec["resolvedFocus"][0]["status"] == "failed"


def test_discovery_focus_adds_regional_text_that_full_frame_ocr_missed() -> None:
    action = {
        "tool": "sens_zoom",
        "reason": "Re-analyze unresolved visible text at higher effective resolution.",
        "reasons": ["unresolved_text_density"],
        "evidence": "unresolved visible text",
        "arguments": {
            "region": {"x": 0, "y": 0, "width": 540, "height": 338}
        },
    }
    document = {
        **_stub_document("reconstruct"),
        "reconstruction": {
            "targetKind": "web",
            "text": [
                {
                    "elementId": 7,
                    "value": "Of Chains",
                    "preferredValue": "Of chains",
                    "status": "confirmed",
                    "boxSource": [57, 227, 461, 298],
                }
            ],
            "focusPlan": [action],
            "textVerificationPlan": [action],
            "semanticStrategy": {"mode": "focused-regions"},
        },
    }
    regional_document = {
        **_stub_document("reconstruct"),
        "semantics_status": "ok",
        "reconstruction": {
            "text": [
                {
                    "elementId": 1,
                    "value": "The Internet",
                    "preferredValue": "The Internet",
                    "status": "confirmed",
                    "confidence": 0.97,
                    "boxSource": [57, 137, 580, 207],
                    "typographyCandidate": {
                        "class": "display-sans",
                        "weight": "black",
                        "confidence": 0.94,
                    },
                },
                {
                    "elementId": 2,
                    "value": "Of Chains",
                    "preferredValue": "Of chains",
                    "status": "confirmed",
                    "confidence": 0.96,
                    "boxSource": [57, 227, 461, 298],
                },
            ],
            "semanticTextCandidate": {
                "text": "The Internet\nOf chains",
                "sourceBox": [0, 0, 540, 338],
            },
        },
    }

    ops._merge_focus_documents(document, [(action, regional_document)])

    entries = document["reconstruction"]["text"]
    assert [entry.get("preferredValue") for entry in entries] == [
        "Of chains",
        "The Internet",
    ]
    discovered = entries[1]
    assert discovered["elementId"] != 1
    assert discovered["resolutionStatus"] == "regional-new-text-candidate"
    assert discovered["geometrySource"] == "bounded-regional-measurement"
    assert document["reconstruction"]["focusPlan"] == []
    assert document["reconstruction"]["resolvedFocus"][0]["textUpdates"] == 2


def test_internal_focus_cannot_drop_currency_or_override_axis_sequence() -> None:
    action = {
        "tool": "sens_zoom",
        "reason": "Resolve grouped live text and typography.",
        "arguments": {
            "region": {"x": 300, "y": 150, "width": 180, "height": 260}
        },
    }
    document = {
        **_stub_document("reconstruct"),
        "reconstruction": {
            "targetKind": "web",
            "text": [
                {
                    "elementId": 1,
                    "value": "$210,429.00",
                    "preferredValue": "$210,429.00",
                    "status": "confirmed",
                    "resolutionStatus": "confirmed",
                    "boxSource": [323, 180, 459, 204],
                },
                {
                    "elementId": 2,
                    "value": "S2oo",
                    "preferredValue": "$200",
                    "status": "candidate",
                    "resolutionStatus": "layout-sequence-inferred",
                    "boxSource": [323, 380, 347, 389],
                },
            ],
            "focusPlan": [action],
            "textVerificationPlan": [action],
            "semanticStrategy": {"mode": "focused-regions"},
        },
    }
    regional_document = {
        **_stub_document("reconstruct"),
        "semantics_status": "ok",
        "reconstruction": {
            "text": [
                {
                    "elementId": 10,
                    "value": "210,429.00",
                    "preferredValue": "210,429.00",
                    "status": "confirmed",
                    "resolutionStatus": "confirmed",
                    "boxSource": [323, 180, 459, 204],
                },
                {
                    "elementId": 11,
                    "value": "210",
                    "preferredValue": "210",
                    "status": "confirmed",
                    "resolutionStatus": "confirmed",
                    "boxSource": [323, 380, 347, 389],
                },
            ],
            "semanticTextCandidate": {
                "text": "210,429.00\n210",
                "sourceBox": [300, 150, 480, 410],
            },
        },
    }

    ops._merge_focus_documents(document, [(action, regional_document)])

    values = {
        entry["elementId"]: entry["preferredValue"]
        for entry in document["reconstruction"]["text"]
    }
    assert values == {1: "$210,429.00", 2: "$200"}


def test_single_line_vlm_reflow_is_removed_from_copy_and_urls() -> None:
    document = {
        "reconstruction": {
            "targetKind": "web",
            "text": [
                {
                    "elementId": 1,
                    "value": "Recentcommisslons",
                    "preferredValue": "Recent com\nmissions",
                    "boxSource": [843, 543, 947, 552],
                    "fontFeatures": {"fontSize": 12},
                },
                {
                    "elementId": 2,
                    "value": "partingre duble ormipregrarmslae mg-prelee t",
                    "preferredValue": "partners.dub\\n\\dub.com/programs/acme-\\nme-projects",
                    "boxSource": [795, 466, 984, 476],
                    "fontFeatures": {"fontSize": 11},
                },
            ],
        }
    }

    ops._sanitize_single_line_preferred_text(document)

    entries = document["reconstruction"]["text"]
    assert entries[0]["preferredValue"] == "Recent commissions"
    assert entries[1]["preferredValue"] == (
        "partners.dub.com/programs/acme-projects"
    )
    assert all(
        entry["resolutionStatus"] == "single-line-reflow-repaired"
        for entry in entries
    )


def test_internal_focus_prefers_semantic_text_when_regional_ocr_only_adds_style() -> None:
    action = {
        "tool": "sens_zoom",
        "reason": "Resolve an exact-text candidate before implementation.",
        "evidence": "5-1OРM",
        "arguments": {
            "region": {"x": 1900, "y": 330, "width": 500, "height": 170}
        },
    }
    document = {
        **_stub_document("reconstruct"),
        "reconstruction": {
            "targetKind": "web",
            "text": [
                {
                    "elementId": 9,
                    "value": "5-1OРM",
                    "preferredValue": None,
                    "status": "candidate",
                    "resolutionStatus": "unresolved",
                    "boxSource": [1993, 337, 2458, 494],
                },
                {
                    "elementId": 10,
                    "value": "TIME",
                    "preferredValue": None,
                    "status": "stable-candidate",
                    "resolutionStatus": "unresolved",
                    "boxSource": [1905, 330, 1960, 350],
                },
            ],
            "focusPlan": [action],
            "textVerificationPlan": [action],
            "semanticStrategy": {"mode": "focused-regions"},
        },
    }
    regional_document = {
        **_stub_document("reconstruct"),
        "semantics_status": "ok",
        "reconstruction": {
            "text": [
                {
                    "elementId": 1,
                    "value": "5-1OРM",
                    "preferredValue": None,
                    "status": "candidate",
                    "resolutionStatus": "unresolved",
                    "boxSource": [1993, 337, 2458, 494],
                    "typographyCandidate": {
                        "class": "slab-serif",
                        "contrast": "high",
                    },
                }
            ],
            "semanticTextCandidate": {
                "text": "5–10 PM",
                "sourceBox": [1900, 330, 2400, 500],
            },
        },
    }

    ops._merge_focus_documents(document, [(action, regional_document)])

    text = document["reconstruction"]["text"][0]
    assert text["preferredValue"] == "5–10 PM"
    assert text["resolutionStatus"] == "vlm-preferred-candidate"
    assert document["reconstruction"]["resolvedFocus"][0]["textUpdates"] == 1


def test_internal_focus_applies_semantic_typography_when_regional_ocr_is_empty() -> None:
    action = {
        "tool": "sens_zoom",
        "reason": "Targeted typography check for a large display.",
        "evidence": "WIth ALDLP",
        "arguments": {
            "region": {"x": 41, "y": 393, "width": 509, "height": 106}
        },
    }
    document = {
        "reconstruction": {
            "targetKind": "web",
            "text": [
                {
                    "elementId": 17,
                    "value": "WIth ALDLP",
                    "preferredValue": "with AI DLP",
                    "boxSource": [60, 400, 531, 492],
                    "typographyCandidate": None,
                }
            ],
            "focusPlan": [action],
            "textVerificationPlan": [action],
            "semanticStrategy": {"mode": "focused-regions"},
        }
    }
    regional_document = {
        "reconstruction": {
            "text": [],
            "semanticTextCandidate": {
                "text": "with AI DLP",
                "typographyRuns": [
                    {
                        "text": "with AI DLP",
                        "class": "script",
                        "contrast": "high",
                        "width": "normal",
                        "weight": "bold",
                        "case": "mixed",
                        "confidence": 0.9,
                    }
                ],
            },
        }
    }

    ops._merge_focus_documents(document, [(action, regional_document)])

    [entry] = document["reconstruction"]["text"]
    assert entry["typographyCandidate"]["class"] == "script"
    assert entry["typographyCandidate"]["method"] == (
        "local-vlm-region-text-inspection"
    )
    assert entry["typographyEvidenceBoxSource"] == [41, 393, 550, 499]


def test_internal_focus_preserves_mixed_word_styles_as_inline_live_text() -> None:
    action = {
        "tool": "sens_zoom",
        "reason": "Targeted typography check for a large display.",
        "evidence": "YOUr N\u00d0\u00a1.",
        "arguments": {
            "region": {"x": 41, "y": 175, "width": 509, "height": 105}
        },
    }
    document = {
        "reconstruction": {
            "targetKind": "web",
            "text": [
                {
                    "elementId": 10,
                    "value": "YOUr N\u00d0\u00a1.",
                    "preferredValue": "Your new",
                    "boxSource": [60, 186, 465, 272],
                    "typographyCandidate": None,
                }
            ],
            "focusPlan": [action],
            "textVerificationPlan": [action],
            "semanticStrategy": {"mode": "focused-regions"},
        }
    }
    regional_document = {
        "reconstruction": {
            "text": [],
            "semanticTextCandidate": {
                "text": "Your new",
                "typographyRuns": [
                    {
                        "text": "Your",
                        "class": "sans-serif",
                        "contrast": "low",
                        "width": "normal",
                        "weight": "light",
                        "slant": "normal",
                        "case": "mixed",
                        "confidence": 0.9,
                    },
                    {
                        "text": "new",
                        "class": "serif",
                        "contrast": "high",
                        "width": "expanded",
                        "weight": "light",
                        "slant": "italic",
                        "case": "lowercase",
                        "confidence": 0.9,
                    },
                ],
            },
        }
    }

    ops._merge_focus_documents(document, [(action, regional_document)])

    [entry] = document["reconstruction"]["text"]
    assert [run["text"] for run in entry["inlineRuns"]] == ["Your ", "new"]
    assert entry["inlineRuns"][0]["typographyCandidate"]["class"] == "sans-serif"
    assert entry["inlineRuns"][1]["typographyCandidate"]["class"] == "serif"
    assert entry["inlineRuns"][1]["typographyCandidate"]["slant"] == "italic"
    assert entry["inlineRunMethod"] == "bounded-regional-vlm-word-styles"
    assert entry["typographyEvidenceBoxSource"] == [41, 175, 550, 280]

    entry["preferredValue"] = "Your new"
    entry["inlineRuns"][0]["text"] = "YOUr "
    entry["inlineRuns"][1]["text"] = "N\u00d0\u00a1."
    ops._reconcile_inline_run_text(document)
    assert [run["text"] for run in entry["inlineRuns"]] == ["Your ", "new"]

    entry["preferredValue"] = "with AI DLP"
    entry["inlineRuns"][0]["text"] = "WIth "
    entry["inlineRuns"][1]["text"] = "ALDLP"
    ops._reconcile_inline_run_text(document)
    assert [run["text"] for run in entry["inlineRuns"]] == [
        "with ",
        "AI DLP",
    ]


def test_grouped_semantic_phrase_is_distributed_over_measured_text_boxes() -> None:
    action = {
        "tool": "sens_zoom",
        "reason": "Resolve grouped live text and typography.",
        "evidence": "STIANDARD | HALL | WE RENTED IT OUT | TIOKETS",
        "arguments": {
            "region": {"x": 126, "y": 780, "width": 577, "height": 409}
        },
    }
    base_values = [
        (1, "WE RENTED IT OUT", [282, 790, 548, 824]),
        (2, "STIANDARD", [146, 842, 683, 955]),
        (3, "HALL", [267, 950, 562, 1072]),
        (4, "TIOKETS", [321, 1131, 512, 1179]),
    ]
    document = {
        "reconstruction": {
            "targetKind": "web",
            "text": [
                {
                    "elementId": element_id,
                    "value": value,
                    "preferredValue": None,
                    "status": "candidate",
                    "resolutionStatus": "unresolved",
                    "boxSource": box,
                }
                for element_id, value, box in base_values
            ],
            "focusPlan": [action],
            "textVerificationPlan": [action],
            "semanticStrategy": {"mode": "focused-regions"},
        }
    }
    regional_document = {
        "reconstruction": {
            "text": [
                {
                    "elementId": 11,
                    "value": "WE RENTED IT OUT",
                    "preferredValue": None,
                    "status": "stable-candidate",
                    "resolutionStatus": "unresolved",
                    "boxSource": [283, 790, 547, 823],
                },
                {
                    "elementId": 12,
                    "value": "SIANDARD",
                    "preferredValue": "STANDARD",
                    "status": "candidate",
                    "resolutionStatus": "vlm-preferred-candidate",
                    "boxSource": [139, 834, 696, 967],
                },
                {
                    "elementId": 13,
                    "value": "HALL",
                    "preferredValue": "HALL",
                    "status": "confirmed",
                    "resolutionStatus": "confirmed",
                    "boxSource": [264, 947, 569, 1073],
                },
                {
                    "elementId": 14,
                    "value": "TIOCKETS",
                    "preferredValue": "TICKETS",
                    "status": "candidate",
                    "resolutionStatus": "vlm-preferred-candidate",
                    "boxSource": [323, 1131, 510, 1179],
                },
            ],
            "semanticTextCandidate": {
                "text": "WE RENTED IT OUT STANDARD HALL TICKETS"
            },
        }
    }

    ops._merge_focus_documents(document, [(action, regional_document)])

    values = [
        entry["preferredValue"]
        for entry in document["reconstruction"]["text"]
    ]
    assert values == ["WE RENTED IT OUT", "STANDARD", "HALL", "TICKETS"]
    assert "WE RENTED IT OUT STANDARD HALL TICKETS" not in values
    assert document["reconstruction"]["resolvedFocus"][0]["textUpdates"] == 4


def test_crop_semantics_partition_a_garbled_multiline_paragraph_without_truncation() -> None:
    action = {
        "tool": "sens_zoom",
        "reason": "Resolve grouped live text and typography.",
        "evidence": (
            "EDL FLAIHE FHUM THE KITEHENE | "
            "OF INIA. CRAFTED FOR THOSE TIHO | CFAIE MOFE. | BUy now"
        ),
        "arguments": {
            "region": {"x": 16, "y": 763, "width": 237, "height": 103}
        },
    }
    document = {
        "reconstruction": {
            "targetKind": "web",
            "text": [
                {
                    "elementId": 10,
                    "value": "EDL FLAIHE FHUM THE KITEHENE",
                    "preferredValue": None,
                    "resolutionStatus": "unresolved",
                    "boxSource": [28, 773, 238, 782],
                },
                {
                    "elementId": 11,
                    "value": "OF INIA. CRAFTED FOR THOSE TIHO",
                    "preferredValue": "OF INDIA. CRAFTED FOR THOSE WHO",
                    "resolutionStatus": "vlm-preferred-candidate",
                    "boxSource": [28, 788, 241, 797],
                },
                {
                    "elementId": 12,
                    "value": "CFAIE MOFE.",
                    "preferredValue": "CRAVE",
                    "resolutionStatus": "vlm-preferred-candidate",
                    "boxSource": [28, 803, 106, 812],
                },
                {
                    "elementId": 14,
                    "value": "BUy now",
                    "preferredValue": "BUY NOW",
                    "resolutionStatus": "vlm-preferred-candidate",
                    "boxSource": [60, 847, 113, 856],
                },
            ],
            "focusPlan": [action],
            "textVerificationPlan": [action],
            "semanticStrategy": {"mode": "focused-regions"},
        }
    }
    regional_document = {
        "reconstruction": {
            "text": [],
            "semanticTextCandidate": {
                "text": (
                    "BOLD FLAVORS FROM THE KITCHENS "
                    "OF INDIA. CRAFTED FOR THOSE WHO CRAVE MORE. BUY NOW"
                )
            },
        }
    }

    ops._merge_focus_documents(document, [(action, regional_document)])

    values = {
        entry["elementId"]: entry["preferredValue"]
        for entry in document["reconstruction"]["text"]
    }
    assert values == {
        10: "BOLD FLAVORS FROM THE KITCHENS",
        11: "OF INDIA. CRAFTED FOR THOSE WHO",
        12: "CRAVE MORE.",
        14: "BUY NOW",
    }
    for entry in document["reconstruction"]["text"][:3]:
        assert entry["resolutionStatus"] == (
            "regional-multiline-semantic-partition"
        )
        assert entry["resolutionMethod"] == "measured-row-fuzzy-partition"


def test_internal_focus_groups_cover_all_live_text_with_four_bounded_crops() -> None:
    entries = [
        (1, "THE SUMMER", [5, 8, 2551, 304]),
        (2, "DRIVE", [677, 254, 1887, 544]),
        (3, "DATE", [273, 299, 356, 331]),
        (4, "06.24.21", [102, 339, 527, 490]),
        (5, "TAGLINE", [268, 594, 2288, 657]),
        (6, "STANDARD HALL", [146, 790, 683, 1072]),
        (7, "RESERVE NOW", [1958, 790, 2411, 1075]),
        (8, "ASCII NOISE", [700, 760, 1800, 1200]),
        (9, "LEFT TICKETS", [321, 1131, 512, 1179]),
        (10, "RIGHT TICKETS", [2093, 1131, 2281, 1179]),
    ]
    spec = {
        "targetKind": "web",
        "canvas": {"width": 2557, "height": 1273},
        "text": [
            {"elementId": element_id, "value": value, "boxSource": box}
            for element_id, value, box in entries
        ],
        "symbolArt": [{"boxSource": [650, 700, 1850, 1250]}],
    }

    actions = ops._grouped_text_focus_actions(spec, 4)

    assert len(actions) == 4
    assert all(action["tool"] == "sens_zoom" for action in actions)
    assert all(action["arguments"]["profile"] == "reconstruct" for action in actions)
    assert all(action["arguments"]["targetKind"] == "web" for action in actions)
    evidence = " ".join(str(action.get("evidence") or "") for action in actions)
    assert "ASCII NOISE" not in evidence
    for _element_id, value, box in [entry for entry in entries if entry[1] != "ASCII NOISE"]:
        center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        assert any(
            (region := action["arguments"]["region"])["x"]
            <= center[0]
            <= region["x"] + region["width"]
            and region["y"]
            <= center[1]
            <= region["y"] + region["height"]
            for action in actions
        ), value
    evidence_sets = [str(action.get("evidence") or "") for action in actions]
    assert "THE SUMMER" in evidence_sets
    assert any("STANDARD HALL" in value and "LEFT TICKETS" in value for value in evidence_sets)
    assert any("RESERVE NOW" in value and "RIGHT TICKETS" in value for value in evidence_sets)
    assert any("DRIVE" in value and "TAGLINE" in value for value in evidence_sets)

    quality_actions = ops._grouped_text_focus_actions(spec, 5)
    assert len(quality_actions) == 5
    assert any(action.get("evidence") == "TAGLINE" for action in quality_actions)


def test_internal_focus_never_maps_a_contained_small_label_to_a_large_heading() -> None:
    action = {
        "tool": "sens_zoom",
        "reason": "Resolve grouped live text and typography.",
        "evidence": "HEMER",
        "arguments": {
            "region": {"x": 0, "y": 0, "width": 1000, "height": 320}
        },
    }
    document = {
        **_stub_document("reconstruct"),
        "reconstruction": {
            "targetKind": "web",
            "text": [
                {
                    "elementId": 1,
                    "value": "HEMER",
                    "preferredValue": None,
                    "status": "candidate",
                    "resolutionStatus": "unresolved",
                    "boxSource": [0, 0, 1000, 300],
                }
            ],
            "focusPlan": [action],
            "textVerificationPlan": [action],
            "semanticStrategy": {"mode": "focused-regions"},
        },
    }
    regional_document = {
        **_stub_document("reconstruct"),
        "semantics_status": "ok",
        "reconstruction": {
            "text": [
                {
                    "elementId": 2,
                    "value": "TIME",
                    "preferredValue": "TIME",
                    "status": "confirmed",
                    "resolutionStatus": "confirmed",
                    "boxSource": [850, 285, 930, 315],
                    "typographyCandidate": {"class": "sans-serif"},
                }
            ],
            "semanticTextCandidate": {"text": "TIME"},
        },
    }

    ops._merge_focus_documents(document, [(action, regional_document)])

    heading = document["reconstruction"]["text"][0]
    assert heading["preferredValue"] is None
    assert heading["status"] == "candidate"
    assert document["reconstruction"]["focusPlan"] == [action]


def test_internal_focus_uses_two_targeted_fallbacks_only_for_unresolved_text() -> None:
    spec = {
        "targetKind": "web",
        "canvas": {"width": 2557, "height": 1273},
        "text": [
            {
                "elementId": 1,
                "value": "DBINЕ",
                "confidence": 0.74,
                "resolutionStatus": "unresolved",
                "boxSource": [677, 254, 1887, 544],
                "typographyCandidate": None,
            },
            {
                "elementId": 2,
                "value": "TIME",
                "confidence": 0.95,
                "resolutionStatus": "unresolved",
                "boxSource": [2190, 299, 2263, 331],
                "typographyCandidate": None,
            },
            {
                "elementId": 3,
                "value": "DATE",
                "confidence": 0.99,
                "resolutionStatus": "confirmed",
                "boxSource": [273, 299, 356, 331],
                "typographyCandidate": {"class": "sans-serif"},
            },
        ],
        "symbolArt": [],
    }

    actions = ops._fallback_text_focus_actions(spec, 2)

    assert [action["evidence"] for action in actions] == ["DBINЕ", "TIME"]
    assert all("fallback" in action["reason"].casefold() for action in actions)


def test_grouped_focus_marks_measured_resolved_text_as_nonblocking() -> None:
    spec = {
        "targetKind": "web",
        "canvas": {"width": 1440, "height": 900},
        "text": [
            {
                "elementId": 1,
                "value": "Explore Chains",
                "verified": True,
                "confidence": 0.998,
                "resolutionStatus": "confirmed",
                "boxSource": [944, 842, 1068, 859],
            },
            {
                "elementId": 2,
                "value": "Book A Call",
                "preferredValue": "Book A Call",
                "resolutionStatus": "regional-new-text-candidate",
                "boxSource": [1217, 842, 1310, 855],
            },
        ],
        "symbolArt": [],
    }

    [action] = ops._grouped_text_focus_actions(spec, 1)

    assert action["blocking"] is False


def test_failed_optional_group_does_not_remain_in_focus_plan() -> None:
    action = {
        "tool": "sens_zoom",
        "reason": "Resolve grouped live text and typography.",
        "evidence": "Explore Chains | Book A Call",
        "blocking": False,
        "arguments": {
            "region": {"x": 900, "y": 800, "width": 430, "height": 90}
        },
    }
    document = {
        "reconstruction": {
            "targetKind": "web",
            "text": [],
            "focusPlan": [action],
            "textVerificationPlan": [action],
        }
    }

    ops._merge_focus_documents(document, [(action, None)])

    spec = document["reconstruction"]
    assert spec["focusPlan"] == []
    assert spec["resolvedFocus"][0]["status"] == "optional-no-update"


def test_targeted_typography_focus_prioritizes_large_numeric_displays() -> None:
    spec = {
        "targetKind": "web",
        "canvas": {"width": 2557, "height": 1273},
        "text": [
            {
                "elementId": 1,
                "value": "06.24.21",
                "preferredValue": "06.24.21",
                "boxSource": [102, 339, 527, 490],
                "fontFeatures": {"fontSize": 158},
                "typographyCandidate": {
                    "class": "sans-serif",
                    "weight": "regular",
                },
            },
            {
                "elementId": 2,
                "value": "5-10 PM",
                "preferredValue": "5-10 PM",
                "boxSource": [1993, 337, 2458, 494],
                "fontFeatures": {"fontSize": 156},
                "typographyCandidate": None,
            },
            {
                "elementId": 3,
                "value": "$210,429.00",
                "boxSource": [320, 179, 454, 203],
                "fontFeatures": {"fontSize": 26},
                "typographyCandidate": {
                    "class": "sans-serif",
                    "weight": "bold",
                },
            },
        ],
    }

    actions = ops._typography_focus_actions(spec, 2)

    assert {action["evidence"] for action in actions} == {"06.24.21", "5-10 PM"}
    assert all("typography" in action["reason"].casefold() for action in actions)


def test_targeted_typography_focus_uses_missing_large_nonnumeric_styles() -> None:
    spec = {
        "targetKind": "web",
        "canvas": {"width": 1440, "height": 900},
        "text": [
            {
                "elementId": 1,
                "value": "Your new",
                "boxSource": [60, 186, 465, 272],
                "fontFeatures": {
                    "fontSize": 103,
                    "weightCandidate": "light",
                    "characterCount": 7,
                    "measuredCharacterCount": 2,
                    "wordBoxesSource": [
                        {"text": "Your", "box": [68, 197, 258, 272]},
                        {"text": "new", "box": [268, 197, 455, 272]},
                    ],
                },
                "typographyCandidate": None,
            },
            {
                "elementId": 2,
                "value": "with AI DLP",
                "boxSource": [60, 400, 531, 492],
                "fontFeatures": {
                    "fontSize": 126,
                    "inkCoverage": 0.62,
                    "characterCount": 9,
                    "measuredCharacterCount": 1,
                },
                "typographyCandidate": None,
            },
            {
                "elementId": 3,
                "value": "Secure Web",
                "boxSource": [74, 268, 548, 334],
                "fontFeatures": {"fontSize": 88},
                "typographyCandidate": {"class": "sans-serif"},
            },
            {
                "elementId": 4,
                "value": "Gateway",
                "boxSource": [62, 327, 448, 426],
                "fontFeatures": {
                    "fontSize": 136,
                    "weightCandidate": "bold",
                    "inkCoverage": 0.31,
                    "characterCount": 7,
                    "measuredCharacterCount": 2,
                },
                "typographyCandidate": None,
            },
        ],
    }

    actions = ops._typography_focus_actions(spec, 3)

    assert {action["evidence"] for action in actions} == {
        "Your new",
        "with AI DLP",
    }
    mixed = next(action for action in actions if action["evidence"] == "Your new")
    assert mixed["semanticCallCost"] == 2
    assert [entry["text"] for entry in mixed["inlineRegions"]] == ["Your", "new"]
    assert mixed["inlineRegions"][0]["region"] == {
        "x": 64,
        "y": 191,
        "width": 198,
        "height": 87,
    }


def test_narrow_typography_focus_is_not_overwritten_by_a_broader_group() -> None:
    narrow = {
        "tool": "sens_zoom",
        "reason": "Targeted typography check for a large numeric display.",
        "evidence": "5-10 PM",
        "arguments": {"region": {"x": 1980, "y": 325, "width": 500, "height": 180}},
    }
    broad = {
        "tool": "sens_zoom",
        "reason": "Resolve grouped live text and typography.",
        "evidence": "DATE | TIME | 5-10 PM | 06.24.21",
        "arguments": {"region": {"x": 80, "y": 285, "width": 2400, "height": 225}},
    }
    document = {
        "reconstruction": {
            "targetKind": "web",
            "text": [
                {
                    "elementId": 6,
                    "value": "5-1OРM",
                    "preferredValue": "5-10 PM",
                    "boxSource": [1993, 337, 2458, 494],
                    "typographyCandidate": {"class": "sans-serif"},
                }
            ],
            "focusPlan": [narrow, broad],
            "textVerificationPlan": [narrow, broad],
            "semanticStrategy": {"mode": "focused-regions"},
        }
    }
    narrow_result = {
        "reconstruction": {
            "text": [
                {
                    "value": "5-10 PM",
                    "preferredValue": "5-10 PM",
                    "boxSource": [13, 12, 478, 169],
                    "typographyCandidate": {"class": "slab-serif", "confidence": 0.91},
                }
            ],
            "semanticTextCandidate": {"text": "5-10 PM"},
        }
    }
    broad_result = {
        "reconstruction": {
            "text": [
                {
                    "value": "5-10 PM",
                    "preferredValue": "5-10 PM",
                    "boxSource": [1913, 52, 2378, 209],
                    "typographyCandidate": {"class": "sans-serif", "confidence": 0.84},
                }
            ],
            "semanticTextCandidate": {"text": "5-10 PM"},
        }
    }

    ops._merge_focus_documents(
        document,
        [(narrow, narrow_result), (broad, broad_result)],
    )

    [entry] = document["reconstruction"]["text"]
    assert entry["typographyCandidate"]["class"] == "slab-serif"
    assert entry["typographyEvidenceBoxSource"] == [1980, 325, 2480, 505]


def test_internal_focus_spends_seven_calls_with_typography_before_groups(
    monkeypatch,
) -> None:
    def action(evidence: str) -> dict:
        return {
            "tool": "sens_zoom",
            "reason": "Resolve text typography.",
            "evidence": evidence,
            "arguments": {"region": {"x": 0, "y": 0, "width": 10, "height": 10}},
        }

    merge_order: list[list[str]] = []
    document = {
        "reconstruction": {
            "targetKind": "web",
            "canvas": {"width": 100, "height": 100},
            "text": [],
            "focusPlan": [action("stale-builder-plan")],
            "textVerificationPlan": [action("stale-builder-plan")],
        }
    }

    monkeypatch.setattr(
        ops,
        "_typography_focus_actions",
        lambda _spec, budget: [action("type-1"), action("type-2")]
        if budget == 2
        else [],
    )
    monkeypatch.setattr(
        ops,
        "_grouped_text_focus_actions",
        lambda _spec, budget: [action(f"group-{index}") for index in range(budget)],
    )
    monkeypatch.setattr(ops, "_control_text_focus_actions", lambda _spec, _budget: [])
    monkeypatch.setattr(ops, "_fallback_text_focus_actions", lambda _spec, _budget: [])
    monkeypatch.setattr(
        ops,
        "see_document",
        lambda *_args, **_kwargs: {"doc": {"reconstruction": {}}},
    )

    def record_merge(_document, resolutions):
        merge_order.append([item[0]["evidence"] for item in resolutions])
        return _document

    monkeypatch.setattr(ops, "_merge_focus_documents", record_merge)

    ops._resolve_focus_plan(
        document,
        "reference.png",
        no_store=False,
        quality=False,
        pack="local",
        intent=None,
        max_semantic_calls=7,
        target_kind="web",
    )

    assert merge_order == [
        ["type-1", "type-2"],
        ["group-0", "group-1", "group-2", "group-3", "group-4"],
    ]
    assert all(
        item.get("evidence") != "stale-builder-plan"
        for item in document["reconstruction"]["focusPlan"]
    )


def test_internal_focus_runs_unresolved_text_discovery_before_typography(
    monkeypatch,
) -> None:
    def action(evidence: str, reasons=None) -> dict:
        return {
            "tool": "sens_zoom",
            "reason": "Resolve visible text.",
            "reasons": reasons or [],
            "evidence": evidence,
            "arguments": {"region": {"x": 0, "y": 0, "width": 20, "height": 20}},
        }

    discovery = action("unresolved visible text", ["unresolved_text_density"])
    document = {
        "reconstruction": {
            "targetKind": "web",
            "canvas": {"width": 100, "height": 100},
            "text": [],
            "focusPlan": [discovery],
            "textVerificationPlan": [discovery],
        }
    }
    merge_order: list[list[str]] = []
    monkeypatch.setattr(
        ops,
        "_typography_focus_actions",
        lambda _spec, budget: [action("type")] if budget == 1 else [],
    )
    monkeypatch.setattr(
        ops,
        "_grouped_text_focus_actions",
        lambda _spec, budget: [action(f"group-{index}") for index in range(budget)],
    )
    monkeypatch.setattr(ops, "_control_text_focus_actions", lambda _spec, _budget: [])
    monkeypatch.setattr(ops, "_fallback_text_focus_actions", lambda _spec, _budget: [])
    monkeypatch.setattr(
        ops,
        "see_document",
        lambda *_args, **_kwargs: {"doc": {"reconstruction": {}}},
    )

    def record_merge(_document, resolutions):
        merge_order.append([item[0]["evidence"] for item in resolutions])
        return _document

    monkeypatch.setattr(ops, "_merge_focus_documents", record_merge)

    ops._resolve_focus_plan(
        document,
        "reference.png",
        no_store=False,
        quality=False,
        pack="local",
        intent=None,
        max_semantic_calls=7,
        target_kind="web",
    )

    assert merge_order == [
        ["unresolved visible text"],
        ["type"],
        ["group-0", "group-1", "group-2", "group-3", "group-4"],
    ]


def test_dense_parallel_halftone_rows_are_not_structural_lines() -> None:
    artwork_rows = [
        {
            "boxSource": [640, 120 + index * 8, 1380 - (index % 3) * 80, 121 + index * 8],
            "orientation": "horizontal",
            "color": "#F54E06" if index % 2 else "#6549D3",
        }
        for index in range(12)
    ]
    real_rule = {
        "boxSource": [40, 500, 1400, 501],
        "orientation": "horizontal",
        "color": "#111111",
    }

    kept, excluded = ops._exclude_dense_parallel_artwork_lines(
        [real_rule, *artwork_rows], 900
    )

    assert kept == [real_rule]
    assert len(excluded) == 12
    assert {item["reason"] for item in excluded} == {
        "dense-parallel-halftone-artwork"
    }


def test_lines_inside_live_symbol_art_are_not_rendered_again() -> None:
    symbol_box = [0, 429, 2555, 1196]
    inside = {
        "boxSource": [1, 742, 1167, 743],
        "orientation": "horizontal",
        "lineStyle": "dashed",
    }
    outside = {
        "boxSource": [40, 400, 2515, 401],
        "orientation": "horizontal",
    }

    kept, excluded = ops._exclude_symbol_art_lines(
        [outside, inside], [{"boxSource": symbol_box}]
    )

    assert kept == [outside]
    assert excluded == [
        {
            **inside,
            "reason": "overlaps-live-symbol-art",
            "representation": "owned-by-preformatted-symbol-art",
        }
    ]


def test_repeated_text_consensus_repairs_a_large_unresolved_word() -> None:
    document = {
        "reconstruction": {
            "text": [
                {
                    "elementId": 1,
                    "value": "DBINЕ",
                    "preferredValue": None,
                    "resolutionStatus": "unresolved",
                },
                {
                    "elementId": 2,
                    "value": "A NO-WORK EVENT IN THE DRVE PORTFOLIO",
                    "preferredValue": "A NO-WORK EVENT IN THE DRIVE PORTFOLIO",
                    "resolutionStatus": "vlm-preferred-candidate",
                },
            ],
            "blockingUncertainties": [
                {"kind": "text_candidate", "elementId": 1},
                {"kind": "text_candidate", "elementId": 2},
            ],
        }
    }

    ops._resolve_repeated_text_consensus(document)

    [title, _sentence] = document["reconstruction"]["text"]
    assert title["preferredValue"] == "DRIVE"
    assert title["resolutionStatus"] == "cross-text-consensus-candidate"
    assert title["resolutionMethod"] == "repeated-word-in-resolved-context"
    assert document["reconstruction"]["blockingUncertainties"] == [
        {"kind": "text_candidate", "elementId": 2}
    ]


def test_repeated_text_consensus_overrides_weak_fragment_join() -> None:
    document = {
        "reconstruction": {
            "text": [
                {
                    "elementId": 1,
                    "value": "DBINE",
                    "preferredValue": "D BIVE",
                    "resolutionStatus": "regional-fragment-consensus",
                },
                {
                    "elementId": 2,
                    "value": "A TEAM EVENT IN THE DRVE PORTFOLIO",
                    "preferredValue": "A TEAM EVENT IN THE DRIVE PORTFOLIO",
                    "resolutionStatus": "vlm-preferred-candidate",
                },
            ],
            "blockingUncertainties": [{"kind": "text_candidate", "elementId": 1}],
        }
    }

    ops._resolve_repeated_text_consensus(document)

    [title, _sentence] = document["reconstruction"]["text"]
    assert title["preferredValue"] == "DRIVE"
    assert title["resolutionStatus"] == "cross-text-consensus-candidate"
    assert document["reconstruction"]["blockingUncertainties"] == []


def test_repeated_text_consensus_leaves_ambiguous_word_unresolved() -> None:
    document = {
        "reconstruction": {
            "text": [
                {
                    "elementId": 1,
                    "value": "DOME",
                    "preferredValue": None,
                    "resolutionStatus": "unresolved",
                },
                {
                    "elementId": 2,
                    "value": "HOME SOME",
                    "preferredValue": "HOME SOME",
                    "resolutionStatus": "confirmed",
                },
            ]
        }
    }

    ops._resolve_repeated_text_consensus(document)

    assert document["reconstruction"]["text"][0]["preferredValue"] is None


def test_merged_semantic_heading_is_partitioned_across_measured_lines() -> None:
    document = {
        "reconstruction": {
            "text": [
                {
                    "elementId": 7,
                    "value": "World-class",
                    "preferredValue": None,
                    "resolutionStatus": "unresolved",
                    "boxSource": [819, 212, 1730, 309],
                },
                {
                    "elementId": 8,
                    "value": "Websites for startups.",
                    "preferredValue": (
                        "World-class branding and websites for startups."
                    ),
                    "resolutionStatus": "vlm-preferred-candidate",
                    "boxSource": [891, 294, 1654, 391],
                },
            ],
            "blockingUncertainties": [
                {"kind": "text_candidate", "elementId": 7},
                {"kind": "text_candidate", "elementId": 8},
            ],
        }
    }

    ops._partition_merged_preferred_text(document)

    [first_line, second_line] = document["reconstruction"]["text"]
    assert first_line["preferredValue"] == "World-class branding and"
    assert second_line["preferredValue"] == "websites for startups."
    assert first_line["resolutionStatus"] == "semantic-span-partition-candidate"
    assert second_line["resolutionStatus"] == "semantic-span-partition-candidate"
    assert first_line["resolutionMethod"] == "measured-line-semantic-partition"
    assert second_line["resolutionMethod"] == "measured-line-semantic-partition"


def test_discovered_ocr_typo_uses_exact_neighbor_to_partition_heading() -> None:
    document = {
        "reconstruction": {
            "text": [
                {
                    "elementId": 11,
                    "value": "The nterne",
                    "preferredValue": "The Internet Of Chains",
                    "resolutionStatus": "semantic-span-partition-candidate",
                    "boxSource": [57, 137, 540, 207],
                },
                {
                    "elementId": 1,
                    "value": "Of Chains",
                    "preferredValue": "Of Chains",
                    "resolutionStatus": "confirmed",
                    "boxSource": [57, 227, 461, 298],
                },
            ]
        }
    }

    ops._partition_merged_preferred_text(document)

    first_line, second_line = document["reconstruction"]["text"]
    assert first_line["preferredValue"] == "The Internet"
    assert second_line["preferredValue"] == "Of Chains"
    assert first_line["resolutionMethod"] == "measured-line-semantic-partition"
    assert second_line["resolutionMethod"] == "measured-line-semantic-partition"


def test_verified_ocr_case_is_not_lowercased_by_semantic_partition() -> None:
    document = {
        "reconstruction": {
            "text": [
                {
                    "elementId": 1,
                    "value": "Of Chains",
                    "preferredValue": "of Chains",
                    "verified": True,
                    "confidence": 0.99,
                    "method": "rapidocr-multiscale-consensus",
                    "resolutionStatus": "semantic-span-partition-candidate",
                }
            ]
        }
    }

    ops._preserve_verified_ocr_orthography(document)

    [text] = document["reconstruction"]["text"]
    assert text["preferredValue"] == "Of Chains"
    assert text["resolutionStatus"] == "confirmed"
    assert text["epistemic"] == "measured"


def test_verified_ocr_copy_and_url_order_outrank_lossy_vlm_rewrites() -> None:
    document = {
        "reconstruction": {
            "text": [
                {
                    "elementId": 28,
                    "value": "Branding",
                    "preferredValue": "reading",
                    "verified": True,
                    "confidence": 1.0,
                    "method": "rapidocr-dual-script-consensus",
                    "resolutionStatus": "vlm-preferred-candidate",
                },
                {
                    "elementId": 40,
                    "value": "partners.dub.com/programs/acme-projects",
                    "preferredValue": "/programs/acme-projects partners.dub.com/",
                    "verified": True,
                    "confidence": 1.0,
                    "method": "rapidocr-dual-script-consensus",
                    "resolutionStatus": "regional-fragment-consensus",
                },
            ]
        }
    }

    ops._preserve_verified_ocr_orthography(document)

    branding, url = document["reconstruction"]["text"]
    assert branding["preferredValue"] == "Branding"
    assert url["preferredValue"] == "partners.dub.com/programs/acme-projects"
    assert branding["resolutionMethod"] == "verified-ocr-authoritative-preservation"
    assert url["resolutionMethod"] == "verified-ocr-authoritative-preservation"


def test_verified_ocr_keeps_semantic_word_boundaries_when_glyph_order_matches() -> None:
    document = {
        "reconstruction": {
            "text": [
                {
                    "elementId": 10,
                    "value": "Yournew",
                    "preferredValue": "Your new",
                    "verified": True,
                    "confidence": 0.999,
                    "method": "rapidocr-dual-script-latin-preferred",
                    "resolutionStatus": "vlm-preferred-candidate",
                },
                {
                    "elementId": 17,
                    "value": "withAIDLP",
                    "preferredValue": "with AI DLP",
                    "verified": True,
                    "confidence": 0.99,
                    "method": "rapidocr-dual-script-latin-preferred",
                    "resolutionStatus": "vlm-preferred-candidate",
                },
            ]
        }
    }

    ops._preserve_verified_ocr_orthography(document)

    your_new, with_ai = document["reconstruction"]["text"]
    assert your_new["preferredValue"] == "Your new"
    assert with_ai["preferredValue"] == "with AI DLP"
    assert your_new["resolutionMethod"] == "verified-ocr-semantic-spacing-preservation"
    assert with_ai["resolutionMethod"] == "verified-ocr-semantic-spacing-preservation"


def test_verified_ocr_rejects_implausible_single_letter_intra_word_gaps() -> None:
    document = {
        "reconstruction": {
            "text": [
                {
                    "value": "Wearable",
                    "preferredValue": "Wearabl e",
                    "verified": True,
                    "confidence": 1.0,
                    "method": "rapidocr-dual-script-consensus",
                },
                {
                    "value": "SERVICES",
                    "preferredValue": "S ERVICES",
                    "verified": True,
                    "confidence": 1.0,
                    "method": "rapidocr-dual-script-consensus",
                },
            ]
        }
    }

    ops._preserve_verified_ocr_orthography(document)

    wearable, services = document["reconstruction"]["text"]
    assert wearable["preferredValue"] == "Wearable"
    assert services["preferredValue"] == "SERVICES"
    assert all(
        entry["resolutionMethod"] == "verified-ocr-authoritative-preservation"
        for entry in (wearable, services)
    )


def test_large_collapsed_line_is_split_from_measured_gaps_and_bounded_ocr(
    monkeypatch,
) -> None:
    image = np.zeros((80, 300, 3), np.uint8)
    image[20:60, 10:120] = 255
    image[20:60, 175:285] = 255
    readings = iter(
        [
            [{"text": "Your", "confidence": 0.999}],
            [{"text": "new", "confidence": 0.999}],
        ]
    )
    monkeypatch.setattr(ops, "run_latin_ocr_image", lambda _crop: next(readings))
    entry = {
        "value": "Yournew",
        "boxSource": [0, 0, 300, 80],
        "color": "#FFFFFF",
    }
    metrics = {"capHeight": 40, "color": "#FFFFFF"}

    restored = ops._restore_word_spaces_from_glyph_gaps(image, entry, metrics)

    assert restored == "Your new"


def test_interface_phrase_arbiter_uses_exact_high_confidence_latin_alternative() -> None:
    document = {
        "reconstruction": {
            "targetKind": "web",
            "canvas": {"width": 1440, "height": 900},
                "visualControlCandidates": [
                    {"elementId": 40, "labelElementIds": [8, 30]}
                ],
            "text": [
                {
                    "elementId": 8,
                    "value": "EOOK A DEMO",
                    "alternatives": [
                        {"text": "EOOK A DEMO", "confidence": 0.973},
                        {"text": "BOOKA DEMO", "confidence": 0.991},
                    ],
                    "boxSource": [1218, 21, 1304, 31],
                },
                {
                    "elementId": 10,
                    "value": "Yournew",
                    "boxSource": [60, 186, 465, 272],
                    "fontFeatures": {"fontSize": 103},
                },
                {
                    "elementId": 22,
                    "value": "See how in 14Os",
                    "alternatives": [
                        {"text": "See how in 140s", "confidence": 0.974}
                    ],
                    "boxSource": [68, 610, 281, 640],
                    "fontFeatures": {"fontSize": 41},
                },
                {
                    "elementId": 30,
                    "value": "START NOW 7",
                    "verified": True,
                    "confidence": 0.946,
                    "boxSource": [1128, 428, 1253, 442],
                },
            ],
        }
    }

    ops._repair_interface_ocr_phrases(document)
    ops._preserve_verified_ocr_orthography(document)

    book, your_new, see_how, start_now = document["reconstruction"]["text"]
    assert book["preferredValue"] == "BOOK A DEMO"
    assert your_new["preferredValue"] == "Your new"
    assert see_how["preferredValue"] == "See how in 140s"
    assert start_now["preferredValue"] == "START NOW ↗"
    assert all(
        entry["resolutionMethod"] == "bounded-interface-phrase-arbitration"
        for entry in (book, your_new, see_how, start_now)
    )


def test_indexed_product_control_labels_recover_space_and_superscript() -> None:
    document = {
        "reconstruction": {
            "targetKind": "web",
            "visualControlCandidates": [
                {"elementId": 14, "labelElementIds": [11]},
                {"elementId": 15, "labelElementIds": [12]},
                {"elementId": 16, "labelElementIds": [13]},
            ],
            "text": [
                {
                    "elementId": 11,
                    "value": "ASense",
                    "verified": True,
                    "alternatives": [
                        {"text": "A1Sense", "confidence": 0.916}
                    ],
                },
                {
                    "elementId": 12,
                    "value": "B'Eye",
                    "verified": True,
                    "alternatives": [
                        {"text": "B1 Eye", "confidence": 0.808}
                    ],
                },
                {
                    "elementId": 13,
                    "value": "A'Neuro",
                    "verified": True,
                    "alternatives": [
                        {"text": "ANeuro", "confidence": 0.998}
                    ],
                },
            ],
        }
    }

    ops._repair_indexed_control_labels(document)
    ops._preserve_verified_ocr_orthography(document)

    labels = document["reconstruction"]["text"]
    assert [entry["preferredValue"] for entry in labels] == [
        "A1 Sense",
        "B1 Eye",
        "A1 Neuro",
    ]
    assert all(entry["indexedLabel"]["superscript"] for entry in labels)


def test_bottom_corner_glyph_becomes_a_semantic_scroll_control() -> None:
    document = {
        "tokens": {
            "color": {"background": {"$type": "color", "$value": "#FFFFFF"}}
        },
        "reconstruction": {
            "targetKind": "web",
            "canvas": {"width": 2557, "height": 1273},
            "icons": [
                {
                    "elementId": 18,
                    "name": "cross",
                    "boxSource": [129, 1194, 135, 1204],
                    "color": "#3E3E3E",
                }
            ],
            "visualControlCandidates": [],
        },
    }

    ops._infer_corner_navigation_controls(document)

    [icon] = document["reconstruction"]["icons"]
    [control] = document["reconstruction"]["visualControlCandidates"]
    assert icon["name"] == "arrow_down"
    assert control["boxSource"] == [120, 1175, 144, 1223]
    assert control["interaction"] == "semantic-button"
    assert control["ariaLabel"] == "Scroll down"
    assert control["zIndex"] == 24


def test_editorial_mixed_display_uses_one_light_serif_family_across_runs() -> None:
    document = {
        "reconstruction": {
            "targetKind": "web",
            "text": [
                {
                    "value": "with AI DLP",
                    "preferredValue": "with AI DLP",
                    "fontFeatures": {
                        "fontSize": 126,
                        "capHeight": 92,
                        "renderFamilyCandidate": "inter-tight",
                        "renderFamilyConfidence": 0.51,
                        "wordBoxesSource": [
                            {
                                "text": "with",
                                "box": [60, 400, 269, 492],
                                "slant": "italic",
                                "slantConfidence": 0.95,
                                "renderFamilyCandidate": "newsreader",
                                "renderFamilyConfidence": 0.52,
                                "renderFamilyScores": {
                                    "inter-tight": 0.85,
                                    "newsreader": 0.83,
                                },
                            },
                            {"text": "AI", "box": [269, 400, 374, 492]},
                            {"text": "DLP", "box": [374, 400, 531, 492]},
                        ],
                    },
                    "inlineRuns": [
                        {
                            "text": "with ",
                            "typographyCandidate": {
                                "slant": "italic",
                                "confidence": 0.95,
                            },
                            "runIndex": 0,
                        },
                        {
                            "text": "AI ",
                            "typographyCandidate": {
                                "slant": "normal",
                                "confidence": 0.87,
                            },
                            "runIndex": 1,
                        },
                        {
                            "text": "DLP",
                            "typographyCandidate": {
                                "slant": "normal",
                                "confidence": 0.71,
                            },
                            "runIndex": 2,
                        },
                    ],
                }
            ],
        }
    }

    ops._resolve_editorial_mixed_display_fonts(document)

    [entry] = document["reconstruction"]["text"]
    assert all(
        word["renderFamily"] == "newsreader"
        and word["renderWeight"] == 300
        for word in entry["fontFeatures"]["wordBoxesSource"]
    )
    assert all(
        run["typographyCandidate"]["class"] == "serif"
        and run["typographyCandidate"]["weight"] == "light"
        for run in entry["inlineRuns"]
    )


def test_control_label_box_is_tightened_without_absorbing_brand_icon(
    monkeypatch,
) -> None:
    image = np.full((100, 300, 3), 40, np.uint8)
    document = {
        "reconstruction": {
            "targetKind": "web",
            "canvas": {"width": 300, "height": 100},
            "text": [
                {
                    "elementId": 23,
                    "value": "Try now with Microsoft",
                    "preferredValue": "Try now with Microsoft",
                    "boxSource": [92, 42, 235, 62],
                    "fontFeatures": {"capHeight": 20},
                }
            ],
            "visualControlCandidates": [
                {
                    "elementId": 39,
                    "kind": "button",
                    "boxSource": [60, 20, 260, 80],
                    "labelElementIds": [23],
                }
            ],
        }
    }
    monkeypatch.setattr(ops, "load_cv", lambda _path: image)
    monkeypatch.setattr(
        ops,
        "run_latin_ocr_image",
        lambda _image: [
            {
                "text": "Try now with Microsoft",
                "box": [55, 22, 175, 44],
                "confidence": 0.999,
            }
        ],
    )

    ops._tighten_control_label_boxes(document, "fixture.png")
    ops._tighten_control_label_boxes(document, "fixture.png")

    [label] = document["reconstruction"]["text"]
    assert label["boxSource"] == [115, 42, 235, 64]
    assert label["geometrySource"] == "bounded-latin-control-label"
    assert label["confirmedBy"] == ["bounded-latin-control-label"]


def test_navigation_prefix_split_requires_a_measured_navigation_rail(
    tmp_path, monkeypatch
) -> None:
    image_path = tmp_path / "hero.png"
    cv2.imwrite(str(image_path), np.full((900, 1440, 3), 20, np.uint8))
    document = {
        "reconstruction": {
            "targetKind": "web",
            "canvas": {"width": 1440, "height": 900},
            "surfaces": [],
            "icons": [],
            "text": [
                {
                    "elementId": 22,
                    "value": "See how in 140s",
                    "preferredValue": "See how in 140s",
                    "boxSource": [72, 613, 280, 637],
                    "fontFeatures": {"fontSize": 26, "color": "#FFFFFF"},
                }
            ],
        }
    }
    monkeypatch.setattr(
        ops,
        "run_latin_ocr_image",
        lambda _image: (_ for _ in ()).throw(AssertionError("must not run")),
    )

    ops._separate_navigation_prefix_icons(document, str(image_path))

    [entry] = document["reconstruction"]["text"]
    assert entry["value"] == "See how in 140s"
    assert entry["boxSource"] == [72, 613, 280, 637]
    assert document["reconstruction"]["icons"] == []


def test_header_word_gaps_become_separate_semantic_navigation_links() -> None:
    document = {
        "reconstruction": {
            "targetKind": "web",
            "canvas": {"width": 1440, "height": 900},
            "text": [
                {
                    "elementId": 1,
                    "value": "Rollups Metalayer",
                    "preferredValue": "Rollups Metalayer",
                    "resolutionStatus": "regional-fragment-consensus",
                    "boxSource": [285, 49, 436, 65],
                    "fontFeatures": {
                        "fontSize": 15,
                        "wordBoxesSource": [
                            {"text": "Rollups", "box": [285, 49, 337, 65]},
                            {"text": "Metalayer", "box": [363, 49, 436, 65]},
                        ],
                    },
                },
                {
                    "elementId": 2,
                    "value": "Blog",
                    "preferredValue": "Blog",
                    "resolutionStatus": "regional-new-text-candidate",
                    "boxSource": [462, 49, 492, 65],
                    "fontFeatures": {"fontSize": 15},
                },
                {
                    "elementId": 3,
                    "value": "Docs",
                    "preferredValue": "Docs",
                    "resolutionStatus": "regional-new-text-candidate",
                    "boxSource": [581, 50, 617, 61],
                    "fontFeatures": {"fontSize": 15},
                },
            ],
            "layoutRegions": [],
            "visualControlCandidates": [],
        }
    }

    ops._partition_navigation_word_gaps(document)
    ops._infer_top_navigation_controls(document)

    spec = document["reconstruction"]
    assert [entry["value"] for entry in spec["text"]] == [
        "Rollups",
        "Metalayer",
        "Blog",
        "Docs",
    ]
    labels_by_id = {
        entry["elementId"]: entry["value"] for entry in spec["text"]
    }
    assert {
        labels_by_id[control["labelElementIds"][0]]
        for control in spec["visualControlCandidates"]
    } == {"Rollups", "Metalayer", "Blog", "Docs"}
    assert all(
        control["interaction"] == "semantic-link"
        for control in spec["visualControlCandidates"]
    )


def test_confirmed_multiscale_text_is_not_truncated_by_semantic_candidate() -> None:
    document = {
        "reconstruction": {
            "text": [
                {
                    "elementId": 7,
                    "value": "World-class branding and",
                    "preferredValue": "World-class",
                    "status": "confirmed",
                    "confidence": 1.0,
                    "alternatives": [
                        {
                            "text": "World-class branding and",
                            "confidence": 1.0,
                            "scale": 1.0,
                        },
                        {
                            "text": "World-class branding and",
                            "confidence": 1.0,
                            "scale": 1.5,
                        },
                    ],
                    "boxSource": [819, 212, 1730, 309],
                }
            ]
        }
    }

    ops._partition_merged_preferred_text(document)

    [heading] = document["reconstruction"]["text"]
    assert heading["preferredValue"] == "World-class branding and"
    assert heading["resolutionStatus"] == "measured-multiscale-confirmed"
    assert heading["resolutionMethod"] == "preserve-confirmed-measured-text"
    assert heading["epistemic"] == "measured"


def test_large_text_box_refinement_excludes_small_neighbor_labels() -> None:
    image = np.full((220, 900, 3), (255, 255, 255), np.uint8)
    cv2.putText(
        image,
        "SUMMER",
        (45, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.8,
        (255, 110, 0),
        12,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "DATE",
        (70, 178),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 110, 0),
        2,
        cv2.LINE_AA,
    )
    entry = {
        "value": "SUMMER",
        "preferredValue": "SUMMER",
        "boxSource": [0, 0, 650, 190],
        "color": "#006EFF",
        "fontFeatures": {"fontSize": 120},
    }

    refined = ops._refine_large_text_box(image, entry)

    assert refined is not None
    assert refined[0] >= 40
    assert refined[1] >= 55
    assert refined[2] < 600
    assert refined[3] < 150


def test_large_text_box_refinement_preserves_already_tight_glyph_box() -> None:
    image = np.full((180, 700, 3), (255, 255, 255), np.uint8)
    cv2.putText(
        image,
        "SUMMER",
        (45, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.8,
        (255, 110, 0),
        12,
        cv2.LINE_AA,
    )
    entry = {
        "value": "SUMMER",
        "preferredValue": "SUMMER",
        "boxSource": [40, 50, 400, 135],
        "color": "#006EFF",
        "fontFeatures": {"fontSize": 120},
    }

    assert ops._refine_large_text_box(image, entry) is None


def test_downscaled_display_box_excludes_attached_edge_artwork_when_ocr_is_tight() -> None:
    image = np.full((300, 800, 3), (235, 235, 235), np.uint8)
    for left in (160, 260, 360, 460, 560):
        cv2.rectangle(image, (left, 60), (left + 59, 239), (0, 0, 0), -1)
    cv2.rectangle(image, (90, 20), (170, 164), (0, 0, 0), -1)
    cv2.rectangle(image, (610, 135), (690, 280), (0, 0, 0), -1)
    entry = {
        "value": "SLUSH",
        "preferredValue": "SLUSH",
        "boxSource": [100, 50, 680, 240],
        "color": "#000000",
        "method": "rapidocr-downscaled-display-latin-preferred",
        "fontFeatures": {"fontSize": 240},
    }

    refined = ops._refine_large_text_box(image, entry)

    assert refined is not None
    assert 155 <= refined[0] <= 165
    assert 615 <= refined[2] <= 625
    assert refined[1:4:2] == [59, 241]


def test_small_text_box_separates_unrecognized_leading_brand_mark() -> None:
    image = np.full((44, 190, 3), (226, 226, 223), np.uint8)
    cv2.fillConvexPoly(
        image,
        np.asarray([[4, 25], [13, 10], [31, 10], [40, 25]], dtype=np.int32),
        (0, 80, 252),
    )
    cv2.putText(
        image,
        "caldera",
        (51, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (0, 80, 252),
        2,
        cv2.LINE_AA,
    )
    entry = {
        "value": "caldera",
        "preferredValue": "caldera",
        "boxSource": [4, 7, 184, 32],
        "geometrySource": "bounded-regional-measurement",
    }
    metrics = {
        "fontSize": 25,
        "capHeight": 18,
        "measuredCharacterCount": 8,
        "inkBox": [4, 7, 184, 32],
    }

    result = ops._refine_small_text_ink_box(image, entry, metrics)

    assert result is not None
    refined, method = result
    assert 42 <= refined[0] <= 58
    assert refined[1:] == [7, 184, 32]
    assert method.endswith("leading-icon-separation")


def test_measured_glyph_match_selects_local_serif_and_sans_renderers() -> None:
    reference = (
        Path(__file__).resolve().parents[2]
        / "qa"
        / "fixtures"
        / "reconstruction-matrix"
        / "references"
        / "summer-drive.png"
    )
    document = {
        "header": {"size": [2557, 1273]},
        "reconstruction": {
            "targetKind": "web",
            "text": [
                {
                    "elementId": 1,
                    "value": "5-10 PM",
                    "preferredValue": "5-10 PM",
                    "boxSource": [1993, 337, 2458, 494],
                    "color": "#006EFF",
                },
                {
                    "elementId": 2,
                    "value": "TICKETS",
                    "preferredValue": "TICKETS",
                    "boxSource": [326, 1139, 504, 1172],
                    "color": "#006EFF",
                },
            ],
        },
    }

    ops._hydrate_measured_typography(document, str(reference))

    time_text, ticket_text = document["reconstruction"]["text"]
    assert time_text["fontFeatures"]["renderFamily"] == "newsreader"
    assert ticket_text["fontFeatures"]["renderFamily"] == "inter-tight"
    assert (
        time_text["fontFeatures"]["renderFamilyMethod"]
        == "bundled-glyph-raster-chamfer-iou"
    )


def test_symmetric_numeric_typography_uses_the_joint_better_renderer() -> None:
    document = {
        "reconstruction": {
            "text": [
                {
                    "elementId": 5,
                    "value": "06.24.21",
                    "boxSource": [91, 285, 416, 377],
                    "color": "#006EFF",
                    "fontFeatures": {
                        "capHeight": 115,
                        "color": "#006EFF",
                        "renderFamily": "inter-tight",
                        "renderFamilyCandidate": "inter-tight",
                        "renderFamilyConfidence": 0.82,
                        "renderWeight": 300,
                        "renderFamilyScores": {
                            "inter-tight": 0.6578,
                            "newsreader": 0.8095,
                        },
                    },
                },
                {
                    "elementId": 6,
                    "value": "5-10 PM",
                    "boxSource": [2015, 285, 2440, 377],
                    "color": "#006EFF",
                    "fontFeatures": {
                        "capHeight": 114,
                        "color": "#006EFF",
                        "renderFamily": "newsreader",
                        "renderFamilyCandidate": "newsreader",
                        "renderFamilyConfidence": 0.95,
                        "renderWeight": 300,
                        "renderFamilyScores": {
                            "inter-tight": 0.8684,
                            "newsreader": 0.6477,
                        },
                    },
                },
            ]
        }
    }

    ops._resolve_symmetric_render_fonts(document)

    families = [
        entry["fontFeatures"]["renderFamily"]
        for entry in document["reconstruction"]["text"]
    ]
    assert families == ["newsreader", "newsreader"]
    assert all(
        entry["fontFeatures"]["renderFamilyMethod"]
        == "symmetric-numeric-render-consensus"
        for entry in document["reconstruction"]["text"]
    )


def test_weak_bundled_font_margin_stays_unresolved_for_sans_fallback() -> None:
    reference = (
        Path(__file__).resolve().parents[2]
        / "qa"
        / "fixtures"
        / "reconstruction-matrix"
        / "references"
        / "summer-drive.png"
    )
    document = {
        "header": {"size": [2557, 1273]},
        "reconstruction": {
            "targetKind": "web",
            "text": [
                {
                    "elementId": 7,
                    "value": (
                        "A NO-WORK, WORK-EVENT FOR TEAMS IN THE DRIVE "
                        "CAPITAL PORTFOLIO"
                    ),
                    "boxSource": [275, 604, 2282, 655],
                    "color": "#006EFF",
                }
            ],
        },
    }

    ops._hydrate_measured_typography(document, str(reference))

    [tagline] = document["reconstruction"]["text"]
    assert tagline["fontFeatures"]["renderFamilyCandidate"] == "newsreader"
    assert "renderFamily" not in tagline["fontFeatures"]
    assert tagline["fontFeatures"]["renderFamilyMethod"].endswith("uncertain")
    assert "inlineRuns" not in tagline


def test_tight_text_box_repairs_background_reported_as_glyph_color(
    monkeypatch, tmp_path
) -> None:
    background_bgr = (20, 55, 134)  # #863714
    ink_bgr = (51, 174, 250)  # #FAAE33
    image = np.full((60, 280, 3), background_bgr, np.uint8)
    cv2.putText(
        image,
        "FIRE ROASTED INDIAN SAUCE",
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        ink_bgr,
        1,
        cv2.LINE_AA,
    )
    image_path = tmp_path / "tight-text.png"
    cv2.imwrite(str(image_path), image)
    metrics = {
        "fontSize": 12,
        "capHeight": 9,
        "color": "#863714",
        "colorSource": "measured-glyph-pixels",
    }
    monkeypatch.setattr(ops, "_glyph_metrics", lambda *_args, **_kwargs: dict(metrics))
    monkeypatch.setattr(
        ops, "_refine_small_text_ink_box", lambda *_args, **_kwargs: None
    )
    document = {
        "header": {"size": [280, 60]},
        "reconstruction": {
            "targetKind": "web",
            "text": [
                {
                    "elementId": 8,
                    "value": "FIRE ROASTED INDIAN SAUCE",
                    "boxSource": [10, 20, 218, 34],
                }
            ],
        },
    }

    ops._hydrate_measured_typography(document, str(image_path))

    [entry] = document["reconstruction"]["text"]
    measured = np.asarray(ops._hex_to_bgr(entry["color"]), dtype=np.float32)
    expected = np.asarray(ink_bgr, dtype=np.float32)
    assert np.linalg.norm(measured - expected) < 65
    assert entry["colorSource"] == "measured-external-ring-contrast-repair"


def test_semantic_candidate_cannot_truncate_measured_large_glyph_run() -> None:
    entry = {
        "value": "06.A2",
        "fontFeatures": {
            "fontSize": 158,
            "characterCount": 5,
            "measuredCharacterCount": 8,
        },
    }

    assert ops._preferred_matches_measured_glyph_count(entry, "06.24.21") is True
    assert ops._preferred_matches_measured_glyph_count(entry, "06.24") is False
    assert ops._preferred_matches_measured_glyph_count(entry, "06") is False


def test_unresolved_control_labels_share_one_bounded_focus_crop() -> None:
    spec = {
        "targetKind": "web",
        "canvas": {"width": 2555, "height": 1273},
        "text": [
            {
                "elementId": 1,
                "value": "START NOW Z",
                "preferredValue": None,
                "resolutionStatus": "unresolved",
                "boxSource": [1122, 422, 1259, 448],
            },
            {
                "elementId": 2,
                "value": "VIEV WORK U",
                "preferredValue": None,
                "resolutionStatus": "unresolved",
                "boxSource": [1298, 423, 1424, 445],
            },
            {
                "elementId": 3,
                "value": "STATUS",
                "preferredValue": "STATUS",
                "resolutionStatus": "confirmed",
                "boxSource": [1145, 168, 1423, 190],
            },
        ],
        "visualControlCandidates": [
            {"boxSource": [1110, 410, 1272, 460], "labelElementIds": [1]},
            {"boxSource": [1282, 410, 1440, 461], "labelElementIds": [2]},
            {"boxSource": [1119, 161, 1432, 196], "labelElementIds": [3]},
        ],
    }

    actions = ops._control_text_focus_actions(spec, 2)

    assert len(actions) == 1
    [action] = actions
    assert action["evidence"] == "START NOW Z | VIEV WORK U"
    assert action["arguments"]["region"] == {
        "x": 1084,
        "y": 397,
        "width": 382,
        "height": 77,
    }


def test_regional_ocr_fragments_rejoin_one_measured_control_label() -> None:
    document = {
        "reconstruction": {
            "text": [
                {
                    "elementId": 9,
                    "value": "START NOW Z",
                    "preferredValue": None,
                    "resolutionStatus": "unresolved",
                    "boxSource": [1122, 422, 1259, 448],
                }
            ],
            "focusPlan": [],
            "textVerificationPlan": [],
            "blockingUncertainties": [],
        }
    }
    action = {
        "reason": "Resolve exact live labels inside measured semantic controls.",
        "evidence": "START NOW Z",
        "arguments": {
            "region": {"x": 1084, "y": 397, "width": 382, "height": 77}
        },
    }
    regional_document = {
        "reconstruction": {
            "text": [
                {
                    "elementId": 1,
                    "value": "START",
                    "preferredValue": "START",
                    "resolutionStatus": "confirmed",
                    "boxSource": [1125, 424, 1190, 445],
                },
                {
                    "elementId": 2,
                    "value": "NOWZ",
                    "preferredValue": "NOW",
                    "resolutionStatus": "vlm-preferred-candidate",
                    "boxSource": [1179, 424, 1258, 446],
                },
            ],
            "semanticTextCandidate": {
                "text": "START NOW ↑ VIEW WORK ↓",
            },
        }
    }

    ops._merge_focus_documents(document, [(action, regional_document)])

    [label] = document["reconstruction"]["text"]
    assert label["preferredValue"] == "START NOW"
    assert label["resolutionStatus"] == "regional-fragment-consensus"
    assert label["resolutionMethod"] == "regional-ocr-fragment-join"


def test_tiny_unresolved_glyph_noise_becomes_an_icon_not_a_focus_call() -> None:
    document = {
        **_stub_document("reconstruct"),
        "reconstruction": {
            "targetKind": "web",
            "canvas": {"width": 1092, "height": 553},
            "text": [
                {
                    "elementId": 64,
                    "value": ".",
                    "preferredValue": None,
                    "status": "candidate",
                    "resolutionStatus": "unresolved",
                    "confidence": 0.50,
                    "boxSource": [763, 181, 782, 198],
                },
                {
                    "elementId": 65,
                    "value": "д",
                    "preferredValue": None,
                    "status": "candidate",
                    "resolutionStatus": "unresolved",
                    "confidence": 0.69,
                    "boxSource": [764, 262, 784, 282],
                },
                {
                    "elementId": 66,
                    "value": "2",
                    "preferredValue": None,
                    "status": "candidate",
                    "resolutionStatus": "unresolved",
                    "confidence": 0.62,
                    "boxSource": [1022, 178, 1042, 199],
                },
                {
                    "elementId": 67,
                    "value": "Tasks",
                    "preferredValue": "Tasks",
                    "status": "confirmed",
                    "resolutionStatus": "confirmed",
                    "confidence": 0.99,
                    "boxSource": [756, 143, 791, 159],
                },
            ],
            "icons": [{"boxSource": [767, 187, 779, 196]}],
            "focusPlan": [],
            "textVerificationPlan": [],
            "blockingUncertainties": [
                {"elementId": 64},
                {"elementId": 65},
                {"elementId": 66},
            ],
        },
    }

    ops._exclude_tiny_glyph_noise(document)

    spec = document["reconstruction"]
    assert [entry["elementId"] for entry in spec["text"]] == [66, 67]
    assert {entry["elementId"] for entry in spec["excludedTextCandidates"]} == {64, 65}
    assert any(icon.get("elementId") == 64 for icon in spec["icons"])
    assert {entry["elementId"] for entry in spec["blockingUncertainties"]} == {66}
    assert [action["evidence"] for action in ops._fallback_text_focus_actions(spec, 3)] == ["2"]


def test_unverified_square_numeric_glyph_in_a_navigation_rail_is_an_icon() -> None:
    document = {
        "reconstruction": {
            "targetKind": "web",
            "canvas": {"width": 1092, "height": 553},
            "text": [
                {
                    "elementId": 66,
                    "value": "88",
                    "preferredValue": None,
                    "status": "candidate",
                    "resolutionStatus": "unresolved",
                    "confidence": 0.998,
                    "verified": False,
                    "boxSource": [9, 147, 44, 182],
                    "fontFeatures": {"inkCoverage": 0.93},
                },
                {
                    "elementId": 67,
                    "value": "Tasks",
                    "preferredValue": "Tasks",
                    "status": "confirmed",
                    "resolutionStatus": "confirmed",
                    "confidence": 0.99,
                    "verified": True,
                    "boxSource": [759, 146, 788, 156],
                },
            ],
            "icons": [],
            "focusPlan": [],
            "textVerificationPlan": [],
            "blockingUncertainties": [{"elementId": 66}],
        }
    }

    ops._exclude_tiny_glyph_noise(document)

    spec = document["reconstruction"]
    assert [entry["elementId"] for entry in spec["text"]] == [67]
    assert spec["excludedTextCandidates"][0]["reason"] == (
        "unverified-square-navigation-rail-glyph"
    )
    assert spec["icons"][0]["representation"] == "preserve-source-decoration"


def test_navigation_prefix_icon_is_split_before_bounded_latin_ocr(
    monkeypatch,
) -> None:
    image = np.full((80, 220, 3), 246, np.uint8)
    cv2.rectangle(image, (20, 30), (30, 41), (70, 70, 70), -1)
    for x0, x1 in ((42, 47), (50, 55), (58, 63), (66, 71), (74, 79), (82, 87)):
        cv2.rectangle(image, (x0, 31), (x1, 41), (70, 70, 70), -1)
    document = {
        "reconstruction": {
            "targetKind": "web",
            "canvas": {"width": 220, "height": 80},
            "text": [
                {
                    "elementId": 15,
                    "value": "8Groups",
                    "preferredValue": None,
                    "status": "candidate",
                    "resolutionStatus": "unresolved",
                    "confidence": 0.96,
                    "verified": True,
                    "boxSource": [20, 30, 88, 42],
                    "color": "#464646",
                }
            ],
            "icons": [],
        }
    }
    monkeypatch.setattr(ops, "load_cv", lambda _path: image)
    monkeypatch.setattr(
        ops,
        "run_latin_ocr_image",
        lambda _image: [
            {
                "text": "Groups",
                "confidence": 0.999,
                "box": [0, 0, 120, 40],
                "method": "rapidocr",
            }
        ],
        raising=False,
    )

    ops._separate_navigation_prefix_icons(document, "fixture.png")

    spec = document["reconstruction"]
    [label] = spec["text"]
    assert label["value"] == "Groups"
    assert label["preferredValue"] == "Groups"
    assert label["verified"] is True
    assert label["boxSource"][0] >= 40
    assert label["resolutionMethod"] == "bounded-latin-ocr-after-icon-prefix-split"
    [icon] = spec["icons"]
    assert icon["kind"] == "navigation-prefix-icon"
    assert icon["boxSource"][2] < label["boxSource"][0]


def test_measured_word_gap_restores_summer_title_without_splitting_tickets() -> None:
    reference = (
        Path(__file__).parents[2]
        / "qa"
        / "fixtures"
        / "reconstruction-matrix"
        / "references"
        / "summer-drive.png"
    )
    image = cv2.imread(str(reference))
    assert image is not None

    restored = ops._restore_word_spaces_from_glyph_gaps(
        image,
        {
            "value": "THESUMMER",
            "boxSource": [50, 58, 2505, 254],
            "color": "#006EFF",
        },
        {"capHeight": 192, "color": "#006EFF"},
    )
    unchanged_button = ops._restore_word_spaces_from_glyph_gaps(
        image,
        {
            "value": "TICKETS",
            "boxSource": [2097, 1139, 2275, 1172],
            "color": "#006EFF",
        },
        {"capHeight": 33, "color": "#006EFF"},
    )

    assert restored == "THE SUMMER"
    assert unchanged_button is None


def test_regional_single_glyph_discovery_is_reclassified_as_an_icon() -> None:
    document = {
        "reconstruction": {
            "targetKind": "web",
            "canvas": {"width": 1440, "height": 900},
            "text": [
                {
                    "elementId": 6,
                    "value": "x",
                    "preferredValue": "x",
                    "status": "candidate",
                    "resolutionStatus": "regional-new-text-candidate",
                    "confidence": 0.93,
                    "boxSource": [1118, 47, 1138, 65],
                },
                {
                    "elementId": 7,
                    "value": "Docs",
                    "preferredValue": "Docs",
                    "status": "stable-candidate",
                    "resolutionStatus": "regional-new-text-candidate",
                    "confidence": 0.97,
                    "boxSource": [581, 50, 617, 61],
                },
            ],
            "icons": [],
            "focusPlan": [],
            "textVerificationPlan": [],
            "blockingUncertainties": [],
        }
    }

    ops._exclude_tiny_glyph_noise(document)

    spec = document["reconstruction"]
    assert [entry["value"] for entry in spec["text"]] == ["Docs"]
    assert spec["excludedTextCandidates"][0]["elementId"] == 6
    assert spec["icons"][0]["elementId"] == 6


def test_bounded_dual_ocr_updates_matching_text_element_without_losing_id() -> None:
    dump = {
        "elements": [
            {
                "id": 7,
                "kind": "text",
                "text": "SERA",
                "box": [20, 20, 80, 40],
                "font": {"fontSize": 16},
            },
            {"id": 8, "kind": "icon", "box": [100, 20, 120, 40]},
        ]
    }
    image = np.full((80, 160, 3), 255, np.uint8)
    fused = [
        {
            "text": "$ERA",
            "box": [20, 20, 80, 40],
            "confidence": 1.0,
            "verified": True,
            "method": "rapidocr-dual-script-visible-sigil-consensus",
        }
    ]

    ops._sync_bounded_ocr_elements(dump, image, fused)

    text = next(element for element in dump["elements"] if element["kind"] == "text")
    assert text["id"] == 7
    assert text["text"] == "$ERA"
    assert text["verified"] is True
    assert any(element["id"] == 8 for element in dump["elements"])


def test_dashboard_currency_axis_is_resolved_from_aligned_ocr_anchors() -> None:
    document = {
        **_stub_document("reconstruct"),
        "reconstruction": {
            "targetKind": "web",
            "canvas": {"width": 1092, "height": 553},
            "text": [
                {"elementId": 1, "value": "Ssoo", "preferredValue": None, "status": "candidate", "resolutionStatus": "unresolved", "confidence": 0.82, "boxSource": [320, 240, 350, 256]},
                {"elementId": 2, "value": "Sоо", "preferredValue": None, "status": "candidate", "resolutionStatus": "unresolved", "confidence": 0.82, "boxSource": [321, 286, 350, 301]},
                {"elementId": 3, "value": "Sоо", "preferredValue": None, "status": "candidate", "resolutionStatus": "unresolved", "confidence": 0.81, "boxSource": [321, 331, 349, 347]},
                {"elementId": 4, "value": "S2оо", "preferredValue": None, "status": "candidate", "resolutionStatus": "unresolved", "confidence": 0.80, "boxSource": [321, 377, 349, 392]},
                {"elementId": 5, "value": "Sо", "preferredValue": None, "status": "candidate", "resolutionStatus": "unresolved", "confidence": 0.61, "boxSource": [332, 423, 349, 437]},
            ],
        },
    }

    ops._resolve_numeric_axis_labels(document)

    entries = document["reconstruction"]["text"]
    assert [entry["preferredValue"] for entry in entries] == [
        "$800",
        "$600",
        "$400",
        "$200",
        "$0",
    ]
    assert all(entry["resolutionStatus"] == "layout-sequence-inferred" for entry in entries)
    assert all(entry["epistemic"] == "inferred" for entry in entries)


def test_full_web_document_cache_skips_rebuilding_the_same_contract(
    monkeypatch, tmp_path
) -> None:
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"stable-reference")
    monkeypatch.setenv("SENS_CACHE_DIR", str(tmp_path / "cache"))
    build_calls = []
    monkeypatch.setattr(
        ops,
        "analyze",
        lambda *_args, **_kwargs: {
            "source": {"id": "sha256:fixture", "mediaType": "image"},
            "somPath": None,
            "design": {"facts": []},
            "warnings": [],
            "ocr": [],
            "elements": [],
        },
    )
    monkeypatch.setattr(ops, "_apply_reconstruction_ocr", lambda *_args: None)
    monkeypatch.setattr(
        ops,
        "_image_for",
        lambda *_args, **_kwargs: np.zeros((10, 10, 3), np.uint8),
    )

    def build_document(*_args, **_kwargs):
        build_calls.append(_kwargs)
        document = _stub_document("reconstruct")
        document["reconstruction"].update(
            {
                "targetKind": "web",
                "text": [],
                "icons": [],
                "focusPlan": [],
                "textVerificationPlan": [],
                "blockingUncertainties": [],
                "allowedRasterRegions": [],
                "rasterAssetRule": {},
                "workflow": {},
            }
        )
        return document

    monkeypatch.setattr(ops.docmod, "build_document", build_document)

    first = ops.see_document(
        str(reference), fast=True, profile="reconstruct", target_kind="web"
    )
    second = ops.see_document(
        str(reference),
        fast=True,
        profile="reconstruct",
        target_kind="web",
        intent="A differently worded request for the same exact web reconstruction.",
    )
    brief = ops.see_document(
        str(reference),
        fast=True,
        profile="reconstruct",
        target_kind="web",
        response="brief",
        asset_output_dir=str(tmp_path / "assets"),
    )

    assert len(build_calls) == 1
    assert first == second
    assert brief["compatibility"]["response"] == "brief"
    assert brief["brief"]["contract"]["path"] == brief["contractPath"]
    assert brief["brief"]["reviewArguments"]["contractPath"] == brief["contractPath"]
    assert Path(brief["brief"]["starterProject"]["entryPath"]).is_file()
    assert brief["brief"]["workflow"]["nextAction"] == (
        "copy-or-serve-starter-then-sens-review"
    )
    geometry_authority = brief["brief"]["geometryAuthority"]
    assert geometry_authority["textDomBox"] == "reconstruction.text[].boxSource"
    assert geometry_authority["textInkBox"] == (
        "reconstruction.text[].fontFeatures.inkBox"
    )
    assert geometry_authority["controls"] == (
        "reconstruction.visualControlCandidates[].boxSource"
    )
    assert "elements[].box_norm" in geometry_authority["nonAuthoritative"]
    assert "claims[].regionNorm" in geometry_authority["nonAuthoritative"]
    assert "never" in geometry_authority["rule"].lower()
    contract = json.loads(Path(brief["contractPath"]).read_text(encoding="utf-8"))
    assert contract["reconstruction"]["geometryAuthority"] == geometry_authority
    assert Path(brief["contractPath"]).is_file()


def test_web_brief_exposes_measured_display_glyph_geometry() -> None:
    document = {
        "reconstruction": {
            "targetKind": "web",
            "canvas": {"width": 1440, "height": 900},
            "text": [
                {
                    "elementId": 32,
                    "value": "SLUSH",
                    "preferredValue": "SLUSH",
                    "boxSource": [345, 184, 1094, 593],
                    "fontFeatures": {
                        "fontSize": 549,
                        "inkBox": [345, 185, 1094, 592],
                        "glyphBoxes": [
                            {"text": "S", "box": [345, 185, 497, 592]},
                            {"text": "L", "box": [506, 189, 619, 588]},
                        ],
                        "measuredCharacterCount": 5,
                        "inkCoverage": 0.6995,
                        "strokeWidthPx": 30.0,
                    },
                }
            ],
        }
    }

    brief = ops._implementation_brief(document, "contract.json")

    assert brief["geometryAuthority"]["textGlyphBoxes"] == (
        "reconstruction.text[].fontFeatures.glyphBoxes"
    )
    table = brief["text"]
    row = json.loads(table["rows"])
    values = dict(zip(table["columns"], row, strict=True))
    assert values["inkBox"] == [345, 185, 1094, 592]
    assert values["glyphBoxes(text,box)"][0] == [
        "S",
        [345, 185, 497, 592],
    ]
    assert values["measuredCharacterCount"] == 5
    assert values["inkCoverage"] == 0.6995
    assert values["strokeWidthPx"] == 30.0
    assert any(
        "glyphBoxes" in rule and "connected slab" in rule
        for rule in brief["implementationRules"]
    )


def test_verified_source_vectors_replace_custom_display_font_with_live_wordmark(
    monkeypatch, tmp_path,
) -> None:
    monkeypatch.setenv("SENS_CACHE_DIR", str(tmp_path / "cache"))
    text = {
        "elementId": 32,
        "value": "SLUSH",
        "preferredValue": None,
        "method": "rapidocr-downscaled-display-latin-preferred",
        "confidence": 0.935,
        "boxSource": [345, 184, 1116, 593],
        "fontFeatures": {"fontSize": 549, "capHeight": 401},
    }
    document = {
        "artifacts": [],
        "reconstruction": {
            "targetKind": "web",
            "canvas": {"width": 1440, "height": 900},
            "text": [text],
        },
    }
    boxes = [
        [349.125, 141.25, 498.875, 636.5],
        [506, 141, 620, 636],
        [629, 141, 776, 637],
        [785, 141, 933, 636],
        [942, 141, 1091, 636],
    ]
    assets = []
    for index, box in enumerate(boxes):
        source = tmp_path / f"letter-{index}.svg"
        source.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
            f'<path d="M{index} 0H10V10Z"/></svg>',
            encoding="utf-8",
        )
        content = source.read_bytes()
        assets.append(
            {
                "vectorIndex": index,
                "domIndex": 100 + index,
                "path": str(source),
                "sha256": hashlib.sha256(content).hexdigest(),
                "sizeBytes": len(content),
                "mediaType": "image/svg+xml",
                "box": box,
                "visible": True,
                "source": "observed",
                "method": "sanitized-live-dom-svg",
            }
        )

    ops._hydrate_source_vector_regions(document, assets)

    regions = document["reconstruction"]["sourceVectorRegions"]
    assert len(regions) == 5
    assert regions[0]["boxSource"] == [349.125, 141.25, 498.875, 636.5]
    assert text["visualRepresentation"] == (
        "source-vector-wordmark-with-selectable-live-label"
    )
    assert text["sourceVectorAssetIds"] == [
        region["elementId"] for region in regions
    ]
    assert all(region["wordmarkText"] == "SLUSH" for region in regions)
    assert all(region["ariaHidden"] is True for region in regions)


def test_dense_screenshot_auto_selects_reconstruction_when_client_omits_prompt(
    monkeypatch,
) -> None:
    captured = {}
    _install_stubs(monkeypatch, captured)
    monkeypatch.setattr(
        ops,
        "analyze",
        lambda *_args, **_kwargs: {
            "somPath": None,
            "design": {"facts": []},
            "warnings": [],
            "ocr": [{"text": str(index)} for index in range(4)],
            "elements": [{"id": index} for index in range(5)],
        },
    )
    monkeypatch.setattr(ops, "_apply_reconstruction_ocr", lambda *_args: None)

    result = ops.see_document("fixture.png", fast=True)

    assert captured["profile"] == "reconstruct"
    assert result["doc"]["profile"] == "reconstruct"


def test_full_response_preserves_the_legacy_projection(monkeypatch) -> None:
    captured = {}
    _install_stubs(monkeypatch, captured)

    result = ops.see_document(
        "fixture.png",
        fast=True,
        profile="analyze",
        response="full",
    )

    assert captured["profile"] == "analyze"
    assert result["document"] == "FULL MARKDOWN"
    assert result["legacy"]["design"] == {
        "issues": [{"kind": "alignment", "detail": "ok"}]
    }
    assert result["compatibility"]["legacyIncluded"] is True


def test_consumer_prompt_teaches_the_strict_reconstruction_loop() -> None:
    russian = ops.vision_prompt("ru")["prompt"]
    english = ops.vision_prompt("en")["prompt"]

    for prompt in (russian, english):
        assert "profile=reconstruct" in prompt
        assert "targetKind=web" in prompt
        assert "response=brief" in prompt
        assert "fit=strict" in prompt
        assert "canComplete=true" in prompt
        assert "sens_review" in prompt
        assert "webPass=true" in prompt
        assert "live" in prompt.lower()
        assert "raster" in prompt.lower()
        assert "serial" in prompt.lower()
        assert "similarityScore" in prompt
        assert "focusPlan" in prompt
        assert "preferredValue" in prompt
        assert "JSONL" in prompt
        assert "starterProject" in prompt
        assert "entryPath" in prompt
        assert "at most four" in prompt or "не больше четырёх" in prompt
        assert "pixel-scanning" in prompt or "пиксельного сканирования" in prompt


def test_element_exposes_reconstruction_role_instead_of_raw_kind_alone(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        ops,
        "analyze",
        lambda *_args, **_kwargs: {
            "image": {"width": 1000, "height": 500},
            "elements": [
                {
                    "id": 1,
                    "kind": "text",
                    "text": "TICKETS",
                    "box": [80, 420, 180, 450],
                },
                {
                    "id": 2,
                    "kind": "button",
                    "box": [40, 400, 220, 470],
                    "borderColor": "#0078FF",
                },
                {
                    "id": 3,
                    "kind": "image",
                    "box": [30, 390, 230, 480],
                },
            ],
        },
    )

    button = ops.element("fixture.png", 2)
    raster = ops.element("fixture.png", 3)

    assert button["rawKind"] == "button"
    assert button["reconstructionRole"] == "semantic-control-candidate"
    assert button["representationGuidance"] == (
        "For web output, use a semantic HTML control; do not flatten it into an image."
    )
    assert raster["rawKind"] == "image"
    assert raster["reconstructionRole"] == "raster-forbidden-overlaps-text"


def test_display_text_background_removal_preserves_artwork_between_glyphs(
    tmp_path,
) -> None:
    source = np.zeros((200, 420, 3), np.uint8)
    for column in range(source.shape[1]):
        source[:, column] = (180 + column % 40, 120, 35)
    cv2.putText(
        source,
        "SLUSH",
        (18, 148),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.6,
        (0, 0, 0),
        12,
        cv2.LINE_AA,
    )
    edge_artwork = np.zeros(source.shape[:2], np.uint8)
    cv2.rectangle(edge_artwork, (351, 80), (354, 150), 255, -1)
    source[edge_artwork > 0] = (0, 0, 0)
    internal_artwork = np.zeros(source.shape[:2], np.uint8)
    cv2.rectangle(internal_artwork, (12, 65), (15, 145), 255, -1)
    source[internal_artwork > 0] = (0, 0, 0)
    reference = tmp_path / "display-reference.png"
    assert cv2.imwrite(str(reference), source)
    document = {
        "source": {"id": "sha256:display"},
        "artifacts": [],
        "reconstruction": {
            "targetKind": "web",
            "displayTextDiscovery": {
                "status": "complete",
                "scale": 0.5,
                "candidateCount": 1,
                "method": "rapidocr-downscaled-display-scan",
            },
            "text": [
                {
                    "elementId": 9,
                    "value": "SLUSH",
                    "boxSource": [12, 62, 350, 160],
                    "color": "#000000",
                    "method": "rapidocr-downscaled-display-latin-preferred",
                    "fontFeatures": {"fontSize": 132, "capHeight": 98},
                }
            ],
            "visualControlCandidates": [],
            "surfaces": [],
            "decorativeShapes": [],
            "icons": [],
            "badges": [],
            "symbolArt": [],
            "structuralLines": [],
            "vectorPaths": [],
            "allowedRasterRegions": [
                {
                    "elementId": "background-artwork",
                    "kind": "alpha-masked-background-artwork",
                    "boxSource": [0, 0, 420, 200],
                }
            ],
            "rasterAssetRule": {},
        },
    }

    ops._materialize_raster_assets(
        document,
        str(reference),
        str(tmp_path / "display-assets"),
        no_store=False,
    )

    layer = document["reconstruction"]["allowedRasterRegions"][0]
    output = cv2.imread(layer["assetPath"], cv2.IMREAD_COLOR)
    original_glyphs = cv2.inRange(source, (0, 0, 0), (20, 20, 20)) > 0
    protected = cv2.dilate(
        original_glyphs.astype(np.uint8), np.ones((7, 7), np.uint8)
    ).astype(bool)
    artwork = ~protected
    unchanged = np.linalg.norm(
        output.astype(np.int16) - source.astype(np.int16), axis=2
    ) <= 3

    assert unchanged[artwork].mean() > 0.98
    assert unchanged[edge_artwork > 0].mean() > 0.95
    assert unchanged[internal_artwork > 0].mean() > 0.95
    assert np.count_nonzero(
        np.linalg.norm(output.astype(np.int16), axis=2)[original_glyphs] < 20
    ) < np.count_nonzero(original_glyphs) * 0.15
    assert document["reconstruction"]["text"][0]["backgroundRemovalMode"] == (
        "measured-display-glyph-mask-inpaint"
    )
    assert layer["semanticResidualProtection"]["displayGlyphMaskElementIds"] == [9]
