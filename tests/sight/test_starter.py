from pathlib import Path

from sight.starter import (
    _font_family,
    _font_style,
    _font_weight,
    _icon_svg_markup,
    materialize_starter_project,
)


def _document(asset_path: Path) -> dict:
    return {
        "source": {"id": "sha256:fixture"},
        "tokens": {
            "color": {"background": {"$type": "color", "$value": "#FCF7EF"}}
        },
        "artifacts": [],
        "reconstruction": {
            "targetKind": "web",
            "canvas": {"width": 800, "height": 500},
            "surfaces": [
                {
                    "boxSource": [10, 10, 790, 490],
                    "background": "#FFFFFF",
                    "borderColor": "#DDDDDD",
                    "borderWidth": 1,
                    "cornerRadius": 12,
                }
            ],
            "decorativeShapes": [],
            "structuralLines": [
                {"boxSource": [20, 200, 780, 202], "color": "#006EFF"},
                {
                    "boxSource": [300, 300, 600, 301],
                    "color": "#E2E2E2",
                    "lineStyle": "dashed",
                    "dashLength": 3,
                    "dashGap": 5,
                },
            ],
            "vectorPaths": [
                {
                    "boxSource": [300, 250, 600, 350],
                    "pointsSource": [[300, 330], [420, 260], [600, 340]],
                    "strokeColor": "#A56EF0",
                    "strokeWidth": 2,
                }
            ],
            "text": [
                {
                    "elementId": 1,
                    "value": "THE SUMMER",
                    "preferredValue": None,
                    "boxSource": [20, 20, 400, 90],
                    "fontFeatures": {
                        "fontSize": 64,
                        "capHeight": 47,
                        "color": "#006EFF",
                    },
                    "typographyCandidate": {
                        "class": "display-sans",
                        "weight": "bold",
                    },
                    "color": "#006EFF",
                },
                {
                    "elementId": 2,
                    "value": "TICKETS",
                    "boxSource": [80, 420, 180, 450],
                    "fontFeatures": {"fontSize": 24, "color": "#006EFF"},
                    "color": "#006EFF",
                },
            ],
            "visualControlCandidates": [
                {
                    "elementId": 3,
                    "boxSource": [40, 400, 220, 470],
                    "background": "#FCF7EF",
                    "borderColor": "#006EFF",
                    "borderWidth": 2,
                    "cornerRadius": 35,
                    "labelElementIds": [2],
                }
            ],
            "symbolArt": [
                {
                    "text": "..<>..\n.<<<<.",
                    "boxSource": [420, 20, 760, 160],
                    "cellWidth": 12,
                    "rowPitch": 20,
                }
            ],
            "icons": [
                {"elementId": 4, "name": "cross", "boxSource": [10, 240, 30, 260]}
            ],
            "allowedRasterRegions": [
                {
                    "elementId": 5,
                    "boxSource": [300, 250, 600, 450],
                    "assetPath": str(asset_path),
                }
            ],
        },
    }


def test_starter_is_semantic_content_addressed_and_raster_bounded(tmp_path) -> None:
    crop = tmp_path / "car.png"
    crop.write_bytes(b"exact-allowed-crop")
    document = _document(crop)

    first = materialize_starter_project(
        document, str(tmp_path / "output"), no_store=False
    )
    second = materialize_starter_project(
        document, str(tmp_path / "output"), no_store=False
    )

    assert first is not None
    assert second is not None
    assert first["directory"] == second["directory"]
    index = Path(first["entryPath"]).read_text(encoding="utf-8")
    css = Path(first["stylesheetPath"]).read_text(encoding="utf-8")
    assert "THE SUMMER" in index
    assert index.count('<span class="sens-text">TICKETS</span>') == 1
    assert 'aria-label="TICKETS"' in index
    assert '<button class="sens-control"' in index
    assert 'data-sens-line="true"' in index
    assert "repeating-linear-gradient(to right,#E2E2E2 0 3px" in index
    assert '<pre class="sens-symbol-art"' in index
    assert '<polyline points="300,330 420,260 600,340"' in index
    assert '<svg class="sens-icon sens-icon-cross"' in index
    assert "..&lt;&gt;..\n.&lt;&lt;&lt;&lt;." in index
    assert index.count('<img class="sens-raster"') == 1
    assert "car.png" not in index
    assert "reference" not in index.casefold()
    assert "width:800px;height:500px" in css
    assert "background:#FCF7EF" in css
    assert "font-family:'Sens Inter Tight'" in css
    assert "font-family:'Sens Newsreader'" in css
    script = (Path(first["directory"]) / "script.js").read_text(encoding="utf-8")
    assert "measureText(text.textContent)" in script
    assert "actualBoundingBoxAscent" in script
    assert "fontBoundingBoxAscent" in script
    assert "slot.dataset.sensCapHeight" in script
    assert "desiredInkHeight" in script
    assert "matrix(${scaleX},0,0,${scaleY},${translateX},${translateY})" in script
    assert 'data-sens-text-box="true"' in index
    assert 'data-sens-cap-height="47"' in index
    assert first["sourceFiles"] == ["index.html", "styles.css", "script.js"]
    copied = {
        item.name: item for item in (Path(first["directory"]) / "assets").iterdir()
    }
    assert len(copied) == 3
    assert first["fontAssetCount"] == 2
    assert copied["sens-inter-tight.ttf"].stat().st_size > 500_000
    assert copied["sens-newsreader.ttf"].stat().st_size > 400_000
    assert any(item.read_bytes() == b"exact-allowed-crop" for item in copied.values())
    assert document["reconstruction"]["starterProject"] == second
    assert document["artifacts"][0]["kind"] == "semantic-web-starter"


def test_starter_fits_measured_words_into_their_individual_source_boxes(
    tmp_path,
) -> None:
    crop = tmp_path / "asset.png"
    crop.write_bytes(b"asset")
    document = _document(crop)
    title = document["reconstruction"]["text"][0]
    title["fontFeatures"]["wordBoxesSource"] = [
        {"text": "THE", "box": [20, 20, 130, 90]},
        {"text": "SUMMER", "box": [145, 20, 400, 90]},
    ]

    result = materialize_starter_project(
        document, str(tmp_path / "word-layout"), no_store=False
    )

    assert result is not None
    index = Path(result["entryPath"]).read_text(encoding="utf-8")
    script = (Path(result["directory"]) / "script.js").read_text(
        encoding="utf-8"
    )
    assert 'class="sens-text-slot sens-measured-words"' in index
    assert index.count('class="sens-word-slot sens-fit-slot"') == 2
    assert 'left:0px;top:0px;width:110px;height:70px' in index
    assert 'left:125px;top:0px;width:255px;height:70px' in index
    assert '<span class="sens-text">THE</span></span> <span' in index
    assert "querySelectorAll('.sens-fit-slot')" in script


def test_measured_words_can_select_different_bundled_render_families(
    tmp_path,
) -> None:
    crop = tmp_path / "asset.png"
    crop.write_bytes(b"asset")
    document = _document(crop)
    title = document["reconstruction"]["text"][0]
    title.update({"value": "Your new", "boxSource": [20, 20, 400, 90]})
    title["fontFeatures"]["wordBoxesSource"] = [
        {
            "text": "Your",
            "box": [20, 20, 180, 90],
            "renderFamily": "inter-tight",
        },
        {
            "text": "new",
            "box": [220, 20, 400, 90],
            "renderFamily": "newsreader",
            "renderWeight": 300,
            "slant": "italic",
        },
    ]

    result = materialize_starter_project(
        document, str(tmp_path / "mixed-family"), no_store=False
    )

    assert result is not None
    index = Path(result["entryPath"]).read_text(encoding="utf-8")
    word_line = next(
        line for line in index.splitlines() if 'data-sens-word-index="0"' in line
    )
    assert "font-family:'Sens Inter Tight'" in word_line
    assert "font-family:'Sens Newsreader'" in word_line
    assert "font-style:italic" in word_line


def test_indexed_control_label_renders_live_superscript(tmp_path) -> None:
    crop = tmp_path / "asset.png"
    crop.write_bytes(b"asset")
    document = _document(crop)
    title = document["reconstruction"]["text"][0]
    title.update(
        {
            "value": "A1 Sense",
            "preferredValue": "A1 Sense",
            "indexedLabel": {
                "prefix": "A",
                "index": "1",
                "label": "Sense",
                "superscript": True,
            },
        }
    )

    result = materialize_starter_project(
        document, str(tmp_path / "indexed-label"), no_store=False
    )

    assert result is not None
    index = Path(result["entryPath"]).read_text(encoding="utf-8")
    assert 'data-sens-indexed-label="true"' in index
    assert '<sup class="sens-indexed-label-index">1</sup>' in index
    assert "A1 Sense" in index


def test_unlabelled_icon_control_keeps_an_accessible_name(tmp_path) -> None:
    crop = tmp_path / "asset.png"
    crop.write_bytes(b"asset")
    document = _document(crop)
    document["reconstruction"]["visualControlCandidates"].append(
        {
            "elementId": "corner-scroll-control:18",
            "boxSource": [20, 420, 44, 468],
            "labelElementIds": [],
            "ariaLabel": "Scroll down",
            "interaction": "semantic-button",
            "background": "#F1F1F1",
            "borderColor": "transparent",
            "borderWidth": 0,
            "cornerRadius": 12,
            "zIndex": 24,
        }
    )

    result = materialize_starter_project(
        document, str(tmp_path / "icon-control"), no_store=False
    )

    assert result is not None
    index = Path(result["entryPath"]).read_text(encoding="utf-8")
    assert 'aria-label="Scroll down"' in index
    assert "z-index:24" in index


def test_tiny_svg_icons_keep_a_visible_one_pixel_stroke() -> None:
    markup = _icon_svg_markup("arrow-down", [129, 1194, 135, 1204], "#333333")

    assert 'stroke-width="4"' in markup


def test_starter_rejects_implausible_word_box_token_assignments(tmp_path) -> None:
    crop = tmp_path / "asset.png"
    crop.write_bytes(b"asset")
    document = _document(crop)
    title = document["reconstruction"]["text"][0]
    title.update(
        {
            "value": "Last 30 days",
            "boxSource": [304, 80, 402, 93],
        }
    )
    title["fontFeatures"]["wordBoxesSource"] = [
        {"text": "Last", "box": [304, 80, 315, 92]},
        {"text": "30", "box": [323, 82, 387, 93]},
        {"text": "days", "box": [395, 85, 402, 89]},
    ]

    result = materialize_starter_project(
        document, str(tmp_path / "invalid-word-layout"), no_store=False
    )

    assert result is not None
    index = Path(result["entryPath"]).read_text(encoding="utf-8")
    assert 'class="sens-text-slot sens-fit-slot"' in index
    assert 'class="sens-word-slot sens-fit-slot"' not in index
    assert '<span class="sens-text">Last 30 days</span>' in index


def test_starter_falls_back_to_natural_spaces_when_measured_words_touch(
    tmp_path,
) -> None:
    crop = tmp_path / "asset.png"
    crop.write_bytes(b"asset")
    document = _document(crop)
    title = document["reconstruction"]["text"][0]
    title.update({"value": "START NOW", "boxSource": [100, 100, 237, 126]})
    title["fontFeatures"].update(
        {
            "fontSize": 32,
            "capHeight": 26,
            "wordBoxesSource": [
                {"text": "START", "box": [100, 100, 186, 126]},
                {"text": "NOW", "box": [186, 100, 237, 126]},
            ],
        }
    )

    result = materialize_starter_project(
        document, str(tmp_path / "touching-words"), no_store=False
    )

    assert result is not None
    index = Path(result["entryPath"]).read_text(encoding="utf-8")
    assert 'class="sens-text-slot sens-fit-slot"' in index
    assert '<span class="sens-text">START NOW</span>' in index
    assert 'class="sens-word-slot sens-fit-slot"' not in index


def test_starter_renders_navigation_controls_as_links(tmp_path) -> None:
    crop = tmp_path / "asset.png"
    crop.write_bytes(b"asset")
    document = _document(crop)
    control = document["reconstruction"]["visualControlCandidates"][0]
    control["interaction"] = "semantic-link"
    control["semanticRole"] = "nav"

    project = materialize_starter_project(
        document, str(tmp_path / "output"), no_store=False
    )

    assert project is not None
    index = Path(project["entryPath"]).read_text(encoding="utf-8")
    assert '<a class="sens-control" href="#" aria-label="TICKETS"' in index
    assert '<button class="sens-control"' not in index


def test_starter_renders_mixed_typography_runs_inside_one_selectable_text_node(
    tmp_path,
) -> None:
    crop = tmp_path / "car.png"
    crop.write_bytes(b"exact-allowed-crop")
    document = _document(crop)
    headline = document["reconstruction"]["text"][0]
    headline["value"] = "Your new"
    headline["preferredValue"] = "Your new"
    headline["inlineRuns"] = [
        {
            "text": "Your ",
            "typographyCandidate": {
                "class": "sans-serif",
                "weight": "light",
                "slant": "normal",
                "confidence": 0.9,
            },
        },
        {
            "text": "new",
            "typographyCandidate": {
                "class": "serif",
                "contrast": "high",
                "weight": "light",
                "slant": "italic",
                "confidence": 0.9,
            },
        },
    ]

    result = materialize_starter_project(
        document, str(tmp_path / "output"), no_store=False
    )

    assert result is not None
    index = Path(result["entryPath"]).read_text(encoding="utf-8")
    script = (Path(result["directory"]) / "script.js").read_text(encoding="utf-8")
    assert 'data-sens-inline-runs="true"' in index
    assert '<span class="sens-inline-run"' in index
    assert "Your " in index
    assert "new</span>" in index
    assert "font-family:'Sens Newsreader'" in index
    assert "font-style:italic" in index
    assert "querySelectorAll('.sens-inline-run')" in script
    assert "measureText(run.textContent)" in script


def test_inline_slant_runs_inherit_the_line_family(tmp_path) -> None:
    crop = tmp_path / "car.png"
    crop.write_bytes(b"exact-allowed-crop")
    document = _document(crop)
    headline = document["reconstruction"]["text"][0]
    headline["value"] = "with AI DLP"
    headline["preferredValue"] = "with AI DLP"
    headline["typographyCandidate"] = {
        "class": "script",
        "weight": "regular",
        "confidence": 0.9,
    }
    headline["inlineRuns"] = [
        {"text": "with ", "typographyCandidate": {"slant": "italic"}},
        {"text": "AI DLP", "typographyCandidate": {"slant": "normal"}},
    ]

    result = materialize_starter_project(
        document, str(tmp_path / "output"), no_store=False
    )

    assert result is not None
    index = Path(result["entryPath"]).read_text(encoding="utf-8")
    assert index.count("font-family:'Sens Newsreader'") >= 2
    assert "font-style:italic" in index
    assert "font-style:normal" in index


def test_starter_places_alpha_masked_artwork_behind_live_dom(tmp_path) -> None:
    crop = tmp_path / "car.png"
    crop.write_bytes(b"foreground")
    background = tmp_path / "background.png"
    background.write_bytes(b"alpha-background")
    document = _document(crop)
    document["reconstruction"]["allowedRasterRegions"].insert(
        0,
        {
            "elementId": "background-artwork",
            "artifactId": "raster:background",
            "kind": "alpha-masked-background-artwork",
            "boxSource": [0, 0, 800, 500],
            "assetPath": str(background),
        },
    )

    result = materialize_starter_project(
        document, str(tmp_path / "output"), no_store=False
    )

    assert result is not None
    index = Path(result["entryPath"]).read_text(encoding="utf-8")
    css = Path(result["stylesheetPath"]).read_text(encoding="utf-8")
    assert 'class="sens-background-artwork"' in index
    assert 'data-sens-raster-role="alpha-masked-background-artwork"' in index
    assert 'data-sens-artifact-id="raster:background"' in index
    assert index.count('<img class="sens-raster"') == 1
    assert ".sens-background-artwork{z-index:0" in css


def test_starter_keeps_preserved_control_semantic_and_skips_duplicate_icon(
    tmp_path,
) -> None:
    crop = tmp_path / "car.png"
    crop.write_bytes(b"foreground")
    document = _document(crop)
    control = document["reconstruction"]["visualControlCandidates"][0]
    control.update(
        {
            "decorationPreservedInBackgroundArtwork": True,
            "background": "#00000000",
            "borderColor": "#00000000",
            "borderWidth": 0,
        }
    )
    document["reconstruction"]["icons"][0][
        "preservedInBackgroundArtwork"
    ] = True

    result = materialize_starter_project(
        document, str(tmp_path / "output"), no_store=False
    )

    assert result is not None
    index = Path(result["entryPath"]).read_text(encoding="utf-8")
    assert '<button class="sens-control"' in index
    assert "background:#00000000" in index
    assert "border:0px solid #00000000" in index
    assert "sens-icon" not in index


def test_starter_does_not_duplicate_non_text_chrome_preserved_in_background(
    tmp_path,
) -> None:
    crop = tmp_path / "car.png"
    crop.write_bytes(b"foreground")
    document = _document(crop)
    for surface in document["reconstruction"]["surfaces"]:
        surface["preservedInBackgroundArtwork"] = True
    for line in document["reconstruction"]["structuralLines"]:
        line["preservedInBackgroundArtwork"] = True
    for vector in document["reconstruction"]["vectorPaths"]:
        vector["preservedInBackgroundArtwork"] = True

    result = materialize_starter_project(
        document, str(tmp_path / "output"), no_store=False
    )

    assert result is not None
    index = Path(result["entryPath"]).read_text(encoding="utf-8")
    assert '<div class="sens-surface"' not in index
    assert "background:transparent" in index
    assert "<polyline" not in index


def test_starter_honors_no_store(tmp_path) -> None:
    crop = tmp_path / "asset.png"
    crop.write_bytes(b"asset")
    document = _document(crop)

    assert (
        materialize_starter_project(document, str(tmp_path / "output"), no_store=True)
        is None
    )
    assert not (tmp_path / "output").exists()


def test_starter_prefers_measured_stroke_weight_and_high_contrast_serif() -> None:
    entry = {
        "fontFeatures": {"weightCandidate": "light"},
        "typographyCandidate": {
            "class": "slab-serif",
            "contrast": "high",
            "weight": "bold",
        },
    }

    assert _font_weight(entry) == 300
    assert _font_family(entry).startswith("'Sens Newsreader'")


def test_starter_prefers_measured_bundled_render_family_and_weight() -> None:
    serif = {
        "fontFeatures": {
            "renderFamily": "newsreader",
            "renderFamilyConfidence": 0.78,
            "renderWeight": 300,
        },
        "typographyCandidate": {
            "class": "sans-serif",
            "confidence": 0.0,
            "weight": "black",
        },
    }

    assert _font_family(serif).startswith("'Sens Newsreader'")
    assert _font_weight(serif) == 300


def test_starter_uses_measured_ink_for_unresolved_ui_weight() -> None:
    revenue = {
        "value": "$210,429.00",
        "fontFeatures": {
            "fontSize": 26,
            "strokeWidthRatio": 0.105,
            "inkCoverage": 0.33,
        },
    }
    compact_label = {
        "value": "Confirm pending payouts",
        "fontFeatures": {
            "fontSize": 15,
            "strokeWidthRatio": 0.18,
            "inkCoverage": 0.51,
        },
        "typographyCandidate": {"weight": "regular"},
    }
    long_url = {
        "value": "partners.dub.com/programs/acme-projects",
        "fontFeatures": {"fontSize": 11, "inkCoverage": 0.51},
    }

    assert _font_weight(revenue) == 600
    assert _font_weight(compact_label) == 600
    assert _font_weight(long_url) == 400


def test_starter_uses_canvas_token_beneath_measured_surfaces(tmp_path) -> None:
    crop = tmp_path / "asset.png"
    crop.write_bytes(b"asset")
    document = _document(crop)
    document["tokens"]["color"]["canvas"] = {
        "$type": "color",
        "$value": "#E5E5E5",
    }

    project = materialize_starter_project(
        document, str(tmp_path / "output"), no_store=False
    )

    assert project is not None
    css = Path(project["stylesheetPath"]).read_text(encoding="utf-8")
    assert "html,body{margin:0;width:100%;min-height:100%;background:#E5E5E5}" in css
    assert ".sens-canvas{position:relative;width:800px;height:500px" in css
    assert "background:#E5E5E5" in css


def test_starter_rejects_display_and_condensed_guesses_for_tiny_blurred_text() -> None:
    tiny_display = {
        "fontFeatures": {"fontSize": 15},
        "typographyCandidate": {"class": "display-sans", "confidence": 0.9},
    }
    tiny_oswald = {
        "fontFeatures": {
            "fontSize": 14,
            "family": "oswald",
            "familyConfidence": 0.4,
        }
    }

    assert _font_family(tiny_display).startswith("Arial")
    assert _font_family(tiny_oswald).startswith("Arial")
    assert _font_family({"fontFeatures": {"fontSize": 26}}).startswith("Arial")
    assert _font_family(
        {
            "fontFeatures": {"fontSize": 64},
            "typographyCandidate": {"class": "display-sans", "confidence": 0.9},
        }
    ).startswith("'Sens Inter Tight'")
    assert _font_family(
        {
            "fontFeatures": {"fontSize": 64},
            "typographyCandidate": {
                "class": "display",
                "width": "expanded",
                "confidence": 0.9,
            },
        }
    ).startswith("Arial")


def test_giant_expanded_high_ink_display_uses_black_system_face() -> None:
    entry = {
        "value": "BOLD FLAVOR",
        "fontFeatures": {
            "fontSize": 221,
            "strokeWidthRatio": 0.124,
            "inkCoverage": 0.564,
            "weightCandidate": "bold",
        },
        "typographyCandidate": {
            "class": "display",
            "width": "expanded",
            "weight": "bold",
            "confidence": 0.9,
        },
    }

    assert _font_family(entry).startswith("'Arial Black'")
    assert _font_weight(entry) == 900
    measured_only = {
        "value": "BOLD FLAVOR",
        "fontFeatures": dict(entry["fontFeatures"]),
        "typographyCandidate": None,
    }
    assert _font_family(measured_only).startswith("'Arial Black'")
    assert _font_weight(measured_only) == 900


def test_large_medium_strokes_are_not_promoted_to_heavy_display_weight() -> None:
    gateway = {
        "value": "Gateway",
        "fontFeatures": {
            "fontSize": 136,
            "strokeWidthRatio": 0.0566,
            "inkCoverage": 0.3065,
            "weightCandidate": "bold",
        },
        "typographyCandidate": {"class": "sans-serif", "confidence": 0.9},
    }
    secure = {
        "value": "Secure Web",
        "fontFeatures": {
            "fontSize": 88,
            "strokeWidthRatio": 0.0938,
            "inkCoverage": 0.4095,
            "weightCandidate": "bold",
        },
        "typographyCandidate": {"class": "sans-serif", "confidence": 0.9},
    }

    assert _font_weight(gateway) == 500
    assert _font_weight(secure) == 600


def test_large_low_coverage_strokes_remain_light() -> None:
    tagline = {
        "value": "A NO-WORK, WORK-EVENT FOR TEAMS",
        "fontFeatures": {
            "fontSize": 60,
            "strokeWidthRatio": 0.0636,
            "inkCoverage": 0.2267,
            "weightCandidate": "bold",
        },
        "typographyCandidate": {"class": "sans-serif", "confidence": 0.0},
    }

    assert _font_weight(tagline) == 300


def test_confident_serif_run_is_not_overridden_by_noisy_photo_stroke_metrics() -> None:
    entry = {
        "value": "with AI DLP",
        "fontFeatures": {
            "fontSize": 126,
            "strokeWidthRatio": 0.122,
            "inkCoverage": 0.622,
            "weightCandidate": "bold",
        },
        "typographyCandidate": {
            "class": "slab-serif",
            "weight": "regular",
            "contrast": "medium",
            "confidence": 0.9,
        },
    }

    assert _font_family(entry).startswith("Georgia")
    assert _font_weight(entry) == 400


def test_script_run_uses_a_live_italic_serif_instead_of_heavy_sans() -> None:
    entry = {
        "value": "with AI DLP",
        "fontFeatures": {
            "fontSize": 126,
            "strokeWidthRatio": 0.122,
            "inkCoverage": 0.622,
            "weightCandidate": "bold",
        },
        "typographyCandidate": {
            "class": "script",
            "weight": "bold",
            "contrast": "high",
            "confidence": 0.9,
        },
    }

    assert _font_family(entry).startswith("'Sens Newsreader'")
    assert _font_weight(entry) == 300
    assert _font_style(entry) == "italic"

    thin_entry = {
        **entry,
        "fontFeatures": {
            **entry["fontFeatures"],
            "weightCandidate": "light",
        },
    }
    assert _font_weight(thin_entry) == 300


def test_starter_caps_tiny_control_label_typography(tmp_path) -> None:
    crop = tmp_path / "asset.png"
    crop.write_bytes(b"asset")
    document = _document(crop)
    label = document["reconstruction"]["text"][1]
    label["fontFeatures"]["fontSize"] = 44
    label["typographyCandidate"] = {
        "class": "display",
        "weight": "bold",
        "confidence": 0.9,
    }
    control = document["reconstruction"]["visualControlCandidates"][0]
    control["boxSource"] = [40, 400, 220, 435]

    project = materialize_starter_project(
        document, str(tmp_path / "output"), no_store=False
    )

    assert project is not None
    index = Path(project["entryPath"]).read_text(encoding="utf-8")
    button = next(line for line in index.splitlines() if "sens-control" in line)
    assert "font-family:Arial" in button
    assert "font-weight:400" in button
    assert "font-size:21.7px" in button
    assert 'data-sens-cap-height="16.5"' in button


def test_starter_renders_cyrillic_starburst_ocr_as_a_visible_icon(tmp_path) -> None:
    crop = tmp_path / "asset.png"
    crop.write_bytes(b"asset")
    document = _document(crop)
    document["reconstruction"]["icons"].append(
        {
            "elementId": 9,
            "name": "starburst",
            "boxSource": [300, 40, 349, 84],
            "color": "#333333",
        }
    )

    project = materialize_starter_project(
        document, str(tmp_path / "starburst"), no_store=False
    )

    assert project is not None
    index = Path(project["entryPath"]).read_text(encoding="utf-8")
    assert '<svg class="sens-icon sens-icon-starburst"' in index
    assert 'fill="#F1F1F1"' in index


def test_starter_renders_numeric_badges_as_live_text_on_css_surfaces(
    tmp_path,
) -> None:
    crop = tmp_path / "asset.png"
    crop.write_bytes(b"asset")
    document = _document(crop)
    document["reconstruction"]["text"].append(
        {
            "elementId": 6,
            "value": "12",
            "preferredValue": "12",
            "boxSource": [706, 48, 718, 57],
            "fontFeatures": {"fontSize": 11, "color": "#3977EA"},
            "color": "#3977EA",
        }
    )
    document["reconstruction"]["badges"] = [
        {
            "elementId": "badge-1",
            "labelElementId": 6,
            "boxSource": [696, 38, 726, 67],
            "background": "#DCEBFF",
            "foreground": "#3977EA",
            "cornerRadius": 10,
            "value": "12",
        }
    ]

    project = materialize_starter_project(
        document, str(tmp_path / "output"), no_store=False
    )

    assert project is not None
    index = Path(project["entryPath"]).read_text(encoding="utf-8")
    css = Path(project["stylesheetPath"]).read_text(encoding="utf-8")
    assert index.count(">12<") == 1
    assert 'class="sens-badge"' in index
    assert 'data-sens-badge="true"' in index
    assert "background:#DCEBFF" in index
    assert "color:#3977EA" in index
    assert ".sens-badge-text{display:block;user-select:text}" in css


def test_starter_renders_measured_symbol_glyphs_as_selectable_text_and_css(
    tmp_path,
) -> None:
    crop = tmp_path / "asset.png"
    crop.write_bytes(b"asset")
    document = _document(crop)
    document["reconstruction"]["symbolArt"] = [
        {
            "text": ".◆\n ◆",
            "boxSource": [100, 100, 160, 160],
            "cellWidth": 12.16,
            "rowPitch": 24.8,
            "firstCellCenterX": 106.0,
            "firstBaselineY": 118.0,
            "color": "#FFFFFF",
            "glyphGeometry": {
                "dot": {
                    "width": 4.0,
                    "height": 5.0,
                    "centerOffsetX": 0.0,
                    "centerOffsetY": 0.0,
                },
                "diamond": {
                    "width": 13.0,
                    "height": 14.0,
                    "centerOffsetX": 0.0,
                    "centerOffsetY": -6.0,
                },
            },
        }
    ]

    project = materialize_starter_project(
        document, str(tmp_path / "output"), no_store=False
    )

    assert project is not None
    index = Path(project["entryPath"]).read_text(encoding="utf-8")
    css = Path(project["stylesheetPath"]).read_text(encoding="utf-8")
    assert ".◆\n ◆" in index
    assert index.count("sens-symbol-dot") == 1
    assert index.count("sens-symbol-diamond") == 2
    assert "color:transparent" in index
    assert "left:4px;top:15.5px;width:4px;height:5px" in index
    assert "left:11.66px;top:5px;width:13px;height:14px" in index
    assert ".sens-symbol-diamond{clip-path:polygon" in css
    assert '<img class="sens-raster"' in index
