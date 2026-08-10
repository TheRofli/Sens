import cv2
import numpy as np

from sight import compare


def test_compare_rejects_dimension_mismatch_without_resampling(tmp_path, monkeypatch) -> None:
    reference_path = tmp_path / "reference.png"
    candidate_path = tmp_path / "candidate.png"
    cv2.imwrite(str(reference_path), np.full((120, 160, 3), 255, np.uint8))
    cv2.imwrite(str(candidate_path), np.full((100, 140, 3), 255, np.uint8))
    monkeypatch.setattr(compare, "run_ocr", lambda _path: [])

    result = compare.compare_images(str(reference_path), str(candidate_path))

    assert result["dimensions"] == {
        "reference": {"width": 160, "height": 120},
        "candidate": {"width": 140, "height": 100},
        "exactMatch": False,
        "aspectRatioDelta": 0.066667,
        "fit": "strict",
        "resampled": False,
    }
    assert result["verdict"] == "fail"
    assert result["canComplete"] is False
    assert "dimension_mismatch" in result["blockingReasons"]
    assert result["requiredAction"] == {
        "kind": "rerender_exact_dimensions",
        "referenceSize": {"width": 160, "height": 120},
        "candidateSize": {"width": 140, "height": 100},
        "reason": "Candidate dimensions must exactly match the immutable reference before visual completion can be evaluated.",
    }


def test_compare_verdict_is_a_hard_completion_gate(tmp_path, monkeypatch) -> None:
    reference_path = tmp_path / "reference.png"
    identical_path = tmp_path / "identical.png"
    shifted_path = tmp_path / "shifted.png"
    reference = np.full((120, 160, 3), 255, np.uint8)
    shifted = reference.copy()
    cv2.rectangle(reference, (20, 30), (80, 70), (0, 0, 0), -1)
    cv2.rectangle(shifted, (55, 30), (115, 70), (0, 0, 0), -1)
    cv2.imwrite(str(reference_path), reference)
    cv2.imwrite(str(identical_path), reference)
    cv2.imwrite(str(shifted_path), shifted)
    monkeypatch.setattr(compare, "run_ocr", lambda _path: [{"text": "SAVE"}])

    passed = compare.compare_images(str(reference_path), str(identical_path))
    failed = compare.compare_images(str(reference_path), str(shifted_path))

    assert passed["verdict"] == "pass"
    assert passed["canComplete"] is True
    assert passed["completionScope"] == "visual-only"
    assert passed["visualPass"] is True
    assert passed["webCompletionWarning"] == (
        "For screenshot-to-web work, visualPass is insufficient; sens_review must also return webPass=true."
    )
    assert passed["blockingReasons"] == []
    assert all(check["passed"] for check in passed["acceptance"]["checks"])
    assert failed["verdict"] == "fail"
    assert failed["canComplete"] is False
    assert failed["blockingReasons"]
    assert failed["hotRegions"][0]["areaRatio"] > 0


def test_explicit_resize_fit_is_compatible_but_cannot_prove_completion(
    tmp_path, monkeypatch
) -> None:
    reference_path = tmp_path / "reference.png"
    candidate_path = tmp_path / "candidate.png"
    cv2.imwrite(str(reference_path), np.full((120, 160, 3), 255, np.uint8))
    cv2.imwrite(str(candidate_path), np.full((60, 80, 3), 255, np.uint8))
    monkeypatch.setattr(compare, "run_ocr", lambda _path: [])

    result = compare.compare_images(
        str(reference_path), str(candidate_path), fit="resize"
    )

    assert result["dimensions"]["fit"] == "resize"
    assert result["dimensions"]["resampled"] is True
    assert result["verdict"] == "fail"
    assert result["canComplete"] is False
    assert "resampled_candidate" in result["blockingReasons"]


def test_compare_exposes_foreground_weighted_mismatch(tmp_path, monkeypatch) -> None:
    reference_path = tmp_path / "reference.png"
    candidate_path = tmp_path / "candidate.png"
    reference = np.full((200, 300, 3), 248, np.uint8)
    candidate = reference.copy()
    cv2.rectangle(reference, (120, 80), (180, 120), (20, 20, 20), -1)
    cv2.rectangle(candidate, (130, 80), (190, 120), (20, 20, 20), -1)
    cv2.imwrite(str(reference_path), reference)
    cv2.imwrite(str(candidate_path), candidate)
    monkeypatch.setattr(compare, "run_ocr", lambda _path: [])

    result = compare.compare_images(str(reference_path), str(candidate_path))

    foreground = result["metrics"]["foreground"]
    assert foreground["coverageRatio"] > 0
    assert foreground["mismatchRatio"] > result["metrics"]["pixel"]["mismatchRatio"]
    assert foreground["method"] == "local-detail-mask-tolerant-xor"
    assert "foreground_mismatch_maximum" in result["blockingReasons"]


def test_foreground_mask_ignores_multiple_panel_fills_and_scores_local_detail(
    tmp_path, monkeypatch
) -> None:
    reference_path = tmp_path / "reference.png"
    candidate_path = tmp_path / "candidate.png"
    reference = np.full((240, 400, 3), 255, np.uint8)
    candidate = reference.copy()
    reference[:, :120] = (238, 238, 238)
    candidate[:, :120] = (238, 238, 238)
    for y in (50, 90, 130, 170):
        cv2.rectangle(reference, (22, y), (92, y + 8), (40, 40, 40), -1)
        cv2.rectangle(candidate, (42, y), (112, y + 8), (40, 40, 40), -1)
    cv2.rectangle(reference, (165, 40), (360, 205), (245, 245, 245), 1)
    cv2.rectangle(candidate, (165, 40), (360, 205), (245, 245, 245), 1)
    cv2.imwrite(str(reference_path), reference)
    cv2.imwrite(str(candidate_path), candidate)
    monkeypatch.setattr(compare, "run_ocr", lambda _path: [])

    result = compare.compare_images(str(reference_path), str(candidate_path))
    foreground = result["metrics"]["foreground"]

    assert foreground["coverageRatio"] < 0.2
    assert foreground["mismatchRatio"] > foreground["threshold"]
    assert "foreground_mismatch_maximum" in result["blockingReasons"]


def test_foreground_geometry_ignores_color_and_antialias_only_differences(
    tmp_path, monkeypatch
) -> None:
    reference_path = tmp_path / "reference.png"
    candidate_path = tmp_path / "candidate.png"
    reference = np.zeros((120, 180, 3), np.uint8)
    candidate = reference.copy()
    cv2.rectangle(reference, (35, 30), (145, 90), (255, 255, 255), -1)
    cv2.rectangle(candidate, (35, 30), (145, 90), (235, 235, 235), -1)
    cv2.imwrite(str(reference_path), reference)
    cv2.imwrite(str(candidate_path), candidate)
    monkeypatch.setattr(compare, "run_ocr", lambda _path: [])

    result = compare.compare_images(str(reference_path), str(candidate_path))

    assert result["metrics"]["pixel"]["rawMismatchRatio"] > 0.3
    assert result["metrics"]["pixel"]["mismatchRatio"] > 0.25
    assert result["metrics"]["foreground"]["mismatchRatio"] == 0.0


def test_foreground_tolerance_scales_with_reference_resolution(
    tmp_path, monkeypatch
) -> None:
    reference_path = tmp_path / "reference.png"
    candidate_path = tmp_path / "candidate.png"
    image = np.full((1024, 1600, 3), 255, np.uint8)
    cv2.rectangle(image, (180, 240), (1380, 780), (0, 0, 0), 8)
    cv2.imwrite(str(reference_path), image)
    cv2.imwrite(str(candidate_path), image)
    monkeypatch.setattr(compare, "run_ocr", lambda _path: [])

    result = compare.compare_images(str(reference_path), str(candidate_path))

    foreground = result["metrics"]["foreground"]
    assert foreground["tolerancePx"] == 4
    assert foreground["toleranceRule"] == "round(min(width,height)/256), clamped 1..6"


def test_color_fidelity_uses_position_independent_palette(tmp_path, monkeypatch) -> None:
    reference_path = tmp_path / "reference.png"
    candidate_path = tmp_path / "candidate.png"
    reference = np.full((160, 240, 3), 250, np.uint8)
    candidate = reference.copy()
    cv2.rectangle(reference, (20, 40), (90, 120), (255, 110, 0), -1)
    cv2.rectangle(candidate, (150, 40), (220, 120), (255, 110, 0), -1)
    cv2.imwrite(str(reference_path), reference)
    cv2.imwrite(str(candidate_path), candidate)
    monkeypatch.setattr(compare, "run_ocr", lambda _path: [])

    result = compare.compare_images(str(reference_path), str(candidate_path))
    color = result["metrics"]["color"]

    assert color["meanLabDelta"] > 0
    assert color["paletteMeanLabDelta"] == 0.0
    assert color["paletteSimilarity"] == 1.0
    assert color["scoreMethod"] == "position-independent-quantized-lab-palette"


def test_color_palette_detects_an_actual_color_change(tmp_path, monkeypatch) -> None:
    reference_path = tmp_path / "reference.png"
    candidate_path = tmp_path / "candidate.png"
    reference = np.full((160, 240, 3), 250, np.uint8)
    candidate = reference.copy()
    cv2.rectangle(reference, (40, 40), (200, 120), (255, 0, 0), -1)
    cv2.rectangle(candidate, (40, 40), (200, 120), (0, 180, 0), -1)
    cv2.imwrite(str(reference_path), reference)
    cv2.imwrite(str(candidate_path), candidate)
    monkeypatch.setattr(compare, "run_ocr", lambda _path: [])

    color = compare.compare_images(
        str(reference_path), str(candidate_path)
    )["metrics"]["color"]

    assert color["paletteMeanLabDelta"] > 10
    assert color["paletteSimilarity"] < 0.8


def test_compare_reports_multisignal_metrics_and_next_focus(tmp_path, monkeypatch) -> None:
    reference_path = tmp_path / "reference.png"
    candidate_path = tmp_path / "candidate.png"
    reference = np.full((120, 160, 3), 255, np.uint8)
    candidate = reference.copy()
    cv2.rectangle(reference, (20, 30), (80, 70), (0, 0, 0), -1)
    cv2.rectangle(candidate, (35, 30), (95, 70), (0, 0, 0), -1)
    cv2.imwrite(str(reference_path), reference)
    cv2.imwrite(str(candidate_path), candidate)

    monkeypatch.setattr(
        compare,
        "run_ocr",
        lambda path: [{"text": "SAVE" if "reference" in path else "SAV"}],
    )

    result = compare.compare_images(str(reference_path), str(candidate_path))

    assert set(result["metrics"]) == {
        "pixel", "foreground", "color", "edge", "text", "layout"
    }
    assert 0.0 <= result["similarityScore"] < 1.0
    assert result["metrics"]["edge"]["mismatchRatio"] > 0
    assert result["metrics"]["text"]["similarity"] < 1
    assert result["hotRegions"]
    assert result["nextActions"][0]["tool"] == "sens_zoom"
    assert result["nextActions"][0]["arguments"]["region"]["width"] > 0


def test_hot_region_area_counts_actual_signal_not_its_sparse_bounding_box() -> None:
    height, width = 300, 500
    mask_u8 = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask_u8, (80, 50), (420, 250), 1, 2)
    mask = mask_u8.astype(bool)
    score = np.zeros((height, width), np.float32)
    score[mask] = 80.0

    zones = compare._extract_hot_regions(score, mask, width, height)

    assert zones
    largest = zones[0]
    assert largest["boundingAreaRatio"] > 0.4
    assert largest["areaRatio"] < 0.03
    assert largest["area"] == largest["signalArea"]
    assert largest["areaRatio"] == largest["signalAreaRatio"]


def test_hot_region_area_still_blocks_a_large_solid_defect() -> None:
    height, width = 300, 500
    mask = np.zeros((height, width), dtype=bool)
    mask[50:250, 80:420] = True
    score = np.zeros((height, width), np.float32)
    score[mask] = 80.0

    largest = compare._extract_hot_regions(score, mask, width, height)[0]

    assert largest["areaRatio"] > compare.PASS_THRESHOLDS["largestHotRegionRatio"]
    assert largest["areaRatio"] == largest["signalAreaRatio"]
    assert largest["boundingAreaRatio"] >= largest["areaRatio"]


def test_dense_broad_hot_region_gets_a_material_bounding_gate() -> None:
    dense = {
        "signalAreaRatio": 0.0465,
        "boundingAreaRatio": 0.1127,
    }
    sparse = {
        "signalAreaRatio": 0.0098,
        "boundingAreaRatio": 0.20,
    }

    assert compare._material_hot_region_bounding_ratio(dense) == 0.1127
    assert compare._material_hot_region_bounding_ratio(sparse) == 0.0
    assert (
        compare._material_hot_region_bounding_ratio(dense)
        > compare.PASS_THRESHOLDS["largestMaterialHotRegionBoundingRatio"]
    )


def test_ocr_text_similarity_tolerates_visual_script_confusion_and_row_order() -> None:
    reference = "partners revenue сommissions landing page"
    candidate = "landing page partners revenue commissions"

    similarity, method = compare._text_similarity(reference, candidate)

    assert similarity >= 0.7
    assert method == "visual-latin-character-bigram-dice"
