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
    assert foreground["method"] == "border-background-union-mask"
    assert "foreground_mismatch_maximum" in result["blockingReasons"]


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
