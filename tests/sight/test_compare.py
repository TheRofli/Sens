import cv2
import numpy as np

from sight import compare


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

    assert set(result["metrics"]) == {"pixel", "color", "edge", "text", "layout"}
    assert 0.0 <= result["similarityScore"] < 1.0
    assert result["metrics"]["edge"]["mismatchRatio"] > 0
    assert result["metrics"]["text"]["similarity"] < 1
    assert result["hotRegions"]
    assert result["nextActions"][0]["tool"] == "sens_zoom"
    assert result["nextActions"][0]["arguments"]["region"]["width"] > 0
