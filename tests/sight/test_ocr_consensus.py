import cv2
import numpy as np

from sight import ocr
from sight.ocr import merge_ocr_passes, merge_script_ocr_passes


def test_multiscale_ocr_confirms_matching_text() -> None:
    base = [
        {
            "text": "THE SUMMER",
            "box": [10, 20, 210, 80],
            "confidence": 0.88,
            "source": "inferred",
            "method": "rapidocr",
        }
    ]
    scaled = [
        {
            "text": "THE SUMMER",
            "box": [15, 30, 315, 120],
            "confidence": 0.94,
            "source": "inferred",
            "method": "rapidocr",
        }
    ]

    [result] = merge_ocr_passes(base, scaled, scale=1.5)

    assert result["text"] == "THE SUMMER"
    assert result["box"] == [10, 20, 210, 80]
    assert result["verified"] is True
    assert result["method"] == "rapidocr-multiscale-consensus"


def test_multiscale_ocr_exposes_disagreement_instead_of_claiming_exact_text() -> None:
    base = [
        {
            "text": "HЕMЕR",
            "box": [10, 20, 210, 80],
            "confidence": 0.71,
            "source": "inferred",
            "method": "rapidocr",
        }
    ]
    scaled = [
        {
            "text": "THE SUMMER",
            "box": [15, 30, 315, 120],
            "confidence": 0.94,
            "source": "inferred",
            "method": "rapidocr",
        }
    ]

    [result] = merge_ocr_passes(base, scaled, scale=1.5)

    assert result["text"] == "THE SUMMER"
    assert result["verified"] is False
    assert result["method"] == "rapidocr-multiscale-disagreement"
    assert {item["text"] for item in result["alternatives"]} == {
        "HЕMЕR",
        "THE SUMMER",
    }


def test_reconstruction_ocr_runs_one_bounded_scaled_pass(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "poster.png"
    cv2.imwrite(str(image_path), np.full((200, 400, 3), 255, np.uint8))
    observed_shape = None

    def fake_run(path):
        nonlocal observed_shape
        image = cv2.imread(path)
        observed_shape = image.shape[:2]
        return [
            {
                "text": "THE SUMMER",
                "box": [15, 30, 315, 120],
                "confidence": 0.94,
                "source": "inferred",
                "method": "rapidocr",
            }
        ]

    monkeypatch.setattr(ocr, "run_ocr", fake_run)
    base = [
        {
            "text": "HЕMЕR",
            "box": [10, 20, 210, 80],
            "confidence": 0.71,
            "source": "inferred",
            "method": "rapidocr",
        }
    ]

    [result] = ocr.refine_ocr_for_reconstruction(
        str(image_path), base, scale=1.5, max_pixels=1_000_000
    )

    assert observed_shape == (300, 600)
    assert result["text"] == "THE SUMMER"
    assert result["verified"] is False


def test_dual_script_ocr_preserves_visible_currency_sigil() -> None:
    base = [
        {
            "text": "SERA",
            "box": [30, 20, 110, 50],
            "confidence": 0.975,
            "method": "rapidocr",
        }
    ]
    latin = [
        {
            "text": "$ERA",
            "box": [31, 20, 111, 50],
            "confidence": 1.0,
            "method": "rapidocr",
        }
    ]

    [result] = merge_script_ocr_passes(base, latin)

    assert result["text"] == "$ERA"
    assert result["verified"] is True
    assert result["method"] == "rapidocr-dual-script-visible-sigil-consensus"
    assert {item["text"] for item in result["alternatives"]} == {"SERA", "$ERA"}


def test_dual_script_ocr_keeps_close_non_sigil_disagreement_unverified() -> None:
    base = [
        {
            "text": "Metalayer",
            "box": [30, 20, 110, 50],
            "confidence": 0.96,
            "method": "rapidocr",
        }
    ]
    latin = [
        {
            "text": "MetaIayer",
            "box": [31, 20, 111, 50],
            "confidence": 0.98,
            "method": "rapidocr",
        }
    ]

    [result] = merge_script_ocr_passes(base, latin)

    assert result["text"] == "Metalayer"
    assert result["verified"] is False
    assert result["method"] == "rapidocr-dual-script-disagreement"


def test_dual_script_ocr_prefers_high_confidence_complete_latin_line() -> None:
    base = [
        {
            "text": "A NO-WORK, WORKEEVENT FOR TEAMS IN THE DRVE OAPITAL PORTFOLIO",
            "box": [268, 594, 2288, 657],
            "confidence": 0.948,
            "method": "rapidocr-multiscale-consensus",
        }
    ]
    latin = [
        {
            "text": "A NO-WORK, WORK-EVENT FOR TEAMS IN THE DRIVE CAPITAL PORTFOLIO",
            "box": [268, 594, 2288, 657],
            "confidence": 0.978,
            "method": "rapidocr",
        }
    ]

    [result] = merge_script_ocr_passes(base, latin)

    assert result["text"] == latin[0]["text"]
    assert result["verified"] is True
    assert result["method"] == "rapidocr-dual-script-latin-preferred"


def test_dual_script_ocr_keeps_measured_space_when_glyphs_match() -> None:
    base = [
        {
            "text": "THE SUMMER",
            "box": [5, 8, 2551, 304],
            "confidence": 1.0,
            "method": "rapidocr-multiscale-consensus",
        }
    ]
    latin = [
        {
            "text": "THESUMMER",
            "box": [5, 8, 2551, 304],
            "confidence": 1.0,
            "method": "rapidocr",
        }
    ]

    [result] = merge_script_ocr_passes(base, latin)

    assert result["text"] == "THE SUMMER"
    assert result["verified"] is False
