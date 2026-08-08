import cv2
import numpy as np

from sight import ocr
from sight.ocr import merge_ocr_passes


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
