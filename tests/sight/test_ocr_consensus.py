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


def test_downscaled_display_ocr_discovers_hero_word_missed_at_native_scale(
    tmp_path, monkeypatch
) -> None:
    image_path = tmp_path / "hero.png"
    cv2.imwrite(str(image_path), np.full((900, 1440, 3), 255, np.uint8))

    monkeypatch.setattr(
        ocr,
        "run_ocr_image",
        lambda _image: [
            {
                "text": "Slsн",
                "box": [103, 42, 537, 330],
                "confidence": 0.664,
                "method": "rapidocr",
            }
        ],
    )
    monkeypatch.setattr(
        ocr,
        "run_latin_ocr_image",
        lambda _image: [
            {
                "text": "SLUSH",
                "box": [103, 42, 537, 330],
                "confidence": 0.99,
                "method": "rapidocr",
            },
            {
                "text": "Your money. Unstuck.",
                "box": [202, 306, 517, 344],
                "confidence": 0.99,
                "method": "rapidocr",
            },
        ],
    )

    [display] = ocr.discover_display_ocr(str(image_path), [], scale=0.5)

    assert display["text"] == "SLUSH"
    assert display["box"] == [206, 84, 1074, 660]
    assert display["confidence"] == 0.99
    assert display["verified"] is True
    assert display["method"] == "rapidocr-downscaled-display-latin-preferred"
    assert display["displayScale"] == 0.5


def test_downscaled_display_prefers_complete_high_confidence_latin_word(
    tmp_path, monkeypatch
) -> None:
    image_path = tmp_path / "hero-with-overlays.png"
    cv2.imwrite(str(image_path), np.full((900, 1440, 3), 255, np.uint8))
    monkeypatch.setattr(
        ocr,
        "run_ocr_image",
        lambda _image: [
            {
                "text": "Ss",
                "box": [69, 19, 650, 355],
                "confidence": 0.852,
                "method": "rapidocr",
            }
        ],
    )
    monkeypatch.setattr(
        ocr,
        "run_latin_ocr_image",
        lambda _image: [
            {
                "text": "SLUSH",
                "box": [69, 19, 650, 355],
                "confidence": 0.947,
                "method": "rapidocr",
            }
        ],
    )

    [display] = ocr.discover_display_ocr(str(image_path), [], scale=0.5)

    assert display["text"] == "SLUSH"
    assert display["confidence"] == 0.947
    assert display["verified"] is True
    assert display["method"] == "rapidocr-downscaled-display-latin-preferred"
    assert {item["text"] for item in display["alternatives"]} == {"Ss", "SLUSH"}


def test_downscaled_display_ocr_does_not_duplicate_existing_live_text(
    tmp_path, monkeypatch
) -> None:
    image_path = tmp_path / "hero.png"
    cv2.imwrite(str(image_path), np.full((900, 1440, 3), 255, np.uint8))
    candidate = {
        "text": "THE SUMMER DRIVE",
        "box": [50, 40, 650, 260],
        "confidence": 0.99,
        "method": "rapidocr",
    }
    monkeypatch.setattr(ocr, "run_ocr_image", lambda _image: [candidate])
    monkeypatch.setattr(ocr, "run_latin_ocr_image", lambda _image: [candidate])
    existing = [
        {
            "text": "THE SUMMER DRIVE",
            "box": [100, 80, 1300, 520],
            "confidence": 0.99,
            "method": "rapidocr",
        }
    ]

    assert ocr.discover_display_ocr(str(image_path), existing, scale=0.5) == []


def test_downscaled_display_ocr_falls_back_when_primary_scale_misses_hero(
    tmp_path, monkeypatch
) -> None:
    image_path = tmp_path / "hero.png"
    cv2.imwrite(str(image_path), np.full((900, 1440, 3), 255, np.uint8))

    monkeypatch.setattr(ocr, "run_ocr_image", lambda _image: [])

    def latin_candidates(image):
        if image.shape[1] != 432:
            return []
        return [
            {
                "text": "SLUSH",
                "box": [50, 13, 389, 213],
                "confidence": 0.98,
                "method": "rapidocr",
            }
        ]

    monkeypatch.setattr(ocr, "run_latin_ocr_image", latin_candidates)

    [display] = ocr.discover_display_ocr(str(image_path), [], scale=0.5)

    assert display["text"] == "SLUSH"
    assert display["displayScale"] == 0.3
    assert display["box"] == [167, 43, 1297, 710]


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
