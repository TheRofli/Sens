import cv2
import numpy as np

from sight.ops import _plausible_controls
from sight.qa import control_style


CREAM_BGR = np.array((239, 247, 252), dtype=np.uint8)
BLUE_BGR = (255, 120, 0)


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
