import cv2
import numpy as np

from sight.symbol_art import detect_symbol_art


def _draw_symbol_grid(lines: list[str]) -> np.ndarray:
    image = np.zeros((260, 420, 3), np.uint8)
    origin_x = 42
    first_baseline = 55
    cell_width = 14
    row_pitch = 22
    for row, line in enumerate(lines):
        baseline = first_baseline + row * row_pitch
        for column, character in enumerate(line):
            center_x = origin_x + column * cell_width
            if character == ".":
                cv2.rectangle(
                    image,
                    (center_x - 1, baseline - 1),
                    (center_x + 1, baseline + 1),
                    (255, 255, 255),
                    -1,
                )
            elif character == "◆":
                cv2.fillConvexPoly(
                    image,
                    np.array(
                        [
                            [center_x, baseline - 13],
                            [center_x + 6, baseline - 7],
                            [center_x, baseline - 1],
                            [center_x - 6, baseline - 7],
                        ],
                        np.int32,
                    ),
                    (255, 255, 255),
                )
    return image


def test_repeated_dot_and_diamond_grid_is_reconstructed_as_text() -> None:
    lines = [
        "  ....      ",
        " ..◆◆..     ",
        ".◆◆◆◆◆.     ",
        "◆◆..◆◆..    ",
        "◆.◆◆.◆..    ",
        ".◆◆..◆◆.    ",
        " ..◆◆..     ",
        "  ....      ",
    ]

    [art] = detect_symbol_art(_draw_symbol_grid(lines))

    reconstructed = art["text"].splitlines()
    assert len(reconstructed) == len(lines)
    assert [line.rstrip() for line in reconstructed] == [
        line.rstrip() for line in lines
    ]
    assert art["alphabet"] == [".", "◆"]
    assert abs(art["cellWidth"] - 14) <= 1
    assert abs(art["rowPitch"] - 22) <= 1
    assert art["strategy"] == "render-as-live-selectable-monospace-text"
    assert art["source"] == "measured"


def test_normal_headline_is_not_symbol_art() -> None:
    image = np.zeros((260, 420, 3), np.uint8)
    cv2.putText(
        image,
        "WORLD CLASS",
        (30, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )

    assert detect_symbol_art(image) == []


def test_fractional_symbol_grid_pitch_and_glyph_geometry_are_preserved() -> None:
    image = np.zeros((280, 460, 3), np.uint8)
    origin_x = 17.3
    first_baseline = 43.6
    cell_width = 13.4
    row_pitch = 21.6
    lines = [
        "..◆◆..        ",
        ".◆◆◆◆.        ",
        "◆◆..◆◆        ",
        "◆.◆◆.◆        ",
        ".◆◆..◆◆.      ",
        "..◆◆◆◆..      ",
        " ◆◆..◆◆       ",
        "  ◆◆◆◆        ",
    ]
    for row, line in enumerate(lines):
        baseline = first_baseline + row * row_pitch
        for column, character in enumerate(line):
            center_x = origin_x + column * cell_width
            if character == ".":
                cv2.rectangle(
                    image,
                    (round(center_x - 2), round(baseline - 2)),
                    (round(center_x + 2), round(baseline + 2)),
                    (255, 255, 255),
                    -1,
                )
            elif character == "◆":
                cv2.fillConvexPoly(
                    image,
                    np.array(
                        [
                            [round(center_x), round(baseline - 12)],
                            [round(center_x + 6), round(baseline - 6)],
                            [round(center_x), round(baseline)],
                            [round(center_x - 6), round(baseline - 6)],
                        ],
                        np.int32,
                    ),
                    (255, 255, 255),
                )

    [art] = detect_symbol_art(image)

    assert abs(art["cellWidth"] - cell_width) <= 0.2
    assert abs(art["rowPitch"] - row_pitch) <= 0.2
    assert abs(art["firstBaselineY"] - first_baseline) <= 1.0
    assert abs(art["firstCellCenterX"] - origin_x) <= 1.0
    assert art["glyphGeometry"]["dot"]["height"] >= 4
    assert art["glyphGeometry"]["diamond"]["height"] >= 10
