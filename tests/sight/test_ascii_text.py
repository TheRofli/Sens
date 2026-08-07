from sight.ascii_text import reconstruct_monospace


def test_reconstructs_fixed_grid_with_whitespace_and_blank_lines() -> None:
    result = reconstruct_monospace(
        [
            {"text": "/\\", "box": [2, 0, 6, 2], "confidence": 0.99},
            {"text": "||", "box": [2, 2, 6, 4], "confidence": 0.98},
        ],
        image_width=10,
        image_height=6,
        cell_width=2,
        line_height=2,
    )

    assert result["text"] == " /\\  \n ||  \n     "
    assert result["grid"] == {"columns": 5, "rows": 3, "cellWidth": 2.0, "lineHeight": 2.0}
    assert result["ambiguities"] == []


def test_low_confidence_characters_are_exposed_as_ambiguous() -> None:
    result = reconstruct_monospace(
        [{"text": "<> ", "box": [0, 0, 6, 2], "confidence": 0.4}],
        image_width=6,
        image_height=2,
        cell_width=2,
        line_height=2,
    )

    assert result["text"] == "?? "
    assert result["ambiguities"] == [{"row": 0, "columns": [0, 2], "observed": "<>"}]
