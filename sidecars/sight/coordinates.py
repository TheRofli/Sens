"""Coordinate transforms between an analyzed image and its source image."""

from __future__ import annotations

from typing import Any


def identity_coordinates(width: int, height: int) -> dict[str, Any]:
    """Describe an analysis performed directly in source-image coordinates."""
    return {
        "sourceSize": [width, height],
        "regionInSource": [0, 0, width, height],
        "analysisSize": [width, height],
        "analysisToSource": {
            "scaleX": 1.0,
            "scaleY": 1.0,
            "offsetX": 0.0,
            "offsetY": 0.0,
        },
    }


def crop_coordinates(
    source_width: int,
    source_height: int,
    region: list[int],
    analysis_width: int,
    analysis_height: int,
) -> dict[str, Any]:
    """Describe a crop, including any resize applied before analysis."""
    x0, y0, x1, y1 = region
    return {
        "sourceSize": [source_width, source_height],
        "regionInSource": [x0, y0, x1, y1],
        "analysisSize": [analysis_width, analysis_height],
        "analysisToSource": {
            "scaleX": (x1 - x0) / analysis_width,
            "scaleY": (y1 - y0) / analysis_height,
            "offsetX": float(x0),
            "offsetY": float(y0),
        },
    }


def box_to_source(box: list[int | float], coordinates: dict[str, Any]) -> list[int]:
    """Map an analysis-space xyxy box back into the original source image."""
    transform = coordinates["analysisToSource"]
    source_width, source_height = coordinates["sourceSize"]

    def map_x(value: int | float) -> int:
        mapped = float(value) * transform["scaleX"] + transform["offsetX"]
        return max(0, min(source_width, round(mapped)))

    def map_y(value: int | float) -> int:
        mapped = float(value) * transform["scaleY"] + transform["offsetY"]
        return max(0, min(source_height, round(mapped)))

    return [map_x(box[0]), map_y(box[1]), map_x(box[2]), map_y(box[3])]
