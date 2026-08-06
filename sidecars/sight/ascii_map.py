"""Luminance ASCII composition map for the visual context document."""
from __future__ import annotations

from typing import Any

import cv2

CHARS = " .:-=+*#%@"


def render_ascii(image: Any, cols: int = 96, rows: int = 48) -> str:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (cols, rows), interpolation=cv2.INTER_AREA)
    lines = []
    for y in range(rows):
        lines.append(
            "".join(
                CHARS[min(len(CHARS) - 1, int(small[y, x]) * len(CHARS) // 256)]
                for x in range(cols)
            )
        )
    return "\n".join(lines)
