import numpy as np

from sight.ascii_map import CHARS, render_ascii


def _gradient() -> np.ndarray:
    img = np.zeros((480, 960, 3), np.uint8)
    for y in range(480):
        img[y, :] = int(255 * y / 479)
    return img


def test_shape() -> None:
    lines = render_ascii(_gradient(), cols=96, rows=48).split("\n")
    assert len(lines) == 48
    assert all(len(line) == 96 for line in lines)


def test_gradient_direction() -> None:
    lines = render_ascii(_gradient(), cols=96, rows=48).split("\n")
    assert lines[0].strip(" ") == ""                      # чёрный верх -> пробелы
    assert lines[-1].count(CHARS[-1]) > 50                # белый низ -> плотный символ


def test_uniform_single_char() -> None:
    img = np.full((100, 100, 3), 128, np.uint8)
    lines = render_ascii(img, cols=10, rows=4).split("\n")
    chars = {c for line in lines for c in line}
    assert len(chars) == 1
