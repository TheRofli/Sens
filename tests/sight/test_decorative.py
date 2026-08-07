import math

from sight.decorative import detect_circular, detect_vertical


def _ring(n: int = 12, cx: float = 500, cy: float = 500, r: float = 200):
    items = []
    for i in range(n):
        a = 2 * math.pi * i / n
        x, y = cx + r * math.cos(a), cy + r * math.sin(a)
        items.append({"id": i + 1, "box": [int(x - 15), int(y - 10), int(x + 15), int(y + 10)]})
    return items


def test_circular_detected_and_ordered() -> None:
    groups = detect_circular(_ring())
    assert len(groups) == 1
    g = groups[0]
    assert g["direction"] == "circular"
    assert sorted(g["ids"]) == list(range(1, 13))
    assert abs(g["radius"] - 200) < 30


def test_circular_ignores_grid() -> None:
    items = [
        {"id": i + 1, "box": [x, y, x + 80, y + 20]}
        for i, y in enumerate(range(0, 400, 40))
        for x in range(0, 400, 120)
    ]
    assert detect_circular(items) == []


def test_vertical_column() -> None:
    items = [
        {"id": 1, "box": [700, 100, 740, 300]},
        {"id": 2, "box": [700, 310, 740, 510]},
        {"id": 3, "box": [700, 520, 740, 720]},
        {"id": 4, "box": [100, 100, 300, 130]},  # горизонтальный — не в группу
    ]
    groups = detect_vertical(items)
    assert len(groups) == 1
    assert groups[0]["ids"] == [1, 2, 3]
