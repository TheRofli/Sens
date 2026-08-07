import numpy as np
import cv2

from sight.capture import motion_events


def _frame(path, x):
    img = np.zeros((300, 300), np.uint8)
    img[100:160, x:x + 60] = 255
    cv2.imwrite(str(path), img)


def test_moving_square_event(tmp_path):
    paths = [tmp_path / f"f{i}.png" for i in range(3)]
    _frame(paths[0], 100)
    _frame(paths[1], 200)
    _frame(paths[2], 300 - 61)
    events = motion_events([str(p) for p in paths])
    assert events, "expected motion events"
    assert any(e["dx"] is not None and 80 <= e["dx"] <= 120 for e in events)
