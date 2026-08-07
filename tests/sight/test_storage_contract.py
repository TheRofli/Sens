from pathlib import Path

import cv2
import numpy as np

from sight import cache as cachemod
from sight import ops


def _stub_heavy_layers(monkeypatch, cache_dir: Path) -> None:
    monkeypatch.setattr(ops, "cache_root", lambda: cache_dir)
    monkeypatch.setattr(cachemod, "cache_root", lambda: cache_dir)
    monkeypatch.setattr(ops, "run_ocr", lambda _path: [])
    monkeypatch.setattr(ops, "layout_blocks", lambda _image: [])
    monkeypatch.setattr(ops, "texture_blocks", lambda _image, _ocr: [])
    monkeypatch.setattr(ops, "attention_map", lambda _image, _ocr: {})
    monkeypatch.setattr(ops, "objects_yolo", lambda _path: [])
    monkeypatch.setattr(ops, "scene_clip", lambda _path: [])
    monkeypatch.setattr(ops, "control_style", lambda _image, _blocks: [])
    monkeypatch.setattr(ops, "control_icons", lambda _image, _blocks, _ocr: [])
    monkeypatch.setattr(
        ops,
        "build_section_tree",
        lambda _image, _background, _textures, _width, _height, _ocr, _controls: {},
    )
    monkeypatch.setattr(ops, "_assign_roles", lambda *_args: None)
    monkeypatch.setattr(ops, "build_element_tree", lambda *_args: [])
    monkeypatch.setattr(ops, "expand_button_subparts", lambda _elements: None)
    monkeypatch.setattr(ops, "summarize_screen", lambda *_args: {})
    monkeypatch.setattr(ops, "layout_skeleton", lambda _image: {})
    monkeypatch.setattr(ops, "layout_gaps", lambda _blocks: [])
    monkeypatch.setattr(ops, "design_qa", lambda _image, _blocks, _ocr: {"facts": []})
    monkeypatch.setattr(ops, "section_style", lambda _image, _blocks, _ocr: [])
    monkeypatch.setattr(ops, "shadow_bands", lambda _image, _blocks, _objects: [])
    monkeypatch.setattr(ops, "cross_verify", lambda _dump: [])


def _write_image(path: Path, bgr: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((16, 16, 3), bgr, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_no_store_creates_no_cache_or_som_artifact(tmp_path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"
    image_path = tmp_path / "input.png"
    _write_image(image_path, (0, 0, 255))
    _stub_heavy_layers(monkeypatch, cache_dir)

    dump = ops.analyze(str(image_path), no_store=True)

    assert dump["somPath"] is None
    assert not cache_dir.exists() or not any(cache_dir.rglob("*"))


def test_som_artifacts_are_content_addressed_not_basename_addressed(
    tmp_path, monkeypatch
) -> None:
    cache_dir = tmp_path / "cache"
    first = tmp_path / "a" / "screen.png"
    second = tmp_path / "b" / "screen.png"
    _write_image(first, (0, 0, 255))
    _write_image(second, (255, 0, 0))
    _stub_heavy_layers(monkeypatch, cache_dir)

    first_dump = ops.analyze(str(first))
    second_dump = ops.analyze(str(second))

    assert first_dump["somPath"] != second_dump["somPath"]
    assert Path(first_dump["somPath"]).is_file()
    assert Path(second_dump["somPath"]).is_file()
