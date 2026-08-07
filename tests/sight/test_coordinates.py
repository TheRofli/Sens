from sight import ops
from sight.coordinates import box_to_source
from tests.sight.test_storage_contract import _stub_heavy_layers, _write_image


def test_crop_analysis_carries_reversible_source_transform(tmp_path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"
    image_path = tmp_path / "source.png"
    _write_image(image_path, (20, 40, 60))
    _stub_heavy_layers(monkeypatch, cache_dir)

    dump = ops.analyze(
        str(image_path),
        region={"x": 4, "y": 6, "width": 8, "height": 4},
        no_store=True,
    )

    coordinates = dump["coordinates"]
    assert coordinates["sourceSize"] == [16, 16]
    assert coordinates["regionInSource"] == [4, 6, 12, 10]
    assert coordinates["analysisSize"] == [32, 16]
    assert box_to_source([0, 0, 32, 16], coordinates) == [4, 6, 12, 10]
