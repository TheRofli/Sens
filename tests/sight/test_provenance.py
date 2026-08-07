import sys
from types import SimpleNamespace

from sight import ocr, perception


def test_ocr_results_are_identified_as_inference(monkeypatch) -> None:
    output = SimpleNamespace(
        boxes=[[[1, 2], [11, 2], [11, 8], [1, 8]]],
        txts=["Sens"],
        scores=[0.97],
    )
    monkeypatch.setattr(ocr, "_ocr_engine", lambda _path: output)

    [item] = ocr.run_ocr("fixture.png")

    assert item["source"] == "inferred"
    assert item["method"] == "rapidocr"


def test_object_results_are_identified_as_inference(monkeypatch) -> None:
    result = SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=[[2.0, 3.0, 12.0, 13.0]],
            cls=[0],
            conf=[0.83],
        ),
        names={0: "button"},
    )
    monkeypatch.setattr(perception, "_yolo", lambda _path, verbose=False: [result])

    [item] = perception.objects_yolo("fixture.png")

    assert item["source"] == "inferred"
    assert item["method"] == "yolov8n"


def test_scene_results_are_identified_as_inference(tmp_path, monkeypatch) -> None:
    class NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return False

    class Scores:
        def softmax(self, **_kwargs):
            return self

        def __getitem__(self, _index):
            return self

        def tolist(self):
            return [0.9] + [0.1] * (len(perception.SCENE_CANDIDATES) - 1)

    class Features:
        @property
        def T(self):
            return self

        def float(self):
            return self

        def norm(self, **_kwargs):
            return 1.0

        def __itruediv__(self, _other):
            return self

        def __rmul__(self, _other):
            return self

        def __matmul__(self, _other):
            return Scores()

    class ImageInput:
        def unsqueeze(self, _axis):
            return self

    image_path = tmp_path / "fixture.png"
    image_path.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de"
            "0000000c4944415408d763f8cfc000000301010018dd8db10000000049454e44ae426082"
        )
    )
    fake_clip = SimpleNamespace(
        encode_image=lambda _input: Features(),
        encode_text=lambda _tokens: Features(),
    )
    monkeypatch.setattr(perception, "_clip", fake_clip)
    monkeypatch.setattr(perception, "_clip_preprocess", lambda _image: ImageInput())
    monkeypatch.setattr(perception, "_clip_tokenizer", lambda _labels: object())
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(no_grad=NoGrad))

    [item, *_] = perception.scene_clip(str(image_path))

    assert item["source"] == "inferred"
    assert item["method"] == "clip-vit-b-32"
