import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

from sight import vlm
from sight import server
from sight.vlm import PACKS


def _downloader_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "download-vision-models.py"
    spec = importlib.util.spec_from_file_location("download_vision_models", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _benchmark_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "benchmark-vision-models.py"
    spec = importlib.util.spec_from_file_location("benchmark_vision_models", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_pack_is_official_qwen3_vl_with_verified_files() -> None:
    pack = PACKS["lite"]

    assert pack["repo"] == "Qwen/Qwen3-VL-2B-Instruct-GGUF"
    assert pack["license"] == "Apache-2.0"
    assert pack["text"] == "Qwen3VL-2B-Instruct-Q4_K_M.gguf"
    assert pack["mmproj"] == "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"
    assert pack["sha256"]["text"] == "089d75c52f4b7ffc56ba998ffc50aae89fcafc755f9e7208aacca281dca6c2ae"
    assert pack["sha256"]["mmproj"] == "f9a68fabba69c3b81e153367b2c7521030b0fa8bb0de400c9599c8e6725f9c82"


def test_downloader_removes_a_file_that_fails_hash_verification(tmp_path, monkeypatch) -> None:
    downloader = _downloader_module()
    destination = tmp_path / "model.gguf"
    monkeypatch.setattr(
        downloader.urllib.request,
        "urlretrieve",
        lambda _url, path: Path(path).write_bytes(b"GGUF-corrupt"),
    )

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        downloader._fetch("https://example.invalid/model", destination, "0" * 64)

    assert not destination.exists()


def test_vlm_host_disables_gpu_and_keeps_only_one_model_loaded(tmp_path, monkeypatch) -> None:
    for pack_name in ("lite", "quality"):
        pack = PACKS[pack_name]
        (tmp_path / pack["text"]).write_bytes(b"GGUF")
        (tmp_path / pack["mmproj"]).write_bytes(b"GGUF")
    constructed = []

    class FakeLlama:
        def __init__(self, **kwargs):
            constructed.append(kwargs)

    class FakeHandler:
        def __init__(self, clip_model_path, use_gpu, verbose=False):
            self.clip_model_path = clip_model_path
            self.use_gpu = use_gpu
            self.verbose = verbose

    monkeypatch.setenv("SENS_MODELS_ROOT", str(tmp_path))
    monkeypatch.setattr(
        vlm,
        "_import_llama",
        lambda: type(
            "FakeModule",
            (),
            {
                "Llama": FakeLlama,
                "llama_chat_format": type(
                    "FakeChatFormat", (), {"MTMDChatHandler": FakeHandler}
                ),
            },
        ),
    )
    monkeypatch.setattr(vlm, "_ACTIVE_HOST", None, raising=False)
    first = vlm.VlmHost("lite")
    second = vlm.VlmHost("quality")

    first._load()
    second._load()

    assert all(call["n_gpu_layers"] == 0 for call in constructed)
    assert all(call["chat_handler"].use_gpu is False for call in constructed)
    assert all("mmproj_path" not in call for call in constructed)
    assert first._llm is None
    assert second._llm is not None


def test_worker_does_not_preload_optional_vlm_without_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("SENS_VLM_PRELOAD", raising=False)

    assert server._warm_vlm_async() is False


def test_multimodal_message_uses_model_card_content_order(tmp_path, monkeypatch) -> None:
    image = tmp_path / "fixture.png"
    cv2.imwrite(str(image), np.zeros((16, 16, 3), np.uint8))
    captured = {}

    class FakeModel:
        def create_chat_completion(self, **kwargs):
            captured.update(kwargs)
            return {"choices": [{"message": {"content": "ok"}}]}

    host = vlm.VlmHost("lite")
    host._llm = FakeModel()
    monkeypatch.setattr(host, "_touch", lambda: None)

    assert host.ask(str(image), "What is shown?") == "ok"
    content = captured["messages"][0]["content"]
    assert [item["type"] for item in content] == ["text", "image_url"]


def test_benchmark_required_groups_score_independent_visual_facts() -> None:
    benchmark = _benchmark_module()
    case = {
        "metric": "required_groups",
        "required_groups": [["creative coding", "illustration"], ["mono x7", "canvas"]],
    }

    assert benchmark.score_case(case, "Illustration surrounds the MONO X7 canvas") == 1.0
    assert benchmark.score_case(case, "Illustration only") == 0.5


def test_vlm_preview_bounds_large_images_before_inference(tmp_path) -> None:
    source = tmp_path / "large.png"
    cv2.imwrite(str(source), np.zeros((1200, 2400, 3), np.uint8))
    host = vlm.VlmHost("lite", max_pixels=400_000)

    prepared, temporary = host._prepare_image(str(source), None)
    try:
        image = cv2.imread(prepared)
        assert temporary is True
        assert image.shape[0] * image.shape[1] <= 400_000
        assert image.shape[1] / image.shape[0] == pytest.approx(2.0, rel=0.01)
    finally:
        Path(prepared).unlink(missing_ok=True)
