"""Lazy local VLM host (llama-cpp GGUF, CPU-only). Packs: lite (default) / quality."""
from __future__ import annotations

import base64
import os
import tempfile
import threading
from pathlib import Path

import cv2

# Serializes native model construction across hosts/threads: the boot-time warm
# thread and a request handler must not build two GGUF models concurrently.
_LOAD_LOCK = threading.Lock()

# Repo layouts verified against HF API on 2026-08-07.
# lite    ~0.7 GB RAM — deterministic-plus semantics on a budget;
# quality ~2.0 GB RAM — SmolVLM2-2.2B Q4_K_M (newer/stronger than moondream2);
# quality_large ~3.5 GB RAM — Qwen2.5-VL-3B, best quality (esp. OCR), opt-in.
PACKS = {
    "lite": {
        "repo": "jc-builds/smolvlm2-500m-gguf",
        "text": "SmolVLM2-500M-Video-Instruct-Q8_0.gguf",
        "mmproj": "mmproj-SmolVLM2-500M-Video-Instruct-f16.gguf",
    },
    "quality": {
        "repo": "ggml-org/SmolVLM2-2.2B-Instruct-GGUF",
        "text": "SmolVLM2-2.2B-Instruct-Q4_K_M.gguf",
        "mmproj": "mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf",
    },
    "quality_large": {
        "repo": "unsloth/Qwen2.5-VL-3B-Instruct-GGUF",
        "text": "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
        "mmproj": "mmproj-F16.gguf",
    },
}

VIBE_PROMPT = (
    "Describe the visual style of this screen in 2-3 sentences: design movement "
    "(brutalism/retro/minimalism/swiss...), palette mood, typography character, overall vibe."
)
DESCRIBE_PROMPT = (
    "Describe this graphic region in 1-2 sentences: kind of graphic "
    "(photo/illustration/dot-matrix pattern/ascii art/3d render) and what it depicts."
)
TRANSCRIBE_PROMPT = "Transcribe ALL text visible in this image exactly as written, in reading order. Output only the text."

VISION_PROMPT = {
    "ru": (
        "У тебя есть зрение через Sens. sens_see возвращает документ экрана: палитра, "
        "типографика, сетка, элементы с номерами [id] и координатами 0–1000, графика с "
        "подписями, ascii-карта композиции, измерения (факты без оценок). Детали запрашивай "
        "sens_zoom(регион|[id]), вопросы про регион — sens_ask, метрики элемента — sens_element, "
        "моушен сайта — sens_motion(url). Ссылайся на элементы по номерам [id]."
    ),
    "en": (
        "You have vision via Sens. sens_see returns a screen document: palette, typography, grid, "
        "elements with [id] and 0–1000 coords, captioned graphics, ascii composition map, "
        "measurements (facts, not judgments). For detail use sens_zoom(region|[id]), region "
        "questions — sens_ask, element metrics — sens_element, site motion — sens_motion(url). "
        "Reference elements by [id]."
    ),
}


def models_root() -> Path:
    return Path(os.environ.get("SENS_MODELS_ROOT", str(Path(__file__).resolve().parents[2] / "models")))


def _data_uri(path: str) -> str:
    raw = Path(path).read_bytes()
    return "data:image/png;base64," + base64.b64encode(raw).decode()


def _import_llama():
    # llama_cpp's Windows loader calls os.add_dll_directory on CUDA_PATH/{bin,lib}
    # and HIP_PATH/{bin,lib} and raises FileNotFoundError if any is missing
    # (stale/partial toolkit installs). Sens is CPU-only, so drop entries whose
    # subdirs are incomplete for the import, then restore them.
    saved = {}
    for key in ("CUDA_PATH", "HIP_PATH"):
        val = os.environ.get(key)
        if val and not all((Path(val) / sub).exists() for sub in ("bin", "lib")):
            saved[key] = os.environ.pop(key)
    try:
        import llama_cpp
    finally:
        os.environ.update(saved)
    return llama_cpp


class VlmHost:
    def __init__(self, pack: str | None = None, idle_seconds: float = 600.0):
        self.pack = pack or os.environ.get("SENS_VISION_PACK", "lite")
        self._llm = None
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()
        self._idle = idle_seconds

    def available(self) -> bool:
        spec = PACKS[self.pack]
        root = models_root()
        return (root / spec["text"]).exists() and (root / spec["mmproj"]).exists()

    def _load(self) -> None:
        with self._lock:
            if self._llm is not None:
                return
            if not self.available():
                raise RuntimeError(
                    "vision models not downloaded; run scripts/download-vision-models.py"
                )
            with _LOAD_LOCK:
                if self._llm is not None:
                    return
                Llama = _import_llama().Llama
                spec = PACKS[self.pack]
                root = models_root()
                self._llm = Llama(
                    model_path=str(root / spec["text"]),
                    mmproj_path=str(root / spec["mmproj"]),
                    n_threads=os.cpu_count() or 4,
                    n_ctx=4096,
                    verbose=False,
                )

    def _touch(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(self._idle, self.unload)
        self._timer.daemon = True
        self._timer.start()

    def unload(self) -> None:
        with self._lock:
            self._llm = None

    def _crop(self, image_path: str, box: list[int]) -> str:
        img = cv2.imread(image_path)
        x1, y1, x2, y2 = [int(v) for v in box]
        crop = img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(tmp.name, crop)
        return tmp.name

    def _chat(self, image_path: str, prompt: str, box: list[int] | None = None) -> str:
        self._load()
        self._touch()
        path = self._crop(image_path, box) if box else image_path
        try:
            res = self._llm.create_chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": _data_uri(path)}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_tokens=300,
                temperature=0.2,
            )
            return str(res["choices"][0]["message"]["content"]).strip()
        finally:
            if box:
                Path(path).unlink(missing_ok=True)

    def vibe(self, image_path: str) -> str:
        return self._chat(image_path, VIBE_PROMPT)

    def describe(self, image_path: str, box: list[int]) -> str:
        return self._chat(image_path, DESCRIBE_PROMPT, box)

    def transcribe(self, image_path: str, box: list[int]) -> str:
        return self._chat(image_path, TRANSCRIBE_PROMPT, box)

    def ask(self, image_path: str, question: str, box: list[int] | None = None) -> str:
        return self._chat(image_path, question, box)
