"""Lazy local VLM host (llama-cpp GGUF, CPU-only). Packs: lite (default) / quality."""
from __future__ import annotations

import base64
import math
import os
import tempfile
import threading
from pathlib import Path

import cv2

# Serializes native model construction across hosts/threads: the boot-time warm
# thread and a request handler must not build two GGUF models concurrently.
_LOAD_LOCK = threading.Lock()
_ACTIVE_HOST = None

# Repo layouts verified against HF API on 2026-08-07.
# Measured on the 2026-08-07 CPU benchmark: lite peaked near 3.4 GiB and
# quality near 3.2 GiB for the complete process. quality_large is intentionally
# opt-in and has not yet passed the same acceptance benchmark.
PACKS = {
    "lite": {
        "repo": "Qwen/Qwen3-VL-2B-Instruct-GGUF",
        "text": "Qwen3VL-2B-Instruct-Q4_K_M.gguf",
        "mmproj": "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf",
        "license": "Apache-2.0",
        "sha256": {
            "text": "089d75c52f4b7ffc56ba998ffc50aae89fcafc755f9e7208aacca281dca6c2ae",
            "mmproj": "f9a68fabba69c3b81e153367b2c7521030b0fa8bb0de400c9599c8e6725f9c82",
        },
        "bytes": {"text": 1107409952, "mmproj": 445053216},
    },
    "quality": {
        "repo": "ggml-org/SmolVLM2-2.2B-Instruct-GGUF",
        "text": "SmolVLM2-2.2B-Instruct-Q4_K_M.gguf",
        "mmproj": "mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf",
        "license": "Apache-2.0",
        "sha256": {
            "text": "0cf76814555b8665149075b74ab6b5c1d428ea1d3d01c1918c12012e8d7c9f58",
            "mmproj": "ae07ea1facd07dd3230c4483b63e8cda96c6944ad2481f33d531f79e892dd024",
        },
        "bytes": {"text": 1112602656, "mmproj": 592523200},
    },
    "quality_large": {
        "repo": "unsloth/Qwen2.5-VL-3B-Instruct-GGUF",
        "text": "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
        "mmproj": "mmproj-F16.gguf",
        "license": "Apache-2.0 (upstream Qwen2.5-VL)",
        "sha256": {
            "text": "c47e8c1f6fb3e8cff6ec58909baff16dbeffb64a5bb3b746b96e05e6334c129f",
            "mmproj": "4c1240f514de94c81b70709b0f9a80c7e3297598ea7c83f39dc00b18ee5be60c",
        },
        "bytes": {"text": 1929901408, "mmproj": 1338428256},
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
        # MTMD is lazily imported by the chat handler; preload it while stale
        # CUDA/HIP variables are still hidden so the CPU DLL can be selected.
        import llama_cpp.llama_chat_format  # noqa: F401
        import llama_cpp.mtmd_cpp  # noqa: F401
    finally:
        os.environ.update(saved)
    return llama_cpp


class VlmHost:
    def __init__(
        self,
        pack: str | None = None,
        idle_seconds: float = 600.0,
        max_pixels: int | None = None,
    ):
        self.pack = pack or os.environ.get("SENS_VISION_PACK", "lite")
        self._llm = None
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()
        self._idle = idle_seconds
        self._max_pixels = max_pixels or int(
            os.environ.get("SENS_VLM_MAX_PIXELS", "512000")
        )

    def available(self) -> bool:
        spec = PACKS[self.pack]
        root = models_root()
        return (root / spec["text"]).exists() and (root / spec["mmproj"]).exists()

    def _load(self) -> None:
        global _ACTIVE_HOST
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
                if _ACTIVE_HOST is not None and _ACTIVE_HOST is not self:
                    _ACTIVE_HOST.unload()
                llama_cpp = _import_llama()
                Llama = llama_cpp.Llama
                spec = PACKS[self.pack]
                root = models_root()
                chat_handler = llama_cpp.llama_chat_format.MTMDChatHandler(
                    clip_model_path=str(root / spec["mmproj"]),
                    verbose=False,
                    use_gpu=False,
                )
                self._llm = Llama(
                    model_path=str(root / spec["text"]),
                    chat_handler=chat_handler,
                    n_threads=os.cpu_count() or 4,
                    n_ctx=4096,
                    n_gpu_layers=0,
                    verbose=False,
                )
                _ACTIVE_HOST = self

    def _touch(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(self._idle, self.unload)
        self._timer.daemon = True
        self._timer.start()

    def unload(self) -> None:
        global _ACTIVE_HOST
        with self._lock:
            self._llm = None
            if _ACTIVE_HOST is self:
                _ACTIVE_HOST = None

    def _prepare_image(
        self, image_path: str, box: list[int] | None
    ) -> tuple[str, bool]:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"cannot decode image: {image_path}")
        changed = False
        if box:
            height, width = img.shape[:2]
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, x2 = max(0, x1), min(width, x2)
            y1, y2 = max(0, y1), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                raise ValueError("VLM region does not intersect the image")
            img = img[y1:y2, x1:x2]
            changed = True

        height, width = img.shape[:2]
        pixels = width * height
        scale = 1.0
        if pixels > self._max_pixels:
            scale = math.sqrt(self._max_pixels / pixels)
        elif min(width, height) < 256:
            scale = min(
                4.0,
                512.0 / min(width, height),
                math.sqrt(self._max_pixels / pixels),
            )
        if abs(scale - 1.0) > 0.001:
            target_width = max(1, math.floor(width * scale))
            target_height = max(1, math.floor(height * scale))
            interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
            img = cv2.resize(img, (target_width, target_height), interpolation=interpolation)
            changed = True
        if not changed:
            return image_path, False
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        cv2.imwrite(tmp.name, img)
        return tmp.name, True

    def _chat(self, image_path: str, prompt: str, box: list[int] | None = None) -> str:
        self._load()
        self._touch()
        path, temporary = self._prepare_image(image_path, box)
        try:
            res = self._llm.create_chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": _data_uri(path)}},
                        ],
                    }
                ],
                max_tokens=300,
                temperature=0.2,
            )
            return str(res["choices"][0]["message"]["content"]).strip()
        finally:
            if temporary:
                Path(path).unlink(missing_ok=True)

    def vibe(self, image_path: str) -> str:
        return self._chat(image_path, VIBE_PROMPT)

    def describe(self, image_path: str, box: list[int]) -> str:
        return self._chat(image_path, DESCRIBE_PROMPT, box)

    def transcribe(self, image_path: str, box: list[int]) -> str:
        return self._chat(image_path, TRANSCRIBE_PROMPT, box)

    def ask(self, image_path: str, question: str, box: list[int] | None = None) -> str:
        return self._chat(image_path, question, box)
