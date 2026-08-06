"""RapidOCR engine setup and image loading."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any



WORKER_DIR = Path(__file__).resolve().parent


CYRILLIC_CONFIG_NAME = "rapidocr-cyrillic.yaml"



# --------------------------------------------------------------------------
# RapidOCR setup (cyrillic rec model)
# --------------------------------------------------------------------------


def ensure_ocr_config() -> str:
    target = WORKER_DIR / CYRILLIC_CONFIG_NAME
    if target.exists():
        return str(target)
    try:
        import rapidocr
    except ImportError as exc:
        raise RuntimeError(
            "Local vision requires rapidocr. Run: pip install rapidocr"
        ) from exc
    default = Path(rapidocr.__file__).parent / "config.yaml"
    text = default.read_text(encoding="utf-8")
    text = re.sub(
        r'(Rec:\n    engine_type: "onnxruntime"\n    lang_type: )"ch"',
        r'\1"cyrillic"',
        text,
    )
    text = re.sub(
        r'(Rec:\n    engine_type: "onnxruntime"\n    lang_type: "cyrillic"\n    model_type: )"small"',
        r'\1"mobile"',
        text,
    )
    text = re.sub(
        r'(Rec:\n    engine_type: "onnxruntime"\n    lang_type: "cyrillic"\n    model_type: "mobile"\n    ocr_version: )"PP-OCRv6"',
        r'\1"PP-OCRv4"',
        text,
    )
    target.write_text(text, encoding="utf-8")
    return str(target)




_ocr_engine: Any = None




def ocr_engine() -> Any:
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr import RapidOCR

        _ocr_engine = RapidOCR(config_path=ensure_ocr_config())
    return _ocr_engine




def run_ocr(image_path: str) -> list[dict[str, Any]]:
    out = ocr_engine()(image_path)
    items: list[dict[str, Any]] = []
    boxes = out.boxes if out.boxes is not None else []
    txts = out.txts if out.txts is not None else []
    scores = out.scores if out.scores is not None else []
    for box, txt, score in zip(boxes, txts, scores):
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        items.append({
            "text": str(txt).strip(),
            "box": [round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys))],
            "confidence": round(float(score), 3),
        })
    return [item for item in items if item["text"]]




# --------------------------------------------------------------------------
# OpenCV layers: colors + layout blocks
# --------------------------------------------------------------------------


def load_cv(image_path: str) -> Any:
    import cv2

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"cannot decode image: {image_path}")
    return image
