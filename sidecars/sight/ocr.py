"""RapidOCR engine setup and image loading."""

from __future__ import annotations

import re
import os
import tempfile
from pathlib import Path
from typing import Any



# sidecars/ root (parent of the sight/ package) — keeps the RapidOCR config
# location identical to the pre-refactor single-file worker.
WORKER_DIR = Path(__file__).resolve().parent.parent


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
            "source": "inferred",
            "method": "rapidocr",
        })
    return [item for item in items if item["text"]]


def _box_iou(left: list[int], right: list[int]) -> float:
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    intersection = (x1 - x0) * (y1 - y0)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / max(1, left_area + right_area - intersection)


def _normalized_ocr_text(value: str) -> str:
    return " ".join(re.findall(r"[\w.-]+", value.casefold()))


def merge_ocr_passes(
    base_items: list[dict[str, Any]],
    scaled_items: list[dict[str, Any]],
    *,
    scale: float,
) -> list[dict[str, Any]]:
    """Merge a second OCR scale without hiding recognition disagreements."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    mapped = []
    for item in scaled_items:
        entry = dict(item)
        entry["box"] = [round(float(value) / scale) for value in item["box"]]
        mapped.append(entry)

    unused = set(range(len(mapped)))
    merged: list[dict[str, Any]] = []
    for base in base_items:
        match_index = None
        match_iou = 0.0
        for index in unused:
            score = _box_iou(base["box"], mapped[index]["box"])
            if score > match_iou:
                match_index = index
                match_iou = score
        if match_index is None or match_iou < 0.30:
            entry = dict(base)
            entry.setdefault("verified", float(entry.get("confidence") or 0.0) >= 0.90)
            merged.append(entry)
            continue

        unused.remove(match_index)
        second = mapped[match_index]
        base_confidence = float(base.get("confidence") or 0.0)
        second_confidence = float(second.get("confidence") or 0.0)
        entry = dict(base)
        alternatives = [
            {
                "text": str(base.get("text", "")),
                "confidence": round(base_confidence, 3),
                "scale": 1.0,
            },
            {
                "text": str(second.get("text", "")),
                "confidence": round(second_confidence, 3),
                "scale": scale,
            },
        ]
        if _normalized_ocr_text(str(base.get("text", ""))) == _normalized_ocr_text(
            str(second.get("text", ""))
        ):
            if second_confidence > base_confidence:
                entry["text"] = second["text"]
            entry["confidence"] = round(max(base_confidence, second_confidence), 3)
            entry["verified"] = True
            entry["method"] = "rapidocr-multiscale-consensus"
        else:
            if second_confidence >= base_confidence + 0.08:
                entry["text"] = second["text"]
                entry["confidence"] = round(second_confidence, 3)
            entry["verified"] = False
            entry["method"] = "rapidocr-multiscale-disagreement"
        entry["alternatives"] = alternatives
        merged.append(entry)

    for index in sorted(unused):
        entry = dict(mapped[index])
        entry["verified"] = False
        entry["method"] = "rapidocr-multiscale-unmatched"
        entry["alternatives"] = [
            {
                "text": str(entry.get("text", "")),
                "confidence": round(float(entry.get("confidence") or 0.0), 3),
                "scale": scale,
            }
        ]
        merged.append(entry)
    return merged


def refine_ocr_for_reconstruction(
    image_path: str,
    base_items: list[dict[str, Any]],
    *,
    scale: float = 1.5,
    max_pixels: int = 12_000_000,
) -> list[dict[str, Any]]:
    """Run one bounded larger OCR pass for exact reconstruction text."""
    import cv2

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"cannot decode image: {image_path}")
    height, width = image.shape[:2]
    bounded_scale = min(
        float(scale),
        (max(1, int(max_pixels)) / max(1, width * height)) ** 0.5,
    )
    if bounded_scale <= 1.05:
        result = [dict(item) for item in base_items]
        for item in result:
            item.setdefault(
                "verified", float(item.get("confidence") or 0.0) >= 0.90
            )
        return result

    resized = cv2.resize(
        image,
        None,
        fx=bounded_scale,
        fy=bounded_scale,
        interpolation=cv2.INTER_CUBIC,
    )
    descriptor, temporary_path = tempfile.mkstemp(
        suffix=".png", prefix="sens-ocr-reconstruct-"
    )
    os.close(descriptor)
    try:
        if not cv2.imwrite(temporary_path, resized):
            raise RuntimeError("could not write bounded OCR preview")
        scaled_items = run_ocr(temporary_path)
    finally:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
    return merge_ocr_passes(base_items, scaled_items, scale=bounded_scale)




# --------------------------------------------------------------------------
# OpenCV layers: colors + layout blocks
# --------------------------------------------------------------------------


def load_cv(image_path: str) -> Any:
    import cv2

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"cannot decode image: {image_path}")
    return image
