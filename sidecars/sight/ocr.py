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
_latin_ocr_engine: Any = None




def ocr_engine() -> Any:
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr import RapidOCR

        _ocr_engine = RapidOCR(config_path=ensure_ocr_config())
    return _ocr_engine


def latin_ocr_engine() -> Any:
    """Default PP-OCR recognizer used only on bounded high-resolution crops."""
    global _latin_ocr_engine
    if _latin_ocr_engine is None:
        from rapidocr import RapidOCR

        _latin_ocr_engine = RapidOCR()
    return _latin_ocr_engine




def _run_with_engine(engine: Any, image_input: Any) -> list[dict[str, Any]]:
    out = engine(image_input)
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


def run_ocr(image_path: str) -> list[dict[str, Any]]:
    return _run_with_engine(ocr_engine(), image_path)


def run_latin_ocr(image_path: str) -> list[dict[str, Any]]:
    return _run_with_engine(latin_ocr_engine(), image_path)


def run_latin_ocr_image(image: Any) -> list[dict[str, Any]]:
    """Run the portable Latin recognizer on an already-bounded image array."""
    return _run_with_engine(latin_ocr_engine(), image)


def _box_iou(left: list[int], right: list[int]) -> float:
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    intersection = (x1 - x0) * (y1 - y0)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / max(1, left_area + right_area - intersection)


def _box_match_score(left: list[int], right: list[int]) -> float:
    """Match OCR polygons even when one recognizer sees a longer full line."""
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    intersection = (x1 - x0) * (y1 - y0)
    left_area = max(1, (left[2] - left[0]) * (left[3] - left[1]))
    right_area = max(1, (right[2] - right[0]) * (right[3] - right[1]))
    containment = intersection / max(1, min(left_area, right_area))
    return max(_box_iou(left, right), 0.82 * containment)


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


_CURRENCY_SIGILS = "$€£¥₽₩₹₿"


def _visible_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def _visible_sigil_correction(left: str, right: str) -> str | None:
    """Return the candidate that preserves a leading measured currency sigil."""
    for sigil_candidate, letter_candidate in ((left, right), (right, left)):
        if (
            len(sigil_candidate) >= 3
            and len(sigil_candidate) == len(letter_candidate)
            and sigil_candidate[0] in _CURRENCY_SIGILS
            and letter_candidate[0].casefold() in {"s", "5"}
            and sigil_candidate[1:].casefold() == letter_candidate[1:].casefold()
        ):
            return sigil_candidate
    return None


def _portable_latin_candidate(value: Any) -> bool:
    text = str(value or "").strip()
    alphanumeric = [character for character in text if character.isalnum()]
    if not alphanumeric:
        return False
    portable = sum(character.isascii() for character in alphanumeric)
    return portable / len(alphanumeric) >= 0.75


def merge_script_ocr_passes(
    base_items: list[dict[str, Any]],
    latin_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fuse Cyrillic-capable and Latin recognizers on one bounded crop."""
    unused = set(range(len(latin_items)))
    merged: list[dict[str, Any]] = []
    for base in base_items:
        match_index = None
        match_iou = 0.0
        for index in unused:
            score = _box_match_score(base["box"], latin_items[index]["box"])
            if score > match_iou:
                match_index = index
                match_iou = score
        if match_index is None or match_iou < 0.30:
            merged.append(dict(base))
            continue
        unused.remove(match_index)
        latin = latin_items[match_index]
        base_text = str(base.get("text") or "")
        latin_text = str(latin.get("text") or "")
        base_confidence = float(base.get("confidence") or 0.0)
        latin_confidence = float(latin.get("confidence") or 0.0)
        alternatives = [
            {
                "text": base_text,
                "confidence": round(base_confidence, 3),
                "script": "cyrillic-capable",
            },
            {
                "text": latin_text,
                "confidence": round(latin_confidence, 3),
                "script": "latin",
            },
        ]
        entry = dict(base)
        sigil = _visible_sigil_correction(base_text, latin_text)
        if sigil is not None and (
            (latin_text == sigil and latin_confidence >= base_confidence - 0.12)
            or (base_text == sigil and base_confidence >= latin_confidence - 0.12)
        ):
            entry["text"] = sigil
            entry["confidence"] = round(
                max(base_confidence, latin_confidence), 3
            )
            entry["verified"] = True
            entry["method"] = "rapidocr-dual-script-visible-sigil-consensus"
        elif _visible_text(base_text) == _visible_text(latin_text):
            if latin_confidence > base_confidence:
                entry["text"] = latin_text
            entry["confidence"] = round(
                max(base_confidence, latin_confidence), 3
            )
            entry["verified"] = True
            entry["method"] = "rapidocr-dual-script-consensus"
        else:
            base_compact = "".join(
                character.casefold()
                for character in base_text
                if character.isalnum()
            )
            latin_compact = "".join(
                character.casefold()
                for character in latin_text
                if character.isalnum()
            )
            base_separators = sum(
                character.isspace() or character in "-.,:/" for character in base_text
            )
            latin_separators = sum(
                character.isspace() or character in "-.,:/" for character in latin_text
            )
            same_visible_glyphs = bool(
                base_compact and base_compact == latin_compact
            )
            similarity = __import__("difflib").SequenceMatcher(
                None, _visible_text(base_text), _visible_text(latin_text)
            ).ratio()
            portable_pair = _portable_latin_candidate(
                base_text
            ) and _portable_latin_candidate(latin_text)
            prefer_latin = bool(
                portable_pair
                and latin_confidence >= 0.90
                and (
                    latin_confidence >= base_confidence + 0.025
                    or base_confidence < 0.80
                )
                and (similarity >= 0.62 or latin_confidence >= base_confidence + 0.12)
            )
            # If both recognizers saw the same glyph sequence, retain the
            # reading with measured word/punctuation boundaries.  This keeps
            # "THE SUMMER" over "THESUMMER" even when both scores round to 1.
            if same_visible_glyphs:
                if latin_separators > base_separators:
                    prefer_latin = True
                elif latin_separators < base_separators:
                    prefer_latin = False
            if prefer_latin:
                entry["text"] = latin_text
                entry["confidence"] = round(latin_confidence, 3)
                entry["verified"] = latin_confidence >= 0.90
                entry["method"] = "rapidocr-dual-script-latin-preferred"
            else:
                entry["verified"] = False
                entry["method"] = "rapidocr-dual-script-disagreement"
        entry["alternatives"] = alternatives
        merged.append(entry)

    for index in sorted(unused):
        entry = dict(latin_items[index])
        if (
            float(entry.get("confidence") or 0.0) < 0.80
            or not _portable_latin_candidate(entry.get("text"))
        ):
            continue
        entry["verified"] = False
        entry["method"] = "rapidocr-latin-bounded-unmatched"
        entry["alternatives"] = [
            {
                "text": str(entry.get("text") or ""),
                "confidence": round(float(entry.get("confidence") or 0.0), 3),
                "script": "latin",
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
