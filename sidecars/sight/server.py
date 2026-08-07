"""NDJSON protocol: handle() and the stdin loop."""

from __future__ import annotations

import json
import os
import sys

from sight.compare import compare_images
from sight.ops import (
    analyze,
    ask,
    capture_op,
    element,
    inspect_target,
    locate_text,
    motion_op,
    see_document,
    vision_prompt,
    warm,
    zoom,
)




# --------------------------------------------------------------------------
# NDJSON protocol loop (same shape as the other sidecar workers)
# --------------------------------------------------------------------------


def handle(message: dict[str, object]) -> dict[str, object]:
    operation = str(message.get("operation", ""))
    payload = message.get("input") or {}
    no_store = bool(message.get("noStore", False))
    if not isinstance(payload, dict):
        raise ValueError("Sight input must be an object")
    if operation == "see":
        max_semantic_calls = max(0, min(4, int(message.get("maxCalls") or 2)))
        return see_document(
            str(payload["imagePath"]),
            payload.get("region"),
            no_store,
            bool(payload.get("fast", False)),
            bool(payload.get("quality", False)),
            payload.get("pack"),
            payload.get("prompt"),
            max_semantic_calls,
        )
    if operation == "read":
        dump = analyze(str(payload["imagePath"]), payload.get("region"), no_store)
        return {"texts": [item["text"] for item in dump["ocr"]], "ocr": dump["ocr"]}
    if operation == "locate":
        dump = analyze(str(payload["imagePath"]), None, no_store)
        return locate_text(dump, str(payload.get("target", "")))
    if operation == "zoom":
        return zoom(
            str(payload["imagePath"]),
            payload.get("region"),
            payload.get("somId"),
            no_store,
            bool(payload.get("quality", False)),
            payload.get("pack"),
        )
    if operation == "ask":
        return ask(
            str(payload["imagePath"]),
            str(payload["question"]),
            payload.get("region"),
            bool(payload.get("quality", False)),
            payload.get("pack"),
        )
    if operation == "element":
        return element(str(payload["imagePath"]), int(payload["id"]), no_store)
    if operation == "vision_prompt":
        return vision_prompt(str(payload.get("lang", "ru")))
    if operation == "warm":
        return warm()
    if operation == "capture":
        return capture_op(str(payload["url"]), payload, no_store)
    if operation == "motion":
        return motion_op(str(payload["url"]), payload, no_store)
    if operation == "inspect":
        region = payload.get("region")
        target = payload.get("target")
        if isinstance(region, dict):
            return analyze(str(payload["imagePath"]), region, no_store)
        if target:
            return inspect_target(str(payload["imagePath"]), str(target), no_store)
        raise ValueError("region or target is required for inspect")
    if operation == "compare":
        return compare_images(
            str(payload["referencePath"]), str(payload["candidatePath"])
        )
    raise ValueError(f"Unsupported Sight operation: {operation}")



def _warm_vlm_async() -> bool:
    """Preload the VLM pack in the background while the worker boots.

    The broker spawns this worker lazily on the first request; the warm thread
    loads the GGUF models concurrently so later see/ask calls hit a hot host.
    Errors are tolerated: the worker must serve deterministic ops without models.
    """
    if os.environ.get("SENS_VLM_PRELOAD", "").lower() not in {"1", "true", "yes"}:
        return False

    import threading

    def _run() -> None:
        try:
            warm()
        except Exception as error:  # noqa: BLE001 - warm must never kill the worker
            sys.stderr.write(f"vlm warm skipped: {error}\n")
            sys.stderr.flush()

    threading.Thread(target=_run, daemon=True, name="vlm-warm").start()
    return True


def _configure_protocol_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="backslashreplace")


def main() -> None:
    _configure_protocol_streams()
    _warm_vlm_async()
    for line in sys.stdin:
        if not line.strip():
            continue
        request_id = None
        try:
            message = json.loads(line)
            request_id = message.get("requestId")
            result = handle(message)
            response = {"ok": True, "requestId": request_id, "result": result}
        except Exception as error:  # noqa: BLE001 - protocol boundary
            response = {
                "ok": False,
                "requestId": request_id,
                "error": {"message": str(error), "type": type(error).__name__},
            }
        try:
            print(json.dumps(response, ensure_ascii=False), flush=True)
        except Exception as error:  # noqa: BLE001 - stdout is broken; exit for the broker to restart us
            sys.stderr.write(f"could not write response: {error}\n")
            sys.stderr.flush()
            raise SystemExit(3)
