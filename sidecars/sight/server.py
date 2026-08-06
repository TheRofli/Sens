"""NDJSON protocol: handle() and the stdin loop."""

from __future__ import annotations

import json
import sys

from sight.compare import compare_images
from sight.ops import analyze, inspect_target, locate_text




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
        return analyze(str(payload["imagePath"]), payload.get("region"), no_store)
    if operation == "read":
        dump = analyze(str(payload["imagePath"]), payload.get("region"), no_store)
        return {"texts": [item["text"] for item in dump["ocr"]], "ocr": dump["ocr"]}
    if operation == "locate":
        dump = analyze(str(payload["imagePath"]), None, no_store)
        return locate_text(dump, str(payload.get("target", "")))
    if operation == "zoom":
        region = payload.get("region")
        if not isinstance(region, dict):
            raise ValueError("region is required for zoom")
        return analyze(str(payload["imagePath"]), region, no_store)
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



def main() -> None:
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
