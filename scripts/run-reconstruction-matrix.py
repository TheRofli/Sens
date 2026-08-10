#!/usr/bin/env python3
"""Run the frozen Sens web-reconstruction matrix through the public pipeline.

The gate intentionally uses ``see_document`` to create the implementation and
``review_web`` to capture the generated site.  It does not inspect pixels with
an independent ad-hoc scorer, so its result matches the contract exposed to an
MCP client such as Z-Code.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import threading
import time
from contextlib import contextmanager
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sidecars"))

from sight.ops import see_document  # noqa: E402
from sight.web_review import review_web  # noqa: E402


MATRIX_ROOT = ROOT / "qa" / "fixtures" / "reconstruction-matrix"
DEFAULT_INTENT = (
    "Reconstruct this exact reference as a real website with live selectable "
    "text, semantic links and buttons, measured geometry, and raster only for "
    "non-text artwork."
)


def _configure_output_stream(stream: object) -> None:
    """Make Unicode repair hints printable in legacy Windows terminals."""
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="backslashreplace")


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, _format: str, *_args: object) -> None:
        return


@contextmanager
def _serve(root: Path) -> Iterator[str]:
    handler = functools.partial(_QuietHandler, directory=str(root))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = int(server.server_address[1])
        yield f"http://127.0.0.1:{port}/index.html"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _reference_path(case: dict[str, object]) -> Path:
    baseline = case if case["kind"] == "image" else case["frozenBaseline"]
    assert isinstance(baseline, dict)
    return (MATRIX_ROOT / str(baseline["reference"])).resolve()


def _starter_index(artifacts: list[dict[str, object]]) -> Path:
    for artifact in artifacts:
        if artifact.get("kind") == "semantic-web-starter":
            return Path(str(artifact["uri"])).resolve()
    raise RuntimeError("see_document did not materialize a semantic web starter")


def _run_case(
    case: dict[str, object],
    output_root: Path,
    *,
    fast: bool,
    max_semantic_calls: int,
) -> dict[str, object]:
    case_id = str(case["id"])
    reference = _reference_path(case)
    case_root = output_root / case_id
    assets_root = case_root / "assets"
    case_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print(f"[{case_id}] reconstruct", file=sys.stderr, flush=True)
    result = see_document(
        str(reference),
        no_store=False,
        fast=fast,
        quality=not fast,
        intent=DEFAULT_INTENT,
        max_semantic_calls=max_semantic_calls,
        profile="reconstruct",
        response="brief",
        target_kind="web",
        resolve_focus=not fast,
        asset_output_dir=str(assets_root),
    )
    contract_path = Path(str(result["contractPath"])).resolve()
    artifacts = list(result.get("artifacts") or [])
    starter_index = _starter_index(artifacts)
    print(f"[{case_id}] sens_review", file=sys.stderr, flush=True)
    with _serve(starter_index.parent) as url:
        review = review_web(
            str(reference),
            url,
            {"contractPath": str(contract_path)},
            out_dir=case_root / "review",
        )

    capture = dict(review.get("capture") or {})
    row: dict[str, object] = {
        "case": case_id,
        "reference": str(reference),
        "elapsedSeconds": round(time.perf_counter() - started, 2),
        "contractPath": str(contract_path),
        "starter": str(starter_index),
        "candidateScreenshot": capture.get("screenshot"),
        "verdict": review.get("verdict"),
        "canComplete": bool(review.get("canComplete")),
        "visualPass": bool(review.get("visualPass")),
        "webPass": bool(review.get("webPass")),
        "textNodeCount": int(capture.get("textNodeCount") or 0),
        "semanticControlCount": int(capture.get("semanticControlCount") or 0),
        "rasterElementCount": int(capture.get("rasterElementCount") or 0),
        "accessibilityAvailable": bool(capture.get("accessibilityAvailable")),
        "repairHints": review.get("repairHints") or {},
    }
    row["pass"] = bool(
        row["canComplete"] and row["visualPass"] and row["webPass"]
    )
    return row


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="Run only this case id. Repeat the flag to select several cases.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Artifact directory (default: output/reconstruction-matrix/<timestamp>).",
    )
    parser.add_argument(
        "--max-semantic-calls",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip the local VLM for a quick plumbing smoke test.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the frozen case ids and exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = json.loads((MATRIX_ROOT / "manifest.json").read_text("utf-8"))
    cases = list(manifest["cases"])
    if args.list:
        print("\n".join(str(case["id"]) for case in cases))
        return 0
    if args.max_semantic_calls < 0 or args.max_semantic_calls > 7:
        raise SystemExit("--max-semantic-calls must be between 0 and 7")
    requested = set(args.cases or [])
    known = {str(case["id"]) for case in cases}
    unknown = requested - known
    if unknown:
        raise SystemExit(f"unknown matrix case(s): {', '.join(sorted(unknown))}")
    selected = [case for case in cases if not requested or case["id"] in requested]
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_root = (
        args.output_root
        or ROOT / "output" / "reconstruction-matrix" / timestamp
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for case in selected:
        try:
            rows.append(
                _run_case(
                    case,
                    output_root,
                    fast=bool(args.fast),
                    max_semantic_calls=int(args.max_semantic_calls),
                )
            )
        except Exception as error:  # Keep the remaining release cases observable.
            rows.append(
                {
                    "case": str(case["id"]),
                    "pass": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    report = {
        "schemaVersion": 1,
        "release": manifest["release"],
        "mode": "fast" if args.fast else "quality",
        "outputRoot": str(output_root),
        "pass": all(bool(row.get("pass")) for row in rows),
        "cases": rows,
    }
    report_path = output_root / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    _configure_output_stream(sys.stdout)
    _configure_output_stream(sys.stderr)
    raise SystemExit(main())
