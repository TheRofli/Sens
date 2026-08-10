#!/usr/bin/env python3
"""Verify every frozen 1.3.7 reconstruction contract fits agent tool budgets."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sidecars"))

from sight.ops import see_document  # noqa: E402


def main() -> int:
    matrix_root = ROOT / "qa" / "fixtures" / "reconstruction-matrix"
    manifest = json.loads(
        (matrix_root / "manifest.json").read_text(encoding="utf-8")
    )
    rows = []
    for case in manifest["cases"]:
        baseline = case if case["kind"] == "image" else case["frozenBaseline"]
        reference = matrix_root / baseline["reference"]
        result = see_document(
            str(reference),
            fast=True,
            profile="reconstruct",
            response="compact",
            target_kind="web",
            intent="Recreate this exact reference as a live website.",
        )
        wrapped_bytes = len(
            json.dumps(
                {"result": {"data": result}},
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8")
        )
        focus_count = len(result["summary"]["nextActions"])
        if wrapped_bytes >= 40_000:
            raise AssertionError(
                f"{case['id']} exceeds pretty tool-result budget: {wrapped_bytes} bytes"
            )
        if focus_count > 4:
            raise AssertionError(
                f"{case['id']} exceeds focus budget: {focus_count} regions"
            )
        rows.append(
            {
                "case": case["id"],
                "prettyWrappedBytes": wrapped_bytes,
                "focusCount": focus_count,
                "textCount": result["doc"]["reconstruction"]["text"]["count"],
            }
        )

    print(json.dumps({"status": "pass", "cases": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
