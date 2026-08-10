"""Freeze live URL hero viewports for the Sens reconstruction matrix."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sidecars"))

from sight.capture import capture_url  # noqa: E402


CASES = {
    "hungry-tiger": "https://www.eathungrytiger.com/",
    "dope-security": "https://dope.security/",
    "caldera": "https://caldera.xyz/",
}
OPTIONS = {
    "viewport": {"width": 1440, "height": 900},
    "dpr": 1.0,
    "theme": "light",
    "locale": "en-US",
    "waitUntil": "load",
    "fullPage": False,
    "timeoutMs": 60_000,
    "settleMs": 2_500,
    "scrollSteps": 0,
}


def freeze(case_id: str, url: str) -> dict:
    scratch = ROOT / "output" / "playwright" / "sens-1.3.7-baselines" / case_id
    references = ROOT / "qa" / "fixtures" / "reconstruction-matrix" / "references"
    metadata_root = ROOT / "qa" / "fixtures" / "reconstruction-matrix" / "live-baselines"
    scratch.mkdir(parents=True, exist_ok=True)
    references.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)

    result = capture_url(url, scratch, OPTIONS)
    digest = result["screenshotSha256"]
    reference = references / f"{case_id}-{digest[:16]}.png"
    shutil.copy2(result["screenshot"], reference)
    metadata = {
        "schemaVersion": 1,
        "id": case_id,
        "requestedUrl": url,
        "observedAt": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "reference": reference.relative_to(ROOT).as_posix(),
        "sha256": digest,
        "settings": result["settings"],
        "captureId": result["captureId"],
        "observed": {
            "domElementCount": len(result.get("dom", [])),
            "textNodeCount": len(result.get("textNodes", [])),
            "semanticControlCount": len(result.get("semanticControls", [])),
            "rasterElementCount": len(result.get("rasterElements", [])),
            "motionEventCount": len(result.get("motion", [])),
        },
    }
    metadata_path = metadata_root / f"{case_id}.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=[*CASES, "all"], default="all")
    args = parser.parse_args()
    selected = CASES if args.case == "all" else {args.case: CASES[args.case]}
    for case_id, url in selected.items():
        print(json.dumps(freeze(case_id, url), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
