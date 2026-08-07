"""Reproducible CPU-only benchmark for Sens local semantic vision packs."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import psutil

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sidecars"))

from sight.vlm import PACKS, VlmHost  # noqa: E402


CASES = [
    {
        "id": "design_exact_ocr",
        "image": "qa/incoming/design-review.png",
        "prompt": "Transcribe every visible word exactly as written, in reading order. Output text only.",
        "expected": "Settings\nProfile\nActions\nfaint hint\nthis label runs off the edge of the panel",
        "metric": "similarity",
    },
    {
        "id": "design_overflow_grounding",
        "image": "qa/incoming/design-review.png",
        "prompt": "Does any text extend outside a bordered panel? Answer YES or NO, then quote that text.",
        "required": ["yes", "this label runs off the edge of the panel"],
        "metric": "required_phrases",
    },
    {
        "id": "negative_hallucination",
        "image": "qa/incoming/design-review.png",
        "prompt": "What bank balance is shown? If this image has no bank balance, answer exactly NONE.",
        "expected": "none",
        "metric": "exact",
    },
    {
        "id": "russian_ui_ocr",
        "image": "qa/incoming/ui-question.png",
        "prompt": "Transcribe the main headline and the two capability names exactly. Output text only.",
        "required": ["чувства модели", "зрение", "слух"],
        "metric": "required_phrases",
    },
    {
        "id": "poster_ui_grounding",
        "image": "qa/incoming/2026-08-06T02-55-08-191Z-fe5fe4-02.png",
        "prompt": "Quote at least two phrases from the central vertical black strip, then identify the product shown in the center. Be concise.",
        "required_groups": [
            ["illustration", "creative coding", "web experiments"],
            ["mono x7", "display", "canvas"],
        ],
        "metric": "required_groups",
    },
]


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"[\w.-]+", text.casefold()))


def score_case(case: dict[str, Any], answer: str) -> float:
    normalized = _normalize(answer)
    metric = case["metric"]
    if metric == "exact":
        return 1.0 if normalized == _normalize(case["expected"]) else 0.0
    if metric == "similarity":
        return round(SequenceMatcher(None, _normalize(case["expected"]), normalized).ratio(), 4)
    if metric == "required_phrases":
        hits = sum(_normalize(phrase) in normalized for phrase in case["required"])
        return round(hits / len(case["required"]), 4)
    if metric == "required_groups":
        hits = sum(
            any(_normalize(option) in normalized for option in group)
            for group in case["required_groups"]
        )
        return round(hits / len(case["required_groups"]), 4)
    raise ValueError(f"unknown metric: {metric}")


def run_pack(pack: str) -> dict[str, Any]:
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    running = True

    def sample_memory() -> None:
        nonlocal peak_rss
        while running:
            peak_rss = max(peak_rss, process.memory_info().rss)
            time.sleep(0.05)

    monitor = threading.Thread(target=sample_memory, daemon=True)
    monitor.start()
    host = VlmHost(pack, idle_seconds=3600)
    results = []
    try:
        for case in CASES:
            started = time.perf_counter()
            answer = host.ask(str(ROOT / case["image"]), case["prompt"])
            results.append(
                {
                    "id": case["id"],
                    "elapsedSeconds": round(time.perf_counter() - started, 3),
                    "score": score_case(case, answer),
                    "answer": answer,
                }
            )
    finally:
        host.unload()
        running = False
        monitor.join(timeout=1)
    return {
        "pack": pack,
        "model": PACKS[pack],
        "cpuOnly": True,
        "peakRssMiB": round(peak_rss / 1024 / 1024, 1),
        "meanScore": round(sum(item["score"] for item in results) / len(results), 4),
        "totalSeconds": round(sum(item["elapsedSeconds"] for item in results), 3),
        "cases": results,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Sens local VLM benchmark",
        "",
        f"Measured: {report['measuredAt']} on {report['machine']}. CPU-only; one model process at a time.",
        "",
        "| Pack | Model | Mean task score | Total time | Peak RSS |",
        "|---|---|---:|---:|---:|",
    ]
    for item in report["results"]:
        lines.append(
            f"| `{item['pack']}` | `{item['model']['repo']}` | {item['meanScore']:.3f} | {item['totalSeconds']:.1f}s | {item['peakRssMiB']:.0f} MiB |"
        )
    lines += [
        "",
        "## Per-task evidence",
        "",
    ]
    for item in report["results"]:
        lines.append(f"### {item['pack']} — {item['model']['repo']}")
        lines.append("")
        for case in item["cases"]:
            safe_answer = case["answer"].replace("\n", " / ")
            lines.append(
                f"- `{case['id']}`: score {case['score']:.3f}, {case['elapsedSeconds']:.2f}s — {safe_answer}"
            )
        lines.append("")
    winner = max(report["results"], key=lambda item: (item["meanScore"], -item["totalSeconds"]))
    lines += [
        "## Decision",
        "",
        f"`{winner['pack']}` (`{winner['model']['repo']}`) is the default semantic pack because it has the highest measured Sens-task score. Deterministic OCR, geometry, color, and comparison remain the primary truth; VLM output remains explicitly inferred.",
        "",
        "Model absence is a supported degraded state. No model is downloaded or loaded implicitly.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=list(PACKS))
    parser.add_argument("--packs", nargs="+", choices=list(PACKS), default=["lite", "quality"])
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(run_pack(args.worker), ensure_ascii=False))
        return

    results = []
    for pack in args.packs:
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", pack],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=True,
        )
        results.append(json.loads(completed.stdout.strip().splitlines()[-1]))
    report = {
        "schemaVersion": "1.0.0",
        "measuredAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "machine": "AMD Ryzen 7 5800XT, Windows, llama-cpp-python 0.3.34",
        "results": results,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered, encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
