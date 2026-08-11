import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import pytest

from sight.ops import see_document


ROOT = Path(__file__).resolve().parents[2]
MATRIX_ROOT = ROOT / "qa" / "fixtures" / "reconstruction-matrix"


def test_release_reconstruction_matrix_has_seven_immutable_baselines() -> None:
    manifest = json.loads((MATRIX_ROOT / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["release"] == "1.3.7"
    assert len(manifest["cases"]) == 7
    assert {case["kind"] for case in manifest["cases"]} == {"image", "url"}

    for case in manifest["cases"]:
        baseline = case if case["kind"] == "image" else case["frozenBaseline"]
        assert baseline is not None, case["id"]
        reference = MATRIX_ROOT / baseline["reference"]
        assert reference.is_file(), case["id"]
        assert hashlib.sha256(reference.read_bytes()).hexdigest() == baseline["sha256"]
        image = cv2.imread(str(reference))
        assert image is not None, case["id"]
        if case["kind"] == "image":
            assert [image.shape[1], image.shape[0]] == [case["width"], case["height"]]
        else:
            viewport = baseline["viewport"]
            assert [image.shape[1], image.shape[0]] == [
                viewport["width"],
                viewport["height"],
            ]


def test_release_matrix_runner_resolves_every_frozen_reference() -> None:
    runner_path = ROOT / "scripts" / "run-reconstruction-matrix.py"
    module_spec = importlib.util.spec_from_file_location(
        "sens_reconstruction_matrix_runner", runner_path
    )
    assert module_spec is not None and module_spec.loader is not None
    runner = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(runner)
    manifest = json.loads((MATRIX_ROOT / "manifest.json").read_text(encoding="utf-8"))

    assert [runner._reference_path(case) for case in manifest["cases"]] == [
        (
            MATRIX_ROOT
            / (
                case["reference"]
                if case["kind"] == "image"
                else case["frozenBaseline"]["reference"]
            )
        ).resolve()
        for case in manifest["cases"]
    ]


def test_matrix_runner_forces_utf8_for_unicode_review_reports() -> None:
    runner_path = ROOT / "scripts" / "run-reconstruction-matrix.py"
    module_spec = importlib.util.spec_from_file_location(
        "sens_reconstruction_matrix_runner_utf8", runner_path
    )
    assert module_spec is not None and module_spec.loader is not None
    runner = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(runner)

    class Stream:
        configured: dict[str, str] | None = None

        def reconfigure(self, **options: str) -> None:
            self.configured = options

    stream = Stream()
    runner._configure_output_stream(stream)

    assert stream.configured == {
        "encoding": "utf-8",
        "errors": "backslashreplace",
    }


def test_matrix_compact_contract_fits_agent_tool_result_budget() -> None:
    pytest.importorskip(
        "rapidocr",
        reason="release-runtime QA; scripts/check-reconstruction-result-budget.py is the portable gate",
    )
    manifest = json.loads((MATRIX_ROOT / "manifest.json").read_text(encoding="utf-8"))

    for case in manifest["cases"]:
        baseline = case if case["kind"] == "image" else case["frozenBaseline"]
        reference = MATRIX_ROOT / baseline["reference"]
        result = see_document(
            str(reference),
            fast=True,
            profile="reconstruct",
            response="compact",
            target_kind="web",
            intent="Recreate this exact reference as a live website.",
        )
        pretty_wrapped = json.dumps(
            {"result": {"data": result}},
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")

        assert len(pretty_wrapped) < 40_000, case["id"]
        assert result["doc"]["reconstruction"]["focusPlan"] == {
            "encoding": "response-reference",
            "path": "summary.nextActions",
            "count": len(result["summary"]["nextActions"]),
        }
        assert len(result["summary"]["nextActions"]) <= 4
