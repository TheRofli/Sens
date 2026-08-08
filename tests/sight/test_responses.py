import numpy as np

from sight import ops


def _stub_document(profile: str) -> dict:
    return {
        "schemaVersion": "2.0.0",
        "profile": profile,
        "source": {"id": "sha256:fixture", "mediaType": "image"},
        "coordinateSpaces": {"source": {"size": [1000, 500]}},
        "header": {"size": [1000, 500]},
        "artifacts": [{"id": "som:fixture", "kind": "set-of-marks"}],
        "warnings": [],
        "nextActions": [],
        "claims": [{"id": "duplicate"}],
        "ascii": "duplicate composition map",
        "elements": [{"id": 1}],
        "semantics_status": "unavailable",
        "reconstruction": {"canvas": {"width": 1000, "height": 500}}
        if profile == "reconstruct"
        else None,
    }


def _install_stubs(monkeypatch, captured) -> None:
    monkeypatch.setattr(
        ops,
        "analyze",
        lambda *_args, **_kwargs: {
            "somPath": "som.png",
            "design": {"facts": [{"kind": "alignment", "detail": "ok"}]},
            "warnings": [],
        },
    )
    monkeypatch.setattr(
        ops,
        "_image_for",
        lambda *_args, **_kwargs: np.zeros((10, 10, 3), np.uint8),
    )

    def build_document(*_args, **kwargs):
        captured.update(kwargs)
        return _stub_document(kwargs["profile"])

    monkeypatch.setattr(ops.docmod, "build_document", build_document)
    monkeypatch.setattr(ops.docmod, "render_markdown", lambda _doc: "FULL MARKDOWN")


def test_see_defaults_to_compact_reconstruction_for_copy_intent(monkeypatch) -> None:
    captured = {}
    _install_stubs(monkeypatch, captured)
    refined = []
    monkeypatch.setattr(
        ops,
        "refine_ocr_for_reconstruction",
        lambda path, items: refined.append((path, items)) or items,
        raising=False,
    )

    result = ops.see_document(
        "fixture.png",
        fast=True,
        intent="Repeat this design exactly from the screenshot",
    )

    assert captured["profile"] == "reconstruct"
    assert refined == [("fixture.png", [])]
    assert set(result) == {"doc", "summary", "artifacts", "pack", "compatibility"}
    assert result["compatibility"] == {
        "response": "compact",
        "legacyIncluded": False,
        "fullResponse": "Set response=full only for legacy debugging.",
    }
    assert "document" not in result
    assert "legacy" not in result
    assert "claims" not in result["doc"]
    assert "ascii" not in result["doc"]
    assert "elements" not in result["doc"]
    assert result["summary"]["nextActions"] == []


def test_dense_screenshot_auto_selects_reconstruction_when_client_omits_prompt(
    monkeypatch,
) -> None:
    captured = {}
    _install_stubs(monkeypatch, captured)
    monkeypatch.setattr(
        ops,
        "analyze",
        lambda *_args, **_kwargs: {
            "somPath": None,
            "design": {"facts": []},
            "warnings": [],
            "ocr": [{"text": str(index)} for index in range(4)],
            "elements": [{"id": index} for index in range(5)],
        },
    )
    monkeypatch.setattr(ops, "_apply_reconstruction_ocr", lambda *_args: None)

    result = ops.see_document("fixture.png", fast=True)

    assert captured["profile"] == "reconstruct"
    assert result["doc"]["profile"] == "reconstruct"


def test_full_response_preserves_the_legacy_projection(monkeypatch) -> None:
    captured = {}
    _install_stubs(monkeypatch, captured)

    result = ops.see_document(
        "fixture.png",
        fast=True,
        profile="analyze",
        response="full",
    )

    assert captured["profile"] == "analyze"
    assert result["document"] == "FULL MARKDOWN"
    assert result["legacy"]["design"] == {
        "issues": [{"kind": "alignment", "detail": "ok"}]
    }
    assert result["compatibility"]["legacyIncluded"] is True


def test_consumer_prompt_teaches_the_strict_reconstruction_loop() -> None:
    russian = ops.vision_prompt("ru")["prompt"]
    english = ops.vision_prompt("en")["prompt"]

    for prompt in (russian, english):
        assert "profile=reconstruct" in prompt
        assert "response=compact" in prompt
        assert "fit=strict" in prompt
        assert "canComplete=true" in prompt
        assert "similarityScore" in prompt
        assert "focusPlan" in prompt
        assert "preferredValue" in prompt
