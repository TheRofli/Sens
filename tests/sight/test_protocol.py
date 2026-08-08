from sight import server


def test_worker_configures_protocol_streams_as_utf8(monkeypatch) -> None:
    calls = []

    class Stream:
        def reconfigure(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(server.sys, "stdout", Stream())
    monkeypatch.setattr(server.sys, "stderr", Stream())

    server._configure_protocol_streams()

    assert calls == [
        {"encoding": "utf-8", "errors": "backslashreplace"},
        {"encoding": "utf-8", "errors": "backslashreplace"},
    ]


def test_compare_forwards_explicit_fit_mode(monkeypatch) -> None:
    captured = {}

    def fake_compare(reference_path, candidate_path, *, fit="strict"):
        captured.update(
            reference_path=reference_path,
            candidate_path=candidate_path,
            fit=fit,
        )
        return {"verdict": "fail"}

    monkeypatch.setattr(server, "compare_images", fake_compare)

    result = server.handle(
        {
            "operation": "compare",
            "input": {
                "referencePath": "reference.png",
                "candidatePath": "candidate.png",
                "fit": "resize",
            },
        }
    )

    assert result == {"verdict": "fail"}
    assert captured == {
        "reference_path": "reference.png",
        "candidate_path": "candidate.png",
        "fit": "resize",
    }


def test_see_forwards_reconstruction_profile_and_response_mode(monkeypatch) -> None:
    captured = {}

    def fake_see(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"doc": {"profile": kwargs["profile"]}}

    monkeypatch.setattr(server, "see_document", fake_see)

    result = server.handle(
        {
            "operation": "see",
            "input": {
                "imagePath": "reference.png",
                "profile": "reconstruct",
                "response": "full",
            },
        }
    )

    assert result == {"doc": {"profile": "reconstruct"}}
    assert captured["kwargs"] == {
        "profile": "reconstruct",
        "response": "full",
    }


def test_zoom_keeps_reconstruction_profile_compact(monkeypatch) -> None:
    captured = {}

    def fake_zoom(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"doc": {"profile": kwargs["profile"]}}

    monkeypatch.setattr(server, "zoom", fake_zoom)

    result = server.handle(
        {
            "operation": "zoom",
            "input": {
                "imagePath": "reference.png",
                "region": {"x": 1, "y": 2, "width": 3, "height": 4},
                "profile": "reconstruct",
                "response": "compact",
            },
        }
    )

    assert result == {"doc": {"profile": "reconstruct"}}
    assert captured["kwargs"] == {
        "profile": "reconstruct",
        "response": "compact",
    }
