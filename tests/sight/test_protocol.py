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
