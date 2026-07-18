"""Local HTTP API server for the Tauri shell (and other local clients).

Runs on ``127.0.0.1`` on an ephemeral port discovered via ``data/api.port``.
All endpoints speak JSON. Read endpoints snapshot immutable state; write
endpoints hop onto the UI thread via ``app.post_ui_sync`` because tkinter is
not thread-safe.

Endpoint map
------------
GET  /api/status           runtime + model + recording state
GET  /api/settings         current settings dict
POST /api/settings         merge-update settings (sync-applied)
GET  /api/models           list presets with install status
POST /api/model            {key} select the active model
POST /api/model/load       load the active model (background)
POST /api/model/unload     unload the model
POST /api/model/install    {key} install a model (async; poll /api/status)
GET  /api/history          ?limit=&q= transcript history
POST /api/history/copy     {id} copy a transcript by id
POST /api/action/copy_last copy the latest transcript
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

from .engines.paths import data_dir

if TYPE_CHECKING:
    from .app import SpeechApp


class SpeechAPIServer:
    """Owns the ThreadingHTTPServer bound to an ephemeral port."""

    def __init__(self, app: "SpeechApp") -> None:
        self.app = app
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.port: int = 0

    def start(self) -> None:
        # Capture the app on the handler via a closure-based factory so the
        # handler does not need a global.
        app = self.app

        class _Handler(SpeechAPIHandler):
            pass

        _Handler.app = app  # type: ignore[attr-defined]
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.port = self._httpd.server_address[1]
        self._write_port(self.port)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, name="speech-api", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._clear_port()

    def _write_port(self, port: int) -> None:
        port_file = data_dir() / "api.port"
        try:
            port_file.parent.mkdir(parents=True, exist_ok=True)
            port_file.write_text(str(port), encoding="utf-8")
        except OSError:
            pass

    def _clear_port(self) -> None:
        port_file = data_dir() / "api.port"
        try:
            port_file.unlink(missing_ok=True)
        except OSError:
            pass


class SpeechAPIHandler(BaseHTTPRequestHandler):
    """BaseHTTPRequestHandler subclass; ``app`` is set by SpeechAPIServer."""

    app: "SpeechApp"  # type: ignore[assignment]

    # Quiet logging: this is a local control channel, not a public server.
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    # -- helpers ---------------------------------------------------------

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Any:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    # -- routing ---------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = parse_qs(parsed.query)
        try:
            if path == "/api/status":
                self._send_json(200, self._status())
            elif path == "/api/settings":
                self._send_json(200, self.app.get_settings_values())
            elif path == "/api/models":
                self._send_json(200, self.app.available_models())
            elif path == "/api/history":
                limit = _first_int(query.get("limit"), default=80)
                text_query = _first_str(query.get("q"), default="").lower()
                self._send_json(200, self._history(limit, text_query))
            else:
                self._send_json(404, {"error": "not found"})
        except Exception as exc:  # noqa: BLE001 - return 500, keep serving
            self._send_json(500, {"error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        body = self._read_json()
        if body is None:
            self._send_json(400, {"error": "invalid json"})
            return
        try:
            handler = {
                "/api/settings": self._post_settings,
                "/api/model": self._post_model,
                "/api/model/load": self._post_model_load,
                "/api/model/unload": self._post_model_unload,
                "/api/model/install": self._post_model_install,
                "/api/history/copy": self._post_history_copy,
                "/api/action/copy_last": self._post_copy_last,
            }.get(path)
            if handler is None:
                self._send_json(404, {"error": "not found"})
            else:
                handler(body)
        except Exception as exc:  # noqa: BLE001 - return 500, keep serving
            self._send_json(500, {"error": str(exc)})

    # -- endpoint bodies -------------------------------------------------

    def _status(self) -> dict[str, Any]:
        app = self.app
        preset = app.model_status()
        return {
            "running": True,
            "engine_enabled": app.engine_enabled(),
            "model_state": app._model_state_label(),  # noqa: SLF001
            "model": app.current_model(),
            "model_label": app.current_model_label(),
            "model_loaded": app.model_loaded(),
            "model_loading": app.model_is_loading(),
            "model_installed": preset.installed,
            "model_size_label": preset.size_label if preset.installed else "Not installed",
            "transcribing": getattr(app, "transcribing", False),
            "device": app.current_device(),
            "backend": app.current_backend(),
            "status_text": app.status_text(),
        }

    def _post_settings(self, body: Any) -> None:
        if not isinstance(body, dict):
            self._send_json(400, {"error": "expected a JSON object"})
            return
        # Merge onto current values so partial updates work and unknown keys
        # are ignored (save_settings_values reads known keys only).
        current = self.app.get_settings_values()
        merged = {**current, **body}
        self.app.post_ui_sync(lambda: self.app.save_settings_values(merged))
        self._send_json(200, {"ok": True, "settings": self.app.get_settings_values()})

    def _post_model(self, body: Any) -> None:
        key = str(body.get("key", "")).strip()
        if not key:
            self._send_json(400, {"error": "key is required"})
            return
        self.app.post_ui_sync(lambda: self.app.set_model(key))
        self._send_json(200, {"ok": True, "model": self.app.current_model()})

    def _post_model_load(self, _body: Any) -> None:
        self.app.post_ui_sync(lambda: self.app.load_model_background())
        self._send_json(200, {"ok": True})

    def _post_model_unload(self, _body: Any) -> None:
        self.app.post_ui_sync(lambda: self.app.unload_model())
        self._send_json(200, {"ok": True})

    def _post_model_install(self, body: Any) -> None:
        key = str(body.get("key", "")).strip() or self.app.current_model()
        # Installation is long-running; do it in a background thread so the
        # request returns immediately. Status is observable via /api/status
        # (model_installed flips once the marker file lands).
        def _install() -> None:
            from .engines.install import install_model

            try:
                install_model(key)
            except Exception:
                pass

        threading.Thread(target=_install, name="speech-install", daemon=True).start()
        self._send_json(202, {"ok": True, "key": key, "message": "installing"})

    def _history(self, limit: int, text_query: str) -> list[dict[str, str]]:
        rows = self.app.history_rows()
        out: list[dict[str, str]] = []
        for entry_id, text in rows:
            if text_query and text_query not in text.lower():
                continue
            out.append({"id": entry_id, "text": text})
            if len(out) >= limit:
                break
        return out

    def _post_history_copy(self, body: Any) -> None:
        entry_id = str(body.get("id", "")).strip()
        if not entry_id:
            self._send_json(400, {"error": "id is required"})
            return
        self.app.post_ui_sync(lambda: self.app.copy_history_entry(entry_id))
        self._send_json(200, {"ok": True})

    def _post_copy_last(self, _body: Any) -> None:
        self.app.post_ui_sync(lambda: self.app.copy_last_transcript())
        self._send_json(200, {"ok": True})


def _first_int(values: list[str] | None, default: int) -> int:
    if not values:
        return default
    try:
        return int(values[0])
    except ValueError:
        return default


def _first_str(values: list[str] | None, default: str) -> str:
    return values[0] if values else default
