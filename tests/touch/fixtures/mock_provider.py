#!/usr/bin/env python3
"""Mock OpenAI-compatible provider for Touch contract tests (stdlib only).

Implements a scripted subset of POST /v1/chat/completions with tool calling:
each request advances through a scenario list. A scenario entry is either:

    {"role": "tool_call", "tool": "read", "args": {...}}
    {"role": "final", "content": "...", "usage": {"prompt_tokens": N, "completion_tokens": M}}

The server records every request (body, Authorization header presence, masked)
to a log file for tests that assert the key never reaches the worker side.

Usage:
    python mock_provider.py --port 0 --scenario scenario.json --log requests.jsonl
"""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SCENARIO_DEFAULT = [
    {
        "role": "tool_call",
        "tool": "read",
        "args": {"path": "tests/touch/fixtures/source_files/useSocket.ts"},
    },
    {
        "role": "final",
        "echo_evidence": True,
        "content": '{"conclusion": "fixture", "confidence": 0.9}',
        "usage": {"prompt_tokens": 1200, "completion_tokens": 300},
    },
]

EVIDENCE_MARKER = "EVIDENCE RECEIPT (reference this evidenceId in claims):"


def extract_evidence_id(messages):
    """Pull the last evidenceId the worker embedded from a tool_result."""
    for message in reversed(messages or []):
        if message.get("role") != "tool":
            continue
        content = message.get("content") or ""
        if EVIDENCE_MARKER not in content:
            continue
        payload = content.split(EVIDENCE_MARKER, 1)[1].strip()
        try:
            receipt = json.loads(payload)
        except json.JSONDecodeError:
            continue
        evidence_id = receipt.get("evidenceId")
        if evidence_id:
            return evidence_id
    return None


class MockProviderHandler(BaseHTTPRequestHandler):
    scenario = SCENARIO_DEFAULT
    lock = threading.Lock()
    step = 0
    log_path = None

    def log_message(self, format, *args):  # keep stderr clean
        pass

    def _record(self, body: dict, auth_present: bool):
        if not self.log_path:
            return
        record = {
            "step": self.step,
            "auth_present": auth_present,
            "auth_masked": "<present>" if auth_present else "<absent>",
            "model": body.get("model"),
            "has_tools": "tools" in body,
            "message_count": len(body.get("messages", [])),
            "last_user_content": str(body.get("messages", [{}])[-1].get("content"))[:200],
        }
        with self.lock:
            with open(self.log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")

    def do_POST(self):  # noqa: N802 (http.server API)
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        self._record(body, bool(self.headers.get("Authorization")))

        with self.lock:
            scenario = type(self).scenario
            step = type(self).step
            entry = scenario[min(step, len(scenario) - 1)]
            type(self).step = step + 1

        if entry["role"] == "tool_call":
            self._json(
                200,
                {
                    "id": f"chatcmpl-mock-{self.step}",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_mock_1",
                                        "type": "function",
                                        "function": {
                                            "name": entry["tool"],
                                            "arguments": json.dumps(entry.get("args", {})),
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 800, "completion_tokens": 60},
                },
            )
            return
        usage = entry.get("usage", {})
        content = entry.get("content", "")
        if entry.get("echo_evidence"):
            evidence_id = extract_evidence_id(body.get("messages", []))
            if evidence_id:
                content = content.replace("{EVIDENCE_ID}", evidence_id)
        self._json(
            200,
            {
                "id": f"chatcmpl-mock-{self.step}",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": entry.get("content", ""),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 1000),
                    "completion_tokens": usage.get("completion_tokens", 200),
                },
            },
        )

    def do_GET(self):  # noqa: N802
        if self.path == "/v1/models":
            self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "mock/deepseek-v4-flash-0731",
                            "object": "model",
                            "owned_by": "mock",
                        }
                    ],
                },
            )
            return
        self._json(404, {"error": "not found"})

    def _json(self, status: int, payload: dict):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--scenario", default=None, help="JSON file with scenario list")
    parser.add_argument("--log", default=None, help="JSONL request log path")
    args = parser.parse_args()

    if args.scenario:
        with open(args.scenario, encoding="utf-8") as handle:
            MockProviderHandler.scenario = json.load(handle)
    MockProviderHandler.log_path = args.log

    server = ThreadingHTTPServer(("127.0.0.1", args.port), MockProviderHandler)
    print(f"mock provider on http://127.0.0.1:{server.server_address[1]}/v1", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
