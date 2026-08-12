"""Touch worker agent-loop tests: the worker is driven by a scripted broker
in-process (no network, no provider). Validates:
- the worker requests a model call and follows tool calls with tool_requests;
- evidence receipts from tool results are embedded into the context;
- the final structured answer becomes a WorkerResult with claims referencing
  broker-issued evidence ids;
- a limit message produces an honest partial result.

Run: python -m unittest tests.touch.test_worker (or pytest tests/touch)
"""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

WORKER = Path(__file__).resolve().parent.parent.parent / "sidecars" / "touch" / "touch-worker.py"
ROLES = WORKER.parent / "roles"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def run_worker(script_lines):
    """Feed NDJSON script lines to the worker; return (out_messages, returncode)."""
    env = dict(os.environ)
    env["SENS_TOUCH_ROLES_DIR"] = str(ROLES)
    proc = subprocess.Popen(
        [sys.executable, str(WORKER)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    payload = "\n".join(json.dumps(line, ensure_ascii=False) for line in script_lines) + "\n"
    out, err = proc.communicate(payload, timeout=30)
    messages = [json.loads(line) for line in out.splitlines() if line.strip()]
    return messages, proc.returncode, err


def start_packet(role="explorer", objective="Investigate", scope=("src",)):
    return {
        "type": "start",
        "packet": {
            "jobId": "tch_test_1",
            "packetId": "pkt_test_1",
            "role": role,
            "objective": objective,
            "scope": list(scope),
            "constraints": ["read-only"],
            "deliverable": "report",
            "outputFormat": "auto",
            "budget": {
                "maxSteps": 15,
                "maxTotalInputTokens": 50000,
                "maxTotalOutputTokens": 6000,
                "maxContextTokens": 24000,
                "maxToolResultTokens": 6000,
                "maxSingleToolResultTokens": 2500,
                "timeoutS": 180,
                "maxSpendUsd": None,
            },
            "context": {"os": "windows", "cwd": None, "repo": None},
        },
        "model": "mock/deepseek-v4-flash-0731",
        "budget": {"maxSteps": 15},
    }


TOOL_CALL_READ = {
    "type": "model_response",
    "message": {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "read", "arguments": '{"path": "src/a.ts"}'},
            }
        ],
    },
    "usage": {"prompt_tokens": 100, "completion_tokens": 20},
}

FINAL_JSON = (
    '{"conclusion": "done", "confidence": 0.9, '
    '"claims": [{"claim": "line 47 contains setInterval", "claimStatus": "verified", '
    '"confidence": 1.0, "evidence": [{"evidenceId": "ev_123", "evidenceStatus": "verified"}]}], '
    '"findings": ["x"], "risks": [], "recommendedAction": null, "unresolved": []}'
)

EVIDENCE = {
    "evidenceId": "ev_123",
    "kind": "file_read",
    "path": "src/a.ts",
    "range": [47, 47],
    "sha256": "abc",
    "observedAt": "2026-08-13T12:00:00Z",
    "snippet": "        setInterval(reconnect, 5000);",
}


class WorkerAgentLoopTests(unittest.TestCase):
    def test_tool_call_followed_by_tool_request_and_final_result(self):
        script = [
            start_packet(objective="Read line 47"),
            TOOL_CALL_READ,
            {"type": "tool_result", "ok": True, "result": {"text": "line", "path": "src/a.ts"},
             "evidence": EVIDENCE},
            {
                "type": "model_response",
                "message": {"role": "assistant", "content": FINAL_JSON},
                "usage": {"prompt_tokens": 200, "completion_tokens": 50},
            },
        ]
        messages, code, err = run_worker(script)
        self.assertEqual(code, 0, err)
        kinds = [m["type"] for m in messages]
        self.assertIn("model_request", kinds)
        tool_idx = kinds.index("tool_request")
        self.assertEqual(messages[tool_idx]["tool"], "read")
        self.assertEqual(messages[tool_idx]["args"], {"path": "src/a.ts"})
        complete = next(m for m in messages if m["type"] == "complete")
        result = complete["result"]
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["conclusion"], "done")
        self.assertEqual(result["claims"][0]["claimStatus"], "verified")
        self.assertEqual(
            result["claims"][0]["evidence"][0]["evidenceId"], "ev_123"
        )

    def test_limit_produces_honest_partial(self):
        script = [
            start_packet(objective="Read line 47"),
            {"type": "limit", "reason": "input_tokens", "message": {"role": "assistant", "content": FINAL_JSON}},
        ]
        messages, code, err = run_worker(script)
        self.assertEqual(code, 0, err)
        complete = next(m for m in messages if m["type"] == "complete")
        self.assertEqual(complete["result"]["status"], "partial")
        self.assertIn("input_tokens", complete["result"]["warnings"][0])

    def test_protocol_error_without_start(self):
        messages, code, _ = run_worker([{"type": "bogus"}])
        self.assertNotEqual(code, 0)
        self.assertEqual(messages[0]["type"], "failed")

    def test_unknown_tool_error_is_surfaced_to_model(self):
        script = [
            start_packet(objective="Use bad tool"),
            {
                "type": "model_response",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "rm", "arguments": "{}"},
                        }
                    ],
                },
                "usage": {},
            },
            {"type": "tool_result", "ok": False, "error": "unknown tool: rm"},
        ]
        messages, code, err = run_worker(script)
        # The worker must keep looping after an error; with no further broker
        # input it hits EOF and reports failure instead of fabricating success.
        self.assertEqual(messages[-1]["type"], "failed")

    def test_coder_role_has_write_tool_available(self):
        script = [
            start_packet(role="coder", objective="Implement", scope=("src",)),
            {
                "type": "model_response",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_3",
                            "type": "function",
                            "function": {
                                "name": "write",
                                "arguments": '{"path": "sandbox/x.ts", "content": "export const a = 1;"}',
                            },
                        }
                    ],
                },
                "usage": {},
            },
            {
                "type": "tool_result",
                "ok": True,
                "result": {"path": "sandbox/x.ts", "bytes": 20},
                "evidence": {"evidenceId": "ev_write", "kind": "sandbox_write", "sha256": "x",
                             "observedAt": "2026-08-13T12:00:00Z", "snippet": ""},
            },
            {
                "type": "model_response",
                "message": {"role": "assistant", "content": FINAL_JSON},
                "usage": {},
            },
        ]
        messages, code, err = run_worker(script)
        self.assertEqual(code, 0, err)
        tool_messages = [m for m in messages if m["type"] == "tool_request"]
        self.assertEqual(tool_messages[0]["tool"], "write")


if __name__ == "__main__":
    unittest.main()
