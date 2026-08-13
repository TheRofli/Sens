#!/usr/bin/env python3
"""Sens Touch worker: the reasoning brain without any authority.

Design v1.1: "Worker requests, Broker permits and executes."
This worker has NO secrets, NO direct filesystem access and NO network.
Every model call is a model_request to the broker (which owns the provider
key and HTTPS); every tool use is a tool_request to the broker (which
enforces scope, executes, and issues evidence receipts).

The worker's only job is the agent loop: decide the next tool call or the
final structured answer, and compress the result.

Protocol (NDJSON over stdin/stdout):

broker -> worker:
    {"type": "start", "packet": {...}, "model": "...", "budget": {...}}
    {"type": "model_response", "message": {...}, "usage": {...}, "cumulative": {...}}
    {"type": "tool_result", "ok": true, "result": {...}, "evidence": {...}|null}
    {"type": "tool_result", "ok": false, "error": "..."}
    {"type": "limit", "reason": "input_tokens|output_tokens|context_tokens|spend", ...}
    {"type": "cancel"}

worker -> broker:
    {"type": "event", "event": {"t": ..., "kind": "step", "tool": ..., "target": ...}}
    {"type": "model_request", "messages": [...], "tools": [...]}
    {"type": "tool_request", "tool": "read", "args": {...}}
    {"type": "complete", "result": {...WorkerResult or CodingResult...}}
    {"type": "failed", "error": "..."}
"""
import json
import os
import sys
import time
from pathlib import Path

ROLES_DIR = Path(os.environ.get("SENS_TOUCH_ROLES_DIR", Path(__file__).resolve().parent / "roles"))

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a text file inside the issued scope. Returns the requested lines and an evidence receipt (evidenceId) that you MUST reference in claims. Never cite files you have not read.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path inside scope"},
                    "startLine": {"type": "integer"},
                    "endLine": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "List files matching a glob pattern inside the issued scope.",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Regex search inside the issued scope. Returns matches with line numbers and an evidence receipt.",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                "required": ["pattern", "path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Write a file inside the coder sandbox ONLY. Never write anywhere else.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch an https page (public hosts only). Returns text and an evidence receipt with sha256 and fetchedAt.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web when a search provider key is configured. Results are unverifiable until the page itself is fetched with web_fetch.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]

FINAL_SCHEMA_INSTRUCTION = (
    "Your final answer MUST be a single JSON object (no markdown fences) with exactly:\n"
    '{"conclusion": string, "confidence": number 0..1, '
    '"claims": [{"claim": string, "claimStatus": "inferred", "confidence": number 0..1, '
    '"evidence": [{"evidenceId": string, "evidenceStatus": "verified"}]}], '
    '"findings": [string], "risks": [string], "recommendedAction": string|null, '
    '"unresolved": [string]}\n'
    "Every evidenceId MUST come from a receipt the broker issued to you in a tool_result. "
    "You cannot cite anything you did not actually read or fetch. "
    "Semantic conclusions stay claimStatus inferred; only facts you directly observed "
    "(e.g. 'line 47 contains X') may use verified, and only with that evidence.\n"
)


def load_role_prompt(role: str) -> str:
    role_file = ROLES_DIR / f"{role}.md"
    if role_file.is_file():
        return role_file.read_text(encoding="utf-8")
    return f"You are a {role} worker. Be precise and honest; cite only broker-issued evidence receipts."


def send(message: dict) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def read_line() -> dict:
    line = sys.stdin.readline()
    if not line:
        return {"type": "eof"}
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return {"type": "protocol_error", "error": line[:200]}


def run() -> int:
    start = read_line()
    if start.get("type") != "start":
        send({"type": "failed", "error": "expected start message"})
        return 1
    packet = start.get("packet", {})
    role = packet.get("role", "explorer")
    model = start.get("model", "")
    budget = start.get("budget", {})
    max_steps = int(budget.get("maxSteps", 15))

    system_prompt = load_role_prompt(role) + "\n\n" + FINAL_SCHEMA_INSTRUCTION
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Objective:\n{packet.get('objective', '')}\n\n"
                f"Scope: {json.dumps(packet.get('scope', []))}\n"
                f"Constraints: {json.dumps(packet.get('constraints', []))}\n"
                f"Deliverable: {packet.get('deliverable', 'auto')}\n"
                "Work in steps. Use tools to gather evidence. Keep your context small: "
                "prefer targeted reads and greps over whole files."
            ),
        },
    ]
    limited_reason = None
    started_at = time.time()

    for step in range(max_steps):
        send(
            {
                "type": "event",
                "event": {
                    "t": round(time.time() - started_at, 2),
                    "kind": "model_call",
                    "tool": None,
                    "target": None,
                },
            }
        )
        send({"type": "model_request", "messages": messages, "tools": TOOLS})
        response = read_line()
        if response.get("type") == "limit":
            limited_reason = response.get("reason")
            # The limit message carries the last assistant message; use it if
            # it is a final answer, otherwise wrap up with what we have.
            message = response.get("message") or {}
            if message.get("content"):
                messages.append({"role": "assistant", "content": message["content"]})
            break
        if response.get("type") == "cancel":
            send({"type": "failed", "error": "cancelled by client"})
            return 0
        if response.get("type") == "eof":
            send({"type": "failed", "error": "broker closed the connection"})
            return 1
        if response.get("type") != "model_response":
            send({"type": "failed", "error": f"unexpected broker message: {response.get('type')}"})
            return 1
        message = response.get("message", {})
        messages.append({"role": "assistant", "content": message.get("content"), **({"tool_calls": message["tool_calls"]} if message.get("tool_calls") else {})})
        if message.get("content"):
            messages[-1]["content"] = message["content"] or ""

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            break
        for call in tool_calls:
            function = call.get("function", {})
            tool_name = function.get("name", "")
            try:
                args = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            send(
                {
                    "type": "event",
                    "event": {
                        "t": round(time.time() - started_at, 2),
                        "kind": "tool_call",
                        "tool": tool_name,
                        "target": args.get("path") or args.get("url") or args.get("query"),
                    },
                }
            )
            send({"type": "tool_request", "tool": tool_name, "args": args})
            tool_result = read_line()
            if tool_result.get("type") == "cancel":
                send({"type": "failed", "error": "cancelled by client"})
                return 0
            if tool_result.get("type") != "tool_result":
                send({"type": "failed", "error": f"expected tool_result, got {tool_result.get('type')}"})
                return 1
            evidence = tool_result.get("evidence")
            if tool_result.get("ok"):
                content = json.dumps(tool_result.get("result", {}), ensure_ascii=False)
            else:
                content = f"ERROR: {tool_result.get('error', 'unknown tool error')}"
            if evidence:
                content += (
                    "\n\nEVIDENCE RECEIPT (reference this evidenceId in claims): "
                    + json.dumps(evidence, ensure_ascii=False)
                )
            messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": content})
    else:
        # max_steps reached without a final answer: honest partial wrap-up.
        send({"type": "failed", "error": "worker exceeded maxSteps without a final answer"})
        return 1

    result = build_result(messages, role, model, packet.get("jobId", ""), limited_reason, max_steps)
    send({"type": "complete", "result": result})
    return 0


def build_result(messages, role, model, job_id, limited_reason, max_steps):
    final_text = ""
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("content"):
            final_text = message["content"]
            break
    claims = []
    conclusion = final_text
    confidence = 0.5
    findings = []
    risks = []
    recommended_action = None
    unresolved = []
    try:
        parsed = json.loads(final_text)
        if isinstance(parsed, dict):
            conclusion = parsed.get("conclusion", conclusion)
            confidence = float(parsed.get("confidence", confidence))
            claims = [
                {
                    "claim": item.get("claim", ""),
                    "claimStatus": item.get("claimStatus", "inferred"),
                    "confidence": float(item.get("confidence", 0.5)),
                    "evidence": item.get("evidence", []),
                }
                for item in parsed.get("claims", [])
                if isinstance(item, dict) and item.get("claim")
            ]
            findings = parsed.get("findings", [])
            risks = parsed.get("risks", [])
            recommended_action = parsed.get("recommendedAction")
            unresolved = parsed.get("unresolved", [])
    except json.JSONDecodeError:
        # The model did not return structured JSON: wrap the text honestly.
        claims = [
            {
                "claim": final_text,
                "claimStatus": "inferred",
                "confidence": confidence,
                "evidence": [],
            }
        ]
    status = "partial" if limited_reason else "complete"
    warnings = [f"limit reached: {limited_reason}"] if limited_reason else []
    return {
        "jobId": job_id,
        "status": status,
        "role": role,
        "provider": "touch",
        "model": model,
        "conclusion": conclusion,
        "confidence": confidence,
        "claims": claims,
        "findings": findings,
        "risks": risks,
        "recommendedAction": recommended_action,
        "unresolved": unresolved,
        "usage": {"steps": max_steps, "totalInputTokens": 0, "totalOutputTokens": 0,
                  "costEstimateUsd": 0.0, "latencyMs": 0},
        "warnings": warnings,
    }


if __name__ == "__main__":
    sys.exit(run())
