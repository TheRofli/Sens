"""Sens 1.4.0 Touch Slice 0 contract tests (stdlib only).

Validates the frozen fixtures against the v1.1 contract:
- two-axis semantics (claim_status vs evidence_status);
- evidence receipts carry sha256/observed_at and match the real file;
- workers cannot cite receipts that were never issued;
- dirty-repo generator, mock provider and mock search behave as documented;
- private-range URL fixture covers the SSRF policy categories.

Run: python -m unittest tests.touch.test_contracts (or pytest tests/touch)
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SOURCE = FIXTURES / "source_files"
RECEIPTS = FIXTURES / "receipts"

USE_SOCKET = SOURCE / "useSocket.ts"
INJECTION = SOURCE / "injection.md"
PRIVATE_URLS = FIXTURES / "private_urls.json"


def load(name: str):
    with open(RECEIPTS / name, encoding="utf-8") as handle:
        return json.load(handle)


class SourceFixtureTests(unittest.TestCase):
    def test_line_47_contains_setinterval(self):
        lines = USE_SOCKET.read_text(encoding="utf-8").splitlines()
        self.assertIn("setInterval", lines[46], "fixture target line must be line 47")

    def test_injection_file_contains_injection_marker(self):
        text = INJECTION.read_text(encoding="utf-8")
        self.assertIn("IGNORE PREVIOUS INSTRUCTIONS", text)
        self.assertIn("untrusted data", text)

    def test_receipt_sha256_matches_file(self):
        receipt = load("file_read_good.json")
        digest = hashlib.sha256(USE_SOCKET.read_bytes()).hexdigest()
        self.assertEqual(receipt["sha256"], digest)
        self.assertTrue(receipt["observedAt"].endswith("Z"))
        self.assertEqual(receipt["range"], [47, 47])
        self.assertEqual(receipt["kind"], "file_read")


class TwoAxisContractTests(unittest.TestCase):
    def test_semantic_claim_stays_inferred(self):
        result = load("worker_result_auto.json")
        claims = {item["claim"] for item in result["claims"]}
        semantic = next(item for item in result["claims"] if "не очищается" in item["claim"])
        self.assertEqual(semantic["claimStatus"], "inferred")
        self.assertLessEqual(semantic["confidence"], 1.0)
        self.assertGreaterEqual(semantic["confidence"], 0.0)
        # verified evidence under an inferred conclusion is the normal case
        self.assertTrue(
            all(e["evidenceStatus"] == "verified" for e in semantic["evidence"])
        )

    def test_predicate_claim_may_be_verified(self):
        result = load("worker_result_auto.json")
        predicate = next(item for item in result["claims"] if item["claimStatus"] == "verified")
        self.assertIn("setInterval", predicate["claim"])
        self.assertTrue(predicate["evidence"], "verified claim must carry evidence")

    def test_nonexistent_evidence_ref_is_flagged(self):
        result = load("worker_result_nonexistent_ref.json")
        warnings = result["warnings"]
        self.assertTrue(
            any("ev_does_not_exist" in warning for warning in warnings),
            "broker must flag references to receipts it never issued",
        )
        claim = result["claims"][0]
        self.assertEqual(claim["evidence"][0]["evidenceStatus"], "unverifiable")

    def test_cumulative_usage_shape(self):
        result = load("worker_result_auto.json")
        usage = result["usage"]
        for key in ("steps", "totalInputTokens", "totalOutputTokens", "costEstimateUsd", "latencyMs"):
            self.assertIn(key, usage)
        # cumulative semantics: total input is the SUM over model calls, so
        # it must be >= the tokens of any single call; fixture documents 24k.
        self.assertGreater(usage["totalInputTokens"], 12_000)

    def test_coding_result_is_patch_producer(self):
        result = load("worker_result_coding.json")
        self.assertIn("patch", result)
        self.assertIn("testsRequired", result)
        self.assertNotIn("testsRun", result, "v1.1: no test execution")


class DirtyRepoFixtureTests(unittest.TestCase):
    def test_generator_produces_expected_dirty_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = FIXTURES / "make_dirty_repo.py"
            target = Path(tmp) / "repo"
            subprocess.run(
                [sys.executable, str(script), "--dir", str(target)],
                check=True,
                capture_output=True,
                text=True,
            )
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=target,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.splitlines()
            kinds = {line[:2] for line in status}
            self.assertIn(" M", kinds, "tracked modified file must be dirty")
            self.assertIn("M ", kinds, "staged modification must exist")
            self.assertIn("A ", kinds, "newly staged file must exist")
            self.assertIn("??", kinds, "untracked file must exist")
            self.assertTrue((target / "injection.md").exists())


class MockProviderTests(unittest.TestCase):
    def test_provider_serves_tool_call_then_final(self):
        import json as _json

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "requests.jsonl"
            proc = subprocess.Popen(
                [
                    sys.executable,
                    str(FIXTURES / "mock_provider.py"),
                    "--port", "0",
                    "--log", str(log_path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            try:
                line = proc.stdout.readline().strip()
                base = line.split(" ")[-1]  # ends with /v1
                url = base + "/chat/completions"
                payload = _json.dumps(
                    {
                        "model": "mock/deepseek-v4-flash-0731",
                        "messages": [{"role": "user", "content": "ping"}],
                        "tools": [{"type": "function", "function": {"name": "read"}}],
                    }
                ).encode()
                request = urllib.request.Request(
                    url, data=payload, headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(request, timeout=10) as response:
                    first = _json.loads(response.read())
                self.assertEqual(first["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "read")

                request2 = urllib.request.Request(
                    url,
                    data=_json.dumps(
                        {"model": "m", "messages": [{"role": "user", "content": "x"}]}
                    ).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request2, timeout=10) as response:
                    second = _json.loads(response.read())
                self.assertEqual(second["choices"][0]["finish_reason"], "stop")

                log_records = [
                    _json.loads(line)
                    for line in log_path.read_text(encoding="utf-8").splitlines()
                ]
                self.assertEqual(len(log_records), 2)
                self.assertTrue(log_records[0]["has_tools"])
                self.assertIn("auth_present", log_records[0])
            finally:
                proc.terminate()
                proc.wait(timeout=10)


class MockSearchTests(unittest.TestCase):
    def test_search_requires_key_and_filters(self):
        import json as _json

        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results.json"
            results.write_text(
                _json.dumps(
                    {
                        "results": [
                            {"title": "Tavily pricing page", "url": "https://a.example/", "content": "pricing 1000 credits"},
                            {"title": "Unrelated", "url": "https://b.example/", "content": "nothing"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            proc = subprocess.Popen(
                [sys.executable, str(FIXTURES / "mock_search.py"), "--port", "0", "--results", str(results)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            try:
                line = proc.stdout.readline().strip()
                base = line.split(" ")[-1]  # ends with /search
                url = base

                no_key = urllib.request.Request(
                    url,
                    data=_json.dumps({"query": "tavily"}).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    urllib.request.urlopen(no_key, timeout=10)
                self.assertEqual(ctx.exception.code, 401)

                with_key = urllib.request.Request(
                    url,
                    data=_json.dumps({"query": "tavily", "api_key": "tvly-test"}).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(with_key, timeout=10) as response:
                    body = _json.loads(response.read())
                self.assertEqual(len(body["results"]), 1)
                self.assertEqual(body["results"][0]["title"], "Tavily pricing page")
            finally:
                proc.terminate()
                proc.wait(timeout=10)


class PrivateUrlFixtureTests(unittest.TestCase):
    def test_categories_cover_ssrf_policy(self):
        fixture = json.loads(PRIVATE_URLS.read_text(encoding="utf-8"))
        categories = fixture["categories"]
        for key in (
            "loopback", "private", "link_local", "multicast", "reserved",
            "ipv6_private", "unspecified",
        ):
            self.assertIn(key, categories, f"missing SSRF category {key}")
            self.assertTrue(categories[key], f"empty category {key}")
        # Provider endpoints are a user-chosen trust boundary, not SSRF.
        self.assertIn("must_allow", fixture)
        self.assertTrue(any("openrouter" in url for url in fixture["must_allow"]))


if __name__ == "__main__":
    unittest.main()
