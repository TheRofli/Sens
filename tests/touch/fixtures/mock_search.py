#!/usr/bin/env python3
"""Mock search provider (Tavily-shaped) for Touch contract tests (stdlib only).

POST /search with {"query": "...", "api_key": "..."} returns scripted results.
Without an api_key the server answers 401 so tests can assert the honest
"web_search disabled / missing key" behavior.

Usage:
    python mock_search.py --port 0 --results results.json
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

RESULTS_DEFAULT = {
    "query": "tavily pricing",
    "results": [
        {
            "title": "Tavily — pricing",
            "url": "https://docs.example.com/tavily/pricing",
            "content": "Tavily pricing starts at $0/month for 1000 credits.",
            "score": 0.95,
        }
    ],
}


class MockSearchHandler(BaseHTTPRequestHandler):
    results = RESULTS_DEFAULT

    def log_message(self, format, *args):  # keep stderr clean
        pass

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if not body.get("api_key"):
            self._json(401, {"detail": "Missing API key"})
            return
        query = body.get("query", "")
        results = [
            item
            for item in self.results.get("results", [])
            if query.lower() in item.get("title", "").lower()
            or query.lower() in item.get("content", "").lower()
        ]
        self._json(
            200,
            {
                "query": query,
                "results": results,
                "answer": None,
                "response_time": 0.42,
            },
        )

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
    parser.add_argument("--results", default=None)
    args = parser.parse_args()
    if args.results:
        with open(args.results, encoding="utf-8") as handle:
            MockSearchHandler.results = json.load(handle)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), MockSearchHandler)
    print(f"mock search on http://127.0.0.1:{server.server_address[1]}/search", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
