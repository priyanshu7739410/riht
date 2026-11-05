#!/usr/bin/env python3
"""
mock_vllm_server.py: a stand-in for the vLLM OpenAI-compatible server.

Use it to exercise pubmed_summarizer.py without a DGX Spark, including the
failure cases: slow responses, 503 overload and dropped connections.

    python mock_vllm_server.py --port 8000                  # healthy, instant
    python mock_vllm_server.py --port 8000 --delay 45       # 45 s per response
    python mock_vllm_server.py --port 8000 --fail-first 2   # two 503s, then OK
    python mock_vllm_server.py --port 8000 --drop           # close socket, no reply

Then, in another terminal:

    python pubmed_summarizer.py --endpoint http://127.0.0.1:8000/v1/chat/completions
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict

MODEL_NAME = "llama-3.3-70b"


class MockState:
    def __init__(self, delay: float = 0.0, fail_first: int = 0, drop: bool = False,
                 model: str = MODEL_NAME) -> None:
        self.delay = delay
        self.fail_first = fail_first
        self.drop = drop
        self.model = model
        self.requests = 0
        self.lock = threading.Lock()


def _fake_summary(user_message: str) -> str:
    match = re.search(r'Abstract:\n"""\n(.*?)\n"""', user_message, re.DOTALL)
    abstract = match.group(1) if match else user_message
    first = re.split(r"(?<=[.!?])\s", abstract.strip(), maxsplit=1)[0]
    return f"[mock] The study reports: {first}"


def make_handler(state: MockState):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # quieter logs
            print(f"[mock] {self.address_string()} {fmt % args}", flush=True)

        def _send(self, status: int, body: Dict[str, Any], headers: Dict[str, str] = None):
            raw = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            if self.path.rstrip("/") == "/v1/models":
                self._send(200, {"object": "list",
                                 "data": [{"id": state.model, "object": "model"}]})
            elif self.path == "/health":
                self._send(200, {})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            if self.path != "/v1/chat/completions":
                self._send(404, {"error": "not found"})
                return

            with state.lock:
                state.requests += 1
                n = state.requests

            if body.get("model") != state.model:
                self._send(404, {"error": {"message": f"The model `{body.get('model')}` "
                                                      f"does not exist."}})
                return
            if n <= state.fail_first:
                self._send(503, {"error": "server overloaded"}, {"Retry-After": "1"})
                return
            if state.delay:
                time.sleep(state.delay)
            if state.drop:
                # Simulate a crashed or restarted container: no response at all.
                self.close_connection = True
                self.connection.close()
                return

            user = next((m["content"] for m in body.get("messages", [])
                         if m.get("role") == "user"), "")
            self._send(200, {
                "id": f"chatcmpl-mock-{n}",
                "object": "chat.completion",
                "model": state.model,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant",
                                         "content": _fake_summary(user)}}],
                "usage": {"prompt_tokens": len(user) // 4, "completion_tokens": 20},
            })

    return Handler


def make_server(host: str = "127.0.0.1", port: int = 0, **kwargs: Any):
    """Create a server (port 0 = pick a free port). Returns (server, state)."""
    state = MockState(**kwargs)
    server = ThreadingHTTPServer((host, port), make_handler(state))
    server.daemon_threads = True
    return server, state


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--delay", type=float, default=0.0, help="seconds to wait before replying")
    p.add_argument("--fail-first", type=int, default=0, help="return 503 for the first N requests")
    p.add_argument("--drop", action="store_true", help="close the connection without replying")
    p.add_argument("--model", default=MODEL_NAME)
    args = p.parse_args()
    server, _ = make_server(args.host, args.port, delay=args.delay,
                            fail_first=args.fail_first, drop=args.drop, model=args.model)
    print(f"[mock] serving {args.model} on http://{args.host}:{args.port} "
          f"(delay={args.delay}s fail_first={args.fail_first} drop={args.drop})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
