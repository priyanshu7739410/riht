"""Tests for pubmed_summarizer. Run with:  python -m unittest -v"""

from __future__ import annotations

import io
import json
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mock_vllm_server  # noqa: E402
import pubmed_summarizer as ps  # noqa: E402


def sample_records():
    return ps.load_records(None)


class CleaningTests(unittest.TestCase):
    def setUp(self):
        self.articles, self.rejected = ps.clean_records(sample_records())

    def test_counts(self):
        self.assertEqual(len(self.articles), 2)
        self.assertEqual(len(self.rejected), 1)

    def test_record_one_canonical(self):
        a = self.articles[0]
        self.assertEqual(a.pmid, "349281")
        self.assertEqual(a.ref, "PMID:349281")
        self.assertEqual(a.title, "Efficacy of Drug X in severe cardiac failure.")
        self.assertEqual(a.authors, ["Smith, J", "Doe, A"])
        self.assertIn("500 patients", a.abstract)

    def test_record_two_integer_id_null_title_unknown_authors(self):
        a = self.articles[1]
        self.assertEqual(a.pmid, "8829102")
        self.assertIsNone(a.title)
        self.assertEqual(a.authors, [])
        self.assertIn("title missing", a.warnings)

    def test_word_null_in_abstract_is_not_treated_as_null(self):
        self.assertTrue(self.articles[1].abstract.startswith("Null hypothesis rejected."))

    def test_record_three_rejected(self):
        r = self.rejected[0]
        self.assertEqual(r.raw_id, "ERR_NOT_FOUND")
        self.assertEqual(r.ref, "row-3")
        self.assertIn("abstract", r.reason)

    def test_pmid_parsing(self):
        cases = {"PMID:349281": "349281", "pmid 00123": "123", 8829102: "8829102",
                 8829102.0: "8829102", "ERR_NOT_FOUND": None, None: None, True: None,
                 -5: None, "": None, "12ab": None}
        for raw, expected in cases.items():
            self.assertEqual(ps.parse_pmid(raw), expected, raw)

    def test_author_parsing(self):
        self.assertEqual(ps.parse_authors("Smith, J; Doe, A"), ["Smith, J", "Doe, A"])
        self.assertEqual(ps.parse_authors("Unknown"), [])
        self.assertEqual(ps.parse_authors([None, "", "Smith, J", "Smith, J"]), ["Smith, J"])
        self.assertEqual(ps.parse_authors([{"LastName": "Roe", "Initials": "R"}]), ["Roe, R"])
        self.assertEqual(ps.parse_authors(42), [])

    def test_messy_inputs(self):
        records = [
            "not a dict",
            {"pmid": "1", "Abstract": "  &lt;b&gt;Short&lt;/b&gt; "},
            {"pmid": "2", "abstract": [{"Label": "Results", "#text": "Mortality fell by 12% overall."}]},
            {"pmid": "2", "abstract": "A duplicate PMID with a long enough abstract."},
            {"PMID": "3", "Abstract_Text": "x" * 50, "title": "N/A"},
        ]
        articles, rejected = ps.clean_records(records, max_abstract_chars=40)
        reasons = [r.reason for r in rejected]
        self.assertTrue(any("not a JSON object" in r for r in reasons))
        self.assertTrue(any("too short" in r for r in reasons))
        self.assertTrue(any("duplicate" in r for r in reasons))
        self.assertEqual(articles[0].abstract, "RESULTS: Mortality fell by 12% overall.")
        self.assertIsNone(articles[1].title)
        self.assertLessEqual(len(articles[1].abstract), 40)
        self.assertTrue(any("truncated" in w for w in articles[1].warnings))

    def test_load_json_lines_and_wrapped(self):
        tmp = ROOT / "tests" / "_tmp_input.jsonl"
        try:
            tmp.write_text('{"id": 1, "abstract": "a"}\n{bad line}\n{"id": 2}\n', encoding="utf-8")
            self.assertEqual(len(ps.load_records(tmp)), 2)
            tmp.write_text('{"articles": [{"id": 1}, {"id": 2}, {"id": 3}]}', encoding="utf-8")
            self.assertEqual(len(ps.load_records(tmp)), 3)
            tmp.write_text("totally not json", encoding="utf-8")
            with self.assertRaises(ps.DataError):
                ps.load_records(tmp)
        finally:
            tmp.unlink(missing_ok=True)


class PayloadTests(unittest.TestCase):
    def test_payload_shape(self):
        article = ps.clean_records(sample_records())[0][1]
        payload = ps.build_payload(article, model="m", max_tokens=64)
        self.assertEqual(payload["model"], "m")
        self.assertEqual(payload["max_tokens"], 64)
        self.assertFalse(payload["stream"])
        user = payload["messages"][1]["content"]
        self.assertIn(ps.PROMPT_INSTRUCTION, user)
        self.assertIn("Title: Not available", user)
        self.assertIn("cohort B", user)


def _response(status=200, body=None, headers=None):
    resp = requests.Response()
    resp.status_code = status
    resp._content = json.dumps(body if body is not None else {}).encode()
    resp.headers.update(headers or {})
    return resp


OK_BODY = {"choices": [{"message": {"content": " Drug X helped 40%. "},
                        "finish_reason": "stop"}]}


class RetryTests(unittest.TestCase):
    def setUp(self):
        self.article = ps.clean_records(sample_records())[0][0]
        self.sleeps = []

    def client(self, side_effect, retries=3):
        session = mock.Mock(spec=requests.Session)
        session.headers = {}
        session.post.side_effect = side_effect
        c = ps.LLMClient("http://x/v1/chat/completions", "m", max_retries=retries,
                         session=session, sleep=self.sleeps.append)
        return c, session

    def test_read_timeout_then_success(self):
        c, s = self.client([requests.exceptions.ReadTimeout(), _response(200, OK_BODY)])
        result = c.summarize(self.article)
        self.assertEqual(result.summary, "Drug X helped 40%.")
        self.assertEqual(result.attempts, 2)
        self.assertEqual(len(self.sleeps), 1)

    def test_connection_dropped_every_time(self):
        c, s = self.client([requests.exceptions.ConnectionError("reset")] * 4)
        with self.assertRaises(ps.LLMError) as ctx:
            c.summarize(self.article)
        self.assertEqual(ctx.exception.attempts, 4)
        self.assertTrue(ctx.exception.server_unreachable)
        self.assertEqual(s.post.call_count, 4)
        # exponential backoff: each sleep >= the one before (minus jitter)
        self.assertTrue(self.sleeps[0] < self.sleeps[2])

    def test_503_honours_retry_after(self):
        c, _ = self.client([_response(503, {}, {"Retry-After": "7"}), _response(200, OK_BODY)])
        c.summarize(self.article)
        self.assertEqual(self.sleeps, [7.0])

    def test_400_not_retried(self):
        c, s = self.client([_response(400, {"error": "prompt too long"})])
        with self.assertRaises(ps.LLMError) as ctx:
            c.summarize(self.article)
        self.assertFalse(ctx.exception.retryable)
        self.assertEqual(s.post.call_count, 1)

    def test_malformed_body(self):
        c, _ = self.client([_response(200, {"unexpected": True})])
        with self.assertRaises(ps.LLMError):
            c.summarize(self.article)


class MockServerIntegrationTests(unittest.TestCase):
    """Talks real HTTP to mock_vllm_server on a random local port."""

    def serve(self, **kwargs):
        server, state = mock_vllm_server.make_server("127.0.0.1", 0, **kwargs)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        host, port = server.server_address
        return f"http://{host}:{port}/v1/chat/completions", state

    def run_pipeline(self, endpoint, **client_kwargs):
        articles, rejected = ps.clean_records(sample_records())
        client = ps.LLMClient(endpoint, mock_vllm_server.MODEL_NAME,
                              sleep=lambda s: None, **client_kwargs)
        out = io.StringIO()
        stats = ps.run(articles, rejected, client, out, max_consecutive_failures=2)
        rows = [json.loads(line) for line in out.getvalue().splitlines()]
        return stats, rows

    def test_happy_path(self):
        endpoint, _ = self.serve()
        stats, rows = self.run_pipeline(endpoint)
        self.assertEqual(stats, {"ok": 2, "failed": 0, "rejected": 1, "skipped": 0})
        self.assertEqual({r["status"] for r in rows}, {"ok", "rejected"})

    def test_slow_server_within_timeout_succeeds_first_try(self):
        endpoint, state = self.serve(delay=0.5)
        stats, rows = self.run_pipeline(endpoint, read_timeout=2.0)
        self.assertEqual(stats["ok"], 2)
        self.assertEqual(state.requests, 2)  # no retries, no duplicate GPU work

    def test_server_slower_than_timeout_trips_circuit_breaker(self):
        endpoint, state = self.serve(delay=0.6)
        stats, _ = self.run_pipeline(endpoint, read_timeout=0.2, max_retries=1)
        self.assertEqual(stats["failed"], 2)
        self.assertEqual(stats["ok"], 0)

    def test_dropped_connection(self):
        endpoint, _ = self.serve(drop=True)
        stats, rows = self.run_pipeline(endpoint, max_retries=1)
        self.assertEqual(stats["failed"], 2)
        self.assertIn("connection dropped", rows[1]["error"])

    def test_transient_503_recovers(self):
        endpoint, _ = self.serve(fail_first=1)
        stats, _ = self.run_pipeline(endpoint)
        self.assertEqual(stats["ok"], 2)

    def test_preflight_catches_wrong_model(self):
        endpoint, _ = self.serve()
        client = ps.LLMClient(endpoint, "wrong-model")
        with self.assertRaises(ps.DataError):
            client.preflight()


if __name__ == "__main__":
    unittest.main()
