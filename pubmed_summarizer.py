#!/usr/bin/env python3
"""
pubmed_summarizer.py
====================

Cleans messy PubMed abstract records from a legacy export and asks a locally
hosted Llama 3.3 70B (vLLM, OpenAI-compatible API) to summarise the clinical
outcome of each study in one sentence.

Pipeline
--------
1. LOAD     Read JSON (or JSON Lines as a fallback) from --input. If no input
            is given, the sample records from the assignment are used.
2. CLEAN    Normalise inconsistent keys, IDs, titles, abstracts and authors.
            Records with no usable abstract are rejected *before* they reach
            the GPU and are written to the output with the reason.
3. PROMPT   Build an OpenAI-style chat-completions payload per record.
4. SEND     POST to the local vLLM server, one request at a time, with
            separate connect/read timeouts, bounded retries, exponential
            backoff with jitter, and a circuit breaker for a dead server.
5. WRITE    Stream one JSON line per record to --output as it finishes, so
            a crash or Ctrl+C never loses completed work.

Exit codes
----------
0  every valid record was summarised
1  finished, but at least one record failed after all retries
2  bad input or configuration (unreadable JSON, wrong model name, ...)
3  aborted early: the server looked down (too many failures in a row)
130 interrupted by the user (Ctrl+C)

Usage
-----
    python pubmed_summarizer.py                      # bundled sample data
    python pubmed_summarizer.py -i data/raw_abstracts.json -o results.jsonl
    python pubmed_summarizer.py --read-timeout 180 --max-retries 4 -v

Every option can also be set with an environment variable (see --help).

Dependencies: Python 3.9+, requests.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import os
import random
import re
import sys
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, TextIO, Tuple

import requests

log = logging.getLogger("pubmed_summarizer")

# --------------------------------------------------------------------------- #
# Configuration defaults
# --------------------------------------------------------------------------- #

DEFAULT_ENDPOINT = "http://192.168.1.100:8000/v1/chat/completions"
# Must match `--served-model-name` in the vLLM command (see README.md).
DEFAULT_MODEL = "llama-3.3-70b"

# Connect timeout: a TCP handshake on the LAN takes milliseconds. If we cannot
# connect within 5 s the box or the container is down, so fail fast.
DEFAULT_CONNECT_TIMEOUT_S = 5.0
# Read timeout: a 70B model on a single DGX Spark decodes at single-digit
# tokens/s, and stream=False means no bytes arrive until the answer is done.
# 45 s is a normal slow response, not a failure, so allow 120 s.
DEFAULT_READ_TIMEOUT_S = 120.0
# Responses slower than this still succeed but are logged as a warning.
SLOW_RESPONSE_WARN_S = 30.0

DEFAULT_MAX_RETRIES = 3            # retries after the first attempt
DEFAULT_BACKOFF_BASE_S = 2.0       # 2 s, 4 s, 8 s ... plus jitter
DEFAULT_BACKOFF_MAX_S = 30.0
DEFAULT_MAX_CONSECUTIVE_FAILURES = 3

# Keep every request well inside the server's --max-model-len (8192):
# ~6000 chars is ~1500 tokens; the prompt plus 128 output tokens is then far
# below the limit, so the server never has to reject or preempt.
DEFAULT_MAX_ABSTRACT_CHARS = 6000
DEFAULT_MAX_TOKENS = 128
MIN_ABSTRACT_CHARS = 20

PROMPT_INSTRUCTION = "Summarize the clinical outcome of this study in one sentence."

SYSTEM_PROMPT = (
    "You are a careful clinical research assistant. Use only the information "
    "in the abstract you are given. Do not invent numbers, outcomes, drug "
    "names or conclusions. If the abstract reports no clinical outcome, say "
    "so. Reply with exactly one sentence and nothing else."
)

RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

# The raw records from the assignment, used when --input is not given so this
# file runs on its own.
SAMPLE_DATA = r"""
[
  {
    "id": "PMID:349281",
    "ArticleTitle": "Efficacy of Drug X in severe cardiac failure.",
    "abstract_text": "We tested 500 patients. The drug worked in 40% of cases but caused nausea.",
    "authorList": ["Smith, J", "Doe, A"]
  },
  {
    "id": 8829102,
    "title": null,
    "Abstract": "Null hypothesis rejected. No significant improvement noted in cohort B vs placebo.",
    "authors": "Unknown"
  },
  {
    "id": "ERR_NOT_FOUND",
    "ArticleTitle": "Corrupted Entry",
    "abstract_text": null
  }
]
"""

# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class DataError(Exception):
    """The input file cannot be read or parsed at all."""


class LLMError(Exception):
    """A request to the model failed for good (after retries, or not retryable)."""

    def __init__(self, message: str, *, attempts: int, retryable: bool,
                 server_unreachable: bool = False) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.retryable = retryable
        # True when the failure means "the server is down", not "this record
        # is bad". The circuit breaker counts only these.
        self.server_unreachable = server_unreachable


# --------------------------------------------------------------------------- #
# Data cleaning
# --------------------------------------------------------------------------- #

# Normalised key -> canonical field. Keys are compared after lower-casing and
# removing everything except letters and digits, so "ArticleTitle",
# "article_title" and "Article Title" all match.
FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "id": ("pmid", "id", "articleid", "pubmedid", "uid"),
    "title": ("articletitle", "title"),
    "abstract": ("abstracttext", "abstract", "abstr", "summary"),
    "authors": ("authorlist", "authors", "author"),
}

# Placeholder values the legacy export uses for "no data". They are matched
# against the WHOLE cleaned value, never as a substring: an abstract that
# begins "Null hypothesis rejected..." is real data and must be kept.
PLACEHOLDERS = frozenset({
    "", "null", "none", "nil", "nan", "n/a", "na", "-", "--", "?",
    "unknown", "undefined", "missing", "not available", "not found",
    "corrupted entry", "corrupted", "err_not_found", "error",
})

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE = re.compile(r"\s+")
_HTML_TAGS = re.compile(r"<[^>]{1,200}>")
_PMID_PREFIX = re.compile(r"^\s*(?:pmid|pubmed)\s*[:#]?\s*", re.IGNORECASE)
_PMID_DIGITS = re.compile(r"^\d{1,10}$")


def _norm_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def clean_text(value: Any) -> Optional[str]:
    """Return a cleaned string, or None if the value carries no information.

    Handles None, non-strings, HTML entities and tags, odd Unicode, control
    characters, repeated whitespace and placeholder values like "Unknown".
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        # A bare number is not a title or an abstract.
        return None
    if not isinstance(value, str):
        return None
    text = html.unescape(value)
    text = _HTML_TAGS.sub(" ", text)
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_CHARS.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    if text.lower().strip(" .") in PLACEHOLDERS:
        return None
    return text


def clean_abstract(value: Any) -> Optional[str]:
    """Abstracts arrive as a string, or (in PubMed XML-to-JSON exports) as a
    list of sections or a dict such as {"#text": ...}. Flatten all of these."""
    if isinstance(value, list):
        parts = [clean_abstract(v) for v in value]
        joined = " ".join(p for p in parts if p)
        return clean_text(joined)
    if isinstance(value, dict):
        label = clean_text(value.get("Label") or value.get("label"))
        body = None
        for key in ("#text", "text", "AbstractText", "value", "content"):
            if key in value:
                body = clean_abstract(value[key])
                if body:
                    break
        if body and label:
            return f"{label.upper()}: {body}"
        return body
    return clean_text(value)


def parse_pmid(value: Any) -> Optional[str]:
    """Normalise a PubMed ID to a digit string.

    "PMID:349281" -> "349281", 8829102 -> "8829102", "ERR_NOT_FOUND" -> None.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value) if value > 0 else None
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() and value > 0 else None
    if isinstance(value, str):
        candidate = _PMID_PREFIX.sub("", value.strip())
        if _PMID_DIGITS.match(candidate) and int(candidate) > 0:
            return str(int(candidate))  # drop leading zeros
    return None


def parse_authors(value: Any) -> List[str]:
    """Return a list of author names; an unknown or missing value gives [].

    Strings are split on ";" or "|" only, never on ",", because PubMed
    names look like "Smith, J".
    """
    if value is None:
        return []
    if isinstance(value, dict):
        # {"LastName": "Smith", "Initials": "J"} or {"Author": [...]}
        if "Author" in value:
            return parse_authors(value["Author"])
        last = clean_text(value.get("LastName") or value.get("lastName"))
        first = clean_text(value.get("Initials") or value.get("ForeName")
                           or value.get("firstName"))
        name = ", ".join(p for p in (last, first) if p)
        if not name:
            name = clean_text(value.get("CollectiveName") or value.get("name")) or ""
        return [name] if name else []
    if isinstance(value, (list, tuple)):
        authors: List[str] = []
        for item in value:
            authors.extend(parse_authors(item))
        # De-duplicate but keep order.
        return list(dict.fromkeys(authors))
    if isinstance(value, str):
        text = clean_text(value)
        if not text:
            return []
        parts = (clean_text(p) for p in re.split(r"[;|]", text))
        return [p for p in parts if p]
    return []


@dataclass
class Article:
    """One cleaned record, ready to send to the model."""

    ref: str                      # stable reference for logs and output
    pmid: Optional[str]
    title: Optional[str]
    abstract: str
    authors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class Rejected:
    """A record that was dropped during cleaning, and why."""

    ref: str
    index: int
    raw_id: Any
    reason: str


def _pick(record: Dict[str, Any], canonical: str) -> Tuple[Any, Optional[str]]:
    """Return (value, source_key) for the first alias that holds a value."""
    normalised = {_norm_key(k): k for k in record}
    for alias in FIELD_ALIASES[canonical]:
        original_key = normalised.get(alias)
        if original_key is not None and record[original_key] is not None:
            return record[original_key], original_key
    return None, None


def clean_records(raw_records: Iterable[Any], *,
                  max_abstract_chars: int = DEFAULT_MAX_ABSTRACT_CHARS
                  ) -> Tuple[List[Article], List[Rejected]]:
    """Turn raw, inconsistent records into Article objects.

    Rules:
      * Not a JSON object                -> rejected
      * No usable abstract               -> rejected (nothing to summarise;
                                            never spend GPU time on it)
      * Abstract shorter than 20 chars   -> rejected
      * Duplicate PMID                   -> rejected (keep the first)
      * Bad or missing ID                -> kept, with ref "row-<n>" and a warning
      * Missing title / authors          -> kept; title None, authors []
      * Very long abstract               -> truncated, with a warning
    """
    articles: List[Article] = []
    rejected: List[Rejected] = []
    seen_pmids: set = set()

    for index, record in enumerate(raw_records, start=1):
        if not isinstance(record, dict):
            rejected.append(Rejected(f"row-{index}", index, None,
                                     f"not a JSON object ({type(record).__name__})"))
            continue

        raw_id, _ = _pick(record, "id")
        pmid = parse_pmid(raw_id)
        ref = f"PMID:{pmid}" if pmid else f"row-{index}"
        warnings: List[str] = []
        if pmid is None:
            warnings.append(f"invalid or missing id {raw_id!r}")

        raw_abstract, _ = _pick(record, "abstract")
        abstract = clean_abstract(raw_abstract)
        if not abstract:
            rejected.append(Rejected(ref, index, raw_id, "missing or empty abstract"))
            continue
        if len(abstract) < MIN_ABSTRACT_CHARS:
            rejected.append(Rejected(ref, index, raw_id,
                                     f"abstract too short ({len(abstract)} chars)"))
            continue

        if pmid is not None:
            if pmid in seen_pmids:
                rejected.append(Rejected(ref, index, raw_id, "duplicate PMID"))
                continue
            seen_pmids.add(pmid)

        if len(abstract) > max_abstract_chars:
            cut = abstract[:max_abstract_chars].rsplit(" ", 1)[0]
            warnings.append(f"abstract truncated from {len(abstract)} to {len(cut)} chars")
            abstract = cut

        title_value, _ = _pick(record, "title")
        title = clean_text(title_value)
        if title is None:
            warnings.append("title missing")

        authors_value, _ = _pick(record, "authors")
        authors = parse_authors(authors_value)

        articles.append(Article(ref=ref, pmid=pmid, title=title, abstract=abstract,
                                authors=authors, warnings=warnings))

    return articles, rejected


def load_records(path: Optional[Path]) -> List[Any]:
    """Load a list of records from JSON or JSON Lines.

    The top level may be a list, a single object, or an object wrapping the
    list (e.g. {"articles": [...]}).
    """
    if path is None:
        text = SAMPLE_DATA
        source = "<bundled sample data>"
    else:
        try:
            # utf-8-sig strips a BOM; "replace" survives stray bad bytes.
            text = path.read_text(encoding="utf-8-sig", errors="replace")
        except OSError as exc:
            raise DataError(f"cannot read {path}: {exc}") from exc
        source = str(path)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        data = _try_json_lines(text)
        if data is None:
            raise DataError(f"{source} is not valid JSON or JSON Lines "
                            f"(line {exc.lineno}, column {exc.colno}: {exc.msg})") from exc
        log.warning("%s parsed as JSON Lines (%d records)", source, len(data))

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("articles", "records", "data", "results", "items", "PubmedArticle"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    raise DataError(f"{source}: expected a list of records, got {type(data).__name__}")


def _try_json_lines(text: str) -> Optional[List[Any]]:
    records: List[Any] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip().rstrip(",")
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            log.warning("skipping unparseable line %d", lineno)
    return records or None


# --------------------------------------------------------------------------- #
# LLM client
# --------------------------------------------------------------------------- #


def build_payload(article: Article, *, model: str,
                  max_tokens: int = DEFAULT_MAX_TOKENS,
                  temperature: float = 0.1) -> Dict[str, Any]:
    """OpenAI-compatible chat-completions body for one article."""
    title = article.title or "Not available"
    user_message = (
        f"{PROMPT_INSTRUCTION}\n\n"
        f"Title: {title}\n\n"
        f"Abstract:\n\"\"\"\n{article.abstract}\n\"\"\""
    )
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "temperature": temperature,      # near-deterministic, factual output
        "max_tokens": max_tokens,        # a hard cap on output (and KV cache) per request
        "stream": False,
    }


@dataclass
class LLMResult:
    summary: str
    attempts: int
    latency_s: float
    finish_reason: Optional[str]
    usage: Optional[Dict[str, Any]]


class LLMClient:
    """A small, defensive client for vLLM's OpenAI-compatible endpoint."""

    def __init__(self, endpoint: str, model: str, *,
                 connect_timeout: float = DEFAULT_CONNECT_TIMEOUT_S,
                 read_timeout: float = DEFAULT_READ_TIMEOUT_S,
                 max_retries: int = DEFAULT_MAX_RETRIES,
                 backoff_base: float = DEFAULT_BACKOFF_BASE_S,
                 backoff_max: float = DEFAULT_BACKOFF_MAX_S,
                 max_tokens: int = DEFAULT_MAX_TOKENS,
                 api_key: Optional[str] = None,
                 session: Optional[requests.Session] = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.endpoint = endpoint
        self.model = model
        self.timeout = (connect_timeout, read_timeout)
        self.max_retries = max(0, max_retries)
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.max_tokens = max_tokens
        self._sleep = sleep
        self.session = session or requests.Session()
        self.session.headers.update({"Content-Type": "application/json",
                                     "Accept": "application/json",
                                     "User-Agent": "pubmed-summarizer/1.0"})
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    # -- helpers ------------------------------------------------------------ #

    def _backoff(self, attempt: int, retry_after: Optional[float] = None) -> float:
        """Exponential backoff with full jitter; honours Retry-After."""
        if retry_after is not None:
            return min(self.backoff_max, max(0.0, retry_after))
        delay = min(self.backoff_max, self.backoff_base * (2 ** (attempt - 1)))
        return delay + random.uniform(0, self.backoff_base)

    @staticmethod
    def _retry_after(resp: requests.Response) -> Optional[float]:
        value = resp.headers.get("Retry-After")
        try:
            return float(value) if value is not None else None
        except ValueError:
            return None

    def models_url(self) -> str:
        base = self.endpoint.split("/v1/", 1)[0]
        return f"{base}/v1/models"

    def preflight(self) -> None:
        """Check the server is up and serving our model name.

        A wrong model name gives HTTP 404 on every request, so we catch it
        once, here, instead of 3 retries x N records later. If the server
        is merely down, only warn: the retry loop and circuit breaker
        handle that.
        """
        try:
            resp = self.session.get(self.models_url(), timeout=self.timeout[0] * 2)
            resp.raise_for_status()
            served = [m.get("id") for m in resp.json().get("data", [])]
        except (requests.RequestException, ValueError) as exc:
            log.warning("preflight: could not list models at %s (%s); continuing",
                        self.models_url(), exc.__class__.__name__)
            return
        if served and self.model not in served:
            raise DataError(f"model {self.model!r} is not served by {self.models_url()}; "
                            f"available: {served}. Use --model.")
        log.info("preflight OK: server is serving %s", self.model)

    # -- main call ---------------------------------------------------------- #

    def summarize(self, article: Article) -> LLMResult:
        payload = build_payload(article, model=self.model, max_tokens=self.max_tokens)
        total_attempts = self.max_retries + 1
        last_error = "unknown error"
        unreachable = False

        for attempt in range(1, total_attempts + 1):
            request_id = f"{article.ref}-{uuid.uuid4().hex[:8]}"
            started = time.monotonic()
            retry_after: Optional[float] = None
            try:
                resp = self.session.post(self.endpoint, json=payload, timeout=self.timeout,
                                         headers={"X-Request-Id": request_id})
                latency = time.monotonic() - started

                if resp.status_code == 200:
                    return self._parse_success(resp, article, attempt, latency)

                body = resp.text[:300].replace("\n", " ")
                last_error = f"HTTP {resp.status_code}: {body}"
                unreachable = resp.status_code in (502, 503, 504)
                if resp.status_code not in RETRYABLE_STATUS:
                    # 400 (bad prompt / too long), 401, 404 (wrong model) ...
                    # will fail the same way every time, so do not retry.
                    raise LLMError(last_error, attempts=attempt, retryable=False)
                retry_after = self._retry_after(resp)

            except requests.exceptions.ConnectTimeout:
                last_error = (f"connect timeout after {self.timeout[0]:.0f}s "
                              f"(host unreachable or server not listening)")
                unreachable = True
            except requests.exceptions.ReadTimeout:
                # Connected, but no response within the read timeout. Our socket
                # closes; vLLM sees the disconnect and aborts the request, which
                # frees its KV cache, so a retry does not double the GPU load.
                last_error = (f"read timeout: no response within {self.timeout[1]:.0f}s "
                              f"(server overloaded or generation too slow)")
                unreachable = True
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.ChunkedEncodingError) as exc:
                # Refused, reset by peer, RemoteDisconnected, or a body cut off
                # halfway (e.g. the container restarted).
                last_error = f"connection dropped: {exc.__class__.__name__}: {_short(exc)}"
                unreachable = True
            except requests.exceptions.RequestException as exc:
                last_error = f"request failed: {exc.__class__.__name__}: {_short(exc)}"
                raise LLMError(last_error, attempts=attempt, retryable=False) from exc

            if attempt < total_attempts:
                delay = self._backoff(attempt, retry_after)
                log.warning("%s attempt %d/%d failed: %s; retrying in %.1fs",
                            article.ref, attempt, total_attempts, last_error, delay)
                self._sleep(delay)
            else:
                log.error("%s attempt %d/%d failed: %s; giving up",
                          article.ref, attempt, total_attempts, last_error)

        raise LLMError(last_error, attempts=total_attempts, retryable=True,
                       server_unreachable=unreachable)

    def _parse_success(self, resp: requests.Response, article: Article,
                       attempt: int, latency: float) -> LLMResult:
        try:
            body = resp.json()
            choice = body["choices"][0]
            content = choice["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"malformed response body: {resp.text[:200]!r}",
                           attempts=attempt, retryable=False) from exc

        summary = _clean_summary(content)
        if not summary:
            raise LLMError("model returned an empty summary", attempts=attempt,
                           retryable=False)

        finish_reason = choice.get("finish_reason")
        if finish_reason == "length":
            log.warning("%s summary hit max_tokens=%d and may be cut off",
                        article.ref, self.max_tokens)
        if latency > SLOW_RESPONSE_WARN_S:
            log.warning("%s slow response: %.1fs (read timeout is %.0fs)",
                        article.ref, latency, self.timeout[1])
        return LLMResult(summary=summary, attempts=attempt, latency_s=latency,
                         finish_reason=finish_reason, usage=body.get("usage"))


def _clean_summary(content: Any) -> str:
    if not isinstance(content, str):
        return ""
    text = _WHITESPACE.sub(" ", content).strip()
    text = re.sub(r"^(?:summary|clinical outcome)\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip().strip('"').strip()


def _short(exc: BaseException, limit: int = 160) -> str:
    text = str(exc)
    return text if len(text) <= limit else text[:limit] + "..."


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def _write(out: TextIO, row: Dict[str, Any]) -> None:
    out.write(json.dumps(row, ensure_ascii=False) + "\n")
    out.flush()  # one line per finished record survives a crash or Ctrl+C


def run(articles: Sequence[Article], rejected: Sequence[Rejected], client: LLMClient,
        out: TextIO, *, max_consecutive_failures: int) -> Dict[str, int]:
    stats = {"ok": 0, "failed": 0, "rejected": len(rejected), "skipped": 0}

    for item in rejected:
        log.warning("rejected %s (input row %d): %s", item.ref, item.index, item.reason)
        _write(out, {"ref": item.ref, "pmid": None, "status": "rejected",
                     "reason": item.reason, "raw_id": item.raw_id})

    consecutive_down = 0
    for position, article in enumerate(articles):
        for warning in article.warnings:
            log.info("%s: %s", article.ref, warning)
        log.info("summarising %s (%d/%d)", article.ref, position + 1, len(articles))
        row: Dict[str, Any] = {"ref": article.ref, "pmid": article.pmid,
                               "title": article.title, "authors": article.authors,
                               "warnings": article.warnings}
        try:
            result = client.summarize(article)
        except LLMError as exc:
            stats["failed"] += 1
            row.update(status="failed", error=str(exc), attempts=exc.attempts)
            _write(out, row)
            consecutive_down = consecutive_down + 1 if exc.server_unreachable else 0
            if consecutive_down >= max_consecutive_failures:
                remaining = articles[position + 1:]
                log.critical("server unavailable for %d records in a row; aborting "
                             "the remaining %d records", consecutive_down, len(remaining))
                for skipped in remaining:
                    stats["skipped"] += 1
                    _write(out, {"ref": skipped.ref, "pmid": skipped.pmid,
                                 "status": "skipped", "error": "run aborted: server unavailable"})
                break
            continue

        consecutive_down = 0
        stats["ok"] += 1
        row.update(status="ok", summary=result.summary, attempts=result.attempts,
                   latency_s=round(result.latency_s, 2),
                   finish_reason=result.finish_reason, usage=result.usage)
        _write(out, row)
        log.info("%s done in %.1fs: %s", article.ref, result.latency_s, result.summary)

    return stats


def _env(name: str, default: Any, cast: Callable[[str], Any] = str) -> Any:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return cast(value)
    except ValueError:
        raise SystemExit(f"invalid value for {name}: {value!r}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Clean PubMed records and summarise clinical outcomes with a local LLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-i", "--input", type=Path, default=_env("PS_INPUT", None, Path),
                   help="JSON / JSON Lines file (env PS_INPUT). Omit to use the bundled sample.")
    p.add_argument("-o", "--output", default=_env("PS_OUTPUT", "-"),
                   help="JSON Lines output path, '-' for stdout (env PS_OUTPUT)")
    p.add_argument("--endpoint", default=_env("PS_ENDPOINT", DEFAULT_ENDPOINT),
                   help="chat-completions URL (env PS_ENDPOINT)")
    p.add_argument("--model", default=_env("PS_MODEL", DEFAULT_MODEL),
                   help="served model name (env PS_MODEL)")
    p.add_argument("--api-key", default=os.environ.get("PS_API_KEY"),
                   help="bearer token if vLLM runs with --api-key (env PS_API_KEY)")
    p.add_argument("--connect-timeout", type=float,
                   default=_env("PS_CONNECT_TIMEOUT", DEFAULT_CONNECT_TIMEOUT_S, float))
    p.add_argument("--read-timeout", type=float,
                   default=_env("PS_READ_TIMEOUT", DEFAULT_READ_TIMEOUT_S, float))
    p.add_argument("--max-retries", type=int,
                   default=_env("PS_MAX_RETRIES", DEFAULT_MAX_RETRIES, int))
    p.add_argument("--backoff-base", type=float,
                   default=_env("PS_BACKOFF_BASE", DEFAULT_BACKOFF_BASE_S, float))
    p.add_argument("--max-tokens", type=int,
                   default=_env("PS_MAX_TOKENS", DEFAULT_MAX_TOKENS, int))
    p.add_argument("--max-abstract-chars", type=int,
                   default=_env("PS_MAX_ABSTRACT_CHARS", DEFAULT_MAX_ABSTRACT_CHARS, int))
    p.add_argument("--max-consecutive-failures", type=int,
                   default=_env("PS_MAX_CONSECUTIVE_FAILURES",
                                DEFAULT_MAX_CONSECUTIVE_FAILURES, int),
                   help="stop the run after this many records fail in a row because "
                        "the server is down")
    p.add_argument("--no-preflight", action="store_true",
                   help="skip the GET /v1/models check at startup")
    p.add_argument("--dry-run", action="store_true",
                   help="clean the data and print the payloads; send nothing")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(message)s",
                        stream=sys.stderr)

    try:
        raw = load_records(args.input)
    except DataError as exc:
        log.error("%s", exc)
        return 2

    articles, rejected = clean_records(raw, max_abstract_chars=args.max_abstract_chars)
    log.info("loaded %d records: %d valid, %d rejected", len(raw), len(articles), len(rejected))

    if args.dry_run:
        for item in rejected:
            log.warning("rejected %s: %s", item.ref, item.reason)
        for article in articles:
            print(json.dumps({"ref": article.ref,
                              "payload": build_payload(article, model=args.model,
                                                       max_tokens=args.max_tokens)},
                             ensure_ascii=False, indent=2))
        return 0

    if not articles:
        log.warning("no valid records to summarise")

    client = LLMClient(args.endpoint, args.model,
                       connect_timeout=args.connect_timeout,
                       read_timeout=args.read_timeout,
                       max_retries=args.max_retries,
                       backoff_base=args.backoff_base,
                       max_tokens=args.max_tokens,
                       api_key=args.api_key)

    if not args.no_preflight and articles:
        try:
            client.preflight()
        except DataError as exc:
            log.error("%s", exc)
            return 2

    out: TextIO
    if args.output == "-":
        out = sys.stdout
    else:
        try:
            out = open(args.output, "w", encoding="utf-8")
        except OSError as exc:
            log.error("cannot open output %s: %s", args.output, exc)
            return 2

    try:
        stats = run(articles, rejected, client, out,
                    max_consecutive_failures=args.max_consecutive_failures)
    except KeyboardInterrupt:
        log.warning("interrupted; results finished so far are saved")
        return 130
    finally:
        if out is not sys.stdout:
            out.close()

    log.info("finished: %(ok)d ok, %(failed)d failed, %(rejected)d rejected, "
             "%(skipped)d skipped", stats)
    if stats["skipped"]:
        return 3
    if stats["failed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
