# card_tools.py — shared helpers for the "card" tools (places, sports, elections, recipe, drafts, docs).
# ─────────────────────────────────────────────────────────────────────────────
# A card tool returns the normal tool-result JSON the model reads:
#     {"success": True, "summary": "...", "card": {"type": "places", ...}}
# gpt2_native.blocks_from_result() lifts `card` out into a {"type": "card", "card": {...}} block, which
# ai_jobs.py writes into the reply doc's `blocks` array like any other block. See CARDS_CONTRACT.md.
#
# Rules every card tool follows:
#   * never raises — failures come back as {"success": False, "error": "..."} so the model can recover
#   * every network call has a timeout
#   * paid provider first (only when its key is set), free provider(s) after it
# ─────────────────────────────────────────────────────────────────────────────
import threading
import time
from typing import Any, Callable, Optional

import requests

UA = "OOOR-backend/1.0 (study assistant; https://github.com/Holixdave/python_backend_ai)"
TIMEOUT_S = 8


class ProviderError(Exception):
    """One provider failed — the chain moves on to the next."""


def http_json(method: str, url: str, *, params: Optional[dict] = None, headers: Optional[dict] = None,
              json_body: Optional[dict] = None, timeout: float = TIMEOUT_S) -> Any:
    h = {"User-Agent": UA, "Accept": "application/json"}
    h.update(headers or {})
    try:
        r = requests.request(method, url, params=params, headers=h, json=json_body, timeout=timeout)
    except requests.RequestException as e:
        raise ProviderError(f"network: {type(e).__name__}") from e
    if r.status_code >= 400:
        raise ProviderError(f"http {r.status_code}")
    try:
        return r.json()
    except ValueError as e:
        raise ProviderError("bad json") from e


class TTLCache:
    """Tiny in-process cache so repeated questions don't burn paid quota or hammer free services."""

    def __init__(self, ttl_s: int = 60, max_items: int = 256):
        self.ttl, self.max = ttl_s, max_items
        self._d: dict = {}
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            hit = self._d.get(key)
            if hit and time.time() - hit[0] < self.ttl:
                return hit[1]
            self._d.pop(key, None)
        return None

    def put(self, key, value):
        with self._lock:
            if len(self._d) >= self.max:
                self._d.pop(next(iter(self._d)))
            self._d[key] = (time.time(), value)
        return value


class Throttle:
    """At most one call per `gap_s` (Nominatim's usage policy is 1 request/second)."""

    def __init__(self, gap_s: float):
        self.gap, self._last, self._lock = gap_s, 0.0, threading.Lock()

    def wait(self):
        with self._lock:
            delay = self._last + self.gap - time.time()
            if delay > 0:
                time.sleep(delay)
            self._last = time.time()


def run_chain(providers: list, *args, **kwargs) -> tuple:
    """providers = [(name, fn), ...] in priority order. Returns (name, result) from the first that works.
    Raises ProviderError listing every failure if none do."""
    failures = []
    for name, fn in providers:
        try:
            result = fn(*args, **kwargs)
            if result is not None:
                return name, result
            failures.append(f"{name}: nothing")
        except ProviderError as e:
            failures.append(f"{name}: {e}")
        except Exception as e:                                  # a parser bug must not kill the turn either
            failures.append(f"{name}: {type(e).__name__}")
    raise ProviderError("; ".join(failures) or "no provider available")


def ok(summary: str, card: Optional[dict] = None, **extra) -> dict:
    out = {"success": True, "summary": summary, **extra}
    if card is not None:
        out["card"] = card
    return out


def fail(error: str, **extra) -> dict:
    return {"success": False, "error": error, **extra}


def clip(text: Any, n: int) -> str:
    s = "" if text is None else str(text)
    return s if len(s) <= n else s[: n - 1] + "…"


def to_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def to_int(v: Any) -> Optional[int]:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None
