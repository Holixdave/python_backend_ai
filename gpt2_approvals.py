#!/usr/bin/env python3
# gpt2_approvals.py — "OOOR wants to fetch ..." / "OOOR wants to run this command" gate
# ─────────────────────────────────────────────────────────────────────────────
# WHAT THIS DOES
#   Sits in front of tool execution (called from gpt2_tools.execute_tool) and decides,
#   in CODE (never by asking the model nicely), whether the user must say yes first:
#
#     FETCH    any tool that opens a link / repo that is NOT
#                - in the user's "Supported webpages" list, or
#                - a URL the user typed/pasted in their own message, or
#                - one of the app's own storage hosts
#              -> asks "OOOR wants to fetch <url>"  (Allow / Deny / Allow + add to list)
#
#     COMMAND  workspace_run commands that are CRITICAL only (delete recursively, install,
#              network, sudo, secrets, leaving the workspace ...). Ordinary commands
#              (ls, cat, grep, pytest, python script.py) just run — asking every time
#              trains people to tap Allow without reading.
#
#   Everything fails CLOSED: deny, timeout and cancel all mean "not approved".
#
# HOW THE QUESTION REACHES THE USER
#   This module knows nothing about Firestore or the app. ai_jobs.py registers an
#   "approver" callable for the running job (set_approver). The gate calls it with a
#   spec dict and gets back "allow" | "allow_add" | "deny" | "timeout" | "cancelled".
#   With NO approver (the old /ai-query-stream path has no UI to answer) the gate does
#   nothing, so that path behaves exactly as before. APPROVALS_STRICT=1 flips that to
#   fail-closed.
#
# SUPPORTED LINKS
#   users/{uid}/ai_config/settings.supported_links  — a list OR a single string
#   (separated by spaces / commas / new lines). Entries match by domain (subdomains
#   included) or by domain + path prefix:  "github.com/googlefonts/" covers everything
#   under that path, "example.com" covers the whole site.
# ─────────────────────────────────────────────────────────────────────────────

import contextvars
import os
import re
import time
from typing import Any, Callable, Optional
from urllib.parse import urlparse

FETCH_TIMEOUT_S = int(os.environ.get("APPROVAL_FETCH_TIMEOUT_S", "120"))
COMMAND_TIMEOUT_S = int(os.environ.get("APPROVAL_COMMAND_TIMEOUT_S", "90"))   # riskier -> expires sooner
MAX_LINKS = 200
_LINKS_CACHE_S = 15

ALLOW = "allow"
ALLOW_ADD = "allow_add"          # allow + the app adds the link to the supported list
DENY = "deny"
TIMEOUT = "timeout"
CANCELLED = "cancelled"
_APPROVED = {ALLOW, ALLOW_ADD}

_approver: contextvars.ContextVar = contextvars.ContextVar("oor_approver", default=None)


def strict() -> bool:
    return os.environ.get("APPROVALS_STRICT", "0").strip().lower() in ("1", "true", "yes", "on")


def set_approver(fn: Optional[Callable[[dict], str]]):
    """Register the function that asks the user. Returns a token for reset_approver()."""
    return _approver.set(fn)


def reset_approver(token) -> None:
    try:
        _approver.reset(token)
    except Exception:
        _approver.set(None)


# ── supported links ──────────────────────────────────────────────────────────
_SPLIT_RE = re.compile(r"[\s,;]+")
_DOMAINISH_RE = re.compile(r"^[a-z0-9][a-z0-9.-]*\.[a-z]{2,}(?::\d+)?(?:[/?#].*)?$", re.I)


def normalize_links(value: Any) -> list:
    """Accepts a list OR one string (space/comma/newline separated) and returns clean, de-duplicated
    https:// strings. Junk entries are dropped. Never raises."""
    if value is None:
        return []
    if isinstance(value, str):
        raw = _SPLIT_RE.split(value.strip())
    elif isinstance(value, (list, tuple, set)):
        raw = []
        for v in value:
            raw.extend(_SPLIT_RE.split(str(v).strip()) if isinstance(v, str) else [])
    else:
        return []
    out, seen = [], set()
    for item in raw:
        item = item.strip().strip("'\"<>()[]")
        if not item:
            continue
        if not re.match(r"^[a-z][a-z0-9+.-]*://", item, re.I):
            if not _DOMAINISH_RE.match(item):
                continue
            item = "https://" + item
        p = urlparse(item)
        if p.scheme.lower() not in ("http", "https") or not p.hostname:
            continue
        key = _link_key(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= MAX_LINKS:
            break
    return out


def _link_key(url: str) -> str:
    p = urlparse(url)
    host = (p.hostname or "").lower().rstrip(".")
    return host + (p.path or "/").rstrip("/").lower()


_links_cache: dict = {}


def supported_links_for(userid: Optional[str]) -> list:
    """users/{uid}/ai_config/settings.supported_links (array or string). Cached a few seconds. Never raises."""
    if not userid:
        return []
    hit = _links_cache.get(userid)
    if hit and time.time() - hit[0] < _LINKS_CACHE_S:
        return hit[1]
    links: list = []
    try:
        from firebase_admin import firestore  # lazy: tests run without Firebase
        snap = (firestore.client().collection("users").document(userid)
                .collection("ai_config").document("settings").get())
        if snap.exists:
            links = normalize_links((snap.to_dict() or {}).get("supported_links"))
    except Exception as e:
        print(f"[APPROVALS] couldn't read supported links for {userid!r}: {type(e).__name__}: {e}")
    _links_cache[userid] = (time.time(), links)
    return links


def url_allowed(url: str, links: list) -> bool:
    """True when `url` is covered by a supported-links entry (same domain or subdomain, and the entry's
    path, if it has one, is a prefix of the url's path on a segment boundary)."""
    try:
        u = urlparse(url)
        if u.scheme.lower() not in ("http", "https") or not u.hostname:
            return False
        host = u.hostname.lower().rstrip(".")
        path = (u.path or "/").lower()
        for link in links:
            lp = urlparse(link)
            lhost = (lp.hostname or "").lower().rstrip(".")
            if not lhost or not (host == lhost or host.endswith("." + lhost)):
                continue
            lpath = (lp.path or "/").lower().rstrip("/")
            if not lpath:
                return True
            if path == lpath or path.startswith(lpath + "/"):
                return True
        return False
    except Exception:
        return False


# ── which tools fetch, and what they fetch ───────────────────────────────────
_GH_RE = re.compile(r"^(?:https?://)?(?:www\.)?github\.com/([^/\s]+)/([^/\s#?]+)", re.I)
_SLUG_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def _github_url(repo: str, path: str = "", ref: str = "") -> Optional[str]:
    repo = (repo or "").strip()
    m = _GH_RE.match(repo)
    if m:
        repo = f"{m.group(1)}/{m.group(2)}"
    repo = re.sub(r"\.git$", "", repo)
    if not _SLUG_RE.match(repo):
        return None
    url = f"https://github.com/{repo}"
    path = (path or "").strip().strip("/")
    if path:
        url += f"/tree/{(ref or 'HEAD').strip()}/{path}"
    return url


def fetch_target(tool: str, args: dict) -> Optional[str]:
    """The link a tool call would open, or None if this tool doesn't open one."""
    a = args or {}
    if tool in ("fetch_webpage", "fetch_document", "fetch_page_preview_image"):
        return str(a.get("url") or "").strip() or None
    if tool == "remove_background":
        return str(a.get("image_url") or "").strip() or None
    if tool in ("fetch_github_file", "workspace_github_clone", "workspace_github_folder"):
        path = "" if tool == "workspace_github_clone" else (a.get("path") or "")
        return _github_url(a.get("repo") or "", path, a.get("ref") or "")
    return None


def _own_hosts() -> set:
    hosts = {"firebasestorage.googleapis.com"}
    for var in ("SUPABASE_URL", "TRUSTED_FETCH_HOSTS"):
        for v in re.split(r"[\s,]+", os.environ.get(var, "")):
            h = urlparse(v if "://" in v else "https://" + v).hostname if v else None
            if h:
                hosts.add(h.lower())
    return hosts


def _user_words(prompt: str) -> str:
    """The user's own typing — NOT text of attached files (those can carry hidden links)."""
    try:
        from gpt2_workspace import _user_words_only  # lazy
        return _user_words_only(prompt or "")
    except Exception:
        return prompt or ""


def _norm_url(u: str) -> str:
    return (u or "").strip().rstrip("/").lower()


def _typed_by_user(url: str, ctx: dict) -> bool:
    words = _user_words(ctx.get("prompt") or "").lower()
    n = _norm_url(url)
    if n and n in words:
        return True
    attached = list(ctx.get("image_urls") or []) + list(ctx.get("file_urls") or [])
    return any(_norm_url(x) == n for x in attached)


# ── critical commands ────────────────────────────────────────────────────────
# A "verb" only counts in command position (start, or after ; & | ( ` $( — optionally behind VAR=x,
# xargs, time, timeout N, nice, env, command), so `python service.py` or `pytest -k eval` don't trip it.
_POS = (r"(?:^|[;&|(`]|\$\()\s*(?:\w+=\S*\s+)*"
        r"(?:(?:xargs|time|timeout\s+\S+|nice|env|command)\s+(?:-\S+\s+)*)*")
_CRITICAL = [
    (_POS + r"rm\b[^|;&]*\s(?:-[a-zA-Z]*[rRf][a-zA-Z]*|--recursive|--force)\b", "deletes files recursively or by force"),
    (_POS + r"rm\b[^|;&]*[*?]", "deletes files with a wildcard"),
    (_POS + r"find\b[^|;&]*\s-(?:delete|exec)\b", "deletes or runs something on many files"),
    (_POS + r"(?:shred|mkfs\w*|dd|truncate|fdisk|mount|umount)\b", "low-level disk or file operation"),
    (_POS + r"(?:sudo|su|doas)\b", "runs with elevated rights"),
    (_POS + r"(?:chmod|chown|chgrp)\b", "changes permissions"),
    (_POS + r"(?:kill|killall|pkill|shutdown|reboot|halt|systemctl|service|crontab|nohup|disown)\b",
     "controls processes or the system"),
    (_POS + r"(?:curl|wget|nc|ncat|netcat|ssh|scp|sftp|rsync|ftp|telnet|socat)\b", "uses the network"),
    (_POS + r"git\s+(?:push|clone|remote|fetch|pull|submodule)\b", "talks to a git remote"),
    (_POS + r"(?:pip3?|pipx|npm|npx|yarn|pnpm|apt(?:-get)?|apk|brew|gem|cargo|conda|poetry)\s+"
     r"(?:install|add|i|get|update|upgrade|uninstall|remove)\b", "installs or removes software"),
    (_POS + r"go\s+(?:get|install)\b", "installs software"),
    (_POS + r"python3?\s+-m\s+pip\b", "installs software"),
    (r"\|\s*(?:sh|bash|zsh|python3?|perl|ruby|node)\b", "pipes text into an interpreter"),
    (_POS + r"(?:eval|exec)\b", "runs dynamically built code"),
    (r"\bbase64\s+(?:-d|--decode)\b", "decodes hidden content"),
    (_POS + r"(?:printenv|env)\s*(?:$|[|;&>])", "prints environment variables"),
    (r"\$\{?[A-Za-z_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL)[A-Za-z_]*", "reads a secret"),
    (r"\.env\b|\.ssh\b|id_rsa|\bcredentials?\b|\.aws\b|\.npmrc|\.netrc", "touches credentials"),
    (r"(?:^|[\s'\"=<>])~(?:/|\s|$)", "touches your home folder"),
    (r"(?:^|[\s'\"=/])\.\.(?:/|\s|$|['\"])", "leaves the workspace folder"),
    (r"(?:^|[\s'\"=<>])/(?!dev/null\b)[A-Za-z]", "uses a path outside the workspace"),
    (r">\s*/(?!dev/null\b)", "writes outside the workspace"),
]
_CRITICAL_RE = [(re.compile(p, re.I), why) for p, why in _CRITICAL]


def critical_reasons(command: str, packages: Any = None) -> list:
    """Why a workspace_run call needs the user's OK. Empty list = it can just run."""
    reasons = []
    if packages:
        reasons.append("installs packages")
    cmd = command or ""
    for rx, why in _CRITICAL_RE:
        if rx.search(cmd) and why not in reasons:
            reasons.append(why)
    return reasons


# ── the gate ─────────────────────────────────────────────────────────────────
def needs_approval(tool: str, args: dict, ctx: dict) -> Optional[dict]:
    """None = go ahead. Otherwise the dialog spec the app shows."""
    args = args or {}

    if tool == "workspace_run":
        reasons = critical_reasons(str(args.get("command") or ""), args.get("packages"))
        if not reasons:
            return None
        return {"kind": "command", "tool": tool, "title": "OOOR wants to run this command",
                "target": str(args.get("command") or "")[:600], "why": "; ".join(reasons),
                "risk": "high", "timeoutS": COMMAND_TIMEOUT_S}

    url = fetch_target(tool, args)
    if url is None:
        return None       # not a link tool, or no usable link in the args (the tool reports that itself)
    if url_allowed(url, supported_links_for(ctx.get("userid"))):
        return None
    if _typed_by_user(url, ctx):
        return None
    host = (urlparse(url).hostname or "").lower()
    if host in _own_hosts():
        return None
    return {"kind": "fetch", "tool": tool, "title": "OOOR wants to fetch", "target": url[:600],
            "why": "This link isn't in your supported webpages.", "risk": "normal",
            "allowAdd": url.lower().startswith("http"), "timeoutS": FETCH_TIMEOUT_S}


def _deny_message(spec: dict, decision: str) -> str:
    what = "run that command" if spec["kind"] == "command" else f"fetch {spec['target']}"
    if decision == TIMEOUT:
        return (f"NOT DONE: the user did not respond in time, so OOOR did not {what}. Nothing was approved. "
                "Do not call this tool again this turn. Tell the user plainly that you didn't do it because "
                "you didn't get an answer, and offer to try again.")
    if decision == CANCELLED:
        return "NOT DONE: the user stopped this request. Do not call this tool again this turn."
    return (f"NOT DONE: the user declined, so OOOR did not {what}. Do not retry it or look for a way around it. "
            "Tell the user plainly you didn't do it, and finish the rest of their request without it if you can.")


def check(tool: str, args: dict, ctx: dict) -> Optional[str]:
    """Called by execute_tool before a tool runs. Returns None to proceed, or the message to hand
    the model INSTEAD of running the tool."""
    try:
        spec = needs_approval(tool, args, ctx)
    except Exception as e:      # a bug in the policy must never run an unchecked tool OR crash the turn
        print(f"[APPROVALS] policy error for {tool}: {type(e).__name__}: {e}")
        spec = None
    if spec is None:
        return None

    approver = _approver.get()
    if approver is None:
        return ("NOT DONE: this needs the user's approval but there is no way to ask in this session. "
                "Tell the user plainly.") if strict() else None

    key = f"{spec['kind']}:{spec['target']}"
    approved = ctx.setdefault("_approved", set())
    refused = ctx.setdefault("_refused", {})
    if key in approved:
        return None
    if key in refused:                      # the model asked again after a no: don't nag
        return _deny_message(spec, refused[key])

    try:
        decision = approver(dict(spec))
    except Exception as e:
        print(f"[APPROVALS] approver failed: {type(e).__name__}: {e}")
        decision = DENY
    if decision in _APPROVED:
        approved.add(key)
        return None
    decision = decision if decision in (TIMEOUT, CANCELLED) else DENY
    refused[key] = decision
    return _deny_message(spec, decision)
