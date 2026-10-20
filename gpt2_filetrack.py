#!/usr/bin/env python3
# gpt2_filetrack.py - remembers which files the AI MADE and which ones the user actually RECEIVED.
# ----------------------------------------------------------------------------
# Fixes three recurring problems, in code (not just prompts):
#   1. AI builds a file, never calls output_files, and the user gets nothing.
#        -> gpt2_native.run_native_turn asks `unsent_from(...)` when the model stops talking,
#           nudges it twice, then sends the file itself (build_delivery_calls).
#   2. User says "drop the file" and the AI rebuilds it from scratch.
#        -> `send_request_paths` spots a short "send/drop the file" message; the pending file is
#           delivered at once, with no model rewriting. `files_note` also tells the model what exists.
#   3. AI types a download link it made up (404 "Object not found").
#        -> `scrub_invented_links` removes storage links that no tool returned.
#
#   python3 gpt2_filetrack.py --selftest
# ----------------------------------------------------------------------------

import json
import os
import re
import threading
import time
from collections import OrderedDict
from typing import Iterable, Optional

DOC_EXTS = {
    ".pdf", ".html", ".htm", ".md", ".txt", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".csv", ".zip",
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg", ".wav", ".mp3", ".mp4", ".json",
}
_SKIP_PARTS = ("uploads/", ".git/", "node_modules/", "__pycache__/")
MAX_TRACKED = 60

_lock = threading.Lock()
_files: dict = {}          # userid -> OrderedDict[path] = {"made": ts, "sent": ts|None, "url": str|None}


# ── helpers ──────────────────────────────────────────────────────────────────
def norm_path(p) -> str:
    p = str(p or "").replace("\\", "/").strip()
    while p.startswith("./"):
        p = p[2:]
    return p.lstrip("/")


def is_deliverable(path: str) -> bool:
    """A file the user would plausibly want to receive (not an upload, repo internals or a cache)."""
    p = norm_path(path)
    if not p or p.endswith("/") or any(p.startswith(s) or f"/{s}" in f"/{p}" for s in _SKIP_PARTS):
        return False
    return p.startswith("outputs/") or os.path.splitext(p)[1].lower() in DOC_EXTS


def _user(userid) -> str:
    return userid or "anonymous"


def note_made(userid, path: str) -> bool:
    p = norm_path(path)
    if not is_deliverable(p):
        return False
    with _lock:
        d = _files.setdefault(_user(userid), OrderedDict())
        d.pop(p, None)                                   # a re-made/edited file is "not sent" again
        d[p] = {"made": time.time(), "sent": None, "url": None}
        while len(d) > MAX_TRACKED:
            d.popitem(last=False)
    return True


def note_sent(userid, path: str, url: Optional[str] = None) -> None:
    p = norm_path(path)
    with _lock:
        d = _files.setdefault(_user(userid), OrderedDict())
        hits = [k for k in d if k == p or k.endswith("/" + p) or (("/" not in p) and os.path.basename(k) == p)]
        if not hits:
            d[p] = {"made": time.time(), "sent": time.time(), "url": url}
            return
        for k in hits:
            d[k]["sent"] = time.time()
            if url:
                d[k]["url"] = url


def is_pending(userid, path: str) -> bool:
    with _lock:
        e = _files.get(_user(userid), {}).get(norm_path(path))
    return bool(e) and e["sent"] is None


def _exists(userid, path: str) -> bool:
    try:
        from gpt2_workspace import _safe_path
        return os.path.isfile(_safe_path(userid, path))
    except Exception:
        return False


def pending(userid, max_age_s: float = 24 * 3600) -> list:
    """Files made but never sent, oldest first, still on disk."""
    now = time.time()
    with _lock:
        items = [(p, e) for p, e in _files.get(_user(userid), {}).items()
                 if e["sent"] is None and now - e["made"] <= max_age_s]
    return [p for p, _e in items if _exists(userid, p)]


def unsent_from(userid, paths: Iterable[str]) -> list:
    """Of `paths` (made this turn), the ones the user still has not received."""
    seen, out = set(), []
    for p in paths:
        p = norm_path(p)
        if p and p not in seen and is_pending(userid, p) and _exists(userid, p):
            seen.add(p)
            out.append(p)
    return out


def _recent_outputs(userid, max_age_s: float, limit: int) -> list:
    """Fallback after a redeploy (memory is empty): newest files in outputs/."""
    try:
        from gpt2_workspace import _safe_path
        root = _safe_path(userid, "outputs")
        found = []
        for dp, dn, fn in os.walk(root):
            for name in fn:
                full = os.path.join(dp, name)
                mt = os.path.getmtime(full)
                if time.time() - mt <= max_age_s:
                    found.append((mt, os.path.relpath(full, _safe_path(userid, "")).replace(os.sep, "/")))
        return [p for _m, p in sorted(found, reverse=True)[:limit]]
    except Exception:
        return []


# ── effect of a tool result on the registry ─────────────────────────────────
def _loads(s):
    try:
        return json.loads(s)
    except Exception:
        return None


def note_result(userid, tool: str, args: dict, result_str: str, ok: bool) -> list:
    """Update the registry from one tool result. Returns the paths this call MADE (for turn tracking)."""
    if not ok:
        return []
    data = _loads(result_str)
    if not isinstance(data, (dict, list)):
        return []
    made = []
    if isinstance(data, dict) and data.get("success") is False and tool != "output_files":
        return []
    if tool in ("workspace_write", "workspace_append", "save_pasted_text") and isinstance(data, dict):
        path = data.get("path")
        if path and note_made(userid, path):
            made.append(norm_path(path))
            delivered = data.get("delivered") or []
            if delivered:
                note_sent(userid, path, (delivered[0] or {}).get("url"))
    elif tool == "run_code" and isinstance(data, dict):
        for f in data.get("saved_to_workspace") or []:
            path = f.get("path") if isinstance(f, dict) else f
            if path and note_made(userid, path):
                made.append(norm_path(path))
    elif tool == "workspace_run" and isinstance(data, dict):
        for path in data.get("changed_files") or []:
            if isinstance(path, str) and os.path.splitext(path)[1].lower() in DOC_EXTS and note_made(userid, path):
                made.append(norm_path(path))
    elif tool == "output_files" and isinstance(data, dict):
        failed = {norm_path(f.get("file")) for f in (data.get("failed") or []) if isinstance(f, dict)}
        entries = (args or {}).get("files")
        entries = [entries] if isinstance(entries, (str, dict)) else (entries or [])
        urls = [d.get("url") for d in (data.get("files") or []) if isinstance(d, dict)]
        for i, e in enumerate(entries):
            path = e if isinstance(e, str) else (e or {}).get("path")
            if not path or str(path).startswith(("http://", "https://")) or norm_path(path) in failed:
                continue
            note_sent(userid, path, urls[i] if i < len(urls) else None)
    return made


# ── "drop the file" -> deliver, don't rebuild ───────────────────────────────
_STRONG_NOUN = re.compile(r"\b(?:files?|pdfs?|docs?|documents?|websites?|sites?|webpages?|pages?|html|zip|txt|"
                          r"downloads?|attachments?|links?|outputs?)\b")
_VERB = re.compile(r"\b(?:drop|send|give|share|attach|upload|resend|re-send|deliver|post|bring|"
                   r"where(?:'s| is| are| dey)?|forgot|forget|didn'?t (?:get|receive|see|send)|"
                   r"did not (?:get|receive|see|send)|not (?:see|get|receive)|no file|never (?:get|got|send|sent))\b")
_SHORT = re.compile(r"^(?:pls |please |abeg |now |ok |okay |hey )*(?:drop|send|give|share|resend|upload) (?:it|am|that|them|those)"
                    r"(?: again| now| here| abeg| pls| please| man| dude)*[.!? ]*$")
_NOT_A_SEND_REQUEST = re.compile(r"\b(?:how|python|code|script|function|api|using|email|whatsapp|telegram|friend|boss|"
                                 r"teacher|mum|mom|dad|to my|to him|to her|to them|why|should i|can i|example)\b")


def is_send_request(prompt: str) -> bool:
    t = re.sub(r"\s+", " ", (prompt or "").strip().lower())
    if not t or len(t) > 90 or len(t.split()) > 12:
        return False
    if _SHORT.match(t):
        return True
    if _NOT_A_SEND_REQUEST.search(t):
        return False
    return bool(_VERB.search(t) and _STRONG_NOUN.search(t))


_DONT_SEND_RE = re.compile(r"\b(?:(?:do ?n[o']?t|do not|no need to|never|without) (?:send|drop|deliver|share|give)|"
                           r"(?:only|just) save|save (?:it )?(?:only|first)|hold (?:it|on)|not yet)\b", re.I)


def says_dont_send(prompt: str) -> bool:
    """The user asked to keep the file for now - the server must not send it on its own."""
    return bool(_DONT_SEND_RE.search(prompt or ""))


def send_request_paths(userid, prompt: str, limit: int = 5) -> list:
    """If `prompt` is a short "drop/send the file" message, the files to hand over (never rebuild)."""
    if not is_send_request(prompt):
        return []
    paths = pending(userid)
    if not paths and not _files.get(_user(userid)):
        paths = _recent_outputs(userid, 6 * 3600, 1 if not re.search(r"files|them|those", prompt.lower()) else limit)
    return paths[-limit:]


def build_delivery_calls(userid, paths: list) -> list:
    """Synthetic tool calls: design_check for each PDF/page not yet looked at, then ONE output_files."""
    calls, n = [], int(time.time() * 1000) % 100000
    try:
        from gpt2_design_check import needs_design_check
        from gpt2_workspace import _safe_path
    except Exception:
        needs_design_check = None
    for p in paths:
        try:
            if needs_design_check and needs_design_check(userid, _safe_path(userid, p)):
                calls.append({"id": f"auto_{n}_{len(calls)}_check", "name": "design_check",
                              "arguments": json.dumps({"path": p})})
        except Exception:
            pass
    if paths:
        calls.append({"id": f"auto_{n}_{len(calls)}_send", "name": "output_files",
                      "arguments": json.dumps({"files": list(paths)})})
    return calls


# ── what the model is told about files that exist ───────────────────────────
def _ago(ts: float) -> str:
    s = max(0, int(time.time() - ts))
    return f"{s}s ago" if s < 90 else f"{s // 60} min ago" if s < 5400 else f"{s // 3600} h ago"


def files_note(userid) -> str:
    with _lock:
        items = list(_files.get(_user(userid), {}).items())
    unsent = [(p, e) for p, e in items if e["sent"] is None and _exists(userid, p)][-6:]
    sent = [(p, e) for p, e in items if e["sent"] is not None and _exists(userid, p)][-4:]
    if not unsent and not sent:
        recent = _recent_outputs(userid, 24 * 3600, 5)
        if not recent:
            return ""
        lines = ["Files that exist in outputs/ (unknown whether they were sent): " + ", ".join(recent)]
    else:
        lines = []
        if unsent:
            lines.append("Made but NOT yet received by the user: " + ", ".join(f"{p} (made {_ago(e['made'])})" for p, e in unsent))
        if sent:
            lines.append("Already sent to the user: " + ", ".join(p for p, _e in sent))
    return (
        "\n\nFILES IN THIS CHAT (internal, never quote this block):\n" + "\n".join("- " + l for l in lines) + "\n"
        "- If the user says drop / send / give / where is the file, or that they did not get it: call output_files with "
        "the exact path above. NEVER recreate, rewrite or rebuild a file that already exists. Only write a file again "
        "when the user asks for a CHANGE to it.\n"
        "- Whenever you make a file the user should have, send it yourself in the same turn with output_files. "
        "Never write a download link or file URL yourself; the file card is the only link."
    )


# ── links the AI made up ────────────────────────────────────────────────────
_URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")
_STORAGE_MARK = "/storage/v1/object/"
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\((https?://[^\s)]+)\)")


def _clean_url(u: str) -> str:
    return u.rstrip(".,;:!?*_`").split("?")[0].split("#")[0]


def collect_urls(text: str) -> set:
    return {_clean_url(u) for u in _URL_RE.findall(text or "")}


def scrub_invented_links(text: str, known_urls: set) -> str:
    """Remove links to OUR file storage that no tool returned (the AI invents them; they 404).
    Markdown links keep their label; bare URLs are dropped. Everything else is left alone."""
    if not text or _STORAGE_MARK not in text:
        return text

    def md(m):
        url = _clean_url(m.group(2))
        return m.group(0) if (_STORAGE_MARK not in url or url in known_urls) else m.group(1)

    out = _MD_LINK_RE.sub(md, text)

    def bare(m):
        url = _clean_url(m.group(0))
        return m.group(0) if (_STORAGE_MARK not in url or url in known_urls) else ""

    out = _URL_RE.sub(bare, out)
    return re.sub(r"[ \t]+\n", "\n", re.sub(r"[ \t]{2,}", " ", out)).strip()


def reset(userid=None) -> None:
    """Forget everything (used by tests)."""
    with _lock:
        if userid is None:
            _files.clear()
        else:
            _files.pop(_user(userid), None)


# ── self-test ────────────────────────────────────────────────────────────────
def _selftest() -> None:
    def line(name, ok, detail=""):
        print(f"[{'OK ' if ok else 'NO '}] {name}" + (f" - {detail}" if detail else ""))

    yes = ["drop the file", "send the file now", "pls drop am", "where is the file", "i didn't get the file",
           "give me the pdf", "send the website", "drop it", "you forgot to drop the file man", "where is my website"]
    no = ["send an email to my boss", "how do i send a file in python", "send a file to my friend", "hello",
          "make me a pdf about cats", "what is a file"]
    line("send requests recognised", all(is_send_request(t) for t in yes), str([t for t in yes if not is_send_request(t)]))
    line("normal chat not mistaken", not any(is_send_request(t) for t in no), str([t for t in no if is_send_request(t)]))
    bad = "see [your file](https://x.supabase.co/storage/v1/object/public/b/u/a.pdf) here"
    line("invented link removed", "supabase" not in scrub_invented_links(bad, set()))
    print("done")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
