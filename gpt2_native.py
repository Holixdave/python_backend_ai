"""
gpt2_native.py — NATIVE tool calling + INTERLEAVED streaming.

Replaces the text convention (<<TOOL_CALL>>...) with the providers' real
function-calling (`tools=[...]`, streamed `tool_calls`), and streams the turn as
ORDERED events so the app can draw it the way it happened:

    text   "Hold on, let me look that up."
    tool   search_web(...)            <- chip
    block  sources / images / file    <- shown right where the tool produced them
    text   "Here's what I found..."
    tool   build file ...
    block  file card
    text   "Done — it's above."
    final  everything again, in order, for saving

Event contract (every yielded dict has a "type"):
  {"type":"text",        "delta": str}
  {"type":"tool_call",   "id", "tool", "args", "text", "icon", "call"}      (a chip)
  {"type":"tool_result", "id", "tool", "ok": bool, "summary": str}
  {"type":"block",       "kind": "images"|"file"|"sources"|"code", ...payload}
  {"type":"ask_user",    ...payload}                  (turn ends, no final)
  {"type":"final",       "answer", "blocks":[...], "segments", "sources", "images",
                         "files", "file", "codes", "suggestions", "provider"}

Saved/final `blocks` (ordered):
  {"type":"text",   "text": str, "segments": [...]}
  {"type":"tool",   "id","tool","text","icon","ok"}
  {"type":"images", "images":[...]}
  {"type":"file",   "file":{"filename","url"}}
  {"type":"sources","sources":[{"title","url"}]}
  {"type":"code",   "code":{...code_result...}}

Everything model-/network-specific is injected, so it is unit-tested offline
(tests/test_native.py) with a fake model.
"""
import inspect
import json
import os
import re
import time
import typing
import uuid
from typing import Callable, Optional

import requests

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}


def native_enabled() -> bool:
    """ON by default. NATIVE_TOOLS=0 falls back to the old text-convention loop."""
    return os.environ.get("NATIVE_TOOLS", "1").strip().lower() in ("1", "true", "yes", "on")


# ─────────────────────────────────────────────────────────────────────────────
# 1. TOOL SCHEMAS — built from the real function signatures, so they can't drift
# ─────────────────────────────────────────────────────────────────────────────
# list params: what the items are (default "string").
_OPTION = {"type": "object", "properties": {"label": {"type": "string"}, "description": {"type": "string"}},
           "required": ["label"]}
_QUESTION = {"type": "object", "properties": {
    "question": {"type": "string"},
    "options": {"type": "array", "items": _OPTION},
    "allow_multiple": {"type": "boolean"},
    "allow_other": {"type": "boolean"}}, "required": ["question", "options"]}
_ITEM_TYPES = {
    # Objects need explicit properties: some providers (Gemini) reject a bare {"type": "object"}.
    ("create_image", "operations"): {"type": "object", "properties": {
        "op": {"type": "string", "description": "rectangle | ellipse | line | polygon | text"},
        "xy": {"type": "array", "items": {"type": "number"}},
        "fill": {"type": "string"}, "text": {"type": "string"}, "width": {"type": "integer"}},
        "required": ["op", "xy"]},
    ("ask_user", "options"): _OPTION,
    ("ask_user", "questions"): _QUESTION,
}


def _unwrap_optional(ann):
    origin = typing.get_origin(ann)
    if origin is typing.Union:
        args = [a for a in typing.get_args(ann) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return ann


def _param_schema(tool: str, pname: str, ann) -> dict:
    ann = _unwrap_optional(ann)
    origin = typing.get_origin(ann) or ann
    if ann is bool:
        return {"type": "boolean"}
    if ann is int:
        return {"type": "integer"}
    if ann is float:
        return {"type": "number"}
    if origin in (list, typing.List, tuple, set):
        item = _ITEM_TYPES.get((tool, pname), "string")
        return {"type": "array", "items": item if isinstance(item, dict) else {"type": item}}
    if origin in (dict, typing.Dict):
        return {"type": "object"}
    return {"type": "string"}


def _doc_args(fn) -> dict:
    """{param: description} parsed from the 'Args:' section of the docstring."""
    doc = inspect.getdoc(fn) or ""
    m = re.search(r"\n\s*Args:\s*\n(.*?)(?:\n\s*\n|\n\s*[A-Z][A-Za-z ]+:\s*\n|\Z)", "\n" + doc, re.S)
    out = {}
    if not m:
        return out
    for line in m.group(1).split("\n"):
        mm = re.match(r"\s*(\w+)\s*(?:\([^)]*\))?:\s*(.+)", line)
        if mm:
            out[mm.group(1)] = mm.group(2).strip()
    return out


def build_tool_schemas(registry: dict, descriptions: dict, injected: set, only: Optional[set] = None,
                       expose: frozenset = frozenset()) -> list:
    """OpenAI-style `tools` list. Session-owned params (userid, history, ...) are NOT
    exposed — the backend fills them in, the model can't set or fake them. `expose` is a set
    of (tool, param) pairs that ARE written by the model even though the name is injected
    elsewhere (e.g. generate_image's own `prompt`)."""
    tools = []
    for name, fn in registry.items():
        if only is not None and name not in only:
            continue
        try:
            sig = inspect.signature(fn)
            hints = typing.get_type_hints(fn)
        except Exception:
            continue
        argdocs = _doc_args(fn)
        props, required = {}, []
        for pname, p in sig.parameters.items():
            if (pname in injected and (name, pname) not in expose) or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            schema = _param_schema(name, pname, hints.get(pname, str))
            if pname in argdocs:
                schema["description"] = argdocs[pname][:300]
            props[pname] = schema
            if p.default is inspect.Parameter.empty:
                required.append(pname)
        desc = (descriptions.get(name) or (inspect.getdoc(fn) or name).split("\n")[0])[:1000]
        fn_def = {"name": name, "description": desc,
                  "parameters": {"type": "object", "properties": props}}
        if required:
            fn_def["parameters"]["required"] = required
        tools.append({"type": "function", "function": fn_def})
    return tools


# ─────────────────────────────────────────────────────────────────────────────
# 2. LIVE TEXT GATE — streams answer text, but never leaks reasoning tags or <<tags>>
# ─────────────────────────────────────────────────────────────────────────────
from leak_guard import (  # noqa: E402  (kept next to its only users)
    scrub_internal, repair_mojibake, strip_reasoning,
    THINK_OPEN_RE, THINK_CLOSE_RE, THINK_BLOCK_RE, THINK_TAG_LITERALS, THINK_TAG_MAX_LEN,
)


def _partial_start(buf: str) -> int:
    """Index where an unfinished '<<' or reasoning tag ('<tho', '</thin' ...) begins at the very
    end of buf — that tail is held back until the next chunk shows what it is. -1 if none."""
    lt = buf.rfind("<")
    if lt == -1:
        return -1
    if lt > 0 and buf[lt - 1] == "<":          # a trailing '<<' (suggestion marker) is held whole
        lt -= 1
    tail = buf[lt:].lower()
    if tail == "<<" or any(t.startswith(tail) for t in THINK_TAG_LITERALS):
        return lt
    return -1


class TextGate:
    """feed(delta) -> text that is safe to show NOW. Everything from the first '<<'
    (suggestion / inline-visual tags) is withheld until the turn ends; reasoning blocks
    (<think>, <thought>, <thinking>, <reasoning> ...) are dropped. `raw` keeps the full
    unfiltered text for final parsing."""

    def __init__(self):
        self.raw = ""
        self._buf = ""
        self._in_think = False
        self._held = False

    def feed(self, delta: str) -> str:
        self.raw += delta
        if self._held:
            return ""
        self._buf += delta
        out = []
        while True:
            if self._in_think:
                m = THINK_CLOSE_RE.search(self._buf)
                if not m:                                   # still inside: keep only what could start a closer
                    self._buf = self._buf[-THINK_TAG_MAX_LEN:]
                    break
                self._buf = self._buf[m.end():]
                self._in_think = False
                continue
            o = THINK_OPEN_RE.search(self._buf)
            k = self._buf.find("<<")
            if o and (k == -1 or o.start() < k):
                out.append(self._buf[:o.start()])
                self._buf = self._buf[o.end():]
                self._in_think = True
                continue
            if k != -1:
                out.append(self._buf[:k])
                self._buf = ""
                self._held = True
                break
            ps = _partial_start(self._buf)
            if ps != -1:
                out.append(self._buf[:ps])
                self._buf = self._buf[ps:]
            else:
                out.append(self._buf)
                self._buf = ""
            break
        return "".join(out)

    def flush(self) -> str:
        if self._held or self._in_think:
            self._buf = ""
            return ""
        rest, self._buf = self._buf, ""
        return rest


_THINK_RE = THINK_BLOCK_RE          # old name, kept so nothing importing it breaks


def clean_round_text(raw: str) -> str:
    """Final text of one model turn: no reasoning blocks / leaked reasoning debris."""
    return repair_mojibake(strip_reasoning(raw))


# ─────────────────────────────────────────────────────────────────────────────
# 3. STREAMING CHAT — one provider after another; if one dies MID-ANSWER the next one
#    carries on from the exact words the user already saw
# ─────────────────────────────────────────────────────────────────────────────
CONTINUE_PROMPT = (
    "Continue your previous reply from exactly the point where it stopped. Do not repeat anything "
    "already written, do not greet, do not apologize and do not mention any interruption. If the "
    "last sentence was cut off, finish it mid-sentence. Keep the same language, tone and formatting."
)


class _Stitcher:
    """Glues a continuation onto words the user already saw. The first ~160 characters of the new
    model's text are held briefly so any words it repeats from the end of `seen` can be dropped —
    after that everything passes straight through."""
    HOLD = 160

    def __init__(self, seen: str):
        self._tail = seen[-200:]
        self._buf = ""
        self._open = False

    def feed(self, delta: str) -> str:
        if self._open:
            return delta
        self._buf += delta
        return self._release() if len(self._buf) >= self.HOLD else ""

    def flush(self) -> str:
        return "" if self._open else self._release()

    def _release(self) -> str:
        self._open = True
        buf, self._buf = self._buf, ""
        for k in range(min(len(self._tail), len(buf)), 7, -1):
            if self._tail.endswith(buf[:k]):
                return buf[k:]
        return buf


def _valid_calls(calls: list) -> list:
    """Tool calls whose arguments are complete JSON — a call cut off mid-arguments is useless."""
    good = []
    for c in calls:
        try:
            json.loads(c.get("arguments") or "{}")
            good.append(c)
        except Exception:
            pass
    return good


class _Health:
    """Remembers which models just failed, so a rate-limited or request-rejecting model is not
    hammered again on every round of every message (that was burning the whole chain per tool call)."""

    def __init__(self, clock: Callable = time.time):
        self._until: dict = {}
        self._clock = clock

    def bad(self, key, seconds: float) -> None:
        self._until[key] = self._clock() + seconds

    def ok(self, key) -> bool:
        return self._until.get(key, 0) <= self._clock()

    def clear(self) -> None:
        self._until.clear()


HEALTH = _Health()


def _request_kind(tools, messages) -> str:
    """Models reject different things: some can't take tools, some choke on tool-call history."""
    if not tools:
        return "plain"
    return "tools+history" if any(m.get("role") == "tool" for m in messages) else "tools"


def apply_model_pin(providers: list) -> list:
    """MODEL_PIN=name[,name...] (env) restricts the turn to just those models, in that order —
    e.g. MODEL_PIN=groq-gpt-oss-120b for a single-model setup. Unknown/disabled names are ignored."""
    pin = (os.environ.get("MODEL_PIN") or "").strip()
    if not pin:
        return providers
    names = [n.strip() for n in pin.split(",") if n.strip()]
    chosen = [p for n in names for p in providers if p.get("name") == n and p.get("enabled")]
    if chosen:
        return chosen
    print(f"[NATIVE] MODEL_PIN={pin!r} matches no enabled model — using the normal chain")
    return providers


def _flatten_tool_history(messages: list) -> list:
    """Same conversation with the tool-call structure turned into plain text. Some models reject
    another model's tool-call history (ids, signatures, strict formats); plain text always works."""
    out: list = []
    pending: list = []

    def flush():
        if not pending:
            return
        text = "Results from the tools you just used:\n\n" + "\n\n".join(pending)
        pending.clear()
        if out and out[-1].get("role") == "user" and isinstance(out[-1].get("content"), str):
            out[-1] = {"role": "user", "content": out[-1]["content"] + "\n\n" + text}
        else:
            out.append({"role": "user", "content": text})

    for m in messages:
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            flush()
            if m.get("content"):
                out.append({"role": "assistant", "content": m["content"]})
            continue
        if role == "tool":
            pending.append(f"[{m.get('name') or 'tool'}]\n{m.get('content') or ''}")
            continue
        flush()
        out.append(m)
    flush()
    return out


def stream_chat(providers: list, messages: list, tools: Optional[list], *, temperature: float,
                max_tokens: int, timeout: int = 120, max_retries: int = 2, retry_delay: float = 1.0,
                reasoning_effort: Optional[str] = None, post: Callable = None, sleep: Callable = time.sleep):
    """
    Generator over ("text", delta) ... then exactly one of
        ("done", {"content": str, "tool_calls": [...], "provider": str, "providers": [str, ...],
                  "interrupted": bool (only when every provider died mid-answer)})
        ("fail", reason)
    A provider that errors BEFORE producing words is skipped or retried. A provider that dies AFTER
    the user already saw words is replaced: the next provider gets the partial answer and is told to
    carry on from exactly there, so the reply just keeps flowing. `providers` lists every provider
    that wrote part of the answer, in order.
    """
    post = post or requests.post
    last_error = "No provider available."
    errors: list = []              # every model's failure, so the log shows the whole story
    shown = ""                     # every word the user has seen this round, across providers
    used: list = []                # providers that contributed words / tool calls, in order
    kind = _request_kind(tools, messages)
    usable = [p for p in providers if p.get("enabled")
              and HEALTH.ok(("rate", p["name"])) and HEALTH.ok((kind, p["name"]))]
    if not usable:                 # everything is cooling down — try them anyway rather than give up
        usable = [p for p in providers if p.get("enabled")]
    for provider in usable:
        name = provider["name"]
        continuing = bool(shown)
        msgs = messages
        if continuing:
            msgs = list(messages) + [{"role": "assistant", "content": shown},
                                     {"role": "user", "content": CONTINUE_PROMPT}]
        payload = {"model": provider["model"], "messages": msgs, "temperature": temperature,
                   "max_tokens": max_tokens, "stream": True}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if reasoning_effort is not None and provider.get("supports_reasoning_effort"):
            payload["reasoning_effort"] = reasoning_effort
            payload["reasoning_format"] = "raw"

        for attempt in range(1, max_retries + 1):
            stitch = _Stitcher(shown) if continuing else None
            wrote = False                                   # this attempt put words in front of the user
            calls: list = []
            try:
                resp = post(provider["url"], headers=provider["headers"], json=payload,
                            timeout=timeout, stream=True)
                if resp.status_code == 429:
                    last_error = f"{name}: rate limited (429)"
                    if attempt < max_retries and not continuing:
                        sleep(retry_delay * attempt)
                        continue
                    errors.append(last_error)
                    try:
                        wait = float((getattr(resp, "headers", None) or {}).get("Retry-After") or 60)
                    except (TypeError, ValueError):
                        wait = 60.0
                    HEALTH.bad(("rate", name), min(max(wait, 15.0), 120.0))
                    break
                if resp.status_code != 200:
                    last_error = f"{name}: HTTP {resp.status_code} — {str(resp.text)[:150]}"
                    errors.append(last_error)
                    if resp.status_code >= 500 or resp.status_code in (401, 403):
                        HEALTH.bad(("rate", name), 30 if resp.status_code >= 500 else 300)
                    else:   # 400/404/413/422: this model won't take THIS kind of request — skip it for a while
                        HEALTH.bad((kind, name), 60 if resp.status_code == 413 else 300)
                    break
                try:
                    resp.encoding = "utf-8"                 # SSE is always UTF-8; without this requests
                except Exception:                           # assumes Latin-1 and turns "•" into "â¢"
                    pass
                for raw in resp.iter_lines(decode_unicode=True):
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8", "replace")
                    if not raw or not raw.startswith("data:"):
                        continue
                    data = raw[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except Exception:
                        continue
                    if isinstance(obj, dict) and obj.get("error"):
                        raise RuntimeError(str(obj["error"])[:150])
                    for ch in (obj.get("choices") or []):
                        d = ch.get("delta") or {}
                        piece = d.get("content")
                        if isinstance(piece, str) and piece:
                            out = stitch.feed(piece) if stitch else piece
                            if out:
                                wrote = True
                                shown += out
                                yield ("text", out)
                        for tc in (d.get("tool_calls") or []):
                            fn = tc.get("function") or {}
                            name = fn.get("name")
                            args = fn.get("arguments")
                            if name:                       # a chunk with a name starts a new call
                                calls.append({"id": tc.get("id") or f"call_{uuid.uuid4().hex[:10]}",
                                              "name": name, "arguments": args if isinstance(args, str) else
                                              json.dumps(args or {})})
                            elif calls and args:
                                calls[-1]["arguments"] += args if isinstance(args, str) else json.dumps(args)
                if stitch:
                    tail = stitch.flush()
                    if tail:
                        wrote = True
                        shown += tail
                        yield ("text", tail)
                if wrote or calls:
                    used.append(provider["name"])
                    yield ("done", {"content": shown, "tool_calls": calls, "provider": provider["name"],
                                    "providers": used})
                    return
                last_error = f"{name}: empty response"
                errors.append(last_error)
                HEALTH.bad(("rate", name), 20)
                break
            except Exception as e:
                last_error = f"{name}: {e}"
                if wrote:
                    used.append(provider["name"])
                if shown:
                    # The user already saw words. A finished tool call can still be used; otherwise the
                    # NEXT provider carries on from here (no retry of this one — it just died).
                    good = _valid_calls(calls)
                    if good:
                        yield ("done", {"content": shown, "tool_calls": good, "provider": provider["name"],
                                        "providers": used})
                        return
                    errors.append(last_error)
                    HEALTH.bad(("rate", name), 30)
                    break
                if attempt < max_retries:
                    sleep(retry_delay * attempt)
                    continue
                errors.append(last_error)
                HEALTH.bad(("rate", name), 30)
                break
    if shown:                                               # every provider died mid-answer — keep what we have
        yield ("done", {"content": shown, "tool_calls": [], "provider": used[-1] if used else None,
                        "providers": used, "interrupted": True})
        return
    summary = " | ".join(errors)[-900:] or last_error
    print(f"[CHAIN] every model failed ({kind}): {summary}")
    yield ("fail", summary)


def _model_label(name: str) -> str:
    """'openrouter/nvidia/nemotron-3-super:free' -> 'nemotron-3-super' — short name for the app's models chip."""
    return (name or "").split("/")[-1].split(":")[0]


# ─────────────────────────────────────────────────────────────────────────────
# 3b. CHIP LABELS — what the user reads on a tool chip (never raw tool names)
# ─────────────────────────────────────────────────────────────────────────────
def _short(v, n=40) -> str:
    v = str(v or "").strip().replace("\n", " ")
    return v if len(v) <= n else v[: n - 1] + "…"


_LABELS = {
    "search_web": lambda a: f"Searching for {_short(a.get('query'))}" if a.get("query") else "Searching the web",
    "search_images": lambda a: f"Searching images for {_short(a.get('query'))}" if a.get("query") else "Searching images",
    "generate_image": lambda a: "Generating an image",
    "create_image": lambda a: "Drawing an image",
    "remove_background": lambda a: "Removing the background",
    "analyze_image": lambda a: "Looking at the image",
    "fetch_webpage": lambda a: f"Reading {_short(a.get('url'))}" if a.get("url") else "Reading a page",
    "fetch_document": lambda a: "Reading a document",
    "fetch_page_preview_image": lambda a: "Getting a page preview",
    "search_github": lambda a: "Searching GitHub",
    "fetch_github_file": lambda a: f"Reading {_short(a.get('path'))} on GitHub",
    "run_code": lambda a: f"Running {a.get('language') or 'python'} code",
    "workspace_list": lambda a: f"Looking in {_short(a.get('path') or 'your workspace')}",
    "workspace_read": lambda a: f"Reading {_short(a.get('path'))}",
    "workspace_write": lambda a: f"Creating {_short(a.get('path'))}",
    "publish_page": lambda a: f"Publishing {_short(a.get('title') or 'page')}",
    "workspace_edit": lambda a: f"Editing {_short(a.get('path'))}",
    "workspace_edit_by_intent": lambda a: f"Editing {_short(a.get('path'))}",
    "workspace_delete": lambda a: f"Deleting {_short(a.get('path'))}",
    "workspace_grep": lambda a: f"Searching your files for {_short(a.get('pattern'), 24)}",
    "workspace_copy": lambda a: f"Copying {_short(a.get('src'))}",
    "workspace_unzip": lambda a: f"Unzipping {_short(a.get('path'))}",
    "workspace_zip": lambda a: "Zipping files",
    "workspace_run": lambda a: "Running a command",
    "workspace_github_clone": lambda a: f"Cloning {_short(a.get('repo'))}",
    "output_files": lambda a: "Sending files",
    "schedule_reminder": lambda a: "Setting a reminder",
    "save_user_note": lambda a: "Saving a note",
    "get_user_notes": lambda a: "Checking your notes",
    "save_study_note": lambda a: "Saving a study note",
    "get_study_notes": lambda a: "Checking your study notes",
    "get_user_profile": lambda a: "Checking your profile",
    "result_more": lambda a: "Reading more",
}


_INPUT_KEYS = ("code", "query", "command", "url", "pattern", "path", "prompt", "title")


def tool_input_preview(tool: str, args: dict, cap: int = 6000) -> str:
    """The thing the tool was given (python code, search text, path…) for the tap-to-open card."""
    args = args or {}
    if tool == "workspace_write":
        return _short(args.get("path"), 200)
    if tool == "publish_page":
        return _short(args.get("title"), 200)
    for k in _INPUT_KEYS:
        v = args.get(k)
        if isinstance(v, str) and v.strip():
            return v if len(v) <= cap else v[:cap] + "\n…"
    return ""


def tool_output_preview(tool: str, ok: bool, result_str: str, cap: int = 1500) -> str:
    """Short, human-readable result line(s) for the tap-to-open card."""
    data = _loads(result_str)
    if tool == "run_code" and isinstance(data, dict):
        out = (data.get("stdout") or "").strip()
        err = (data.get("stderr") or "").strip()
        text = out if ok and out else (err or out)
        return text[:cap]
    if isinstance(data, list):
        return f"{len(data)} result{'s' if len(data) != 1 else ''}"
    if isinstance(data, dict):
        if data.get("error") and not ok:
            return str(data["error"])[:cap]
        files = data.get("files") or data.get("delivered")
        if isinstance(files, list) and files:
            return ", ".join(str(f.get("filename") or f.get("url"))[:60] for f in files if isinstance(f, dict))[:cap]
        if data.get("url"):
            return str(data["url"])[:cap]
    return (result_str or "")[:cap]


def friendly_label(tool: str, args: dict) -> str:
    fn = _LABELS.get(tool)
    try:
        return fn(args or {}) if fn else f"Using {tool.replace('_', ' ')}"
    except Exception:
        return f"Using {tool.replace('_', ' ')}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. TOOL RESULT -> BLOCKS (what the user SEES, emitted the moment it exists)
# ─────────────────────────────────────────────────────────────────────────────
def _loads(s):
    try:
        return json.loads(s)
    except Exception:
        return None


_IMAGE_URL_HINTS = (
    "/crawler/media", "lookaside.fbsbx.com", "fbcdn.net", "cdninstagram.com", "encrypted-tbn",
    "gstatic.com/images", "googleusercontent.com", "pbs.twimg.com", "images.unsplash.com",
    "i.pinimg.com", "upload.wikimedia.org", "/image/", "/images/", "/media/",
)


def _is_image(name: str) -> bool:
    low = (name or "").lower()
    ext = os.path.splitext(low.split("?")[0])[1]
    if ext in IMAGE_EXTS:
        return True
    if ext in (".pdf", ".zip", ".wav", ".mp3", ".mp4", ".txt", ".json", ".csv", ".docx", ".xlsx", ".html", ".py", ".js"):
        return False                                    # a known non-image type is never an image
    return low.startswith("http") and any(h in low for h in _IMAGE_URL_HINTS)


def _image_item(url, title="", generated=False):
    return {"image": url, "thumbnail": url, "title": title or "", "source": "", "generated": generated}


def _nice_name(name, url: str) -> str:
    """A real filename for the card — never a URL path like COM/LOOKASIDE/CRAWLER/MEDIA/."""
    name = (name or "").strip()
    if name and "/" not in name and not name.lower().startswith("http"):
        return name
    tail = os.path.basename((url or "").split("?")[0].rstrip("/")) or "file"
    return tail[:60]


def blocks_from_result(tool: str, args: dict, result_str: str, ok: bool, shown_images: Optional[set] = None) -> list:
    """Visible blocks (images / file cards / code run) a tool result produces. Sources are
    handled separately (they need de-duplication across the whole turn)."""
    if not ok:
        return []
    data = _loads(result_str)
    out = []

    def add_files(files):
        imgs, others = [], []
        for f in files or []:
            if not isinstance(f, dict) or not f.get("url"):
                continue
            name = f.get("filename") or ""
            if shown_images and f["url"] in shown_images:
                continue                                  # already in a gallery above — don't repeat it as a file card
            (imgs if (f.get("kind") == "image" or _is_image(name) or _is_image(f["url"])) else others).append(f)
        if imgs:
            out.append({"kind": "images", "images": [_image_item(f["url"], f.get("filename")) for f in imgs]})
        for f in others:
            out.append({"kind": "file", "file": {"filename": _nice_name(f.get("filename"), f["url"]), "url": f["url"]}})

    if tool == "search_images" and isinstance(data, list):
        imgs = [d for d in data if isinstance(d, dict) and d.get("image")]
        if imgs:
            out.append({"kind": "images", "images": imgs})
    elif tool == "generate_image" and isinstance(data, dict):
        imgs = [d for d in (data.get("images") or []) if isinstance(d, dict) and d.get("image")]
        if imgs:
            out.append({"kind": "images", "images": imgs})
    elif tool in ("output_files", "workspace_write") and isinstance(data, dict):
        add_files(data.get("files") or data.get("delivered"))
    elif tool in ("workspace_zip", "remove_background", "create_image") and isinstance(data, dict):
        if data.get("success") and data.get("url"):
            add_files([{"filename": data.get("filename"), "url": data["url"]}])
    elif tool == "run_code" and isinstance(data, dict):
        out.append({"kind": "code", "code": data})
        add_files(data.get("files"))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 5. THE TURN
# ─────────────────────────────────────────────────────────────────────────────
def _compact_old_tool_results(messages: list, keep_full: int = 4, old_chars: int = 700) -> None:
    idx = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    for i in idx[:-keep_full] if keep_full else idx:
        c = messages[i].get("content") or ""
        if len(c) > old_chars:
            messages[i]["content"] = c[:old_chars] + f"\n[…shortened — {len(c) - old_chars} more chars]"


def run_native_turn(messages: list, providers: list, session_context: dict, *,
                    tools: list, execute: Callable, describe_call: Callable,
                    sources_fn: Optional[Callable] = None, normalize_ask: Optional[Callable] = None,
                    stream: Callable = stream_chat, max_rounds: int = 20, temperature: float = 0.4,
                    max_tokens: int = 4000, reasoning_effort: Optional[str] = None,
                    post_process: Optional[Callable] = None, parse_suggestions: Optional[Callable] = None,
                    strip_markers: Optional[Callable] = None, build_segments: Optional[Callable] = None,
                    on_failure: Optional[Callable] = None, state: Optional[dict] = None):
    """
    Generator of the events described at the top of this file. `state` (optional dict) is
    filled with tools_called / tools_ok / file_results / code_results / image_results /
    sources so the caller can build its skills context from the same lists.
    """
    state = state if state is not None else {}
    for k in ("tools_called", "tools_ok", "file_results", "code_results", "image_results", "sources"):
        state.setdefault(k, [])
    strip_markers = strip_markers or (lambda t: t)
    parse_suggestions = parse_suggestions or (lambda t: (t, []))
    build_segments = build_segments or (lambda t: [{"type": "text", "content": t}] if t else [])

    providers = apply_model_pin(providers)
    blocks: list = []                       # ordered, final form
    seen_sources = set()
    provider_name = None
    forced_final = False
    sticky = None                           # the model that handled the last round — it goes first next round
    flat_retry = False                      # the plain-text-history retry is used at most once per turn

    # Last resort only — every model died mid-answer. Plain words, no server talk, and "continue" really does resume.
    BUSY_NOTE = "I got cut off right here. Send \"continue\" and I'll pick up from this exact point."
    # The last step never produced a single word (not a cut-off) — say that instead.
    STEP_NOTE = "I hit a problem on that last step. Send \"continue\" and I'll try again."
    used_models: list = []                  # every model that wrote part of this reply, in order

    def models_info() -> dict:
        return {"models": [{"name": n, "label": _model_label(n)} for n in used_models],
                "model_count": len(used_models)}

    def fail_final(reason, mid_answer=False):
        # Words, tool results or images already produced this turn are KEPT — never wipe a good partial answer.
        kept = [b for b in blocks if not (b["type"] == "text" and not b["text"].strip())]
        has_content = any(b["type"] in ("text", "images", "file", "code") for b in kept)
        if has_content:
            print(f"[NATIVE] giving up (partial kept): {reason}")
            for b in kept:
                if b["type"] == "text":
                    b["text"] = repair_mojibake(scrub_internal(strip_markers(b["text"])))
                    b["segments"] = build_segments(b["text"])
            note = BUSY_NOTE if mid_answer else STEP_NOTE
            kept.append({"type": "text", "text": note, "segments": build_segments(note)})
            ans = "\n\n".join(b["text"] for b in kept if b["type"] == "text").strip()
            return {"type": "final", "answer": ans, "blocks": kept, "segments": build_segments(ans),
                    "sources": state["sources"], "images": state["image_results"],
                    "files": state["file_results"], "file": (state["file_results"] or [None])[-1],
                    "codes": state["code_results"], "suggestions": [], "provider": provider_name, **models_info()}
        msg = on_failure() if on_failure else "Something went wrong. Please try again."
        print(f"[NATIVE] giving up: {reason}")
        return {"type": "final", "answer": msg, "blocks": [{"type": "text", "text": msg, "segments": build_segments(msg)}],
                "segments": build_segments(msg), "sources": state["sources"], "images": state["image_results"],
                "files": state["file_results"], "file": (state["file_results"] or [None])[-1],
                "codes": state["code_results"], "suggestions": [], "provider": provider_name, **models_info()}

    for round_no in range(max_rounds + 1):
        use_tools = None if (forced_final or round_no == max_rounds) else tools
        gate = TextGate()
        done = None
        failure = None
        round_providers = ([p for p in providers if p.get("name") == sticky] +
                           [p for p in providers if p.get("name") != sticky]) if sticky else providers
        for kind, val in stream(round_providers, messages, use_tools, temperature=temperature,
                                max_tokens=max_tokens, reasoning_effort=reasoning_effort):
            if kind == "text":
                safe = gate.feed(val)
                if safe:
                    yield {"type": "text", "delta": safe}
            elif kind == "done":
                done = val
            elif kind == "fail":
                failure = val
        if done is None:
            if state["tools_called"] and not flat_retry:
                # Every model refused the follow-up to a tool call. Often that is a model choking on the
                # tool-call history, not a real outage — retry once with the history as plain text, no tools.
                flat_retry = True
                forced_final = True
                messages[:] = _flatten_tool_history(messages)
                messages.append({"role": "user", "content":
                                 "Now reply to the user in plain words using the tool results above. "
                                 "Do not call any more tools."})
                print(f"[NATIVE] follow-up after a tool call failed — retrying with plain-text history: {failure}")
                continue
            yield fail_final(failure or "no answer")
            return
        tail = gate.flush()
        if tail:
            yield {"type": "text", "delta": tail}
        provider_name = done.get("provider") or provider_name
        sticky = done.get("provider") or sticky
        for n in (done.get("providers") or [done.get("provider")]):
            if n and n not in used_models:
                used_models.append(n)
        round_text = clean_round_text(gate.raw)
        calls = (done.get("tool_calls") or [])[:1]   # one tool per round: the model speaks between every tool call

        if round_text:
            blocks.append({"type": "text", "text": round_text})
        if done.get("interrupted") and not (done.get("tool_calls") or []):
            yield fail_final("stream interrupted", mid_answer=True)
            return

        if not calls:
            if not round_text and not forced_final:
                # The model said nothing (or only wrote reasoning) — ask once more, tools off, so the user gets words.
                forced_final = True
                messages.append({"role": "user", "content": (
                    "Now reply to the user in plain words using what you found. Do not call any more tools."
                    if state["tools_called"] else
                    "Reply to the user directly in plain words. Do not write your reasoning or use any tags.")})
                continue
            break

        # ── the model asked for tools: record the assistant turn, run each call in order ──
        messages.append({
            "role": "assistant", "content": round_text or None,
            "tool_calls": [{"id": c["id"], "type": "function",
                            "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}} for c in calls],
        })
        for c in calls:
            name, cid = c["name"], c["id"]
            try:
                args = json.loads(c["arguments"] or "{}")
                if not isinstance(args, dict):
                    raise ValueError("arguments must be a JSON object")
                bad_args = None
            except Exception as e:
                args, bad_args = {}, f"Your arguments for {name} were not valid JSON ({e}). Send a complete JSON object."

            if name == "ask_user" and normalize_ask and not bad_args:
                payload = normalize_ask(args)
                if payload:
                    yield {"type": "ask_user", **payload}
                    return

            state["tools_called"].append(name)
            info = describe_call(name, args)
            chip = {"type": "tool", "id": cid, "tool": name, "text": info.get("text") or name,
                    "icon": info.get("icon") or "tool", "ok": None,
                    "input": tool_input_preview(name, args), "output": None}
            blocks.append(chip)
            yield {"type": "tool_call", "id": cid, "tool": name, "args": info.get("args", args),
                   "text": chip["text"], "icon": chip["icon"], "call": info.get("call"), "input": chip["input"]}

            if bad_args:
                ok, result = False, bad_args
            else:
                ok, result = execute(name, args, session_context)
            chip["ok"] = bool(ok)
            chip["output"] = tool_output_preview(name, bool(ok), result)
            if ok:
                state["tools_ok"].append(name)
            yield {"type": "tool_result", "id": cid, "tool": name, "ok": bool(ok), "summary": (result or "")[:300],
                   "output": chip["output"]}

            shown_urls = {i.get("image") for i in state["image_results"] if isinstance(i, dict)}
            new_blocks = blocks_from_result(name, args, result, ok, shown_urls)
            if ok and sources_fn:
                fresh = []
                for s in (sources_fn(name, result, args) or []):
                    u = s.get("url")
                    if u and u not in seen_sources:
                        seen_sources.add(u)
                        fresh.append({"title": s.get("title") or u, "url": u})
                if fresh:
                    new_blocks.append({"kind": "sources", "sources": fresh})
            for b in new_blocks:
                kind = b["kind"]
                if kind == "images":
                    state["image_results"] = list(state["image_results"]) + b["images"]
                    session_context["image_results"] = state["image_results"]
                    blocks.append({"type": "images", "images": b["images"]})
                elif kind == "file":
                    state["file_results"].append({"type": "file_result", "success": True, **b["file"]})
                    blocks.append({"type": "file", "file": b["file"]})
                elif kind == "sources":
                    state["sources"].extend(b["sources"])
                    blocks.append({"type": "sources", "sources": b["sources"]})
                elif kind == "code":
                    state["code_results"].append(b["code"])
                    blocks.append({"type": "code", "code": b["code"]})
                yield {"type": "block", **b}

            messages.append({"role": "tool", "tool_call_id": cid, "name": name, "content": result or ""})
        _compact_old_tool_results(messages)
    else:
        pass

    # ── finish: clean the LAST text block (suggestions, tags, honesty), build segments ──
    suggestions = []
    last_text = max((i for i, b in enumerate(blocks) if b["type"] == "text"), default=None)
    for i, b in enumerate(blocks):
        if b["type"] != "text":
            continue
        t = repair_mojibake(scrub_internal(strip_markers(b["text"])))
        if i == last_text:
            t, suggestions = parse_suggestions(t)
            if post_process:
                t = post_process(t)
        b["text"] = t
        b["segments"] = build_segments(t)
    blocks = [b for b in blocks if b["type"] != "text" or b["text"].strip()]
    answer = "\n\n".join(b["text"] for b in blocks if b["type"] == "text").strip()
    if not answer and not any(b["type"] in ("images", "file", "code") for b in blocks):
        yield fail_final("empty answer")
        return
    yield {"type": "final", "answer": answer, "blocks": blocks, "segments": build_segments(answer),
           "sources": state["sources"], "images": state["image_results"], "files": state["file_results"],
           "file": (state["file_results"] or [None])[-1], "codes": state["code_results"],
           "suggestions": suggestions, "provider": provider_name, **models_info()}
