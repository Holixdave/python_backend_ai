#!/usr/bin/env python3
# gpt2_tools.py — function manifest + echo-trigger loop
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: lets the AI see a menu of real backend functions and decide FOR
# ITSELF when it wants to call one — instead of us hardcoding "if intent==X
# then call function Y". The AI never gets direct code-execution access;
# it only ever gets to name a function from TOOL_REGISTRY and supply args,
# which we validate and dispatch on the Python side.
#
# FLOW (2 round trips per tool the AI decides to use):
#   1. The AI is shown a short manifest (name + one-line purpose) and told
#      how to REQUEST a tool by echoing a <<TOOL_REQUEST>> block.
#   2. We detect that echo, and instead of guessing what args it needs, we
#      hand the AI the REAL source of that function (inspect.getsource) so
#      it knows the exact signature — no hand-maintained docs to go stale.
#   3. The AI replies with a <<TOOL_CALL>> block containing real args.
#   4. We parse + execute the real function, feed the result back, and the
#      AI continues to its final answer.
#
# This file only imports from gpt2_functions.py — it does NOT import
# gpt2_test.py, so it can't create a new circular-import chain. gpt2_test.py
# is the one that imports THIS file and drives the loop (it already owns
# _call_provider_chain via gpt2_functions).
# ─────────────────────────────────────────────────────────────────────────────

import inspect
import json
import re
from typing import Optional

from gpt2_functions import (
    search_web,
    search_images,
    _fetch_og_image,
    _verify_image_relevance,
    ask_with_vision,
    build_file_with_continuation,
)

# ---------------------------------------------------------------------------
# REGISTRY — every function the AI is allowed to self-trigger. Keys are the
# names the AI sees and echoes; values are the real, callable functions.
# "All of em" per the current build phase — narrow this later if any of
# these turn out to be too easy to misuse standalone.
# ---------------------------------------------------------------------------
TOOL_REGISTRY = {
    "search_web": search_web,
    "search_images": search_images,
    "fetch_page_preview_image": _fetch_og_image,
    "verify_image_relevance": _verify_image_relevance,
    "analyze_image": ask_with_vision,
    "build_file": build_file_with_continuation,
}

# Short, hand-written purpose lines for the initial manifest only — this is
# deliberately NOT the full docs. Once the AI picks one, it gets the real
# source instead of trusting this one-liner for anything beyond "should I
# use this at all".
TOOL_DESCRIPTIONS = {
    "search_web": "Search the live web for current info; returns text + source links.",
    "search_images": "Search the web for candidate images matching a query.",
    "fetch_page_preview_image": "Grab a specific webpage's own declared preview image.",
    "verify_image_relevance": "Check a batch of candidate images against a query using vision.",
    "analyze_image": "Look at image(s) and answer a question about what's in them.",
    "build_file": "Build a complete downloadable file and upload it for the user.",
}

# Params that should NEVER come from the AI's own JSON — they belong to the
# current request/session and get auto-injected by gpt2_test.py instead, so
# the AI can't (and doesn't need to) invent a userid, fabricate history, etc.
# image_results is here too: it chains automatically from whatever
# search_images just returned this turn, rather than requiring the model to
# retype a huge JSON array of results by hand — that was the actual cause
# of verify_image_relevance calls failing (the model tried to paste the
# full result set back in, and it got cut off mid-JSON by the token cap).
SESSION_INJECTED_PARAMS = {"prompt", "history", "userid", "image_urls", "image_results"}

MAX_TOOL_ROUNDS = 3

# ---------------------------------------------------------------------------
# MARKERS — distinctive on purpose. A bare "name()" or a loose JSON object
# is too easy for a model to almost-produce by accident while just talking
# about code. Wrapping real JSON in a tag this specific only fires when the
# model actually means to invoke it.
# ---------------------------------------------------------------------------
_TOOL_REQUEST_RE = re.compile(r"<<TOOL_REQUEST>>\s*(\{.*?\})\s*<<END_TOOL_REQUEST>>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<<TOOL_CALL>>\s*(\{.*?\})\s*<<END_TOOL_CALL>>", re.DOTALL)


def build_tool_manifest() -> str:
    """
    Renders TOOL_REGISTRY into the short menu shown to the AI up front, plus
    the exact marker syntax it must use to request one. Regenerated from the
    registry every time, so a function added/removed from TOOL_REGISTRY is
    automatically reflected — nothing to keep in sync by hand.
    """
    lines = [
        "AVAILABLE TOOLS — you are not limited to your own knowledge. If one "
        "of these would genuinely help answer the user, you may request it. "
        "Do not request a tool for things you already know confidently.",
        "",
    ]
    for name, fn in TOOL_REGISTRY.items():
        desc = TOOL_DESCRIPTIONS.get(name, "No description available.")
        lines.append(f"- {name}: {desc}")
    lines.append("")
    lines.append(
        "TO REQUEST A TOOL: output exactly one block in this form, with "
        "nothing else on those lines:\n"
        '<<TOOL_REQUEST>>{"tool": "tool_name_here"}<<END_TOOL_REQUEST>>\n'
        "You will then be shown that tool's real source code and asked to "
        "call it with real arguments — so just name the tool here, don't "
        "guess arguments yet."
    )
    return "\n".join(lines)


def strip_tool_markers(text: Optional[str]) -> str:
    """
    Safety net — removes any raw <<TOOL_REQUEST>>/<<TOOL_CALL>> block from
    text before it's ever shown to the user. This is deliberately called
    unconditionally on every final answer in gpt2_test.py, not just when
    the tool loop thinks it handled something — a model that skips the
    intended protocol, produces malformed JSON, or gets cut off by
    MAX_TOOL_ROUNDS should never be able to leak raw marker syntax into a
    real chat bubble, no matter which of those causes it.
    """
    if not text:
        return text
    text = _TOOL_REQUEST_RE.sub("", text)
    text = _TOOL_CALL_RE.sub("", text)
    return text.strip()


def detect_tool_request(text: Optional[str]) -> Optional[str]:
    """
    Scans model output for a <<TOOL_REQUEST>> block. Returns the requested
    tool name if it's a real, valid JSON block naming a tool that actually
    exists in TOOL_REGISTRY — None otherwise (including malformed JSON or
    an unknown tool name, so a hallucinated tool name never gets dispatched).
    """
    if not text:
        return None
    match = _TOOL_REQUEST_RE.search(text)
    if not match:
        return None
    try:
        data = json.loads(match.group(1))
        tool_name = data.get("tool")
    except Exception:
        return None
    if tool_name in TOOL_REGISTRY:
        return tool_name
    return None


def get_tool_source(tool_name: str) -> str:
    """
    Returns the real source of a registered tool via inspect.getsource() —
    this is the AI's actual documentation, always in sync with the code
    because it IS the code. Falls back to the docstring/signature if source
    can't be read for some reason (e.g. running from a frozen build).
    """
    fn = TOOL_REGISTRY.get(tool_name)
    if fn is None:
        return ""
    try:
        return inspect.getsource(fn)
    except Exception:
        sig = str(inspect.signature(fn)) if _safe_signature(fn) else "(...)"
        doc = fn.__doc__ or "No docstring available."
        return f"def {tool_name}{sig}:\n    \"\"\"{doc}\"\"\""


def _safe_signature(fn) -> bool:
    try:
        inspect.signature(fn)
        return True
    except Exception:
        return False


def parse_tool_call(text: Optional[str]) -> Optional[dict]:
    """
    Scans model output for a <<TOOL_CALL>> block containing the real tool
    name + args to execute. Returns {"tool": str, "args": dict} or None if
    the block is missing, malformed, or names an unregistered tool.
    """
    if not text:
        return None
    match = _TOOL_CALL_RE.search(text)
    if not match:
        return None
    try:
        data = json.loads(match.group(1))
    except Exception:
        return None
    tool_name = data.get("tool")
    if tool_name not in TOOL_REGISTRY:
        return None
    args = data.get("args") or {}
    if not isinstance(args, dict):
        return None
    return {"tool": tool_name, "args": args}


def execute_tool(tool_name: str, ai_args: dict, session_context: dict):
    """
    Runs the real function. Session-owned params (prompt/history/userid/
    image_urls — see SESSION_INJECTED_PARAMS) are pulled from
    session_context automatically rather than trusted from the AI's own
    JSON; everything else comes from ai_args. Handles both normal
    functions and generator-based ones (like build_file_with_continuation,
    which yields progress events) by draining the generator and keeping
    only its final meaningful result.

    Returns (success: bool, result_str: str). Never raises — a tool failure
    is just fed back to the AI as a failure message so it can recover
    (e.g. answer without the tool, or try a different one).
    """
    fn = TOOL_REGISTRY.get(tool_name)
    if fn is None:
        return False, f"Unknown tool '{tool_name}'."

    try:
        sig = inspect.signature(fn)
        final_args = {}
        for pname in sig.parameters:
            if pname in SESSION_INJECTED_PARAMS and pname in session_context:
                final_args[pname] = session_context[pname]
        final_args.update(ai_args)  # AI-supplied args always win for non-session params

        if inspect.isgeneratorfunction(fn):
            last_event = None
            for event in fn(**final_args):
                last_event = event
            if last_event is None:
                return False, "Tool produced no output."
            return True, json.dumps(last_event, default=str)[:4000]

        result = fn(**final_args)
        return True, json.dumps(result, default=str)[:4000]

    except TypeError as e:
        # Wrong/missing args — surfaced plainly so the AI can retry with a
        # corrected <<TOOL_CALL>> instead of silently failing.
        return False, f"Called {tool_name} with invalid arguments: {e}"
    except Exception as e:
        return False, f"{tool_name} failed: {e}"
