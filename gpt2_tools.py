# @hint: "gpt2_tools.py"
# @tags: user-upload
# @date: 2026-10-15T11:53:05.806603
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

import gpt2_approvals

def see_tool_arg(tool_name: str) -> str:
    """
    Meta-tool — lets the AI look up any OTHER registered tool's real source
    code on demand, without waiting for the backend-forced REQUEST/CALL
    round trip. Call this BEFORE calling an unfamiliar tool if unsure of
    its exact argument names, instead of guessing.
    """
    if tool_name not in TOOL_REGISTRY:
        return f"No tool named '{tool_name}'. Available tools: {', '.join(TOOL_REGISTRY.keys())}"
    return get_tool_source(tool_name)


MAX_ASK_QUESTIONS = 3


def ask_user_multi_enabled() -> bool:
    """True once the app's bottom sheet can show several questions with a Next
    button (ON by default; ASK_USER_MULTI=0 turns it off). When off only ONE
    question is sent per sheet — extra questions are asked one at a time."""
    import os
    return os.environ.get("ASK_USER_MULTI", "1").strip().lower() in ("1", "true", "yes", "on")


def _normalize_one_question(args: dict) -> Optional[dict]:
    """One question + its 2-6 options, or None if unusable."""
    import uuid
    if not isinstance(args, dict):
        return None
    question = str(args.get("question") or "").strip()
    raw_options = args.get("options")
    if not question or not isinstance(raw_options, list):
        return None

    options, seen, seen_labels = [], set(), set()
    for i, raw in enumerate(raw_options):
        if isinstance(raw, dict):
            label = str(raw.get("label") or "").strip()
            desc = str(raw.get("description") or "").strip() or None
            oid = str(raw.get("id") or "").strip()
        else:
            label, desc, oid = str(raw or "").strip(), None, ""
        if not label:
            continue
        oid = oid or f"opt{i + 1}"
        if oid in seen or label.lower() in seen_labels:
            continue
        seen_labels.add(label.lower())
        seen.add(oid)
        options.append({"id": oid, "label": label[:80], "description": desc[:160] if desc else None})
        if len(options) == 6:  # the sheet is meant for quick taps, not a menu
            break
    if len(options) < 2:
        return None

    return {
        "question_id": uuid.uuid4().hex[:16],
        "question": question[:300],
        "options": options,
        "allow_multiple": bool(args.get("allow_multiple", False)),
        "allow_other": bool(args.get("allow_other", True)),
    }


def normalize_ask_user(args: dict) -> Optional[dict]:
    """
    Validates/cleans the arguments of an ask_user tool call into the exact
    payload the app's bottom sheet expects. Returns None if the call is not
    usable (no question, or fewer than 2 usable options) so the caller can
    hand an error back to the model instead of showing a broken sheet.

    SINGLE question (always works): {question, options, allow_multiple, allow_other}
      -> payload {question_id, question, options, allow_multiple, allow_other}.

    SEVERAL questions: pass "questions": [ {question, options, ...}, ... ] (max 3).
      -> the same top-level fields as the FIRST question (so an app that only
         knows the single-question sheet still works and shows question 1),
         plus "questions": [all cleaned questions] and "total": n.
      The extra questions are only sent when ask_user_multi_enabled() is True;
      otherwise only the first is sent and the model asks the next one after
      the user answers.
    """
    if not isinstance(args, dict):
        return None

    raw_qs = args.get("questions")
    if isinstance(raw_qs, list) and raw_qs:
        cleaned = [q for q in (_normalize_one_question(r) for r in raw_qs[:MAX_ASK_QUESTIONS]) if q]
        if not cleaned:
            return None
        if not ask_user_multi_enabled():
            cleaned = cleaned[:1]
        payload = dict(cleaned[0])
        if len(cleaned) > 1:
            payload["questions"] = cleaned
            payload["total"] = len(cleaned)
        return payload

    return _normalize_one_question(args)


def ask_user(question: str = "", options: Optional[list] = None, allow_multiple: bool = False,
             allow_other: bool = True, questions: Optional[list] = None):
    """
    Ask the user a quick multiple-choice question in a bottom sheet in
    their app, instead of guessing. The turn STOPS here: the sheet opens,
    the user taps an answer, and their answer comes back as their next
    message — so write NOTHING else in the same reply.

    One question: pass question + options. Several related questions in one
    sheet (the user answers each, then taps Next): pass questions=[{question,
    options, allow_multiple, allow_other}, ...] (max 3). If the app's sheet
    does not support several yet, only the FIRST is shown — so put the most
    important one first, and ask the next one after they answer.

    Use it only when a wrong guess would waste the user's time and the
    request genuinely can't be answered well without knowing their choice
    (which subject/topic/level, which of several meanings, which file
    format/style before you build something big). Do NOT use it for things
    you can reasonably assume or for simple factual questions. Don't ask again
    once the user has answered unless one essential detail is still missing
    (never more than 3 questions in total for one request) — otherwise proceed.

    Args:
        question: the question, one short sentence (no markdown).
        options: 2-6 choices. Each is either a plain string, or
            {"label": "Short label", "description": "optional hint"}.
        allow_multiple: True if the user may pick more than one option.
        allow_other: True (default) lets the user type their own answer.

    Example:
    <<TOOL_CALL>>{"tool": "ask_user", "args": {"question": "Which subject should the quiz cover?", "options": [{"label": "Mathematics"}, {"label": "Physics", "description": "Mechanics and waves"}, "Chemistry"], "allow_multiple": false, "allow_other": true}}<<END_TOOL_CALL>>
    """
    # Normally never reached: gpt2_test.py intercepts a valid ask_user call and
    # ends the turn with an ask_user event. This only runs when the call was
    # invalid, so the model gets a clear reason and can fix it or just answer.
    payload = normalize_ask_user({"question": question, "options": options,
                                  "allow_multiple": allow_multiple, "allow_other": allow_other,
                                  "questions": questions})
    if payload is None:
        return {"error": "ask_user needs a non-empty question and 2-6 non-empty options "
                         "(or a questions list where each has both). "
                         "Fix the call, or skip the question and answer with your best assumption."}
    return {"asked": True}


from gpt2_functions import (
    search_web,
    search_images,
    _fetch_og_image,
    ask_with_vision,
    build_file,
    build_multiple_files,
    build_zip_file,
    edit_file,
    edit_file_by_intent,
    remove_background,
    create_image,
    redisplay_file,
    fetch_webpage,
    search_github,
    fetch_github_file,
    fetch_document,
    get_user_profile,
    save_user_note,
    get_user_notes,
    save_study_note,
    get_study_notes,
    schedule_reminder,
    redisplay_images,
    generate_image,
    is_web_search_enabled,
)
from gpt2_sandbox import run_code
from gpt2_output import output_files
from gpt2_workspace_run import workspace_run
from gpt2_workspace import (
    workspace_list,
    workspace_read,
    workspace_write,
    workspace_delete,
    workspace_grep,
    workspace_unzip,
    workspace_github_clone,
    workspace_github_folder,
    workspace_zip,
    workspace_copy,
    workspace_edit,
    workspace_edit_by_intent,
    workspace_append,
    save_pasted_text,
)
from gpt2_design_check import design_check
from gpt2_codecheck import python_check

# Tools that reach out to the internet on the user's behalf. Every one of
# these is gated by the user's own web_search toggle (see
# is_web_search_enabled in gpt2_functions.py) before it's allowed to run —
# not just search_web. fetch_page_preview_image/fetch_webpage/search_github/
# fetch_github_file/fetch_document all hit external URLs/APIs too.
WEB_TOOLS = {
    "search_web",
    "search_images",
    "fetch_page_preview_image",
    "fetch_webpage",
    "search_github",
    "fetch_github_file",
    "fetch_document",
    "remove_background",
    "workspace_github_clone",
    "workspace_github_folder",
}

# ---------------------------------------------------------------------------
# REGISTRY — every function the AI is allowed to self-trigger. Keys are the
# names the AI sees and echoes; values are the real, callable functions.
# "All of em" per the current build phase — narrow this later if any of
# these turn out to be too easy to misuse standalone.
# ---------------------------------------------------------------------------
from publish_pages import publish_page as _publish_page_impl


def publish_page_tool(html: str, title: str = "page") -> dict:
    """Publish one self-contained HTML page to a public link (GitHub Pages)."""
    return _publish_page_impl(html, title)


TOOL_REGISTRY = {
    "search_web": search_web,
    "search_images": search_images,
    "fetch_page_preview_image": _fetch_og_image,
    "analyze_image": ask_with_vision,
    "remove_background": remove_background,
    "create_image": create_image,
    "fetch_webpage": fetch_webpage,
    "search_github": search_github,
    "fetch_github_file": fetch_github_file,
    "fetch_document": fetch_document,
    "get_user_profile": get_user_profile,
    "save_user_note": save_user_note,
    "get_user_notes": get_user_notes,
    "save_study_note": save_study_note,
    "get_study_notes": get_study_notes,
    "schedule_reminder": schedule_reminder,
    "generate_image": generate_image,
    "run_code": run_code,
    "ask_user": ask_user,
    # Persistent per-user workspace (gpt2_workspace.py)
    "workspace_list": workspace_list,
    "workspace_read": workspace_read,
    "workspace_write": workspace_write,
    "workspace_delete": workspace_delete,
    "workspace_grep": workspace_grep,
    "workspace_unzip": workspace_unzip,
    "workspace_github_clone": workspace_github_clone,
    "workspace_github_folder": workspace_github_folder,
    "workspace_zip": workspace_zip,
    "workspace_copy": workspace_copy,
    "output_files": output_files,
    "workspace_edit": workspace_edit,
    "workspace_edit_by_intent": workspace_edit_by_intent,
    "workspace_run": workspace_run,
    "workspace_append": workspace_append,
    "save_pasted_text": save_pasted_text,
    "design_check": design_check,
    "python_check": python_check,
    "publish_page": publish_page_tool,
}
# Short, hand-written purpose lines for the initial manifest only — this is
# deliberately NOT the full docs. Once the AI picks one, it gets the real
# source instead of trusting this one-liner for anything beyond "should I
# use this at all".
TOOL_DESCRIPTIONS = {
    "search_web": "Search the live web for current info; returns text + source links.",
    "search_images": "Search the web for candidate images matching a query.",
    "fetch_page_preview_image": "Grab a specific webpage's own declared preview image.",
    "analyze_image": "Look at image(s) and answer a question about what's in them.",
    "remove_background": "Remove the background from an existing image (given its url) and get back a transparent PNG. ALWAYS use this for background removal — never try it with run_code (far too slow there).",
    "create_image": "Draw a real image from simple shapes/text (rectangles, ellipses, lines, polygons, text) — describe what to draw as a list of operations, this renders and uploads the actual image.",
    "fetch_webpage": "Open a Specific url and read its actual page text/content.",
    "search_github": "Search GitHub for repositories or code matching a query.",
    "fetch_github_file": "Fetch a specific file's content from a GitHub repo by path.",
    "fetch_document": "Download a PDF or Word (.docx) document from a URL and read its text content.",
    "get_user_profile": "Look up the current user's real name/display name so you can address them personally.",
    "save_user_note": "Save a fact or preference about the current user for future conversations.",
    "get_user_notes": "Retrieve previously saved facts/notes about the current user.",
    "save_study_note": "Save a drafted question or note to the user's JAMB study notebook.",
    "get_study_notes": "Retrieve the user's saved JAMB study notes/questions to review or help solve.",
    "schedule_reminder": "Schedule a push-notification reminder for the user at a specific future time.",
    "generate_image": "Generate a brand-new AI image from a text description and show it in a real gallery, same as search_images.",
    "run_code": "Actually execute code you wrote (python/javascript/bash) in a sandbox and get back real stdout/stderr/exit_code — use this whenever the user asks you to run/execute/test code, not just to verify build_file's output. Supports an optional packages list (python only) to auto-install third-party imports like numpy/requests before running — no separate pip install step needed. To READ files the user already has, pass input_files=[workspace paths/folders] — they are copied into ./inputs/ for that run (open('inputs/data.csv')); without it the code starts in an EMPTY folder and cannot see any workspace file. For a whole project with its own files/tests use workspace_run. To produce a real file (audio, image, pdf, zip, csv, etc.), write it into the ./outputs/ folder from your code, e.g. open('outputs/tone.wav','wb'); every file saved there lands in your workspace under outputs/ and is listed in `saved_to_workspace` — it is NOT shown to the user until you call output_files with those paths. Never claim the user has a file unless output_files returned it.",
    "ask_user": "Ask the user a quick multiple-choice question in a bottom sheet in their app (2-6 options, optional 'Other' box). Pass questions=[...] (max 3) to ask several related questions in one sheet; if the app can only show one, only the first appears, so put the most important first and ask the next after they answer. Use it when the request is genuinely ambiguous and a wrong guess would waste their time. The turn stops when you call it and their answer comes back as their next message — so write nothing else in that reply.",
    "workspace_github_clone": "Download a WHOLE GitHub repo (owner/name or URL, optional ref/branch) into your workspace — no truncation. Use this instead of fetch_github_file whenever you need more than one small file. Then explore it with workspace_list / workspace_grep / workspace_read.",
    "workspace_github_folder": "Download ONE FOLDER of a GitHub repo (repo + path inside it, optional ref/branch) into an isolated folder in your workspace. Use this for BIG repos you cannot clone whole - e.g. repo \"googlefonts/noto-emoji\", path \"png/128\". Capped at 1500 files / 60MB; if it is bigger, pick a deeper sub-folder. Then explore it with workspace_list / workspace_grep / workspace_read.",
    "workspace_list": "Show the workspace file tree (or one folder in it) with file sizes. Start here after a clone/unzip to see what exists.",
    "workspace_grep": "Regex-search all text files in the workspace (or a folder) and get file:line matches. Use it to FIND code in a big repo, then workspace_read that range.",
    "workspace_read": "Read a workspace text file by line range (start_line/end_line) with total_lines — read big files in pieces; the result says which start_line to continue from.",
    "workspace_write": "Create or overwrite a text file in the workspace (path + COMPLETE content you wrote yourself). Pass deliver=true to ALSO send it to the user as a download card — this is how you hand over a file (call once per file).",
    "workspace_delete": "Delete a file or folder from the workspace.",
    "workspace_unzip": "Open a .zip that is in the workspace and extract it into a folder. (A .zip the user attaches is already saved and unzipped for you — just workspace_list it.)",
    "output_files": "DELIVER existing files to the user — nothing from run_code or the workspace is sent automatically, you choose. files = list of workspace paths (e.g. \"outputs/chart.png\") or http(s) URLs (e.g. from generate_image or earlier in the chat), mixed freely. Images show in an image gallery, other files as download cards, right where you call this. Do NOT pass images that search_images or generate_image already showed — those are displayed automatically. Only claim files that come back in the result.",
    "workspace_edit": "Change PART of a workspace file by exact find/replace (find_text must match once, or replace_all=true) — never retype a whole file for a small change. On a miss it tells you which line to re-read.",
    "workspace_edit_by_intent": "Edit ONE workspace file by describing the change in plain English — e.g. \"make the header text bigger\". You do NOT need the exact current text; a separate step reads the real file and works out the precise edit (retrying itself if not unique). Use this instead of workspace_edit when you can't see the file's exact wording. Files over ~60,000 characters: workspace_read + workspace_edit instead.",
    "workspace_run": "Run a shell command INSIDE a workspace project (e.g. \"pytest -q\", \"python app.py\", \"ls\") and get real stdout/stderr/exit_code. Runs on a copy of `cwd`; files it creates/changes are copied back and listed. Optional packages=[...] pip installs first. 60s default, 180s max. Use this (not run_code) whenever the code needs the workspace's files.",
    "result_more": "Get the NEXT part of a tool result that ended with '[truncated ... result_id ...]'. Args: result_id, offset (use the offset the note gives you).",
    "workspace_copy": "Copy a file or folder inside the workspace — mainly from uploads/ (where the user's attachments land) into the main workspace so you can edit or run it. The original stays in uploads/. Args: src, dest (optional, defaults to the same name at the top level).",
    "publish_page": "Publish ONE self-contained HTML page (all CSS/JS inline) to a public https link the user can open and share. ONLY call this after the user has said yes to publishing. Args: html (complete page), title (short). Returns the public url.",
    "workspace_append": "Add text to the END of a workspace file (creates it if missing). Use it to build a LONG file in parts: workspace_write part 1, then workspace_append each next part (each ending at a full paragraph/section/function), deliver=true only on the last part. Never send a long file in one huge call.",
    "save_pasted_text": "Save text the USER PASTED into a .txt/.md file exactly as they wrote it - the backend copies it, you do NOT retype it. Use whenever the user pastes a long text and asks to package/put/save it in a text file. source='this_message' (default) or 'earlier_message' when they pasted it before. Delivers the file by default.",
    "design_check": "LOOK at a visual file you built - a website (.html), a .pdf or an image - and get a real verdict: it runs static checks, renders the pages, and shows each to a vision reviewer. Call it after you build a page/PDF/design and again after each fix (max 3 rounds). Returns status passed / needs_fixes / not_visually_verified plus the exact issues to fix.",
    "python_check": "Check Python file(s) in the workspace for SYNTAX errors (and undefined names when available) WITHOUT running them; path = a .py file or folder. Call after every write/edit of Python, fix the listed lines, call again. A clean result is NOT proof it runs - then run it with workspace_run.",
    "workspace_zip": "Pack a workspace folder (or everything) into ONE .zip and get a download URL (then pass that URL to output_files to show it).",
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

MAX_TOOL_ROUNDS = 30  # raised from 3 — allows deeper multi-step chains (e.g. several rounds of search_web self-correction, or search + verify_image_relevance + search again)

# ---------------------------------------------------------------------------
# MARKERS — distinctive on purpose. A bare "name()" or a loose JSON object
# is too easy for a model to almost-produce by accident while just talking
# about code. Wrapping real JSON in a tag this specific only fires when the
# model actually means to invoke it.
#
# FIXED: weaker/free models (unlike Groq's bigger models) are less
# reliable at reproducing the exact "<<END_TOOL_REQUEST>>" closing tag —
# they sometimes drop the final ">", producing "<<END_TOOL_REQUEST>"
# instead. The old strict regex required an exact match, so that single
# dropped character meant: (a) detect_tool_request() never recognized the
# tool call, so it never actually ran, and (b) strip_tool_markers() used
# the SAME regex, so it never stripped the raw marker either — it leaked
# straight into the chat bubble as visible garbage. ">>?" below matches
# either ">" or ">>" on both the opening and closing tag, on both markers.
# ---------------------------------------------------------------------------
_TOOL_REQUEST_RE = re.compile(r"<<TOOL_REQUEST>>?\s*(\{.*?\})\s*<<END_TOOL_REQUEST>>?", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<<TOOL_CALL>>?\s*(\{.*?\})\s*<<END_TOOL_CALL>>?", re.DOTALL)

# Catch-all safety nets — if a model opens a marker but the closing tag
# never shows up at all (cut off by max_tokens, or some other malformed
# variant we haven't seen yet), strip everything from the opening tag to
# the end of the string rather than leaking a half-finished tool call.
_TOOL_REQUEST_OPEN_RE = re.compile(r"<<TOOL_REQUEST>>?.*", re.DOTALL)
_TOOL_CALL_OPEN_RE = re.compile(r"<<TOOL_CALL>>?.*", re.DOTALL)


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


_SUGGESTIONS_RE = re.compile(r"<<SUGGESTIONS>>\s*(\[.*?\])\s*<<END_SUGGESTIONS>>", re.DOTALL)


def extract_suggestions(text: Optional[str]) -> tuple:
    """
    Pulls a <<SUGGESTIONS>>[...]<<END_SUGGESTIONS>> block out of a final
    answer, same pattern as the tool markers. Returns (cleaned_text,
    suggestions_list) — cleaned_text has the marker stripped so it never
    leaks into the visible bubble, suggestions_list is [] if there was no
    block, it didn't parse as valid JSON, or wasn't a list of strings.

    Each string in the list is meant to be the EXACT, literal text that
    gets sent as the user's next message if they tap it — not a
    description of an option. The frontend renders these as tappable
    chips below the answer.
    """
    if not text:
        return text, []
    match = _SUGGESTIONS_RE.search(text)
    if not match:
        return text, []
    cleaned = _SUGGESTIONS_RE.sub("", text).strip()
    try:
        data = json.loads(match.group(1))
    except Exception:
        return cleaned, []
    if not isinstance(data, list):
        return cleaned, []
    suggestions = [s.strip() for s in data if isinstance(s, str) and s.strip()]
    return cleaned, suggestions[:4]  # cap at 4 — a wall of chips isn't useful


_INLINE_VISUAL_RE = re.compile(
    r"<<in_line_svg>>\s*(.*?)\s*<<in_line_svg>>"
    r"|<<inline_html>>\s*(.*?)\s*<<inline_html>>",
    re.DOTALL | re.IGNORECASE,
)
# Safety net for a malformed/cut-off tag pair (opening tag with no closing
# tag, e.g. the model got cut off by the token cap mid-visual). The main
# regex above simply won't match an unclosed pair, so the raw marker text
# would otherwise fall through untouched into a "text" segment and leak to
# the user. Same pattern as _ORPHAN_THINK_CLOSE_RE in gpt2_functions.py.
_ORPHAN_VISUAL_TAG_RE = re.compile(r"<<in_line_svg>>|<<inline_html>>", re.IGNORECASE)


# Blocks whose INTERIOR must never be scanned for inline visuals. The frontend
# renders these as their own widgets and one of them (<<exam>>) legitimately
# nests <<in_line_svg>>/<<inline_html>> inside a <<question>> as that
# question's graphic; a fenced code block may also just *show* the tags as an
# example. Splitting inside any of these tears the block in half (exam) or
# turns documentation into a live widget (fence).
_PROTECTED_BLOCK_RE = re.compile(
    r"```.*?```"
    r"|<<note\s[^>]*?>{1,2}.*?<<note>{1,2}"
    r"|<<exam\s[^>]*?>{1,2}.*?<<exam>{1,2}",
    re.DOTALL | re.IGNORECASE,
)


def build_answer_segments(text: Optional[str]) -> list:
    """
    Splits a cleaned model answer into an ORDERED list of typed segments,
    so the frontend never has to regex-scan raw text for <<in_line_svg>>
    / <<inline_html>> markers itself: it iterates this list and switches on
    segment["type"].

    Returns, in original order:
      {"type": "text", "content": "..."}                  -- markdown (may still contain
                                                             code fences, <<note>>, <<exam>>,
                                                             <<suggestion>> -- rendered client-side)
      {"type": "inline_svg", "content": "<svg>...</svg>"}  -- raw SVG markup
      {"type": "inline_html", "content": "<div>...</div>"} -- raw HTML/JS

    Fenced code blocks and <<note>>/<<exam>> blocks are PROTECTED: they are
    kept whole inside a text segment and never scanned for visuals (see
    _PROTECTED_BLOCK_RE). Everywhere else, tags are never nested and each
    pair holds exactly one visual, so one left-to-right scan is enough.

    Always returns at least one segment (possibly an empty "text" one) so
    callers can safely iterate blindly.
    """
    if not text:
        return [{"type": "text", "content": text or ""}]

    segments: list = []
    buf: list = []

    def flush_text():
        joined = "".join(buf).strip()
        buf.clear()
        if joined:
            segments.append({"type": "text", "content": joined})

    def scan_unprotected(chunk: str):
        pos = 0
        for match in _INLINE_VISUAL_RE.finditer(chunk):
            # Orphan/malformed opening tags (cut-off generation) are removed
            # from plain text ONLY here -- never from protected blocks, where
            # the literal tag text is legitimate content.
            buf.append(_ORPHAN_VISUAL_TAG_RE.sub("", chunk[pos:match.start()]))
            flush_text()
            svg_content, html_content = match.group(1), match.group(2)
            if svg_content is not None:
                segments.append({"type": "inline_svg", "content": svg_content.strip()})
            else:
                segments.append({"type": "inline_html", "content": html_content.strip()})
            pos = match.end()
        buf.append(_ORPHAN_VISUAL_TAG_RE.sub("", chunk[pos:]))

    cursor = 0
    for block in _PROTECTED_BLOCK_RE.finditer(text):
        scan_unprotected(text[cursor:block.start()])
        buf.append(block.group(0))
        cursor = block.end()
    scan_unprotected(text[cursor:])
    flush_text()

    if not segments:
        segments.append({"type": "text", "content": ""})
    return segments


def flatten_segments_to_text(segments: list) -> str:
    """
    Legacy-compat helper (not currently wired in — kept available for when
    an OLDER client needs a flat string and shouldn't see raw SVG/HTML
    markup dumped into a chat bubble). Collapses a segments list back into
    one plain string, replacing visual segments with a short readable
    placeholder instead of their raw markup.
    """
    parts = []
    for seg in segments:
        if seg["type"] == "text":
            parts.append(seg["content"])
        elif seg["type"] == "inline_svg":
            parts.append("[diagram]")
        elif seg["type"] == "inline_html":
            parts.append("[interactive widget]")
    return "\n\n".join(p for p in parts if p.strip()).strip()


# Internal memory notes are appended to the assistant's SAVED history (see
# main.py _build_memory_reply). The model sees them in its own past messages
# and imitates the format, echoing a fake note into the visible answer.
# This strips any such note (well-formed or cut off) before the user sees it.
_INTERNAL_NOTE_RE = re.compile(
    r"\s*\[INTERNAL MEMORY NOTE.*?\]\s*(?=\n\n|\Z)", re.DOTALL | re.IGNORECASE
)
_INTERNAL_NOTE_OPEN_RE = re.compile(r"\s*\[INTERNAL MEMORY NOTE.*\Z", re.DOTALL | re.IGNORECASE)


def strip_internal_notes(text: Optional[str]) -> str:
    if not text:
        return text
    text = _INTERNAL_NOTE_RE.sub("", text)
    text = _INTERNAL_NOTE_OPEN_RE.sub("", text)  # unterminated / truncated note
    return text.strip()


def strip_tool_markers(text: Optional[str]) -> str:
    """
    Safety net — removes any raw <<TOOL_REQUEST>>/<<TOOL_CALL>> block from
    text before it's ever shown to the user. This is deliberately called
    unconditionally on every final answer in gpt2_test.py, not just when
    the tool loop thinks it handled something — a model that skips the
    intended protocol, produces malformed JSON, or gets cut off by
    MAX_TOOL_ROUNDS should never be able to leak raw marker syntax into a
    real chat bubble, no matter which of those causes it.

    Two passes: first the well-formed (or near-well-formed, see the >>?
    tolerance above) open+close pair, then a catch-all that strips from an
    opening tag to end-of-string if no closing tag showed up at all — a
    half-emitted tool call is never useful to show the user either way.
    """
    if not text:
        return text
    text = _TOOL_REQUEST_RE.sub("", text)
    text = _TOOL_CALL_RE.sub("", text)
    text = _TOOL_REQUEST_OPEN_RE.sub("", text)
    text = _TOOL_CALL_OPEN_RE.sub("", text)
    text = strip_internal_notes(text)
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


# ---------------------------------------------------------------------------
# PAGED RESULTS — replaces the old hard `[:4000]` cut that dropped the end of
# every long tool result (logs, search text, pages). Long results are kept
# server-side; the model gets the first PAGE_CHARS plus an exact note on how to
# fetch the rest with result_more. Small results pass through untouched.
# ---------------------------------------------------------------------------
PAGE_CHARS = 6000
_RESULT_STORE: dict = {}   # userid -> {result_id: full_text}, newest last, max 10 each


def _page_result(userid, tool_name: str, text: str) -> str:
    if len(text) <= PAGE_CHARS:
        return text
    import uuid
    rid = uuid.uuid4().hex[:8]
    store = _RESULT_STORE.setdefault(userid or "anonymous", {})
    store[rid] = text
    while len(store) > 10:
        store.pop(next(iter(store)))
    return (text[:PAGE_CHARS]
            + f"\n[truncated: {len(text) - PAGE_CHARS} more chars. Call result_more with "
              f'{{"result_id": "{rid}", "offset": {PAGE_CHARS}}} to read the next part.]')


def result_more(userid: str, result_id: str, offset: int = 0) -> dict:
    """Read the next page of a long earlier tool result (see the '[truncated ...]' note)."""
    text = (_RESULT_STORE.get(userid or "anonymous") or {}).get(result_id)
    if text is None:
        return {"success": False, "error": "Unknown or expired result_id — re-run the original tool."}
    offset = max(0, int(offset or 0))
    chunk = text[offset:offset + PAGE_CHARS]
    nxt = offset + PAGE_CHARS
    out = {"success": True, "offset": offset, "text": chunk, "total_chars": len(text)}
    if nxt < len(text):
        out["next_offset"] = nxt
        out["note"] = f'More remains: call result_more again with offset {nxt}.'
    return out


# Registered here, after the real definition above (registering it inside the
# TOOL_REGISTRY literal earlier in the file raised NameError at import time).
TOOL_REGISTRY["result_more"] = result_more


def compact_tool_history(messages: list, keep_full: int = 3, old_chars: int = 500) -> None:
    """
    Keeps long agent loops from drowning the context window: only the newest
    `keep_full` tool results stay in full; older ones shrink to a short head +
    a note. Edits `messages` in place. The model can re-run a tool or use
    result_more if it needs an older result again.
    """
    idxs = [i for i, m in enumerate(messages)
            if m.get("role") == "user" and isinstance(m.get("content"), str)
            and m["content"].startswith("Tool result from ")]
    for i in idxs[:-keep_full] if keep_full else idxs:
        c = messages[i]["content"]
        if len(c) > old_chars + 200 and "[older tool result trimmed" not in c:
            head = c.split("\n\nContinue and give the user", 1)[0][:old_chars]
            messages[i]["content"] = head + "\n[older tool result trimmed to save space — re-run the tool if you need it again]"


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

    if tool_name in WEB_TOOLS and not is_web_search_enabled(session_context.get("userid")):
        return False, (
            "Web search is turned OFF in this user's account settings — "
            f"'{tool_name}' was blocked before it ran, and no internet "
            "tool will run this turn. Tell the user plainly that web "
            "search is currently disabled on their account, and that they "
            "can turn it back on from the toggle in the chat input bar "
            "(the search icon/chip next to the text field) — then answer "
            "the rest of their request using only what you already know, "
            "if possible. Do not request search_web or any other internet "
            "tool again this turn."
        )

    # User approval ("OOOR wants to fetch ..." / "OOOR wants to run this command"): decided in code by
    # gpt2_approvals, fails closed (deny / timeout / cancel = not run). The refusal text goes back to
    # the model instead of the tool result so it can tell the user and carry on.
    denied = gpt2_approvals.check(tool_name, ai_args, session_context)
    if denied:
        return False, denied

    # generate_image is NOT gated: free users generate unlimited images via
    # Pollinations, premium users get the paid providers (see
    # is_user_premium / generate_image in gpt2_functions.py).

    try:
        sig = inspect.signature(fn)
        # BUG FIX: session values MUST be applied last. The old order did
        # final_args.update(ai_args) AFTER injecting session values, which
        # meant a guessed/wrong userid (or prompt/history/image_urls) from
        # the AI's own JSON silently overwrote the real, securely-injected
        # session value — completely defeating the purpose of
        # SESSION_INJECTED_PARAMS. Real symptom this caused: get_user_profile
        # kept getting called with the model's own literal "current_user"
        # guess instead of the actual logged-in user's id, every time the
        # model included a userid key in its own tool-call JSON at all.
        final_args = dict(ai_args)  # AI-supplied args fill in first...
        for pname in sig.parameters:
            if pname in SESSION_INJECTED_PARAMS and pname in session_context:
                final_args[pname] = session_context[pname]  # ...session values ALWAYS win for these

        if inspect.isgeneratorfunction(fn):
            last_event = None
            for event in fn(**final_args):
                last_event = event
            if last_event is None:
                return False, "Tool produced no output."
            return True, _page_result(session_context.get("userid"), tool_name, json.dumps(last_event, default=str))

        result = fn(**final_args)
        # search_web returns (text, sources). The sources sit at the END of the JSON, and _page_result cuts long
        # results at PAGE_CHARS — which chops the sources off, so json.loads failed later and the Sources chip
        # silently never appeared. Keep the real list out of band, before any truncation.
        if tool_name == "search_web" and isinstance(result, (list, tuple)) and len(result) == 2 and isinstance(result[1], list):
            session_context["_last_web_sources"] = result[1]
        # fetch_github_file and search_images both get a much higher cap
        # than other tools. For fetch_github_file, a source file cut off
        # mid-import/mid-syntax just confuses the model. For search_images,
        # this is even more critical: gpt2_test.py's gallery-population
        # code does json.loads() on this exact string to build the real
        # image gallery — a [:4000] cut lands mid-JSON-string on anything
        # over ~10 candidates, throws inside that parse, gets silently
        # swallowed by its try/except, and the gallery just never renders
        # even though the search itself succeeded. This was a live bug.
        if tool_name in ("fetch_github_file", "search_images", "workspace_read", "workspace_list", "workspace_grep"):
            return True, json.dumps(result, default=str)[:20000]
        return True, _page_result(session_context.get("userid"), tool_name, json.dumps(result, default=str))

    except TypeError as e:
        # Wrong/missing args — surfaced plainly so the AI can retry with a
        # corrected <<TOOL_CALL>> instead of silently failing.
        return False, f"Called {tool_name} with invalid arguments: {e}"
    except Exception as e:
        return False, f"{tool_name} failed: {e}"
