"""
leak_guard.py — keeps backend internals out of what the user reads.

Three jobs:

1. Reasoning tags. Some models write their thinking inline as <think>, <thought>,
   <thinking>, <reasoning> ... blocks. strip_reasoning() / THINK_*_RE remove every
   spelling so none of it reaches the app.
2. Internal talk. scrub_internal() removes any sentence that quotes or describes our
   instructions, rules, tool names, config/env names — from live text, final text and
   thought-process steps. Fenced code is never touched.
3. Encoding. repair_mojibake() fixes text where UTF-8 was decoded as Latin-1/cp1252
   (a bullet "•" turning into "â¢"), as a safety net on top of the real fix in the
   streaming reader.

The "[ADVANCED THINKING ENABLED ...]" directive lives in the system prompt
(prompts.PREMIUM_MODE_PROMPT), never in the user turn; scrub_internal() still removes
any sentence that repeats it.
"""
import re

# ── 1. reasoning tags ────────────────────────────────────────────────────────
THINK_TAGS = ("think", "thinking", "thought", "thoughts", "reasoning", "scratchpad")
_T = "|".join(THINK_TAGS)
THINK_OPEN_RE = re.compile(rf"<(?:{_T})>", re.I)
THINK_CLOSE_RE = re.compile(rf"</(?:{_T})>", re.I)
THINK_BLOCK_RE = re.compile(rf"<(?:{_T})>.*?</(?:{_T})>", re.S | re.I)
# Literal tag strings, used to decide whether a half-streamed "<tho" could still become one.
THINK_TAG_LITERALS = tuple(f"<{n}>" for n in THINK_TAGS) + tuple(f"</{n}>" for n in THINK_TAGS)
THINK_TAG_MAX_LEN = max(len(t) for t in THINK_TAG_LITERALS)


def strip_reasoning(text: str) -> str:
    """Remove closed reasoning blocks, text before a lone closer, and an unclosed opener's tail."""
    t = THINK_BLOCK_RE.sub("", text or "")
    closers = list(THINK_CLOSE_RE.finditer(t))
    if closers:                                   # a lone closer: everything before it was reasoning
        t = t[closers[-1].end():]
    opener = THINK_OPEN_RE.search(t)
    if opener:                                    # an opener that never closed: the rest is reasoning
        t = t[:opener.start()]
    return t.strip()


# ── 2. internal talk ─────────────────────────────────────────────────────────
_BRACKET_DIRECTIVE = re.compile(r"\[\s*ADVANCED THINKING[^\]]*\]", re.I | re.S)

# Distinctive phrases from our own internal directives. A sentence/line containing one of
# these is describing the backend, not answering the user.
_MARKERS = (
    "advanced thinking enabled",
    "advanced thinking mode",
    "dont talk too much",
    "don't talk too much",
    "uneccesary talk",
    "unnecessary talk",
    "satisfy the user fully",
    "code-first approach",
    "system instruction",
    "system prompt",
    "prompt engineering directive",
    "my instructions say",
    "my instructions tell me",
    "i was instructed to",
    "i am instructed to",
    "i've been instructed to",
    "backend prompt",
    "hidden instruction",
    "honesty rule",
    "private rules",
    "private section",
    "according to the rules",
    "according to my rules",
    "according to my guidelines",
    "according to my instructions",
    "my guidelines say",
    "my rules say",
    "tool call failed",
    "the tool returned",
    "tool result",
)

# Names of our own tools — never shown to a user, who only sees plain-words chips.
_TOOL_NAMES = (
    "analyze_image", "ask_user", "create_image", "fetch_document", "fetch_github_file",
    "fetch_page_preview_image", "fetch_webpage", "generate_image", "get_study_notes",
    "get_user_notes", "get_user_profile", "output_files", "publish_page", "remove_background",
    "run_code", "save_study_note", "save_user_note", "schedule_reminder", "search_github",
    "search_images", "search_web", "result_more", "build_file", "edit_file",
    "edit_file_by_intent",
)
_INTERNAL_NAME_RE = re.compile(
    r"\b(?:workspace_[a-z_]+|" + "|".join(_TOOL_NAMES) + r")\b"       # tool / function names
    r"|\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*_(?:TOKEN|KEY|REPO|SECRET|PASSWORD)\b"  # env / config names
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _is_internal(sentence: str) -> bool:
    low = sentence.lower()
    return any(m in low for m in _MARKERS) or bool(_INTERNAL_NAME_RE.search(sentence))


def scrub_internal(text: str) -> str:
    """Drop reasoning blocks, bracketed directives and any sentence that quotes or describes
    internal instructions, tool names or config names. Fenced code is never touched (a user's
    own code may legitimately contain these words)."""
    if not text:
        return text
    if "```" in text:
        parts = text.split("```")
        # even index = prose, odd index = inside a fence (an unclosed last fence is still code)
        return "```".join(p if i % 2 else _scrub_prose(p) for i, p in enumerate(parts))
    return _scrub_prose(text)


def _scrub_prose(text: str) -> str:
    t = THINK_BLOCK_RE.sub("", text)
    t = _BRACKET_DIRECTIVE.sub("", t)
    if not _is_internal(t):
        return t
    keep = []
    for piece in re.split(r"(\n+)", t):          # keep newlines (markdown structure) intact
        if piece.strip() == "" or "\n" in piece:
            keep.append(piece)
            continue
        sentences = re.split(r"(?<=[.!?])(\s+)", piece)
        out = []
        for s in sentences:
            if s.strip() and _is_internal(s):
                continue
            out.append(s)
        keep.append("".join(out))
    return re.sub(r"[ \t]+\n", "\n", "".join(keep))


# ── 3. encoding ──────────────────────────────────────────────────────────────
# UTF-8 bytes read as Latin-1/cp1252 show up as a lead char (Â Ã â ð) followed by C1 control
# characters or Latin-1 symbols. The bullet "•" (E2 80 A2) becomes "â" + U+0080 + "¢".
_MOJIBAKE_RUN = re.compile(r"[Â-ô][\u0080-¿]+")
_CP1252_TO_BYTE = {  # chars cp1252 uses in place of C1 controls (so "â€¢" is repaired as well as "â\x80¢")
    "€": 0x80, "‚": 0x82, "ƒ": 0x83, "„": 0x84, "…": 0x85, "†": 0x86, "‡": 0x87, "ˆ": 0x88,
    "‰": 0x89, "Š": 0x8A, "‹": 0x8B, "Œ": 0x8C, "Ž": 0x8E, "‘": 0x91, "’": 0x92, "“": 0x93,
    "”": 0x94, "•": 0x95, "–": 0x96, "—": 0x97, "˜": 0x98, "™": 0x99, "š": 0x9A, "›": 0x9B,
    "œ": 0x9C, "ž": 0x9E, "Ÿ": 0x9F,
}


def _undo_latin1(s: str) -> str:
    try:
        raw = bytes((_CP1252_TO_BYTE.get(ch) or ord(ch)) for ch in s)
        return raw.decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return s


def repair_mojibake(text: str) -> str:
    """Fix 'â¢', 'â€¢', 'Ã©' style garbage. Text with no sign of it is returned untouched."""
    if not text or not re.search(r"[Â-ô]", text):
        return text
    if not _MOJIBAKE_RUN.search(text) and "â€" not in text and "Ã" not in text:
        return text
    whole = _undo_latin1(text)
    if whole != text and not _MOJIBAKE_RUN.search(whole):
        return whole
    # mixed text (some real non-Latin characters too): repair each garbled run on its own
    pat = re.compile(r"[Â-ô][\u0080-¿€‚-›ˆ˜ŠšŒœŽžŸƒ™¡-¿]+")
    return pat.sub(lambda m: _undo_latin1(m.group(0)), text)
