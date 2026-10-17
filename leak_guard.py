"""
leak_guard.py — keeps backend instructions out of what the user reads.

The "[ADVANCED THINKING ENABLED ...]" directive used to be glued onto the user's
message, so the model treated it as something the USER said and sometimes echoed,
summarised or "explained" it. Two defences now:

1. The directive lives in the system prompt (prompts.PREMIUM_MODE_PROMPT), never in the user turn.
2. scrub_internal() removes any sentence that still repeats it, from live text, final text
   and thought-process steps.
"""
import re

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
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def scrub_internal(text: str) -> str:
    """Drop bracketed directives and any sentence that quotes/describes internal instructions.
    Fenced code is never touched (a user's own code may legitimately contain these words)."""
    if not text:
        return text
    if "```" in text:
        parts = text.split("```")
        # even index = prose, odd index = inside a fence (an unclosed last fence is still code)
        return "```".join(p if i % 2 else _scrub_prose(p) for i, p in enumerate(parts))
    return _scrub_prose(text)


def _scrub_prose(text: str) -> str:
    t = _BRACKET_DIRECTIVE.sub("", text)
    low = t.lower()
    if not any(m in low for m in _MARKERS):
        return t
    keep = []
    for piece in re.split(r"(\n+)", t):          # keep newlines (markdown structure) intact
        if piece.strip() == "" or "\n" in piece:
            keep.append(piece)
            continue
        sentences = re.split(r"(?<=[.!?])(\s+)", piece)
        out = []
        for s in sentences:
            if s.strip() and any(m in s.lower() for m in _MARKERS):
                continue
            out.append(s)
        keep.append("".join(out))
    return re.sub(r"[ \t]+\n", "\n", "".join(keep))
