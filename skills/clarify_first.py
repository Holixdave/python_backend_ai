"""clarify_first — vague "build me X" requests get a question, not a build.

pre only. Fires when the user's message is a short build request with no
details (e.g. "create website", "build an app"). It tells the model to call
ask_user with tappable options instead of generating files.
"""
import re

NAME = "clarify_first"
PRIORITY = 30

_BUILD = re.compile(
    r"^\W*(?:please\s+|pls\s+|can you\s+|dude\s+|bro\s+)*"
    r"(?:create|build|make|design|develop|code|write)\s+(?:me\s+)?(?:a|an|my|the)?\s*"
    r"(?:[\w'-]+\s+){0,3}?"
    r"(?:website|web\s?site|web\s?app|app|landing page|page|site|dashboard|game|portfolio|system|platform)\b",
    re.I,
)
_HAS_DETAILS = re.compile(
    r"\b(?:with|that|which|using|including|include|features?|"
    r"in (?:react|flutter|html|python|next|vue)|for (?:my|our) \w+ (?:business|company|school|store))\b",
    re.I,
)

PRE = (
    "CLARIFY FIRST: the user's request is a short, vague build request. Do NOT build "
    "anything yet and do NOT write files or code. Call the ask_user tool now with 1-3 "
    "short multiple-choice questions (main purpose, must-have features, style/colours), "
    "2-4 options each. Build only after they answer."
)


def pre(ctx):
    text = (ctx.get("prompt") or "").strip()
    if not text or len(text.split()) > 12:
        return ""
    if _BUILD.search(text) and not _HAS_DETAILS.search(text):
        return PRE
    return ""
