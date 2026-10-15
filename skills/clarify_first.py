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

PRE_MULTI = (
    "CLARIFY FIRST: the user's request is a short, vague build request. Do NOT build "
    "anything yet and do NOT write files or code. Call ask_user ONCE with "
    "questions=[...] — 2 or 3 short questions (main purpose, must-have features, "
    "style/colours), each with 2-4 options. Build only after they answer."
)
PRE_SINGLE = (
    "CLARIFY FIRST: the user's request is a short, vague build request. Do NOT build "
    "anything yet and do NOT write files or code. The app's sheet shows ONE question at a "
    "time, so call ask_user with the single most important question (usually the main "
    "purpose), 2-4 options. After they answer, if one more essential detail is still "
    "missing (features or style) ask that ONE next question; never more than 3 questions "
    "in total, then build."
)
PRE = PRE_SINGLE  # kept for backwards compatibility


def pre(ctx):
    text = (ctx.get("prompt") or "").strip()
    if not text or len(text.split()) > 12:
        return ""
    if _BUILD.search(text) and not _HAS_DETAILS.search(text):
        import os  # same switch as gpt2_tools.ask_user_multi_enabled(); read directly to avoid importing the heavy tools module
        multi = os.environ.get("ASK_USER_MULTI", "0").strip().lower() in ("1", "true", "yes", "on")
        return PRE_MULTI if multi else PRE_SINGLE
    return ""
