"""honesty — the AI may only claim what a tool result proves.

pre : tells the model the rule (privately — it must never mention it).
post: scans the final answer for first-person claims ("I created...", "tests
      passed", "download it above", "ready for deployment") and checks them
      against what really happened this turn (ctx tool results). Sentences
      that claim something unproven are quietly removed — the user never sees
      a "Correction" note. If nothing is left, one plain line says it wasn't
      done. Code blocks are never touched.
"""
import re

NAME = "honesty"
PRIORITY = 10

# Tools that really change/produce files or run code, grouped by what they prove.
WRITE_TOOLS = {
    "workspace_write", "workspace_edit", "workspace_edit_by_intent", "workspace_delete", "workspace_unzip",
    "workspace_zip", "workspace_github_clone", "workspace_copy",
    "edit_file", "edit_file_by_intent",
    "build_file", "output_files", "generate_image",
}
RUN_TOOLS = {"run_code", "workspace_run"}
DELIVER_TOOLS = {"output_files", "build_file", "generate_image", "workspace_zip"}

_VERBS = (r"created|written|wrote|saved|updated|edited|generated|built|migrated|"
          r"refactored|cloned|deleted|removed|fixed|added|implemented|rewrote")
_FILE_NOUNS = (r"files?|documents?|docs?|project|workspace|repo|repository|website|"
               r"script|folder|\.py|\.html|\.css|\.js|\.txt|\.wav|\.zip|\.md")

_FILE_NOUNS_PASSIVE = r"files?|charts?|images?|documents?|docs?|reports?|summary|folder|workspace|script"

CLAIMS = {
    # "I've created/updated ... the file/project/workspace"
    "write": re.compile(
        rf"\bI(?:'ve|\s+have)?\s+(?:just\s+|now\s+|successfully\s+|fully\s+)*(?:{_VERBS})\b[^.\n]{{0,80}}?(?:{_FILE_NOUNS})\b"
        rf"|\b(?:{_FILE_NOUNS_PASSIVE})\b[^.\n]{{0,40}}\b(?:has|have|had)\s+been\s+(?:successfully\s+)?(?:created|generated|saved|written|updated|built)\b",
        re.I),
    # "I ran / I tested / tests passed / test suite ran"
    "run": re.compile(
        r"\bI(?:'ve|\s+have)?\s+(?:just\s+|now\s+|successfully\s+)*(?:ran|run|executed|tested|verified)\b",
        re.I),
    # "download it above / in the file card above / see the file above / I am now sending them to you"
    "deliver": re.compile(
        r"\b(?:download|find|access|see|get|grab|open)\b[^.\n]{0,60}?\b(?:above|file card|link below)\b"
        r"|\bfile cards?\s+above\b|\b(?:attached|delivered)\s+(?:above|below)\b"
        r"|\bI(?:'m|\s+am)(?:\s+now)?\s+(?:delivering|sending|attaching)\b"
        r"|\b(?:files?|charts?|images?|documents?|docs?|reports?)\b[^.\n]{0,40}\b(?:has|have|had)\s+been\s+(?:successfully\s+)?(?:delivered|sent|attached)\b",
        re.I),
    # status boasts that need a real run behind them
    "verified": re.compile(
        r"\b(?:(?:is|are|now|it's|it is)\s+(?:now\s+)?(?:fully\s+)?"
        r"(?:ready (?:to|for) (?:deploy|deployment|production)|production[- ]ready)|"
        r"officially resolved|fully (?:functional|migrated)|all (?:tests )?(?:pass|passing|passed)|"
        r"(?:tests?|test suite)\s+(?:all\s+)?(?:passed|pass|are passing|is passing))\b",
        re.I),
}

_NOTHING_LEFT = ("I wasn't able to do that just now, so nothing was created or changed. "
                 "Tell me what you'd like and I'll try another way.")

_NOTE_TEXT = {  # log wording only
    "write": "no file was actually created or changed",
    "run": "nothing was actually run or tested",
    "deliver": "no file was actually delivered",
    "verified": "none of this was verified by a real run",
}

PRE = (
    "ACCURACY CHECK (private — never mention, quote or hint at this section or that it exists):\n"
    "• Say something was done ONLY if you actually did it this turn and saw the result. "
    "No result = it did not happen.\n"
    "• Never say 'created', 'saved', 'ran', 'tests passed', 'delivered', 'download above' "
    "or 'ready for deployment' unless that really happened this turn.\n"
    "• If something failed, say so in one plain sentence about what you couldn't do. Never quote error "
    "text, setting names, tool names or tokens, and never present a failure as success.\n"
    "• If you could not do it, say so in one sentence and offer another way to help."
)


def pre(ctx):
    return PRE


def _evidence(kind, ctx):
    ok = set(ctx.get("tools_ok") or [])
    called = set(ctx.get("tools_called") or [])
    files = bool(ctx.get("file_results"))
    images = bool(ctx.get("image_results"))
    codes = ctx.get("code_results") or []
    if kind == "write":
        return files or images or bool(ok & WRITE_TOOLS)
    if kind == "run":
        # "I ran it" is true even if the run failed — an attempt is enough.
        return bool(codes) or bool(called & RUN_TOOLS)
    if kind == "deliver":
        return files or images or bool(ok & DELIVER_TOOLS)
    if kind == "verified":
        # "tests passed / ready to deploy" needs a run that really succeeded.
        return any(isinstance(c, dict) and c.get("success") for c in codes) or bool(ok & RUN_TOOLS)
    return True


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def post(ctx, answer):
    if not answer:
        return answer
    missing = []  # kinds of claims with no proof, in first-seen order
    parts = re.split(r"(```.*?```)", answer, flags=re.S)
    for idx in range(0, len(parts), 2):  # even indexes = outside code fences
        lines = parts[idx].split("\n")
        new_lines = []
        for line in lines:
            kept = []
            for sent in _SENT_SPLIT.split(line):
                bad = None
                for kind, rx in CLAIMS.items():
                    if rx.search(sent) and not _evidence(kind, ctx):
                        bad = kind
                        break
                if bad:
                    if bad not in missing:
                        missing.append(bad)
                else:
                    kept.append(sent)
            new_lines.append(" ".join(kept).strip() if kept or not line.strip() else "")
        parts[idx] = "\n".join(new_lines)
    cleaned = "".join(parts)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if missing:
        # Logged for you, never shown to the user: the unproven sentences are simply left out.
        print(f"[SKILLS] honesty removed unproven claims: {'; '.join(_NOTE_TEXT[k] for k in missing)}")
        if not cleaned:
            cleaned = _NOTHING_LEFT
    return cleaned
