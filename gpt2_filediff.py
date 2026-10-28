#!/usr/bin/env python3
# gpt2_filediff.py — "what did that script change?" for run_code and workspace_run.
# ─────────────────────────────────────────────────────────────────────────────
# Same diff format the edit tools use, so the app renders it with one component:
#     @@ line N @@        hunk header
#     - removed line      (red)
#     + added line        (green)
#       unchanged line    (context)
# A changed file is {path, status: new|edited|deleted, added, removed, diff}.
# format_changes() turns a list of them into the text block the tool card shows:
#     --- changed files ---
#     === config.json (edited)
#     @@ line 2 @@ ...
# Only small UTF-8 text files get a diff; binaries are listed by the tools as before.
# ─────────────────────────────────────────────────────────────────────────────
import difflib
from typing import Optional

MAX_TEXT_BYTES = 100_000
MAX_DIFF_LINES = 40
NEW_FILE_PREVIEW_LINES = 25
MAX_FILES_SHOWN = 6
SEPARATOR = "--- changed files ---"


def text_of_bytes(data: bytes, cap: int = MAX_TEXT_BYTES) -> Optional[str]:
    """UTF-8 text of small non-binary data, else None."""
    if len(data) > cap or b"\x00" in data[:4096]:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def read_text(path: str, cap: int = MAX_TEXT_BYTES) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            return text_of_bytes(f.read(cap + 1), cap)
    except OSError:
        return None


def file_diff(old: Optional[str], new: str, max_lines: int = MAX_DIFF_LINES):
    """(added, removed, diff_text). old=None means a brand-new file (every line is added)."""
    new_lines = new.split("\n")
    if new_lines and new_lines[-1] == "":
        new_lines.pop()
    if old is None:
        shown = ["+ " + l for l in new_lines[:NEW_FILE_PREVIEW_LINES]]
        if len(new_lines) > NEW_FILE_PREVIEW_LINES:
            shown.append(f"  … {len(new_lines) - NEW_FILE_PREVIEW_LINES} more lines")
        return len(new_lines), 0, "\n".join(shown)
    old_lines = old.split("\n")
    if old_lines and old_lines[-1] == "":
        old_lines.pop()
    sm = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    out, added, removed = [], 0, 0
    for group in sm.get_grouped_opcodes(1):
        out.append(f"@@ line {group[0][1] + 1} @@")
        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                out += ["  " + l for l in old_lines[i1:i2]]
            else:
                out += ["- " + l for l in old_lines[i1:i2]]
                out += ["+ " + l for l in new_lines[j1:j2]]
                removed += i2 - i1
                added += j2 - j1
    if len(out) > max_lines:
        extra = len(out) - max_lines
        out = out[:max_lines] + [f"  … {extra} more lines"]
    return added, removed, "\n".join(out)


def change_entry(path: str, old_text: Optional[str], new_text: str) -> dict:
    added, removed, diff = file_diff(old_text, new_text)
    return {"path": path, "status": "new" if old_text is None else "edited",
            "added": added, "removed": removed, "diff": diff}


def format_changes(changes: list, max_files: int = MAX_FILES_SHOWN) -> str:
    """The 'changed files' block appended to a tool card's output. '' when nothing changed."""
    changes = [c for c in (changes or []) if isinstance(c, dict) and c.get("path")]
    if not changes:
        return ""
    parts = [SEPARATOR]
    for c in changes[:max_files]:
        parts.append(f"=== {c['path']} ({c.get('status', 'edited')})")
        if c.get("diff"):
            parts.append(c["diff"])
    if len(changes) > max_files:
        parts.append(f"… and {len(changes) - max_files} more files")
    return "\n".join(parts)
