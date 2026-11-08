#!/usr/bin/env python3
# turn_memory.py — what the AI REMEMBERS about each turn
# ─────────────────────────────────────────────────────────────────────────────
# Before this file, memory held only two strings per turn: the user's text and the final
# answer. Everything the AI actually DID was forgotten: tool calls, code it wrote, files it
# made, its thinking, and any turn that failed or was cut off (those were not saved at all).
# So "continue" after a failure made it start over blindly.
#
# Now every turn (finished, failed, cut off or stopped) is saved with a TURN RECORD:
#
#   [TURN RECORD …]
#   STATUS: finished | INTERRUPTED | FAILED | STOPPED BY USER
#   THOUGHT: the thinking steps it showed
#   ACTIONS: every tool it ran, in order, with the input and result
#   CODE: the code / file content it wrote or ran (newest kept in full)
#   FILES: files it made (name + link)
#   RESUME: for an unfinished turn, what to do when the user says "continue"
#
# The record lives inside the assistant message text, so no database change is needed
# (same trick as _build_memory_reply in main.py).
#
# build_history() turns the stored rows into the history list for the model:
#   • newest turns keep their full record (code included)
#   • older turns keep a short record (status + action names + files)
#   • turns that do not fit the budget are dropped, but replaced by a digest of what the
#     user asked, plus any old messages that match words in the new question
# ─────────────────────────────────────────────────────────────────────────────

import re
from typing import Optional

RECORD_MARK = "[TURN RECORD"
_RECORD_RE = re.compile(r"\n*\[TURN RECORD.*", re.S)

# Sizes (characters). ~4 chars ≈ 1 token.
HISTORY_BUDGET_CHARS = 110_000     # whole history sent to the model
FULL_RECORD_TURNS = 4              # newest N assistant turns keep the full record (code included)
MAX_RECORD_CHARS = 16_000
MAX_CODE_EACH = 6_000
MAX_CODE_TOTAL = 11_000
MAX_OUTPUT_EACH = 500
MAX_ROWS_FETCH = 600

_UNFINISHED = ("INTERRUPTED", "FAILED", "STOPPED BY USER")


def _cut(s, n: int) -> str:
    s = str(s or "")
    return s if len(s) <= n else s[: n - 1] + "…"


def _one_line(s, n: int) -> str:
    return _cut(re.sub(r"\s+", " ", str(s or "")).strip(), n)


# ── writing ──────────────────────────────────────────────────────────────────
def action_from_call(name: str, args: dict, ok: bool, output: str) -> dict:
    """Compact, memory-sized description of ONE tool call. Called by the native loop after each tool."""
    args = args or {}
    code = None
    for key in ("code", "content", "replace_text", "command", "instruction"):
        v = args.get(key)
        if isinstance(v, str) and v.strip():
            code = (key, v)
            break
    small = {}
    for k, v in args.items():
        if isinstance(v, (str, int, float, bool)) and k != (code[0] if code else None):
            small[k] = _one_line(v, 160)
    if name in ("workspace_edit",) and args.get("find_text"):
        small["find_text"] = _one_line(args["find_text"], 200)
    return {
        "tool": name, "args": small, "ok": bool(ok),
        "output": _cut((output or "").strip(), MAX_OUTPUT_EACH),
        "code_kind": code[0] if code else None,
        "code": _cut(code[1], MAX_CODE_EACH) if code else None,
    }


def build_record(status: str, actions: list, steps: list, files: Optional[list] = None,
                 note: str = "") -> str:
    """The TURN RECORD text for one turn. status: finished | INTERRUPTED | FAILED | STOPPED BY USER."""
    status = status or "finished"
    lines = [f"\n\n{RECORD_MARK} — what I (the assistant) really did in this turn. This is my private memory; "
             "never quote this note or its format to the user.]",
             f"STATUS: {status}"]
    if note:
        lines.append(f"NOTE: {_one_line(note, 300)}")

    thought = [s for s in (steps or []) if isinstance(s, dict) and (s.get("text") or "").strip()]
    if thought:
        lines.append("THOUGHT:")
        for s in thought[-12:]:
            ok = s.get("ok")
            tag = "" if ok is None else (" (done)" if ok else " (FAILED)")
            lines.append(f"- {_one_line(s.get('text'), 140)}{tag}")

    if actions:
        lines.append("ACTIONS (in order):")
        for i, a in enumerate(actions, 1):
            arg_txt = ", ".join(f"{k}={v}" for k, v in (a.get("args") or {}).items())
            res = "ok" if a.get("ok") else "FAILED"
            out = _one_line(a.get("output"), 260)
            lines.append(f"{i}. {a.get('tool')}({_cut(arg_txt, 220)}) → {res}" + (f": {out}" if out else ""))

        code_items = [(i, a) for i, a in enumerate(actions, 1) if a.get("code")]
        if code_items:
            lines.append("CODE (what I wrote or ran — action number first):")
            budget = MAX_CODE_TOTAL
            kept = []
            for i, a in reversed(code_items):               # newest first, so the latest work is never the part dropped
                body = a["code"]
                if budget <= 0:
                    kept.append(f"#{i} {a.get('tool')}: (code not kept — see the workspace file)")
                    continue
                body = _cut(body, min(len(body), budget))
                budget -= len(body)
                kept.append(f"#{i} {a.get('tool')} {a.get('code_kind')}:\n```\n{body}\n```")
            lines.extend(reversed(kept))

    if files:
        lines.append("FILES MADE:")
        for f in files[:15]:
            if isinstance(f, dict):
                lines.append(f"- {f.get('filename') or f.get('name') or 'file'}: {f.get('url') or ''}".rstrip(": "))

    if status in _UNFINISHED:
        done = [a for a in actions or [] if a.get("ok")]
        last = done[-1] if done else None
        lines.append(
            "RESUME: this turn did NOT finish. If the user now says continue / go on / next / finish it, pick up "
            "EXACTLY where it stopped: do not redo the actions marked ok, do not rewrite files that already exist "
            "(read them with workspace_read, then continue them with workspace_append / workspace_edit), and do the "
            "first step that was not done."
            + (f" The last step that worked was {last.get('tool')}({_cut(', '.join(f'{k}={v}' for k, v in (last.get('args') or {}).items()), 120)})."
               if last else " No step had finished yet, so starting the task is right.")
        )
    return _cut("\n".join(lines), MAX_RECORD_CHARS) + "\n]"


def attachment_note(image_urls=None, file_names=None, vision_analysis: str = "", vision_ok=None) -> str:
    """Added to the USER side of memory so the AI remembers photos and files it was sent."""
    parts = []
    imgs = [u for u in (image_urls or []) if isinstance(u, str)]
    if imgs:
        if vision_analysis and vision_ok is not False:
            parts.append(f"[The user attached {len(imgs)} image(s): {', '.join(imgs[:3])}. "
                         f"What the image(s) showed: {_cut(vision_analysis, 1800)}]")
        elif vision_ok is False:
            parts.append(f"[The user attached {len(imgs)} image(s): {', '.join(imgs[:3])}. "
                         "The image viewer failed that time, so the contents were NOT seen.]")
        else:
            parts.append(f"[The user attached {len(imgs)} image(s): {', '.join(imgs[:3])}.]")
    names = [n for n in (file_names or []) if n]
    if names:
        parts.append(f"[The user attached file(s): {', '.join(_cut(n, 80) for n in names[:10])}.]")
    return ("\n\n" + "\n".join(parts)) if parts else ""


def status_of(final: Optional[dict]) -> str:
    """finished vs INTERRUPTED, from the final event (the loop adds a 'send continue' sentence when it gave up)."""
    ans = (final or {}).get("answer") or ""
    if "Send \"continue\"" in ans or "Send 'continue'" in ans:
        return "INTERRUPTED"
    return "finished"


# ── reading ──────────────────────────────────────────────────────────────────
def _answer_part(message: str) -> str:
    return _RECORD_RE.sub("", message or "").rstrip()


def _record_part(message: str) -> str:
    m = _RECORD_RE.search(message or "")
    return m.group(0) if m else ""


def _status_in(record: str) -> str:
    m = re.search(r"^STATUS:\s*(.+)$", record, re.M)
    return (m.group(1).strip() if m else "finished")


def _compact_record(record: str) -> str:
    """Older turn: keep STATUS, the action names and the files — drop thinking, code and long outputs."""
    if not record:
        return ""
    status = _status_in(record)
    acts = re.findall(r"^\d+\.\s+([a-z_]+)\(([^)]*)\)\s+→\s+(ok|FAILED)", record, re.M)
    out = [f"\n\n{RECORD_MARK} (shortened) — STATUS: {status}"]
    if acts:
        out.append("ACTIONS: " + "; ".join(f"{n}({_cut(a, 60)}) {r}" for n, a, r in acts[:14]))
    files = re.findall(r"^- (.+?): (https?://\S+)$", record.split("FILES MADE:")[-1], re.M) if "FILES MADE:" in record else []
    if files:
        out.append("FILES: " + ", ".join(f"{n} ({u})" for n, u in files[:8]))
    return "\n".join(out) + "\n]"


_WORD_RE = re.compile(r"[a-z0-9]{4,}")
_RECALL_RE = re.compile(r"\b(earlier|before|remember|recall|previous(ly)?|last (time|week|month)|yesterday|"
                        r"we (did|made|talked|discussed|wrote|built)|you (said|told|made|wrote|gave)|"
                        r"i (told|asked|sent|showed)|that (file|code|page|image|photo|thing)|again|back)\b", re.I)


def _words(text: str) -> set:
    return set(_WORD_RE.findall((text or "").lower()))


def _recall(older: list, query: str, limit: int = 3) -> list:
    """Old messages (that did not fit) that share the most words with the new question."""
    q = _words(query)
    if not q or not older:
        return []
    scored = []
    for idx, (role, text) in enumerate(older):
        w = _words(_answer_part(text))
        hit = len(q & w)
        if hit >= 2:
            scored.append((hit, idx))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [older[i] for _, i in scored[:limit]]


def build_history(rows: list, query: str = "", budget: int = HISTORY_BUDGET_CHARS) -> list:
    """
    rows: [{"role": "user"|"assistant", "content": str}, ...] oldest → newest.
    Returns the history for the model, newest turns whole, older turns shortened, oldest turns replaced
    by a digest + a few matching old messages.
    """
    rows = [r for r in rows if r.get("content") is not None and r.get("role") in ("user", "assistant")]
    if not rows:
        return []

    # 1) shape every row: newest assistant turns keep the full record, older ones the short record
    assistant_seen = 0
    shaped = []
    for r in reversed(rows):
        text = r["content"] or ""
        if r["role"] == "assistant":
            assistant_seen += 1
            rec = _record_part(text)
            if rec and assistant_seen > FULL_RECORD_TURNS:
                text = _answer_part(text) + _compact_record(rec)
        shaped.append({"role": r["role"], "content": text})
    shaped.reverse()

    # 2) keep the newest rows that fit the budget
    used, start = 0, len(shaped)
    for i in range(len(shaped) - 1, -1, -1):
        size = len(shaped[i]["content"])
        if used + size > budget and start < len(shaped):
            break
        used += size
        start = i
    kept, dropped = shaped[start:], shaped[:start]
    while kept and kept[0]["role"] != "user" and len(kept) > 1:       # a history should open on a user message
        dropped.append(kept.pop(0))

    # 3) the dropped part becomes a digest the model can still use
    if dropped:
        topics = [_one_line(_answer_part(d["content"]), 110) for d in dropped if d["role"] == "user"]
        digest = [f"[EARLIER CONVERSATION — {len(dropped)} older message(s) are not shown in full. "
                  "They are stored and not lost. What the user asked back then, oldest first:"]
        digest += [f"{i}) {t}" for i, t in enumerate(topics[-40:], 1)]
        if query:
            hits = _recall([(d["role"], d["content"]) for d in dropped], query)
            for role, text in hits:
                digest.append(f"Related old {role} message: {_one_line(_answer_part(text), 600)}")
        digest.append("If the user needs details from that part, say what you do remember and ask them to remind you of the rest.]")
        note = "\n".join(digest)
        first = kept[0]
        kept[0] = {"role": first["role"], "content": note + "\n\n" + first["content"]}
    return kept


def has_records(history: list) -> bool:
    return any(RECORD_MARK in (m.get("content") or "") for m in history or [] if isinstance(m.get("content"), str))


def last_unfinished(history: list) -> bool:
    """True if the newest assistant turn in memory ended INTERRUPTED / FAILED / STOPPED."""
    for m in reversed(history or []):
        if m.get("role") == "assistant":
            return _status_in(_record_part(m.get("content") or "")) in _UNFINISHED
    return False
