from datetime import datetime, timedelta, timezone

from chat_repository import (earliest_created_at, get_first_messages, get_recent_messages, insert_backfill,
                             save_message)
from firestore_repository import (backfill_done, get_app_chat_before, get_recent_messages_from_firestore,
                                  mark_backfill_done, save_message_to_firestore)
from turn_memory import attachment_note, build_history

_BACKFILL_CHECKED: set = set()   # users already checked by THIS process (the Firestore flag covers restarts)
BACKFILL_MAX_CHARS = 8000


def _backfill_old_chat(db, user_id: str):
    """
    One-time, best effort: messages from BEFORE the backend saved history live only in the app's own chat
    (users/{uid}/chat_history). Copy the ones OLDER than the first backend-saved row into the backend DB with their
    original time, so the AI remembers them too. Never raises, never duplicates (older-than cut-off + a per-user flag).
    """
    if not user_id or user_id in _BACKFILL_CHECKED:
        return
    _BACKFILL_CHECKED.add(user_id)
    try:
        if backfill_done(user_id):
            return
        first = earliest_created_at(db, user_id)                       # naive UTC
        cutoff = first if first is not None else datetime.utcnow() - timedelta(seconds=30)   # keeps out the message being sent right now
        docs = get_app_chat_before(user_id, cutoff.replace(tzinfo=timezone.utc))
        rows = []
        for m in docs:
            text = (m.get("text") or "").strip()
            if not text or m.get("status") in ("running", "error", "cancelled"):
                continue
            role = "user" if m.get("sender") == "user" else "assistant"
            ts = m.get("timestamp")
            when = ts.astimezone(timezone.utc).replace(tzinfo=None) if getattr(ts, "tzinfo", None) else (ts or cutoff)
            if role == "user":
                text = text[:BACKFILL_MAX_CHARS] + attachment_note(m.get("imageUrls"), m.get("fileNames"))
            else:
                text = text[:BACKFILL_MAX_CHARS]
                if m.get("builtFileNames"):
                    text += "\n\n[Files I made in that message: " + ", ".join(m["builtFileNames"][:8]) + "]"
            rows.append({"role": role, "message": text, "created_at": when})
        n = insert_backfill(db, user_id, rows)
        mark_backfill_done(user_id)
        print(f"[MEMORY] backfilled {n} old message(s) for user_id={user_id!r}")
    except Exception as e:
        try:
            db.rollback()
        except Exception:
            pass
        print(f"[MEMORY] backfill skipped for user_id={user_id!r}: {type(e).__name__}: {e}")

# How many turns get pulled from the DB per request. This is intentionally
# bigger than the 6-message window get_lean_history() ultimately sends to
# the model (see gpt2_functions.py) — that final trim is just about keeping
# the PROMPT lean/cheap. This number is about giving classify_intent() and
# any future summarization step a wider, real window of what actually
# happened, since it's now a cheap DB read instead of a payload the
# frontend had to carry on every request.
MEMORY_WINDOW = 600   # rows pulled from the DB; turn_memory.build_history() decides what fits the prompt


def build_memory(db, user_id: str, query: str = "") -> list:
    """
    Loads this user's persistent chat history from the backend DB and
    shapes it into the exact same [{"role": ..., "content": ...}, ...]
    list that ask_gpt2() / ask_gpt2_stream() already expect as `history`.

    This is the whole fix: nothing downstream (classify_intent,
    get_lean_history, the tool loop, build_file_with_continuation) needs to
    change at all — only WHERE this list comes from changed. Previously it
    was whatever array the frontend replayed; now it's the backend's own
    record of the conversation, keyed by user_id.
    """
    _backfill_old_chat(db, user_id)

    rows = get_recent_messages(db, user_id, limit=MEMORY_WINDOW)

    if rows:
        # The chat's very first messages are always loaded too, even when the chat is longer than the window,
        # so build_history() can always name them.
        have = {r.id for r in rows}
        firsts = [r for r in get_first_messages(db, user_id, 10) if r.id not in have]
        return build_history(
            [{"role": row.role, "content": row.message} for row in firsts + rows], query=query)

    # FALLBACK: Postgres/SQLite came back empty for this user_id. On Render
    # this happens right after a redeploy wipes the DB (or a fresh SQLite
    # file locally) — the user still exists, they just lost their fast-path
    # history. Firestore is the durable copy, so pull it from there instead
    # of handing the model an empty conversation.
    if not user_id:
        return []

    firestore_rows = get_recent_messages_from_firestore(user_id, limit=MEMORY_WINDOW)

    if firestore_rows:
        print(f"[MEMORY] Postgres empty for user_id={user_id!r}, "
              f"recovered {len(firestore_rows)} message(s) from Firestore")

    return build_history(
        [{"role": row.get("role"), "content": row.get("message")} for row in firestore_rows], query=query)


def remember_turn(db, user_id: str, user_message: str, assistant_reply: str):
    """
    Call once per completed turn (after the AI has answered) to persist
    both sides to the DB. If user_id is missing, this is a silent no-op —
    the request still works, it just isn't remembered (same as a guest/
    anonymous session on the working app).
    """
    save_message(db, user_id, "user", user_message)
    save_message(db, user_id, "assistant", assistant_reply)

    # Dual-write to Firestore too. This is the durable copy that survives
    # a Render redeploy wiping Postgres/SQLite — see firestore_repository.py.
    # Best-effort: if Firestore is down, the Postgres write above already
    # succeeded, so the turn isn't lost, just not backed up this once.
    save_message_to_firestore(user_id, "user", user_message)
    save_message_to_firestore(user_id, "assistant", assistant_reply)
