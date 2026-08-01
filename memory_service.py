from chat_repository import get_recent_messages, save_message
from firestore_repository import save_message_to_firestore, get_recent_messages_from_firestore

# How many turns get pulled from the DB per request. This is intentionally
# bigger than the 6-message window get_lean_history() ultimately sends to
# the model (see gpt2_functions.py) — that final trim is just about keeping
# the PROMPT lean/cheap. This number is about giving classify_intent() and
# any future summarization step a wider, real window of what actually
# happened, since it's now a cheap DB read instead of a payload the
# frontend had to carry on every request.
MEMORY_WINDOW = 40


def build_memory(db, user_id: str) -> list:
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
    rows = get_recent_messages(db, user_id, limit=MEMORY_WINDOW)

    if rows:
        return [
            {"role": row.role, "content": row.message}
            for row in rows
        ]

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

    return [
        {"role": row.get("role"), "content": row.get("message")}
        for row in firestore_rows
    ]


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
