"""
Firestore chat memory backup.

Why this exists: Postgres on Render's free plan (or the local SQLite
fallback) is the FAST path for memory, but on a redeploy Render can spin
up a fresh disk/DB and the fast path comes back empty. Firestore is the
durable path — every turn gets written here too, and if Postgres ever
comes back empty for a user_id, we pull that user's history back out of
Firestore so the AI doesn't lose context on a redeploy.

Structure (this is the "user col -> user_id -> col -> documents" shape):

    users/{user_id}/messages/{auto_id}
        role: "user" | "assistant"
        message: str
        created_at: server timestamp

Every function here is best-effort and NEVER raises — a Firestore hiccup
should degrade to "memory just uses Postgres for this turn", not take
the whole chat endpoint down.
"""

import firebase_config  # noqa: F401  (ensures firebase_admin is initialized)
from firebase_admin import firestore

_db = None


def _client():
    global _db
    if _db is None:
        _db = firestore.client()
    return _db


def save_message_to_firestore(user_id: str, role: str, message: str):
    if not user_id:
        return

    try:
        _client().collection("users").document(user_id).collection("utme26_messages").add({
            "role": role,
            "message": message,
            "created_at": firestore.SERVER_TIMESTAMP,
        })
    except Exception as e:
        print(f"[FIRESTORE] save_message failed for user_id={user_id!r}: {e}")


def get_recent_messages_from_firestore(user_id: str, limit: int = 40) -> list:
    if not user_id:
        return []

    try:
        docs = (
            _client()
            .collection("users")
            .document(user_id)
            .collection("utme26_messages")
            .order_by("created_at", direction=firestore.Query.DESCENDING)
            .limit(limit)
            .stream()
        )

        rows = [d.to_dict() for d in docs]
        rows.reverse()  # oldest -> newest, matching Postgres ordering
        return rows

    except Exception as e:
        print(f"[FIRESTORE] get_recent_messages failed for user_id={user_id!r}: {e}")
        return []
