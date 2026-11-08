from datetime import datetime

from sqlalchemy import func
from sqlalchemy.orm import Session

from db_models import ChatHistory


def save_message(
    db: Session,
    user_id: str,
    role: str,
    message: str,
):
    # No user_id -> nothing to key the memory on. Stay stateless for that
    # request instead of erroring, same as before this fix.
    #
    # THIS is almost certainly why your local uvicorn test showed an empty
    # table: the table gets created fine on startup (Base.metadata.create_all
    # always runs), but if the request body never included "userid", every
    # save_message() call silently returns here and nothing is ever
    # inserted. Loud print so this is impossible to miss next time.
    if not user_id:
        print("[MEMORY] save_message skipped — no userid on request. "
              "Send a non-empty \"userid\" field in /ai-query to test memory.")
        return

    chat = ChatHistory(
        user_id=user_id,
        role=role,
        message=message,
    )

    db.add(chat)
    db.commit()


def get_recent_messages(
    db: Session,
    user_id: str,
    limit: int = 30,
):
    if not user_id:
        return []

    rows = (
        db.query(ChatHistory)
        .filter(ChatHistory.user_id == user_id)
        .order_by(ChatHistory.created_at.desc(), ChatHistory.id.desc())   # created_at first: backfilled old rows are inserted later but are OLDER
        .limit(limit)
        .all()
    )

    rows.reverse()

    return rows


def get_first_messages(db: Session, user_id: str, n: int = 10):
    """The very first rows of this user's history (oldest first)."""
    if not user_id:
        return []
    return (
        db.query(ChatHistory)
        .filter(ChatHistory.user_id == user_id)
        .order_by(ChatHistory.created_at.asc(), ChatHistory.id.asc())
        .limit(n)
        .all()
    )


def earliest_created_at(db: Session, user_id: str):
    if not user_id:
        return None
    return db.query(func.min(ChatHistory.created_at)).filter(ChatHistory.user_id == user_id).scalar()


def insert_backfill(db: Session, user_id: str, rows: list) -> int:
    """rows: [{"role", "message", "created_at": naive-UTC datetime}] — older messages from before the backend
    saved history. Keeps their ORIGINAL time so they sort before everything already saved."""
    if not user_id or not rows:
        return 0
    for r in rows:
        db.add(ChatHistory(user_id=user_id, role=r["role"], message=r["message"],
                           created_at=r.get("created_at") or datetime.utcnow()))
    db.commit()
    return len(rows)
