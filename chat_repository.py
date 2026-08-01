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
        .order_by(ChatHistory.id.desc())
        .limit(limit)
        .all()
    )

    rows.reverse()

    return rows
