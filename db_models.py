from sqlalchemy import Column, Integer, String, Text, DateTime
from datetime import datetime

from database import Base


class ChatHistory(Base):
    __tablename__ = "utme26_chat_history"

    id = Column(Integer, primary_key=True, index=True)

    user_id = Column(String, index=True)

    role = Column(String)

    message = Column(Text)

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )
