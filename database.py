import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import Session

# Own DB for this app's chat memory — same pattern as the Zindryx/Gemini
# backend, but its own instance (falls back to a local SQLite file if no
# DATABASE_URL is set, exactly like the other app does).
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./utme26_chat.db")

# Render's Postgres gives you a URL starting with "postgres://", but
# SQLAlchemy + psycopg2 require "postgresql://". Normalize it here so you
# can paste Render's URL straight in without editing it.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()


def get_db():

    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()
