import os, sys, types, unittest
from datetime import datetime, timedelta, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Fake Firestore layer (no network): the app's old chat + the backfill flag.
fs = types.ModuleType("firestore_repository")
fs.APP = []
fs.FLAG = {"done": False}
fs.save_message_to_firestore = lambda *a, **k: None
fs.get_recent_messages_from_firestore = lambda *a, **k: []
fs.backfill_done = lambda uid: fs.FLAG["done"]
fs.mark_backfill_done = lambda uid: fs.FLAG.update(done=True)
def _before(uid, before, limit=2000):
    return [m for m in fs.APP if m["timestamp"] < before]
fs.get_app_chat_before = _before
sys.modules["firestore_repository"] = fs

try:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from db_models import Base, ChatHistory
    import memory_service as ms
    HAVE_DB = True
except ImportError:          # sqlalchemy is not installed in this environment
    HAVE_DB = False
import turn_memory as tm


def utc(dt):
    return dt.replace(tzinfo=timezone.utc)


@unittest.skipUnless(HAVE_DB, "sqlalchemy not installed")
class Backfill(unittest.TestCase):
    def setUp(self):
        eng = create_engine("sqlite://")
        Base.metadata.create_all(eng)
        self.db = sessionmaker(bind=eng)()
        ms._BACKFILL_CHECKED.clear()
        fs.FLAG["done"] = False
        now = datetime.utcnow()
        # 4 app messages from long before the backend saved anything
        fs.APP = [
            {"sender": "user", "text": "my very first question about biology", "status": "done", "timestamp": utc(now - timedelta(days=30)), "imageUrls": [], "fileNames": [], "builtFileNames": []},
            {"sender": "ai", "text": "biology answer", "status": "done", "timestamp": utc(now - timedelta(days=30, minutes=-1)), "imageUrls": [], "fileNames": [], "builtFileNames": ["notes.pdf"]},
            {"sender": "user", "text": "broken one", "status": "error", "timestamp": utc(now - timedelta(days=29)), "imageUrls": [], "fileNames": [], "builtFileNames": []},
            {"sender": "user", "text": "second question", "status": "done", "timestamp": utc(now - timedelta(days=28)), "imageUrls": ["https://x/a.jpg"], "fileNames": [], "builtFileNames": []},
        ]
        # backend DB already has this week's turn
        self.db.add(ChatHistory(user_id="u", role="user", message="new question", created_at=now - timedelta(hours=1)))
        self.db.add(ChatHistory(user_id="u", role="assistant", message="new answer", created_at=now - timedelta(hours=1)))
        self.db.commit()

    def test_backfill_orders_old_first_and_skips_errors(self):
        h = ms.build_memory(self.db, "u", "next")
        texts = [m["content"] for m in h]
        self.assertIn("my very first question about biology", texts[0])
        self.assertTrue(any("notes.pdf" in t for t in texts))
        self.assertFalse(any("broken one" in t for t in texts))
        self.assertTrue(any("a.jpg" in t for t in texts))
        self.assertEqual(texts[-2], "new question")
        self.assertEqual(texts[-1], "new answer")
        self.assertTrue(fs.FLAG["done"])

    def test_runs_only_once(self):
        ms.build_memory(self.db, "u", "")
        n = self.db.query(ChatHistory).count()
        ms._BACKFILL_CHECKED.clear()
        fs.FLAG["done"] = True
        ms.build_memory(self.db, "u", "")
        self.assertEqual(self.db.query(ChatHistory).count(), n)

    def test_first_messages_survive_in_a_long_chat(self):
        now = datetime.utcnow()
        for i in range(80):
            self.db.add(ChatHistory(user_id="u", role="user", message=f"filler {i} " + "x" * 1500, created_at=now - timedelta(minutes=100 - i)))
            self.db.add(ChatHistory(user_id="u", role="assistant", message="ok " + "y" * 1500, created_at=now - timedelta(minutes=100 - i, seconds=-1)))
        self.db.commit()
        old = tm.HISTORY_BUDGET_CHARS
        try:
            h = ms.build_memory(self.db, "u", "")
        finally:
            tm.HISTORY_BUDGET_CHARS = old
        self.assertIn("my very first question about biology", h[0]["content"])   # named in the head note
        self.assertIn("CONVERSATION SO FAR", h[-1]["content"])                  # whole-chat summary at the end
        self.assertIn("my very first question about biology", h[-1]["content"])

    def test_no_user_is_safe(self):
        self.assertEqual(ms.build_memory(self.db, "", ""), [])


if __name__ == "__main__":
    unittest.main()
