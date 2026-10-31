#!/usr/bin/env python3
# offline_board.py — leaderboard for people using the app WITHOUT an account
# ─────────────────────────────────────────────────────────────────────────────
#   POST /offline-board/submit   {guestId, nickname, studentType, score}
#   GET  /offline-board/top?limit=200&guestId=...
#
# Guests have no Firebase account. The phone sends its best score here; THIS
# server (admin key) writes it to Firestore  offlineBoard/{guestId}. Firestore
# rules for that collection stay closed to clients.
#
# Ranked by the best score ("weeklyHighScore"). Nothing is reset here.
# Honest limit: the score comes from the phone, so it can be faked. We cap it
# at the real maximum, only ever let it go UP, and rate-limit per phone and
# per IP, which stops casual abuse but not a determined one.
# ─────────────────────────────────────────────────────────────────────────────

import re
import threading
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

import firebase_config  # noqa: F401 (initialises firebase_admin)
from firebase_admin import firestore

router = APIRouter()

COLLECTION = "offlineBoard"
MAX_SCORE = 400                 # 4 subjects x 100, same as CbtSession.totalMaxScore()
MIN_SECONDS_BETWEEN = 15        # per phone
NEW_GUESTS_PER_IP_PER_DAY = 25
MAX_TOP = 200
_ID_RE = re.compile(r"^[A-Za-z0-9-]{16,64}$")
_NICK_RE = re.compile(r"[^\w .\-]", re.UNICODE)
_TYPES = {"Science", "Art", "Commercial", "Other"}

_lock = threading.Lock()
_last_submit: dict = {}         # guestId -> time
_ip_new: dict = {}              # (ip, yyyy-mm-dd) -> set(guestId)

_db_client = None


def _db():
    global _db_client
    if _db_client is None:
        _db_client = firestore.client()
    return _db_client


def clean_nick(raw: str) -> str:
    n = _NICK_RE.sub("", (raw or "")).strip()[:20]
    return n if len(n) >= 3 else ""


class SubmitBody(BaseModel):
    guestId: str
    nickname: str
    studentType: str = "Other"
    score: int


def _ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "?"


@router.post("/offline-board/submit")
def submit(body: SubmitBody, request: Request):
    if not _ID_RE.match(body.guestId):
        raise HTTPException(400, "bad id")
    nick = clean_nick(body.nickname)
    if not nick:
        raise HTTPException(400, "bad name")
    if not (0 <= body.score <= MAX_SCORE):
        raise HTTPException(400, "bad score")
    stype = body.studentType if body.studentType in _TYPES else "Other"

    now = time.time()
    day = time.strftime("%Y-%m-%d", time.gmtime(now))
    ip = _ip(request)
    with _lock:
        if now - _last_submit.get(body.guestId, 0) < MIN_SECONDS_BETWEEN:
            raise HTTPException(429, "slow down")
        seen = _ip_new.setdefault((ip, day), set())
        if body.guestId not in seen and len(seen) >= NEW_GUESTS_PER_IP_PER_DAY:
            raise HTTPException(429, "too many new phones from this network")
        seen.add(body.guestId)
        _last_submit[body.guestId] = now
        if len(_last_submit) > 50_000:
            _last_submit.clear()
        for k in [k for k in _ip_new if k[1] != day]:
            _ip_new.pop(k, None)

    ref = _db().collection(COLLECTION).document(body.guestId)
    snap = ref.get()
    old = (snap.to_dict() or {}) if snap.exists else {}
    best = int(old.get("weeklyHighScore", 0) or 0)
    update = {"nickname": nick, "studentType": stype, "exams": firestore.Increment(1)}
    if body.score > best:
        update["weeklyHighScore"] = body.score
        update["weeklyHighTimestamp"] = firestore.SERVER_TIMESTAMP
        best = body.score
    ref.set(update, merge=True)
    return {"ok": True, "best": best}


def _rank_of(score: int) -> int:
    q = _db().collection(COLLECTION).where("weeklyHighScore", ">", score)
    try:
        return int(q.count().get()[0][0].value) + 1
    except Exception:
        return sum(1 for _ in q.select([]).stream()) + 1


@router.get("/offline-board/top")
def top(limit: int = Query(200, ge=1, le=MAX_TOP), guestId: Optional[str] = None):
    docs = (
        _db().collection(COLLECTION)
        .where("weeklyHighScore", ">", 0)
        .order_by("weeklyHighScore", direction=firestore.Query.DESCENDING)
        .limit(limit)
        .stream()
    )
    rows = []
    for i, d in enumerate(docs, start=1):
        x = d.to_dict() or {}
        ts = x.get("weeklyHighTimestamp")
        rows.append({
            "rank": i,
            "id": d.id if guestId and d.id == guestId else "",
            "nickname": x.get("nickname", ""),
            "studentType": x.get("studentType", ""),
            "score": int(x.get("weeklyHighScore", 0) or 0),
            "ts": int(ts.timestamp() * 1000) if ts else None,
        })
    me = None
    if guestId and _ID_RE.match(guestId):
        s = _db().collection(COLLECTION).document(guestId).get()
        if s.exists:
            x = s.to_dict() or {}
            sc = int(x.get("weeklyHighScore", 0) or 0)
            if sc > 0:
                me = {"rank": _rank_of(sc), "score": sc, "nickname": x.get("nickname", "")}
    return {"rows": rows, "me": me, "updatedAt": int(time.time() * 1000)}
