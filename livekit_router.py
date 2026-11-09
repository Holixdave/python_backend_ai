"""LiveKit (self-hosted media server) for big group calls.

Two endpoints:
  GET  /livekit/config  -> {"enabled": bool}   the app asks this before starting a group call.
  POST /livekit/token   -> {"url", "token"}    a short-lived ticket for ONE call's room, members only.

Env vars (set on Render):
  LIVEKIT_URL         wss://lk.yourdomain.com     (the public address of your LiveKit server)
  LIVEKIT_API_KEY     the key from livekit.yaml
  LIVEKIT_API_SECRET  the secret from livekit.yaml

Until all three are set, /livekit/config says enabled=false and the app keeps using the phone-to-phone mesh.
"""
import os
import time

import firebase_config  # noqa: F401 (initialises firebase_admin)
import jwt  # PyJWT
from fastapi import APIRouter, Header, HTTPException
from firebase_admin import auth as fb_auth, firestore
from pydantic import BaseModel

router = APIRouter()

MAX_PEOPLE = 40               # one call, voice or video
TICKET_SECONDS = 60 * 60 * 2  # a ticket is only needed to JOIN; 2 h covers the 1 h 30 call limit


def _cfg():
    return (
        os.getenv("LIVEKIT_URL", "").strip(),
        os.getenv("LIVEKIT_API_KEY", "").strip(),
        os.getenv("LIVEKIT_API_SECRET", "").strip(),
    )


def _enabled() -> bool:
    url, key, secret = _cfg()
    return bool(url and key and secret)


def _uid(authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Sign in first")
    try:
        return fb_auth.verify_id_token(authorization.split(" ", 1)[1].strip())["uid"]
    except Exception:
        raise HTTPException(status_code=401, detail="Bad sign-in")


@router.get("/livekit/config")
def livekit_config():
    return {"enabled": _enabled(), "maxPeople": MAX_PEOPLE}


class TokenRequest(BaseModel):
    callId: str


@router.post("/livekit/token")
def livekit_token(req: TokenRequest, authorization: str | None = Header(default=None)):
    if not _enabled():
        raise HTTPException(status_code=503, detail="Media server is not set up")
    uid = _uid(authorization)
    db = firestore.client()

    call = db.collection("calls").document(req.callId).get()
    if not call.exists:
        raise HTTPException(status_code=404, detail="No such call")
    data = call.to_dict() or {}
    if not data.get("isGroup"):
        raise HTTPException(status_code=400, detail="Not a group call")
    if data.get("status") in ("ended", "missed"):
        raise HTTPException(status_code=410, detail="Call is over")
    if uid != data.get("callerId") and uid not in (data.get("members") or []):
        raise HTTPException(status_code=403, detail="You are not in this group")
    active = [m for m in (data.get("activeParticipants") or []) if m != uid]
    if len(active) >= MAX_PEOPLE:
        raise HTTPException(status_code=409, detail="This call is full")

    # a banned account never gets a ticket
    user = db.collection("users").document(uid).get().to_dict() or {}
    if user.get("banned") is True:
        raise HTTPException(status_code=403, detail="Account banned")
    name = user.get("name") or user.get("username") or uid

    url, key, secret = _cfg()
    now = int(time.time())
    token = jwt.encode(
        {
            "iss": key,
            "sub": uid,
            "name": name,
            "nbf": now - 5,
            "exp": now + TICKET_SECONDS,
            "video": {
                "room": req.callId,
                "roomJoin": True,
                "canPublish": True,
                "canSubscribe": True,
                "canPublishData": False,
            },
        },
        secret,
        algorithm="HS256",
    )
    return {"url": url, "token": token}
