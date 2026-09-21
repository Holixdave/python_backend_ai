# tts_router.py
#
# Wired version of tts_backend_snippet.py: same two routes, same rules
# (Firebase token check + Pro/Max plan check), but as an APIRouter so
# main.py can include it with one line.
#
# Changes from the snippet (all bug/safety fixes, behaviour is the same):
#  1. Uses APIRouter — the snippet used @app.get/@app.post but `app` was
#     commented out, so importing it would raise NameError.
#  2. Missing ELEVENLABS_API_KEY no longer crashes the WHOLE backend at
#     startup (os.environ[...] did). Only /tts-synthesize returns 503.
#  3. Routes are plain `def`, so FastAPI runs them in a threadpool. The
#     Firebase / Firestore / ElevenLabs calls are blocking and would
#     otherwise freeze /ai-query and /ai-query-stream while audio renders.
#  4. Text length cap, so nobody can burn your ElevenLabs credits with a
#     giant payload.
#
# pip install elevenlabs        (firebase-admin is already in your project)

import os
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import Response
from firebase_admin import auth as fb_auth, firestore
from pydantic import BaseModel

import firebase_config  # noqa: F401  (initialises firebase_admin before Firestore is used)

router = APIRouter(tags=["tts"])

MAX_TTS_CHARS = 2000

# ── Curated voice catalogue (unchanged) ────────────────────────────────────
VOICE_CATALOGUE = [
    {"id": "pNInz6obpgDQGcFmaJgB", "name": "Adam", "gender": "male", "locale": "en-US", "preview_text": "Hey, I'm Adam."},
    {"id": "ErXwobaYiN019PkySvjV", "name": "Antoni", "gender": "male", "locale": "en-US", "preview_text": "Hi there, Antoni here."},
    {"id": "VR6AewLTigWG4xSOukaG", "name": "Arnold", "gender": "male", "locale": "en-US", "preview_text": "Arnold speaking."},
    {"id": "EXAVITQu4vr4xnSDxMaL", "name": "Bella", "gender": "female", "locale": "en-US", "preview_text": "Hi, I'm Bella."},
]


class SynthesizeRequest(BaseModel):
    text: str
    voiceId: str


def _get_uid(authorization: Optional[str]) -> str:
    """The app sends its Firebase ID token; verify it server-side."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing auth token")
    token = authorization.removeprefix("Bearer ").strip()
    try:
        return fb_auth.verify_id_token(token)["uid"]
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid auth token")


def _user_can_use_online_voice(uid: str) -> bool:
    doc = firestore.client().collection("users").document(uid).get()
    if not doc.exists:
        return False
    data = doc.to_dict() or {}
    return bool(data.get("pro_status", False)) and data.get("plan", "free") in ("pro", "max")


def _synthesize(text: str, voice_id: str) -> bytes:
    """
    The ONLY provider-specific code. To move to your own voice server
    (or Google/Azure), replace the body of this function — routes and
    checks above/below stay exactly the same. Must return MP3 bytes.
    """
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="Online voices are not configured on the server")

    from elevenlabs.client import ElevenLabs  # lazy: missing package can't crash startup

    client = ElevenLabs(api_key=api_key)
    return b"".join(
        client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id="eleven_turbo_v2",
            output_format="mp3_44100_128",
        )
    )


@router.get("/tts-voices")
def tts_voices():
    # The list itself isn't gated (app shows online voices greyed out for
    # Free users); SYNTHESIS is gated below.
    return VOICE_CATALOGUE


@router.post("/tts-synthesize")
def tts_synthesize(req: SynthesizeRequest, authorization: Optional[str] = Header(None)):
    uid = _get_uid(authorization)

    if not _user_can_use_online_voice(uid):
        raise HTTPException(status_code=403, detail="Online voices require Pro or Max")

    if req.voiceId not in {v["id"] for v in VOICE_CATALOGUE}:
        raise HTTPException(status_code=400, detail="Unknown voiceId")

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")
    if len(text) > MAX_TTS_CHARS:
        raise HTTPException(status_code=400, detail=f"Text too long (max {MAX_TTS_CHARS} characters)")

    try:
        audio = _synthesize(text, req.voiceId)
    except HTTPException:
        raise
    except Exception as e:
        print(f"[TTS] synth failed for {uid}: {e}")
        raise HTTPException(status_code=502, detail="Voice provider failed, try again")

    return Response(content=audio, media_type="audio/mpeg")
