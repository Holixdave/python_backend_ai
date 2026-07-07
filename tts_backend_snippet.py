# tts_backend_snippet.py
#
# Add these two routes to your existing Python AI backend (same app that
# serves /ai-query-stream). Written for FastAPI since that's what your
# SSE streaming style (/ai-query-stream) strongly suggests — if you're
# actually on Flask, tell me and I'll convert it.
#
# WHY THIS LIVES ON THE BACKEND, NOT THE APP:
#   The ElevenLabs (or whichever provider) API key must never ship inside
#   the Flutter app — anyone can decompile an APK and pull it out. The
#   backend holds the key; the app only ever talks to YOUR server.
#
# WHY PLAN CHECKING HAPPENS HERE TOO, NOT JUST IN THE APP:
#   The app already greys out online voices for Free users, but that's a
#   UI-layer restriction only — nothing stops someone from calling
#   /tts-synthesize directly with curl. This route re-checks the user's
#   plan against Firestore before calling the paid provider, same source
#   of truth your SubscriptionService already uses.
#
# pip install elevenlabs firebase-admin

import os
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import Response
from pydantic import BaseModel
import firebase_admin
from firebase_admin import auth as fb_auth, firestore

# Assumes firebase_admin is already initialized elsewhere in your app
# (firebase_admin.initialize_app(...)) — do NOT call it twice.
db = firestore.client()

ELEVENLABS_API_KEY = os.environ["ELEVENLABS_API_KEY"]  # set in your env / Firebase config

# ── Curated voice catalogue ─────────────────────────────────────────────────
# These map to real ElevenLabs premade voice IDs. Swap in your own IDs if
# you clone custom voices instead of using the premade ones.
VOICE_CATALOGUE = [
    {"id": "pNInz6obpgDQGcFmaJgB", "name": "Adam",   "gender": "male",   "locale": "en-US", "preview_text": "Hey, I'm Adam."},
    {"id": "ErXwobaYiN019PkySvjV", "name": "Antoni",  "gender": "male",   "locale": "en-US", "preview_text": "Hi there, Antoni here."},
    {"id": "VR6AewLTigWG4xSOukaG", "name": "Arnold",  "gender": "male",   "locale": "en-US", "preview_text": "Arnold speaking."},
    {"id": "EXAVITQu4vr4xnSDxMaL", "name": "Bella",   "gender": "female", "locale": "en-US", "preview_text": "Hi, I'm Bella."},
]

# app = FastAPI()  # <- use your existing app instance instead of this


class SynthesizeRequest(BaseModel):
    text: str
    voiceId: str


def _get_uid(authorization: str | None) -> str:
    """Same pattern you should already be using to protect other routes —
    the app sends the Firebase ID token, we verify it server-side."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing auth token")
    token = authorization.removeprefix("Bearer ")
    try:
        decoded = fb_auth.verify_id_token(token)
        return decoded["uid"]
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid auth token")


def _user_can_use_online_voice(uid: str) -> bool:
    doc = db.collection("users").document(uid).get()
    if not doc.exists:
        return False
    data = doc.to_dict() or {}
    plan = data.get("plan", "free")
    pro_status = data.get("pro_status", False)
    return pro_status and plan in ("pro", "max")


@app.get("/tts-voices")
async def tts_voices():
    # Voice catalogue itself isn't gated — showing the list (greyed out) is
    # fine, it's SYNTHESIS that's gated below.
    return VOICE_CATALOGUE


@app.post("/tts-synthesize")
async def tts_synthesize(req: SynthesizeRequest, authorization: str | None = Header(None)):
    uid = _get_uid(authorization)

    if not _user_can_use_online_voice(uid):
        raise HTTPException(status_code=403, detail="Online voices require Pro or Max")

    valid_ids = {v["id"] for v in VOICE_CATALOGUE}
    if req.voiceId not in valid_ids:
        raise HTTPException(status_code=400, detail="Unknown voiceId")

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    # ElevenLabs call — swap for Google/Azure if you prefer a different provider
    from elevenlabs.client import ElevenLabs
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

    audio_bytes = b"".join(
        client.text_to_speech.convert(
            voice_id=req.voiceId,
            text=req.text,
            model_id="eleven_turbo_v2",
            output_format="mp3_44100_128",
        )
    )

    return Response(content=audio_bytes, media_type="audio/mpeg")


# ─────────────────────────────────────────────────────────────────────────
# FLUTTER SIDE — you'll need to send the Firebase ID token on these two
# calls too, since the routes now require it. In tts_service.dart, add:
#
#   final idToken = await FirebaseAuth.instance.currentUser?.getIdToken();
#   headers: {
#     'Content-Type': 'application/json',
#     'Authorization': 'Bearer $idToken',
#   }
#
# on both the /tts-voices GET and /tts-synthesize POST calls. Want me to
# make that edit to tts_service.dart now?
# ─────────────────────────────────────────────────────────────────────────
