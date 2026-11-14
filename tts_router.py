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
import subprocess
import tempfile
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


# Maps each existing voice id (what the app sends) to a free Piper voice.
# Same ids/names as before, so the Flutter app needs NO change.
PIPER_VOICE_FOR = {
    "pNInz6obpgDQGcFmaJgB": "en_US-ryan-medium",   # Adam
    "ErXwobaYiN019PkySvjV": "en_US-joe-medium",    # Antoni
    "VR6AewLTigWG4xSOukaG": "en_GB-alan-medium",   # Arnold
    "EXAVITQu4vr4xnSDxMaL": "en_US-amy-medium",    # Bella
}
PIPER_BIN = os.getenv("PIPER_BIN", "/opt/piper/piper")
PIPER_MODEL_DIR = os.getenv("PIPER_MODEL_DIR", "/app/models")


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
    Self-hosted Piper TTS (free, no API key). Piper makes a WAV; ffmpeg
    (already in the Dockerfile) converts it to MP3 so the app still gets
    audio/mpeg exactly like before.
    """
    voice = PIPER_VOICE_FOR[voice_id]
    model = os.path.join(PIPER_MODEL_DIR, f"{voice}.onnx")
    if not os.path.exists(model):
        raise HTTPException(status_code=503, detail="Voice model not installed on the server")

    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "out.wav")
        p = subprocess.run(
            [PIPER_BIN, "--model", model, "--config", model + ".json",
             "--output_file", wav, "--sentence_silence", "0.2"],
            input=text.encode("utf-8"), capture_output=True, timeout=45,
        )
        if p.returncode != 0:
            raise RuntimeError("piper: " + p.stderr.decode("utf-8", "ignore")[:200])
        m = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", wav,
             "-codec:a", "libmp3lame", "-b:a", "96k", "-f", "mp3", "pipe:1"],
            capture_output=True, timeout=30,
        )
        if m.returncode != 0 or not m.stdout:
            raise RuntimeError("ffmpeg: " + m.stderr.decode("utf-8", "ignore")[:200])
        return m.stdout


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
