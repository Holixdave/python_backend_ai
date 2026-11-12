"""
Pictures for the app's Thought sheet ("Looking at the image…"), served by THIS server.

Nothing is uploaded to any storage bucket. A small JPEG thumbnail is kept in memory (and /tmp as a backup)
for a few hours and served from  GET /step-image/{token}.  After that the app shows "Expired" for it.
"""
import io
import os
import secrets
import threading
import time
from collections import OrderedDict

import requests
from fastapi import APIRouter
from fastapi.responses import Response

router = APIRouter()

PUBLIC_BASE = os.getenv("PUBLIC_BASE_URL", "https://chadivavirtuals.name.ng").rstrip("/")
TTL_SECONDS = int(os.getenv("STEP_IMAGE_TTL", str(6 * 3600)))
MAX_ITEMS = 300
MAX_BYTES = 80 * 1024 * 1024
_DIR = os.path.join(os.getenv("TMPDIR", "/tmp"), "step_images")

_lock = threading.Lock()
_mem: "OrderedDict[str, tuple]" = OrderedDict()   # token -> (expires_at, bytes, content_type)
_total = 0


def _evict() -> None:
    global _total
    now = time.time()
    for t in [t for t, v in _mem.items() if v[0] < now]:
        _total -= len(_mem.pop(t)[1])
    while _mem and (len(_mem) > MAX_ITEMS or _total > MAX_BYTES):
        _, v = _mem.popitem(last=False)
        _total -= len(v[1])


def _jpeg(data: bytes, max_side: int = 720) -> tuple:
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(data)).convert("RGB")
        im.thumbnail((max_side, max_side * 2))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=72, optimize=True)
        return buf.getvalue(), "image/jpeg"
    except Exception:
        return data, "image/png"


def store(data: bytes) -> str:
    """Keeps a picture (shrunk to a thumbnail) and returns the public URL the app can load."""
    global _total
    body, ctype = _jpeg(data)
    token = secrets.token_urlsafe(12)
    with _lock:
        _mem[token] = (time.time() + TTL_SECONDS, body, ctype)
        _total += len(body)
        _evict()
    try:
        os.makedirs(_DIR, exist_ok=True)
        with open(os.path.join(_DIR, token), "wb") as f:
            f.write(body)
        with open(os.path.join(_DIR, token + ".t"), "w") as f:
            f.write(f"{time.time() + TTL_SECONDS}|{ctype}")
    except Exception:
        pass
    return f"{PUBLIC_BASE}/step-image/{token}"


def thumb_url(src_url: str) -> str:
    """Downloads a picture the user attached and serves a thumbnail of it from here. Falls back to the original URL."""
    try:
        r = requests.get(src_url, timeout=8)
        if r.ok and r.content:
            return store(r.content)
    except Exception as e:
        print(f"[STEP_IMAGE] could not fetch {src_url[:80]}: {type(e).__name__}")
    return src_url


def thumb_urls(urls: list) -> dict:
    """{original: served url}, fetched in parallel."""
    from concurrent.futures import ThreadPoolExecutor
    urls = list(dict.fromkeys(urls))
    if not urls:
        return {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        return dict(zip(urls, ex.map(thumb_url, urls)))


@router.get("/step-image/{token}")
def step_image(token: str):
    token = token.split(".")[0]
    with _lock:
        v = _mem.get(token)
        if v and v[0] >= time.time():
            return Response(v[1], media_type=v[2], headers={"Cache-Control": "private, max-age=3600"})
    try:  # memory was wiped (restart) but the backup file is still there
        meta = open(os.path.join(_DIR, token + ".t")).read().split("|", 1)
        if float(meta[0]) >= time.time():
            body = open(os.path.join(_DIR, token), "rb").read()
            return Response(body, media_type=meta[1], headers={"Cache-Control": "private, max-age=3600"})
    except Exception:
        pass
    return Response(status_code=404)
