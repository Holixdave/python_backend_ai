#!/usr/bin/env python3
# gpt2_workspace_store.py — makes workspaces survive Render redeploys for free.
# ─────────────────────────────────────────────────────────────────────────────
#   local disk      = fast working copy (wiped on every redeploy)
#   Supabase        = the file BYTES: one zip snapshot per user
#                     (<userid>/_workspace/<token>.zip)
#   Firebase        = the METADATA: users/{userid}/workspace/meta
#                     {token, snapshot_url, bytes, files, updated_at, error}
#
# Flow
#   - first time a user's workspace is touched after a restart, hydrate():
#     read meta from Firestore, download the snapshot, unpack it locally.
#   - every change calls mark_dirty(); ~4s later ONE snapshot is uploaded
#     (changes in that window are coalesced; also flushed on shutdown).
#   - if anything fails (Firebase/Supabase down, too big) the workspace keeps
#     working locally and the failure is reported by status() — it never
#     breaks a tool call.
#
# SAFETY RAILS
#   - if hydrate() FAILS we refuse to persist, so a transient outage can never
#     overwrite a good snapshot with an empty folder.
#   - snapshot name contains a random token (the Supabase bucket is public).
#   - node_modules/.git/etc. are not persisted (re-downloadable).
#   - snapshots over MAX_SNAPSHOT_BYTES (Supabase free upload cap is 50MB) are
#     skipped and the user/AI is told.
# Disable entirely with WORKSPACE_PERSIST=0.
# ─────────────────────────────────────────────────────────────────────────────

import atexit
import io
import os
import secrets
import threading
import time
import zipfile
from typing import Optional

import requests

PERSIST_DELAY_S = 4
MAX_SNAPSHOT_BYTES = 45 * 1024 * 1024
MARKER = ".ws_hydrated"
HYDRATE_RETRY_S = 30
_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".idea", ".dart_tool"}

_lock = threading.Lock()
_hydrated: set = set()          # users whose local copy is known-good
_failed_at: dict = {}           # userid -> last failed hydrate time
_timers: dict = {}              # userid -> pending threading.Timer
_tokens: dict = {}              # userid -> snapshot token
_status: dict = {}              # userid -> human string


def _enabled(userid: Optional[str]) -> bool:
    return bool(userid) and userid != "anonymous" and os.environ.get("WORKSPACE_PERSIST", "1") != "0"


def _meta_ref(userid: str):
    import firebase_config  # noqa: F401  (initialises firebase_admin)
    from firebase_admin import firestore
    return firestore.client().collection("users").document(userid).collection("workspace").document("meta")


def _supabase():
    from gpt2_functions import _upload_bytes_to_supabase
    return _upload_bytes_to_supabase


# ── restore ──────────────────────────────────────────────────────────────────
def _restore(root: str, data: bytes) -> int:
    zf = zipfile.ZipFile(io.BytesIO(data))
    root = os.path.realpath(root)
    n = 0
    for info in zf.infolist():
        if info.filename.endswith("/"):
            continue
        target = os.path.realpath(os.path.join(root, info.filename))
        if not target.startswith(root + os.sep):
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with zf.open(info) as src, open(target, "wb") as out:
            out.write(src.read())
        n += 1
    return n


def hydrate(userid: Optional[str], root: str) -> None:
    """Called whenever a user's workspace root is first resolved in this process."""
    if not _enabled(userid) or userid in _hydrated:
        return
    marker = os.path.join(root, MARKER)
    if os.path.exists(marker):
        _hydrated.add(userid)
        return
    if time.time() - _failed_at.get(userid, 0) < HYDRATE_RETRY_S:
        return
    try:
        snap = _meta_ref(userid).get()
        if snap.exists:
            d = snap.to_dict() or {}
            _tokens[userid] = d.get("token")
            url = d.get("snapshot_url")
            if url:
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                n = _restore(root, r.content)
                print(f"[WORKSPACE_STORE] restored {n} file(s) for {userid!r}")
        open(marker, "w").close()
        _hydrated.add(userid)
        _status[userid] = "restored" if snap.exists else "new"
    except Exception as e:
        _failed_at[userid] = time.time()
        _status[userid] = f"couldn't restore saved workspace yet ({type(e).__name__}) — saving is paused so nothing gets overwritten"
        print(f"[WORKSPACE_STORE] hydrate failed for {userid!r}: {e}")


# ── persist ──────────────────────────────────────────────────────────────────
def mark_dirty(userid: Optional[str]) -> None:
    if not _enabled(userid) or userid not in _hydrated:
        return
    with _lock:
        if userid in _timers:
            return                      # a snapshot is already scheduled; it will include this change
        t = threading.Timer(PERSIST_DELAY_S, _timer_fire, args=(userid,))
        t.daemon = True
        _timers[userid] = t
        t.start()


def _timer_fire(userid: str) -> None:
    with _lock:
        _timers.pop(userid, None)
    persist_now(userid)


def persist_now(userid: str) -> bool:
    if not _enabled(userid) or userid not in _hydrated:
        return False
    from gpt2_workspace import _root_path  # lazy: avoid import cycle
    root = _root_path(userid)
    if not os.path.isdir(root):
        return False
    try:
        buf = io.BytesIO()
        files = 0
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for dp, dn, fn in os.walk(root):
                dn[:] = [d for d in dn if d not in _SKIP_DIRS and not os.path.islink(os.path.join(dp, d))]
                for f in fn:
                    if f == MARKER:
                        continue
                    fp = os.path.join(dp, f)
                    if os.path.islink(fp):
                        continue
                    zf.write(fp, os.path.relpath(fp, root))
                    files += 1
        data = buf.getvalue()

        if len(data) > MAX_SNAPSHOT_BYTES:
            msg = (f"workspace is {len(data)//1024//1024}MB compressed — over the "
                   f"{MAX_SNAPSHOT_BYTES//1024//1024}MB save limit, so it is NOT being saved across restarts. Delete big files.")
            _status[userid] = msg
            try:
                _meta_ref(userid).set({"error": msg, "updated_at": int(time.time())}, merge=True)
            except Exception:
                pass
            return False

        token = _tokens.get(userid) or secrets.token_urlsafe(18)
        _tokens[userid] = token
        url = _supabase()(userid, f"_workspace/{token}.zip", data, "application/zip")
        if not url:
            _status[userid] = "last save to Supabase failed — will retry on the next change"
            return False
        _meta_ref(userid).set({
            "token": token, "snapshot_url": url, "bytes": len(data),
            "files": files, "updated_at": int(time.time()), "error": None,
        })
        _status[userid] = f"saved ({files} files, {len(data)//1024} KB)"
        return True
    except Exception as e:
        _status[userid] = f"last save failed ({type(e).__name__})"
        print(f"[WORKSPACE_STORE] persist failed for {userid!r}: {e}")
        return False


def status(userid: Optional[str]) -> Optional[str]:
    if not _enabled(userid):
        return None
    return _status.get(userid)


def flush_all() -> None:
    """Persist anything still waiting on its debounce timer (runs at shutdown)."""
    with _lock:
        pending = list(_timers.items())
        _timers.clear()
    for uid, t in pending:
        t.cancel()
        persist_now(uid)


atexit.register(flush_all)
