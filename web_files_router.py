"""web_files_router.py — raw file access for the OOOR website, straight from the Render disk (no Supabase).

Every route needs `Authorization: Bearer <Firebase ID token>`. Each user only ever sees their own workspace
(gpt2_workspace._safe_path blocks any path that escapes it).

  GET    /web/files                 -> {"files":[{name,path,size,modified}]}
  POST   /web/files/upload          -> multipart field "file"  -> {"path","name","size"}
  GET    /web/files/download?path=  -> the raw file (Content-Disposition: attachment)
  DELETE /web/files?path=           -> {"ok":true}
"""
import mimetypes
import os
from typing import Optional

import firebase_config  # noqa: F401  (initialises firebase_admin)
from fastapi import APIRouter, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from firebase_admin import auth as fb_auth

from gpt2_workspace import (MARKER, MAX_FILE_BYTES, SKIP_DIRS, _rel, _safe_path, _user_root,
                            workspace_delete, workspace_save_upload)

router = APIRouter(prefix="/web")
MAX_LIST = 500


def _uid(authorization: Optional[str]) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Sign in first")
    try:
        return fb_auth.verify_id_token(authorization.split(" ", 1)[1].strip())["uid"]
    except Exception:
        raise HTTPException(status_code=401, detail="Session expired. Sign in again.")


@router.get("/files")
def list_files(authorization: Optional[str] = Header(default=None)):
    uid = _uid(authorization)
    root = _user_root(uid)
    out = []
    for base, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for n in names:
            if n == MARKER or n.startswith("."):
                continue
            full = os.path.join(base, n)
            try:
                st = os.stat(full)
            except OSError:
                continue
            out.append({"name": n, "path": _rel(uid, full), "size": st.st_size, "modified": int(st.st_mtime)})
    out.sort(key=lambda f: f["modified"], reverse=True)
    return {"files": out[:MAX_LIST]}


@router.post("/files/upload")
async def upload_file(file: UploadFile = File(...), authorization: Optional[str] = Header(default=None)):
    uid = _uid(authorization)
    data = await file.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail=f"File is over {MAX_FILE_BYTES // (1024 * 1024)} MB")
    if not data:
        raise HTTPException(status_code=400, detail="File is empty")
    res = workspace_save_upload(uid, file.filename or "upload", data)
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("error") or "Could not save the file")
    path = res.get("path")
    return {"path": path, "name": os.path.basename(path or file.filename or "upload"), "size": len(data)}


@router.get("/files/download")
def download_file(path: str, authorization: Optional[str] = Header(default=None)):
    uid = _uid(authorization)
    try:
        full = _safe_path(uid, path)
    except ValueError:
        raise HTTPException(status_code=400, detail="Bad path")
    if not os.path.isfile(full):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(full, filename=os.path.basename(full),
                        media_type=mimetypes.guess_type(full)[0] or "application/octet-stream")


@router.delete("/files")
def delete_file(path: str, authorization: Optional[str] = Header(default=None)):
    uid = _uid(authorization)
    res = workspace_delete(uid, path)
    if not res.get("success", True):
        raise HTTPException(status_code=400, detail=res.get("error") or "Could not delete")
    return {"ok": True}
