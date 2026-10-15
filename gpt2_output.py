#!/usr/bin/env python3
# gpt2_output.py — output_files: the MODEL decides what the user receives.
# ─────────────────────────────────────────────────────────────────────────────
# Before: run_code uploaded everything in ./outputs/ automatically (max 5),
#         so the user got whatever happened to be lying there.
# Now:    every tool just leaves its files in the workspace (run_code copies
#         ./outputs/* into workspace "outputs/"). Nothing reaches the user
#         until the model calls output_files with exactly the files it wants
#         to hand over. It can mix sources in ONE call:
#           - workspace paths   ("outputs/report.pdf", "myrepo/src/app.py")
#           - saved docs        ({"doc_id": "help-1.html"})   <- build_file etc.
#           - existing URLs     ("https://...")               <- generate_image,
#                                                               earlier uploads
# Images come back as kind="image" and the dispatcher (gpt2_test.py) puts
# them in the image gallery shown ABOVE the reply; every other file becomes
# a normal file card.
# ─────────────────────────────────────────────────────────────────────────────

import mimetypes
import os
import re
import uuid
from typing import Optional

from gpt2_workspace import _safe_path, _is_probably_text

MAX_OUTPUT_FILES = 20
MAX_OUTPUT_FILE_BYTES = 10 * 1024 * 1024
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]")


def _kind(name: str) -> str:
    return "image" if os.path.splitext(name.lower().split("?")[0])[1] in IMAGE_EXTS else "file"


def output_files(userid: str, files: list) -> dict:
    """
    Deliver files to the user. YOU choose what they get — nothing from run_code
    or the workspace is sent automatically. `files` is a list; each entry is
    either a workspace path string ("outputs/chart.png"), an http(s) URL string,
    or a dict: {"path": ...} | {"doc_id": ...} | {"url": ..., "filename": ...}.
    Images appear in the image gallery above your reply; other files appear as
    file cards. Returns what was delivered and what failed — only claim files
    that appear in the returned `files` list.
    """
    from gpt2_functions import _upload_bytes_to_supabase  # lazy: avoid import cycle

    if isinstance(files, (str, dict)):
        files = [files]
    if not files:
        return {"success": False, "error": "Give a list of files to output.", "files": []}

    delivered, failed = [], []
    for item in list(files)[:MAX_OUTPUT_FILES]:
        spec = {"path": item} if isinstance(item, str) and not item.startswith(("http://", "https://")) \
            else ({"url": item} if isinstance(item, str) else dict(item or {}))
        label = spec.get("path") or spec.get("doc_id") or spec.get("url") or "?"
        try:
            # 1) already a public URL — pass straight through
            if spec.get("url"):
                url = spec["url"]
                if not url.startswith(("http://", "https://")):
                    raise ValueError("url must start with http(s)://")
                name = spec.get("filename") or os.path.basename(url.split("?")[0]) or "file"
                delivered.append({"filename": name, "url": url, "kind": _kind(name)})
                continue

            # 2) a saved doc (build_file / uploads) — upload its content
            if spec.get("doc_id"):
                from user_doc_manager import UserDocManager
                doc = UserDocManager(userid).get_doc(spec["doc_id"])
                if not doc:
                    raise ValueError("no saved doc with that id")
                data = (doc.get("content") or "").encode("utf-8")
                name = doc.get("filename") or spec["doc_id"]
            # 3) a workspace file
            elif spec.get("path"):
                full = _safe_path(userid, spec["path"])
                if not os.path.isfile(full):
                    raise ValueError("not a file in the workspace")
                if os.path.getsize(full) > MAX_OUTPUT_FILE_BYTES:
                    raise ValueError("over the 10MB delivery limit")
                with open(full, "rb") as f:
                    data = f.read()
                name = spec.get("filename") or os.path.basename(full)
            else:
                raise ValueError("entry needs path, doc_id or url")

            if not data:
                raise ValueError("file is empty")
            if len(data) > MAX_OUTPUT_FILE_BYTES:
                raise ValueError("over the 10MB delivery limit")
            safe = _SAFE_NAME_RE.sub("_", os.path.basename(name))[:80] or "file"
            ctype = mimetypes.guess_type(safe)[0] or "application/octet-stream"
            url = _upload_bytes_to_supabase(userid or "anonymous", f"{uuid.uuid4().hex[:8]}_{safe}", data, ctype)
            if not url:
                raise ValueError("upload failed")
            delivered.append({"filename": safe, "url": url, "kind": _kind(safe), "size": len(data)})
        except Exception as e:
            failed.append({"file": label, "error": str(e)})

    out = {"success": bool(delivered), "files": delivered}
    if failed:
        out["failed"] = failed
    if len(files) > MAX_OUTPUT_FILES:
        out["note"] = f"Only the first {MAX_OUTPUT_FILES} were delivered; call output_files again for the rest."
    return out
