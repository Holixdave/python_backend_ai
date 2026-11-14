# collab_docs.py — editable, shareable docs (create / read / update / share / export)
# ─────────────────────────────────────────────────────────────────────────────
# Independent of user_doc_manager.py (not imported, not touched).
#
# Storage: Firestore  shared_docs/{docId}   { ownerUid, title, markdown, version, createdAt, updatedAt,
#                                             lastEditedBy, collaborators: {uid: "edit"|"view"},
#                                             linkShare: {mode: "off"|"view"|"edit", token} }
#          shared_docs/{docId}/versions/{n} (last 20 snapshots, so an AI/user overwrite can be undone)
# Concurrency: every write is a Firestore transaction with an optimistic `base_version` check, so the AI
#   can never silently overwrite what the user typed in the app (409 / conflict message instead).
# AI tools: create_doc, list_docs, read_doc, update_doc, share_doc, export_doc
# App REST (Firebase ID token, same as /ai-job): GET/PUT/DELETE /documents/{id}, GET /documents, POST /documents/{id}/share,
#   POST /documents/{id}/export, GET /documents/shared/{token}, POST /documents/shared/{token}/join
# Export: docx (python-docx), pdf (reportlab + the repo's Unicode fonts), md — uploaded with the existing
#   Supabase helper, so the file URL works exactly like every other file card.
# ─────────────────────────────────────────────────────────────────────────────
import functools
import io
import os
import re
import secrets
import time
from typing import List, Optional

from card_tools import clip, fail, ok, to_int

MAX_MARKDOWN = 200_000
KEEP_VERSIONS = 20
COLLECTION = "shared_docs"


class DocError(Exception):
    def __init__(self, code: str, message: str = "", **extra):
        super().__init__(message or code)
        self.code, self.extra = code, extra


def _now_ms() -> int:
    return int(time.time() * 1000)


def _db():
    import firebase_config  # noqa: F401  (initialises firebase_admin before firestore.client())
    from firebase_admin import firestore
    return firestore.client()


def _role(doc: dict, uid: str) -> Optional[str]:
    if doc.get("ownerUid") == uid:
        return "owner"
    return (doc.get("collaborators") or {}).get(uid)


def _public_view(doc_id: str, doc: dict, uid: str, with_markdown: bool = True) -> dict:
    out = {"doc_id": doc_id, "title": doc.get("title") or "Untitled", "version": int(doc.get("version", 1)),
           "updatedAt": doc.get("updatedAt"), "createdAt": doc.get("createdAt"), "role": _role(doc, uid),
           "ownerUid": doc.get("ownerUid"), "lastEditedBy": doc.get("lastEditedBy"),
           "linkMode": (doc.get("linkShare") or {}).get("mode", "off")}
    if with_markdown:
        out["markdown"] = doc.get("markdown") or ""
    return out


def _check_md(markdown: str) -> str:
    md = str(markdown or "")
    if len(md) > MAX_MARKDOWN:
        raise DocError("too_big", f"Document is over {MAX_MARKDOWN:,} characters.")
    return md


# ── storage operations (shared by the AI tools and the REST router) ─────────
def create_record(uid: str, title: str, markdown: str) -> dict:
    md = _check_md(markdown)
    ref = _db().collection(COLLECTION).document()
    now = _now_ms()
    doc = {"ownerUid": uid, "title": clip(title or "Untitled", 120), "markdown": md, "version": 1, "createdAt": now,
           "updatedAt": now, "lastEditedBy": uid, "collaborators": {}, "linkShare": {"mode": "off", "token": None}}
    ref.set(doc)
    return _public_view(ref.id, doc, uid)


def get_record(uid: str, doc_id: str) -> dict:
    snap = _db().collection(COLLECTION).document(doc_id).get()
    if not snap.exists:
        raise DocError("not_found", "No such document.")
    doc = snap.to_dict() or {}
    if not _role(doc, uid):
        raise DocError("forbidden", "You don't have access to this document.")
    return _public_view(doc_id, doc, uid)


def update_record(uid: str, doc_id: str, markdown: Optional[str] = None, title: Optional[str] = None,
                  base_version: Optional[int] = None) -> dict:
    from firebase_admin import firestore
    client = _db()
    ref = client.collection(COLLECTION).document(doc_id)

    @firestore.transactional
    def txn_fn(txn):
        snap = ref.get(transaction=txn)
        if not snap.exists:
            raise DocError("not_found", "No such document.")
        d = snap.to_dict() or {}
        if _role(d, uid) not in ("owner", "edit"):
            raise DocError("forbidden", "You can't edit this document.")
        cur = int(d.get("version", 1))
        if base_version is not None and int(base_version) != cur:
            raise DocError("conflict", f"Edited since version {base_version}; it is now version {cur}.",
                           current=_public_view(doc_id, d, uid))
        new_md = d.get("markdown") or "" if markdown is None else _check_md(markdown)
        txn.set(ref.collection("versions").document(str(cur)),
                {"markdown": d.get("markdown") or "", "title": d.get("title"), "by": d.get("lastEditedBy"), "at": d.get("updatedAt")})
        if cur - KEEP_VERSIONS > 0:
            txn.delete(ref.collection("versions").document(str(cur - KEEP_VERSIONS)))
        patch = {"markdown": new_md, "version": cur + 1, "updatedAt": _now_ms(), "lastEditedBy": uid}
        if title:
            patch["title"] = clip(title, 120)
        txn.update(ref, patch)
        return {**d, **patch}

    final = txn_fn(client.transaction())
    return _public_view(doc_id, final, uid)


def list_records(uid: str, limit: int = 50) -> List[dict]:
    col = _db().collection(COLLECTION)
    seen, out = set(), []
    for q in (col.where("ownerUid", "==", uid).limit(limit),
              col.where(f"collaborators.{uid}", "in", ["edit", "view"]).limit(limit)):
        for s in q.stream():
            if s.id not in seen:
                seen.add(s.id)
                out.append(_public_view(s.id, s.to_dict() or {}, uid, with_markdown=False))
    return sorted(out, key=lambda d: d.get("updatedAt") or 0, reverse=True)[:limit]


def delete_record(uid: str, doc_id: str) -> None:
    ref = _db().collection(COLLECTION).document(doc_id)
    snap = ref.get()
    if not snap.exists:
        raise DocError("not_found", "No such document.")
    if (snap.to_dict() or {}).get("ownerUid") != uid:
        raise DocError("forbidden", "Only the owner can delete a document.")
    for v in ref.collection("versions").stream():
        v.reference.delete()
    ref.delete()


def share_record(uid: str, doc_id: str, mode: str) -> dict:
    mode = str(mode or "").lower()
    if mode not in ("off", "view", "edit"):
        raise DocError("bad_request", "mode must be off, view or edit.")
    ref = _db().collection(COLLECTION).document(doc_id)
    snap = ref.get()
    if not snap.exists:
        raise DocError("not_found", "No such document.")
    d = snap.to_dict() or {}
    if d.get("ownerUid") != uid:
        raise DocError("forbidden", "Only the owner can change sharing.")
    token = (d.get("linkShare") or {}).get("token") or secrets.token_urlsafe(16)
    if mode == "off":                      # turning sharing off kills the link; a new one is minted next time
        ref.update({"linkShare": {"mode": "off", "token": None}})
        return {"doc_id": doc_id, "mode": "off", "url": None}
    ref.update({"linkShare": {"mode": mode, "token": token}})
    base = (os.getenv("PUBLIC_BASE_URL") or os.getenv("RENDER_EXTERNAL_URL") or "").rstrip("/")
    return {"doc_id": doc_id, "mode": mode, "token": token, "url": f"{base}/documents/shared/{token}" if base else None}


def _by_token(token: str):
    docs = list(_db().collection(COLLECTION).where("linkShare.token", "==", token).limit(1).stream())
    if not docs or (docs[0].to_dict().get("linkShare") or {}).get("mode") not in ("view", "edit"):
        raise DocError("not_found", "This link doesn't work (it may have been turned off).")
    return docs[0]


def read_shared(token: str) -> dict:
    s = _by_token(token)
    d = s.to_dict() or {}
    return {"doc_id": s.id, "title": d.get("title"), "markdown": d.get("markdown") or "", "version": int(d.get("version", 1)),
            "mode": d["linkShare"]["mode"], "updatedAt": d.get("updatedAt")}


def join_shared(uid: str, token: str) -> dict:
    s = _by_token(token)
    d = s.to_dict() or {}
    if d.get("ownerUid") != uid:
        s.reference.update({f"collaborators.{uid}": d["linkShare"]["mode"]})
        d.setdefault("collaborators", {})[uid] = d["linkShare"]["mode"]
    return _public_view(s.id, d, uid)


# ── markdown -> blocks (one tiny parser feeds both exporters) ───────────────
_LIST_RE = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
_HR_RE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")


def _cells(line: str) -> List[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse_markdown(md: str) -> list:
    lines, i, blocks = (md or "").replace("\r\n", "\n").split("\n"), 0, []
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            i += 1
        elif ln.strip().startswith("```"):
            i, code = i + 1, []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1
            blocks.append({"t": "code", "text": "\n".join(code)})
        elif re.match(r"^#{1,6}\s+", ln):
            m = re.match(r"^(#{1,6})\s+(.*?)\s*#*$", ln)
            blocks.append({"t": "h", "level": len(m.group(1)), "text": m.group(2)})
            i += 1
        elif _HR_RE.match(ln):
            blocks.append({"t": "hr"})
            i += 1
        elif "|" in ln and i + 1 < len(lines) and _TABLE_SEP.match(lines[i + 1]):
            rows, i = [_cells(ln)], i + 2
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                rows.append(_cells(lines[i]))
                i += 1
            blocks.append({"t": "table", "rows": rows})
        elif _LIST_RE.match(ln):
            items, ordered = [], bool(re.match(r"^\s*\d", ln))
            while i < len(lines) and _LIST_RE.match(lines[i]):
                m = _LIST_RE.match(lines[i])
                if not m.group(1) and bool(re.match(r"\d", m.group(2))) != ordered:
                    break                                  # bullets then numbers (or back) = a new list
                items.append({"level": min(len(m.group(1).replace("\t", "  ")) // 2, 2), "text": m.group(3)})
                i += 1
            blocks.append({"t": "list", "ordered": ordered, "items": items})
        elif ln.lstrip().startswith(">"):
            q = []
            while i < len(lines) and lines[i].lstrip().startswith(">"):
                q.append(lines[i].lstrip()[1:].strip())
                i += 1
            blocks.append({"t": "quote", "text": " ".join(q)})
        else:
            para = []
            while i < len(lines) and lines[i].strip() and not (re.match(r"^#{1,6}\s+", lines[i]) or lines[i].strip().startswith("```")
                                                              or _LIST_RE.match(lines[i]) or _HR_RE.match(lines[i])):
                para.append(lines[i].strip())
                i += 1
            blocks.append({"t": "p", "text": " ".join(para)})
    return blocks


_INLINE = re.compile(r"(\*\*[^*]+\*\*|__[^_]+__|\*[^*\s][^*]*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))")


def _inline_parts(text: str) -> list:
    """-> [(kind, text, extra)] kind: plain|b|i|code|link"""
    parts = []
    for tok in _INLINE.split(text or ""):
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**") and len(tok) > 4:
            parts.append(("b", tok[2:-2], None))
        elif tok.startswith("__") and tok.endswith("__") and len(tok) > 4:
            parts.append(("b", tok[2:-2], None))
        elif tok.startswith("`") and tok.endswith("`") and len(tok) > 2:
            parts.append(("code", tok[1:-1], None))
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            parts.append(("i", tok[1:-1], None))
        elif tok.startswith("["):
            m = re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", tok)
            parts.append(("link", m.group(1), m.group(2)) if m else ("plain", tok, None))
        else:
            parts.append(("plain", tok, None))
    return parts


# ── exporters ───────────────────────────────────────────────────────────────
def build_docx(title: str, markdown: str) -> bytes:
    from docx import Document
    from docx.shared import Pt
    d = Document()
    d.add_heading(title or "Untitled", 0)

    def runs(par, text):
        for kind, t, extra in _inline_parts(text):
            r = par.add_run(f"{t} ({extra})" if kind == "link" else t)
            r.bold, r.italic = kind == "b", kind == "i"
            if kind == "code":
                r.font.name = "Courier New"

    for b in parse_markdown(markdown):
        t = b["t"]
        if t == "h":
            runs(d.add_heading("", min(b["level"], 4)), b["text"])
        elif t == "p":
            runs(d.add_paragraph(), b["text"])
        elif t == "list":
            base = "List Number" if b["ordered"] else "List Bullet"
            for it in b["items"]:
                runs(d.add_paragraph(style=base if it["level"] == 0 else f"{base} {it['level'] + 1}"), it["text"])
        elif t == "quote":
            runs(d.add_paragraph(style="Quote"), b["text"])
        elif t == "code":
            r = d.add_paragraph().add_run(b["text"])
            r.font.name, r.font.size = "Courier New", Pt(9)
        elif t == "hr":
            d.add_paragraph("─" * 30)
        elif t == "table" and b["rows"]:
            width = max(len(r) for r in b["rows"])
            tbl = d.add_table(rows=len(b["rows"]), cols=width)
            tbl.style = "Table Grid"
            for ri, row in enumerate(b["rows"]):
                for ci in range(width):
                    cell = tbl.cell(ri, ci)
                    cell.text = ""
                    runs(cell.paragraphs[0], row[ci] if ci < len(row) else "")
                    if ri == 0:
                        for r in cell.paragraphs[0].runs:
                            r.bold = True
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def _rl_inline(text: str, font: str, markup_fn) -> str:
    from xml.sax.saxutils import escape
    out = []
    for kind, t, extra in _inline_parts(text):
        s = markup_fn(escape(t), font) if markup_fn else escape(t)
        if kind == "b":
            s = f"<b>{s}</b>"
        elif kind == "i":
            s = f"<i>{s}</i>"
        elif kind == "code":
            s = f'<font face="Courier">{escape(t)}</font>'
        elif kind == "link":
            s = f'<a href="{escape(extra, {chr(34): "&quot;"})}" color="blue">{s}</a>'
        out.append(s)
    return "".join(out)


def build_pdf(title: str, markdown: str) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import HRFlowable, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle

    font, markup_fn = "Helvetica", None
    try:                                                    # the repo's fonts cover Naira, Yoruba letters, maths
        from fonts import pdf_fonts
        font, markup_fn = pdf_fonts.register("Inter"), pdf_fonts.markup
    except Exception as e:
        print(f"[DOCS] Unicode fonts unavailable ({type(e).__name__}); PDF falls back to Helvetica")

    body = ParagraphStyle("body", fontName=font, fontSize=10.5, leading=15, spaceAfter=6)
    h = {n: ParagraphStyle(f"h{n}", parent=body, fontSize=s, leading=s + 5, spaceBefore=10, spaceAfter=6)
         for n, s in {1: 20, 2: 16, 3: 13.5, 4: 12, 5: 11, 6: 10.5}.items()}
    story = [Paragraph(_rl_inline(title or "Untitled", font, markup_fn), ParagraphStyle("t", parent=h[1], fontSize=24, leading=29)),
             Spacer(1, 6)]
    for b in parse_markdown(markdown):
        t = b["t"]
        if t == "h":
            story.append(Paragraph(f"<b>{_rl_inline(b['text'], font, markup_fn)}</b>", h[b["level"]]))
        elif t == "p":
            story.append(Paragraph(_rl_inline(b["text"], font, markup_fn), body))
        elif t == "quote":
            story.append(Paragraph(f"<i>{_rl_inline(b['text'], font, markup_fn)}</i>",
                                   ParagraphStyle("q", parent=body, leftIndent=14, textColor=colors.HexColor("#555555"))))
        elif t == "list":
            n = 0
            for it in b["items"]:
                n += 1
                story.append(Paragraph(_rl_inline(it["text"], font, markup_fn),
                                       ParagraphStyle("li", parent=body, leftIndent=16 + 14 * it["level"], bulletIndent=4 + 14 * it["level"], spaceAfter=3),
                                       bulletText=f"{n}." if b["ordered"] else "•"))
        elif t == "code":
            story.append(Preformatted(b["text"], ParagraphStyle("c", parent=body, fontName="Courier", fontSize=8.5, leading=11,
                                                                  backColor=colors.HexColor("#F3F3F3"), borderPadding=4)))
            story.append(Spacer(1, 6))
        elif t == "hr":
            story.append(HRFlowable(width="100%", color=colors.lightgrey, spaceBefore=6, spaceAfter=6))
        elif t == "table" and b["rows"]:
            width = max(len(r) for r in b["rows"])
            cell = ParagraphStyle("cell", parent=body, fontSize=9, leading=12, spaceAfter=0)
            data = [[Paragraph(_rl_inline(r[c] if c < len(r) else "", font, markup_fn), cell) for c in range(width)] for r in b["rows"]]
            tbl = Table(data, repeatRows=1)
            tbl.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEEEEE")),
                                     ("VALIGN", (0, 0), (-1, -1), "TOP")]))
            story += [tbl, Spacer(1, 8)]
    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=18 * mm,
                      title=title or "Untitled").build(story)
    return buf.getvalue()


_EXPORTS = {
    "docx": ("docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", lambda t, m: build_docx(t, m)),
    "pdf": ("pdf", "application/pdf", lambda t, m: build_pdf(t, m)),
    "md": ("md", "text/markdown; charset=utf-8", lambda t, m: f"# {t}\n\n{m}\n".encode("utf-8")),
}


def export_record(uid: str, doc_id: str, fmt: str) -> dict:
    fmt = {"word": "docx", "markdown": "md"}.get(str(fmt or "docx").lower(), str(fmt or "docx").lower())
    if fmt not in _EXPORTS:
        raise DocError("bad_request", "format must be docx, pdf or md.")
    doc = get_record(uid, doc_id)
    ext, mime, builder = _EXPORTS[fmt]
    try:
        data = builder(doc["title"], doc["markdown"])
    except Exception as e:
        raise DocError("export_failed", f"Couldn't build the {fmt}: {type(e).__name__}: {e}")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", doc["title"]).strip("-")[:50] or "document"
    filename = f"{slug}-v{doc['version']}.{ext}"
    from gpt2_functions import _upload_bytes_to_supabase           # lazy: heavy module, same helper build_file uses
    url = _upload_bytes_to_supabase(uid, filename, data, mime)
    if not url:
        raise DocError("upload_failed", "The file was built but couldn't be uploaded (storage is not configured).")
    return {"doc_id": doc_id, "format": fmt, "filename": filename, "url": url, "version": doc["version"]}


# ── AI tools (return dicts; never raise) ────────────────────────────────────
def _card(view: dict) -> dict:
    return {"type": "doc", "doc_id": view["doc_id"], "title": view["title"], "version": view["version"],
            "role": view.get("role"), "preview": clip((view.get("markdown") or "").replace("\n", " "), 240),
            "updatedAt": view.get("updatedAt")}


def _guard(fn):
    @functools.wraps(fn)          # keeps the REAL signature: tool schemas and userid injection read it
    def wrapper(*a, **k):
        try:
            return fn(*a, **k)
        except DocError as e:
            if e.code == "conflict":
                return fail("The user edited this document since you read it. Call read_doc again, apply your change "
                            "to the NEW text, and update with the new base_version.", code="conflict")
            return fail(str(e), code=e.code)
        except Exception as e:
            print(f"[DOCS] tool error {fn.__name__}: {type(e).__name__}: {e}")
            return fail(f"Document service error ({type(e).__name__}). Tell the user it didn't work.")
    return wrapper


@_guard
def create_doc(userid: str, title: str, markdown: str) -> dict:
    """
    Create a NEW editable document the user can open, edit, share and download as Word or PDF. Use it for any
    document they want to keep: report, letter, CV, essay, notes, lesson plan, contract, guide. Write the COMPLETE
    text in markdown (headings #, lists, **bold**, tables).

    Args:
        title: the document title
        markdown: the full document text in markdown
    """
    view = create_record(userid, title, markdown)
    return ok(f"Created the document \"{view['title']}\" (id {view['doc_id']}). A card with Open / Share / Download "
              f"buttons is shown. Don't paste the text again; say what you wrote in a line.", _card(view), doc_id=view["doc_id"])


@_guard
def list_docs(userid: str) -> dict:
    """List the user's documents (own and shared with them), newest first — use it to find a doc_id."""
    items = list_records(userid)
    return ok(f"{len(items)} document(s).", None, documents=[{"doc_id": d["doc_id"], "title": d["title"], "version": d["version"],
                                                             "role": d["role"]} for d in items])


@_guard
def read_doc(userid: str, doc_id: str) -> dict:
    """
    Read a document's current text and version before changing it. Always do this first if the user may have
    edited the document themselves.

    Args:
        doc_id: the document id (from create_doc or list_docs)
    """
    v = get_record(userid, doc_id)
    return ok(f"Version {v['version']}. When you call update_doc pass base_version={v['version']}.", None,
              doc_id=doc_id, title=v["title"], version=v["version"], role=v["role"], markdown=v["markdown"][:60000],
              truncated=len(v["markdown"]) > 60000)


@_guard
def update_doc(userid: str, doc_id: str, markdown: str, base_version: Optional[int] = None, title: str = "") -> dict:
    """
    Replace a document's text with the COMPLETE new markdown (not a fragment). Previous versions are kept.

    Args:
        doc_id: the document id
        markdown: the full new text of the whole document
        base_version: the version you read with read_doc (protects the user's own edits)
        title: new title, only if it should change
    """
    v = update_record(userid, doc_id, markdown, title or None, to_int(base_version))
    return ok(f"Updated \"{v['title']}\" to version {v['version']}. The card shows it; don't repaste the text.", _card(v),
              doc_id=doc_id, version=v["version"])


@_guard
def share_doc(userid: str, doc_id: str, mode: str = "view") -> dict:
    """
    Turn on (or off) link sharing for a document so others can read or edit it. Only the owner can share.

    Args:
        doc_id: the document id
        mode: view (anyone with the link can read), edit (people who open it in the app can edit), or off
    """
    r = share_record(userid, doc_id, mode)
    if r["mode"] == "off":
        return ok("Link sharing is now OFF; the old link no longer works.", None, **r)
    return ok(f"Link sharing is on ({r['mode']}). Give the user this link: {r.get('url') or '(server has no public URL set)'}", None, **r)


@_guard
def export_doc(userid: str, doc_id: str, format: str = "docx") -> dict:
    """
    Turn a document into a downloadable file and send it to the user as a file card.

    Args:
        doc_id: the document id
        format: docx (Word), pdf, or md
    """
    r = export_record(userid, doc_id, format)
    return {"success": True, "url": r["url"], "filename": r["filename"], "format": r["format"], "version": r["version"]}


# ── REST router for the app ─────────────────────────────────────────────────
def build_router():
    from typing import Optional as Opt
    from fastapi import APIRouter, Header, HTTPException
    from firebase_admin import auth as fb_auth
    from pydantic import BaseModel

    router = APIRouter()

    def uid_of(authorization: Opt[str]) -> str:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing auth token")
        try:
            return fb_auth.verify_id_token(authorization.removeprefix("Bearer ").strip())["uid"]
        except Exception:
            raise HTTPException(status_code=401, detail="Invalid auth token")

    def run(fn, *a, **k):
        try:
            return fn(*a, **k)
        except DocError as e:
            status = {"not_found": 404, "forbidden": 403, "conflict": 409, "too_big": 413, "bad_request": 400}.get(e.code, 500)
            detail = {"code": e.code, "message": str(e)}
            if e.code == "conflict":
                detail["current"] = e.extra.get("current")
            raise HTTPException(status_code=status, detail=detail)

    class EditBody(BaseModel):
        markdown: Opt[str] = None
        title: Opt[str] = None
        base_version: Opt[int] = None

    class ShareBody(BaseModel):
        mode: str

    class ExportBody(BaseModel):
        format: str = "docx"

    class CreateBody(BaseModel):
        title: str = "Untitled"
        markdown: str = ""

    @router.get("/documents")
    def docs_list(authorization: Opt[str] = Header(None)):
        return {"documents": run(list_records, uid_of(authorization))}

    @router.post("/documents", status_code=201)
    def docs_create(body: CreateBody, authorization: Opt[str] = Header(None)):
        return run(create_record, uid_of(authorization), body.title, body.markdown)

    @router.get("/documents/shared/{token}")
    def docs_shared_read(token: str):
        return run(read_shared, token)

    @router.post("/documents/shared/{token}/join")
    def docs_shared_join(token: str, authorization: Opt[str] = Header(None)):
        return run(join_shared, uid_of(authorization), token)

    @router.get("/documents/{doc_id}")
    def docs_get(doc_id: str, authorization: Opt[str] = Header(None)):
        return run(get_record, uid_of(authorization), doc_id)

    @router.put("/documents/{doc_id}")
    def docs_put(doc_id: str, body: EditBody, authorization: Opt[str] = Header(None)):
        return run(update_record, uid_of(authorization), doc_id, body.markdown, body.title, body.base_version)

    @router.delete("/documents/{doc_id}", status_code=204)
    def docs_delete(doc_id: str, authorization: Opt[str] = Header(None)):
        run(delete_record, uid_of(authorization), doc_id)

    @router.post("/documents/{doc_id}/share")
    def docs_share(doc_id: str, body: ShareBody, authorization: Opt[str] = Header(None)):
        return run(share_record, uid_of(authorization), doc_id, body.mode)

    @router.post("/documents/{doc_id}/export")
    def docs_export(doc_id: str, body: ExportBody, authorization: Opt[str] = Header(None)):
        return run(export_record, uid_of(authorization), doc_id, body.format)

    return router
