#!/usr/bin/env python3
# gpt2_design_check.py — design_check tool: the AI LOOKS at what it built before handing it over.
# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE (all pip-only: PyMuPDF + Pillow + bs4, no browser, no Docker):
#   1. STATIC checks on the file itself (blank pages, text running off the page,
#      placeholder/filler text, missing local files, no viewport tag, fixed widths...)
#   2. RENDER to images: PDF pages with PyMuPDF; HTML with PyMuPDF's Story renderer
#      (390pt phone width; NO JavaScript, flex/grid/media queries are not drawn);
#      image files are used as they are.
#   3. LOOK: each image goes to the vision model ONE AT A TIME with a strict QA
#      checklist and must answer JSON {"ok": bool, "issues": [...]}.
#   4. Returns a verdict (status = passed | needs_fixes | not_visually_verified).
#      The AI fixes ONLY the listed issues and calls this again (max 3 rounds).
#
# Honest by design: if the vision model can't answer, the status says so — it never
# reports "passed" for a page nobody looked at.
#
#   python3 gpt2_design_check.py --selftest     (checks PyMuPDF rendering on THIS server)
# ─────────────────────────────────────────────────────────────────────────────

import base64
import io
import json
import os
import re
import tempfile
import threading
import time
from collections import Counter
from typing import Callable, Optional

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_HTML_BYTES = 600_000
MAX_VIEW_PAGES = 4          # images sent to the vision model per call
MAX_STORY_PAGES = 8         # HTML pages laid out before we stop
VIEW_WIDTH_PX = 900
PHONE_W, PHONE_H = 390, 844
MAX_ROUNDS = 3

_PLACEHOLDER_RE = re.compile(
    r"lorem ipsum|dolor sit amet|your (?:text|name|title|content|company|logo|tagline) here|"
    r"insert (?:text|image|title|content|name) here|click to edit|sample text|placeholder text|\btodo\b",
    re.I,
)
_SKIP_REF = ("http:", "https:", "//", "data:", "#", "mailto:", "tel:", "javascript:", "blob:")
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif")

_attempts: dict = {}
_attempts_lock = threading.Lock()


# ── helpers ──────────────────────────────────────────────────────────────────
def _pm():
    try:
        import pymupdf as pm
    except ImportError:
        import fitz as pm       # older PyMuPDF
    return pm


def _to_data_url(img_bytes: bytes) -> str:
    """Shrink to ~900px wide JPEG so a vision call stays small and fast."""
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        if im.width > VIEW_WIDTH_PX:
            im = im.resize((VIEW_WIDTH_PX, max(1, int(im.height * VIEW_WIDTH_PX / im.width))))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=72)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return "data:image/png;base64," + base64.b64encode(img_bytes).decode()


def _render_pdf_pages(pm, doc, limit: int = MAX_VIEW_PAGES) -> list:
    """[(page_number, png_bytes)] for the first `limit` pages."""
    out = []
    for i in range(min(doc.page_count, limit)):
        page = doc.load_page(i)
        zoom = min(2.5, VIEW_WIDTH_PX / max(page.rect.width, 1.0))
        pix = page.get_pixmap(matrix=pm.Matrix(zoom, zoom), alpha=False)
        out.append((i + 1, pix.tobytes("png")))
    return out


def _html_to_pdf_doc(pm, html: str, base_dir: Optional[str] = None):
    """Lay HTML out as phone-width pages with PyMuPDF's Story. No JS; flex/grid/@media not drawn."""
    story = pm.Story(html=html, archive=pm.Archive(base_dir) if base_dir else None)
    mediabox = pm.Rect(0, 0, PHONE_W, PHONE_H)
    where = mediabox + (12, 12, -12, -12)
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.close()
    try:
        writer = pm.DocumentWriter(tmp.name)
        more, pages = 1, 0
        while more and pages < MAX_STORY_PAGES:
            dev = writer.begin_page(mediabox)
            more, _filled = story.place(where)
            story.draw(dev)
            writer.end_page()
            pages += 1
        writer.close()
        return pm.open(tmp.name), bool(more)
    finally:
        try:
            os.unlink(tmp.name)      # PyMuPDF has the file loaded by now on POSIX; best effort
        except OSError:
            pass


# ── static checks ────────────────────────────────────────────────────────────
def _repeated_filler(text: str) -> Optional[str]:
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) >= 15]
    if not lines:
        return None
    line, n = Counter(lines).most_common(1)[0]
    return f"the same line appears {n} times ('{line[:50]}') — looks like filler" if n >= 4 else None


def _static_pdf(doc) -> tuple:
    errors, warns, texts = [], [], []
    for i in range(min(doc.page_count, 40)):
        page, n = doc.load_page(i), i + 1
        text = page.get_text()
        texts.append(text)
        if not text.strip() and not page.get_images() and not page.get_drawings():
            errors.append(f"page {n} is blank")
        w, h = page.rect.width, page.rect.height
        off, edge = False, False
        for b in page.get_text("blocks"):
            if len(b) > 6 and b[6] != 0:
                continue                      # image block
            x0, y0, x1, y1 = b[:4]
            if x0 < -1 or y0 < -1 or x1 > w + 1 or y1 > h + 1:
                off = True
            elif x0 < 10 or x1 > w - 10:
                edge = True
        if off:
            errors.append(f"page {n}: text runs off the edge of the page")
        elif edge:
            warns.append(f"page {n}: text sits within 10pt of the page edge")
    joined = "\n".join(texts)
    if "\ufffd" in joined or "□" in joined or "■" in joined:
        errors.append("some characters print as boxes (font is missing them)")
    if _PLACEHOLDER_RE.search(joined):
        errors.append("placeholder text found (lorem ipsum / 'your text here' / TODO)")
    rep = _repeated_filler(joined)
    if rep:
        errors.append(rep)
    return errors[:8], warns[:5]


def _strip_media_blocks(css: str) -> str:
    return re.sub(r"@media[^{]*\{(?:[^{}]*\{[^{}]*\})*[^{}]*\}", "", css, flags=re.S)


def _static_html(html: str, ref_exists: Callable[[str], bool]) -> tuple:
    from bs4 import BeautifulSoup
    errors, warns = [], []
    soup = BeautifulSoup(html, "html.parser")

    css = "\n".join(t.get_text() for t in soup.find_all("style"))
    css += "\n" + "\n".join(f"x{{{t['style']}}}" for t in soup.find_all(style=True))
    css_no_media = _strip_media_blocks(css)

    if not soup.find("meta", attrs={"name": re.compile("^viewport$", re.I)}):
        errors.append("missing <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"> — breaks on phones")
    if not soup.title or not soup.title.get_text(strip=True):
        warns.append("page has no <title>")

    missing, remote = [], 0
    for tag in soup.find_all(["img", "script", "link", "source", "video", "audio"]):
        if tag.name == "link" and not {"stylesheet", "icon"} & set(tag.get("rel") or []):
            continue
        ref = tag.get("src") or tag.get("href")
        if not ref:
            continue
        ref = ref.strip()
        if ref.lower().startswith(("http:", "https:", "//")):
            if tag.name == "img":
                remote += 1
            continue
        if ref.lower().startswith(_SKIP_REF):
            continue
        if not ref_exists(ref):
            missing.append(ref)
    if missing:
        errors.append("missing local file(s): " + ", ".join(sorted(set(missing))[:5]))
    if remote:
        warns.append(f"{remote} remote image(s) can't be checked here (they may be blocked or slow)")

    wide = sorted({int(v) for v in re.findall(r"(?<![\w-])width\s*:\s*(\d{3,4})px", css_no_media) if int(v) >= 420})
    if wide:
        errors.append(f"fixed width of {wide[0]}px is wider than a 390px phone — use max-width:100% or %")
    if "box-sizing" not in css:
        warns.append("no 'box-sizing: border-box' rule — padded boxes can overflow the screen")
    dead = len(soup.find_all("a", href="#"))
    if dead >= 3:
        warns.append(f"{dead} links go nowhere (href=\"#\")")

    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    text = soup.get_text("\n")
    if len(text.strip()) < 20:
        errors.append("page has almost no visible text")
    if _PLACEHOLDER_RE.search(text):
        errors.append("placeholder text found (lorem ipsum / 'your text here' / TODO)")
    rep = _repeated_filler(text)
    if rep:
        errors.append(rep)
    return errors[:8], warns[:5]


def _inline_local_css(html: str, read_local: Callable[[str], Optional[str]]) -> str:
    """Story ignores <link rel=stylesheet>: paste local CSS files into <style> tags first."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    for link in soup.find_all("link", rel=lambda r: r and "stylesheet" in r):
        href = (link.get("href") or "").strip()
        if not href or href.lower().startswith(_SKIP_REF):
            continue
        css = read_local(href)
        if css:
            style = soup.new_tag("style")
            style.string = css
            link.replace_with(style)
    return str(soup)


# ── the vision "look" ────────────────────────────────────────────────────────
_CHECKLIST = (
    "text cut off or running past the edge; overlapping text or elements; unreadable text (low contrast or tiny); "
    "large empty or blank areas; images that show as empty boxes or are missing; characters printed as squares; "
    "layout that looks unstyled/default; obvious misalignment; content that looks like filler."
)


def _critic_prompt(kind: str, brief: str, label: str) -> str:
    extra = ""
    if kind == "html":
        extra = (" This preview comes from a simplified renderer (no JavaScript, CSS flex/grid shown stacked, no media "
                 "queries). Do NOT report problems that are caused by that; report only real content and styling problems.")
    goal = f" It is meant to be: {brief.strip()[:300]}." if brief and brief.strip() else ""
    return (
        f"You are a strict visual QA reviewer. You see ONE {label} of a {kind} someone just built.{goal}{extra} "
        f"Look for: {_CHECKLIST} "
        'Reply with ONLY JSON: {"ok": true or false, "issues": ["short, specific, fixable issue", ...]}. '
        "ok is true only if nothing here would look broken to a normal user. Max 5 issues. No other text."
    )


def _parse_verdict(answer: str) -> Optional[dict]:
    m = re.search(r"\{.*\}", answer or "", re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("ok"), bool):
        return None
    issues = [str(i).strip()[:160] for i in (obj.get("issues") or []) if str(i).strip()][:5]
    if obj["ok"] is False and not issues:
        issues = ["the reviewer marked this page as broken without details"]
    return {"ok": obj["ok"], "issues": issues if obj["ok"] is False else []}


def _look(kind: str, brief: str, label: str, data_url: str) -> Optional[dict]:
    """One vision call for ONE image. None = the reviewer could not answer (NOT a pass)."""
    try:
        from gpt2_functions import ask_with_vision      # lazy: avoids a circular import at startup
        res = ask_with_vision(_critic_prompt(kind, brief, label), [data_url], [])
    except Exception as e:
        print(f"[DESIGN_CHECK] vision call failed: {e}")
        return None
    return _parse_verdict((res or {}).get("answer") or "")


# ── the tool ─────────────────────────────────────────────────────────────────
def design_check(userid: str, path: str, kind: str = "auto", brief: str = "") -> dict:
    """
    LOOK at a visual file you built (website .html, .pdf, or an image) and get a real verdict before handing it over.
    Runs static checks, renders the pages, and shows each page to a vision reviewer one at a time.
    Call it after every build and again after every fix.

    Args:
        path: workspace path of the file to check, e.g. "outputs/guide.pdf" or "site/index.html".
        kind: "auto" (from the file extension), "pdf", "html" or "image".
        brief: one sentence on what the design is meant to be, e.g. "mobile landing page for a bakery".
    """
    from gpt2_workspace import _rate_ok, _safe_path

    if not path or not str(path).strip():
        return {"success": False, "error": "path is required"}
    if not _rate_ok(userid, "design_check", 40):
        return {"success": False, "error": "design_check limit reached (40 per hour) — try again later."}
    try:
        full = _safe_path(userid, path)
    except Exception as e:
        return {"success": False, "error": str(e)}
    if not os.path.isfile(full):
        return {"success": False, "error": f"'{path}' isn't a file in the workspace."}
    if os.path.getsize(full) > MAX_FILE_BYTES:
        return {"success": False, "error": "file is over 10MB — too big to check."}

    ext = os.path.splitext(full)[1].lower()
    kind = (kind or "auto").strip().lower()
    if kind == "auto":
        kind = "pdf" if ext == ".pdf" else "html" if ext in (".html", ".htm") else "image" if ext in _IMAGE_EXTS else ""
    if kind not in ("pdf", "html", "image"):
        return {"success": False, "error": f"can't check '{ext or path}' files yet — supported: .pdf, .html, .png/.jpg/.webp."}

    key = (userid or "anonymous", path)
    with _attempts_lock:
        now = time.time()
        for k in [k for k, (_n, ts) in _attempts.items() if now - ts > 3600]:
            _attempts.pop(k, None)
        attempt = _attempts.get(key, (0, now))[0] + 1
        _attempts[key] = (attempt, now)

    try:
        pm = None
        images: list = []                  # [(label, png/jpg bytes)]
        pages_total = 0
        truncated_note = None
        limits: list = []

        if kind == "pdf":
            pm = _pm()
            doc = pm.open(full)
            pages_total = doc.page_count
            static_errors, static_warns = _static_pdf(doc)
            images = [(f"page {n}", b) for n, b in _render_pdf_pages(pm, doc)]
            doc.close()

        elif kind == "html":
            if os.path.getsize(full) > MAX_HTML_BYTES:
                return {"success": False, "error": "HTML is over 600KB — too big to check."}
            with open(full, "r", encoding="utf-8", errors="replace") as fh:
                html = fh.read()
            rel_dir = os.path.dirname(path.replace("\\", "/").lstrip("/"))

            def _resolve(ref: str) -> str:
                ref = ref.split("#")[0].split("?")[0]
                return _safe_path(userid, os.path.normpath(os.path.join(rel_dir, ref)).replace(os.sep, "/"))

            def _exists(ref: str) -> bool:
                try:
                    return os.path.exists(_resolve(ref))
                except Exception:
                    return False

            def _read(ref: str) -> Optional[str]:
                try:
                    with open(_resolve(ref), "r", encoding="utf-8", errors="replace") as fh:
                        return fh.read()
                except Exception:
                    return None

            static_errors, static_warns = _static_html(html, _exists)
            pm = _pm()
            doc, more = _html_to_pdf_doc(pm, _inline_local_css(html, _read), os.path.dirname(full))
            pages_total = doc.page_count
            images = [(f"screen {n}", b) for n, b in _render_pdf_pages(pm, doc)]
            doc.close()
            if more:
                truncated_note = f"page is longer than {MAX_STORY_PAGES} screens; only the first {MAX_VIEW_PAGES} were viewed"
            limits = ["JavaScript was not run", "flex/grid layout and media queries are not drawn in the preview"]

        else:  # image
            static_errors, static_warns = [], []
            with open(full, "rb") as fh:
                images = [("image", fh.read())]
            pages_total = 1

        pages = []
        for label, data in images:
            verdict = _look(kind, brief, label, _to_data_url(data))
            pages.append({
                "page": label,
                "ok": None if verdict is None else verdict["ok"],
                "issues": [] if verdict is None else verdict["issues"],
            })

        visual_issues = [f"{p['page']}: {i}" for p in pages if p["ok"] is False for i in p["issues"]]
        unchecked = [p["page"] for p in pages if p["ok"] is None]
        blocking = list(static_errors) + visual_issues

        if blocking:
            status = "needs_fixes"
        elif unchecked or not pages:
            status = "not_visually_verified"
        else:
            status = "passed"

        res = {
            "success": True, "kind": kind, "status": status, "passed": status == "passed",
            "attempt": attempt, "pages_viewed": len(pages) - len(unchecked), "pages_total": pages_total,
            "issues": blocking[:12], "warnings": list(static_warns)[:5],
            "pages": pages,
        }
        if unchecked:
            res["not_viewed"] = unchecked
        if truncated_note:
            res["note"] = truncated_note
        if limits:
            res["not_tested"] = limits

        if status == "needs_fixes":
            if attempt >= MAX_ROUNDS:
                res["stop_fixing"] = True
                res["next_step"] = ("Round limit reached. Do NOT fix again. Deliver the best version and tell the user, in plain "
                                    "words, each problem in `issues` that is still there.")
            else:
                res["next_step"] = ("Fix ONLY the listed issues with workspace_edit (do not rewrite the file), then call "
                                    f"design_check again. This was round {attempt} of {MAX_ROUNDS}.")
        elif status == "not_visually_verified":
            res["next_step"] = ("The visual reviewer could not look at the pages. Static checks found no problems. Do not loop. "
                                "Deliver, and tell the user the look was NOT visually verified.")
        else:
            res["next_step"] = ("Checks passed. Deliver now. Report only what was checked (pages viewed, issues fixed) and what "
                                "was not tested; never say 'looks perfect'.")
        return res
    except Exception as e:
        print(f"[DESIGN_CHECK] failed: {type(e).__name__}: {e}")
        return {"success": False, "error": f"design_check failed: {type(e).__name__}: {str(e)[:200]}"}


# ── self-test (run on the server: python3 gpt2_design_check.py --selftest) ──
def _selftest() -> None:
    def line(name, ok, detail=""):
        print(f"[{'OK ' if ok else 'NO '}] {name}" + (f" - {detail}" if detail else ""))

    try:
        pm = _pm()
        doc = pm.open()
        page = doc.new_page()
        page.insert_text((72, 100), "Hello design check", fontsize=22)
        pngs = _render_pdf_pages(pm, doc)
        line("PDF -> PNG", bool(pngs) and len(pngs[0][1]) > 500, f"{len(pngs[0][1])} bytes")
        errs, warns = _static_pdf(doc)
        line("PDF static checks", True, f"errors={errs} warnings={warns}")
        line("data URL for vision", _to_data_url(pngs[0][1]).startswith("data:image/"))
    except Exception as e:
        line("PDF render", False, f"{type(e).__name__}: {e}")
        return
    try:
        html = ("<html><head><title>t</title><meta name='viewport' content='width=device-width,initial-scale=1'>"
                "<style>*{box-sizing:border-box}h1{color:#c0392b}</style></head><body><h1>Bakery</h1>"
                "<p>Fresh bread every morning, baked in town since 1998.</p></body></html>")
        errs, warns = _static_html(html, lambda r: True)
        line("HTML static checks", not errs, f"errors={errs} warnings={warns}")
        d2, more = _html_to_pdf_doc(pm, html)
        shots = _render_pdf_pages(pm, d2)
        line("HTML -> PNG (no browser)", bool(shots) and len(shots[0][1]) > 500, f"{d2.page_count} page(s), more={more}")
    except Exception as e:
        line("HTML render", False, f"{type(e).__name__}: {e}")
    print("done")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
