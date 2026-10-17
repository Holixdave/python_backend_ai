"""
publish_pages.py — publishes an HTML page the AI wrote to GitHub Pages.

Setup (once, ~5 minutes — see PUBLISH_SETUP.md):
  1. Make a NEW public GitHub account for OOOR, then a public repo, e.g. "ooor-pages".
  2. Repo -> Settings -> Pages -> Source: "Deploy from a branch", branch main, folder /(root).
  3. Account -> Settings -> Developer settings -> Fine-grained token: access to ONLY that repo,
     permission "Contents: Read and write".
  4. Set these env vars on Render:
        PAGES_GITHUB_TOKEN = <the token>
        PAGES_GITHUB_REPO  = <account>/ooor-pages
        PAGES_BASE_URL     = https://<account>.github.io/ooor-pages     (optional; derived if unset)
"""
import base64
import os
import re
import secrets
import time

import requests

_BANNER = (
    '<div style="position:fixed;bottom:0;left:0;right:0;z-index:99999;background:#111;color:#ddd;'
    'font:12px/1.4 system-ui,sans-serif;padding:6px 10px;text-align:center">'
    "Published on the OOOR public page host. It works now, but please move it to your own domain."
    "</div>"
)


def _config():
    token = os.environ.get("PAGES_GITHUB_TOKEN", "").strip()
    repo = os.environ.get("PAGES_GITHUB_REPO", "").strip()
    if not token or not repo or "/" not in repo:
        return None
    owner, name = repo.split("/", 1)
    base = os.environ.get("PAGES_BASE_URL", "").strip().rstrip("/") or f"https://{owner}.github.io/{name}"
    return token, repo, base


def publish_page(html: str, title: str = "page") -> dict:
    """Upload one self-contained HTML page and return its public URL."""
    cfg = _config()
    if not cfg:
        return {"success": False, "error": "Publishing is not set up on the server yet (PAGES_GITHUB_TOKEN / PAGES_GITHUB_REPO missing)."}
    if not isinstance(html, str) or len(html.strip()) < 20:
        return {"success": False, "error": "html is empty."}
    if len(html) > 900_000:
        return {"success": False, "error": "Page is too large to publish (limit ~900 KB). Trim embedded assets."}
    token, repo, base = cfg

    slug = re.sub(r"[^a-z0-9]+", "-", (title or "page").lower()).strip("-")[:32] or "page"
    page_id = f"{slug}-{secrets.token_hex(3)}"
    path = f"p/{page_id}/index.html"
    body = html if "</body>" not in html.lower() else re.sub(r"(?i)</body>", _BANNER + "</body>", html, count=1)

    r = requests.put(
        f"https://api.github.com/repos/{repo}/contents/{path}",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        json={"message": f"publish {page_id}", "content": base64.b64encode(body.encode("utf-8")).decode("ascii")},
        timeout=30,
    )
    if r.status_code not in (200, 201):
        return {"success": False, "error": f"GitHub said {r.status_code}: {r.text[:200]}"}
    return {
        "success": True,
        "url": f"{base}/p/{page_id}/",
        "title": title,
        "note": "GitHub Pages can take up to a minute to go live the first time. Tell the user the link and that it may need a moment.",
        "published_at": int(time.time()),
    }
