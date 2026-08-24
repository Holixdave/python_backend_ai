#!/usr/bin/env python3
# gpt2_functions.py — helper/tool functions used by gpt2_test.py
# ─────────────────────────────────────────────────────────────────────────────
# Split out of gpt2_test.py to keep that file to just config + the 3 public
# entry points (ask_gpt2, ask_gpt2_stream, _ask_gpt2_core). Everything below
# is pure content moved verbatim — no logic was changed in this split.
#
# Contains: web search chain, image search chain, intent classifier,
# reasoning/formatting helpers, provider-chain callers, vision helper,
# unsure-answer detection, and the file-build tool.
# ─────────────────────────────────────────────────────────────────────────────

import os
import re
import time
import json
import random
import requests
from typing import Optional
from user_doc_manager import UserDocManager
from bs4 import BeautifulSoup
# Pulled back from gpt2_test.py — see the note over the matching import
# there for why this is safe despite being circular.
from gpt2_test import (
    GROQ_API_KEY,
    BRAVE_API_KEY,
    TAVILY_API_KEY,
    TEXT_PROVIDERS,
    VISION_PROVIDERS,
    MAX_RETRIES_PER_PROVIDER,
    RETRY_BASE_DELAY,
    REQUEST_TIMEOUT,
    REASONING_STEP_ICONS,
)


from prompts import INTENT_SYSTEM_PROMPT
import firebase_config  # noqa: F401 (ensures firebase_admin is initialized)
from firebase_admin import firestore

# ---------------------------------------------------------------------------
# WEB SEARCH — multi-engine fallback chain (ddgs -> Brave -> Tavily)
# ---------------------------------------------------------------------------
SEARCH_TRIGGER_KEYWORDS = [
    "search", "find", "look up", "look for", "link", "download",
    "latest", "recent", "news", "where can i", "netnaija", "website",
    "what is the price of", "current", "today", "2025", "2026",
    "who won", "result", "show me", "get me",
]

def needs_web_search(prompt: str) -> bool:
    t = prompt.lower()
    return any(k in t for k in SEARCH_TRIGGER_KEYWORDS)

def build_search_query(prompt: str) -> str:
    replacements = [
        "can i get", "can you get", "can you find", "find me",
        "search for", "look for", "get me", "show me", "i want",
        "please", "dude", "man", "kindly", "help me",
        "abeg", "biko", "pls", "plz", "sha", "na so", "una",
    ]
    q = prompt.lower()
    for r in replacements:
        q = q.replace(r, "")
    return q.strip()


def _search_ddgs(query: str, max_results: int):
    from ddgs import DDGS
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    return [
        {"title": r.get("title", "N/A"), "href": r.get("href", ""), "body": r.get("body", "")}
        for r in results
    ]


def _search_brave(query: str, max_results: int):
    if not BRAVE_API_KEY:
        return None
    resp = requests.get(
        "https://api.search.brave.com/res/v1/web/search",
        headers={"Accept": "application/json", "X-Subscription-Token": BRAVE_API_KEY},
        params={"q": query, "count": max_results},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    items = data.get("web", {}).get("results", [])[:max_results]
    return [
        {"title": it.get("title", "N/A"), "href": it.get("url", ""), "body": it.get("description", "")}
        for it in items
    ]


def _search_tavily(query: str, max_results: int):
    if not TAVILY_API_KEY:
        return None
    resp = requests.post(
        "https://api.tavily.com/search",
        json={"api_key": TAVILY_API_KEY, "query": query, "max_results": max_results},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    items = data.get("results", [])[:max_results]
    return [
        {"title": it.get("title", "N/A"), "href": it.get("url", ""), "body": it.get("content", "")}
        for it in items
    ]


def _search_ddgs_images(query: str, max_results: int = 200):
    """
    Fetch up to 200 image candidates from DDGS.
    These are only candidates—they must still be verified by the vision model.
    """
    from ddgs import DDGS

    seen = set()
    images = []

    with DDGS() as ddgs:
        for r in ddgs.images(query, max_results=max_results):
            url = r.get("image")
            if not url or url in seen:
                continue

            seen.add(url)

            images.append({
                "image": url,
                "thumbnail": r.get("thumbnail") or url,
                "title": r.get("title", ""),
                "source": r.get("url") or r.get("source", ""),
            })

    return images

_IMAGE_QUERY_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "for", "and", "or", "to", "with",
    "is", "are", "photo", "photos", "image", "images", "picture", "pictures",
}


def _image_candidate_score(query: str, candidate: dict) -> int:
    """
    Cheap, free relevance pre-check using only text metadata (title + the
    source URL) — no network or vision call needed. Runs BEFORE the
    expensive vision-based _verify_image_relevance, so candidates whose own
    title/URL actually mention the subject get checked first, and results
    with no textual connection at all (e.g. a Maldives-beach travel photo
    for a "laptop" query) sink to the back of the queue instead of eating
    one of verify_image_relevance's limited checks before anything good
    gets a chance. This doesn't throw anything away — a real match can
    still have an unhelpful title — it just reorders so the most promising
    candidates are checked first.
    """
    query_words = {
        w for w in re.findall(r"[a-z0-9]+", query.lower())
        if w not in _IMAGE_QUERY_STOPWORDS and len(w) > 2
    }
    if not query_words:
        return 0
    haystack = f"{candidate.get('title', '')} {candidate.get('source', '')}".lower()
    return sum(1 for w in query_words if w in haystack)


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
SERPERAPI_KEY = os.getenv("SERPERAPI_KEY")  # serpapi.com
SERPER_API_KEY = os.getenv("SERPER_API_KEY")  # serper.dev

def _search_google_images(query: str, max_results: int):
    if not (GOOGLE_API_KEY and GOOGLE_CSE_ID):
        return None
    resp = requests.get(
        "https://www.googleapis.com/customsearch/v1",
        params={
            "key": GOOGLE_API_KEY, "cx": GOOGLE_CSE_ID,
            "q": query, "searchType": "image", "num": min(max_results, 10),
        },
        timeout=10,
    )
    resp.raise_for_status()
    items = resp.json().get("items", [])
    return [
        {"image": it["link"], "thumbnail": it.get("image", {}).get("thumbnailLink") or it["link"],
         "title": it.get("title", ""), "source": it.get("image", {}).get("contextLink", "")}
        for it in items
    ] or None

def _search_serpapi_images(query: str, max_results: int):
    if not SERPERAPI_KEY:
        return None
    resp = requests.get(
        "https://serpapi.com/search",
        params={"engine": "google_images", "q": query, "api_key": SERPERAPI_KEY, "num": max_results},
        timeout=10,
    )
    resp.raise_for_status()
    items = resp.json().get("images_results", [])[:max_results]
    return [
        {"image": it.get("original"), "thumbnail": it.get("thumbnail") or it.get("original"),
         "title": it.get("title", ""), "source": it.get("link", "")}
        for it in items
    ] or None

def _search_serper_images(query: str, max_results: int):
    if not SERPER_API_KEY:
        return None
    resp = requests.post(
        "https://google.serper.dev/images",
        headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
        json={"q": query, "num": max_results},
        timeout=10,
    )
    resp.raise_for_status()
    items = resp.json().get("images", [])[:max_results]
    return [
        {"image": it.get("imageUrl"), "thumbnail": it.get("thumbnailUrl") or it.get("imageUrl"),
         "title": it.get("title", ""), "source": it.get("link", "")}
        for it in items
    ] or None

def search_images(query: str, max_results: int = 20):
    for engine_name, engine_fn in (
        ("google", _search_google_images),
        ("serpapi", _search_serpapi_images),
        ("serper", _search_serper_images),
        ("ddgs", lambda q, n: _search_ddgs_images(q, n)),
    ):
        try:
            candidates = engine_fn(query, max_results)
        except Exception as e:
            print(f"[IMAGE SEARCH] {engine_name} failed: {e}")
            continue
        if candidates:
            print(f"[IMAGE SEARCH] succeeded via {engine_name} ({len(candidates)} results)")
            candidates.sort(key=lambda c: _image_candidate_score(query, c), reverse=True)
            return candidates
    print("[IMAGE SEARCH] all engines failed")
    return []


def _fetch_og_image(url: str) -> Optional[str]:
    """
    Pulls a page's own declared preview image (og:image meta tag) — the
    page author's explicit answer to "what image represents this", which
    is a far stronger relevance signal than a blind keyword-matched image
    search. Never raises; a failure here just means falling back to
    search_images() instead.
    """
    if not url:
        return None
    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        match = re.search(
            r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
            resp.text, re.IGNORECASE,
        )
        if match:
            return match.group(1)
    except Exception as e:
        print(f"[OG_IMAGE] failed for {url}: {e}")
    return None


GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

def search_github(query: str, kind: str = "repositories", max_results: int = 5):
    """
    Search GitHub for repositories or code matching a query.
    kind: "repositories" or "code". Returns a list of dicts.
    """
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    endpoint = "code" if kind == "code" else "repositories"
    resp = requests.get(
        f"https://api.github.com/search/{endpoint}",
        params={"q": query, "per_page": max_results},
        headers=headers,
        timeout=10,
    )
    resp.raise_for_status()
    items = resp.json().get("items", [])

    if kind == "code":
        return [
            {"path": i["path"], "repo": i["repository"]["full_name"], "url": i["html_url"]}
            for i in items
        ]
    return [
        {"name": i["full_name"], "description": i.get("description", ""),
         "stars": i["stargazers_count"], "url": i["html_url"]}
        for i in items
    ]


def fetch_github_file(repo: str, path: str, ref: str = "main") -> str:
    """
    Fetch a single file's raw text content from a GitHub repo.
    repo: "owner/name", path: file path in repo, ref: branch/commit/tag.
    """
    headers = {"Accept": "application/vnd.github.raw+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    resp = requests.get(
        f"https://api.github.com/repos/{repo}/contents/{path}",
        params={"ref": ref}, headers=headers, timeout=10,
    )
    resp.raise_for_status()
    return resp.text


def fetch_document(url: str, max_chars: int = 15000) -> str:
    """
    Downloads a PDF or Word (.docx) document from a URL and extracts its
    text content. Returns the extracted text, truncated to max_chars.
    Raises a clear error if the file type isn't supported or download fails.
    """
    resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "").lower()

    is_pdf = "pdf" in content_type or url.lower().endswith(".pdf")
    is_docx = "wordprocessingml" in content_type or url.lower().endswith(".docx")

    if is_pdf:
        import io
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(resp.content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)

    elif is_docx:
        import io
        from docx import Document
        doc = Document(io.BytesIO(resp.content))
        text = "\n".join(p.text for p in doc.paragraphs)

    else:
        raise ValueError(f"Unsupported document type for URL: {url} (content-type: {content_type})")

    text = text.strip()
    if not text:
        return "Document was fetched but no extractable text was found (it may be a scanned/image-based file)."
    return text[:max_chars]


# ---------------------------------------------------------------------------
# USER IDENTITY / MEMORY — reuses the same firebase_config init and
# users/{userid} Firestore structure already established in
# firestore_repository.py (chat history lives at
# users/{userid}/utme26_messages/{auto_id}). These tools read the user's
# own profile doc directly, and add a NEW subcollection
# users/{userid}/ai_notes/{auto_id} for facts the AI jots down about the
# user across conversations — separate from raw chat history, deliberately
# small/curated rather than a full transcript.
# ---------------------------------------------------------------------------

def get_user_profile(userid: str) -> dict:
    """
    Fetch this user's display name from Firestore (users/{userid} doc).
    Checks 'name', 'displayName', 'username' fields in that order —
    whichever is actually set on the account. Returns
    {"name": str|None, "found": bool}. Never raises; a Firestore hiccup
    just means the AI doesn't know the user's name this turn, not a
    broken request.
    """
    if not userid:
        return {"name": None, "found": False}
    try:
        doc = firestore.client().collection("users").document(userid).get()
        if not doc.exists:
            return {"name": None, "found": False}
        data = doc.to_dict() or {}
        name = data.get("name") or data.get("displayName") or data.get("username")
        return {"name": name, "found": bool(name)}
    except Exception as e:
        print(f"[USER_PROFILE] failed for userid={userid!r}: {e}")
        return {"name": None, "found": False}


def save_user_note(userid: str, note: str) -> bool:
    """
    Jot down a fact/preference about the user for later recall, stored at
    users/{userid}/ai_notes/{auto_id}. Call this when the user shares
    something worth remembering across conversations (their goals,
    preferences, ongoing projects) — not for routine chat content, which
    is already saved separately by remember_turn(). Never raises; returns
    False on failure so the AI can tell the user it couldn't save it.
    """
    if not userid or not note:
        return False
    try:
        firestore.client().collection("users").document(userid).collection("ai_notes").add({
            "note": note,
            "created_at": firestore.SERVER_TIMESTAMP,
        })
        return True
    except Exception as e:
        print(f"[USER_NOTES] save failed for userid={userid!r}: {e}")
        return False


def get_user_notes(userid: str, limit: int = 20) -> list:
    """
    Retrieve previously saved notes about this user, oldest first. Never
    raises; returns [] on any failure or if nothing's been saved yet.
    """
    if not userid:
        return []
    try:
        docs = (
            firestore.client()
            .collection("users").document(userid)
            .collection("ai_notes")
            .order_by("created_at", direction=firestore.Query.DESCENDING)
            .limit(limit)
            .stream()
        )
        notes = [d.to_dict().get("note") for d in docs if d.to_dict().get("note")]
        notes.reverse()
        return notes
    except Exception as e:
        print(f"[USER_NOTES] fetch failed for userid={userid!r}: {e}")
        return []


def save_study_note(userid: str, content: str, title: str = None) -> bool:
    """
    Save a drafted question/note to the user's JAMB study notebook, stored
    at users/{userid}/study_notes/{auto_id}. Never raises; returns False
    on failure so the AI can tell the user it couldn't save it.
    """
    if not userid or not content:
        return False
    try:
        firestore.client().collection("users").document(userid).collection("study_notes").add({
            "content": content,
            "title": title,
            "timestamp": firestore.SERVER_TIMESTAMP,
        })
        return True
    except Exception as e:
        print(f"[STUDY_NOTES] save failed for userid={userid!r}: {e}")
        return False


def get_study_notes(userid: str, limit: int = 20) -> list:
    """
    Retrieve the user's saved study notes/questions, newest first — so the
    AI can look them up when the user says "check my study notes" or
    "help me solve the questions I saved". Never raises; returns [] on
    any failure or if nothing's saved yet.
    """
    if not userid:
        return []
    try:
        docs = (
            firestore.client()
            .collection("users").document(userid)
            .collection("study_notes")
            .order_by("created_at", direction=firestore.Query.DESCENDING)
            .limit(limit)
            .stream()
        )
        return [
            {"content": d.to_dict().get("content"), "subject": d.to_dict().get("subject")}
            for d in docs if d.to_dict().get("content")
        ]
    except Exception as e:
        print(f"[STUDY_NOTES] fetch failed for userid={userid!r}: {e}")
        return []


def schedule_reminder(userid: str, message: str, send_at: str) -> dict:
    """
    Schedule a push-notification reminder for the user at a specific time.
    send_at MUST be a full ISO 8601 datetime string (e.g.
    "2026-08-07T14:00:00") — resolve any relative time ("by 2", "tomorrow
    morning") to a real date/time before calling this. Treated as UTC if
    no timezone is included.

    This only WRITES the reminder to Firestore's top-level
    scheduled_reminders collection — it does NOT send anything itself.
    The separate notifier service (zindryx-notifier, always-on) polls
    that collection every ~60s and delivers any reminder whose send_at
    has passed, using the user's fcmToken stored on their users/{userid}
    doc. Never raises; returns {"success": bool, "error": str|None}.
    """
    if not userid or not message or not send_at:
        return {"success": False, "error": "Missing userid, message, or send_at."}
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(send_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        firestore.client().collection("scheduled_reminders").add({
            "userid": userid,
            "message": message,
            "send_at": dt,
            "sent": False,
            "created_at": firestore.SERVER_TIMESTAMP,
        })
        return {"success": True, "error": None}
    except Exception as e:
        print(f"[SCHEDULE_REMINDER] failed for userid={userid!r}: {e}")
        return {"success": False, "error": str(e)}


def redisplay_images(images: list) -> list:
    """
    Re-renders a real gallery from images the AI already found earlier in
    THIS conversation — for when the user explicitly says something like
    "don't search again, just show me what you found before." Does NOT
    hit any search engine or the internet at all; it only reshapes
    whatever list of image dicts the AI recalls (e.g. from its own
    [INTERNAL MEMORY NOTE] in earlier history — see main.py's
    _build_memory_reply) into the exact same shape search_images returns,
    so gpt2_test.py's gallery-population logic renders it identically.

    images: list of dicts, each needs at minimum {"image": <url>}.
    "title", "thumbnail", "source" are optional and passed through as-is.
    Malformed/missing-url entries are silently skipped rather than
    raising, since a partial gallery beats a hard failure.
    """
    out = []
    for item in images or []:
        if not isinstance(item, dict):
            continue
        url = item.get("image") or item.get("url")
        if not url:
            continue
        out.append({
            "image": url,
            "thumbnail": item.get("thumbnail") or url,
            "title": item.get("title", ""),
            "source": item.get("source", ""),
        })
    return out


def generate_image(prompt: str, width: int = 1024, height: int = 1024) -> list:
    """
    Generates a brand-new AI image from a text prompt and returns it in
    the EXACT same list-of-dicts shape search_images does — so it flows
    through the identical gallery-rendering pipeline in gpt2_test.py with
    zero frontend changes needed. This is for genuinely new/imaginary
    images (see IMAGE_GEN_AWARENESS in prompts.py); use search_images
    instead for real photos of things that already exist.

    Uses Pollinations.ai — free, no API key required, returns a direct
    image URL (the URL itself generates the image on first fetch, no
    separate upload/hosting step needed). A prompt-derived seed keeps
    repeat calls with the same exact prompt visually consistent rather
    than random each time.
    """
    import urllib.parse
    if not prompt:
        return []

    # Defensive cleanup: the model has been observed passing along leaked
    # instructional preamble (e.g. a bracketed "[ADVANCED THINKING ENABLED
    # - ...]" system/persona block that precedes the real user message in
    # some setups) as if it were part of the visual description, instead
    # of extracting just the actual thing to draw. A prompt like that
    # produces unrelated, garbage generations — Pollinations tries to
    # render literal instruction text. Strip any long leading bracketed
    # block and hard-cap length; a real image prompt doesn't need to be
    # this long, and this alone fixes it regardless of where the leak is
    # actually coming from upstream.
    prompt = re.sub(r'^\s*\[[^\]]{20,}\]\s*', '', prompt).strip()
    prompt = prompt[:300]
    if not prompt:
        return []

    encoded = urllib.parse.quote(prompt)
    seed = abs(hash(prompt)) % 1_000_000
    url = (
        f"https://image.pollinations.ai/prompt/{encoded}"
        f"?width={width}&height={height}&seed={seed}&nologo=true"
    )
    return [{
        "image": url,
        "thumbnail": url,
        "title": prompt[:80],
        "source": "AI-generated",
    }]


def _call_provider_chain(providers: list, messages: list, temperature: float, max_tokens: int, reasoning_effort: str = None):
    """
    Walks `providers` in order. For each enabled provider: retries a couple
    times on 429 (rate limit), but moves to the next provider immediately on
    any other failure (bad key, out of credits, network error, etc.) instead
    of burning time/retries on a dead provider.

    reasoning_effort ("default" or "none") is only ever sent to a provider
    whose config sets supports_reasoning_effort — currently just Qwen 3.6
    27B on Groq. Every other provider ignores this parameter entirely so
    passing it never breaks a non-Qwen call.

    Returns (content, provider_name) on success, or (None, None) if every
    provider in the chain failed.
    """
    last_error = "No provider available."
    print(f"[AI] provider chain starting — {len(providers)} configured, "
          f"temperature={temperature}, max_tokens={max_tokens}, reasoning_effort={reasoning_effort}")

    for provider in providers:
        if not provider["enabled"]:
            print(f"[AI] skipping {provider['name']} — no API key configured")
            continue

        payload = {
            "model": provider["model"],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if reasoning_effort is not None and provider.get("supports_reasoning_effort"):
            payload["reasoning_effort"] = reasoning_effort
            # FIXED — without this, Groq's reasoning-capable models default
            # to putting reasoning in a separate `message.reasoning` field
            # instead of inlining it as <think>...</think> tags in
            # `message.content`. _split_thinking() only ever looks inside
            # `content`, so the reasoning was very likely being generated
            # the whole time and just landing somewhere this code never
            # read — no error, nothing to catch, it just silently never
            # showed up as thinking steps on the frontend.
            payload["reasoning_format"] = "raw"

        print(f"[AI] trying {provider['name']} ({provider['model']})...")

        for attempt in range(1, MAX_RETRIES_PER_PROVIDER + 1):
            try:
                response = requests.post(
                    provider["url"],
                    headers=provider["headers"],
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                )
                if response.status_code == 200:
                    result = response.json()
                    content = result.get("choices", [{}])[0].get("message", {}).get("content")
                    if content and _is_degenerate_output(content):
                        last_error = f"{provider['name']}: degenerate/looping output"
                        print(f"[AI] {provider['name']} returned 200 but content is a repetition loop — trying next provider")
                        break  # try next provider
                    if content:
                        print(f"[AI] answered via {provider['name']} ({len(content)} chars)")
                        return content, provider["name"]
                    last_error = f"{provider['name']}: empty content"
                    print(f"[AI] {provider['name']} returned 200 but empty content — trying next provider")
                    break  # try next provider

                if response.status_code == 429:
                    # Rate limited — worth a quick retry before giving up on this provider
                    if attempt < MAX_RETRIES_PER_PROVIDER:
                        print(f"[AI] {provider['name']} rate limited (429), retry {attempt}/{MAX_RETRIES_PER_PROVIDER}")
                        time.sleep(RETRY_BASE_DELAY * attempt)
                        continue
                    last_error = f"{provider['name']}: rate limited (429)"
                    print(f"[AI] {last_error} — giving up on this provider")
                    break

                # Any other status (401 bad key, 402 out of credit, 404 model
                # gone, 500, etc.) — this provider is down, move on now.
                last_error = f"{provider['name']}: HTTP {response.status_code} — {response.text[:150]}"
                print(f"[AI] {last_error}")
                break

            except requests.exceptions.RequestException as e:
                last_error = f"{provider['name']}: {e}"
                if attempt < MAX_RETRIES_PER_PROVIDER:
                    print(f"[AI] {last_error} — retry {attempt}/{MAX_RETRIES_PER_PROVIDER}")
                    time.sleep(RETRY_BASE_DELAY * attempt)
                    continue
                print(f"[AI] {last_error}")
                break

    print(f"[AI] all providers exhausted — last error: {last_error}")
    return None, None

def _verify_image_relevance(
    query: str,
    prompt: str,
    image_results: list,
    max_verified: int = 12,
):
    """
    Verify DDGS image candidates using the vision model.
    Scans through all candidates until enough verified images are found.
    """

    verified = []

    verify_system = (
        "You are a strict image relevance checker. "
        "Reply with EXACTLY one word: YES or NO. "
        "YES only if the image clearly matches the requested subject."
    )

    for candidate in image_results:

        if len(verified) >= max_verified:
            break

        image_url = candidate.get("image")
        if not image_url:
            continue

        messages = [
            {"role": "system", "content": verify_system},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f'Does this image genuinely show "{query}"? Reply YES or NO only.'
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            },
        ]

        for provider in VISION_PROVIDERS:
            if not provider["enabled"]:
                continue

            answer, _ = _call_provider_chain(
                [provider],
                messages,
                temperature=0.0,
                max_tokens=10,
            )

            if answer:
                if answer.strip().upper().startswith("YES"):
                    verified.append(candidate)
                break

    return verified
def search_web(query: str, max_results: int = 20):
    """
    Tries each search engine in order. Returns (formatted_text, sources).
    sources is a list of {"title": str, "url": str} for the frontend to render.

    ORDER CHANGED: Brave and Tavily (real APIs) now go first — ddgs (free,
    unofficial DuckDuckGo scraper) goes LAST. ddgs frequently gets
    rate-limited/soft-blocked on server IPs and silently returns generic,
    low-relevance junk instead of raising an error, which used to make this
    function think the search "succeeded" and stop before ever trying the
    real APIs.

    Also added: a relevance check. If NONE of the query's real keywords
    appear anywhere in a given engine's results, that engine's output is
    treated as junk and we move to the next engine instead of trusting it.

    On total failure, returns ("", []) silently — no raw error text gets
    injected into the model's context or shown to the user.
    """
    print(f"[SEARCH TRIGGERED] Query: {query}")

    query_words = set(re.findall(r"[a-z0-9]+", query.lower()))
    query_words = {w for w in query_words if len(w) > 2}  # drop tiny stopword-ish tokens

    for engine_name, engine_fn in (
        ("brave", _search_brave),
        ("tavily", _search_tavily),
        ("ddgs", _search_ddgs),  # unreliable on server IPs — last resort only
    ):
        try:
            results = engine_fn(query, max_results)
        except ImportError:
            print(f"[SEARCH] {engine_name} not installed, skipping")
            continue
        except Exception as e:
            print(f"[SEARCH] {engine_name} failed: {e}")
            continue

        if not results:
            continue

        # Relevance check — reject results that don't actually match the
        # query. This is what catches ddgs's silent junk-fallback behavior
        # (e.g. "JAMB registration" returning door-jamb architecture pages).
        if query_words:
            hits = 0
            for r in results:
                haystack = f"{r.get('title', '')} {r.get('body', '')}".lower()
                haystack_words = set(re.findall(r"[a-z0-9]+", haystack))
                if query_words & haystack_words:
                    hits += 1
            if hits == 0:
                print(f"[SEARCH] {engine_name} results don't match query keywords "
                      f"({query_words}) — treating as junk, trying next engine")
                continue

        formatted = ""
        sources = []
        for i, r in enumerate(results, 1):
            formatted += (
                f"{i}. Title: {r['title']}\n"
                f"   Link: {r['href']}\n"
                f"   Summary: {r['body']}\n\n"
            )
            if r["href"]:
                sources.append({"title": r["title"], "url": r["href"]})

        print(f"[SEARCH] succeeded via {engine_name} ({len(results)} results)")
        return formatted.strip(), sources

    print("[SEARCH] all engines failed, were unavailable, or returned irrelevant junk")
    return "", []
def fetch_webpage(url: str, max_chars: int = 6000) -> str:
    """
    Fetches a webpage by URL and returns its main readable text content.
    Never raises — returns an error string instead, matching the fail-quiet
    pattern the rest of this file uses (og_image, search_web, etc).
    """
    if not url:
        return "No URL provided."
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split())
        return text[:max_chars]
    except Exception as e:
        print(f"[FETCH_WEBPAGE] failed for {url}: {e}")
        return f"Couldn't load that page ({e})."
# ---------------------------------------------------------------------------
# INTENT CLASSIFIER — one small, cheap call per prompt.
#
# This replaces keyword-matching as the primary decision-maker. It gets
# ONLY the user's raw prompt (no backend system prompt, no rules) and
# returns a tiny JSON object. Everything else in ask_gpt2() reads off this
# result instead of scanning the prompt for keywords itself.
#
# If the classifier call fails for any reason (network, bad JSON, rate
# limit) we silently fall back to the old keyword heuristics below — the
# user never sees an error, they just get the slightly-dumber-but-safe
# routing instead.
# ---------------------------------------------------------------------------
INTENT_MODEL = "llama-3.1-8b-instant"

CODING_KEYWORDS = [
    "code", "write", "build", "create", "implement", "function",
    "class", "widget", "dart", "flutter", "python", "javascript",
    "fix", "debug", "error", "screen", "app", "file",
]

# NOTE: the intent-classifier system prompt used to be defined right here
# as _INTENT_SYSTEM_PROMPT — it now lives in prompts.py as
# INTENT_SYSTEM_PROMPT (imported above) so every prompt is in one file.


def _fallback_intent(prompt: str) -> dict:
    t = prompt.lower()
    if any(k in t for k in ["jamb", "utme", "zindryx", "waec exam", "post utme"]):
        topic = "jamb"
    elif any(k in t for k in ["mojizela", "coin price", "buy coins", "wallet icon", "tiktok creator"]):
        topic = "mojizela"
    else:
        topic = "general"
    
    # Check for user_docs intent (remember, do you have, check my files, etc.)
    user_docs_keywords = ["remember", "do you have", "check my", "my files", "my previous", "my doc", "from my"]
    is_user_docs = any(keyword in t for keyword in user_docs_keywords)
    
    # Check for web search need
    is_web = needs_web_search(prompt) and not is_user_docs
    
    search_type = "user_docs" if is_user_docs else ("web" if is_web else "none")
    search_query = ""
    if is_web:
        search_query = build_search_query(prompt)
    elif is_user_docs:
        search_query = prompt  # use raw prompt as hint for user docs search

    return {
        "search_type": search_type,
        "search_query": search_query,
        "complex": any(k in t for k in CODING_KEYWORDS),
        "topic": topic,
    }


def classify_intent(prompt: str, history: Optional[list] = None) -> dict:
    """
    Single cheap call that decides: does this need a web search (and what
    to actually search for, resolved against recent context), does it need
    the deep/complex track, and which knowledge-base topic (if any)
    applies. Sees a short window of recent history so it can resolve vague
    references ("the damn church", "that place") to the real proper noun
    — no backend rules leak into this call, so it stays fast and cheap.
    """
    context_lines = []
    for msg in (history or [])[-6:]:
        if isinstance(msg.get("content"), str):
            role = "User" if msg.get("role") == "user" else "Assistant"
            context_lines.append(f"{role}: {msg['content'][:400]}")
    context_block = "\n".join(context_lines)

    user_payload = (
        (f"CONVERSATION SO FAR:\n{context_block}\n\n" if context_block else "")
        + f"NEWEST MESSAGE: {prompt}"
    )

    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": INTENT_MODEL,
                "messages": [
                    {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_payload},
                ],
                "temperature": 0.0,
                "max_tokens": 200,
            },
            timeout=10,
        )
        if resp.status_code == 200:
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
            data = json.loads(raw)
            topic = data.get("topic")
            if topic not in ("jamb", "mojizela", "general"):
                topic = "general"
            
            search_type = data.get("search_type", "none")
            if search_type not in ("web", "user_docs", "none"):
                search_type = "none"

            search_query = (data.get("search_query") or "").strip()
            if search_type in ("web", "user_docs") and not search_query:
                # classifier said yes but forgot the query — fall back to prompt
                search_query = build_search_query(prompt) if search_type == "web" else prompt

            # Safety net: if the classifier still just echoed the raw prompt
            # back, or handed back something way longer than a real search
            # query should be, run it through the keyword-stripping fallback
            # as an extra distillation pass rather than sending the user's
            # literal sentence to DDGS.
            if search_type == "web" and search_query:
                is_verbatim_echo = search_query.strip().lower() == prompt.strip().lower()
                is_too_long = len(search_query.split()) > 12
                if is_verbatim_echo or is_too_long:
                    search_query = build_search_query(search_query)
            
            return {
                "search_type": search_type,
                "search_query": search_query,
                "complex": bool(data.get("complex", True)),
                "topic": topic,
            }
        print(f"[INTENT] classifier HTTP {resp.status_code}, falling back to keywords")
        
    except Exception as e:
        print(f"[INTENT] classifier failed ({e}), falling back to keywords")

    return _fallback_intent(prompt)

# ---------------------------------------------------------------------------
# HELPERS — unchanged
# ---------------------------------------------------------------------------
def get_lean_history(history):
    """
    FIXED — this used to hard-cap history and cut individual messages down
    with a "...[Truncated]..." marker. Per direction: don't strip anything.
    memory_service.py's MEMORY_WINDOW is the real limit on how much history
    exists at all (currently 40 messages) — this function just passes all
    of it straight through, untouched, to match how the working app treats
    history (no re-trimming downstream of the DB fetch).

    Returns (history, False) — kept as a tuple for compatibility with every
    existing caller that does `get_lean_history(history)[0]` or unpacks
    both values; was_truncated is always False now since nothing here
    truncates anymore.
    """
    return history, False

# ---------------------------------------------------------------------------
# NEW — Qwen 3.6 (reasoning_effort on) returns its chain-of-thought inline
# as <think>...</think> inside the same content string, instead of a
# separate field. Left alone, that raw block leaks straight into the
# frontend's answer bubble. These two helpers pull it out and break it
# into individual steps so it can be sent as real "status" events instead
# — the chat bubble only ever gets the clean answer, and the full,
# uncut reasoning shows up step-by-step in the Thought sheet.
# ---------------------------------------------------------------------------
_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_ORPHAN_THINK_CLOSE_RE = re.compile(r"^.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_STEP_SPLIT_RE = re.compile(r"(?:^|\n)\s*\d+[\.\)]\s+")


def _split_thinking(raw):
    # type: (str) -> tuple
    """
    Strips a <think>...</think> block out of a raw model response.
    Returns (cleaned_answer, thinking_text_or_None). If there's no
    <think> block, returns the raw text unchanged and None.

    Uses Optional/Tuple-free annotations on purpose — `tuple[str, str | None]`
    is 3.10+ only syntax and will crash the whole module on import for
    older Python runtimes, taking every endpoint down with it.
    """
    if not raw:
        return raw, None
    match = _THINK_BLOCK_RE.search(raw)
    if not match:
        # FIXED — weaker models sometimes start their answer already
        # mid-reasoning and only emit the closing "</think>" with no
        # opening tag at all. That fragment (plus whatever meta-
        # commentary came before it) used to leak straight into the
        # visible answer since it's not a complete <think>...</think>
        # pair. If we see a lone closer, treat everything up to and
        # including it as leaked reasoning debris and drop it.
        if "</think>" in raw.lower():
            stripped = _ORPHAN_THINK_CLOSE_RE.sub("", raw, count=1).strip()
            if stripped:
                return stripped, None
        return raw, None
    thinking = match.group(1).strip()
    cleaned = _THINK_BLOCK_RE.sub("", raw).strip()
    # NEW — if the model's ENTIRE response was just the <think> block
    # (nothing after it), don't hand back an empty string. Fall back to
    # the raw, unstripped text so the user at least sees something
    # instead of a blank bubble.
    if not cleaned:
        return raw.strip(), (thinking or None)
    return cleaned, (thinking or None)


def _split_into_steps(thinking):
    # type: (str) -> list
    """
    Breaks an extracted <think> block into individual step strings, full
    text, nothing cut. Expects blank-line-separated paragraphs, each ideally
    starting with a **Bold Header** (from the current REASONING_STEP_HINT),
    but falls back gracefully to numbered-list splitting for older/other
    models that don't follow the hint, and a single step if it's just one
    paragraph.
    """
    if not thinking:
        return []
    # Blank-line splitting first now — that's the shape the new
    # REASONING_STEP_HINT actually asks for (bold-headed paragraphs, not
    # numbers). Numbered-list splitting is now the fallback, for models
    # that ignore the hint and still number their steps out of habit.
    parts = [p.strip() for p in re.split(r"\n\s*\n", thinking) if p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in _STEP_SPLIT_RE.split(thinking) if p.strip()]
    return parts or [thinking.strip()]


_LEADING_BOLD_HEADER_RE = re.compile(r"^\s*\*\*(.+?)\*\*\s*:?", re.DOTALL)
_LEADING_ICON_TAG_RE = re.compile(r"^\s*\[([a-zA-Z_]+)\]\s*")


def _extract_step_icon(step_text: str, default: str = "thinking") -> tuple:
    """
    Pulls a leading [icon_name] tag off a reasoning step, per the format
    REASONING_STEP_HINT asks the model for — e.g. "[verifying] **Checking
    definition:** ...". Returns (icon, cleaned_step_text) where cleaned_step_text
    has the tag stripped so it doesn't show up twice (once as the real icon,
    once as leftover text in the detail).

    Falls back to `default` whenever there's no tag, the tag isn't one of
    REASONING_STEP_ICONS (guards against a hallucinated icon name reaching
    the frontend), or the model ignored the format entirely — this can
    never break a step's display, it just won't get a specific icon.
    """
    if not step_text:
        return default, step_text

    match = _LEADING_ICON_TAG_RE.match(step_text)
    if not match:
        return default, step_text

    icon = match.group(1).strip().lower()
    cleaned = step_text[match.end():].lstrip()
    if icon not in REASONING_STEP_ICONS:
        icon = default
    return icon, cleaned


def _derive_step_label(step_text: str, index: int) -> str:
    """
    Turns a raw chain-of-thought step into a short label for the collapsed
    row, instead of the literal "Reasoning step N". Prefers the step's own
    leading **Bold Header**, falls back to the first few words of plain
    text, and only uses a numbered fallback if there's truly nothing to
    work with.
    """
    if not step_text:
        return f"Reasoning step {index}"

    header_match = _LEADING_BOLD_HEADER_RE.match(step_text)
    if header_match:
        label = header_match.group(1).strip().rstrip(":")
        if label:
            return label

    first_line = step_text.strip().splitlines()[0]
    words = first_line.strip("*# ").split()
    if words:
        snippet = " ".join(words[:7])
        return snippet + ("…" if len(words) > 7 else "")

    return f"Reasoning step {index}"


# ---------------------------------------------------------------------------
# GENERIC OPENAI-COMPATIBLE CALLER — used by both text and vision chains
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# NEW — degenerate output guard. A model can return HTTP 200 with non-empty
# content that is actually a broken repetition loop (e.g. the same 20-40
# char chunk repeated dozens of times instead of stopping). That used to be
# accepted as a real answer since the old check only looked for "is content
# non-empty". This catches that failure mode and treats it exactly like an
# empty response — try the next provider in the chain instead of shipping
# garbage to the user.
# ---------------------------------------------------------------------------
_DEGENERATE_LOOP_RE = re.compile(r"(.{10,80}?)\1{3,}", re.DOTALL)


def _is_degenerate_output(content: str) -> bool:
    if not content or len(content) < 200:
        return False
    # Only scan the tail — loops like this build up gradually and are
    # always still looping by the end of the response.
    tail = content[-3000:]
    return bool(_DEGENERATE_LOOP_RE.search(tail))



# ---------------------------------------------------------------------------
# GENERIC OPENAI-COMPATIBLE CALLER — used by both text and vision chains
# ---------------------------------------------------------------------------


def _call_provider_chain_full(providers: list, messages: list, temperature: float, max_tokens: int, reasoning_effort: str = None):
    """
    Identical to _call_provider_chain(), except it also returns the
    provider's finish_reason ("stop" | "length" | ...). This is a separate
    function (rather than changing _call_provider_chain's return signature)
    so every existing caller that does answer, provider = _call_provider_chain(...)
    keeps working untouched. Only the file-build tool below needs the
    third value — finish_reason is the deterministic signal for "did the
    model actually finish, or did it get cut off mid-file" — no guessing,
    no second AI judging completeness, just what the provider itself reports.

    Returns (content, provider_name, finish_reason) on success,
    or (None, None, None) if every provider failed.
    """
    last_error = "No provider available."
    for provider in providers:
        if not provider["enabled"]:
            continue

        payload = {
            "model": provider["model"],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if reasoning_effort is not None and provider.get("supports_reasoning_effort"):
            payload["reasoning_effort"] = reasoning_effort
            payload["reasoning_format"] = "raw"  # same fix as _call_provider_chain — see comment there

        for attempt in range(1, MAX_RETRIES_PER_PROVIDER + 1):
            try:
                response = requests.post(
                    provider["url"], headers=provider["headers"], json=payload, timeout=REQUEST_TIMEOUT,
                )
                if response.status_code == 200:
                    result = response.json()
                    choice = result.get("choices", [{}])[0]
                    content = choice.get("message", {}).get("content")
                    finish_reason = choice.get("finish_reason")
                    if content and _is_degenerate_output(content):
                        last_error = f"{provider['name']}: degenerate/looping output"
                        print(f"[FILEBUILD] {provider['name']} returned a repetition loop — trying next provider")
                        break
                    if content:
                        print(f"[FILEBUILD] {provider['name']} answered ({len(content)} chars, finish_reason={finish_reason})")
                        return content, provider["name"], finish_reason
                    last_error = f"{provider['name']}: empty content"
                    break
                if response.status_code == 429:
                    if attempt < MAX_RETRIES_PER_PROVIDER:
                        time.sleep(RETRY_BASE_DELAY * attempt)
                        continue
                    last_error = f"{provider['name']}: rate limited (429)"
                    break
                last_error = f"{provider['name']}: HTTP {response.status_code} — {response.text[:150]}"
                break
            except requests.exceptions.RequestException as e:
                last_error = f"{provider['name']}: {e}"
                if attempt < MAX_RETRIES_PER_PROVIDER:
                    time.sleep(RETRY_BASE_DELAY * attempt)
                    continue
                break

    print(f"[FILEBUILD] all providers exhausted — {last_error}")
    return None, None, None

def _friendly_failure_message() -> str:
    """Returns a randomized, dynamic server overload or maintenance message."""
    messages = [
        "The server is currently overloaded. Please try your request again in a moment.",
        "We are performing routine maintenance to improve performance. OOOR will be back in a bit.",
        "High traffic alert. The server queue is completely full right now. Please give it a minute and try again.",
    ]
    
    return random.choice(messages)


# ---------------------------------------------------------------------------
# VISION — called when imageUrls is present
#
# FIXED (see header notes 7): previously this sent every image in one
# request to the first vision provider and trusted the HTTP 200 response
# unconditionally. If the model itself replied with something like "I'm
# unable to view the images" — which happens on multi-image requests to
# some vision models — that refusal text got returned as if it were a
# real, successful description. Now:
#   1. Each provider's answer is checked against a refusal-phrase list
#      before being accepted.
#   2. If a provider refuses, the next provider in VISION_PROVIDERS is
#      tried with the SAME full image batch.
#   3. If every provider refuses a multi-image batch, we retry by sending
#      images one at a time and merge the per-image descriptions — this
#      isolates whether the batch itself (not the provider) was the
#      problem.
# ---------------------------------------------------------------------------
_VISION_REFUSAL_PHRASES = [
    "unable to view", "can't view", "cannot view",
    "unable to see the image", "can't see the image", "cannot see the image",
    "don't have access to the image", "do not have access to the image",
    "no image was provided", "not able to view", "not able to see the image",
    "i can't process images", "i cannot process images",
]


def _looks_like_vision_refusal(answer: Optional[str]) -> bool:
    if not answer:
        return True
    a = answer.lower()
    return any(p in a for p in _VISION_REFUSAL_PHRASES)


def _vision_messages(prompt: str, image_urls: list, history: list) -> list:
    content = [{"type": "text", "text": prompt}]
    for url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})

    vision_system = (
        "You are a smart visual AI assistant. "
        "Analyse the provided image(s) carefully and answer the user's question accurately. "
        "Describe what you see in detail when asked. "
        "If asked to read text in an image, transcribe it exactly. "
        "If asked to solve a math problem shown in an image, solve it step by step. "
        "Be concise, clear, and helpful. "
        "Current year: 2026."
    )

    messages = [{"role": "system", "content": vision_system}]
    lean_history, _ = get_lean_history(history)
    for msg in lean_history:
        if isinstance(msg["content"], str):
            messages.append(msg)
    messages.append({"role": "user", "content": content})
    return messages


def ask_with_vision(prompt: str, image_urls: list, history: list = []) -> dict:
    image_urls = image_urls[:4]
    print(f"[VISION TRIGGERED] Images: {len(image_urls)}, Prompt: {prompt[:60]}")

    messages = _vision_messages(prompt, image_urls, history)

    # Pass 1: try the full batch against each enabled provider in order.
    for provider in VISION_PROVIDERS:
        if not provider["enabled"]:
            continue
        answer, used_provider = _call_provider_chain(
            [provider], messages, temperature=0.5, max_tokens=1024
        )
        if answer and not _looks_like_vision_refusal(answer):
            return {"answer": answer, "sources": [], "provider": used_provider}
        if answer:
            print(f"[VISION] {provider['name']} returned a refusal-like answer on the "
                  f"full {len(image_urls)}-image batch — trying next provider")

    # Pass 2: every provider refused (or failed outright) on the full
    # batch. If there's more than one image, retry describing each image
    # separately, then merge — this is the one case where batching itself
    # seems to be what a provider can't handle, not the provider being
    # generally unavailable.
    if len(image_urls) > 1:
        print("[VISION] full batch failed on every provider — retrying images one at a time")
        descriptions = []
        for i, url in enumerate(image_urls, start=1):
            single_messages = _vision_messages(prompt, [url], history)
            for provider in VISION_PROVIDERS:
                if not provider["enabled"]:
                    continue
                answer, used_provider = _call_provider_chain(
                    [provider], single_messages, temperature=0.5, max_tokens=1024
                )
                if answer and not _looks_like_vision_refusal(answer):
                    descriptions.append(f"Image {i}: {answer}")
                    break
        if descriptions:
            return {
                "answer": "\n\n".join(descriptions),
                "sources": [],
                "provider": "vision-per-image-fallback",
            }

    return {"answer": _friendly_failure_message(), "sources": [], "provider": None}


_UNSURE_PHRASES = [
    "i don't know", "i do not know", "i'm not sure", "i am not sure",
    "i don't have information", "i do not have information",
    "as of my last update", "as of my knowledge", "i cannot verify",
    "i can't verify", "no information available", "i'm unable to confirm",
    "unable to verify", "unable to confirm", "i'm not certain",
    "i am not certain", "i don't have verified", "i do not have verified",
    "can't guarantee", "cannot guarantee", "i'm unable to provide",
]


def _looks_unsure(answer: str) -> bool:
    a = answer.lower()
    return any(p in a for p in _UNSURE_PHRASES)
# ---------------------------------------------------------------------------
# FILE BUILD TOOL — "universal file builder", the Claude-style artifact
# generator. Detects truncation via the real finish_reason (never an AI
# guessing "does this look done"), auto-continues until the model actually
# reports "stop", uploads the finished file to Supabase Storage, and
# registers a reference in the user's docs so it's searchable later.
# ---------------------------------------------------------------------------
SUPABASE_URL         = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")  # service_role key — server-side only, never the anon key
SUPABASE_BUCKET      = "ooor_bucket"
FILE_BUILD_MAX_CONTINUATIONS = 8

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n|```$", re.MULTILINE)


def _upload_to_supabase(userid: str, filename: str, content: str) -> Optional[str]:
    """
    Uploads file content to Supabase Storage and returns a public URL, or
    None on failure. Never raises — a failed upload should degrade to
    "here's your file inline" rather than crash the whole build.
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        print("[FILEBUILD] SUPABASE_URL / SUPABASE_SERVICE_KEY not set — skipping upload")
        return None

    storage_path = f"{userid}/{filename}"
    upload_url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{storage_path}"

    try:
        resp = requests.post(
            upload_url,
            headers={
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "Content-Type": "text/plain",
                "x-upsert": "true",  # overwrite if same filename already exists
            },
            data=content.encode("utf-8"),
            timeout=30,
        )
        if resp.status_code not in (200, 201):
            print(f"[FILEBUILD] Supabase upload failed: {resp.status_code} — {resp.text[:200]}")
            return None
        return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{storage_path}"
    except Exception as e:
        print(f"[FILEBUILD] Supabase upload error: {e}")
        return None


_CHAT_LEAD_RE = re.compile(
    r"^\s*(sure[,!.]|okay[,!.]|alright[,!.]|certainly[,!.]|of course[,!.]|"
    r"here'?s?\b|here is\b|here are\b|i'?ll\b.*\b(continue|resume|finish)\b|"
    r"continuing from\b|\[continuing)",
    re.IGNORECASE,
)
_CHAT_TRAIL_RE = re.compile(
    r"^\s*(let me know\b|i hope this helps\b|hope this helps\b|"
    r"feel free to\b|that'?s (it|all)\b|done[!.]?\s*$)",
    re.IGNORECASE,
)


def _sanitize_file_chunk(text: str) -> str:
    """
    Strips obvious conversational framing from the very start/end of a
    single generation round — "Here's the code:", "Let me know if you need
    anything else", "Continuing from here:", etc. Deliberately
    conservative: only touches the first/last lines of a chunk, never the
    middle, since mid-chunk regex stripping risks deleting real comments
    or content. This runs on EVERY round, including continuations — that's
    what stops a stray "Continuing from where I left off:" line from a
    later round getting permanently baked into full_content, which is
    exactly what was corrupting continuations before: each round's answer
    was appended raw, so any chatter in round 1 stayed in the file's
    context for every round after it, compounding.

    Mid-file drift across the WHOLE assembled file is a different problem,
    handled separately by _verify_file_is_clean() below, which uses a
    model to classify specific lines instead of guessing with regex.
    """
    lines = text.splitlines()
    while lines and (lines[0].strip() == "" or _CHAT_LEAD_RE.match(lines[0])):
        lines.pop(0)
    while lines and (lines[-1].strip() == "" or _CHAT_TRAIL_RE.match(lines[-1])):
        lines.pop()
    return "\n".join(lines)


def _verify_file_is_clean(filename: str, content: str) -> tuple:
    """
    Runs once, on the FULLY ASSEMBLED file, right before upload — this is
    the model tracing its own finished work, not a hardcoded "done" status.
    Deliberately does NOT ask the model to regenerate the file (that risks
    it silently paraphrasing or altering real content while "cleaning" it
    — a worse bug than the one being fixed). Instead it only asks for the
    LINE NUMBERS of any leftover chatter; removal itself happens
    deterministically in Python, so real content can never be rewritten,
    only exact flagged lines dropped.

    Returns (was_dirty: bool, cleaned_content: str, removed_line_numbers: list).
    On any failure (no provider, bad JSON), fails safe: returns not-dirty
    and the original content untouched rather than risking corruption.
    """
    lines = content.splitlines()
    numbered = "\n".join(f"{i + 1}: {line}" for i, line in enumerate(lines))

    verify_system = (
        f"You will be shown a file named '{filename}', with line numbers "
        "prefixed (format 'N: content'). Find any lines that are leftover "
        "conversational AI chatter that doesn't belong in the actual file — "
        "things like 'Here's the code:', 'I'll continue from here', 'Let me "
        "know if you need anything else', greetings, or meta-commentary "
        "about the build process. Do NOT flag real comments that are "
        "genuinely part of the code/document itself (like '# this function "
        "validates input').\n\n"
        "Reply with ONLY a raw JSON array of line numbers to remove, e.g. "
        "[1, 47, 48]. If nothing needs removing, reply with exactly: []"
    )
    messages = [
        {"role": "system", "content": verify_system},
        {"role": "user", "content": numbered},
    ]

    answer, _ = _call_provider_chain(TEXT_PROVIDERS, messages, temperature=0.0, max_tokens=500)
    if not answer:
        return False, content, []

    try:
        raw = answer.strip().strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
        bad_lines = set(json.loads(raw))
    except Exception:
        return False, content, []

    if not bad_lines:
        return False, content, []

    cleaned_lines = [line for i, line in enumerate(lines, start=1) if i not in bad_lines]
    return True, "\n".join(cleaned_lines), sorted(bad_lines)


def build_file_with_continuation(prompt: str, filename: str, userid: Optional[str], history: list):
    """
    Generator — builds a complete file, auto-continuing past truncation
    using the provider's own finish_reason as ground truth, then uploads
    it and registers it in the user's docs. Yields the same
    {"type": "status", ...} shape as the rest of the pipeline, plus a
    final {"type": "file_result", ...} event.
    """
    build_system = (
        "You are building a complete, real file for the user based on their "
        "request. Output ONLY the file's raw content — no explanation before "
        "or after, no markdown fences unless the file format itself is "
        "markdown. If your response gets cut off before the file is "
        "complete, you will be asked to continue — when that happens, "
        "resume EXACTLY where you left off, mid-line if necessary, with no "
        "repeated content and no re-introduction. When the file is fully, "
        "genuinely complete, end your output with exactly: <<<FILE_DONE>>>"
    )

    messages = [
        {"role": "system", "content": build_system},
        {"role": "user", "content": prompt.strip()},
    ]

    full_content = ""
    for round_num in range(1, FILE_BUILD_MAX_CONTINUATIONS + 1):
        yield {
            "type": "status",
            "text": f"Building {filename}..." if round_num == 1 else f"Continuing {filename} (part {round_num})...",
            "detail": None,
            "icon": "build"
        }

        answer, provider, finish_reason = _call_provider_chain_full(
            TEXT_PROVIDERS, messages, temperature=0.3, max_tokens=16000,
        )

        if answer is None:
            yield {"type": "status", "text": "Build failed — no provider available", "detail": None, "icon": "warning"}
            yield {"type": "file_result", "success": False, "url": None, "filename": filename}
            return 
        clean_chunk = _sanitize_file_chunk(answer)
        full_content += clean_chunk
        messages.append({"role": "assistant", "content": clean_chunk})

        done_marker_found = "<<<FILE_DONE>>>" in full_content
        if done_marker_found:
            full_content = full_content.replace("<<<FILE_DONE>>>", "").rstrip()

        if finish_reason == "stop" or done_marker_found:
            break

        # finish_reason == "length" (or anything not "stop") — genuinely
        # truncated mid-file. Ask it to continue from exactly where it
        # stopped, with the full accumulated content as context.
        yield {
            "type": "status",
            "text": "Response was cut off mid-file — continuing automatically",
            "detail": f"finish_reason was '{finish_reason}', not 'stop' — the file isn't done yet.",
            "icon": "warning"
        }
        messages.append({
            "role": "user",
            "content": "Continue exactly where you stopped. Do not repeat anything already written.",
        })
    else:
        yield {
            "type": "status",
            "text": f"Stopped after {FILE_BUILD_MAX_CONTINUATIONS} continuation rounds — file may be incomplete",
            "detail": None,
            "icon": "warning"
        }

    # Strip a single wrapping ```fence``` if the model added one anyway
    full_content = _FENCE_RE.sub("", full_content).strip()

    # ── Final trace pass — the AI checks its OWN finished file, with real,
    # visible steps, before anything gets uploaded. This is what actually
    # replaces the old hardcoded "Done" status: instead of just declaring
    # success, it looks at what it built and reports what it found.
    yield {
        "type": "status",
        "text": f"Reviewing {filename} for accuracy...",
        "detail": "Tracing through the finished file to check nothing conversational leaked in.",
        "icon": "search",
    }
    was_dirty, full_content, removed_lines = _verify_file_is_clean(filename, full_content)
    if was_dirty:
        yield {
            "type": "status",
            "text": f"Found {len(removed_lines)} stray line(s) — cleaned up",
            "detail": f"Removed line(s) {removed_lines} that weren't real file content.",
            "icon": "warning",
        }
    else:
        yield {
            "type": "status",
            "text": f"{filename} checks out — no stray content found",
            "detail": None,
            "icon": "success",
        }

    yield {"type": "status", "text": "Uploading file...", "detail": None, "icon": "upload"}
    file_url = _upload_to_supabase(userid or "anonymous", filename, full_content)

    if file_url and userid:
        try:
            manager = UserDocManager(userid)
            manager.save_doc(
                filename=f"ref_{filename}.md",
                content=f"File stored at: {file_url}",
                hint=filename,
                tags=["ai-built-file", filename.split(".")[-1]],
                metadata={"supabase_url": file_url, "original_filename": filename},
            )
        except Exception as e:
            print(f"[FILEBUILD] failed to register doc reference: {e}")

    yield {
        "type": "status",
        "text": "Done" if file_url else "File built but upload failed",
        "detail": None,
        "icon": "success" if file_url else "warning"
    }
    yield {
        "type": "file_result",
        "success": bool(file_url),
        "url": file_url,
        "filename": filename,
    }
