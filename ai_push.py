# ai_push.py — FCM pushes for the AI chat (same style as the chat notifier in app.py)
# ─────────────────────────────────────────────────────────────────────────────
# When a job reaches a moment the user may be waiting for, this sends ONE data-only push to users/{uid}.fcmToken:
#
#   notifType      when                                       buttons the phone draws (AiNotifications.kt)
#   ai_reply       the answer is ready                        suggestion 1 · suggestion 2 · Open chat   (Continue first if cut off)
#   ai_error       the turn failed                            Continue (if it can resume) · Open chat
#   ai_question    the AI asks a multiple-choice question     option 1 · Skip · Open chat
#   ai_approval    OOOR needs a yes/no (command, fetch, MCP)  Approve · Deny · View command   (long text under "see more")
#   ai_clear       that approval is over (answered/expired)   — removes the notification so a stale Approve can't be tapped
#
# Same wire rules as the chat pushes: DATA-ONLY (no `notification` block, so the app's ZindryxMessagingService always
# draws it), android priority high, every value a string capped at 600 chars (FCM rejects a push over ~4 KB).
# The BACKEND cannot know whether the chat screen is open, so it always sends; the PHONE suppresses the banner when the
# AI chat is on screen (ActiveAiChat). `recipientId` lets a phone ignore a token that another account on it reused.
#
# Switch off per user: users/{uid}.aiPushEnabled = false. Nothing here ever raises or delays a job: sends run on a small
# thread pool and every failure is just logged (the transcript is never logged).
# ─────────────────────────────────────────────────────────────────────────────
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import Any, List, Optional, Tuple

MAX_VALUE = 600                 # same cap as the chat notifier
PREVIEW_CHARS = 150             # the collapsed notification line
BIG_CHARS = 560                 # the expanded ("see more") text
MAX_SUGGESTIONS = 3
LABEL_CHARS = 34                # what fits on a notification button
REPLY_TTL = timedelta(hours=24)

_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ai-push")
_token_cache: dict = {}         # uid -> (expires_at, token or None, enabled)
_TOKEN_TTL_S = 60


class StaleToken(Exception):
    """FCM says this token is dead (app uninstalled / token rotated)."""


# ── text helpers (pure, unit-tested) ────────────────────────────────────────
_SUGGESTION = re.compile(r"(?is)<{1,2}suggestion>{1,2}(.*?)<{1,2}/?suggestion>{1,2}")
_DANGLING = re.compile(r"(?is)<{1,2}suggestion>{1,2}.*$")
_FENCE = re.compile(r"(?s)```.*?```")
_TAGS = re.compile(r"<{1,2}/?(?:note|callout|suggestion|think|thinking)[^<>]*>{1,2}", re.I)
_HTML = re.compile(r"<[^>\n]{1,200}>")


def clip(s: Any, n: int = MAX_VALUE) -> str:
    t = "" if s is None else str(s)
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


def extract_suggestions(answer: str) -> Tuple[str, List[str]]:
    """'text <<suggestion>>Okay<<suggestion>>' -> ('text', ['Okay']). Tags never leak into a preview."""
    found = [m.group(1).strip() for m in _SUGGESTION.finditer(answer or "")]
    text = _DANGLING.sub("", _SUGGESTION.sub("", answer or ""))
    return text.strip(), [s for s in found if s]


def clean_preview(markdown: str, limit: int = PREVIEW_CHARS) -> str:
    """Markdown -> one plain line for a notification (tables/code/tags/links flattened), cut at a word."""
    t = _FENCE.sub(" [code] ", markdown or "")
    t = _TAGS.sub(" ", t)
    t = re.sub(r"(?m)^\s*\|.*\|\s*$", " ", t)                       # table rows
    t = re.sub(r"(?m)^\s*(?:[-*_]\s*){3,}$", " ", t)                # dividers
    t = re.sub(r"!\[([^\]]*)]\([^)]*\)", r"\1", t)                  # images
    t = re.sub(r"\[([^\]]+)]\([^)]*\)", r"\1", t)                   # links -> their text
    t = _HTML.sub(" ", t)
    t = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", t)                     # headings
    t = re.sub(r"(?m)^\s*>+\s?", "", t)                             # quotes
    t = re.sub(r"(?m)^\s*(?:[-*+•]|\d+[.)])\s+", "", t)             # bullets
    t = re.sub(r"(\*\*|__|~~|==|`)", "", t)                         # emphasis / highlight / code ticks
    t = re.sub(r"(?<![\w*])\*(?!\s)|(?<!\s)\*(?![\w*])", "", t)     # lone * italics
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) <= limit:
        return t
    cut = t[:limit]
    if " " in cut[limit // 2:]:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip(" ,;:.-") + "…"


def _suggestion_items(raw: Any, answer: str) -> Tuple[str, List[dict]]:
    """Suggestions from final['suggestions'] (strings or {text}) or, failing that, the tags inside the answer."""
    text, from_tags = extract_suggestions(answer)
    items: List[str] = []
    for s in (raw or []):
        v = s.get("text") if isinstance(s, dict) else s
        if isinstance(v, str) and v.strip():
            items.append(v.strip())
    items = items or from_tags
    out, seen = [], set()
    for s in items:
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"l": clip(s, LABEL_CHARS), "t": clip(s, 200)})        # l = button label, t = what gets sent
        if len(out) >= MAX_SUGGESTIONS:
            break
    return text, out


# ── building each push (pure) ───────────────────────────────────────────────
def _base(uid: str, kind: str, job_id: str, title: str, body: str, big: str = "") -> dict:
    return {"notifType": kind, "jobId": job_id, "recipientId": uid, "sentAt": str(int(time.time() * 1000)),
            "title": clip(title, 80), "body": clip(body, MAX_VALUE), "bigText": clip(big or body, MAX_VALUE)}


def build_reply(uid: str, job_id: str, final: dict, continue_flag: Optional[dict] = None) -> Optional[dict]:
    answer = (final or {}).get("answer") or ""
    text, sugg = _suggestion_items((final or {}).get("suggestions"), answer)
    preview = clean_preview(text)
    if not preview:
        n_cards = sum(1 for b in (final or {}).get("blocks") or [] if isinstance(b, dict) and b.get("type") in ("card", "file", "images"))
        if not n_cards:
            return None
        preview = "Your answer is ready."
    interrupted = bool(continue_flag and continue_flag.get("show"))
    data = _base(uid, "ai_reply", job_id, "OOOR", preview, clean_preview(text, BIG_CHARS))
    data["suggestions"] = json.dumps(sugg, ensure_ascii=False)
    data["canContinue"] = "true" if interrupted else "false"
    return data


def build_error(uid: str, job_id: str, message: str, continue_flag: Optional[dict] = None) -> dict:
    data = _base(uid, "ai_error", job_id, "OOOR hit a problem", clean_preview(message) or "Something went wrong.")
    data["suggestions"] = "[]"
    data["canContinue"] = "true" if continue_flag and continue_flag.get("show") else "false"
    return data


def build_question(uid: str, job_id: str, q: dict) -> Optional[dict]:
    question = clean_preview((q or {}).get("question") or "", 200)
    if not question:
        return None
    opts: List[dict] = []
    for o in (q or {}).get("options") or []:
        label = o.get("label") if isinstance(o, dict) else o
        if isinstance(label, str) and label.strip():
            opts.append({"l": clip(label.strip(), LABEL_CHARS), "t": clip(label.strip(), 200)})
        if len(opts) >= 2:
            break
    data = _base(uid, "ai_question", job_id, "OOOR has a question", question)
    data["question"] = clip(question, 200)
    data["options"] = json.dumps(opts, ensure_ascii=False)
    return data


def build_approval(uid: str, job_id: str, card: dict) -> dict:
    kind = str(card.get("kind") or "")
    title = clean_preview(card.get("title") or "", 80) or "OOOR needs your approval"
    target = str(card.get("target") or "").strip()
    why = str(card.get("why") or "").strip()
    big = "\n".join(x for x in (why, target) if x)
    data = _base(uid, "ai_approval", job_id, title, clip(target or why or "Tap to review", 120), big)
    data.update({"approvalId": str(card.get("id") or ""), "approvalKind": kind, "risk": str(card.get("risk") or "normal"),
                 "expiresAt": str(card.get("expiresAt") or "")})
    return data


# ── delivery ────────────────────────────────────────────────────────────────
def _user_doc(uid: str) -> dict:
    import firebase_config  # noqa: F401
    from firebase_admin import firestore
    snap = firestore.client().collection("users").document(uid).get()
    return (snap.to_dict() or {}) if snap.exists else {}


def _token_for(uid: str) -> Tuple[Optional[str], bool]:
    now = time.time()
    hit = _token_cache.get(uid)
    if hit and hit[0] > now:
        return hit[1], hit[2]
    doc = _user_doc(uid)
    token = doc.get("fcmToken") or None
    enabled = doc.get("aiPushEnabled") is not False
    _token_cache[uid] = (now + _TOKEN_TTL_S, token, enabled)
    return token, enabled


def _fcm_send(token: str, data: dict, ttl: timedelta) -> None:
    """The only function that touches Firebase Messaging (tests replace it)."""
    from firebase_admin import messaging
    try:
        messaging.send(messaging.Message(token=token, data=data, android=messaging.AndroidConfig(priority="high", ttl=ttl)))
    except (messaging.UnregisteredError, messaging.SenderIdMismatchError) as e:
        raise StaleToken(str(e))


def _forget_token(uid: str, token: str) -> None:
    """Remove a dead token, but only if it is still the one we just failed on (the app may have refreshed it)."""
    try:
        import firebase_config  # noqa: F401
        from firebase_admin import firestore
        ref = firestore.client().collection("users").document(uid)
        if (ref.get().to_dict() or {}).get("fcmToken") == token:
            ref.update({"fcmToken": firestore.DELETE_FIELD})
    except Exception as e:
        print(f"[AI-PUSH] couldn't clear a stale token: {type(e).__name__}")
    _token_cache.pop(uid, None)


def deliver(uid: str, data: dict, ttl: timedelta = REPLY_TTL) -> bool:
    """Send one push. True only if FCM accepted it. Never raises."""
    try:
        token, enabled = _token_for(uid)
        if not token or not enabled:
            return False
        safe = {str(k): ("" if v is None else str(v))[:MAX_VALUE] for k, v in data.items()}
        try:
            _fcm_send(token, safe, ttl)
        except StaleToken:
            _forget_token(uid, token)
            return False
        return True
    except Exception as e:
        print(f"[AI-PUSH] {data.get('notifType')} not sent: {type(e).__name__}")
        return False


def _run(fn, *args) -> None:
    _pool.submit(fn, *args)


def _approval_ttl(card: dict) -> timedelta:
    try:
        left = (int(card.get("expiresAt")) - int(time.time() * 1000)) / 1000
    except (TypeError, ValueError):
        left = 300
    return timedelta(seconds=max(30, min(left, 3600)))      # a stale approval must not arrive after it expired


# ── the hooks ai_jobs.py calls (all fire-and-forget) ────────────────────────
def notify_reply(uid: str, job_id: str, final: dict, continue_flag: Optional[dict] = None) -> None:
    try:
        data = build_reply(uid, job_id, final, continue_flag)
        if data:
            _run(deliver, uid, data)
    except Exception as e:
        print(f"[AI-PUSH] reply hook: {type(e).__name__}")


def notify_error(uid: str, job_id: str, message: str, continue_flag: Optional[dict] = None) -> None:
    try:
        _run(deliver, uid, build_error(uid, job_id, message, continue_flag))
    except Exception as e:
        print(f"[AI-PUSH] error hook: {type(e).__name__}")


def notify_question(uid: str, job_id: str, q: dict) -> None:
    try:
        data = build_question(uid, job_id, q)
        if data:
            _run(deliver, uid, data)
    except Exception as e:
        print(f"[AI-PUSH] question hook: {type(e).__name__}")


def notify_approval(uid: str, job_id: str, card: dict) -> None:
    try:
        _run(deliver, uid, build_approval(uid, job_id, card), _approval_ttl(card))
    except Exception as e:
        print(f"[AI-PUSH] approval hook: {type(e).__name__}")


def notify_approval_closed(uid: str, job_id: str, approval_id: str, status: str) -> None:
    """Tell the phone to drop the approval notification (answered in the app, expired or cancelled)."""
    try:
        data = {"notifType": "ai_clear", "jobId": job_id, "recipientId": uid, "approvalId": approval_id, "status": status,
                "sentAt": str(int(time.time() * 1000))}
        _run(deliver, uid, data, timedelta(minutes=10))
    except Exception as e:
        print(f"[AI-PUSH] clear hook: {type(e).__name__}")
