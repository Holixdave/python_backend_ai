#!/usr/bin/env python3
# ai_jobs.py — POST /ai-job
# ─────────────────────────────────────────────────────────────────────────────
# Why this exists
#   /ai-query-stream ties the reply to the open connection, and the app saves
#   the reply into Firestore itself. Close the app mid-reply and the answer
#   has nowhere to go, so it is lost.
#
# What this does instead
#   1. The app POSTs the question with its Firebase ID token (identity comes
#      from the verified token, NOT from the request body).
#   2. We create users/{uid}/chat_history/{aiMessageId} with status "running"
#      and return 202 straight away.
#   3. The turn runs in a background worker, independent of the connection.
#      Progress (thinking steps, tool cards) is written to that same doc.
#   4. On finish we write the full reply (status "done"), charge the credits
#      server-side, and save the turn to the backend memory.
#
# The app already listens to users/{uid}/chat_history with a snapshot
# listener, so it just renders that doc. Closing and reopening the app shows
# the finished reply (or the still-running shimmer).
#
# The doc uses the SAME field names as ChatMessageFirestore.kt, plus:
#   status        "running" | "done" | "error"
#   updatedAt     ms epoch, heartbeat while running (app treats stale as dead)
#   cancelRequested  set by the app's Stop button
#   charge        {chars, fromFree, fromWallet}  (free users only)
# ─────────────────────────────────────────────────────────────────────────────

import json
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException
from firebase_admin import auth as fb_auth
from firebase_admin import firestore
from google.api_core.exceptions import AlreadyExists
from pydantic import BaseModel, Field

import firebase_config  # noqa: F401 (initialises firebase_admin before firestore.client())
import gpt2_approvals
from database import SessionLocal
from gpt2_functions import is_friendly_failure, premium_from_user_doc
from gpt2_test import ask_gpt2_stream
from leak_guard import scrub_internal
from memory_service import build_memory, remember_turn

router = APIRouter()

# ── tunables ────────────────────────────────────────────────────────────────
FREE_CHARS_PER_DAY = 6_000          # keep in sync with CreditPolicy.FREE_CHARS_PER_DAY (Kotlin)
MODEL_FREE = "OOOR 270"
MODEL_PREMIUM = "OOOR 370 beta"
STEP_WRITE_INTERVAL = 0.5           # seconds; coalesce thinking-step writes
HEARTBEAT_SECONDS = 12              # updatedAt ping + cancel check
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


# Premium turns get this prefix. It used to be added by the app (client-controlled
# prompt text); the server now decides, from the verified tier.
ADVANCED_TAG = (
    "[ADVANCED THINKING ENABLED - Respond with deep analysis, be thorough and professional, "
    "satisfy the user fully, dont talk too much give user answer to thier question  if it code "
    "output code instaed of too much uneccesary talk the only way you are allowed to talked when "
    "writing code is telling user where the code go in ]\n\n"
)


def _contextual_query(req, query: str) -> str:
    """User text + reply/clarification context (what gets remembered). No tier tag."""
    parts = []
    if (req.promptContext or "").strip():
        parts.append(req.promptContext.strip())
    if (req.replyToText or "").strip():
        parts.append(f"Replying to {req.replyToSender or 'user'}: {req.replyToText.strip()[:500]}")
    context = "\n".join(parts)
    return query if not context else f"{context}\n\nUser message: {query}"


# ── request model ───────────────────────────────────────────────────────────
class JobRequest(BaseModel):
    query: str
    aiMessageId: str
    userMessageId: Optional[str] = None
    userTimestamp: Optional[int] = None      # ms; the AI reply is stamped after it
    replyToId: Optional[str] = None
    replyToText: Optional[str] = None
    replyToSender: Optional[str] = None
    promptContext: Optional[str] = None     # e.g. "Clarification question: ..." — goes to the model, not into the chat bubble
    imageUrls: List[str] = Field(default_factory=list)
    fileUrls: List[str] = Field(default_factory=list)
    fileNames: List[str] = Field(default_factory=list)


# ── small helpers ───────────────────────────────────────────────────────────
def _get_uid(authorization: Optional[str]) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing auth token")
    try:
        return fb_auth.verify_id_token(authorization.removeprefix("Bearer ").strip())["uid"]
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid auth token")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _ts(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _clean(value):
    """Firestore only takes plain JSON-ish values; round-trip anything odd."""
    return json.loads(json.dumps(value, default=str))


def _history_col(uid: str):
    return firestore.client().collection("users").document(uid).collection("chat_history")


def _read_user(uid: str) -> dict:
    return firestore.client().collection("users").document(uid).get().to_dict() or {}


def _free_left(user: dict) -> int:
    used = int(user.get("freeCharsUsed", 0) or 0) if user.get("freeCharsPeriod") == _today() else 0
    return max(0, FREE_CHARS_PER_DAY - used)


def _wallet(user: dict) -> int:
    return max(0, int(user.get("aiCreditChars", 0) or 0))


def _safe_update(ref, data: dict):
    try:
        ref.update(data)
    except Exception as e:  # doc deleted by the user, or a transient Firestore error
        print(f"[AI-JOB] update failed for {ref.path}: {e}")


# ── credits (server-side, in a transaction) ─────────────────────────────────
@firestore.transactional
def _charge_txn(txn, ref, chars: int, today: str) -> dict:
    user = ref.get(transaction=txn).to_dict() or {}
    used = int(user.get("freeCharsUsed", 0) or 0) if user.get("freeCharsPeriod") == today else 0
    wallet = max(0, int(user.get("aiCreditChars", 0) or 0))
    free_left = max(0, FREE_CHARS_PER_DAY - used)
    from_free = min(chars, free_left)
    from_wallet = min(chars - from_free, wallet)
    txn.set(ref, {
        "freeCharsPeriod": today,
        "freeCharsUsed": used + from_free,
        "aiCreditChars": wallet - from_wallet,
    }, merge=True)
    return {"chars": chars, "fromFree": from_free, "fromWallet": from_wallet}


def _charge(uid: str, chars: int) -> Optional[dict]:
    if chars <= 0:
        return None
    try:
        client = firestore.client()
        ref = client.collection("users").document(uid)
        return _charge_txn(client.transaction(), ref, chars, _today())
    except Exception as e:
        print(f"[AI-JOB] charge failed for {uid!r}: {e}")
        return None


# ── asking the user ("OOOR wants to fetch ..." / "OOOR wants to run this command") ───────────
# The turn runs in a worker thread, so asking is simple: write the question onto the reply doc
# (the app already listens to it), then wait for the app's answer in `approvalDecision`, or for
# the timeout. Anything but an explicit Allow means NOT approved (see gpt2_approvals.py).
APPROVAL_POLL_SECONDS = 1.5
_DECISIONS = {gpt2_approvals.ALLOW, gpt2_approvals.ALLOW_ADD, gpt2_approvals.DENY}


def _make_approver(ref, state: dict):
    def ask(spec: dict) -> str:
        approval_id = uuid.uuid4().hex[:16]
        timeout_s = max(10, int(spec.get("timeoutS") or gpt2_approvals.FETCH_TIMEOUT_S))
        now = _now_ms()
        card = {
            "id": approval_id, "kind": spec.get("kind"), "tool": spec.get("tool"),
            "title": spec.get("title"), "target": spec.get("target"), "why": spec.get("why"),
            "risk": spec.get("risk", "normal"), "allowAdd": bool(spec.get("allowAdd")),
            "status": "pending", "createdAt": now, "expiresAt": now + timeout_s * 1000,
        }
        try:
            ref.update({"approval": card, "approvalDecision": firestore.DELETE_FIELD, "updatedAt": now})
        except Exception as e:
            print(f"[AI-JOB] couldn't post approval request: {e}")
            return gpt2_approvals.DENY

        def close(status: str, result: str) -> str:
            _safe_update(ref, {"approval.status": status, "updatedAt": _now_ms()})
            print(f"[AI-JOB] approval {approval_id} ({card['kind']}) -> {status}")
            return result

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if state["cancel"]:
                return close("cancelled", gpt2_approvals.CANCELLED)
            try:
                snap = ref.get()
                if not snap.exists:
                    return gpt2_approvals.CANCELLED
                data = snap.to_dict() or {}
            except Exception as e:
                print(f"[AI-JOB] approval poll error: {e}")
                time.sleep(APPROVAL_POLL_SECONDS)
                continue
            answer = data.get("approvalDecision")
            if isinstance(answer, dict) and answer.get("id") == approval_id:     # only THIS request's answer counts
                decision = answer.get("decision")
                if decision in _DECISIONS:
                    approved = decision in (gpt2_approvals.ALLOW, gpt2_approvals.ALLOW_ADD)
                    return close("approved" if approved else "denied", decision)
            time.sleep(APPROVAL_POLL_SECONDS)
        return close("expired", gpt2_approvals.TIMEOUT)

    return ask


# ── POST /ai-job ────────────────────────────────────────────────────────────
@router.post("/ai-job", status_code=202)
def start_job(req: JobRequest, background: BackgroundTasks, authorization: Optional[str] = Header(None)):
    uid = _get_uid(authorization)

    query = req.query.strip()
    if not _ID_RE.match(req.aiMessageId or ""):
        raise HTTPException(status_code=400, detail="Bad aiMessageId")
    if req.userMessageId and not _ID_RE.match(req.userMessageId):
        raise HTTPException(status_code=400, detail="Bad userMessageId")
    if not query and not (req.imageUrls or req.fileUrls):
        raise HTTPException(status_code=400, detail="Empty message")

    user = _read_user(uid)
    premium = premium_from_user_doc(user)
    if not premium and (_free_left(user) + _wallet(user)) <= 0:
        raise HTTPException(status_code=402, detail="no_credits")

    model_name = MODEL_PREMIUM if premium else MODEL_FREE
    col = _history_col(uid)
    now = _now_ms()
    ai_ts = max(now, (req.userTimestamp or 0) + 1)

    # Idempotent: a retried POST with the same id must not run the turn twice.
    try:
        col.document(req.aiMessageId).create({
            "sender": "ai",
            "text": "",
            "timestamp": _ts(ai_ts),
            "modelUsed": model_name,
            "status": "running",
            "updatedAt": now,
            "cancelRequested": False,
            "thinkingSteps": [],
            "sources": [], "images": [], "imageUrls": [], "fileUrls": [], "fileNames": [],
            "builtFileUrls": [], "builtFileNames": [], "codeResults": [],
            "segments": [], "suggestions": [],
        })
    except AlreadyExists:
        return {"status": "exists", "aiMessageId": req.aiMessageId}

    # The user's own message too, so it survives even if the app was killed
    # before its own write flushed. Same doc id → harmless if it already exists.
    if req.userMessageId:
        try:
            col.document(req.userMessageId).create({
                "sender": "user",
                "text": query,
                "timestamp": _ts(req.userTimestamp or now),
                "modelUsed": model_name,
                "replyToId": req.replyToId,
                "replyToText": req.replyToText,
                "replyToSender": req.replyToSender,
                "imageUrls": req.imageUrls,
                "fileUrls": req.fileUrls,
                "fileNames": req.fileNames,
                "sources": [], "thinkingSteps": [], "images": [],
                "builtFileUrls": [], "builtFileNames": [], "codeResults": [],
                "segments": [], "suggestions": [],
            })
        except AlreadyExists:
            pass
        except Exception as e:
            print(f"[AI-JOB] user message write failed: {e}")

    background.add_task(_run_job, uid, req, query, premium, model_name)
    return {"status": "started", "aiMessageId": req.aiMessageId}


# ── the worker ──────────────────────────────────────────────────────────────
def _heartbeat(ref, state: dict, stop: threading.Event):
    """While the turn runs: prove we're alive (updatedAt) and notice Stop."""
    while not stop.wait(HEARTBEAT_SECONDS):
        try:
            snap = ref.get()
            data = snap.to_dict() or {}
            if not snap.exists or data.get("cancelRequested"):
                state["cancel"] = True
                return
            ref.update({"updatedAt": _now_ms()})
        except Exception as e:
            print(f"[AI-JOB] heartbeat error: {e}")


def _text_of(segments: list) -> str:
    return "\n\n".join(
        s.get("content", "") for s in segments if s.get("type") == "text" and s.get("content", "").strip()
    ).strip()


def _build_done_payload(final: dict, steps: list, charge: Optional[dict]) -> dict:
    answer = final.get("answer") or ""
    segments = final.get("segments") or [{"type": "text", "content": answer}]

    files = list(final.get("files") or [])
    if final.get("file"):
        files.append(final["file"])
    seen, built_urls, built_names = set(), [], []
    for f in files:
        if f.get("success") is True and f.get("url") and f["url"] not in seen:
            seen.add(f["url"])
            built_urls.append(f["url"])
            built_names.append(f.get("filename") or "file")

    sources = []
    for s in (final.get("sources") or []):
        href = s.get("href") or s.get("url") or s.get("link") or ""
        if href:
            sources.append({"title": s.get("title") or s.get("name") or "", "href": href})

    payload = {
        "text": answer,
        "segments": segments,
        "sources": sources,
        "images": final.get("images") or [],
        "builtFileUrls": built_urls,
        "builtFileNames": built_names,
        "builtFileUrl": built_urls[0] if built_urls else None,
        "builtFileName": built_names[0] if built_names else None,
        "codeResults": final.get("codes") or [],
        "suggestions": final.get("suggestions") or [],
        "blocks": final.get("blocks") or [],
        "thinkingSteps": steps,
        "status": "done",
        "updatedAt": _now_ms(),
        "charge": charge,
    }
    return _clean(payload)


def _run_job(uid: str, req: JobRequest, query: str, premium: bool, model_name: str):
    ref = _history_col(uid).document(req.aiMessageId)
    steps: list = []
    state = {"cancel": False}
    stop = threading.Event()
    threading.Thread(target=_heartbeat, args=(ref, state, stop), daemon=True).start()
    approver_token = gpt2_approvals.set_approver(_make_approver(ref, state))

    last_write = 0.0

    def push_steps(force: bool = False):
        nonlocal last_write
        now = time.time()
        if not force and now - last_write < STEP_WRITE_INTERVAL:
            return
        last_write = now
        _safe_update(ref, {"thinkingSteps": _clean(steps), "updatedAt": _now_ms()})

    # Interleaved turn (gpt2_native.py): the ordered blocks as they happen, written to the
    # message doc while it runs so the app draws text / tool chips / images / files in order.
    live_blocks: list = []
    last_block_write = 0.0

    def push_blocks(force: bool = False):
        nonlocal last_block_write
        now = time.time()
        if not force and now - last_block_write < STEP_WRITE_INTERVAL:
            return
        last_block_write = now
        shown = [dict(b, text=scrub_internal(b.get("text") or "")) if b.get("type") == "text" else b for b in live_blocks]
        _safe_update(ref, {"blocks": _clean(shown), "updatedAt": _now_ms()})

    def apply_stream_event(event: dict) -> bool:
        """Folds one text/tool_call/tool_result/block event into live_blocks. True if it was one."""
        et = event.get("type")
        if et == "text":
            if live_blocks and live_blocks[-1].get("type") == "text":
                live_blocks[-1]["text"] += event.get("delta") or ""
            else:
                live_blocks.append({"type": "text", "text": event.get("delta") or ""})
            push_blocks()
        elif et == "tool_call":
            live_blocks.append({"type": "tool", "id": event.get("id"), "tool": event.get("tool"),
                                "text": event.get("text") or "", "icon": event.get("icon"), "ok": None,
                                "input": event.get("input"), "output": None})
            push_blocks(force=True)
        elif et == "tool_result":
            for b in live_blocks:
                if b.get("type") == "tool" and b.get("id") == event.get("id"):
                    b["ok"] = bool(event.get("ok"))
                    b["output"] = event.get("output")
            push_blocks(force=True)
        elif et == "block":
            kind = event.get("kind")
            if kind == "images":
                live_blocks.append({"type": "images", "images": event.get("images") or []})
            elif kind == "file":
                live_blocks.append({"type": "file", "file": event.get("file") or {}})
            elif kind == "sources":
                live_blocks.append({"type": "sources", "sources": event.get("sources") or []})
            elif kind == "code":
                live_blocks.append({"type": "code", "code": event.get("code") or {}})
            push_blocks(force=True)
        else:
            return False
        return True

    db = SessionLocal()
    final = None
    question = None
    memory_reply = None
    contextual = _contextual_query(req, query)
    # The premium directive is a SYSTEM-prompt rule now (premium_mode below). It must never be
    # glued onto the user's message: the model then treats it as user text and repeats it.
    model_query = contextual
    try:
        history = build_memory(db, uid)
        print(f"[AI-JOB] start {req.aiMessageId} uid={uid!r} premium={premium} q={query[:80]!r}")

        # Math-first short-circuit (same as /ai-query-stream). Lazy import: main.py owns these helpers.
        try:
            from main import _looks_like_equation, _equation_solved_ok, solve_equation_with_steps
            if query and _looks_like_equation(query):
                eq_answer = solve_equation_with_steps(query)
                if _equation_solved_ok(eq_answer):
                    steps.append({"text": "Solving equation...", "detail": None, "icon": None, "kind": "status", "tool": None})
                    final = {"answer": eq_answer, "segments": [{"type": "text", "content": eq_answer}]}
        except Exception as e:
            print(f"[AI-JOB] equation check skipped: {e}")

        if final is None:
            for event in ask_gpt2_stream(
                model_query, history=history,
                image_urls=req.imageUrls or None,
                file_urls=req.fileUrls or None,
                file_names=req.fileNames or None,
                userid=uid,
                premium_mode=premium,
            ):
                if state["cancel"]:
                    break
                etype = event.get("type")
                if apply_stream_event(event):
                    continue
                if etype == "status":
                    steps.append({
                        "text": event.get("text") or "", "detail": event.get("detail"),
                        "icon": event.get("icon"), "kind": "status", "tool": event.get("tool"),
                    })
                    push_steps(force=bool(event.get("tool")) or event.get("icon") in ("image", "sandbox"))
                elif etype == "think":
                    steps.append({
                        "text": scrub_internal(event.get("text") or ""), "detail": None,
                        "icon": event.get("icon"), "kind": "think", "tool": None,
                    })
                    push_steps()
                elif etype == "final":
                    final = event
                elif etype == "ask_user":
                    question = event
                    break

        if state["cancel"]:
            print(f"[AI-JOB] {req.aiMessageId} cancelled by user")
            try:
                ref.delete()
            except Exception as e:
                print(f"[AI-JOB] cancel delete failed: {e}")
            return

        # The AI chose to ask the user a multiple-choice question instead of answering.
        # Write it onto the reply doc as status "asking"; the app opens the bottom sheet,
        # removes this doc, and sends the tapped answer as the next message. Nothing is
        # charged for a question (only real answers are billed).
        if question is not None:
            q = {
                "question_id": question.get("question_id"),
                "question": question.get("question"),
                "options": question.get("options") or [],
                "allow_multiple": bool(question.get("allow_multiple")),
                "allow_other": bool(question.get("allow_other", True)),
            }
            if question.get("questions"):          # several questions in one sheet (Next button)
                q["questions"] = question["questions"]
                q["total"] = question.get("total") or len(question["questions"])
            ref.set(_clean({"status": "asking", "question": q, "text": "", "updatedAt": _now_ms()}), merge=True)
            remember_turn(db, uid, contextual, q["question"] or "")
            print(f"[AI-JOB] asking {req.aiMessageId}: {q['question']!r}")
            return

        if final is None or is_friendly_failure(final.get("answer")):
            msg = (final or {}).get("answer") or "Something went wrong. Please try again."
            _safe_update(ref, {"status": "error", "text": msg, "error": msg, "updatedAt": _now_ms()})
            return

        # Charge server-side (free users only), once, only for a real answer.
        charge = None
        if not premium:
            plain = _text_of(final.get("segments") or [{"type": "text", "content": final.get("answer") or ""}])
            charge = _charge(uid, len(query) + len(plain))

        payload = _build_done_payload(final, steps, charge)
        try:
            ref.set(payload, merge=True)
        except Exception as e:
            # Most likely the 1 MiB doc limit (huge inline HTML / code output). Keep the text, drop the heavy parts.
            print(f"[AI-JOB] full write failed ({e}); retrying with a reduced payload")
            small = (final.get("answer") or "")[:30000]
            ref.set({**payload, "text": small, "segments": [{"type": "text", "content": small}],
                     "codeResults": [], "images": payload.get("images", [])[:10]}, merge=True)

        try:
            from main import _build_memory_reply
            memory_reply = _build_memory_reply(final.get("answer"), final.get("sources", []), final.get("images", []))
        except Exception:
            memory_reply = final.get("answer")
        remember_turn(db, uid, contextual, memory_reply)
        print(f"[AI-JOB] done {req.aiMessageId}")

    except Exception as e:
        print(f"[AI-JOB] crashed {req.aiMessageId}: {type(e).__name__}: {e}")
        _safe_update(ref, {"status": "error", "text": "Something went wrong. Please try again.",
                           "error": "Something went wrong. Please try again.", "updatedAt": _now_ms()})
    finally:
        stop.set()
        gpt2_approvals.reset_approver(approver_token)
        _safe_update(ref, {"approval": firestore.DELETE_FIELD, "approvalDecision": firestore.DELETE_FIELD})
        db.close()
