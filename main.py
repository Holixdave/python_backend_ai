# main.py
from fastapi import FastAPI, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from sqlalchemy.orm import Session
import os
import json
import re

from math_solver import solve_math_with_explanation
from equation_solver import solve_equation_with_steps
from gpt2_test import ask_gpt2, ask_gpt2_stream
from user_doc_manager import UserDocManager

# ── Backend-side chat memory (NEW) ──────────────────────────────────────
# Same pattern as the working Zindryx/Gemini backend: a DB-backed history
# table keyed by userid. The frontend no longer needs to replay the whole
# conversation on every request — this app now remembers it server-side.
from database import get_db, Base, engine
from memory_service import build_memory, remember_turn
import firebase_config  # noqa: F401 (must init before firestore_repository is used)

app = FastAPI(
    title="UTME26 AI Backend",
    description="Brilliant AI Study Assistant"
)

# Creates the chat_history table on startup if it doesn't exist yet —
# same as the working app's main.py does with Base.metadata.create_all.
Base.metadata.create_all(bind=engine)


# -------------------------------------------------------
# Pydantic models — DO NOT CHANGE (frontend depends on these)
# -------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class QuestionRequest(BaseModel):
    query: str
    # DEPRECATED: the backend now keeps its own memory per userid (see
    # database.py / memory_service.py) and ignores this field even if the
    # frontend still sends it. Left on the model, still accepted, purely so
    # older app builds don't break with a validation error mid-rollout —
    # safe to delete from the frontend payload whenever convenient.
    history: List[ChatMessage] = Field(default_factory=list)
    imageUrls: List[str] = Field(default_factory=list)
    fileUrls: List[str] = Field(default_factory=list)    # NEW — non-image attachments (html/txt/md/etc)
    fileNames: List[str] = Field(default_factory=list)   # NEW — matching filenames, same order as fileUrls
    userid: str = None


# -------------------------------------------------------
# Doc Storage Models
# -------------------------------------------------------
class SaveDocRequest(BaseModel):
    filename: str
    content: str
    hint: str = None
    tags: List[str] = Field(default_factory=list)


class DocMetadata(BaseModel):
    id: str
    filename: str
    hint: str
    tags: List[str] = Field(default_factory=list)
    date: str
    size: int


class DocWithContent(BaseModel):
    id: str
    filename: str
    content: str
    hint: str = None
    tags: List[str] = Field(default_factory=list)
    date: str
    size: int


# -------------------------------------------------------
# Math detection helper
# -------------------------------------------------------
MATH_KEYWORDS = [
    "solve", "find x", "find y", "equation", "derivative", "integral",
    "calculus", "differentiate", "integrate", "plus", "minus", "times",
    "divide", "sum of", "subtract", "multiply", "squared", "square root",
    "cube of", "cube root", "% of", "percent of", "calculate",
]


def is_math_question(text: str) -> bool:
    t = text.lower()
    if any(kw in t for kw in MATH_KEYWORDS):
        return True
    if re.search(r'\d+\s*[\+\-\*\/\^]\s*\d+', t):
        return True
    return False


# -------------------------------------------------------
# Build a UTME-focused prompt so GPT-2 stays on topic
# -------------------------------------------------------
def build_prompt(user_query: str) -> str:
    return (
        "You are UTME26 AI, a brilliant Nigerian study assistant helping "
        "students prepare for JAMB UTME exams. You answer questions clearly "
        "and accurately about Math, English, Physics, Chemistry, Biology, "
        "History, Geography, Economics, and Computer Science.\n\n"
        f"Student: {user_query}\n"
        "UTME26 AI:"
    )


# ── FIXED: the equation short-circuit used to hijack plain-text messages.
# Two problems: (1) it called the solver on ANY input, even ordinary
# sentences, and (2) it only checked for the substring "Could not" — other
# failure phrasings like "No solution found for this equation." slipped
# through and got returned as if they were a real answer. Now we only even
# attempt the solver on text that actually looks like an equation, and we
# check against every known failure phrasing before trusting the result.
_EQUATION_FAILURE_MARKERS = (
    "could not", "no solution", "unable to solve", "invalid equation", "error",
)


def _looks_like_equation(text: str) -> bool:
    # Needs at least one digit AND (an '=' sign OR a math operator next to
    # a letter/number) — cheap filter that ordinary sentences won't pass,
    # while "2x + 5y - 3z = 11" or "3x^2 = 16" will.
    if not re.search(r"\d", text):
        return False
    return "=" in text or bool(re.search(r"[a-zA-Z]\s*[\^]|[\+\-\*/]\s*\d|\d\s*[\+\-\*/]", text))


def _equation_solved_ok(eq_answer: str) -> bool:
    if not eq_answer:
        return False
    lowered = eq_answer.lower()
    return not any(marker in lowered for marker in _EQUATION_FAILURE_MARKERS)


# ── NEW: tool-result memory enrichment ──────────────────────────────────
# The problem this fixes: remember_turn() only ever persisted the plain
# answer text. Everything a tool call actually found this turn — image
# URLs, search source links — existed only in the live response and was
# gone the moment the request finished. Ask "show me that image again"
# two turns later and the model has nothing to work with, because its own
# history never contained what it found.
#
# Fix: persist a RICHER string to history than what the user actually
# sees. The HTTP response to the frontend always uses the original, clean
# `result["answer"]` — untouched. Only what gets written to the DB (and
# therefore read back by build_memory() on a future turn) gets this
# additional internal-note suffix appended, so a later turn's model call
# can see "oh, I already found these images/sources" in its own past
# message instead of drawing a blank.
def _build_memory_reply(answer: str, sources: list = None, images: list = None) -> str:
    parts = [answer or ""]

    if sources:
        lines = []
        for s in sources[:10]:
            title = s.get("title") or s.get("name") or ""
            url = s.get("url") or s.get("link") or ""
            if url:
                lines.append(f"- {title}: {url}")
        if lines:
            parts.append(
                "\n\n[INTERNAL MEMORY NOTE — sources found this turn via "
                "search_web, for your own future recall only. Never quote "
                "this note or its formatting back to the user verbatim; if "
                "asked about these again later, use the real information "
                "naturally, or re-search if you need fresher results:\n"
                + "\n".join(lines) + "]"
            )

    if images:
        lines = []
        for i in images[:10]:
            title = i.get("title") or ""
            url = i.get("image") or i.get("url") or ""
            if url:
                lines.append(f"- {title}: {url}")
        if lines:
            parts.append(
                "\n\n[INTERNAL MEMORY NOTE — images found this turn via "
                "search_images, for your own future recall only. Never "
                "quote this note or its formatting back to the user "
                "verbatim; if asked to see these again later, you may "
                "reference that you found them, or call search_images "
                "again for a fresh/live gallery rather than assuming "
                "these old links still work:\n"
                + "\n".join(lines) + "]"
            )

    return "".join(parts)


# -------------------------------------------------------
# POST /ai-query — main chat endpoint (UNCHANGED path/schema for frontend;
# only an additive "sources" field is included in the response now)
# -------------------------------------------------------
@app.post("/ai-query")
async def ask_ai(request: QuestionRequest, db: Session = Depends(get_db)):
    user_question = request.query.strip()

    # NEW: history is loaded from the backend's own DB (keyed by userid),
    # not from request.history. If no userid was sent, this comes back
    # empty and the turn behaves statelessly, same as before this fix.
    chat_history = build_memory(db, request.userid)

    if not user_question:
        return {"label": "unknown", "answer": "Please type a question!", "sources": []}

    print(f"[REQUEST] /ai-query: {user_question[:100]!r} (memory={len(chat_history)} msgs, userid={request.userid!r})")

    # 1. Check for Math/Algebra first (Calculators don't need history) —
    # only attempted when the text actually looks like an equation.
    if _looks_like_equation(user_question):
        eq_answer = solve_equation_with_steps(user_question)
        print(f"[EQUATION] input looked equation-like, solver said: {eq_answer!r}")
        if _equation_solved_ok(eq_answer):
            # Equation answers are still saved to memory so a later "what
            # was that equation I solved earlier" question can find it.
            remember_turn(db, request.userid, user_question, eq_answer)
            return {"label": "algebra", "answer": eq_answer, "sources": []}

    # 2. AI provider chain (Groq -> fallback providers) — now with web search
    # sources returned alongside the answer when search was used.
    image_urls = request.imageUrls or []
    result = ask_gpt2(user_question, history=chat_history, image_urls=image_urls if image_urls else None, file_urls=request.fileUrls or None, file_names=request.fileNames or None, userid=request.userid)

    # NEW: persist this turn to the backend memory for next time. The DB
    # gets the enriched version (answer + tool-result notes); the frontend
    # still gets back the original, clean result["answer"] below.
    memory_reply = _build_memory_reply(result["answer"], result.get("sources"), result.get("images"))
    remember_turn(db, request.userid, user_question, memory_reply)

    return {
        "label": "general",
        "answer": result["answer"],
        "sources": result.get("sources", []),
        "images": result.get("images", []),
    }


# -------------------------------------------------------
# POST /ai-query-stream — SSE version of /ai-query. Same math-first check,
# same ask_gpt2 logic underneath, but streams real progress events as they
# actually happen instead of returning one blob at the end.
#
# Event shapes (each is one "data: <json>\n\n" line):
#   {"type": "status", "text": "..."}                    -- real progress
#   {"type": "final", "answer": "...", "sources": [...]} -- once, last
# -------------------------------------------------------
@app.post("/ai-query-stream")
async def ask_ai_stream(request: QuestionRequest, db: Session = Depends(get_db)):
    user_question = request.query.strip()

    # NEW: history loaded from backend DB memory, same as /ai-query above.
    chat_history = build_memory(db, request.userid)
    image_urls = request.imageUrls or []

    async def event_generator():
        if not user_question:
            yield f"data: {json.dumps({'type': 'final', 'answer': 'Please type a question!', 'sources': []})}\n\n"
            return

        print(f"[REQUEST] /ai-query-stream: {user_question[:100]!r} (memory={len(chat_history)} msgs, userid={request.userid!r})")

        # 1. Math/algebra short-circuit
        if _looks_like_equation(user_question):
            eq_answer = solve_equation_with_steps(user_question)
            print(f"[EQUATION] input looked equation-like, solver said: {eq_answer!r}")
            if _equation_solved_ok(eq_answer):
                yield f"data: {json.dumps({'type': 'status', 'text': 'Solving equation...'})}\n\n"
                yield f"data: {json.dumps({'type': 'final', 'answer': eq_answer, 'sources': []})}\n\n"
                remember_turn(db, request.userid, user_question, eq_answer)
                return

        # 2. Real generator from gpt2_test
        final_answer = None
        final_sources = []
        final_images = []
        for event in ask_gpt2_stream(user_question, history=chat_history, image_urls=image_urls if image_urls else None, file_urls=request.fileUrls or None, file_names=request.fileNames or None, userid=request.userid):
            if event["type"] == "status":
                yield f"data: {json.dumps({'type': 'status', 'text': event['text'], 'detail': event.get('detail'), 'icon': event.get('icon')})}\n\n"
            elif event["type"] == "final":
                final_answer = event["answer"]
                final_sources = event.get("sources", [])
                final_images = event.get("images", [])
                yield f"data: {json.dumps({'type': 'final', 'answer': event['answer'], 'sources': event.get('sources', []), 'images': event.get('images', []), 'file': event.get('file')})}\n\n"

        # NEW: persist the completed turn once the stream is done — the
        # "final" event above always carries the full answer text plus
        # sources/images, so there's nothing to accumulate chunk-by-chunk
        # here. Same enrichment as /ai-query: DB gets the richer version,
        # the SSE stream above already sent the clean answer to the user.
        if final_answer is not None:
            memory_reply = _build_memory_reply(final_answer, final_sources, final_images)
            remember_turn(db, request.userid, user_question, memory_reply)

    # FIXED: Moved 4 spaces to the left out of the inner event_generator block!
    return StreamingResponse(event_generator(), media_type="text/event-stream")


# -------------------------------------------------------
# POST /generate-question — UNCHANGED for frontend
# -------------------------------------------------------
@app.post("/generate-question")
async def generate_question(request: GenerateRequest):
    prompt = (
        f"Generate a JAMB UTME exam question about: {request.prompt}\n"
        "Question:"
    )
    try:
        result = ask_gpt2(prompt)
        answer = result["answer"]
        if answer and "unavailable" not in answer.lower():
            if "Question:" in answer:
                answer = answer.split("Question:")[-1].strip()
            return {"question": answer}
        return {"error": "Could not generate question. Try a more specific topic."}
    except Exception as e:
        return {"error": f"Generation failed: {e}"}


# -------------------------------------------------------
# GET / — UNCHANGED
# -------------------------------------------------------
@app.get("/")
async def root():
    return {
        "message": "UTME26 AI backend is running and ready!",
        "endpoints": ["/ai-query", "/generate-question", "/health"]
    }


# -------------------------------------------------------
# GET /health — UNCHANGED (used by loading screen), now also reports which
# fallback provider keys are loaded so you can see at a glance what's active
# -------------------------------------------------------
@app.get("/health")
async def health():
    from gpt2_test import GROQ_API_KEY, OPENROUTER_API_KEY, CEREBRAS_API_KEY, GEMINI_API_KEY
    return {
        "status": "healthy",
        "groq_key_loaded": GROQ_API_KEY is not None,
        "openrouter_key_loaded": OPENROUTER_API_KEY is not None,
        "cerebras_key_loaded": CEREBRAS_API_KEY is not None,
        "gemini_key_loaded": GEMINI_API_KEY is not None,
    }


# -------------------------------------------------------
# POST /user/{userid}/doc/save — save a doc (HTML, SVG, Markdown, etc.)
# -------------------------------------------------------
@app.post("/user/{userid}/doc/save")
async def save_user_doc(userid: str, request: SaveDocRequest):
    try:
        manager = UserDocManager(userid)
        doc_meta = manager.save_doc(
            filename=request.filename,
            content=request.content,
            hint=request.hint,
            tags=request.tags,
        )
        print(f"[DOC] saved {userid}/{request.filename} ({doc_meta['size']} bytes)")
        return {"status": "saved", "doc": doc_meta}
    except Exception as e:
        print(f"[DOC] save failed for {userid}/{request.filename}: {e}")
        return {"status": "error", "message": str(e)}, 400


# -------------------------------------------------------
# GET /user/{userid}/doc/search?q=hint — search user's docs by hint/tag
# -------------------------------------------------------
@app.get("/user/{userid}/doc/search")
async def search_user_docs(userid: str, q: str, limit: int = 10):
    try:
        manager = UserDocManager(userid)
        results = manager.search_by_hint(q, limit=limit)
        print(f"[DOC] search {userid} for '{q}' -> {len(results)} results")
        return {"status": "ok", "query": q, "results": results}
    except Exception as e:
        print(f"[DOC] search failed for {userid}: {e}")
        return {"status": "error", "message": str(e)}, 400


# -------------------------------------------------------
# GET /user/{userid}/doc/list — list all user's docs (metadata only)
# -------------------------------------------------------
@app.get("/user/{userid}/doc/list")
async def list_user_docs(userid: str):
    try:
        manager = UserDocManager(userid)
        docs = manager.list_all_docs()
        print(f"[DOC] list {userid} -> {len(docs)} docs")
        return {"status": "ok", "userid": userid, "docs": docs}
    except Exception as e:
        print(f"[DOC] list failed for {userid}: {e}")
        return {"status": "error", "message": str(e)}, 400


# -------------------------------------------------------
# GET /user/{userid}/doc/file/{docid} — retrieve full doc (content + metadata)
# -------------------------------------------------------
@app.get("/user/{userid}/doc/file/{docid}")
async def get_user_doc(userid: str, docid: str):
    try:
        manager = UserDocManager(userid)
        doc = manager.get_doc(docid)
        if not doc:
            return {"status": "not_found", "docid": docid}, 404
        print(f"[DOC] retrieved {userid}/{docid} ({doc['size']} bytes)")
        return {"status": "ok", "doc": doc}
    except Exception as e:
        print(f"[DOC] get failed for {userid}/{docid}: {e}")
        return {"status": "error", "message": str(e)}, 400


# -------------------------------------------------------
# DELETE /user/{userid}/doc/file/{docid} — delete a doc
# -------------------------------------------------------
@app.delete("/user/{userid}/doc/file/{docid}")
async def delete_user_doc(userid: str, docid: str):
    try:
        manager = UserDocManager(userid)
        deleted = manager.delete_doc(docid)
        if not deleted:
            return {"status": "not_found", "docid": docid}, 404
        print(f"[DOC] deleted {userid}/{docid}")
        return {"status": "deleted", "docid": docid}
    except Exception as e:
        print(f"[DOC] delete failed for {userid}/{docid}: {e}")
        return {"status": "error", "message": str(e)}, 400
