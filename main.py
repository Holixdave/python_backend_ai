# main.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List
import os
import json
import re
from math_solver import solve_math_with_explanation
from equation_solver import solve_equation_with_steps
from ai_backend_core import ask_gpt2_stream
from user_doc_manager import UserDocManager

app = FastAPI(
    title="UTME26 AI Backend",
    description="Brilliant AI Study Assistant"
)

# -------------------------------------------------------
# Pydantic models — DO NOT CHANGE (frontend depends on these)
# -------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class QuestionRequest(BaseModel):
    query:     str
    history:   List[ChatMessage] = Field(default_factory=list)
    imageUrls: List[str]         = Field(default_factory=list)
    userid:    str                = None


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
    import re
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


# -------------------------------------------------------
# POST /ai-query  — main chat endpoint (UNCHANGED path/schema for frontend;
# only an additive "sources" field is included in the response now)
# -------------------------------------------------------
@app.post("/ai-query")
async def ask_ai(request: QuestionRequest):
    user_question = request.query.strip()

    # Convert Pydantic history objects to simple dictionaries for the AI
    chat_history = [m.model_dump() for m in request.history] if request.history else []

    if not user_question:
        return {"label": "unknown", "answer": "Please type a question!", "sources": []}

    print(f"[REQUEST] /ai-query: {user_question[:100]!r} (history={len(chat_history)} msgs)")

    # 1. Check for Math/Algebra first (Calculators don't need history) —
    #    only attempted when the text actually looks like an equation.
    if _looks_like_equation(user_question):
        eq_answer = solve_equation_with_steps(user_question)
        print(f"[EQUATION] input looked equation-like, solver said: {eq_answer!r}")
        if _equation_solved_ok(eq_answer):
            return {"label": "algebra", "answer": eq_answer, "sources": []}

    # 2. AI provider chain (Groq -> fallback providers) — now with web search
    #    sources returned alongside the answer when search was used.
    image_urls = request.imageUrls or []
    result = ask_gpt2(user_question, history=chat_history, image_urls=image_urls if image_urls else None, userid=request.userid)

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
#   {"type": "status", "text": "..."}                         -- real progress
#   {"type": "final", "answer": "...", "sources": [...]}       -- once, last
# -------------------------------------------------------
@app.post("/ai-query-stream")
async def ask_ai_stream(request: QuestionRequest):
    user_question = request.query.strip()
    chat_history = [m.model_dump() for m in request.history] if request.history else []
    image_urls = request.imageUrls or []

    async def event_generator():
        if not user_question:
            yield f"data: {json.dumps({'type': 'final', 'answer': 'Please type a question!', 'sources': []})}\n\n"
            return

        print(f"[REQUEST] /ai-query-stream: {user_question[:100]!r} (history={len(chat_history)} msgs)")

        # 1. Math/algebra short-circuit — same fixed guard as /ai-query
        if _looks_like_equation(user_question):
            eq_answer = solve_equation_with_steps(user_question)
            print(f"[EQUATION] input looked equation-like, solver said: {eq_answer!r}")
            if _equation_solved_ok(eq_answer):
                yield f"data: {json.dumps({'type': 'status', 'text': 'Solving equation...'})}\n\n"
                yield f"data: {json.dumps({'type': 'final', 'answer': eq_answer, 'sources': []})}\n\n"
                return

        # 2. Real generator from gpt2_test — every status event here reflects
        #    an actual step that just ran (classifier call, real search hit,
        #    provider call), nothing synthetic.
        for event in ask_gpt2_stream(user_question, history=chat_history, image_urls=image_urls if image_urls else None, userid=request.userid):
            if event["type"] == "status":
                yield f"data: {json.dumps({'type': 'status', 'text': event['text'], 'detail': event.get('detail')})}\n\n"
            elif event["type"] == "final":
                yield f"data: {json.dumps({'type': 'final', 'answer': event['answer'], 'sources': event.get('sources', []), 'images': event.get('images', [])})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# -------------------------------------------------------
# POST /generate-question  — UNCHANGED for frontend
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
# GET /  — UNCHANGED
# -------------------------------------------------------
@app.get("/")
async def root():
    return {
        "message": "UTME26 AI backend is running and ready!",
        "endpoints": ["/ai-query", "/generate-question", "/health"]
    }
# -------------------------------------------------------
# GET /health  — UNCHANGED (used by loading screen), now also reports which
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
        print(f"[DOC] search {userid} for '{q}' → {len(results)} results")
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
        print(f"[DOC] list {userid} → {len(docs)} docs")
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
