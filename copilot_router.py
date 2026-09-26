# copilot_router.py
"""
OpenAI-compatible endpoints so this backend can be plugged into VS Code
extensions (Continue, Cline, etc.) as a custom "OpenAI Compatible" provider.

Design notes:
- GET /v1/models (and /models) returns a fake fixed list so clients that
  probe it before allowing chat don't 404 — see FAKE_MODELS below.
- POST /v1/chat/completions (and /chat/completions) uses this backend's own
  DB-backed memory per userid (build_memory/remember_turn), the same as
  /ai-query — it does NOT replay the client's resent `messages` history,
  only the latest user message. See _resolve_userid for how userid is
  determined from a header, body field, or the Authorization bearer token.
- Any "system" messages the client sends (e.g. Cline's tool-call-format
  instructions) are forwarded into ask_gpt2/ask_gpt2_stream via
  extra_system_prompt, appended after this backend's own system prompt —
  see _client_system_prompt and gpt2_test.py's _ask_gpt2_core.
- Streaming is NOT real token-by-token generation. ask_gpt2_stream()
  yields backend "status" events (e.g. "Searching web...") followed by a
  single "final" event containing the whole answer. To stay compatible
  with streaming clients, status text is sent as small delta chunks (so
  something visibly happens), then the complete final answer is sent as
  one last delta chunk, then the standard finish/[DONE] sequence. If you
  don't want the status lines mixed into the visible reply, set
  INCLUDE_STATUS_IN_STREAM = False below.
"""
import json
import time
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from gpt2_test import ask_gpt2, ask_gpt2_stream
from database import get_db
from memory_service import build_memory, remember_turn

router = APIRouter()

# Some OpenAI-compatible clients (Cline included) call GET /v1/models before
# letting you chat — either to populate a model dropdown or just to sanity-
# check the connection. Real model IDs don't matter here since this backend
# ignores whatever `model` string is sent anyway; we just need SOMETHING in
# the list so that probe request succeeds instead of 404ing.
FAKE_MODELS = ["utme26-ai"]


@router.get("/v1/models")
@router.get("/models")
async def list_models():
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "created": now, "owned_by": "you"}
            for m in FAKE_MODELS
        ],
    }

# Flip to False if you'd rather stream stay silent until the final answer.
INCLUDE_STATUS_IN_STREAM = True

# Used ONLY when a request supplies no userid at all (see _resolve_userid).
# Every request that falls back to this shares ONE memory bucket with every
# other such request — fine for solo testing, wrong the moment a second
# person points a client at this endpoint without setting an id.
DEFAULT_USERID = "vscode-default"


class OAIMessage(BaseModel):
    role: str
    content: str = ""


class ChatCompletionsRequest(BaseModel):
    model: str = "utme26-ai"
    messages: List[OAIMessage]
    stream: bool = False
    userid: Optional[str] = None  # if your client can send a custom field
    user: Optional[str] = None    # OpenAI's standard "end-user id" field


def _resolve_userid(body: ChatCompletionsRequest, header_userid: Optional[str], bearer_token: Optional[str]) -> str:
    """
    Priority: X-User-Id header > body.userid > body.user > Authorization
    bearer token > DEFAULT_USERID.

    The bearer-token fallback exists because Cline (and most OpenAI-
    compatible clients) always send SOME value in the "API Key" field as
    an Authorization: Bearer <value> header, whether or not it's a real
    key — this backend doesn't validate it as a key, it just reuses
    whatever string is there as a stable per-person id. So: put a unique
    string like "james-vscode" in Cline's API Key field and that alone is
    enough, with no header-support gambling. Continue can also just set
    a custom `X-User-Id` header directly in its config if you prefer that.
    """
    return header_userid or body.userid or body.user or bearer_token or DEFAULT_USERID


def _last_user_message(messages: List[OAIMessage]) -> str:
    """
    We deliberately do NOT forward the client's resent conversation as
    history. This backend keeps its own DB-backed memory per userid (same
    pattern as /ai-query and /ai-query-stream in main.py) — replaying the
    client's copy on top of that would just create two disagreeing
    versions of the conversation. Only the newest user message is used;
    prior turns come from the backend's own memory for this userid.
    """
    user_msgs = [m.content for m in messages if m.role == "user"]
    if not user_msgs:
        raise ValueError("No user message found in `messages`.")
    return user_msgs[-1]


def _client_system_prompt(messages: List[OAIMessage]) -> Optional[str]:
    """
    Concatenates any "system" role messages the client sent (Cline sends
    one big one describing its tool-call format — read_file/write_to_file/
    replace_in_file/etc — so it can parse a diff out of the reply and show
    you an Accept button). Returned as-is for _ask_gpt2_core to fold into
    its own system prompt via extra_system_prompt. None if the client sent
    no system message (e.g. Continue's plain chat mode usually doesn't).
    """
    sys_msgs = [m.content for m in messages if m.role == "system" and m.content]
    return "\n\n".join(sys_msgs) if sys_msgs else None


def _sse_chunk(completion_id: str, model: str, delta: dict, finish_reason=None) -> str:
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {"index": 0, "delta": delta, "finish_reason": finish_reason}
        ],
    }
    return f"data: {json.dumps(payload)}\n\n"


@router.post("/v1/chat/completions")
@router.post("/chat/completions")  # some extensions omit the /v1 prefix
async def chat_completions(
    request: ChatCompletionsRequest,
    db: Session = Depends(get_db),
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
    authorization: Optional[str] = Header(default=None),
):
    try:
        prompt = _last_user_message(request.messages)
    except ValueError as e:
        return {"error": {"message": str(e), "type": "invalid_request_error"}}

    bearer_token = None
    if authorization and authorization.lower().startswith("bearer "):
        bearer_token = authorization[7:].strip() or None

    userid = _resolve_userid(request, x_user_id, bearer_token)
    client_system_prompt = _client_system_prompt(request.messages)
    chat_history = build_memory(db, userid)
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    print(f"[COPILOT] /v1/chat/completions userid={userid!r} stream={request.stream} memory={len(chat_history)} msgs client_system_prompt={'yes' if client_system_prompt else 'no'}")

    # Deferred import: main.py imports this router at startup, so importing
    # main.py here at module load time would be circular. By request time
    # (when this function actually runs) main.py is fully loaded, so a
    # local import inside the handler works fine and lets us reuse the
    # exact same "richer memory record" helper /ai-query already uses.
    from main import _build_memory_reply

    # ---- Non-streaming: one request, one JSON response -------------------
    if not request.stream:
        result = ask_gpt2(prompt, history=chat_history, userid=userid, extra_system_prompt=client_system_prompt, copilot_mode=True)
        memory_reply = _build_memory_reply(result["answer"], result.get("sources"), result.get("images"))
        remember_turn(db, userid, prompt, memory_reply)
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result["answer"]},
                    "finish_reason": "stop",
                }
            ],
        }

    # ---- Streaming: SSE chunks in OpenAI's chat.completion.chunk shape ---
    def event_stream():
        yield _sse_chunk(completion_id, request.model, {"role": "assistant"})

        final_answer = ""
        final_sources = []
        final_images = []
        for event in ask_gpt2_stream(prompt, history=chat_history, userid=userid, extra_system_prompt=client_system_prompt, copilot_mode=True):
            if INCLUDE_STATUS_IN_STREAM and event["type"] == "status" and event.get("text"):
                yield _sse_chunk(completion_id, request.model, {"content": f"\n> {event['text']}\n"})
            elif event["type"] == "final":
                final_answer = event.get("answer", "")
                final_sources = event.get("sources", [])
                final_images = event.get("images", [])

        yield _sse_chunk(completion_id, request.model, {"content": final_answer})
        yield _sse_chunk(completion_id, request.model, {}, finish_reason="stop")
        yield "data: [DONE]\n\n"

        # Persisted after the client has everything it needs — same
        # ordering /ai-query-stream in main.py already uses.
        memory_reply = _build_memory_reply(final_answer, final_sources, final_images)
        remember_turn(db, userid, prompt, memory_reply)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
