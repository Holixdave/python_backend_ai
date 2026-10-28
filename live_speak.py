# live_speak.py
# ──────────────────────────────────────────────────────────────────────────────
# POST /ai-live-stream — the "Live speak session" endpoint.
#
# It reuses EVERYTHING the normal chat already has (ask_gpt2_stream: intent
# routing, web search, file building, images, code, memory) and only changes how
# the result is DELIVERED to a phone that is talking out loud:
#
#   * Nothing is meant to be read. Instead of one long answer, the stream carries
#     `speak` events: short sentences the app hands to its TTS, in order.
#   * While tools run, the AI narrates what it is doing ("Hold on while I search
#     the web", "I'm building that file now", ...), so the user is never in silence.
#   * The real events (status / think / think_end / final) are STILL sent, unchanged,
#     so the app can fill the AI chat afterwards with the exact same bubbles, files,
#     images and code results a typed message would have produced.
#
# v2 (streaming): the answer is spoken AS IT IS WRITTEN. The model's `text` deltas are cut into short sentences
# and sent one by one as `speak` events, so the first sentence is heard while the rest is still being made.
# Tool / file / image / code progress is forwarded live (tool_call, tool_result, block) so the phone can show
# what is being built while it is being built.
#
# Event shapes (each is one "data: <json>\n\n" line):
#   {"type": "speak", "text": "...", "kind": "ack|tool|wait|answer|ask|error", "n": 3}  -- say this out loud AND show it as a caption
#   {"type": "tool_call" | "tool_result" | "block" | "ask_user", ...}        -- identical to /ai-query-stream (live progress)
#   {"type": "status" | "think" | "think_end" | "final", ...}                -- identical to /ai-query-stream
#   {"type": "done"}                                                          -- always last
#
# Request body: same QuestionRequest as /ai-query-stream (query, userid, imageUrls, ...) plus the voice fields below.
# ──────────────────────────────────────────────────────────────────────────────
import asyncio
import json
import re
import threading

from typing import List, Optional

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database import get_db
from memory_service import build_memory, remember_turn
from gpt2_test import ask_gpt2_stream

router = APIRouter()

# Light touch on purpose: the answer text still goes into the chat bubble afterwards, so it stays a normal,
# well-formatted answer. Only the SPOKEN copy is derived from it (see _to_speech).
LIVE_SPEAK_PROMPT = (
    "LIVE VOICE SESSION: the user is talking to you out loud and your reply will be spoken by a voice, "
    "then kept in their chat. Be conversational and concise: lead with the answer in plain sentences, "
    "keep it short unless they asked for depth, and do not pad. If you build a file, create an image or run "
    "code, say in one short sentence what you made — the full result is saved in their chat, so never read "
    "code, file contents, tables, long lists or links out loud.\n"
    "SPEECH-RECOGNITION RULES: the user's words were typed by a speech recogniser, so some words, names or numbers "
    "may be WRONG. If the request is unclear, or one word looks mis-heard (a name, a subject, a number, a term "
    "that does not fit), do NOT guess and do NOT build anything: ask ONE short confirming question first "
    "(for example: \"Did you mean the Lekki Headmaster?\"). Once they confirm, carry on straight away.\n"
    "SPEAKING RULES: start with the answer in one or two short sentences, then stop unless they asked for more; "
    "offer the rest in one short line (\"Want the details?\"). Never read a whole long answer aloud: say the key "
    "point and tell them the rest is on their screen."
)


def _turn_note(req) -> str:
    """Extra per-turn rules from what the phone knows about this turn."""
    notes = []
    if getattr(req, "interrupted", False):
        heard = (req.spokenSoFar or "").strip()
        notes.append(
            "The user INTERRUPTED your last spoken reply. "
            + (f"They only heard this much of it: \"{heard[-300:]}\". " if heard else "")
            + "Do not repeat what they already heard; answer what they just said, straight away, in one or two sentences."
        )
    conf = getattr(req, "heardConfidence", None)
    if conf is not None and conf < 0.6:
        notes.append("The recogniser was NOT confident about this sentence: confirm what you understood before acting on it.")
    return ("\n" + " ".join(notes)) if notes else ""

# What the AI says when it starts a tool. Same wording every time keeps it feeling natural.
_TOOL_PHRASES = {
    "search_web": "Hold on while I search the web.",
    "search_images": "Let me look for some pictures.",
    "fetch_page_preview_image": "Let me grab the picture from that page.",
    "fetch_webpage": "I'm opening that page to read it.",
    "fetch_document": "I'm opening that document to read it.",
    "search_github": "I'm searching GitHub for that.",
    "fetch_github_file": "I'm pulling that file from GitHub.",
    "analyze_image": "I'm taking a look at the image.",
    "build_file": "I'm building that file now.",
    "build_multiple_files": "I'm building those files now.",
    "build_zip_file": "I'm packing everything into a zip.",
    "edit_file": "I'm making that edit now.",
    "edit_file_by_intent": "I'm making that edit now.",
    "workspace_edit_by_intent": "I'm making that edit now.",
    "workspace_copy": "I'm copying that file into your workspace.",
    "remove_background": "I'm removing the background now.",
    "create_image": "Give me a moment while I create that image.",
    "generate_image": "Give me a moment while I create that image.",
    "run_code": "I'm running that code now.",
    "redisplay_file": "Let me bring that file back up.",
    "redisplay_images": "Let me bring those pictures back up.",
    "save_study_note": "I'm saving that to your notes.",
    "save_user_note": "I'll remember that.",
    "schedule_reminder": "I'm setting that reminder.",
}

_STATUS_TEXT_PHRASES = (
    ("reading attached file", "I'm reading your file."),
    ("looking at the image", "I'm looking at the image."),
    ("checking jamb", "Checking my JAMB notes."),
    ("solving equation", "Let me work that equation out."),
)

_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF\U0000FE0F\U0000200D]+",
    flags=re.UNICODE,
)


def _to_speech(answer: str, has_extras: bool, max_chars: int = 700) -> str:
    """Turn a markdown answer into something a voice can read naturally."""
    t = answer or ""
    # whole visual blocks the chat renders (svg / html widgets) have no spoken form
    t = re.sub(r"<svg.*?</svg>", " ", t, flags=re.DOTALL | re.IGNORECASE)
    t = re.sub(r"<(html|iframe|script|style)[\s\S]*?</\1>", " ", t, flags=re.IGNORECASE)
    cut_code = bool(re.search(r"```", t))
    t = re.sub(r"```[\s\S]*?```", " ", t)                    # fenced code
    t = re.sub(r"`([^`]*)`", r"\1", t)                       # inline code → its words
    t = re.sub(r"<[^>]+>", " ", t)                           # leftover tags
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", t)              # images
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)           # [text](url) → text
    t = re.sub(r"https?://\S+", " ", t)                      # bare urls
    t = re.sub(r"^\s*\|.*\|\s*$", " ", t, flags=re.MULTILINE)  # table rows
    t = re.sub(r"^\s{0,3}#{1,6}\s*", "", t, flags=re.MULTILINE)  # headings
    t = re.sub(r"^\s*[-*•]\s+", "", t, flags=re.MULTILINE)    # bullets
    t = re.sub(r"(\*\*|__|\*|_|~~)", "", t)                  # emphasis
    t = _EMOJI_RE.sub(" ", t)
    t = re.sub(r"\s*\n+\s*", ". ", t)                        # paragraph breaks → a pause
    t = re.sub(r"\.\s*\.", ".", t)
    t = re.sub(r"\s{2,}", " ", t).strip(" .")
    if not t:
        return "Done. I've put everything in your chat." if (has_extras or cut_code) else ""
    suffix = ""
    if len(t) > max_chars:
        cut = t[:max_chars]
        end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
        t = (cut[: end + 1] if end > 200 else cut.rsplit(" ", 1)[0]).strip()
        suffix = " The rest is in your chat."
    elif has_extras or cut_code:
        suffix = " I've put the details in your chat."
    if t and t[-1] not in ".!?":
        t += "."
    return t + suffix


SPOKEN_BUDGET = 380        # chars of ANSWER spoken per turn; the rest is only in the chat
_BOUNDARY = re.compile(r'(?<=[.!?])["\')\]]*\s+|\n+')
_MIN_SENTENCE = 22         # glue tiny pieces ("Yes." / "e.g.") onto the next sentence


def _clean_sentence(t: str) -> str:
    """Markdown / links / emoji out of ONE sentence so a voice can read it."""
    t = re.sub(r"`([^`]*)`", r"\1", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"^\s{0,3}#{1,6}\s*", "", t)
    t = re.sub(r"^\s*[-*•]\s+", "", t)
    t = re.sub(r"^\s*\d+[.)]\s+", "", t)
    t = re.sub(r"(\*\*|__|\*|_|~~)", "", t)
    t = _EMOJI_RE.sub(" ", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t


class _SpeechSplitter:
    """Cuts the model's streamed text into short spoken sentences, skipping fenced code, and stops at a budget."""

    def __init__(self, budget: int = SPOKEN_BUDGET):
        self.raw = ""          # not yet classified (fence detection)
        self.text = ""         # prose waiting for a sentence end
        self.in_code = False
        self.had_code = False
        self.spoken = 0
        self.over = False      # budget reached: the rest is chat-only
        self.sent = 0
        self.budget = budget

    def _take(self, sentence: str, out: list):
        if self.over:
            return
        c = _clean_sentence(sentence)
        if not c or not re.search(r"[A-Za-z0-9]", c):
            return
        c = c.rstrip(":;,— -")
        if c[-1:] not in (".", "!", "?"):
            c += "."
        out.append(c)
        self.sent += 1
        self.spoken += len(c)
        if self.spoken >= self.budget:
            self.over = True

    def feed(self, delta: str) -> list:
        out: list = []
        self.raw += delta
        while True:
            if self.in_code:
                j = self.raw.find("```")
                if j < 0:
                    self.raw = self.raw[-2:] if self.raw.endswith("`") else ""
                    break
                self.raw = self.raw[j + 3:]
                self.in_code = False
            else:
                j = self.raw.find("```")
                if j < 0:
                    hold = 0
                    while hold < 2 and self.raw.endswith("`" * (hold + 1)):
                        hold += 1
                    self.text += self.raw[: len(self.raw) - hold]
                    self.raw = self.raw[len(self.raw) - hold:]
                    break
                self.text += self.raw[:j]
                self.raw = self.raw[j + 3:]
                self.in_code = True
                self.had_code = True
        pos = 0
        while True:
            m = _BOUNDARY.search(self.text, pos)
            if not m:
                break
            piece = self.text[: m.start()]
            if (len(piece.strip()) < _MIN_SENTENCE or re.search(r"(?i)\b(e\.g|i\.e|vs|etc|approx|no)$", piece.strip())) and "\n" not in m.group(0):
                pos = m.end()          # too short: keep gathering
                continue
            self._take(piece, out)
            self.text = self.text[m.end():]
            pos = 0
        return out

    def flush(self) -> list:
        out: list = []
        if not self.in_code:
            self.text += self.raw
        self.raw = ""
        if self.text.strip():
            self._take(self.text, out)
        self.text = ""
        return out


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, default=str)}\n\n"


def _tool_phrase(event: dict):
    tool = event.get("tool")
    if tool and tool in _TOOL_PHRASES:
        return _TOOL_PHRASES[tool]
    low = (event.get("text") or "").lower()
    for needle, phrase in _STATUS_TEXT_PHRASES:
        if needle in low:
            return phrase
    if event.get("icon") == "search" and not tool:
        return _TOOL_PHRASES["search_web"]
    return None


class LiveRequest(BaseModel):
    """Same fields as main.QuestionRequest, so the app sends the exact body it already sends to /ai-query-stream."""
    query: str
    history: List[dict] = Field(default_factory=list)   # ignored: memory lives in the backend DB
    imageUrls: List[str] = Field(default_factory=list)
    fileUrls: List[str] = Field(default_factory=list)
    fileNames: List[str] = Field(default_factory=list)
    userid: Optional[str] = None
    # ── voice fields (all optional: an old app that does not send them still works) ──
    voice: bool = True                       # the text was SPOKEN by the user
    interrupted: bool = False                # the user cut the previous spoken reply short
    spokenSoFar: Optional[str] = None        # what the user actually heard of that reply
    heardConfidence: Optional[float] = None  # 0..1 from the speech recogniser, when it gives one


_EQ_FAIL = ("could not", "no solution", "unable to solve", "invalid equation", "error")


def _looks_like_equation(text: str) -> bool:
    if not re.search(r"\d", text):
        return False
    return "=" in text or bool(re.search(r"[a-zA-Z]\s*[\^]|[\+\-\*/]\s*\d|\d\s*[\+\-\*/]", text))


def _equation_ok(answer: str) -> bool:
    return bool(answer) and not any(m in answer.lower() for m in _EQ_FAIL)


def _memory_reply(answer: str, sources=None, images=None) -> str:
    """Same idea as main._build_memory_reply: remember what the tools found, not only the words."""
    parts = [answer or ""]
    src = [f"- {s.get('title') or ''}: {s.get('url') or s.get('link') or ''}" for s in (sources or [])[:10]
           if (s.get("url") or s.get("link"))]
    if src:
        parts.append("\n\n[INTERNAL MEMORY NOTE — sources found this turn, for your own recall only; never quote this note:\n"
                     + "\n".join(src) + "]")
    img = [f"- {i.get('title') or ''}: {i.get('image') or i.get('url') or ''}" for i in (images or [])[:10]
           if (i.get("image") or i.get("url"))]
    if img:
        parts.append("\n\n[INTERNAL MEMORY NOTE — images found this turn, for your own recall only; never quote this note:\n"
                     + "\n".join(img) + "]")
    return "".join(parts)


@router.post("/ai-live-stream")
async def ai_live_stream(request: LiveRequest, db: Session = Depends(get_db)):
    from equation_solver import solve_equation_with_steps as solve_equation
    build_memory_reply = _memory_reply
    looks_like_equation = _looks_like_equation
    equation_ok = _equation_ok
    question = (request.query or "").strip()
    history = build_memory(db, request.userid)
    image_urls = request.imageUrls or []

    async def gen():
        if not question:
            yield _sse({"type": "speak", "text": "I didn't catch anything. Try again.", "kind": "error"})
            yield _sse({"type": "done"})
            return

        print(f"[LIVE] /ai-live-stream: {question[:100]!r} (memory={len(history)} msgs, userid={request.userid!r})")

        # 1. Same math short-circuit as the normal chat.
        if looks_like_equation(question):
            eq = solve_equation(question)
            if equation_ok(eq):
                yield _sse({"type": "status", "text": "Solving equation...", "icon": "thinking"})
                yield _sse({"type": "speak", "text": "Let me work that equation out.", "kind": "tool"})
                yield _sse({"type": "speak", "text": _to_speech(eq, False), "kind": "answer"})
                yield _sse({"type": "final", "answer": eq, "segments": [], "sources": [], "images": [],
                            "files": [], "codes": [], "suggestions": []})
                remember_turn(db, request.userid, question, eq)
                yield _sse({"type": "done"})
                return

        # 2. The real pipeline, pumped from a worker thread so we can talk while it is quiet.
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()

        def pump():
            try:
                for ev in ask_gpt2_stream(
                    question, history=history, image_urls=image_urls or None,
                    file_urls=request.fileUrls or None, file_names=request.fileNames or None,
                    userid=request.userid, extra_system_prompt=LIVE_SPEAK_PROMPT + _turn_note(request),
                ):
                    loop.call_soon_threadsafe(q.put_nowait, ev)
            except Exception as e:  # noqa: BLE001
                print(f"[LIVE] pipeline error: {e}")
                loop.call_soon_threadsafe(q.put_nowait, {"type": "__error__"})
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        threading.Thread(target=pump, daemon=True).start()

        said = set()
        waits = 0
        thought = False
        final_event = None
        splitter = _SpeechSplitter()
        n_spoken = 0
        streamed = False       # did any answer text arrive as deltas?
        extras_seen = False    # file / image / code made this turn
        told_rest = False

        def speak(text, kind="answer"):
            nonlocal n_spoken
            n_spoken += 1
            return _sse({"type": "speak", "text": text, "kind": kind, "n": n_spoken})

        while True:
            try:
                ev = await asyncio.wait_for(q.get(), timeout=14)
            except asyncio.TimeoutError:
                # Long tool run: don't leave the user in silence.
                if waits < 3:
                    waits += 1
                    line = ("Still working on it." if waits == 1 else
                            "This is taking a little longer, almost there." if waits == 2 else
                            "Thanks for waiting, nearly done.")
                    yield _sse({"type": "speak", "text": line, "kind": "wait"})
                else:
                    yield ": keepalive\n\n"
                continue
            if ev is None:
                break
            kind = ev.get("type")

            if kind == "__error__":
                yield _sse({"type": "speak", "text": "Sorry, something went wrong on my side. Please try again.", "kind": "error"})
                continue

            if kind == "status":
                yield _sse({"type": "status", "text": ev.get("text"), "detail": ev.get("detail"), "icon": ev.get("icon"),
                            "tool": ev.get("tool"), "args": ev.get("args"), "call": ev.get("call")})
                phrase = _tool_phrase(ev)
                if phrase and phrase not in said:
                    said.add(phrase)
                    yield _sse({"type": "speak", "text": phrase, "kind": "tool"})
            elif kind == "think":
                yield _sse({"type": "think", "step": ev.get("step"), "text": ev.get("text"), "icon": ev.get("icon")})
                if not thought:
                    thought = True
                    yield _sse({"type": "speak", "text": "Let me think this through.", "kind": "ack"})
            elif kind == "think_end":
                yield _sse({"type": "think_end"})
            elif kind == "text":
                delta = ev.get("delta") or ev.get("text") or ""
                if delta:
                    streamed = True
                    for sent in splitter.feed(delta):
                        yield speak(sent)
                    if splitter.over and not told_rest:
                        told_rest = True
            elif kind in ("tool_call", "tool_result", "block", "ask_user"):
                if kind == "block" and ev.get("kind") in ("images", "file", "code"):
                    extras_seen = True
                yield _sse(ev)
                if kind == "tool_call":
                    phrase = _tool_phrase(ev)
                    if phrase and phrase not in said and not streamed:
                        said.add(phrase)
                        yield speak(phrase, "tool")
            elif kind == "final":
                final_event = ev
                files = ev.get("files") or ([ev["file"]] if ev.get("file") else [])
                has_extras = bool(
                    [f for f in files if isinstance(f, dict) and f.get("success")]
                    or ev.get("images") or ev.get("codes")
                )
                has_extras = has_extras or extras_seen or splitter.had_code
                if streamed:
                    # Answer was spoken live; say only what is left, then point at the chat.
                    for sent in splitter.flush():
                        yield speak(sent)
                    if told_rest:
                        yield speak("The rest is in your chat.", "answer")
                    elif has_extras:
                        yield speak("I've put the details in your chat.", "answer")
                else:
                    spoken = _to_speech(ev.get("answer", ""), has_extras, max_chars=SPOKEN_BUDGET)
                    if spoken:
                        yield speak(spoken)
                yield _sse({
                    "type": "final", "answer": ev["answer"], "segments": ev.get("segments", []),
                    "suggestions": ev.get("suggestions", []), "sources": ev.get("sources", []),
                    "images": ev.get("images", []), "file": ev.get("file"), "files": ev.get("files", []),
                    "codes": ev.get("codes", []),
                })

        if final_event is not None:
            reply = build_memory_reply(final_event["answer"], final_event.get("sources"), final_event.get("images"))
            remember_turn(db, request.userid, question, reply)
        yield _sse({"type": "done"})

    return StreamingResponse(gen(), media_type="text/event-stream")
