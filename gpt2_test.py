#!/usr/bin/env python3


# gpt2_test.py — Multi-provider fallback edition
# ─────────────────────────────────────────────────────────────────────────────
# WHAT CHANGED FROM THE OLD VERSION
#   1. ask_gpt2() / ask_with_vision() no longer hit Groq only. They walk a
#      chain of providers (Groq -> OpenRouter free models -> Cerebras free
#      tier -> Gemini) and fail over automatically. If Groq is out of
#      credits or rate-limited, the user never sees an error — the next
#      provider in the chain just answers instead.
#   2. search_web() walks a chain too: ddgs -> Brave -> Tavily. If every
#      search engine fails, we silently return no web context instead of
#      stuffing a "Search failed: ..." string into the prompt (which used
#      to sometimes leak through to the user as the actual answer).
#   3. ask_gpt2() now returns a dict: {"answer": str, "sources": list,
#      "provider": str}. main.py uses this to add an optional "sources"
#      field to the API response — nothing existing was removed.
#   4. Every new provider is OPTIONAL — controlled by env var presence. If
#      you never set OPENROUTER_API_KEY / CEREBRAS_API_KEY / GEMINI_API_KEY
#      / BRAVE_API_KEY / TAVILY_API_KEY, this behaves exactly like before,
#      just with Groq retried smarter.
#
#   FIXES APPLIED IN THIS PASS (from the Test 1-4 debug session):
#   5. The "Thinking it through..." / "Writing answer..." status event no
#      longer fabricates a fake explanation about temperature/token
#      budgets. That text was hardcoded and shown unconditionally — it
#      never reflected anything the model actually did. Real reasoning
#      (the <think> block, when the model returns one) is still split out
#      and shown as its own separate status events further down; this was
#      just decorative filler sitting in front of it.
#   6. wants_image is now forced to False whenever the user already
#      uploaded image(s) for this turn. Previously the intent classifier
#      had no idea an image had already been provided, so prompts that
#      merely contained words like "image" or "screenshot" would trigger
#      an unrelated online image search on top of the real vision
#      analysis — confusing and wasteful.
#   7. ask_with_vision() no longer trusts a vision provider just because
#      the HTTP call returned 200. Some vision models (notably Groq's
#      llama-4-scout on multi-image requests) can return a normal 200
#      response where the model's own content is a refusal like "I'm
#      unable to view the images" — that used to be accepted as the real
#      answer. Now that's detected and treated as a failure: the next
#      provider is tried, and if every provider refuses a multi-image
#      batch, we fall back to describing each image separately and
#      merging the results.
# ─────────────────────────────────────────────────────────────────────────────

import os
import re
import time
import json
import requests
from datetime import datetime, timedelta, timezone
from typing import Optional
from user_doc_manager import UserDocManager
import random
# ---------------------------------------------------------------------------
# CONFIG — API keys. Only GROQ_API_KEY is required. Everything else is an
# optional fallback: if the env var isn't set, that provider is just skipped.
# ---------------------------------------------------------------------------
GROQ_API_KEY:       Optional[str] = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")
CEREBRAS_API_KEY:   Optional[str] = os.getenv("CEREBRAS_API_KEY")
GEMINI_API_KEY:     Optional[str] = os.getenv("GEMINI_API_KEY")
BRAVE_API_KEY:      Optional[str] = os.getenv("BRAVE_API_KEY")
TAVILY_API_KEY:     Optional[str] = os.getenv("TAVILY_API_KEY")

MAX_RETRIES_PER_PROVIDER: int = 2     # quick retries before moving to the next provider
RETRY_BASE_DELAY:         float = 1.0
REQUEST_TIMEOUT:          int   = 120  # raised from 45 — big max_tokens generations (long code, big tool-call JSON) take longer

# NEW — real answers/tool-calls no longer get artificially capped short.
# This is what was cutting off long code (limited to a few hundred lines)
# and truncating large tool-call JSON payloads mid-array before they could
# close. Raised well above what a single answer realistically needs so the
# model's own context window is the real limit, not this number.
MAX_ANSWER_TOKENS: int = 16000

if not OPENROUTER_API_KEY:
    raise EnvironmentError("OPENROUTER_API_KEY environment variable is not set.")

# ---------------------------------------------------------------------------
# PROVIDER CHAINS (OpenAI-compatible /chat/completions shape)
# Order = priority. First enabled provider that succeeds wins.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# PROVIDER CHAINS (OpenAI-compatible /chat/completions shape)
# Order = priority. First enabled provider that succeeds wins.
#
# OpenRouter is now PRIMARY: it walks its own list of free models (tested
# working via test_models.py) top-to-bottom before ever touching Groq.
# Groq only gets called if every OpenRouter free model in this chain fails
# or is rate-limited. Cerebras stays as the last resort.
# ---------------------------------------------------------------------------
GEMINI_MODELS_CHAIN = [
    "models/gemini-flash-latest",
    "models/gemini-3.5-flash",
    "models/gemini-3-flash-preview",
    "models/gemini-3.1-flash-lite",
    "models/gemini-3.1-flash-lite-preview",
    "models/gemini-flash-lite-latest",
    "models/gemma-4-31b-it",
    "models/gemma-4-26b-a4b-it",
]
# Ordered best -> weakest, based on your test_models.py run. Reasoning/
# thinking model is included (nemotron-3-nano-omni-30b-a3b-reasoning) —
# that's OpenRouter's free "thinking" option.
FREE_MODEL_CHAIN = [
    "nvidia/nemotron-3-ultra-550b-a55b:free",             # biggest, best general reasoning
    "nvidia/nemotron-3-super-120b-a12b:free",             # strong mid-tier
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", # dedicated reasoning/thinking model
    "inclusionai/ling-3.0-flash:free",
    "google/gemma-4-26b-a4b-it:free",
    "cohere/north-mini-code:free",                        # good for code-specific prompts
    "poolside/laguna-s-2.1:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "poolside/laguna-xs-2.1:free",
    "nvidia/nemotron-nano-9b-v2:free",
    "openai/gpt-oss-20b:free",
]

def _openrouter_provider(model_name: str) -> dict:
    return {
        "name": f"openrouter/{model_name}",
        "enabled": bool(OPENROUTER_API_KEY),
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "headers": {"Content-Type": "application/json", "Authorization": f"Bearer {OPENROUTER_API_KEY}"},
        "model": model_name,
    }
# ---------------------------------------------------------------------------
# GEMINI PROVIDER BUILDERS (Translates list strings to native Google API shapes)
# ---------------------------------------------------------------------------
def _gemini_text_provider(model_name: str) -> dict:
    raw_model_id = model_name.replace("models/", "")
    return {
        "name": f"google/{raw_model_id}",
        "enabled": bool(GEMINI_API_KEY),
        "url": "https://generativelanguage.googleapis.com/v1beta/chat/completions",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + GEMINI_API_KEY
        },
        "model": raw_model_id,
    }

TEXT_PROVIDERS = [
    # ── Primary: Native Google Gemini endpoints to handle large context & bypass 429s ──
    *[_gemini_text_provider(m) for m in GEMINI_MODELS_CHAIN],

    # ── Secondary: OpenRouter Free Chain ─────────────────────────────────
    *[_openrouter_provider(m) for m in FREE_MODEL_CHAIN],

    # ── Fallback: Groq ──────────────────────────────────────────────────
    {
        "name": "groq-qwen3.6-27b",
        "enabled": bool(GROQ_API_KEY),
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}"
        },
        "model": "qwen/qwen3.6-27b",
        "supports_reasoning_effort": True
    },
    {
        "name": "groq-gpt-oss-20b",
        "enabled": bool(GROQ_API_KEY),
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "headers": {"Content-Type": "application/json", "Authorization": f"Bearer {GROQ_API_KEY}"},
        "model": "openai/gpt-oss-20b",
    },

    # ── Last resort ───────────────────────────────────────────────────────
    {
        "name": "cerebras-llama",
        "enabled": bool(CEREBRAS_API_KEY),
        "url": "https://api.cerebras.ai/v1/chat/completions",
        "headers": {"Content-Type": "application/json", "Authorization": f"Bearer {CEREBRAS_API_KEY}"},
        "model": "gpt-oss-120b",
    },
]

VISION_PROVIDERS = [
    # ── Primary Vision: Native Gemini Pro / Flash Multimodal endpoints ──
    _gemini_text_provider("models/gemini-3.5-flash"),
    _gemini_text_provider("models/gemini-flash-latest"),

    # ── Fallback Vision: Groq & OpenRouter ──────────────────────────────
    {
        "name": "groq-vision",
        "enabled": bool(GROQ_API_KEY),
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "headers": {"Content-Type": "application/json", "Authorization": f"Bearer {GROQ_API_KEY}"},
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
    },
    {
        "name": "openrouter-vision-free",
        "enabled": bool(OPENROUTER_API_KEY),
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "headers": {"Content-Type": "application/json", "Authorization": f"Bearer {OPENROUTER_API_KEY}"},
        "model": "qwen/qwen2.5-vl-32b-instruct:free",
    },
]

# ---------------------------------------------------------------------------
# KNOWLEDGE BASE + IMAGE-DECISION PROMPT — content moved to prompts.py, see
# that file's header for the full map of where every prompt block now lives.
# ---------------------------------------------------------------------------
from prompts import ZINDRYX_INFO, MOJIZELA_INFO, IMAGE_GEN_AWARENESS

# ---------------------------------------------------------------------------
# REAL SERVER-SIDE DATE/TIME — computed fresh on every call. This is the
# fix for date/time questions going through DDGS: those "what's the date"
# aggregator sites are dynamically rendered per-visitor, so a scraped
# snapshot is often stale or wrong, and the model would hallucinate on top
# of bad snippets. The backend's own clock is ground truth and needs no
# search at all.
# ---------------------------------------------------------------------------
_WAT = timezone(timedelta(hours=1))  # West Africa Time — Nigeria, no DST


def _current_datetime_line() -> str:
    now_wat = datetime.now(_WAT)
    formatted = now_wat.strftime("%A, %B %d, %Y, %I:%M %p")
    return (
        f"REAL CURRENT DATE AND TIME: {formatted} WAT (West Africa Time, GMT+1 — Nigeria). "
        "This is the actual current date/time from the server's own clock — it is always "
        "correct. Never search the web or guess for what today's date or the current time is; "
        "just use this value directly. Only search the web for things that actually require "
        "it (news, prices, events, specific facts) — never for the date or time itself."
    )


# NEUTRAL_SYSTEM_PROMPT (identity/tone/formatting/code/math rules) moved to
# prompts.py — see that file's header for the full map.
from prompts import NEUTRAL_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Only appended to the system prompt when intent["complex"] == True (already
# false for greetings/small talk, so trivial messages never get this at
# all). This is sent to EVERY provider in the chain via the same `messages`
# payload — not just Qwen — but the old wording only said "when you think
# through this internally", which silently assumes the model already has a
# native thinking mechanism to expose. That's only true for Qwen 3.6 (via
# the reasoning_effort API param). A fallback provider like an OpenRouter
# free model or Cerebras has no native reasoning toggle, so that old
# phrasing did nothing for them — nothing ever told them a <think> block
# was expected, so if Qwen failed and the chain fell through, the fallback
# just answered flat with no reasoning shown at all.
#
# Fixed by explicitly asking for a literal <think>...</think> block by
# name — this works as a pure prompting instruction on ANY model, native
# reasoning support or not. Qwen's own native block (when reasoning_effort
# is on) just satisfies this instruction automatically; every other
# provider now has to actually produce one on request. _THINK_BLOCK_RE
# extracts it identically either way, so nothing downstream needed to
# change — same content-quality rules as before: no generic numbered
# filler, real problem-specific reasoning only.
# ---------------------------------------------------------------------------
# REASONING_STEP_ICONS / REASONING_STEP_HINT moved to prompts.py. Still
# imported into this module's namespace (not just used locally) because
# gpt2_functions.py imports REASONING_STEP_ICONS back from gpt2_test — see
# the circular-import note on that import below.
from prompts import REASONING_STEP_ICONS, REASONING_STEP_HINT


# ---------------------------------------------------------------------------
# Helper/tool functions live in gpt2_functions.py — imported here so every
# existing call inside ask_gpt2 / ask_gpt2_stream / _ask_gpt2_core below
# works completely unchanged. This import happens down here (not at the
# very top of the file) on purpose: gpt2_functions.py imports several
# constants (GROQ_API_KEY, BRAVE_API_KEY, TAVILY_API_KEY, TEXT_PROVIDERS,
# VISION_PROVIDERS) back from THIS module, so those constants must already
# be defined above before gpt2_functions is loaded, or Python's circular
# import resolution breaks.
# ---------------------------------------------------------------------------
from gpt2_functions import (
    search_web,
    build_search_query,
    classify_intent,
    get_lean_history,
    _split_thinking,
    _split_into_steps,
    _derive_step_label,
    _extract_step_icon,
    _call_provider_chain,
    _friendly_failure_message,
    ask_with_vision,
    _looks_unsure,
    build_file_with_continuation,
)

# ---------------------------------------------------------------------------
# TOOL LOOP — gpt2_tools.py. Lets the AI request one of the functions above
# for itself mid-answer by echoing a <<TOOL_REQUEST>> block, instead of us
# hardcoding which function fires when. See gpt2_tools.py header for the
# full flow. No circular import risk here — gpt2_tools.py only imports
# gpt2_functions.py, never this file.
# ---------------------------------------------------------------------------
from gpt2_tools import (
    build_tool_manifest,
    detect_tool_request,
    get_tool_source,
    parse_tool_call,
    execute_tool,
    strip_tool_markers,
    extract_suggestions,
    MAX_TOOL_ROUNDS,
)

# SUGGESTION_HINT (static) and TOOL_USE_HINT_TAIL (static wrapper text
# around the dynamic manifest) moved to prompts.py. The manifest itself is
# still generated at import time here since build_tool_manifest() is a
# real function call, not static text.
from prompts import SUGGESTION_HINT, TOOL_USE_HINT_TAIL, MEMORY_TRUNCATED_NOTE

TOOL_USE_HINT = "\n\n" + build_tool_manifest() + TOOL_USE_HINT_TAIL


# ---------------------------------------------------------------------------
# MAIN ASK FUNCTION
# ---------------------------------------------------------------------------
def ask_gpt2(
    prompt: str,
    history: Optional[list] = None,
    image_urls: Optional[list] = None,
    userid: Optional[str] = None,
) -> dict:
    """
    Non-streaming entry point — unchanged signature/behaviour for existing
    callers (main.py's /ai-query and /generate-question). Internally just
    drains _ask_gpt2_core() and keeps the final result.
    """
    final = None
    for event in _ask_gpt2_core(prompt, history=history, image_urls=image_urls, userid=userid):
        if event["type"] == "final":
            final = event
    return {
        "answer": final["answer"],
        "sources": final["sources"],
        "images": final.get("images", []),
        "provider": final["provider"],
        "file": final.get("file"),
    }


def ask_gpt2_stream(
    prompt: str,
    history: Optional[list] = None,
    image_urls: Optional[list] = None,
    userid: Optional[str] = None,
):
    """
    Streaming entry point for the /ai-query-stream SSE endpoint. Yields the
    exact same real progress events _ask_gpt2_core() produces — nothing
    synthetic. main.py wraps these as SSE frames.
    """
    yield from _ask_gpt2_core(prompt, history=history, image_urls=image_urls, userid=userid)


def _ask_gpt2_core(
    prompt: str,
    history: Optional[list] = None,
    image_urls: Optional[list] = None,
    userid: Optional[str] = None,
):
    """
    Shared generator. Yields:
      {"type": "status", "text": str}                                  -- real progress, as it happens
      {"type": "final", "answer": str, "sources": list, "provider": str|None}  -- exactly once, last
    """
    if history is None:
        history = []

    valid_image_urls = [
        url for url in (image_urls or [])
        if isinstance(url, str) and url.startswith(("http://", "https://"))
    ]

    image_results = []  # populated automatically whenever search_images succeeds
    file_result = None  # populated later only if the model calls build_file itself

    if valid_image_urls:
        yield {"type": "status", "text": "Looking at the image...", "detail": None, "icon": "vision"}
        vision_result = ask_with_vision(prompt, valid_image_urls, history)

        # Fold what vision saw into the prompt, then fall through to the
        # normal classify_intent() + search flow below — this is what lets
        # "browse for that" / "bring back pictures of this" actually
        # trigger a real web search instead of dead-ending on
        # "I can't browse online". No `return` here on purpose.
        image_description = vision_result.get("answer", "")
        prompt = f"{prompt}\n\n[Image analysis: {image_description}]"

    # ── Normal text flow ─────────────────────────────────────────────────
    yield {"type": "status", "text": "Reading your question...", "detail": None, "icon": "thinking"}
    intent = classify_intent(prompt, history=history)

    # FIXED (see header notes 6): the classifier only ever sees text, so it
    # has no way of knowing an image was already uploaded and analysed
    # above. The old fix forced a classifier field (wants_image) to False;
    # that field no longer exists (the model decides image search itself
    # now — see TOOL_USE_HINT), so the equivalent fix is telling the model
    # directly: don't bother reaching for search_images this turn, real
    # image(s) are already in hand.
    already_has_image_note = (
        "\n\nNOTE: The user already uploaded real image(s) with this message "
        "and they were already analysed above — do not call search_images "
        "this turn, you already have what you need."
        if valid_image_urls else ""
    )

    print(f"[INTENT] search_type={intent['search_type']} complex={intent['complex']} topic={intent['topic']} "
          f"query={intent.get('search_query')!r}")

    current_identity = (
        NEUTRAL_SYSTEM_PROMPT + "\n\n" + IMAGE_GEN_AWARENESS + already_has_image_note
        + "\n\n" + _current_datetime_line()
    )

    if intent["topic"] == "jamb":
        yield {
            "type": "status",
            "text": "Checking JAMB/UTME study notes...",
            "detail": "Pulling in the JAMB/UTME/WAEC exam-prep knowledge base for this reply.",
            "icon": "docs"
        }
        current_identity = (
            f"{NEUTRAL_SYSTEM_PROMPT}\n\n{IMAGE_GEN_AWARENESS}{already_has_image_note}\n\n{_current_datetime_line()}\n\n"
            f"CURRENT CONTEXT: {ZINDRYX_INFO}"
        )
    elif intent["topic"] == "mojizela":
        yield {
            "type": "status",
            "text": "Checking Mojizela app details...",
            "detail": "Pulling in the Mojizela app/coins/wallet knowledge base for this reply.",
            "icon": "docs"
        }
        current_identity = f"{NEUTRAL_SYSTEM_PROMPT}\n\n{IMAGE_GEN_AWARENESS}{already_has_image_note}\n\n{_current_datetime_line()}\n\nCURRENT CONTEXT: {MOJIZELA_INFO}"

    # ── Inject user docs or web search results if needed ──────────────────
    sources = []
    user_docs = []
    
    if intent["search_type"] == "user_docs":
        if userid:
            yield {
                "type": "status",
                "text": "Checking your saved files...",
                "detail": f'Searching for: "{intent["search_query"]}"',
                "icon": "docs"
            }
            try:
                manager = UserDocManager(userid)
                user_docs = manager.search_by_hint(intent["search_query"], limit=5)
                if user_docs:
                    yield {
                        "type": "status",
                        "text": f"Found {len(user_docs)} file(s) in your docs",
                        "detail": " • ".join([d.get("hint", d.get("filename", ""))[:30] for d in user_docs[:3]]),
                        "icon": "docs"
                    }
                    # Inject user docs into context
                    docs_context = "USER'S SAVED FILES (from their document storage):\n"
                    for doc in user_docs:
                        hint = doc.get("hint", doc.get("filename", ""))
                        tags = ", ".join(doc.get("tags", []))
                        docs_context += f"- {hint} (tags: {tags})\n"
                    current_identity += f"\n\n{docs_context}"
                else:
                    yield {
                        "type": "status",
                        "text": "No matching files found",
                        "detail": "Falling back to knowledge base answer",
                        "icon": "warning"
                    }
            except Exception as e:
                print(f"[USER_DOCS] search failed for {userid}: {e}")
                yield {
                    "type": "status",
                    "text": "Couldn't access your files",
                    "detail": str(e)[:50],
                    "icon": "warning"
                }
        else:
            yield {
                "type": "status",
                "text": "No user ID provided, skipping doc search",
                "detail": None,
                "icon": "warning"
            }
    
    elif intent["search_type"] == "web":
        clean_query = intent["search_query"] or build_search_query(prompt)
        print(f"[SEARCH] query={clean_query!r}")
        yield {
            "type": "status",
            "text": "Searching the web...",
            "detail": f'Searching for: "{clean_query}"',
            "icon": "search"
        }
        web_results, sources = search_web(clean_query)
        if web_results:
            titles = [s.get("title", "").strip() for s in sources if s.get("title")]
            yield {
                "type": "status",
                "text": f"Found {len(sources)} source(s)",
                "detail": " • ".join(titles[:5]) if titles else None,
                "icon": "search"
            }
            current_identity += (
                f"\n\n[BACKEND NOTE — not from the user]: The system distilled the "
                f"user's message into the search query \"{clean_query}\" and fetched "
                f"the results below on their behalf. This is reference material, not "
                f"something the user typed — answer their actual question naturally "
                f"using it, don't treat this block as their message or refer to the "
                f"distilled query itself. Always include relevant links when available:\n\n"
                + web_results
            )
        # if web_results is empty (every search engine failed), we simply
        # don't mention search at all — the model answers from its own
        # knowledge instead of relaying a search-failed error to the user.

    # ── Build messages ──────────────────────────────────────────────────
    lean_history, history_truncated = get_lean_history(history)
    if history_truncated:
        # `history` here now comes from the backend's own DB (memory_service
        # .build_memory), not a raw array replayed by the frontend — the
        # full conversation is safely persisted server-side either way.
        # get_lean_history() still trims what's SENT to the model to a lean
        # window for speed/cost, so we tell the model that honestly instead
        # of letting it guess about earlier turns it can't see right now.
        current_identity += MEMORY_TRUNCATED_NOTE
    # ORDER MATTERS HERE — TOOL_USE_HINT goes first, REASONING_STEP_HINT
    # goes last. Previously it was the other way around: TOOL_USE_HINT
    # (long, dense, always-on) was the very last thing the model saw
    # before generating, right after the reasoning requirement. That's
    # exactly the setup where recency bias can let a long, unconditional
    # instruction quietly crowd out a shorter, complexity-gated one — the
    # model would spend real time internally reasoning (Qwen's
    # reasoning_effort genuinely costs latency) but never actually write
    # the <think> block out, as if the tool-use rules were the last thing
    # it "remembered" needing to follow. Putting REASONING_STEP_HINT last
    # instead means it's the most recent instruction in context right
    # before the model starts writing its response.
    current_identity += TOOL_USE_HINT
    current_identity += SUGGESTION_HINT
    if intent["complex"]:
        current_identity += REASONING_STEP_HINT

    messages = [{"role": "system", "content": current_identity}]
    messages.extend(lean_history)
    messages.append({"role": "user", "content": prompt.strip()})

    # FIXED (see header notes 5): this used to always include a fabricated
    # "detail" line about temperature/token budgets, regardless of what
    # actually happened — it was cosmetic text, not a real report of
    # anything the model did. The genuine reasoning (the model's own
    # <think> block, when reasoning_effort is on) is extracted below and
    # emitted as its own separate, real status events — so this one just
    # announces the stage honestly, with nothing invented.
    yield {
        "type": "status",
        "text": "Thinking it through..." if intent["complex"] else "Writing answer...",
        "detail": None,
        "icon": "thinking"
    }

    answer, provider = _call_provider_chain(
        TEXT_PROVIDERS,
        messages,
        temperature=0.2 if intent["complex"] else 0.6,
        max_tokens=MAX_ANSWER_TOKENS,
        reasoning_effort="default" if intent["complex"] else "none",
    )

    if answer is None:
        yield {"type": "final", "answer": _friendly_failure_message(), "sources": [], "images": image_results, "provider": None, "file": file_result, "suggestions": []}
        return

    # NEW — pull any <think>...</think> block Qwen returned inline out of
    # the answer, and emit it as one "status" event per numbered step
    # (full, uncut text in each step's detail) instead of letting it leak
    # into the answer bubble as one giant blob.
    answer, model_thinking = _split_thinking(answer)

    # COMPLIANCE RETRY — a real, code-level safety net, not another prompt
    # rewrite. Only the primary provider (Qwen 3.6) has native reasoning
    # support (supports_reasoning_effort=True); the 3 fallback providers
    # rely purely on the soft <think> instruction in the prompt with
    # nothing backing it up model-side. If the chain fell through to one
    # of those and it just skipped <think> entirely — dumping its
    # reasoning straight into the visible answer instead — retry ONCE with
    # an explicit callout of what it just did wrong. This mirrors the same
    # recovery pattern already used for invalid tool calls elsewhere in
    # this function, rather than silently accepting a blank Thought
    # Process every time a fallback model answers a complex request.
    if intent["complex"] and not model_thinking:
        messages.append({"role": "assistant", "content": answer})
        messages.append({
            "role": "user",
            "content": (
                "You just answered without ever using a <think></think> "
                "block — the reasoning requirement from earlier in this "
                "conversation still applies. Rewrite your answer now: put "
                "your actual step-by-step reasoning inside <think></think> "
                "tags first (using the [icon] **Label:** format), THEN "
                "write your real final answer after the closing tag. "
                "Don't skip this."
            ),
        })
        compliance_answer, compliance_provider = _call_provider_chain(
            TEXT_PROVIDERS, messages, temperature=0.3, max_tokens=MAX_ANSWER_TOKENS, reasoning_effort="default",
        )
        if compliance_answer is not None:
            compliance_answer, compliance_thinking = _split_thinking(compliance_answer)
            if compliance_thinking:
                # Compliance recovered — use the retried, properly-tagged
                # version instead of the original non-compliant one.
                answer, model_thinking, provider = compliance_answer, compliance_thinking, compliance_provider
        # If the retry ALSO comes back without a <think> block, we just
        # move on with whatever answer is on hand — this particular model
        # genuinely won't comply, and an unbounded retry loop isn't the
        # right trade for one extra round trip.

    if model_thinking:
        for i, step in enumerate(_split_into_steps(model_thinking), start=1):
            step_icon, step_clean = _extract_step_icon(step)
            yield {
                "type": "status",
                "text": _derive_step_label(step_clean, i),
                "detail": step_clean,
                "icon": step_icon
            }

    # ── TOOL LOOP — the AI decided it wants to call one of TOOL_REGISTRY's
    # real functions for itself. Detected from whatever it just echoed
    # (checked against both the visible answer and the extracted thinking,
    # since a model may drop the request inside its <think> block). Each
    # round is a real, visible round trip — nothing here is faked or
    # pre-scripted; the "detail" on every status event is the AI's own
    # echoed text or the tool's real result, verbatim.
    session_context = {
        "prompt": prompt,
        "history": history,
        "userid": userid,
        "image_urls": valid_image_urls,
    }
    tool_round = 0
    search_text = (model_thinking or "") + "\n" + (answer or "")

    # FIXED — verify_image_relevance (and build_file, analyze_image) take a
    # session-injected param (image_results / history / image_urls) that's
    # supposed to be auto-filled, never typed by the model. That guidance
    # only ever got shown during the "show real source" round trip below.
    # When a model skipped straight to a one-shot <<TOOL_CALL>> instead —
    # which several free models do — it never saw that guidance, guessed a
    # plausible-but-wrong param name (e.g. "candidates" instead of
    # "image_results"), and tried to hand-paste the entire previous
    # search_images result set as literal JSON. That routinely got cut off
    # mid-array by the token cap, produced invalid JSON, and leaked the
    # raw half-finished <<TOOL_CALL>> straight into the visible answer.
    # Forcing these specific tools through the guided round trip no matter
    # which way the model requested them closes that off entirely.
    NEEDS_SOURCE_ROUNDTRIP = {"build_file", "analyze_image"}

    while tool_round < MAX_TOOL_ROUNDS:
        # Some models don't reliably follow the intended 2-step protocol
        # (REQUEST tool name -> we show real source -> model sends CALL with
        # real args) and jump straight to a full <<TOOL_CALL>> on the first
        # try. Previously only the REQUEST marker was ever detected, so a
        # model doing this got silently ignored and its raw <<TOOL_CALL>>
        # text leaked straight into the visible answer. Now: check for a
        # ready-to-run call FIRST (it's more specific/complete) and skip
        # straight to execution if found; only fall back to the
        # source-code round trip when just a bare tool name was requested
        # OR when the tool is one of NEEDS_SOURCE_ROUNDTRIP above.
        direct_call = parse_tool_call(search_text)
        requested_tool = direct_call["tool"] if direct_call else detect_tool_request(search_text)
        if not requested_tool:
            break
        # Discard a guessed one-shot call for these tools — force the
        # guided round trip instead of trusting hand-typed session args.
        if direct_call and requested_tool in NEEDS_SOURCE_ROUNDTRIP:
            direct_call = None
        tool_round += 1

        yield {
            "type": "status",
            "text": f"Reaching for {requested_tool}...",
            "detail": search_text.strip(),
            "icon": "tool",
        }

        if direct_call:
            # Model already supplied real args in one shot — no need to
            # show it the source and round-trip for a second reply.
            call_data = direct_call
            call_answer = answer
        else:
            tool_source = get_tool_source(requested_tool)
            # Tell the model which params it does NOT need to fill in itself
            # — this is the actual fix for verify_image_relevance calls
            # failing: the model was seeing image_results: list in the real
            # source and dutifully trying to retype the entire previous
            # search_images result set as literal JSON, which routinely got
            # cut off mid-array by the token cap and produced invalid JSON.
            auto_supplied_note = ""
            if requested_tool in ("build_file", "analyze_image"):
                auto_supplied_note = (
                    "\n\nNOTE: some of this function's parameters are session "
                    "data you do NOT need to supply — they're filled in "
                    "automatically: `prompt`, `history`, `userid`, "
                    "`image_urls`, and `image_results` (this last one comes "
                    "straight from whatever search_images you already ran "
                    "this turn). Do NOT try to retype those yourself — that's "
                    "what caused failures before. Just fill in the other real "
                    "arguments (e.g. query, filename, max_verified)."
                )
            messages.append({"role": "assistant", "content": answer})
            messages.append({
                "role": "user",
                "content": (
                    f"Here is the real source code for `{requested_tool}`:\n\n"
                    f"```python\n{tool_source}\n```\n\n"
                    f"Now call it for real by replying with ONLY this block, "
                    f"filled in with the actual arguments it needs:\n"
                    f'<<TOOL_CALL>>{{"tool": "{requested_tool}", "args": {{...}}}}<<END_TOOL_CALL>>'
                    + auto_supplied_note
                ),
            })

            yield {
                "type": "status",
                "text": f"Reviewing {requested_tool}'s code...",
                "detail": f"```python\n{tool_source}\n```{auto_supplied_note}",
                "icon": "tool",
            }
            call_answer, _ = _call_provider_chain(
                TEXT_PROVIDERS, messages, temperature=0.0, max_tokens=MAX_ANSWER_TOKENS,
            )
            call_data = parse_tool_call(call_answer)
            if not call_data:
                # RECOVERY, not a dead end — this used to just `break`,
                # which left whatever stale pre-tool-loop `answer` was
                # sitting around (usually just the bare tool marker) as the
                # final response once markers got stripped from it — that's
                # exactly what produced "no response" before. Now: tell the
                # model its call didn't parse, let it either answer without
                # the tool or try again, and get a REAL follow-up answer
                # before this loop iteration ends.
                yield {
                    "type": "status",
                    "text": f"Couldn't get a valid call for {requested_tool} — moving on",
                    "detail": call_answer,
                    "icon": "warning",
                }
                messages.append({"role": "assistant", "content": call_answer})
                messages.append({
                    "role": "user",
                    "content": (
                        f"That call for {requested_tool} wasn't valid — it may "
                        f"have been cut off or malformed. Skip that tool call "
                        f"and just answer the user now with what you already "
                        f"know, or request a different tool if genuinely needed."
                    ),
                })
                yield {
                    "type": "status",
                    "text": "Continuing with the answer...",
                    "detail": None,
                    "icon": "thinking",
                }
                answer, provider = _call_provider_chain(
                    TEXT_PROVIDERS, messages, temperature=0.3, max_tokens=MAX_ANSWER_TOKENS,
                )
                if answer is None:
                    yield {"type": "final", "answer": _friendly_failure_message(), "sources": sources, "images": image_results, "provider": None, "file": file_result, "suggestions": []}
                    return
                answer, model_thinking = _split_thinking(answer)
                if model_thinking:
                    for i, step in enumerate(_split_into_steps(model_thinking), start=1):
                        step_icon, step_clean = _extract_step_icon(step)
                        yield {
                            "type": "status",
                            "text": _derive_step_label(step_clean, i),
                            "detail": step_clean,
                            "icon": step_icon,
                        }
                search_text = (model_thinking or "") + "\n" + (answer or "")
                continue

        yield {
            "type": "status",
            "text": f"Calling {call_data['tool']}...",
            "detail": json.dumps(call_data["args"]),
            "icon": "tool",
        }

        # SPECIAL-CASED DISPATCH — two tools need more than execute_tool's
        # generic "drain silently, stringify the result" handling:
        #
        #   build_file: it's a generator that yields real, user-visible
        #   progress ("Building output.txt...", "Reviewing for accuracy...")
        #   — draining it silently (like execute_tool does for any other
        #   generator tool) would hide that from the sheet. Called directly
        #   here instead so those events stream live, and its real
        #   file_result dict is kept (not just stringified) so the frontend
        #   still gets a proper downloadable file card.
        #
        #   verify_image_relevance: its return value needs to end up as the
        #   real `image_results` list (structured data for the gallery),
        #   not just text fed back to the model.
        if call_data["tool"] == "build_file":
            file_args = {
                "prompt": session_context["prompt"],
                "filename": call_data["args"].get("filename") or "output.txt",
                "userid": session_context["userid"],
                "history": session_context["history"],
            }
            file_event = None
            for event in build_file_with_continuation(**file_args):
                if event.get("type") == "file_result":
                    file_event = event
                else:
                    yield event
            success = bool(file_event and file_event.get("success"))
            tool_result = json.dumps(file_event, default=str) if file_event else "Tool produced no output."
            if success:
                file_result = file_event  # carried through to the final yield below
        else:
            success, tool_result = execute_tool(call_data["tool"], call_data["args"], session_context)
            if success and call_data["tool"] == "search_images":
                # search_images results ARE the gallery data now — no
                # separate verify_image_relevance parsing step exists
                # anymore, so this has to populate the real `image_results`
                # (what actually reaches the frontend as "images" below),
                # not just session_context for a follow-up call that will
                # never happen.
                try:
                    parsed = json.loads(tool_result)
                    if isinstance(parsed, list):
                        image_results = parsed
                        session_context["image_results"] = parsed
                except Exception:
                    pass

        yield {
            "type": "status",
            "text": f"{requested_tool} {'succeeded' if success else 'failed'}",
            "detail": tool_result[:500],
            "icon": "success" if success else "warning",
        }

        messages.append({"role": "assistant", "content": call_answer})

        # AGENTIC SEARCH RETRY — only added when the tool that just ran was
        # search_web. Deliberately no hardcoded "if fewer than N sources"
        # check here — we don't compute a quality score ourselves and force
        # a retry off it, because that's just a different hardcoded rule.
        # Instead the AI is handed the real results and asked to judge them
        # itself: thin, off-topic, or stale results should prompt IT to
        # rewrite the query and request search_web again through the exact
        # same <<TOOL_REQUEST>> mechanism — this loop already supports that
        # (up to MAX_TOOL_ROUNDS), it just wasn't being told to use it for
        # self-correction before now.
        retry_guidance = ""
        if call_data["tool"] == "search_web":
            retry_guidance = (
                "\n\nJudge these results yourself before using them: are they "
                "specific, current, and actually relevant to what the user "
                "asked — not just loosely related? If they're thin, off-topic, "
                "or clearly missing what's needed, don't settle for a weak "
                "answer. Instead, rewrite the search query — tighter, more "
                "specific, or worded differently — and request search_web "
                "again the exact same way. Only move on to your final answer "
                "once the results genuinely support it, or you've run out of "
                "reasonable ways to rephrase the query."
            )

        # FIXED — a failed call to one of the session-injected-param tools
        # used to just get told "you may request the tool again the same
        # way", relying on the model correctly re-triggering the full
        # source-showing round trip on its own. In practice it often just
        # retried blind with another guessed arg name, failed again, and
        # burned all MAX_TOOL_ROUNDS the same way. Now: on failure, hand it
        # the real source + auto-supplied-param note directly in this same
        # message, so every retry is grounded in the real signature, not
        # a second guess.
        if not success and call_data["tool"] in NEEDS_SOURCE_ROUNDTRIP:
            retry_tool_source = get_tool_source(call_data["tool"])
            retry_note = (
                "\n\nHere is the real source code again — call it with ONLY "
                f"this block:\n```python\n{retry_tool_source}\n```\n\n"
                'Reply with ONLY: <<TOOL_CALL>>{"tool": "'
                f'{call_data["tool"]}", "args": {{...}}'
                '}<<END_TOOL_CALL>>\n\n'
                "Some of these parameters are session data you do NOT need "
                "to supply — they're filled in automatically: `prompt`, "
                "`history`, `userid`, `image_urls`, and `image_results`. "
                "Do NOT retype those yourself, that's what caused this "
                "failure. Only fill in the other real arguments."
            )
            retry_guidance = retry_guidance + retry_note

        messages.append({
            "role": "user",
            "content": (
                f"Tool result from {call_data['tool']}:\n{tool_result}\n\n"
                f"Continue and give the user your actual answer now, using "
                f"this if it helps. If you need another tool, you may "
                f"request one the same way; otherwise just answer normally."
                + retry_guidance
            ),
        })

        yield {
            "type": "status",
            "text": "Continuing with the answer...",
            "detail": None,
            "icon": "thinking",
        }
        answer, provider = _call_provider_chain(
            TEXT_PROVIDERS, messages, temperature=0.3, max_tokens=MAX_ANSWER_TOKENS,
        )
        if answer is None:
            yield {"type": "final", "answer": _friendly_failure_message(), "sources": sources, "images": image_results, "provider": None, "file": file_result, "suggestions": []}
            return

        answer, model_thinking = _split_thinking(answer)
        if model_thinking:
            for i, step in enumerate(_split_into_steps(model_thinking), start=1):
                step_icon, step_clean = _extract_step_icon(step)
                yield {
                    "type": "status",
                    "text": _derive_step_label(step_clean, i),
                    "detail": step_clean,
                    "icon": step_icon,
                }
        search_text = (model_thinking or "") + "\n" + (answer or "")

    # ── Safety net: classifier said "no search needed", but the model
    # itself came back unsure. Rather than let a guess through, run one
    # search now and re-ask with real web context. Sources always get
    # attached when this fires. (Skip if user_docs search was already done.)
    if intent["search_type"] == "none" and not sources and _looks_unsure(answer):
        clean_query = build_search_query(prompt)
        yield {
            "type": "status",
            "text": "Not fully sure — double-checking online...",
            "detail": f'The first draft wasn\'t confident, so searching for: "{clean_query}"',
            "icon": "search"
        }
        web_results, fallback_sources = search_web(clean_query)
        if web_results:
            titles = [s.get("title", "").strip() for s in fallback_sources if s.get("title")]
            yield {
                "type": "status",
                "text": f"Found {len(fallback_sources)} source(s)",
                "detail": " • ".join(titles[:5]) if titles else None,
                "icon": "search"
            }
            retry_identity = current_identity + (
                f"\n\n[BACKEND NOTE — not from the user]: The system distilled the "
                f"user's message into the search query \"{clean_query}\" and fetched "
                f"the results below on their behalf. This is reference material, not "
                f"something the user typed — answer their actual question naturally "
                f"using it, don't treat this block as their message or refer to the "
                f"distilled query itself. Always include relevant links when available:\n\n"
                + web_results
            )
            retry_messages = [{"role": "system", "content": retry_identity}]
            retry_messages.extend(get_lean_history(history)[0])
            retry_messages.append({"role": "user", "content": prompt.strip()})

            yield {
                "type": "status",
                "text": "Rewriting answer with sources...",
                "detail": "Rewriting the answer now with real search results available.",
                "icon": "search"
            }
            retry_answer, retry_provider = _call_provider_chain(
                TEXT_PROVIDERS, retry_messages, temperature=0.5, max_tokens=MAX_ANSWER_TOKENS, reasoning_effort="none"
            )
            if retry_answer:
                # NEW — same safety strip, in case a provider ever returns
                # an inline <think> block here too.
                retry_answer, retry_thinking = _split_thinking(retry_answer)
                if retry_thinking:
                    for i, step in enumerate(_split_into_steps(retry_thinking), start=1):
                        step_icon, step_clean = _extract_step_icon(step)
                        yield {
                            "type": "status",
                            "text": _derive_step_label(step_clean, i),
                            "detail": step_clean,
                            "icon": step_icon
                        }
                retry_answer_clean, retry_suggestions = extract_suggestions(strip_tool_markers(retry_answer))
                yield {"type": "final", "answer": retry_answer_clean, "sources": fallback_sources, "images": image_results, "provider": retry_provider, "file": file_result, "suggestions": retry_suggestions}
                return

    final_answer_clean, final_suggestions = extract_suggestions(strip_tool_markers(answer))
    yield {"type": "final", "answer": final_answer_clean, "sources": sources, "images": image_results, "provider": provider, "file": file_result, "suggestions": final_suggestions}

