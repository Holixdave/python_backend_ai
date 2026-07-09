#!/usr/bin/env python3
"""
ai_backend_flow.py — Main conversation flow, intent classification, thinking extraction.
Part 2 of the refactored backend.

Handles:
  - Intent classification (search needed? complex reasoning?)
  - History management (lean history, truncation detection)
  - Dynamic thinking step extraction (AI generates its own steps, not hardcoded)
  - Main answer generation pipeline (_ask_gpt2_core)
  
CRITICAL CHANGES FROM OLD VERSION:
  1. NO hardcoded thinking steps ("Thinking it through..." / "Writing answer...")
  2. AI generates its own <think> blocks; we extract and emit them dynamically
  3. History is passed to ALL model calls (intent classifier, vision, retries)
  4. No IMAGE_GEN_AWARENESS prompt (removed)
  5. Image results are ONLY from web search, never generated
  6. search_images() only called if wants_image=True AND actual search runs
"""

import os
import re
import json
import requests
from typing import Optional, Tuple, List, Dict
from ai_backend_core import (
    _call_provider_chain, 
    search_web, 
    search_images,
    ask_with_vision,
    TEXT_PROVIDERS,
    INTENT_CLASSIFIER_PROVIDERS,
    REQUEST_TIMEOUT,
)

# ─────────────────────────────────────────────────────────────────────────────
# HISTORY MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def get_lean_history(history: List[Dict], max_turns: int = 6) -> Tuple[List[Dict], bool]:
    """
    Keep recent turns only, drop old ones, track if truncated.
    Returns (lean_history, was_truncated).
    """
    if not history or len(history) <= max_turns:
        return history, False

    # Take the last max_turns
    lean = history[-max_turns:]
    return lean, True


# ─────────────────────────────────────────────────────────────────────────────
# INTENT CLASSIFICATION (WITH HISTORY)
# ─────────────────────────────────────────────────────────────────────────────

def classify_intent(prompt: str, history: List[Dict] = None) -> Dict:
    """
    Lightweight classifier to decide: search needed? complex reasoning?
    NOW PASSES HISTORY so the model understands context.
    
    Returns: {
        "search_type": "web" | "images" | "user_docs" | "none",
        "wants_image": bool,
        "complex": bool,
        "search_query": str or None,
    }
    """
    if history is None:
        history = []

    system_prompt = """You are a query classifier. Respond ONLY with valid JSON, nothing else.
Analyze the user's request and return:
{
  "search_type": "web" (for current info/facts/news), "images" (for pictures), "user_docs" (for personal data), or "none",
  "wants_image": true/false (does the user want images shown?),
  "complex": true/false (requires reasoning/multiple steps?),
  "search_query": "optimized search query string" or null
}"""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    result_text, _ = _call_provider_chain(
        INTENT_CLASSIFIER_PROVIDERS,
        messages,
        temperature=0.2,
        max_tokens=200,
    )

    if not result_text:
        # Fallback on classifier failure
        return {
            "search_type": "none",
            "wants_image": "image" in prompt.lower() or "picture" in prompt.lower(),
            "complex": len(prompt.split()) > 20,
            "search_query": None,
        }

    try:
        return json.loads(result_text)
    except json.JSONDecodeError:
        print(f"[CLASSIFIER] bad JSON: {result_text}")
        return {
            "search_type": "none",
            "wants_image": "image" in prompt.lower(),
            "complex": len(prompt.split()) > 20,
            "search_query": None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# THINKING EXTRACTION & STEP PARSING
# ─────────────────────────────────────────────────────────────────────────────

def _split_thinking(answer: str) -> Tuple[str, str]:
    """
    Extract <think>...</think> block from answer if present.
    Returns (answer_without_think, thinking_text).
    """
    match = re.search(r'<think>(.*?)</think>', answer, re.DOTALL)
    if not match:
        return answer, ""

    thinking = match.group(1).strip()
    answer_clean = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
    return answer_clean, thinking


def _split_into_steps(thinking: str) -> List[str]:
    """
    Split thinking into numbered steps.
    Handles both "1. Title: content" and "1. **Title**: content" formats.
    """
    # Match patterns like "1. Something:" or "1. **Something**:" 
    pattern = r'(\d+)\.\s*(?:\*\*)?(.*?)(?:\*\*)?\s*:(.+?)(?=\d+\.\s|$)'
    matches = re.findall(pattern, thinking, re.DOTALL)
    
    if not matches:
        # No numbered structure found, return whole thing as one step
        return [thinking] if thinking.strip() else []
    
    steps = []
    for num, title, content in matches:
        step_text = f"{num}. **{title.strip()}**: {content.strip()}"
        steps.append(step_text)
    
    return steps


def _derive_step_label(step: str, step_num: int) -> str:
    """
    Extract a short label from a step (first 50 chars of title).
    """
    # Try to extract the title part
    match = re.match(r'\d+\.\s*\*?\*?(.*?)\*?\*?\s*:', step)
    if match:
        title = match.group(1).strip()
        return title[:50]
    
    # Fallback
    return f"Step {step_num}"


def _looks_unsure(answer: str) -> bool:
    """
    Heuristic: does the answer contain uncertainty phrases?
    """
    unsure_phrases = [
        "i don't know", "i do not know", "i'm not sure", "i am not sure",
        "i don't have information", "i do not have information",
        "as of my last update", "as of my knowledge", "i cannot verify",
        "unable to", "cannot determine", "i cannot access",
        "without current data", "without access to", "not available",
    ]
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in unsure_phrases)


def _friendly_failure_message() -> str:
    return (
        "I encountered an error while processing your request. "
        "Please try again, or rephrase your question."
    )


def build_search_query(prompt: str) -> str:
    """
    Distill user prompt into a search query.
    Very simple for now — in production, a classifier call would be better.
    """
    # Remove common filler words
    stop_words = {
        "a", "an", "the", "is", "are", "was", "were", "what", "when", "where",
        "why", "how", "do", "does", "did", "can", "could", "will", "would",
        "should", "must", "might", "may", "and", "or", "but", "if", "then",
        "me", "you", "i", "we", "they", "he", "she", "it",
    }
    
    words = prompt.lower().split()
    query_words = [w for w in words if w not in stop_words and len(w) > 2]
    return " ".join(query_words[:8])  # Keep first 8 words


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPTS (NO IMAGE GENERATION, NO HARDCODED THINKING)
# ─────────────────────────────────────────────────────────────────────────────

BASE_IDENTITY = """You are OOOR, a helpful AI assistant.
- Be clear, concise, and accurate.
- If you're unsure, say so plainly rather than guessing.
- When you have real web search results, use them and cite sources.
- Format code properly with markdown (```lang fenced blocks).
- For SVG diagrams, use ```svg code blocks when appropriate.
- For HTML pages, use ```html code blocks."""

REASONING_STEP_HINT = (
    "\n\n[REASONING INSTRUCTIONS]: You have reasoning capability enabled. "
    "In your <think> block, break your thought process into clear numbered steps. "
    "Each step should be: NUMBER. **TITLE**: explanation. "
    "For example:\n"
    "1. **Analyzing Question**: The user is asking about...\n"
    "2. **Gathering Context**: Relevant history shows...\n"
    "3. **Formulating Answer**: Based on this, I should..."
)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FLOW: _ask_gpt2_core
# ─────────────────────────────────────────────────────────────────────────────

def _ask_gpt2_core(
    prompt: str,
    history: Optional[List[Dict]] = None,
    image_urls: Optional[List[str]] = None,
    userid: Optional[str] = None,
):
    """
    Main generator. Yields status events, then one final event.
    
    CRITICAL CHANGES:
    - AI generates its own thinking steps dynamically (extracted from <think> blocks)
    - History passed to intent classifier and all AI calls
    - No hardcoded "Thinking it through..." messages
    - Images only from web search, never generated
    """
    if history is None:
        history = []

    valid_image_urls = [
        url for url in (image_urls or [])
        if isinstance(url, str) and url.startswith(("http://", "https://"))
    ]

    image_results = []  # Only populated if web search runs AND finds images
    sources = []

    # ─────────────────────────────────────────────────────────────────────────
    # VISION: If images provided, analyze them
    # ─────────────────────────────────────────────────────────────────────────
    if valid_image_urls:
        yield {"type": "status", "text": "Looking at the image...", "detail": None}
        vision_result = ask_with_vision(prompt, valid_image_urls, history)
        image_description = vision_result.get("answer", "")
        # Fold vision into prompt, don't return — let normal flow continue
        prompt = f"{prompt}\n\n[Image analysis: {image_description}]"

    # ─────────────────────────────────────────────────────────────────────────
    # CLASSIFY INTENT (with history)
    # ─────────────────────────────────────────────────────────────────────────
    yield {"type": "status", "text": "Reading your question...", "detail": None}
    intent = classify_intent(prompt, history=history)

    current_identity = BASE_IDENTITY

    # ─────────────────────────────────────────────────────────────────────────
    # SEARCH: If classifier said to search
    # ─────────────────────────────────────────────────────────────────────────
    if intent["search_type"] == "web":
        search_query = intent.get("search_query") or build_search_query(prompt)
        yield {"type": "status", "text": "Searching the web...", "detail": f'Query: "{search_query}"'}
        
        web_results, sources = search_web(search_query)
        if web_results:
            yield {
                "type": "status",
                "text": f"Found {len(sources)} source(s)",
                "detail": " • ".join([s.get("title", "")[:40] for s in sources[:3]]),
            }
            current_identity += f"\n\n[Search Results]\nHere is the current web information:\n{web_results}"

        # ONLY search for images if user asked for them AND we're doing web search
        if intent.get("wants_image"):
            image_query = intent.get("search_query") or search_query
            yield {"type": "status", "text": "Looking for images...", "detail": f'Query: "{image_query}"'}
            image_results = search_images(image_query)
            if image_results:
                yield {
                    "type": "status",
                    "text": f"Found {len(image_results)} image(s)",
                    "detail": None,
                }
    elif intent["search_type"] == "none" and not image_results:
        # No search, no images
        pass

    # ─────────────────────────────────────────────────────────────────────────
    # BUILD MESSAGES (with history)
    # ─────────────────────────────────────────────────────────────────────────
    lean_history, history_truncated = get_lean_history(history)
    
    if history_truncated:
        current_identity += (
            "\n\n[NOTE: Only recent chat history is shown; earlier messages exist "
            "but aren't included. If asked about the start of the conversation, "
            "say you don't have access to that.]"
        )

    # Add reasoning hint only if complex reasoning will be used
    if intent["complex"]:
        current_identity += REASONING_STEP_HINT

    messages = [{"role": "system", "content": current_identity}]
    messages.extend(lean_history)
    messages.append({"role": "user", "content": prompt.strip()})

    # ─────────────────────────────────────────────────────────────────────────
    # CALL TEXT MODEL (AI generates its own thinking in <think> block)
    # ─────────────────────────────────────────────────────────────────────────
    answer, provider = _call_provider_chain(
        TEXT_PROVIDERS,
        messages,
        temperature=0.3 if intent["complex"] else 0.6,
        max_tokens=4096 if intent["complex"] else 2048,
        reasoning_effort="default" if intent["complex"] else "none",
    )

    if answer is None:
        yield {
            "type": "final",
            "answer": _friendly_failure_message(),
            "sources": [],
            "images": image_results,
            "provider": None,
        }
        return

    # ─────────────────────────────────────────────────────────────────────────
    # EXTRACT & EMIT THINKING STEPS (from model's <think> block)
    # ─────────────────────────────────────────────────────────────────────────
    answer, model_thinking = _split_thinking(answer)
    if model_thinking:
        steps = _split_into_steps(model_thinking)
        for i, step in enumerate(steps, start=1):
            label = _derive_step_label(step, i)
            yield {
                "type": "status",
                "text": label,
                "detail": step,
            }

    # ─────────────────────────────────────────────────────────────────────────
    # SAFETY NET: Model was unsure, re-ask with web search
    # ─────────────────────────────────────────────────────────────────────────
    if intent["search_type"] == "none" and not sources and _looks_unsure(answer):
        clean_query = build_search_query(prompt)
        yield {
            "type": "status",
            "text": "Not fully confident — searching online...",
            "detail": f'Searching for: "{clean_query}"',
        }
        
        web_results, fallback_sources = search_web(clean_query)
        if web_results and fallback_sources:
            yield {
                "type": "status",
                "text": f"Found {len(fallback_sources)} source(s)",
                "detail": " • ".join([s.get("title", "")[:40] for s in fallback_sources[:3]]),
            }

            retry_identity = current_identity + (
                f"\n\n[Backend Note: System performed web search for: \"{clean_query}\"]\n\n"
                f"{web_results}"
            )

            retry_messages = [{"role": "system", "content": retry_identity}]
            retry_messages.extend(lean_history)
            retry_messages.append({"role": "user", "content": prompt.strip()})

            retry_answer, retry_provider = _call_provider_chain(
                TEXT_PROVIDERS,
                retry_messages,
                temperature=0.5,
                max_tokens=2048,
                reasoning_effort="none",
            )

            if retry_answer:
                retry_answer, retry_thinking = _split_thinking(retry_answer)
                if retry_thinking:
                    steps = _split_into_steps(retry_thinking)
                    for i, step in enumerate(steps, start=1):
                        label = _derive_step_label(step, i)
                        yield {
                            "type": "status",
                            "text": label,
                            "detail": step,
                        }
                yield {
                    "type": "final",
                    "answer": retry_answer,
                    "sources": fallback_sources,
                    "images": image_results,
                    "provider": retry_provider,
                }
                return

    # ─────────────────────────────────────────────────────────────────────────
    # SUCCESS: Return final answer
    # ─────────────────────────────────────────────────────────────────────────
    yield {
        "type": "final",
        "answer": answer,
        "sources": sources,
        "images": image_results,
        "provider": provider,
    }
