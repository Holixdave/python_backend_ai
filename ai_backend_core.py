#!/usr/bin/env python3
"""
ai_backend_core.py — Core provider chains, search engines, vision handling.
Part 1 of the refactored backend (split from monolithic gpt2_test.py for manageability).

Handles:
  - Provider chains (Groq, OpenRouter, Cerebras, Gemini fallbacks)
  - Web search (DDGS, Brave, Tavily)
  - Image search (DDGS images only — no generation)
  - Vision analysis (multi-image + fallback to single-image analysis)
"""

import os
import re
import time
import json
import requests
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, List, Dict
from user_doc_manager import UserDocManager

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — API keys. Only GROQ_API_KEY is required.
# ─────────────────────────────────────────────────────────────────────────────
GROQ_API_KEY:       Optional[str] = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")
CEREBRAS_API_KEY:   Optional[str] = os.getenv("CEREBRAS_API_KEY")
GEMINI_API_KEY:     Optional[str] = os.getenv("GEMINI_API_KEY")
BRAVE_API_KEY:      Optional[str] = os.getenv("BRAVE_API_KEY")
TAVILY_API_KEY:     Optional[str] = os.getenv("TAVILY_API_KEY")

MAX_RETRIES_PER_PROVIDER: int = 2
RETRY_BASE_DELAY:         float = 1.0
REQUEST_TIMEOUT:          int   = 45

if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY environment variable is not set.")

# ─────────────────────────────────────────────────────────────────────────────
# PROVIDER CHAINS
# ─────────────────────────────────────────────────────────────────────────────
TEXT_PROVIDERS = [
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
        "name": "openrouter-qwen",
        "enabled": bool(OPENROUTER_API_KEY),
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENROUTER_API_KEY}"
        },
        "model": "qwen/qwen-max",
        "supports_reasoning_effort": False
    },
    {
        "name": "cerebras",
        "enabled": bool(CEREBRAS_API_KEY),
        "url": "https://api.cerebras.ai/v1/chat/completions",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {CEREBRAS_API_KEY}"
        },
        "model": "llama-3.3-70b",
        "supports_reasoning_effort": False
    },
    {
        "name": "gemini",
        "enabled": bool(GEMINI_API_KEY),
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "headers": {"Content-Type": "application/json"},
        "model": "gemini-2.0-flash",
        "supports_reasoning_effort": False,
        "api_key_param": "key"
    },
]

VISION_PROVIDERS = [
    {
        "name": "groq-vision",
        "enabled": bool(GROQ_API_KEY),
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}"
        },
        "model": "llama-4-scout",
    },
    {
        "name": "openrouter-vision",
        "enabled": bool(OPENROUTER_API_KEY),
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENROUTER_API_KEY}"
        },
        "model": "gpt-4-vision",
    },
]

INTENT_CLASSIFIER_PROVIDERS = [
    {
        "name": "groq-intent",
        "enabled": bool(GROQ_API_KEY),
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}"
        },
        "model": "llama-3.1-8b-instant",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# SEARCH: WEB IMAGES
# ─────────────────────────────────────────────────────────────────────────────

def _search_ddgs_images(query: str, max_results: int = 4):
    """Search for images via DDGS. Returns list of image dicts."""
    from ddgs import DDGS
    with DDGS() as ddgs:
        results = list(ddgs.images(query, max_results=max_results))
    return [
        {
            "image": r.get("image", ""),
            "thumbnail": r.get("thumbnail", r.get("image", "")),
            "title": r.get("title", ""),
            "source": r.get("url", r.get("source", "")),
        }
        for r in results if r.get("image")
    ]


def search_images(query: str, max_results: int = 4) -> List[Dict]:
    """
    Search for images to accompany text results.
    Returns list of image dicts, or [] on failure (never raises).
    
    IMPORTANT: This ONLY returns real search results, never generated images.
    """
    try:
        return _search_ddgs_images(query, max_results)
    except Exception as e:
        print(f"[IMAGE SEARCH] failed: {e}")
        return []


def _search_ddgs(query: str, max_results: int = 4):
    from duckduckgo_search import DDGS as DDGSearch
    results = DDGSearch().text(query, max_results=max_results)
    return results


def _search_brave(query: str, max_results: int = 4):
    """Brave Search API"""
    if not BRAVE_API_KEY:
        raise ImportError("BRAVE_API_KEY not set")
    url = "https://api.search.brave.com/res/v1/web/search"
    params = {"q": query, "count": max_results}
    headers = {"X-Subscription-Token": BRAVE_API_KEY}
    resp = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return [
        {
            "title": r.get("title", ""),
            "href": r.get("url", ""),
            "body": r.get("description", ""),
        }
        for r in data.get("web", {}).get("results", [])
    ]


def _search_tavily(query: str, max_results: int = 4):
    """Tavily Search API"""
    if not TAVILY_API_KEY:
        raise ImportError("TAVILY_API_KEY not set")
    url = "https://api.tavily.com/search"
    payload = {"api_key": TAVILY_API_KEY, "query": query, "max_results": max_results}
    resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return [
        {
            "title": r.get("title", ""),
            "href": r.get("url", ""),
            "body": r.get("content", ""),
        }
        for r in data.get("results", [])
    ]


def search_web(query: str, max_results: int = 4) -> Tuple[str, List[Dict]]:
    """
    Search the web. Tries each engine in order.
    Returns (formatted_text, sources) where sources = [{"title": str, "url": str}, ...]
    On total failure, returns ("", []) silently.
    """
    print(f"[SEARCH TRIGGERED] Query: {query}")
    for engine_name, engine_fn in (
        ("ddgs", _search_ddgs),
        ("brave", _search_brave),
        ("tavily", _search_tavily),
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

    print("[SEARCH] all engines failed or unavailable")
    return "", []


# ─────────────────────────────────────────────────────────────────────────────
# PROVIDER CHAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def _call_provider_chain(
    providers: List[Dict],
    messages: List[Dict],
    temperature: float = 0.7,
    max_tokens: int = 2048,
    reasoning_effort: str = "none",
) -> Tuple[Optional[str], Optional[str]]:
    """
    Try each provider in order until one succeeds.
    Returns (answer, provider_name) or (None, None) on total failure.
    """
    for provider in providers:
        if not provider.get("enabled"):
            continue

        for attempt in range(MAX_RETRIES_PER_PROVIDER):
            try:
                url = provider["url"]
                headers = provider["headers"].copy()
                
                payload = {
                    "model": provider["model"],
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }

                if provider.get("supports_reasoning_effort") and reasoning_effort != "none":
                    payload["reasoning_effort"] = reasoning_effort

                if "api_key_param" in provider:
                    payload[provider["api_key_param"]] = provider["headers"]["Authorization"].split(" ")[-1]

                resp = requests.post(url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
                data = resp.json()

                # Check if model returned a refusal wrapped in a 200 response
                if data.get("choices"):
                    content = data["choices"][0].get("message", {}).get("content", "")
                    if content and "unable" not in content.lower():
                        return content, provider["name"]

            except requests.exceptions.Timeout:
                print(f"[{provider['name']}] timeout on attempt {attempt + 1}")
                if attempt < MAX_RETRIES_PER_PROVIDER - 1:
                    time.sleep(RETRY_BASE_DELAY * (2 ** attempt))
                continue
            except Exception as e:
                print(f"[{provider['name']}] failed: {e}")
                continue

    return None, None


def _vision_messages(prompt: str, image_urls: List[str], history: List[Dict]) -> List[Dict]:
    """Build messages for vision analysis."""
    messages = []
    for msg in history:
        messages.append(msg)
    
    content = [{"type": "text", "text": prompt}]
    for url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})
    
    messages.append({"role": "user", "content": content})
    return messages


def ask_with_vision(
    prompt: str, 
    image_urls: List[str], 
    history: List[Dict] = None
) -> Dict:
    """
    Analyze images. Handles multi-image batch fallback.
    Returns {"answer": str, "sources": list}.
    """
    if history is None:
        history = []
    
    image_urls = image_urls[:4]
    print(f"[VISION TRIGGERED] Images: {len(image_urls)}, Prompt: {prompt[:60]}")

    messages = _vision_messages(prompt, image_urls, history)

    for provider in VISION_PROVIDERS:
        if not provider.get("enabled"):
            continue

        try:
            resp = requests.post(
                provider["url"],
                json={
                    "model": provider["model"],
                    "messages": messages,
                    "temperature": 0.5,
                    "max_tokens": 1024,
                },
                headers=provider["headers"],
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            answer = data["choices"][0]["message"]["content"]

            # Check for refusal
            if answer and "unable" not in answer.lower():
                return {"answer": answer, "sources": []}

            print(f"[VISION] {provider['name']} refused multi-image batch — trying single fallback")

        except Exception as e:
            print(f"[VISION] {provider['name']} failed: {e}")

    # Fallback: analyze each image separately
    print("[VISION] falling back to single-image analysis")
    individual_answers = []
    for i, url in enumerate(image_urls, start=1):
        msg_single = _vision_messages(f"{prompt} (Image {i})", [url], history)
        ans, _ = _call_provider_chain(VISION_PROVIDERS, msg_single, temperature=0.5, max_tokens=512)
        if ans:
            individual_answers.append(ans)

    combined = "\n\n".join(individual_answers) if individual_answers else "Could not analyze images"
    return {"answer": combined, "sources": []}


def ask_gpt2(
    prompt: str,
    history: Optional[List[Dict]] = None,
    image_urls: Optional[List[str]] = None,
    userid: Optional[str] = None,
) -> Dict:
    """
    Non-streaming entry point. Collects all events and returns final dict.
    """
    if history is None:
        history = []
    
    final = None
    for event in ask_gpt2_stream(prompt, history=history, image_urls=image_urls, userid=userid):
        if event["type"] == "final":
            final = event

    return {
        "answer": final["answer"] if final else "Failed to generate response",
        "sources": final.get("sources", []) if final else [],
        "images": final.get("images", []) if final else [],
        "provider": final.get("provider") if final else None,
    }


def ask_gpt2_stream(
    prompt: str,
    history: Optional[List[Dict]] = None,
    image_urls: Optional[List[str]] = None,
    userid: Optional[str] = None,
):
    """
    Streaming entry point. Yields status events and one final event.
    """
    if history is None:
        history = []
    
    yield from _ask_gpt2_core(prompt, history=history, image_urls=image_urls, userid=userid)
