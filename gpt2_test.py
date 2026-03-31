"""
gpt2_test.py — AI Engine using Groq API
=========================================
Groq provides free, ultra-fast inference for Llama 3.1 70B.
It uses an OpenAI-compatible endpoint so the code is clean and simple.

Setup (Render Dashboard):
  1. Go to https://console.groq.com → sign up free → API Keys → Create Key
  2. In Render: Environment Variables → Add:
       Key:   GROQ_API_KEY
       Value: gsk_xxxxxxxxxxxxxxxxxxxx
  3. Redeploy.

Circuit Breaker Logic (for the AI persona):
  CLOSED  → Normal, all requests go through.
  OPEN    → 3 consecutive failures → reject for 10s cooldown.
  HALF-OPEN → After cooldown, allow one trial request.
"""

import os
import time
import requests
from typing import Optional

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
API_URL: str = "https://api.groq.com/openai/v1/chat/completions"

# Fast + free on Groq. Alternatives:
#   "llama-3.1-70b-versatile"  ← most capable, still free
#   "llama-3.1-8b-instant"     ← fastest, lightest
#   "mixtral-8x7b-32768"       ← great for long contexts
MODEL: str = "llama-3.3-70b-versatile"

MAX_RETRIES: int = 3
RETRY_BASE_DELAY: float = 2.0   # exponential backoff: 2s, 4s, 8s
REQUEST_TIMEOUT: int = 30        # Groq is fast — 30s is plenty

# ---------------------------------------------------------------------------
# VALIDATION — fail loud at startup, not mid-request
# ---------------------------------------------------------------------------
if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY environment variable is not set.\n"
        "1. Get a free key at https://console.groq.com\n"
        "2. Add it in Render: Environment Variables → GROQ_API_KEY"
    )

HEADERS: dict = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json",
}

# ---------------------------------------------------------------------------
# SYSTEM PROMPT / PERSONA
# ---------------------------------------------------------------------------
SYSTEM_PROMPT: str = """
You are a coding writing machine you also help in other area.
You are an expert in Clean Architecture Coding, and High-Performance systems across Python, Rust, Go, TypeScript, C++, and Dart.
1. GUIDELINES:
    - Be concise and professional.
    - If I ask a general question, answer as a senior mentor.
    - Only provide code snippets when they are relevant to my current question.
    - Always use Type Hints and follow PEP8.
    - Do not talk too much, always answer what you are been asked and if you do not understand, ask for details.
    - Do not over-Write more than you are asked to do.
    - Always listen to what a user is saying.
    - Pay Attention to user specifications.
    - You are to also make suggestions for answers to questions you are asking a user.
    - Also note that you may be asked to write full projects so keep you mind in track of discussions between you and users.
    - You also help users fix their code.
    - when a code is pasted to you, You can ask users question on waht they want if you dont understand yet and you are also allowed to add changes and make code better if user suggest such to you.
    - When a user ask you if they should paste code, Your answer should be "paste it" then you are to fix based on what they described to you.
    - Always write fully complete code of what you are aked to write without placeholders or omissions, regardless of length.
    - Output full code. Do not use comments like '// ... rest of code' or 'implement logic here'. Provide every single line.
    - When also replying a user you can add emoji for illustrations too.
    - You can also draw or sketch structural output so they understand also.
    - Remember you are not to talk too much or say unneccesary stuffs.

### TECHNICAL SPECIFICATIONS (STRICT COMPLIANCE REQUIRED):
2. ARCHITECTURE: Follow 'Clean Architecture' / 'Hexagonal' pattern.
3. STATE ENCAPSULATION: Use 'ContextVars' for thread-local safety.
4. DATA INTEGRITY:
   - Pydantic V2 'Transaction' model.
   - Fields: tx_id (UUID), amount (Decimal), sender_key (str), signature (str).
   - Validator: amount > 0, sender_key must start with "0x".
5. SECURITY: 'SignatureVerifier' protocol with SHA-256 checksum simulation.
6. CONCURRENCY:
   - asyncio.Semaphore limiting DB writes to 5 concurrent max.
   - PriorityQueue: transactions > $10,000 processed first.
7. FAULT TOLERANCE:
   - Circuit Breaker for MockDatabase (3 fails → open 10s → half-open).
   - Exponential Backoff decorator on process_tx.
8. MONITORING via MetricsCollector (DefaultDict):
   - Successful vs failed tx counts.
   - Average latency (ms) per tx type.
   - Current queue depth.
9. ASYNC: 1 Ingestor, 3 Workers (asyncio.gather), 1 Heartbeat (every 2s).

### OUTPUT QUALITY:
- PEP8, full Type Hints (mypy compatible).
- Developer Readme comment explaining Circuit Breaker.
- Assert-based Unit Tests for the Validator.
- Minimize O(n) ops in PriorityQueue.
""".strip()


# ---------------------------------------------------------------------------
# CORE FUNCTION
# ---------------------------------------------------------------------------
def ask_gpt2(prompt: str, history: Optional[list] = None) -> str:
    """
    Sends a prompt + conversation history to Groq's Llama 3.1 70B model.

    Args:
        prompt:  The user's current message.
        history: List of prior {"role": ..., "content": ...} dicts.

    Returns:
        The model's reply as a string, or a descriptive error message.
    """
    if history is None:
        history = []

    # Truncate at word boundary
    if len(prompt) > 5000:
        prompt = prompt[:5000].rsplit(" ", 1)[0] + "..."
        print("[WARN] Prompt truncated to 5000 characters.")

    messages: list = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": prompt.strip()})

    payload: dict = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 10000,
        "temperature": 0.5,
        "top_p": 0.9,
        "stream": False,
    }

    # --- Retry loop with exponential backoff ---
    for attempt in range(1, MAX_RETRIES + 3):
        try:
            response = requests.post(
                API_URL,
                headers=HEADERS,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code != 200:
                error_msg = f"[API Error {response.status_code}] {response.text[:300]}"
                print(error_msg)
                # Don't retry client errors — they won't fix themselves
                if response.status_code in (400, 401, 403, 422):
                    return error_msg
                raise requests.HTTPError(error_msg)

            result: dict = response.json()
            choices = result.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "").strip()
                if content:
                    return content

            return "[Error] Model returned an empty response."

        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            wait = RETRY_BASE_DELAY ** attempt
            print(f"[Attempt {attempt}/{MAX_RETRIES}] Transient error: {e}. "
                  f"Retrying in {wait:.1f}s...")
            time.sleep(wait)

        except Exception as e:
            return f"[Fatal Exception] {type(e).__name__}: {e}"

    return "[Error] All retry attempts exhausted. Check GROQ_API_KEY or network."
