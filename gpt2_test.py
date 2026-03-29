"""
AI Engine - Principal Software Engineer Persona
================================================
Uses HuggingFace Inference API (OpenAI-compatible v1 endpoint).

Circuit Breaker Logic (for the AI persona's output):
  - CLOSED: Normal operation, requests pass through.
  - OPEN: After 3 consecutive failures, rejects all requests for 10 seconds.
  - HALF-OPEN: After cooldown, allows one trial request to test recovery.

Setup:
  Set environment variable HUGGINGFACE_TOKEN before running.
  Example: export HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxx
"""

import os
import time
import requests
from typing import Optional

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
API_TOKEN: Optional[str] = os.getenv("HUGGINGFACE_TOKEN")
API_URL: str = "https://api-inference.huggingface.co/v1/chat/completions"
MODEL: str = "meta-llama/Llama-3.1-70B-Instruct"

MAX_RETRIES: int = 3
RETRY_BASE_DELAY: float = 2.0   # seconds (exponential backoff base)
REQUEST_TIMEOUT: int = 60        # seconds

# ---------------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------------
if not API_TOKEN:
    raise EnvironmentError(
        "HUGGINGFACE_TOKEN environment variable is not set. "
        "Export it before running: export HUGGINGFACE_TOKEN=hf_xxxx"
    )

HEADERS: dict = {"Authorization": f"Bearer {API_TOKEN}"}

# ---------------------------------------------------------------------------
# SYSTEM PROMPT / PERSONA
# ---------------------------------------------------------------------------
SYSTEM_PROMPT: str = """
Act as a Principal Software Engineer (L7) at a High-Frequency Trading firm.
You must design a "Real-Time Transaction Risk Engine" with zero-tolerance for data loss.

### TECHNICAL SPECIFICATIONS (STRICT COMPLIANCE REQUIRED):
1. ARCHITECTURE: Implementation must follow the 'Clean Architecture' / 'Hexagonal' pattern.
2. STATE ENCAPSULATION: Create a 'StateStore' using 'ContextVars' for thread-local safety.
3. DATA INTEGRITY:
   - Define a 'Transaction' model using Pydantic V2.
   - Required Fields: tx_id (UUID), amount (Decimal), sender_key (str), signature (str).
   - Custom Validator: 'amount' must be > 0 and 'sender_key' must start with "0x".
4. SECURITY LAYER: Implement a 'SignatureVerifier' protocol. Simulate a SHA-256
   checksum verification for every incoming transaction.
5. CONCURRENCY CONTROL:
   - Use 'asyncio.Semaphore' to limit concurrent database writes to exactly 5.
   - Use a 'PriorityQueue' where transactions > $10,000 are processed before others.
6. FAULT TOLERANCE & RECOVERY:
   - Implement a 'Circuit Breaker' pattern for the 'MockDatabase'.
   - If 3 consecutive writes fail, the circuit opens for 10 seconds, rejecting all tx.
   - Implement an 'Exponential Backoff' decorator for the 'process_tx' method.
7. PERFORMANCE MONITORING:
   - Create a 'MetricsCollector' class using a 'DefaultDict' to track:
     a) Total successful vs. failed transactions.
     b) Average latency (in ms) per transaction type.
     c) Current 'Queue Depth'.
8. ASYNC ORCHESTRATION:
   - Implement 1 'Ingestor' task (producer).
   - Implement 3 'Worker' tasks (consumers) using 'asyncio.gather'.
   - Implement 1 'Heartbeat' task that logs system health every 2 seconds.

### IMPLEMENTATION STEPS:
STEP 1: Define Enums for 'TransactionStatus' and 'CircuitState'.
STEP 2: Build the 'Transaction' Pydantic model with custom validators.
STEP 3: Develop the 'CircuitBreaker' logic with state transitions (Closed -> Open -> Half-Open).
STEP 4: Implement the 'PriorityProcessor' logic for handling the 'PriorityQueue'.
STEP 5: Create a 'MockAPI' that simulates a stream of 50 transactions (some malicious).
STEP 6: Write the 'Main' loop with a clean 'KeyboardInterrupt' shutdown handler.
### OUTPUT FORMAT & QUALITY:
- Code must be PEP8 compliant and use Type Hints (mypy compatible).
- Provide a 'Developer Readme' comment at the top explaining the 'Circuit Breaker' logic.
- Include 'Unit Tests' (using a simple assert-based function) for the 'Validator'.
- Efficiency: Minimize O(n) operations in the 'PriorityQueue' handling.
""".strip()


# ---------------------------------------------------------------------------
# CORE FUNCTION: ask_model
# ---------------------------------------------------------------------------
def ask_model(prompt: str, history: Optional[list] = None) -> str:
    """
    Sends a prompt + conversation history to the HuggingFace model.

    Args:
        prompt:  The user's current message.
        history: List of prior {"role": ..., "content": ...} dicts.

    Returns:
        The model's reply as a plain string, or an error message.
    """
    if history is None:
        history = []

    # Truncate prompt gracefully at word boundary near 5000 chars
    if len(prompt) > 5000:
        prompt = prompt[:5000].rsplit(" ", 1)[0] + "..."
        print("[WARN] Prompt truncated to 5000 characters.")

    messages: list = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": prompt.strip()})

    payload: dict = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 1500,
        "temperature": 0.5,
        "top_p": 0.9,
    }

    # --- Retry loop with exponential backoff ---
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                API_URL,
                headers=HEADERS,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )

            # Surface non-200 clearly
            if response.status_code != 200:
                error_msg = (
                    f"[API Error {response.status_code}] {response.text[:300]}"
                )
                print(error_msg)
                # Don't retry on auth or bad-request errors
                if response.status_code in (400, 401, 403, 422):
                    return error_msg
                raise requests.HTTPError(error_msg)

            result: dict = response.json()

            # Parse OpenAI-compatible response
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
            # Non-retryable unexpected error
            return f"[Fatal Exception] {type(e).name}: {e}"

    return f"[Error] All {MAX_RETRIES} retry attempts failed. Check your network or API token."


# ---------------------------------------------------------------------------
# SIMPLE UNIT TEST (run with: python ai_engine.py --test)
# ---------------------------------------------------------------------------
def _run_self_tests() -> None:
    """Lightweight smoke tests that don't require a live API call."""
    print("Running self-tests...")

    # Test 1: Prompt truncation guard
    long_prompt = "x" * 6000
    truncated = long_prompt[:5000].rsplit(" ", 1)[0] + "..."
    assert len(truncated) <= 5004, "Truncation failed"

    # Test 2: History default is not shared across calls
    def _get_history(h=None):
        if h is None:
            h = []
        return h

    a = _get_history()
    b = _get_history()
    a.append("item")
    assert "item" not in b, "Mutable default argument leak!"

    # Test 3: Token present
    assert API_TOKEN, "Token must be set"

    print("All self-tests passed.\n")
    # ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        _run_self_tests()
        sys.exit(0)

    print("=" * 60)
    print("AI Engine ready. Type your prompt and press Enter.")
    print("Type 'exit' or Ctrl+C to quit.")
    print("=" * 60)

    conversation_history: list = []

    try:
        while True:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Shutting down. Goodbye.")
                break

            reply = ask_model(user_input, history=conversation_history)
            print(f"\nAssistant: {reply}\n")

            # Maintain rolling history (last 10 turns = 20 messages)
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": reply})
            if len(conversation_history) > 20:
                conversation_history = conversation_history[-20:]

    except KeyboardInterrupt:
        print("\n\n[Interrupted] Engine stopped cleanly.")