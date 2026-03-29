"""
gpt2_test.py — AI Engine wrapper
Exports ask_gpt2 so main.py can do: from gpt2_test import ask_gpt2
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
RETRY_BASE_DELAY: float = 2.0
REQUEST_TIMEOUT: int = 60

if not API_TOKEN:
    raise EnvironmentError(
        "HUGGINGFACE_TOKEN is not set. "
        "Add it as an Environment Variable in your Render dashboard."
    )

HEADERS: dict = {"Authorization": f"Bearer {API_TOKEN}"}

# ---------------------------------------------------------------------------
# SYSTEM PROMPT
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
4. SECURITY LAYER: Implement a 'SignatureVerifier' protocol. Simulate SHA-256
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
STEP 3: Develop the 'CircuitBreaker' logic (Closed -> Open -> Half-Open).
STEP 4: Implement 'PriorityProcessor' logic for the 'PriorityQueue'.
STEP 5: Create a 'MockAPI' simulating 50 transactions (some malicious).
STEP 6: Write the 'Main' loop with a clean 'KeyboardInterrupt' shutdown handler.

### OUTPUT FORMAT & QUALITY:
- PEP8 compliant, Type Hints (mypy compatible).
- 'Developer Readme' comment explaining Circuit Breaker logic.
- Unit Tests using simple assert-based functions for the Validator.
- Minimize O(n) operations in PriorityQueue handling.
""".strip()


# ---------------------------------------------------------------------------
# MAIN FUNCTION — named ask_gpt2 so main.py import works unchanged
# ---------------------------------------------------------------------------
def ask_gpt2(prompt: str, history: Optional[list] = None) -> str:
    """
    Sends prompt + conversation history to the HuggingFace model.

    Args:
        prompt:  The user's current message.
        history: List of prior {"role": ..., "content": ...} dicts.

    Returns:
        The model's reply as a string, or a descriptive error message.
    """
    if history is None:
        history = []
        # Truncate at word boundary to avoid cutting mid-sentence
    if len(prompt) > 5000:
        prompt = prompt[:5000].rsplit(" ", 1)[0] + "..."
        print("[WARN] Prompt was truncated to 5000 characters.")

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

    # Retry loop with exponential backoff
    for attempt in range(1, MAX_RETRIES + 1):
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
                # No point retrying auth/bad-request errors
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
            print(f"[Attempt {attempt}/{MAX_RETRIES}] Error: {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)

        except Exception as e:
            return f"[Fatal Exception] {type(e).name}: {e}"

    return f"[Error] All {MAX_RETRIES} retry attempts exhausted."