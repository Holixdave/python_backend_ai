import os
import time
import requests
from typing import Optional, List, Dict

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
MODEL: str = "models/chat-bison-001"  # Gemini's native chat model
MAX_RETRIES: int = 5
RETRY_BASE_DELAY: float = 2.0
REQUEST_TIMEOUT: int = 90

if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY environment variable is not set.\n"
        "1. Get a key at https://cloud.google.com/vertex-ai/docs/generative-ai\n"
        "2. Set it in your environment variables as GEMINI_API_KEY"
    )

HEADERS: dict = {
    "Authorization": f"Bearer {GEMINI_API_KEY}",
    "Content-Type": "application/json",
}

# ---------------------------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------------------------
SYSTEM_PROMPT: str = """
You are a coding writing machine. You are an expert in Clean Architecture.
Always use Type Hints and follow PEP8. Provide full code without placeholders.
""".strip()

# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def get_lean_history(history: List[Dict]) -> List[Dict]:
    """Keep only the last few messages, truncate long content."""
    lean = []
    for msg in history[-6:]:
        content = msg["content"]
        if len(content) > 2000:
            content = content[:1000] + "... [Old Code Truncated] ..."
        lean.append({"role": msg["role"], "content": content})
    return lean

# ---------------------------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------------------------
def ask_gpt2(prompt: str, history: Optional[List[Dict]] = None) -> str:
    if history is None:
        history = []

    clean_prompt = "".join(ch for ch in prompt if ord(ch) >= 32 or ch in "\n\r\t")

    messages = [{"author": "system", "content": SYSTEM_PROMPT}]
    messages.extend(get_lean_history(history))
    messages.append({"author": "user", "content": clean_prompt.strip()})

    payload = {
        "messages": messages,
        "temperature": 0.5,
        "top_p": 0.9,
    }

    # Gemini native endpoint
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateMessage?key={GEMINI_API_KEY}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(api_url, headers=HEADERS, json=payload, timeout=REQUEST_TIMEOUT)

            if response.status_code != 200:
                error_msg = f"[Gemini Error {response.status_code}] {response.text[:300]}"
                if response.status_code in (400, 401, 403, 422):
                    return error_msg
                raise requests.HTTPError(error_msg)

            result = response.json()
            # Gemini returns output in: result['candidates'][0]['content']
            candidates = result.get("candidates", [])
            if candidates:
                return candidates[0].get("content", "")
            return "Error: No response content from Gemini."

        except Exception as e:
            if attempt == MAX_RETRIES:
                return f"Error after {MAX_RETRIES} attempts: {str(e)}"
            time.sleep(RETRY_BASE_DELAY * attempt)

    return "Error: Request failed."