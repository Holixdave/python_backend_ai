import os
import time
import requests
from typing import Optional

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
# Gemini OpenAI-compatible endpoint requires the key in the URL
API_URL: str = f"https://googleapis.com{GEMINI_API_KEY}"
MODEL: str = "gemini-1.5-flash"

MAX_RETRIES: int = 5
RETRY_BASE_DELAY: float = 2.0
REQUEST_TIMEOUT: int = 90

# ---------------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------------
if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY environment variable is not set.\n"
        "1. Get a key at https://google.com\n"
        "2. Add it in Render: Environment Variables → GEMINI_API_KEY"
    )

HEADERS: dict = {
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
def get_lean_history(history):
    lean = []
    for msg in history[-6:]:
        content = msg["content"]
        if len(content) > 2000:
            content = content[:1000] + "... [Old Code Truncated] ..."
        lean.append({"role": msg["role"], "content": content})
    return lean

def ask_gpt2(prompt: str, history: Optional[list] = None) -> str:
    if history is None:
        history = []

    clean_prompt = "".join(ch for ch in prompt if ord(ch) >= 32 or ch in "\n\r\t")

    messages: list = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(get_lean_history(history))
    messages.append({"role": "user", "content": clean_prompt.strip()})

    payload: dict = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.5,
        "top_p": 0.9,
        "stream": False,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                API_URL,
                headers=HEADERS,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code != 200:
                error_msg = f"[Gemini Error {response.status_code}] {response.text[:300]}"
                if response.status_code in (400, 401, 403, 422):
                    return error_msg
                raise requests.HTTPError(error_msg)

            result = response.json()
            choices = result.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return "Error: No response content from Gemini."

        except Exception as e:
            if attempt == MAX_RETRIES:
                return f"Error after {MAX_RETRIES} attempts: {str(e)}"
            time.sleep(RETRY_BASE_DELAY * attempt)
    
    return "Error: Request failed."
