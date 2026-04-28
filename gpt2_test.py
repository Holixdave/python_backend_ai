import os
import time
import requests
from typing import Optional

# ---------------------------------------------------------------------------
# CONFIG - SWAPPING GEMINI FOR GROQ
# ---------------------------------------------------------------------------
GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
API_URL: str = "https://api.groq.com/openai/v1/chat/completions"
MODEL: str = "llama3-70b-8192" 
MAX_RETRIES: int = 3
RETRY_BASE_DELAY: float = 1.0
REQUEST_TIMEOUT: int = 30

# ---------------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------------
if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY environment variable is not set.")

HEADERS: dict = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {GROQ_API_KEY}"
}

SYSTEM_PROMPT: str = (
    "You are UTME26 AI, a brilliant Nigerian study assistant. "
    "You help students prepare for JAMB UTME exams with expert knowledge. "
    "Be concise, accurate, and encouraging. Always follow Clean Architecture "
    "and PEP8 when writing code."
).strip()

# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def get_lean_history(history):
    lean = []
    for msg in history[-6:]:
        content = msg["content"]
        if len(content) > 1500:
            content = content[:750] + "... [Truncated] ..." + content[-750:]
        lean.append({"role": msg["role"], "content": content})
    return lean

def ask_gpt2(prompt: str, history: Optional[list] = None) -> str:
    if history is None:
        history = []
        
    messages: list = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(get_lean_history(history))
    messages.append({"role": "user", "content": prompt.strip()})

    payload: dict = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.6,
        "max_tokens": 2048,
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
            
            if response.status_code == 200:
                result = response.json()
                # FIX: Access the first choice in the list
                return result["choices"][0]["message"]["content"]
            
            if response.status_code == 429:
                time.sleep(RETRY_BASE_DELAY * attempt * 2)
                continue
                
            return f"[Groq Error {response.status_code}] {response.text[:200]}"
            
        except Exception as e:
            if attempt == MAX_RETRIES:
                return f"Connection Error: {str(e)}"
            time.sleep(RETRY_BASE_DELAY * attempt)
    
    return "Error: Request failed."
