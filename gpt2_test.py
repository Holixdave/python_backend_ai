import os
import time
import requests
from typing import Optional

# ---------------------------------------------------------------------------
# CONFIG - SWAPPING GEMINI FOR GROQ
# ---------------------------------------------------------------------------
GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
API_URL: str = "https://groq.com"
# Using Llama-3 70B for the most intelligent "Elite" responses
MODEL: str = "llama3-70b-8192" 
MAX_RETRIES: int = 3
RETRY_BASE_DELAY: float = 1.0
REQUEST_TIMEOUT: int = 30 # Groq is faster, we don't need 90s

# ---------------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------------
if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY environment variable is not set.\n"
        "Ensure you have added it to your Render Environment Variables."
    )

HEADERS: dict = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {GROQ_API_KEY}"
}

# ---------------------------------------------------------------------------
# SYSTEM PROMPT (UTME26 AI Personality)
# ---------------------------------------------------------------------------
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
    # Keep the last 6 messages to maintain context without hitting token limits
    lean = []
    for msg in history[-6:]:
        content = msg["content"]
        if len(content) > 1500:
            content = content[:750] + "... [Truncated] ..." + content[-750:]
        lean.append({"role": msg["role"], "content": content})
    return lean

def ask_gpt2(prompt: str, history: Optional[list] = None) -> str:
    """
    Main function to query the AI. Renamed to ask_gpt2 to maintain 
    compatibility with your main.py routes.
    """
    if history is None:
        history = []
        
    messages: list = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(get_lean_history(history))
    messages.append({"role": "user", "content": prompt.strip()})

    payload: dict = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.6, # Balanced between creative and factual
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
                return result["choices"][0]["message"]["content"]
            
            # Handle Rate Limits (Groq specific)
            if response.status_code == 429:
                time.sleep(RETRY_BASE_DELAY * attempt * 2)
                continue
                
            return f"[Groq Error {response.status_code}] {response.text[:200]}"
            
        except Exception as e:
            if attempt == MAX_RETRIES:
                return f"Connection Error: {str(e)}"
            time.sleep(RETRY_BASE_DELAY * attempt)

    
    return "Error: Request failed."
