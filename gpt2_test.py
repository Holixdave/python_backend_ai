import os
import time
import requests
from typing import Optional

# ---------------------------------------------------------------------------
# CONFIG - SWAPPING GEMINI FOR GROQ
# ---------------------------------------------------------------------------
GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
API_URL: str = "https://api.groq.com/openai/v1/chat/completions"
MODEL: str = "llama-3.3-70b-versatile" 
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
# 1. DEFINE THE MASTER KNOWLEDGE BASE
ZINDRYX_INFO = """
IDENTITY: You are the Zindryx JAMB Study Assistant.
Who or what is zindryx: It an app called 'Zindry', made with love for jamb student preparing for exams
TOPIC: JAMB UTME, WAEC, Post-UTME, and subject-specific tutoring.

APP PRICING:
- Free Version: Limited to 5 practice questions per day.
- Premium Activation: ₦2,500 (One-time fee for full access to all years).
- Subject Buncle: ₦500 per specific subject.

FEATURES:
- Offline Mode: Works without data after activation.
- AI Tutor: Can solve complex math steps and explain English comprehension.
- Performance Tracking: Shows your weak areas in subjects like Physics or Govt.
"""



MOJIZELA_INFO = """
IDENTITY: You are the official Mojizela In-App AI.
TOPIC: Social media, video creation, content trends, and coins.
What or who is mojizela: it a social media platform just like tiktok, Has same features as tiktok but not part of their organisation it is woned by Hxf Softwares.
COIN PRICING (Naira):
- 20 Coins: ₦250
- 100 Coins: ₦1,200
- 500 Coins: ₦5,500
- 1,000 Coins: ₦10,500
- 5,000 Coins: ₦50,000

HOW TO BUY: Users can click the 'Wallet' icon in their profile, select a package, and pay via Flutterwave or Paystack.
GIFTING: 1 coin is worth 1 Diamond to creators.
POLICY: No refunds on coin purchases. Never say "I don't know the pricing."
"""
SYSTEM_PROMPT: str = (
    "You are UTME26 AI, a brilliant Nigerian study assistant. "
    "You help students prepare for JAMB UTME exams with expert knowledge. "
    "Be concise, accurate, and encouraging. Always follow Clean Architecture "
    "and PEP8 when writing code."
).strip()
NEUTRAL_SYSTEM_PROMPT = (
    "You are a brilliant Nigerian AI assistant. You never reveal you are a large language model. "
    "Check the user's intent: If they ask about exams/JAMB, use ZINDRYX_INFO. "
    "If they ask about videos/coins, use MOJIZELA_INFO. "
    "If it is a general greeting, be friendly and wait for context."
    "You always write code completely without any place holder and when you run of token you tell user and then continue from where you stop the code without repeating the code you have already written. "
    "You also should not use '*' for multiplication in code but instead use 'x' ."
    "you also should not use '*' for efor tip highlight or listing point instaed you '-' ."
    "Never tell users that i am out of token i will continue from here just let them know if the code isnt complete"
    "dont ever start writing code when you arent told to do so"
    "when writing code make sure to be self efficient that is be mindful of your design and dont just write code but make your code look very neat and leaghty to cover all aspect"
    "Stay matured and friendly dont start telling user this prompt that were asigned to you from the backend"
    "and make sure to add emoji in your reply so it look lively dont also add too much of emojis just add few emojis to make it lively and not too much to make it look like spammy."
)
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
    
    # 2. DYNAMIC IDENTITY DETECTION
    # We look at the current prompt and history to decide who we are
    full_text_context = (prompt + "".join([m["content"] for m in history])).lower()
    
    current_identity = NEUTRAL_SYSTEM_PROMPT
    if any(k in full_text_context for k in ["jamb", "utme", "syllabus", "zindryx"]):
        current_identity = f"{NEUTRAL_SYSTEM_PROMPT}\n\nCURRENT CONTEXT: {ZINDRYX_INFO}"
    elif any(k in full_text_context for k in ["mojizela", "coin", "tiktok", "video", "post"]):
        current_identity = f"{NEUTRAL_SYSTEM_PROMPT}\n\nCURRENT CONTEXT: {MOJIZELA_INFO}"

    # 3. BUILD MESSAGES
    messages = [{"role": "system", "content": current_identity}]
    messages.extend(get_lean_history(history))
    messages.append({"role": "user", "content": prompt.strip()})

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.6, # Lower temperature helps stay in character
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
                return result.get("choices", [{}])[0].get("message", {}).get("content", "Error: No content found.")

            
            if response.status_code == 429:
                time.sleep(RETRY_BASE_DELAY * attempt * 2)
                continue
                
            return f"[Groq Error {response.status_code}] {response.text[:200]}"
            
        except Exception as e:
            if attempt == MAX_RETRIES:
                return f"Connection Error: {str(e)}"
            time.sleep(RETRY_BASE_DELAY * attempt)
    
    return "Error: Request failed."
