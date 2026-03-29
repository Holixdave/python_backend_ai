import requests
import re
import os

# Get your token from Render Environment Variables
API_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# The Featherless Router is perfect for Llama 3
API_URL = "https://router.huggingface.co/v1/chat/completions"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

# Keep this for your main.py health checks
_pipe = True 

def ask_gpt2(prompt: str) -> str:
    """
    Using Llama-3-8B: The closest free equivalent to GPT-4 performance.
    """
    payload = {
        # UPGRADE: Switching from Zephyr to Llama-3 for better accuracy
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "messages": [
            {
                "role": "system", 
                "content": "You are a helpful AI assistant. The current date is March 2026. "
                           "The current President of Nigeria is Bola Ahmed Tinubu (since May 2023). "
                           "Always provide up-to-date and accurate information."
            },
            {"role": "user", "content": prompt.strip()[:400]}
        ],
        "max_tokens": 200,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=15)
        
        if response.status_code != 200:
            return f"API Error {response.status_code}: {response.text}"

        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            # Correctly accessing the message content
            new_text = result["choices"][0]["message"]["content"]
            return _clean_response(new_text)
            
        return "AI is processing, please try again in a moment."

    except Exception as e:
        return f"System Error: {str(e)}"

def _clean_response(text: str) -> str:
    """
    Cleans the output to ensure professional delivery.
    """
    # Remove AI 'hallucinations' or roleplay artifacts
    for stop in ["Student:", "UTME26 AI:", "Question:", "\n\n"]:
        text = text.split(stop)[0].strip()

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text if text else "I'm not sure how to answer that. Can you rephrase?"
