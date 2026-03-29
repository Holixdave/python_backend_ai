import requests
import re
import os

# Get your token from Render Environment Variables
API_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# This is the "Router" URL from your image - it's much faster!
API_URL = "https://router.huggingface.co"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

# Keep this for your main.py health checks
_pipe = True 

def ask_gpt2(prompt: str) -> str:
    """
    Generates a response using the Zephyr model via Featherless AI.
    """
    # This matches the 'OpenAI' format required by the Featherless router
    payload = {
        "model": "HuggingFaceH4/zephyr-7b-beta:featherless-ai",
        "messages": [
            {"role": "user", "content": prompt.strip()[:400]}
        ],
        "max_tokens": 150,
        "temperature": 0.7,
        "top_p": 0.95
    }
    
    try:
        # Increased timeout to 15 seconds just in case
        response = requests.post(API_URL, headers=headers, json=payload, timeout=15)
        result = response.json()

        # Handle specific API errors
        if "error" in result:
            error_msg = result["error"].get("message", "")
            if "loading" in error_msg.lower():
                return "AI is warming up on the new provider. Please retry in 10 seconds!"
            return f"AI Error: {error_msg}"

        # Featherless/OpenAI format returns 'choices'
        if "choices" in result and len(result["choices"]) > 0:
            new_text = result["choices"][0]["message"]["content"]
            return _clean_response(new_text)
            
        return "The AI sent an empty response. Try rephrasing your question."

    except Exception as e:
        return f"Connection error: {e}"

def _clean_response(text: str) -> str:
    """
    Cleans the output to remove AI roleplay and cut at clean sentences.
    """
    # Remove common AI 'hallucinations' or roleplay headers
    for stop in ["Student:", "UTME26 AI:", "Question:", "\n\n"]:
        text = text.split(stop)[0].strip()

    # Find the last clean sentence ending (. ! ?)
    match = re.search(r'[^.!?]*[.!?]', text[::-1])
    if match:
        text = text
    
    return text.strip() if text else "I'm not sure how to answer that. Can you ask differently?"
