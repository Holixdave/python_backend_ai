import requests
import re
import os

# Get your token from Hugging Face Settings > Access Tokens
# It's safer to set this as an Environment Variable in Render
API_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

# Keep this for your main.py health checks
_pipe = True 

def ask_gpt2(prompt: str) -> str:
    payload = {
        "inputs": prompt.strip()[:400],
        "parameters": {
            "max_new_tokens": 120,
            "temperature": 0.65,
            "top_p": 0.92,
            "repetition_penalty": 1.3,
            "return_full_text": False  # Only gives us the new text
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
        result = response.json()

        # Handle API errors or model warming up
        if "error" in result:
            return "AI is warming up on Hugging Face. Please try again in 30 seconds!"

        if isinstance(result, list) and len(result) > 0:
            new_text = result[0].get('generated_text', "")
            return _clean_response(new_text)
            
        return "I'm not sure about that. Try a different question!"

    except Exception as e:
        return f"Connection error: {e}"

def _clean_response(text: str) -> str:
    # Removes AI role-playing and cuts at clean sentence boundaries
    for stop in ["Student:", "UTME26 AI:", "Question:", "\n\n"]:
        text = text.split(stop)[0].strip()

    # Find last sentence ending (. ! ?)
    match = re.search(r'[^.!?]*[.!?]', text[::-1])
    if match:
        text = text[:len(text) - match.start()]
    
    return text.strip() if text else "I'm not sure. Can you rephrase?"
