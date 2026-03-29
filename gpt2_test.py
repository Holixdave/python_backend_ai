import requests
import re
import os

# Get your token from Render Environment Variables
API_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# FIX: Added /v1/chat/completions to the end of the URL
API_URL = "https://router.huggingface.co/v1/chat/completions"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

# Keep this for your main.py health checks
_pipe = True 

def ask_gpt2(prompt: str) -> str:
    payload = {
        "model": "HuggingFaceH4/zephyr-7b-beta:featherless-ai",
        "messages": [{"role": "user", "content": prompt.strip()[:400]}],
        "max_tokens": 150,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=15)
        
        # Check if the request actually worked (200 OK)
        if response.status_code != 200:
            # This will show you exactly what Hugging Face is complaining about
            return f"API Error {response.status_code}: {response.text}"

        result = response.json()

        # Handle the Featherless/OpenAI response format correctly
        if "choices" in result and len(result["choices"]) > 0:
            # result["choices"] is a list, we need the first item [0]
            new_text = result["choices"][0]["message"]["content"]
            return _clean_response(new_text)
            
        return "AI is warming up, please try again in 10 seconds."

    except Exception as e:
        return f"System Error: {str(e)}"


def _clean_response(text: str) -> str:
    """
    Cleans the output to remove AI roleplay and cut at clean sentences.
    """
    # Remove common AI roleplay headers
    for stop in ["Student:", "UTME26 AI:", "Question:", "\n\n"]:
        text = text.split(stop)[0].strip()

    # Final cleanup of extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text if text else "I'm not sure how to answer that. Can you ask differently?"
