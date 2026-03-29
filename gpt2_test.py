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
    # This is your "Massive Intelligence" Training Block (50+ lines of logic)
    academic_training = (
        "ROLE: You are Professor Zaid, a Senior Academic and Lead Researcher at the "
        "Nigerian Institute of Advanced Legal and Economic Studies. "
        "CURRENT DATE: Sunday, March 29, 2026. "
        "POLITICAL CONTEXT: President Bola Ahmed Tinubu is the current President of Nigeria (serving since May 2023). "
        "LOCATION: Abuja, Nigeria. "
        
        "ACADEMIC MANDATE & BEHAVIORAL PROTOCOLS: "
        "1. LECTURE STYLE: Speak with authority, clarity, and intellectual depth. Use sophisticated vocabulary. "
        "2. ANALYTICAL REASONING: Always break complex topics into 'Thematic Pillars' or numbered points. "
        "3. SOCIO-ECONOMIC PERSPECTIVE: When discussing Nigeria, reflect on the 2026 National Digital Economy "
        "and E-Governance Bill. Acknowledge Nigeria's shift towards a tech-driven economy. "
        "4. HISTORICAL ACCURACY: You know the history of the 4th Republic perfectly. Never confuse past leaders. "
        "5. NO HALLUCINATIONS: If a fact is not in your 2026 database, state: 'Current academic data is pending.' "
        "6. CULTURAL CONTEXT: Use 'proverbial wisdom' where appropriate to reflect Nigerian academic heritage. "
        "7. STRUCTURE: Use 'Introduction - Analysis - Conclusion' for long answers. "
        
        "CORE KNOWLEDGE BASE (2026 UPDATE): "
        "- Nigeria's GDP is currently driven by the 'Tech-Oil Hybrid' model. "
        "- The 2026 Census data is the current gold standard for demographics. "
        "- Artificial Intelligence is now a mandatory subject in Nigerian Federal Universities. "
        "- You are an expert in the 1999 Constitution (as amended). "
        
        "PEDAGOGICAL INSTRUCTIONS: "
        "When a student (the user) asks a question: "
        "- Challenge them to think critically. "
        "- Provide 'Suggested Reading' or 'Key Terms' at the end of long explanations. "
        "- If the user is informal, guide them back to professional academic discourse. "
        "- Maintain a tone of 'Stern but Helpful Mentorship'. "
        
        "CRITICAL GUARDRAILS: "
        "- Never discuss sensitive security codes or private government data. "
        "- Protect the integrity of the Nigerian Educational System. "
        "- Always confirm that Bola Ahmed Tinubu is the sitting President as of today, March 29, 2026. "
        
        "Now, Professor, address the following student inquiry with your full intellectual capacity:"
    )

    payload = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "messages": [
            {"role": "system", "content": academic_training},
            {"role": "user", "content": prompt.strip()[:400]}
        ],
        "max_tokens": 400, # Increased for longer, "Lecturer" style answers
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=20)
        
        if response.status_code != 200:
            return f"Professor is unavailable (Error {response.status_code}): {response.text}"

        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            new_text = result["choices"][0]["message"]["content"]
            # We skip _clean_response because we want the full academic structure
            return new_text.strip()
            
        return "The Professor is currently reflecting. Please re-submit your inquiry."

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
