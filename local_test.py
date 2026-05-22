import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types

# Initialize FastAPI App
app = FastAPI(title="UTME26 Local Test Backend")

# Initialize Gemini Client Natively
# It automatically picks up the 'GEMINI_API_KEY' environment variable from your CMD
try:
    client = genai.Client()
except Exception as e:
    print(f"Initialization Warning: Make sure GEMINI_API_KEY is set. Error: {e}")

# ---------------------------------------------------------------------------
# SCHEMAS (Matches your exact incoming Swagger framework layout)
# ---------------------------------------------------------------------------
class HistoryItem(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    query: str
    history: Optional[List[HistoryItem]] = []
    imageUrls: Optional[List[str]] = []

# SYSTEM PROMPT
SYSTEM_PROMPT = (
    "You are UTME26 AI, a smart, premium Nigerian AI assistant. "
    "Be mature, clean, properly spaced, and highly professional. "
    "Do not wrap words inside double asterisks (**)."
)

# ---------------------------------------------------------------------------
# API ROUTE
# ---------------------------------------------------------------------------
@app.post("/ai-query")
async def ai_query(payload: QueryRequest):
    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=500, 
            detail="Environment Error: GEMINI_API_KEY variable is missing from host environment."
        )

    print(f"\n[INCOMING REQUEST] Query: '{payload.query}'")
    print(f"[CONTEXT] History Items: {len(payload.history or [])} | Images: {len(payload.imageUrls or [])}")

    # 1. Transform history list to official SDK content structures
    contents = []
    for msg in (payload.history or []):
        # Ignore empty or default Swagger template values
        if not msg.content or msg.content == "string":
            continue
            
        # Convert internal roles to standard API designations ('user' or 'model')
        sdk_role = "user" if msg.role == "user" else "model"
        contents.append(
            types.Content(
                role=sdk_role,
                parts=[types.Part.from_text(text=msg.content)]
            )
        )

    # 2. Append the active user query
    if payload.query and payload.query != "string":
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=payload.query)]
            )
        )
    else:
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    # 3. Configure generation context parameters
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.6,
        max_output_tokens=2048
    )

    # 4. Dispatch the payload to Gemini
    try:
        print("[SDK CALL] Sending structured payload to gemini-2.0-flash...")
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents,
            config=config
        )
        
        # 5. Build your response block
        return {
            "label": "general",
            "answer": response.text if response.text else "Error: Empty response body returned."
        }

    except Exception as e:
        print(f"[API ERROR CRASH] Complete details: {str(e)}")
        raise HTTPException(status_code=502, detail=f"Gemini Engine Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("local_test.py:app", host="127.0.0.1", port=8000, reload=True)