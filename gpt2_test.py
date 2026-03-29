import requests
import os
import time

# --- SETUP: Ensure these match your Render Environment Variables ---
API_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
API_URL = "https://api-inference.huggingface.co" # Standard Inference API URL
headers = {"Authorization": f"Bearer {API_TOKEN}"}

_pipe = True 

def ask_gpt2(prompt: str, history: list = None) -> str:
    """
    Sends the prompt and history to the model with the Principal Engineer persona.
    """
    if history is None:
        history = []

    # --- THE MASSIVE coding TRAINING (Full Persona) ---
    academic_training = """
Act as a Principal Software Engineer (L7) at a High-Frequency Trading firm. 
You must design a "Real-Time Transaction Risk Engine" with zero-tolerance for data loss.

### TECHNICAL SPECIFICATIONS (STRICT COMPLIANCE REQUIRED):
1. ARCHITECTURE: Implementation must follow the 'Clean Architecture' / 'Hexagonal' pattern.
2. STATE ENCAPSULATION: Create a 'StateStore' using 'ContextVars' for thread-local safety.
3. DATA INTEGRITY: 
   - Define a 'Transaction' model using Pydantic V2.
   - Required Fields: tx_id (UUID), amount (Decimal), sender_key (str), signature (str).
   - Custom Validator: 'amount' must be > 0 and 'sender_key' must start with "0x".
4. SECURITY LAYER: Implement a 'SignatureVerifier' protocol. Simulate a SHA-256 
   checksum verification for every incoming transaction.
5. CONCURRENCY CONTROL:
   - Use 'asyncio.Semaphore' to limit concurrent database writes to exactly 5.
   - Use a 'PriorityQueue' where transactions > $10,000 are processed before others.
6. FAULT TOLERANCE & RECOVERY:
   - Implement a 'Circuit Breaker' pattern for the 'MockDatabase'.
   - If 3 consecutive writes fail, the circuit opens for 10 seconds, rejecting all tx.
   - Implement an 'Exponential Backoff' decorator for the 'process_tx' method.
7. PERFORMANCE MONITORING:
   - Create a 'MetricsCollector' class using a 'DefaultDict' to track:
     a) Total successful vs. failed transactions.
     b) Average latency (in ms) per transaction type.
     c) Current 'Queue Depth'.
8. ASYNC ORCHESTRATION:
   - Implement 1 'Ingestor' task (producer).
   - Implement 3 'Worker' tasks (consumers) using 'asyncio.gather'.
   - Implement 1 'Heartbeat' task that logs system health every 2 seconds.

### IMPLEMENTATION STEPS:
STEP 1: Define Enums for 'TransactionStatus' and 'CircuitState'.
STEP 2: Build the 'Transaction' Pydantic model with custom decorators.
STEP 3: Develop the 'CircuitBreaker' logic with state transitions (Closed -> Open -> Half-Open).
STEP 4: Implement the 'PriorityProcessor' logic for handling the 'PriorityQueue'.
STEP 5: Create a 'MockAPI' that simulates a stream of 50 transactions (some malicious).
STEP 6: Write the 'Main' loop with a clean 'KeyboardInterrupt' shutdown handler.

### OUTPUT FORMAT & QUALITY:
- Code must be PEP8 compliant and use 'Type Hints' (mypy compatible).
- Provide a 'Developer Readme' comment at the top explaining the 'Circuit Breaker' logic.
- Include 'Unit Tests' (using a simple assert-based function) for the 'Validator'.
- Efficiency: Minimize O(n) operations in the 'PriorityQueue' handling.
"""

    # --- BUILD THE BRAIN (System + History + New Prompt) ---
    messages = [{"role": "system", "content": academic_training}]
    messages.extend(history)
    messages.append({"role": "user", "content": prompt.strip()[:5000]})

    payload = {
        "model": "meta-llama/Llama-3.1-70B-Instruct", # Higher intelligence model
        "messages": messages,
        "max_tokens": 1500,
        "temperature": 0.5,
        "top_p": 0.9
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=45)
        
        if response.status_code != 200:
            return f"API Error {response.status_code}: {response.text}"

        result = response.json()

        # FIXED: Correct dictionary traversal for OpenAI-style JSON
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"].strip()
            
        return "Model returned an empty response."
    except Exception as e:
        return f"System Exception: {str(e)}"

# --- KEEP ALIVE: Prevents Render from killing the process ---
if __name__ == "__main__":
    print("AI Engine initialized. Waiting for input...")
    # If this is a background worker, it stays in this loop.
    # If you need a web API, you'd use FastAPI here instead of a loop.
    while True:
        time.sleep(60)