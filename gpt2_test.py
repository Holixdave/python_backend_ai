# gpt2_test.py
# Uses FREE local GPT-2 via HuggingFace Transformers — NO API key needed!
# Install: pip install transformers torch
# Model downloads automatically on first run (~500MB)

from transformers import pipeline
import re

_pipe = None


def ask_gpt2(prompt: str) -> str:
    """
    Generates a response using GPT-2.
    Prompt should already be formatted before calling this.
    """
    global _pipe
    try:
        if _pipe is None:
            print("Loading GPT-2 model... (first time only, please wait)")
            _pipe = pipeline(
                "text-generation",
                model="gpt2",
                pad_token_id=50256,
            )
            print("GPT-2 model loaded successfully!")

        # Trim prompt to avoid exceeding GPT-2's 1024 token limit
        clean_prompt = prompt.strip()[:400]

        result = _pipe(
            clean_prompt,
            max_new_tokens=120,       # More tokens = fuller answers
            do_sample=True,
            temperature=0.65,         # Lower = more focused, less random
            top_p=0.92,
            top_k=50,
            repetition_penalty=1.3,   # Prevents repeating itself
            num_return_sequences=1,
        )

        full_text = result[0]["generated_text"]

        # Extract only the NEW text after the prompt
        new_text = full_text[len(clean_prompt):].strip()

        if not new_text:
            return "I'm not sure about that one. Try asking a more specific question!"

        # Cut off at a clean sentence ending
        new_text = _clean_response(new_text)

        return new_text

    except Exception as e:
        print(f"GPT-2 error: {e}")
        return f"GPT-2 unavailable: {e}"


def _clean_response(text: str) -> str:
    """
    Cleans GPT-2 output:
    - Cuts at clean sentence boundary
    - Removes incomplete trailing sentences
    - Strips newlines and extra spaces
    """
    # Remove everything after a new "Student:" turn (GPT-2 sometimes role-plays both sides)
    for stop in ["Student:", "UTME26 AI:", "Question:", "\n\n"]:
        if stop in text:
            text = text.split(stop)[0].strip()

    # Find last clean sentence ending
    endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n', '.', '!', '?']
    last_end = -1
    for ending in endings:
        idx = text.rfind(ending)
        if idx > last_end and idx > 15:  # Must have at least 15 chars of content
            last_end = idx + len(ending)

    if last_end > 15:
        text = text[:last_end].strip()

    # Final cleanup
    text = re.sub(r'\s+', ' ', text).strip()

    return text if text else "I'm not sure about that one. Try asking a more specific question!"