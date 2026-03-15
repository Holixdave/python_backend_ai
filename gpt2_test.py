# gpt2_test.py
# NOTE: Despite the filename, this uses OpenAI's API (gpt-4o-mini).
# If you want FREE local GPT-2, uncomment the HuggingFace section below
# and comment out the OpenAI section. Install with: pip install transformers torch

import os
from openai import OpenAI


def ask_gpt2(prompt: str) -> str:
    """
    Calls OpenAI gpt-4o-mini. Requires OPENAI_API_KEY environment variable.
    Falls back gracefully if key is missing.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "⚠️ GPT request failed: OPENAI_API_KEY not set."

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful AI tutor for Nigerian students preparing for UTME exams."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ GPT request failed: {e}"


# -------------------------------------------------------
# OPTIONAL: Real FREE local GPT-2 (no API key needed)
# pip install transformers torch
# Uncomment below and comment out the OpenAI version above
# -------------------------------------------------------
# from transformers import pipeline
# _pipe = None
#
# def ask_gpt2(prompt: str) -> str:
#     global _pipe
#     if _pipe is None:
#         _pipe = pipeline("text-generation", model="gpt2")
#     result = _pipe(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
#     return result[0]["generated_text"][len(prompt):].strip()