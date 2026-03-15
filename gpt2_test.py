# gpt2_test.py
# Uses FREE local GPT-2 via HuggingFace Transformers — NO API key needed!
# Install once with: pip install transformers torch
# The model downloads automatically the first time (~500MB)

from transformers import pipeline

_pipe = None


def ask_gpt2(prompt: str) -> str:
    """
    Generates a response using the free local GPT-2 model.
    No API key, no internet needed after first download.
    """
    global _pipe
    try:
        if _pipe is None:
            print("Loading GPT-2 model for the first time... (this takes a moment)")
            _pipe = pipeline(
                "text-generation",
                model="gpt2",
                pad_token_id=50256
            )
            print("GPT-2 model loaded!")

        clean_prompt = prompt.strip()[:200]

        result = _pipe(
            clean_prompt,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            num_return_sequences=1,
        )

        full_text = result[0]["generated_text"]
        new_text = full_text[len(clean_prompt):].strip()

        if new_text:
            for ending in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
                idx = new_text.find(ending)
                if idx != -1 and idx > 10:
                    new_text = new_text[:idx + 1].strip()
                    break

        return new_text if new_text else "I'm not sure about that one. Try asking a more specific question!"

    except Exception as e:
        return f"GPT-2 unavailable: {e}"