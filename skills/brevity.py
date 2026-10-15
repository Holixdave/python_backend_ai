"""brevity — short answers, answer first, no filler.

pre : tells the model the length rules.
post: removes generic closer lines ("Is there anything else you need?",
      "Let me know if you need anything", "Feel free to ask"). Specific
      follow-up offers ("Want me to add the giving page?") are kept.
"""
import re

NAME = "brevity"
PRIORITY = 20

PRE = (
    "BREVITY RULES (override the formatting rules above whenever they conflict):\n"
    "• Lead with the answer or the result in the first sentence. No greeting, no restating the question.\n"
    "• Simple question: 1-3 sentences. Normal answer: under ~120 words. Go longer only when the user "
    "asks for detail, code, or a document.\n"
    "• Use ### headers and bold ONLY for long multi-section answers. Short replies have no headers.\n"
    "• Do not repeat earlier status ('migration complete', 'ready to deploy') in unrelated replies.\n"
    "• Do not end with a generic offer ('anything else?'). Ask at most ONE question, and only if you "
    "truly need the answer to continue."
)

_FILLER_LINE = re.compile(
    r"^\W*(?:\*\*)?\s*(?:"
    r"is there anything else[^\n]*|anything else (?:you )?(?:need|want|would)[^\n]*|"
    r"let me know if (?:you )?(?:need|want|have)[^\n]*|feel free to (?:ask|reach)[^\n]*|"
    r"i(?:'m| am) here if you need[^\n]*|hope (?:this|that) helps[^\n]*|"
    r"(?:do you )?need (?:any )?(?:further|more) (?:help|assistance)[^\n]*"
    r")\s*(?:\*\*)?\W*$",
    re.I,
)


def pre(ctx):
    return PRE


def post(ctx, answer):
    if not answer:
        return answer
    parts = re.split(r"(```.*?```)", answer, flags=re.S)
    for idx in range(0, len(parts), 2):
        lines = parts[idx].split("\n")
        parts[idx] = "\n".join(l for l in lines if not _FILLER_LINE.match(l))
    out = re.sub(r"\n{3,}", "\n\n", "".join(parts)).strip()
    return out or answer
