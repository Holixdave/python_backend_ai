# search_query.py — the pure, dependency-free gate that decides whether a
# string is something worth sending to a search engine. Lives in its own file
# so tests can import it without Firebase/Groq/etc. gpt2_functions.py imports
# everything below from here.
import re
from typing import Optional

SEARCH_TRIGGER_KEYWORDS = [
    "search", "look up", "look for",
    "latest", "news", "where can i", "netnaija",
    "what is the price of", "who won",
]
# REMOVED "find", "link", "download", "recent": they fire on ordinary code/chat
# ("find the bug", "download button", "recent changes") and caused junk searches.

# Words that carry no search value. If nothing is left after removing these,
# there is nothing to look up — searching anyway is what produced rubbish.
_QUERY_STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "you", "your", "yours", "can", "could", "would",
    "should", "will", "just", "that", "this", "these", "those", "with", "have", "has", "had",
    "not", "but", "its", "it's", "what", "who", "how", "why", "when", "where", "which", "there",
    "their", "them", "then", "than", "too", "very", "also", "about", "some", "any", "all",
    "from", "into", "out", "get", "got", "let", "lets", "need", "want", "like", "know", "think",
    "say", "said", "tell", "make", "made", "does", "did", "dont", "doesnt", "cant", "okay", "ok",
    "yes", "yeah", "thanks", "thank", "hello", "hey", "hii", "hmm", "well", "really", "thing",
    "things", "stuff", "something", "anything", "everything", "again", "more", "now", "one",
    "model", "ai", "bot", "search", "searching", "look", "find", "info", "information",
}
_STATEMENT_STARTS = ("i ", "i'm ", "im ", "i've ", "i'd ", "we ", "you ", "my ", "your ", "it ", "that ", "this ", "he ", "she ", "they ",
                     "can you ", "can u ", "could you ", "would you ", "will you ", "do you ", "are you ",
                     "explain ", "tell me ", "why do", "why does", "why is", "how do you")


def is_meaningful_query(query: str, original_prompt: Optional[str] = None) -> bool:
    """
    True only if `query` looks like something a search engine can answer: a
    short noun-phrase with at least one real content word. False for empty
    strings, pure filler, long sentences, statements/opinions addressed to
    the assistant ("i think you are wrong"), or a verbatim copy of a long
    user message. Deterministic on purpose — this is the last gate before
    the web gets hit, so it must never depend on another model call.
    """
    q = (query or "").strip().lower()
    if not q:
        return False
    tokens = re.findall(r"[a-z0-9\u00c0-\u024f']+", q)
    if not tokens or len(tokens) > 10:
        return False
    content = [t for t in tokens if t not in _QUERY_STOPWORDS and len(t) > 2]
    if not content:
        return False
    if len(tokens) > 4 and len(content) / len(tokens) < 0.4:
        return False                       # mostly filler words = a sentence, not a query
    if q.startswith(_STATEMENT_STARTS) and len(tokens) > 3:
        return False                       # "i think...", "you are...", "my code..." = statement
    if original_prompt:
        op = original_prompt.strip().lower()
        if q == op and len(tokens) > 6:
            return False                   # the user's sentence pasted back as a query
    return True


def needs_web_search(prompt: str) -> bool:
    t = prompt.lower()
    if not any(k in t for k in SEARCH_TRIGGER_KEYWORDS):
        return False
    return is_meaningful_query(build_search_query(prompt), prompt)


_QUERY_FILLER_PHRASES = [
    "can you find me", "can u find me", "can you get me", "can i get", "can you get", "can you find", "can u find", "find me",
    "search for", "search", "look for", "look up", "get me", "show me", "i want",
    "please", "dude", "man", "kindly", "help me",
    "abeg", "biko", "pls", "plz", "sha", "na so", "una",
]
# Longest first so "search for" is removed before "search".
_FILLER_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p in sorted(_QUERY_FILLER_PHRASES, key=len, reverse=True)) + r")\b"
)


def build_search_query(prompt: str) -> str:
    # BUG FIXED: this used plain str.replace(), so "man" was cut out of
    # "human", "manage", "woman", "Manchester" and "una"/"sha" out of
    # "tuna"/"shake" — producing rubbish queries like "hu", "age", "chester".
    # Now whole words only.
    q = _FILLER_RE.sub(" ", (prompt or "").lower())
    return re.sub(r"\s+", " ", q).strip(" ,.!?-")


