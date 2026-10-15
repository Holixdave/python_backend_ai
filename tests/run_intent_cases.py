"""ONLINE check of the real intent classifier (needs GROQ_API_KEY + firebase env).
Run:  python tests/run_intent_cases.py     — add a case to intent_cases.json every time routing misbehaves."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpt2_functions import classify_intent

cases = json.load(open(os.path.join(os.path.dirname(__file__), "intent_cases.json")))
bad = 0
for c in cases:
    got = classify_intent(c["prompt"])
    ok = got["search_type"] == c["search_type"]
    bad += (not ok)
    print(("PASS" if ok else "FAIL"), repr(c["prompt"]), "->", got["search_type"], repr(got.get("search_query")),
          "" if ok else f"(wanted {c['search_type']})")
print(f"\n{len(cases) - bad}/{len(cases)} passed")
sys.exit(1 if bad else 0)
