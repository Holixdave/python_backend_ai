"""Skill loader — every non-underscore .py file in this folder is a skill.

A skill is a small module with:
    NAME      str   short id (used in logs)
    PRIORITY  int   lower runs first (default 50)
    pre(ctx)  -> str   OPTIONAL. Text appended to the system prompt BEFORE the
                       model answers (rules it must follow this turn).
    post(ctx, answer) -> str   OPTIONAL. Runs on the final answer; returns the
                       (possibly corrected) answer.

ctx is a plain dict built once per turn in gpt2_test.py:
    prompt, tools_called, tools_ok, file_results, code_results,
    image_results, sources

Rules of the loader:
  • Drop a new .py file in this folder and it is picked up automatically.
  • A broken skill is logged and skipped — it can never kill a reply.
"""
import importlib
import pkgutil
from typing import List

_CACHE: List = []


def load_skills(force: bool = False) -> List:
    global _CACHE
    if _CACHE and not force:
        return _CACHE
    mods = []
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"{__name__}.{info.name}")
            if not getattr(mod, "NAME", None):
                mod.NAME = info.name
            mods.append(mod)
        except Exception as e:
            print(f"[SKILLS] failed to load {info.name}: {e}")
    mods.sort(key=lambda m: (getattr(m, "PRIORITY", 50), m.NAME))
    _CACHE = mods
    return _CACHE


def run_pre(ctx: dict) -> str:
    """Collect every skill's pre-answer rules into one block."""
    parts = []
    for mod in load_skills():
        fn = getattr(mod, "pre", None)
        if not fn:
            continue
        try:
            text = fn(ctx)
            if text:
                parts.append(text.strip())
        except Exception as e:
            print(f"[SKILLS] {mod.NAME}.pre failed: {e}")
    return ("\n\n" + "\n\n".join(parts) + "\n") if parts else ""


def run_post(ctx: dict, answer: str) -> str:
    """Pipe the final answer through every skill's post-check."""
    for mod in load_skills():
        fn = getattr(mod, "post", None)
        if not fn:
            continue
        try:
            fixed = fn(ctx, answer)
            if isinstance(fixed, str):
                answer = fixed
        except Exception as e:
            print(f"[SKILLS] {mod.NAME}.post failed: {e}")
    return answer
