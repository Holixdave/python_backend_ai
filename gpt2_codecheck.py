#!/usr/bin/env python3
# gpt2_codecheck.py - python_check tool: find Python SYNTAX errors (and, when pyflakes is
# installed, undefined names / other real bugs) in workspace files BEFORE running them.
# ----------------------------------------------------------------------------
# A syntax check is NOT a run test. The result always says so: after a clean check the AI must
# still run the code (workspace_run / run_code) and only then may it say the code "works".
#
#   python3 gpt2_codecheck.py --selftest
# ----------------------------------------------------------------------------

import ast
import io
import os
from typing import Optional

MAX_FILES = 150
MAX_FILE_BYTES = 1_500_000
MAX_ISSUES = 15
_SKIP_DIRS = {"node_modules", ".git", "__pycache__", "venv", ".venv", "env", "site-packages", ".mypy_cache"}


def _py_files(base: str) -> list:
    if os.path.isfile(base):
        return [base] if base.endswith(".py") else []
    found = []
    for root, dirs, files in os.walk(base):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS and not d.startswith("."))
        for name in sorted(files):
            if name.endswith(".py"):
                found.append(os.path.join(root, name))
                if len(found) >= MAX_FILES:
                    return found
    return found


def check_source(src: str, filename: str = "<code>") -> dict:
    """Pure function (no workspace): {'syntax': [issue...], 'flakes': [issue...], 'flakes_ran': bool}."""
    out = {"syntax": [], "flakes": [], "flakes_ran": False}
    try:
        tree = compile(src, filename, "exec", flags=ast.PyCF_ONLY_AST, dont_inherit=True)
    except SyntaxError as e:
        line = e.lineno or 0
        lines = src.splitlines()
        text = (e.text or (lines[line - 1] if 0 < line <= len(lines) else "")).rstrip("\n")
        out["syntax"].append({
            "line": line, "col": e.offset or 0,
            "error": f"{type(e).__name__}: {e.msg}", "code": text.strip()[:160],
        })
        return out
    except (ValueError, RecursionError, MemoryError) as e:
        out["syntax"].append({"line": 0, "col": 0, "error": f"{type(e).__name__}: {str(e)[:120]}", "code": ""})
        return out

    try:
        from pyflakes import checker as _pc, messages as _pm   # optional dependency
    except Exception:
        return out
    try:
        w = _pc.Checker(tree, filename)
        out["flakes_ran"] = True
        for m in sorted(w.messages, key=lambda m: m.lineno):
            text = m.message % m.message_args
            low = text.lower()
            if isinstance(m, (_pm.UndefinedName, _pm.UndefinedLocal, _pm.UndefinedExport, _pm.DuplicateArgument,
                              _pm.ReturnOutsideFunction, _pm.ContinueOutsideLoop, _pm.BreakOutsideLoop)) \
                    or "undefined name" in low:
                out["flakes"].append({"line": m.lineno, "col": getattr(m, "col", 0), "error": text})
    except Exception as e:                                  # a checker bug must never block the tool
        print(f"[CODECHECK] pyflakes failed: {type(e).__name__}: {e}")
        out["flakes_ran"] = False
    return out


def python_check(userid: str, path: str = "") -> dict:
    """
    Check Python files in the workspace for syntax errors (and undefined names when available)
    WITHOUT running them. Call it after you write or edit any .py file, fix every listed problem
    with workspace_edit, and call it again. A clean result still does NOT prove the code runs:
    afterwards run it for real with workspace_run (or run_code) before saying it works.

    Args:
        path: a .py file or a folder in the workspace (empty = the whole workspace).
    """
    from gpt2_workspace import _rate_ok, _safe_path, _rel

    if not _rate_ok(userid, "python_check", 120):
        return {"success": False, "error": "python_check limit reached (120 per hour) - try again later."}
    try:
        base = _safe_path(userid, path or "")
    except Exception as e:
        return {"success": False, "error": str(e)}
    if not os.path.exists(base):
        return {"success": False, "error": f"'{path}' doesn't exist in the workspace."}

    files = _py_files(base)
    if not files:
        return {"success": False, "error": "No .py files found there."}

    errors, flakes, skipped, flakes_ran = [], [], [], False
    for fp in files:
        rel = _rel(userid, fp)
        try:
            if os.path.getsize(fp) > MAX_FILE_BYTES:
                skipped.append(f"{rel} (too big)")
                continue
            with open(fp, "rb") as fh:
                raw = fh.read()
            try:
                src = raw.decode("utf-8")
            except UnicodeDecodeError:
                errors.append({"file": rel, "line": 0, "col": 0, "error": "file is not valid UTF-8 text", "code": ""})
                continue
        except OSError as e:
            skipped.append(f"{rel} ({e.__class__.__name__})")
            continue
        res = check_source(src, rel)
        flakes_ran = flakes_ran or res["flakes_ran"]
        errors += [dict(i, file=rel) for i in res["syntax"]]
        flakes += [dict(i, file=rel) for i in res["flakes"]]

    problems = errors + flakes
    out = {
        "success": True,
        "files_checked": len(files) - len(skipped),
        "ok": not problems,
        "syntax_errors": errors[:MAX_ISSUES],
        "likely_bugs": flakes[:MAX_ISSUES],
        "deep_check": "pyflakes (undefined names)" if flakes_ran else "not available - syntax only",
        "not_tested": "this does not run the code",
    }
    if skipped:
        out["skipped"] = skipped[:10]
    if len(problems) > MAX_ISSUES:
        out["more_not_shown"] = len(problems) - MAX_ISSUES
    if problems:
        out["next_step"] = ("Fix ONLY the listed lines with workspace_edit (read the lines around them first), then call "
                            "python_check again. Do not rewrite whole files.")
    else:
        out["next_step"] = ("No syntax errors found. Now RUN it for real with workspace_run (or run_code) - e.g. the main file, "
                            "or its tests - and fix anything that crashes. Only say it works after the run succeeded, and "
                            "say exactly what you ran.")
    return out


# -- self-test: python3 gpt2_codecheck.py --selftest ------------------------
def _selftest() -> None:
    def line(name, ok, detail=""):
        print(f"[{'OK ' if ok else 'NO '}] {name}" + (f" - {detail}" if detail else ""))

    good = check_source("def f(x):\n    return x + 1\nprint(f(2))\n")
    line("good code has no syntax errors", not good["syntax"])
    bad = check_source("def f(x)\n    return x\n")
    line("missing colon is caught", bool(bad["syntax"]) and bad["syntax"][0]["line"] == 1, str(bad["syntax"]))
    ind = check_source("def f():\nreturn 1\n")
    line("bad indentation is caught", bool(ind["syntax"]), str(ind["syntax"]))
    fl = check_source("def f():\n    return undefined_thing\n")
    line("undefined name", True, f"pyflakes ran={fl['flakes_ran']} found={fl['flakes']}")
    print("done")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
