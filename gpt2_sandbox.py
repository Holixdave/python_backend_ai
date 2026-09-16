#!/usr/bin/env python3
# gpt2_sandbox.py — run_code tool: lets the AI actually execute code it
# wrote and see real stdout/stderr/exit_code back, instead of just
# guessing whether it works.
# ─────────────────────────────────────────────────────────────────────────────
# SAME SHAPE as generate_image() in gpt2_functions.py on purpose:
#   - a generator that yields {"type": "status", ...} progress events as
#     it goes, then a single final {"type": "code_result", ...} event —
#     so gpt2_test.py can special-case it exactly like generate_image/
#     create_image/build_file and stream real progress to the frontend
#     instead of sitting on dead air while the code runs.
#   - a PROVIDER CHAIN, same idea as _build_image_provider_chain(): try
#     the fast path first (local subprocess), fall through to the next
#     engine (Docker) if that one has a problem — missing interpreter,
#     permission error, Docker itself not installed, etc. Whichever
#     engine actually ran is reported back in the result as "engine".
#
# WHAT THIS IS NOT: a full multi-tenant isolation story. subprocess mode
# is resource-limited (CPU/memory/procs) and env-scrubbed but shares the
# host kernel and can still reach the network — treat it as "misbehaving
# code protection", not "hostile code protection". Docker mode is the
# real boundary (own filesystem, `--network none`, dropped capabilities)
# and is what this chain prefers when it's available and the code needs
# stronger isolation; wire in an external sandbox API (Piston/E2B/etc.)
# as a third link in the chain later if you need multi-tenant-safe
# execution without owning a Docker host at all.
# ─────────────────────────────────────────────────────────────────────────────

import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from typing import Optional

# ---------------------------------------------------------------------------
# LIMITS — deliberately conservative. Raise per-call via the `timeout_s`
# arg if a real use case needs more, never by changing these defaults.
# ---------------------------------------------------------------------------
DEFAULT_TIMEOUT_S = 10
MAX_TIMEOUT_S = 30
MAX_CODE_CHARS = 20_000
MAX_OUTPUT_CHARS = 8_000          # per stream (stdout/stderr), truncated past this
MAX_MEMORY_BYTES = 256 * 1024 * 1024   # 256 MB, subprocess mode only
MAX_PROCS = 32                          # fork-bomb guard, subprocess mode only

# ---------------------------------------------------------------------------
# LANGUAGE_RUNNERS — one entry per supported language. Each entry knows how
# to run itself both ways (subprocess + docker), so adding a language means
# adding one dict here, not touching the chain/execution logic at all.
# ---------------------------------------------------------------------------
LANGUAGE_RUNNERS = {
    "python": {
        "ext": "py",
        "subprocess_cmd": lambda path: ["python3", "-I", "-S", path],
        "docker_image": "python:3.12-slim",
        "docker_cmd": lambda container_path: ["python3", container_path],
    },
    "javascript": {
        "ext": "js",
        "subprocess_cmd": lambda path: ["node", "--no-addons", path],
        "docker_image": "node:20-slim",
        "docker_cmd": lambda container_path: ["node", container_path],
    },
    "bash": {
        "ext": "sh",
        "subprocess_cmd": lambda path: ["bash", "--noprofile", "--norc", path],
        "docker_image": "bash:5",
        "docker_cmd": lambda container_path: ["bash", container_path],
    },
}
LANGUAGE_ALIASES = {"py": "python", "js": "javascript", "node": "javascript", "sh": "bash", "shell": "bash"}


def _normalize_language(language: str) -> Optional[str]:
    lang = (language or "python").strip().lower()
    lang = LANGUAGE_ALIASES.get(lang, lang)
    return lang if lang in LANGUAGE_RUNNERS else None


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated, {len(text) - limit} more chars]"


def _clean_env() -> dict:
    # Deliberately NOT os.environ.copy() — this is the whole point. A
    # subprocess that inherited the real environment gets every API key/
    # DB credential/secret this backend process holds, for free. Hand it
    # a minimal, boring environment instead.
    return {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME": "/tmp",
        "LANG": "C.UTF-8",
    }


def _resource_limits():
    # preexec_fn for subprocess.Popen — applies OS-level caps to the CHILD
    # only, never this process. Linux-only (resource module); silently
    # skipped on platforms without it (e.g. during local Windows dev) —
    # subprocess mode still works there, just without these guardrails,
    # so prefer the docker link in the chain for anything untrusted on
    # non-Linux hosts.
    try:
        import resource
    except ImportError:
        return None

    def _apply():
        resource.setrlimit(resource.RLIMIT_CPU, (MAX_TIMEOUT_S, MAX_TIMEOUT_S))
        resource.setrlimit(resource.RLIMIT_AS, (MAX_MEMORY_BYTES, MAX_MEMORY_BYTES))
        resource.setrlimit(resource.RLIMIT_NPROC, (MAX_PROCS, MAX_PROCS))
        resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))  # 10MB file writes max
        os.setsid()  # own process group, so a timeout kill takes any children with it

    return _apply


def _docker_available() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        result = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=3
        )
        return result.returncode == 0
    except Exception:
        return False


def _run_subprocess(runner: dict, code_path: str, timeout_s: int) -> dict:
    cmd = runner["subprocess_cmd"](code_path)
    if not shutil.which(cmd[0]):
        return {"ok": False, "skip_reason": f"'{cmd[0]}' not installed on this host"}

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=os.path.dirname(code_path),
            env=_clean_env(),
            capture_output=True,
            timeout=timeout_s,
            preexec_fn=_resource_limits(),
            text=True,
        )
        return {
            "ok": True,
            "exit_code": proc.returncode,
            "stdout": _truncate(proc.stdout),
            "stderr": _truncate(proc.stderr),
            "duration_s": round(time.time() - start, 2),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as e:
        return {
            "ok": True,
            "exit_code": None,
            "stdout": _truncate(e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")),
            "stderr": _truncate((e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")) + f"\n[killed — exceeded {timeout_s}s timeout]"),
            "duration_s": round(time.time() - start, 2),
            "timed_out": True,
        }
    except Exception as e:
        return {"ok": False, "skip_reason": f"subprocess launch failed: {e}"}


def _run_docker(runner: dict, code_path: str, timeout_s: int) -> dict:
    if not _docker_available():
        return {"ok": False, "skip_reason": "docker not installed or daemon not reachable"}

    container_name = f"sandbox-{uuid.uuid4().hex[:12]}"
    container_path = f"/code/main.{runner['ext']}"
    cmd = [
        "docker", "run", "--rm",
        "--name", container_name,
        "--network", "none",                 # real isolation subprocess mode can't offer
        "--cap-drop", "ALL",
        "--memory", f"{MAX_MEMORY_BYTES}",
        "--pids-limit", str(MAX_PROCS),
        "--user", "nobody",
        "-v", f"{os.path.dirname(code_path)}:/code:ro",
        runner["docker_image"],
        *runner["docker_cmd"](container_path),
    ]

    start = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout_s + 5, text=True)
        return {
            "ok": True,
            "exit_code": proc.returncode,
            "stdout": _truncate(proc.stdout),
            "stderr": _truncate(proc.stderr),
            "duration_s": round(time.time() - start, 2),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", container_name], capture_output=True)
        return {
            "ok": True,
            "exit_code": None,
            "stdout": "",
            "stderr": f"[killed — exceeded {timeout_s}s timeout]",
            "duration_s": round(time.time() - start, 2),
            "timed_out": True,
        }
    except Exception as e:
        return {"ok": False, "skip_reason": f"docker run failed: {e}"}


def _build_sandbox_chain(prefer_isolation: bool) -> list:
    # Mirrors _build_image_provider_chain's shape in gpt2_functions.py:
    # an ordered list this function just walks, falling through on any
    # entry that reports ok=False. prefer_isolation flips the order —
    # untrusted/"just run whatever this looks risky" code should hit
    # Docker first rather than subprocess.
    subprocess_link = {"name": "subprocess", "call": _run_subprocess}
    docker_link = {"name": "docker", "call": _run_docker}
    return [docker_link, subprocess_link] if prefer_isolation else [subprocess_link, docker_link]


def run_code(code: str, language: str = "python", timeout_s: int = DEFAULT_TIMEOUT_S,
             prefer_isolation: bool = False, userid: Optional[str] = None):
    """
    Executes AI-written code for real and returns actual stdout/stderr/
    exit_code — so the AI can verify code works instead of just asserting
    it does. Generator — yields {"type": "status", ...} progress events
    (mount a loading box the instant a run starts, same as generate_image),
    then a single final {"type": "code_result", ...} event.

    language: one of LANGUAGE_RUNNERS' keys or an alias in LANGUAGE_ALIASES
    (currently python / javascript / bash). Unknown language -> failure
    result naming what's supported, no exception raised.

    Engine chain: tries subprocess first (fast, resource-limited, scrubbed
    env, no inherited secrets), falls through to Docker (real isolation —
    own filesystem, --network none, dropped capabilities, 'nobody' user)
    if subprocess isn't usable on this host or the interpreter is missing.
    Pass prefer_isolation=True to flip that order for code you'd rather
    not run outside a container at all. Whichever engine actually ran is
    reported back as "engine" on the final event.

    userid is accepted (and auto-injected from session_context by
    gpt2_tools.execute_tool / the special-cased dispatch in gpt2_test.py,
    same as every other tool) for future use — logging/rate-limiting a
    per-user daily execution quota the same way check_image_quota gates
    generate_image — not used for anything yet.
    """
    lang = _normalize_language(language)
    if lang is None:
        yield {
            "type": "code_result", "success": False, "engine": None,
            "language": language, "stdout": "", "stderr": "",
            "exit_code": None,
            "error": f"Unsupported language '{language}'. Supported: {', '.join(LANGUAGE_RUNNERS)}.",
        }
        return

    if not code or not code.strip():
        yield {
            "type": "code_result", "success": False, "engine": None,
            "language": lang, "stdout": "", "stderr": "", "exit_code": None,
            "error": "empty code",
        }
        return

    code = code[:MAX_CODE_CHARS]
    timeout_s = max(1, min(int(timeout_s or DEFAULT_TIMEOUT_S), MAX_TIMEOUT_S))

    # detail carries the language, same slot generate_image uses for its
    # prompt — the frontend reads event.statusDetail off the FIRST
    # icon:"sandbox" status to know what to show on the running-card.
    yield {"type": "status", "text": f"Running {lang} code...", "detail": lang, "icon": "sandbox"}

    runner = LANGUAGE_RUNNERS[lang]
    run_dir = tempfile.mkdtemp(prefix="sandbox-")
    code_path = os.path.join(run_dir, f"main.{runner['ext']}")
    try:
        with open(code_path, "w") as f:
            f.write(code)

        chain = _build_sandbox_chain(prefer_isolation)
        for link in chain:
            yield {"type": "status", "text": f"Executing via {link['name']}...", "detail": None, "icon": "sandbox"}
            outcome = link["call"](runner, code_path, timeout_s)
            if outcome.get("ok"):
                print(f"[SANDBOX] ran {lang} via {link['name']} — exit={outcome.get('exit_code')}, "
                      f"timed_out={outcome.get('timed_out')}, userid={userid!r}")
                yield {
                    "type": "code_result",
                    "success": (outcome.get("exit_code") == 0) and not outcome.get("timed_out"),
                    "engine": link["name"],
                    "language": lang,
                    "stdout": outcome.get("stdout", ""),
                    "stderr": outcome.get("stderr", ""),
                    "exit_code": outcome.get("exit_code"),
                    "timed_out": outcome.get("timed_out", False),
                    "duration_s": outcome.get("duration_s"),
                }
                return
            print(f"[SANDBOX] {link['name']} unavailable for {lang} ({outcome.get('skip_reason')}), trying next engine")

        yield {
            "type": "code_result", "success": False, "engine": None,
            "language": lang, "stdout": "", "stderr": "", "exit_code": None,
            "error": "No execution engine available (subprocess interpreter missing and Docker not reachable).",
        }
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
