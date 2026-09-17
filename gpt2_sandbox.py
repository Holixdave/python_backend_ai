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
# PACKAGE INSTALL — a persistent, SHARED target directory every run_code
# call points PYTHONPATH at, rather than a fresh venv/site-packages per
# call. This is the actual fix for "it fails multiple times installing,
# but it still works": a package like rembg pulls in onnxruntime, a
# 100MB+ binary wheel — installing that fresh on every single call is
# slow enough on a small host that it looks like repeated failures (each
# one probably a timeout, not a real error), even though the previous
# attempt likely finished fine, just not before the caller gave up
# waiting. Sharing one target dir across calls means only the FIRST
# request for a given package ever pays that cost — everything after
# is an instant no-op.
# ---------------------------------------------------------------------------
PACKAGE_INSTALL_DIR = os.environ.get("SANDBOX_PACKAGE_DIR", "/tmp/sandbox_site_packages")
PACKAGE_INSTALL_TIMEOUT_S = 120   # generous — first pull of a heavy package (torch, onnxruntime) is slow
MAX_PACKAGES_PER_RUN = 6
_PACKAGE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")   # blocks flags/paths/shell metacharacters


def _installed_package_names() -> set:
    # pip's --target layout drops a `<name>-<version>.dist-info` folder
    # per package — reading those names back out is how we know what's
    # already there without re-invoking pip (which would itself take a
    # network round trip just to report "already satisfied").
    names = set()
    if os.path.isdir(PACKAGE_INSTALL_DIR):
        for entry in os.listdir(PACKAGE_INSTALL_DIR):
            base = entry.split("-")[0].split(".")[0]
            if base:
                names.add(base.lower().replace("_", "-"))
    return names


def _ensure_packages(packages: list, timeout_s: int = PACKAGE_INSTALL_TIMEOUT_S):
    """
    Installs any of `packages` not already present in the shared
    PACKAGE_INSTALL_DIR. Returns (ok: bool, message: str) — message is
    empty on success (nothing worth telling the AI), or a clear reason
    on failure so it can be surfaced in the code_result error rather
    than the run just silently failing with a confusing ModuleNotFoundError.
    """
    if not packages:
        return True, ""
    packages = packages[:MAX_PACKAGES_PER_RUN]
    for pkg in packages:
        if not _PACKAGE_NAME_RE.match(pkg):
            return False, (
                f"'{pkg}' isn't a valid package name — only letters, digits, "
                "'.', '_', '-' are allowed, no flags, paths, or version pins "
                "with extra characters."
            )

    os.makedirs(PACKAGE_INSTALL_DIR, exist_ok=True)
    already_there = _installed_package_names()
    to_install = [p for p in packages if p.lower().replace("_", "-") not in already_there]
    if not to_install:
        return True, ""

    cmd = [
        "python3", "-m", "pip", "install",
        "--target", PACKAGE_INSTALL_DIR,
        "--no-input", "--disable-pip-version-check", "--no-warn-script-location",
        *to_install,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout_s, text=True)
    except subprocess.TimeoutExpired:
        return False, (
            f"Installing {', '.join(to_install)} took longer than {timeout_s}s "
            "and was stopped. Large binary packages (onnxruntime, torch, "
            "opencv, etc.) can be slow to pull on a small host — the "
            "download that timed out is often still partially cached, so "
            "retrying the same run_code call again is worth trying before "
            "assuming it's broken."
        )
    except Exception as e:
        return False, f"pip failed to run at all: {e}"

    if proc.returncode != 0:
        return False, f"pip install failed:\n{_truncate(proc.stderr, 2000)}"
    return True, ""

# ---------------------------------------------------------------------------
# LANGUAGE_RUNNERS — one entry per supported language. Each entry knows how
# to run itself both ways (subprocess + docker), so adding a language means
# adding one dict here, not touching the chain/execution logic at all.
# ---------------------------------------------------------------------------
LANGUAGE_RUNNERS = {
    "python": {
        "ext": "py",
        "subprocess_cmd": lambda path: ["python3", "-I", "-S", path],
        # Used instead of subprocess_cmd when packages were requested.
        # -I (isolated mode) ignores PYTHONPATH entirely, which would
        # make pip-installed packages unreachable no matter where they
        # live — so packages drop -I but keep -S, which still skips the
        # HOST's own global/user site-packages. Only our own
        # PACKAGE_INSTALL_DIR (set via PYTHONPATH in _clean_env) becomes
        # importable, never whatever happens to be installed on the host.
        "subprocess_cmd_with_packages": lambda path: ["python3", "-S", path],
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


def _clean_env(extra_pythonpath: Optional[str] = None) -> dict:
    # Deliberately NOT os.environ.copy() — this is the whole point. A
    # subprocess that inherited the real environment gets every API key/
    # DB credential/secret this backend process holds, for free. Hand it
    # a minimal, boring environment instead.
    env = {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME": "/tmp",
        "LANG": "C.UTF-8",
    }
    if extra_pythonpath:
        # Only ever set to PACKAGE_INSTALL_DIR (our own pip --target dir),
        # never anything caller-supplied — this is not a general env
        # passthrough, just how the installed packages become importable.
        env["PYTHONPATH"] = extra_pythonpath
    return env


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


def _run_subprocess(runner: dict, code_path: str, timeout_s: int, packages_dir: Optional[str] = None) -> dict:
    if packages_dir and "subprocess_cmd_with_packages" in runner:
        cmd = runner["subprocess_cmd_with_packages"](code_path)
    else:
        cmd = runner["subprocess_cmd"](code_path)
    if not shutil.which(cmd[0]):
        return {"ok": False, "skip_reason": f"'{cmd[0]}' not installed on this host"}

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=os.path.dirname(code_path),
            env=_clean_env(extra_pythonpath=packages_dir),
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


def _run_docker(runner: dict, code_path: str, timeout_s: int, packages_dir: Optional[str] = None) -> dict:
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
    ]
    if packages_dir and os.path.isdir(packages_dir):
        # Same shared install dir subprocess mode uses — mounted read-only
        # so a container can import what was pip-installed on the host
        # without re-downloading anything itself.
        cmd += ["-v", f"{packages_dir}:/packages:ro", "-e", "PYTHONPATH=/packages"]
    cmd += [runner["docker_image"], *runner["docker_cmd"](container_path)]

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
             prefer_isolation: bool = False, userid: Optional[str] = None,
             packages: Optional[list] = None):
    """
    Executes AI-written code for real and returns actual stdout/stderr/
    exit_code — so the AI can verify code works instead of just asserting
    it does. Generator — yields {"type": "status", ...} progress events
    (mount a loading box the instant a run starts, same as generate_image),
    then a single final {"type": "code_result", ...} event.

    language: one of LANGUAGE_RUNNERS' keys or an alias in LANGUAGE_ALIASES
    (currently python / javascript / bash). Unknown language -> failure
    result naming what's supported, no exception raised.

    packages: optional list of pip package names (python only — leave
    empty/None for javascript/bash) your code imports beyond the standard
    library, e.g. ["numpy", "requests"]. These get installed automatically
    into a shared, persistent directory before your code runs — you do
    NOT need to run `pip install` yourself as a separate bash command,
    just list what you need here and import it normally in your code. The
    first request for a given package pays the real install cost; every
    call after that (from anyone, not just you) reuses it instantly. Max
    6 packages per call; invalid-looking names (anything that isn't a
    plain package name — no flags, paths, or shell syntax) are rejected
    before anything runs.

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

    if packages and lang != "python":
        yield {
            "type": "code_result", "success": False, "engine": None,
            "language": lang, "stdout": "", "stderr": "", "exit_code": None,
            "error": f"packages is only supported for python right now, not {lang}.",
        }
        return

    code = code[:MAX_CODE_CHARS]
    timeout_s = max(1, min(int(timeout_s or DEFAULT_TIMEOUT_S), MAX_TIMEOUT_S))

    packages_dir = None
    if packages:
        yield {
            "type": "status",
            "text": f"Installing {', '.join(packages[:MAX_PACKAGES_PER_RUN])}...",
            "detail": lang, "icon": "sandbox",
        }
        ok, install_err = _ensure_packages(packages)
        if not ok:
            yield {
                "type": "code_result", "success": False, "engine": None,
                "language": lang, "stdout": "", "stderr": "", "exit_code": None,
                "error": f"Couldn't install required packages: {install_err}",
            }
            return
        packages_dir = PACKAGE_INSTALL_DIR

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
            outcome = link["call"](runner, code_path, timeout_s, packages_dir)
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
