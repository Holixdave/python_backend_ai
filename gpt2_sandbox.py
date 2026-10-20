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

import importlib.metadata
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from typing import Optional

try:                      # Linux/macOS only; local Windows dev just skips the cross-process lock
    import fcntl
except ImportError:       # pragma: no cover
    fcntl = None

# gRPC / event-engine libraries print these to stderr and sometimes abort at interpreter
# shutdown (exit code != 0) even though the user's code finished fine. They made good runs
# show up as "Run failed" (and the AI then re-ran the same code a second time).
_RUNTIME_NOISE = re.compile(
    r"^.*(ev_epoll1_linux\.cc|epoll_wait error|epoll_ctl failed|\(event_engine\)|"
    r"fork_posix\.cc|Other threads are currently calling into gRPC|"
    r"All log messages before absl::InitializeLog).*$\n?",
    re.M,
)


def _strip_runtime_noise(text: str) -> str:
    return _RUNTIME_NOISE.sub("", text or "").strip()

# ---------------------------------------------------------------------------
# LIMITS — run time raised (was 10s default / 30s max) so real work like image
# processing, package installs and data crunching finishes. A call can still
# ask for less or more via the `timeout_s` arg, up to MAX_TIMEOUT_S.
# ---------------------------------------------------------------------------
DEFAULT_TIMEOUT_S = 60
MAX_TIMEOUT_S = 180
MAX_CODE_CHARS = 20_000
MAX_OUTPUT_CHARS = 8_000          # per stream (stdout/stderr), truncated past this
MAX_MEMORY_BYTES = 1024 * 1024 * 1024  # 1 GB address space, subprocess mode only (256 MB was too small for numpy + video frames)
MAX_PROCS = 128                         # fork-bomb guard, subprocess mode only. Counts THREADS too, and is shared by every
                                        # process running as 'nobody' — 32 caused "fork: Resource temporarily unavailable".

# ---------------------------------------------------------------------------
# OUTPUT FILES - anything the code writes into ./outputs/ is copied into the
# user's workspace (outputs/<name>) after the run. It is NOT delivered
# automatically: the model calls output_files to choose what the user gets.
# ---------------------------------------------------------------------------
OUTPUT_SUBDIR = "outputs"
MAX_OUTPUT_FILES = 20
MAX_OUTPUT_FILE_BYTES = 10 * 1024 * 1024   # matches the RLIMIT_FSIZE write cap
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]")

# ---------------------------------------------------------------------------
# INPUT FILES - run_code used to run in an EMPTY folder, so code could not
# read anything the user already has. Now the model can pass input_files=[...]
# (workspace paths/folders OR saved-doc ids) and they are COPIED into
# ./inputs/ before the code starts. Copy-in only: the real files are never
# touched by the run. Anything the code wants to keep goes in ./outputs/.
# ---------------------------------------------------------------------------
INPUT_SUBDIR = "inputs"
MAX_INPUT_REFS = 20
MAX_INPUT_FILES = 60
MAX_INPUT_TOTAL_BYTES = 25 * 1024 * 1024
_INPUT_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".idea", ".dart_tool", "build", "dist"}

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
PACKAGE_INSTALL_TIMEOUT_S = 240   # allow slower binary wheels on small hosted runners
MAX_PACKAGES_PER_RUN = 6
_PACKAGE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")   # blocks flags/paths/shell metacharacters


def _installed_package_names() -> set:
    # Only distribution metadata proves that pip completed an installation.
    # Do NOT count import-package folders (e.g. `numpy/`) as installed:
    # an interrupted pip run can leave that folder behind without its compiled
    # extension, which previously caused NumPy to be skipped forever.
    names = set()
    if os.path.isdir(PACKAGE_INSTALL_DIR):
        for entry in os.listdir(PACKAGE_INSTALL_DIR):
            if not (entry.endswith(".dist-info") or entry.endswith(".egg-info")):
                continue
            metadata_name = os.path.join(PACKAGE_INSTALL_DIR, entry, "METADATA")
            if not os.path.isfile(metadata_name):
                metadata_name = os.path.join(PACKAGE_INSTALL_DIR, entry, "PKG-INFO")
            try:
                with open(metadata_name, "r", encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        if line.lower().startswith("name:"):
                            names.add(line.split(":", 1)[1].strip().lower().replace("_", "-"))
                            break
            except OSError:
                continue
    return names


def _distribution_imports_healthy(package: str) -> bool:
    """Return False when an installed distribution's import is broken."""
    normalized = package.lower().replace("_", "-").split("[", 1)[0]
    candidates = []
    try:
        for entry in os.listdir(PACKAGE_INSTALL_DIR):
            if not (entry.endswith(".dist-info") or entry.endswith(".egg-info")):
                continue
            metadata = os.path.join(PACKAGE_INSTALL_DIR, entry, "METADATA")
            try:
                with open(metadata, "r", encoding="utf-8", errors="replace") as fh:
                    content = fh.read(20000)
            except OSError:
                continue
            match = re.search(r"(?im)^Name:\s*(.+?)\s*$", content)
            if match and match.group(1).strip().lower().replace("_", "-") == normalized:
                top_level = os.path.join(PACKAGE_INSTALL_DIR, entry, "top_level.txt")
                try:
                    with open(top_level, "r", encoding="utf-8", errors="replace") as fh:
                        candidates = [line.strip() for line in fh if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", line.strip())]
                except OSError:
                    pass
                break
    except OSError:
        return False

    # Some wheels omit top_level.txt. At least force a clean reinstall of
    # known compiled packages when their files are incomplete.
    if not candidates:
        candidates = {"numpy": ["numpy"], "pillow": ["PIL"], "opencv-python": ["cv2"],
                      "scikit-learn": ["sklearn"], "beautifulsoup4": ["bs4"]}.get(normalized, [])
    if not candidates:
        return True

    probe = "import importlib; " + "; ".join(f"importlib.import_module({name!r})" for name in candidates)
    try:
        result = subprocess.run(
            [sys.executable, "-S", "-c", probe],
            env=_clean_env(extra_pythonpath=PACKAGE_INSTALL_DIR),
            capture_output=True, text=True, timeout=15,
        )
        return result.returncode == 0
    except Exception:
        return False


def _remove_broken_distribution(package: str) -> None:
    """Remove stale files for one broken distribution before reinstalling it."""
    normalized = package.lower().replace("_", "-").split("[", 1)[0]
    if not os.path.isdir(PACKAGE_INSTALL_DIR):
        return
    remove_paths = set()
    for entry in os.listdir(PACKAGE_INSTALL_DIR):
        if entry.lower().replace("_", "-").startswith(normalized.replace("-", "_") + "-") or entry.lower().replace("_", "-").startswith(normalized + "-"):
            if entry.endswith(".dist-info") or entry.endswith(".egg-info"):
                remove_paths.add(os.path.join(PACKAGE_INSTALL_DIR, entry))
    # Use known import-folder names for common compiled packages and recover
    # other packages from their top_level.txt when available.
    aliases = {"numpy": ["numpy"], "pillow": ["PIL"], "opencv-python": ["cv2"],
               "scikit-learn": ["sklearn"], "beautifulsoup4": ["bs4"]}
    import_names = aliases.get(normalized, [])
    for entry in os.listdir(PACKAGE_INSTALL_DIR):
        if entry.endswith(".dist-info") or entry.endswith(".egg-info"):
            metadata = os.path.join(PACKAGE_INSTALL_DIR, entry, "METADATA")
            try:
                with open(metadata, "r", encoding="utf-8", errors="replace") as fh:
                    content = fh.read(20000)
            except OSError:
                continue
            match = re.search(r"(?im)^Name:\s*(.+?)\s*$", content)
            if match and match.group(1).strip().lower().replace("_", "-") == normalized:
                top_level = os.path.join(PACKAGE_INSTALL_DIR, entry, "top_level.txt")
                try:
                    with open(top_level, "r", encoding="utf-8", errors="replace") as fh:
                        import_names = [line.strip() for line in fh if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", line.strip())]
                except OSError:
                    pass
                break
    for name in import_names:
        for path in (os.path.join(PACKAGE_INSTALL_DIR, name), os.path.join(PACKAGE_INSTALL_DIR, name + ".py")):
            if os.path.isdir(path) and not os.path.islink(path):
                remove_paths.add(path)
            elif os.path.isfile(path):
                remove_paths.add(path)
    for path in remove_paths:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                os.remove(path)
            except OSError:
                pass


_INSTALL_THREAD_LOCK = threading.Lock()
_INTERPRETER_STAMP = os.path.join(PACKAGE_INSTALL_DIR, ".python-stamp")


def _installed_in_host_python(package: str) -> bool:
    """True when the interpreter we run user code with (sys.executable, WITHOUT -S)
    already has this distribution — e.g. numpy/pillow/pandas baked into the image.
    Those never need a pip --target copy, and a second copy is exactly what broke."""
    name = package.split("[", 1)[0]
    try:
        importlib.metadata.version(name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False
    except Exception:
        return False


def _reset_dir_if_wrong_python() -> None:
    """Compiled wheels (numpy, pillow...) only load on the Python they were built for.
    If PACKAGE_INSTALL_DIR was filled by a different interpreter (3.11 vs 3.13, as in
    the `_multiarray_umath` / `_imaging` errors), wipe it so it reinstalls cleanly."""
    stamp = f"{sys.version_info.major}.{sys.version_info.minor} {sys.executable}"
    try:
        with open(_INTERPRETER_STAMP, "r", encoding="utf-8") as fh:
            if fh.read().strip() == stamp:
                return
    except OSError:
        pass
    if os.path.isdir(PACKAGE_INSTALL_DIR):
        for entry in os.listdir(PACKAGE_INSTALL_DIR):
            path = os.path.join(PACKAGE_INSTALL_DIR, entry)
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                try:
                    os.remove(path)
                except OSError:
                    pass
    os.makedirs(PACKAGE_INSTALL_DIR, exist_ok=True)
    try:
        with open(_INTERPRETER_STAMP, "w", encoding="utf-8") as fh:
            fh.write(stamp)
    except OSError:
        pass


def _ensure_packages(packages: list, timeout_s: int = PACKAGE_INSTALL_TIMEOUT_S):
    """
    Installs any of `packages` not already available. Returns (ok, message).
    Serialised with a thread lock + file lock so two runs can never pip into the
    same folder at once (that left half-written numpy/PIL installs behind).
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
    lock_path = PACKAGE_INSTALL_DIR.rstrip("/\\") + ".lock"
    with _INSTALL_THREAD_LOCK:
        lock_fh = None
        try:
            if fcntl is not None:
                lock_fh = open(lock_path, "w")
                fcntl.flock(lock_fh, fcntl.LOCK_EX)
            # ALWAYS validate the shared dir first: it goes on PYTHONPATH ahead of the image's
            # own site-packages, so a stale dir built by another Python would shadow a good numpy.
            _reset_dir_if_wrong_python()
            # Anything the image already ships (numpy, pillow, pandas, imageio, ...) needs no install.
            packages = [p for p in packages if not _installed_in_host_python(p)]
            if not packages:
                return True, ""
            return _ensure_packages_locked(packages, timeout_s)
        finally:
            if lock_fh is not None:
                try:
                    fcntl.flock(lock_fh, fcntl.LOCK_UN)
                finally:
                    lock_fh.close()


def _ensure_packages_locked(packages: list, timeout_s: int):
    already_there = _installed_package_names()
    to_install = []
    for package in packages:
        normalized = package.lower().replace("_", "-")
        if normalized not in already_there:
            # Clear abandoned package folders left by an interrupted install.
            _remove_broken_distribution(package)
            # Also remove an orphan import folder when no distribution metadata
            # exists (the exact failure mode from the NumPy traceback).
            if normalized == "numpy":
                stale_numpy = os.path.join(PACKAGE_INSTALL_DIR, "numpy")
                if os.path.isdir(stale_numpy):
                    shutil.rmtree(stale_numpy, ignore_errors=True)
            to_install.append(package)
        elif not _distribution_imports_healthy(package):
            _remove_broken_distribution(package)
            to_install.append(package)
    if not to_install:
        return True, ""

    cmd = [
        sys.executable, "-m", "pip", "install",
        "--upgrade", "--target", PACKAGE_INSTALL_DIR,
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
        # sys.executable (never bare "python3") so pip, the health probe and user code all use
        # the SAME interpreter — mixing 3.11 and 3.13 is what broke numpy/PIL's compiled parts.
        # -S was removed: it hid the numpy/pillow/pandas/imageio already installed in the image,
        # forcing a fragile second copy into /tmp/sandbox_site_packages. -I still ignores
        # PYTHONPATH/user site, so a no-packages run sees only the image's own packages.
        "subprocess_cmd": lambda path: [sys.executable, "-I", path],
        # Packages requested: drop -I so PYTHONPATH (PACKAGE_INSTALL_DIR, set in _clean_env)
        # is honoured. Extras are found first, everything else falls back to the image.
        "subprocess_cmd_with_packages": lambda path: [sys.executable, path],
        # Must match the host Python minor version: the mounted pip --target dir holds
        # wheels built for it (a 3.13 numpy will not load in a 3.12 container).
        "docker_image": f"python:{sys.version_info.major}.{sys.version_info.minor}-slim",
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


_FFMPEG_CACHE: dict = {}


def _ffmpeg_exe() -> Optional[str]:
    """Absolute path of a usable ffmpeg, or None. Found HERE (in the server process) and handed to
    sandboxed code via IMAGEIO_FFMPEG_EXE, so imageio / imageio-ffmpeg never have to discover it
    themselves inside the scrubbed environment (that lookup raised "No ffmpeg exe could be found"
    even though the bundled binary and /usr/bin/ffmpeg both existed). Prefers the binary bundled
    with imageio-ffmpeg, falls back to a system ffmpeg."""
    if "exe" in _FFMPEG_CACHE:
        return _FFMPEG_CACHE["exe"]
    exe = None
    try:
        import glob
        import importlib.util
        spec = importlib.util.find_spec("imageio_ffmpeg")
        if spec and spec.origin:
            pattern = os.path.join(os.path.dirname(spec.origin), "binaries", "ffmpeg-*")
            for cand in sorted(glob.glob(pattern)):
                if os.path.isfile(cand) and os.access(cand, os.X_OK):
                    exe = cand
                    break
    except Exception:
        exe = None
    if not exe:
        exe = shutil.which("ffmpeg")
    _FFMPEG_CACHE["exe"] = exe
    return exe


def _fonts_dir() -> Optional[str]:
    """Folder of bundled .ttf fonts (see fonts/FONTS.txt). FONTS_DIR env wins (the Dockerfile sets it);
    otherwise the fonts/ folder next to this file. None when neither exists."""
    for cand in (os.environ.get("FONTS_DIR"), os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")):
        if cand and os.path.isdir(cand):
            return cand
    return None


def _clean_env(extra_pythonpath: Optional[str] = None) -> dict:
    # Deliberately NOT os.environ.copy() — this is the whole point. A
    # subprocess that inherited the real environment gets every API key/
    # DB credential/secret this backend process holds, for free. Hand it
    # a minimal, boring environment instead.
    env = {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME": "/tmp",
        "LANG": "C.UTF-8",
        # BLAS/OpenMP otherwise start one thread per CPU core; thread count is charged against
        # RLIMIT_NPROC and virtual memory (RLIMIT_AS) and caused numpy import + fork failures.
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "MPLBACKEND": "Agg",
    }
    ffmpeg = _ffmpeg_exe()
    if ffmpeg:
        env["IMAGEIO_FFMPEG_EXE"] = ffmpeg      # a path, not a secret
    if os.environ.get("U2NET_HOME"):
        env["U2NET_HOME"] = os.environ["U2NET_HOME"]      # a path to the pre-downloaded background-removal model, not a secret
    fonts = _fonts_dir()
    if fonts:
        env["FONTS_DIR"] = fonts                          # a path to the bundled fonts, not a secret
    if extra_pythonpath:
        # Only ever set to PACKAGE_INSTALL_DIR (our own pip --target dir),
        # never anything caller-supplied — this is not a general env
        # passthrough, just how the installed packages become importable.
        env["PYTHONPATH"] = extra_pythonpath
    return env


def _uid_thread_count(uid: int) -> int:
    """Processes + threads currently owned by `uid` — exactly what RLIMIT_NPROC counts."""
    total = 0
    try:
        entries = os.listdir("/proc")
    except OSError:
        return 0
    for pid in entries:
        if not pid.isdigit():
            continue
        try:
            real_uid, threads = None, 1
            with open(f"/proc/{pid}/status", "r") as fh:
                for line in fh:
                    if line.startswith("Uid:"):
                        real_uid = int(line.split()[1])
                    elif line.startswith("Threads:"):
                        threads = int(line.split()[1])
            if real_uid == uid:
                total += threads
        except (OSError, ValueError, IndexError):
            continue
    return total


def _nproc_limit(headroom: int) -> Optional[int]:
    """RLIMIT_NPROC is OFF by default (returns None = leave the inherited limit alone).

    Why: it is a per-USER cap counted by the kernel across every process of that uid on the
    machine, including ones this container cannot see in /proc (other containers on a shared
    host, the web server itself). On Render the sandbox child is the same uid as the server, so
    ANY fixed or usage-based number fails every fork with EAGAIN ('Resource temporarily
    unavailable') — even `true` could not start — while only ~29 threads were visible.
    Fork-bomb protection comes from the container's own pids cgroup limit (and the CPU/time
    caps) instead. To force a cap anyway, set env SANDBOX_NPROC_HEADROOM=<n>: the child may
    then run <n> more processes/threads than this user's visible total."""
    raw = os.environ.get("SANDBOX_NPROC_HEADROOM", "").strip()
    if not raw.isdigit() or not hasattr(os, "geteuid"):
        return None
    target = 65534 if os.geteuid() == 0 else os.geteuid()
    return _uid_thread_count(target) + int(raw)


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

    nproc = _nproc_limit(MAX_PROCS)      # computed in the parent, applied in the child

    def _apply():
        resource.setrlimit(resource.RLIMIT_CPU, (MAX_TIMEOUT_S, MAX_TIMEOUT_S))
        resource.setrlimit(resource.RLIMIT_AS, (MAX_MEMORY_BYTES, MAX_MEMORY_BYTES))
        if nproc is not None:            # opt-in only, see _nproc_limit()
            _, hard = resource.getrlimit(resource.RLIMIT_NPROC)
            n = nproc if hard == resource.RLIM_INFINITY else min(nproc, hard)
            resource.setrlimit(resource.RLIMIT_NPROC, (n, n))
        resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))  # 10MB file writes max
        os.setsid()  # own process group, so a timeout kill takes any children with it
        # When the server runs as root (Render's Docker does) the sandboxed code
        # must NOT: drop to 'nobody' so it can't read workspaces (mode 0700),
        # app files that aren't world-readable, or other users' data.
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            os.setgroups([])
            os.setgid(65534)
            os.setuid(65534)

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
        "-w", "/code",                       # relative paths (./inputs, ./outputs) resolve inside the mount
        "-v", f"{os.path.dirname(code_path)}:/code:ro",
        # Writable outputs folder nested inside the read-only /code mount,
        # so docker-mode runs can also produce files.
        "-v", f"{os.path.join(os.path.dirname(code_path), OUTPUT_SUBDIR)}:/code/{OUTPUT_SUBDIR}:rw",
    ]
    if packages_dir and os.path.isdir(packages_dir):
        # Same shared install dir subprocess mode uses — mounted read-only
        # so a container can import what was pip-installed on the host
        # without re-downloading anything itself.
        cmd += ["-v", f"{packages_dir}:/packages:ro", "-e", "PYTHONPATH=/packages"]
    fonts = _fonts_dir()
    if fonts:
        cmd += ["-v", f"{fonts}:/fonts:ro", "-e", "FONTS_DIR=/fonts"]     # same bundled fonts as subprocess mode
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


def _collect_outputs_into_workspace(run_dir: str, userid: Optional[str], unchanged: Optional[dict] = None) -> list:
    """
    Copies every regular file the code left in <run_dir>/outputs/ into the
    user's WORKSPACE (under outputs/) and returns [{"path","size"}]. It does
    NOT upload or show anything to the user — the model decides what gets
    delivered by calling output_files with these paths. Never raises.
    """
    from gpt2_workspace import workspace_write_bytes  # lazy: no import cycle

    out_dir = os.path.join(run_dir, OUTPUT_SUBDIR)
    if not os.path.isdir(out_dir):
        return []

    results = []
    for name in sorted(os.listdir(out_dir))[:MAX_OUTPUT_FILES]:
        path = os.path.join(out_dir, name)
        if os.path.islink(path) or not os.path.isfile(path):
            continue   # skip symlinks (could point at host files) and subfolders
        size = os.path.getsize(path)
        if size == 0 or size > MAX_OUTPUT_FILE_BYTES:
            print(f"[SANDBOX] skipping output {name!r}: size {size} out of range")
            continue
        safe_name = _SAFE_FILENAME_RE.sub("_", os.path.basename(name))[:80]
        try:
            with open(path, "rb") as f:
                data = f.read()
            if unchanged:
                import hashlib
                if (unchanged.get(f"{OUTPUT_SUBDIR}/{name}") == hashlib.sha256(data).hexdigest()
                        and int(os.path.getmtime(path)) == 1_000_000_000):
                    continue   # a copy of an existing workspace file the code never changed
            saved = workspace_write_bytes(userid, f"{OUTPUT_SUBDIR}/{safe_name}", data)
            if saved.get("success"):
                results.append({"path": saved["path"], "size": size})
        except Exception as e:
            print(f"[SANDBOX] couldn't save output {name!r}: {e}")
    return results


def _stage_inputs(run_dir: str, userid: Optional[str], refs: list):
    """
    Copies each ref into <run_dir>/inputs/ and returns (staged, missing).
    A ref is a workspace file path, a workspace folder, or a saved-doc id
    (looked up in that order). staged = [{"path","source","size"}] where
    path is what the code should open (e.g. "inputs/data/sales.csv").
    Never raises: problems are reported in `missing` with a reason.
    """
    from gpt2_workspace import _safe_path  # lazy: no import cycle

    in_dir = os.path.join(run_dir, INPUT_SUBDIR)
    os.makedirs(in_dir, exist_ok=True)
    in_real = os.path.realpath(in_dir)
    staged, missing = [], []
    state = {"files": 0, "bytes": 0}

    def _room(size: int) -> bool:
        return state["files"] < MAX_INPUT_FILES and state["bytes"] + size <= MAX_INPUT_TOTAL_BYTES

    def _dest_for(rel: str) -> Optional[str]:
        rel = rel.replace("\\", "/").lstrip("/")
        dest = os.path.realpath(os.path.join(in_real, rel))
        return dest if dest.startswith(in_real + os.sep) else None

    def _copy_file(src: str, rel: str, source: str, ref: str) -> None:
        size = os.path.getsize(src)
        dest = _dest_for(rel)
        if dest is None:
            missing.append(f"{ref} (unsafe path)")
            return
        if not _room(size):
            missing.append(f"{rel} (over the input size/count limit)")
            return
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copyfile(src, dest)
        state["files"] += 1
        state["bytes"] += size
        staged.append({"path": f"{INPUT_SUBDIR}/{rel.replace(os.sep, '/')}", "source": source, "size": size})

    for raw in (refs or [])[:MAX_INPUT_REFS]:
        ref = str(raw or "").strip()
        if not ref:
            continue
        try:
            full = _safe_path(userid, ref)
        except Exception:
            missing.append(f"{ref} (outside the workspace)")
            continue
        try:
            if os.path.isfile(full):
                _copy_file(full, ref.replace("\\", "/").lstrip("/"), "workspace", ref)
            elif os.path.isdir(full):
                base = ref.replace("\\", "/").strip("/")
                found = 0
                for dp, dn, fn in os.walk(full):
                    dn[:] = [d for d in dn if d not in _INPUT_SKIP_DIRS and not os.path.islink(os.path.join(dp, d))]
                    for name in sorted(fn):
                        fp = os.path.join(dp, name)
                        if os.path.islink(fp):
                            continue
                        sub = os.path.relpath(fp, full)
                        _copy_file(fp, f"{base}/{sub}" if base else sub, "workspace", ref)
                        found += 1
                if not found:
                    missing.append(f"{ref} (folder is empty)")
            else:
                missing.append(f"{ref} (not found in the workspace — use workspace_list)")
                continue
                staged.append({"path": f"{INPUT_SUBDIR}/{name}", "source": "saved_doc", "size": len(data)})
        except Exception as e:
            missing.append(f"{ref} ({str(e)[:80]})")

    # readable by the sandbox user ('nobody' when the server runs as root)
    try:
        is_root = hasattr(os, "geteuid") and os.geteuid() == 0
        for dp, dn, fn in os.walk(in_dir):
            os.chmod(dp, 0o755)
            if is_root:
                os.chown(dp, 65534, 65534)
            for name in fn:
                fp = os.path.join(dp, name)
                os.chmod(fp, 0o644)
                if is_root:
                    os.chown(fp, 65534, 65534)
    except Exception as e:
        print(f"[SANDBOX] couldn't set input permissions: {e}")
    return staged, missing


_PATH_LITERAL_RE = re.compile(r"""(['"])([^'"\n\\]{1,200}\.[A-Za-z0-9]{1,8})\1""")


_DISCOVERY_RE = re.compile(r"os\.walk|os\.listdir|os\.scandir|glob\.|\.glob\(|\.rglob\(|\.iterdir\(|listdir\(", re.I)
_DATA_EXTS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".csv", ".tsv", ".json", ".txt", ".md", ".xlsx", ".xls",
              ".docx", ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".mp4", ".zip", ".html"}


def _stage_workspace_data(run_dir: str, userid: str, placed: dict) -> None:
    """Code that searches folders (os.walk('.'), glob, listdir) finds nothing in an empty run folder.
    Copy the user's data files (outputs/ first, then other media/documents; never source code or
    repos' dot-folders) into the run folder at their workspace-relative paths, within the size limits."""
    import hashlib
    try:
        from gpt2_workspace import _safe_path
        root = _safe_path(userid, "")
    except Exception:
        return
    if not os.path.isdir(root):
        return
    run_real = os.path.realpath(run_dir)
    found = []
    for dp, dn, fn in os.walk(root):
        dn[:] = sorted(d for d in dn if d not in _INPUT_SKIP_DIRS and not d.startswith("."))
        for name in sorted(fn):
            full = os.path.join(dp, name)
            if os.path.islink(full) or os.path.splitext(name)[1].lower() not in _DATA_EXTS:
                continue
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            found.append((0 if rel.startswith(OUTPUT_SUBDIR + "/") else 1, rel, full))
    found.sort()
    count, total = 0, 0
    for _, rel, full in found:
        try:
            size = os.path.getsize(full)
            if count >= MAX_INPUT_FILES or total + size > MAX_INPUT_TOTAL_BYTES:
                continue
            dest = os.path.realpath(os.path.join(run_real, rel))
            if not dest.startswith(run_real + os.sep):
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copyfile(full, dest)
            with open(dest, "rb") as f:
                placed[rel] = hashlib.sha256(f.read()).hexdigest()
            os.utime(dest, (1_000_000_000, 1_000_000_000))
            count += 1
            total += size
        except Exception as e:
            print(f"[SANDBOX] discovery-stage of {rel!r} failed: {e}")


def _auto_stage_referenced(run_dir: str, userid: Optional[str], code: str) -> dict:
    """
    Models often open a workspace file by name in run_code and forget
    input_files, so the code runs in an empty folder and fails with
    FileNotFoundError. This looks at the quoted file names in the code, finds
    the matching WORKSPACE files, and copies each one to exactly the path the
    code uses (so open("outputs/a.wav") or open("a.wav") both just work).
    Copy-in only. Returns {relpath_in_run_dir: sha256} for what it placed, so
    unchanged copies are not saved back as "new" outputs. Never raises.
    """
    import hashlib
    placed = {}
    if not userid:
        return placed
    if _DISCOVERY_RE.search(code or ""):
        _stage_workspace_data(run_dir, userid, placed)
    try:
        from gpt2_workspace import _safe_path  # lazy: no import cycle
    except Exception:
        return placed
    run_real = os.path.realpath(run_dir)
    total = 0
    seen = set()
    for m in _PATH_LITERAL_RE.finditer(code or ""):
        lit = m.group(2).replace("\\", "/").strip()
        rel = lit
        while rel.startswith("./"):
            rel = rel[2:]
        rel = rel.lstrip("/")
        if not rel or ".." in rel.split("/") or rel in seen or rel == "main.py":
            continue
        seen.add(rel)
        if len(placed) >= MAX_INPUT_FILES:
            break
        src = None
        candidates = [rel[len(INPUT_SUBDIR) + 1:]] if rel.startswith(INPUT_SUBDIR + "/") else []
        candidates.append(rel)
        for cand in candidates:
            try:
                full = _safe_path(userid, cand)
            except Exception:
                continue
            if os.path.isfile(full) and not os.path.islink(full):
                src = full
                break
        if src is None and "/" not in rel:
            # bare file name: look for it in the workspace (outputs/ first, then anywhere)
            try:
                root = _safe_path(userid, "")
            except Exception:
                root = None
            if root and os.path.isdir(root):
                hit = None
                for dp, dn, fn in os.walk(root):
                    dn[:] = [d for d in dn if d not in _INPUT_SKIP_DIRS]
                    if rel in fn:
                        cand = os.path.join(dp, rel)
                        if not os.path.islink(cand):
                            if os.path.basename(dp) == OUTPUT_SUBDIR:
                                hit = cand
                                break
                            hit = hit or cand
                src = hit
        if src is None:
            continue
        try:
            size = os.path.getsize(src)
            if total + size > MAX_INPUT_TOTAL_BYTES:
                continue
            dest = os.path.realpath(os.path.join(run_real, rel))
            if not dest.startswith(run_real + os.sep):
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copyfile(src, dest)
            with open(dest, "rb") as f:
                placed[rel] = hashlib.sha256(f.read()).hexdigest()
            os.utime(dest, (1_000_000_000, 1_000_000_000))   # marks it as "not written by the code"; a rewrite gets a fresh time
            total += size
        except Exception as e:
            print(f"[SANDBOX] auto-stage of {rel!r} failed: {e}")
    if placed:
        print(f"[SANDBOX] auto-staged workspace files for the run: {sorted(placed)}")
    return placed


def _chown_tree(root: str):
    try:
        if not (hasattr(os, "geteuid") and os.geteuid() == 0):
            return
        for dp, dn, fn in os.walk(root):
            os.chown(dp, 65534, 65534)
            for name in fn:
                os.chown(os.path.join(dp, name), 65534, 65534)
    except Exception as e:
        print(f"[SANDBOX] chown failed: {e}")


def _sweep_stray_files(run_dir: str, placed: Optional[dict] = None):
    """Code that saves a new file next to main.py (instead of ./outputs/) would lose it when the
    run folder is deleted. Move such new files into ./outputs/ so they are kept like any other output."""
    try:
        out_dir = os.path.join(run_dir, OUTPUT_SUBDIR)
        for name in os.listdir(run_dir):
            path = os.path.join(run_dir, name)
            if (name.startswith("main.") or os.path.islink(path) or not os.path.isfile(path)
                    or os.path.exists(os.path.join(out_dir, name))):
                continue
            if name in (placed or {}) and int(os.path.getmtime(path)) == 1_000_000_000:
                continue   # an input copy the code only read
            shutil.move(path, os.path.join(out_dir, name))
    except Exception as e:
        print(f"[SANDBOX] stray-file sweep failed: {e}")


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
             packages: Optional[list] = None, input_files: Optional[list] = None):
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

    input_files: optional list of files your code needs to READ — workspace
    paths ("data/sales.csv"), workspace folders ("my_project"), or saved-doc
    ids ("notes.md"). They are copied (read-only in effect, the originals are
    never changed) into ./inputs/ before your code starts, so open them as
    "inputs/data/sales.csv" / "inputs/notes.md". The result lists what was
    staged under "inputs" and anything not found under "inputs_missing".
    To keep files your code makes, write them to ./outputs/ (saved into the
    workspace under outputs/). For a whole project with its own tests use
    workspace_run instead.

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

    if packages and any(re.sub(r"[^a-z]", "", str(x).lower()).startswith(("rembg", "onnxruntime", "torch", "tensorflow")) for x in packages):
        yield {
            "type": "code_result", "success": False, "engine": None,
            "language": lang, "stdout": "", "stderr": "", "exit_code": None,
            "error": ("That package is too heavy to install inside run_code (it would time out). "
                      "For removing an image background use the remove_background tool instead."),
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

        staged_inputs, missing_inputs = [], []
        if input_files:
            yield {"type": "status", "text": "Loading your files into the sandbox...",
                   "detail": f"{len(input_files)} file(s)/folder(s)", "icon": "sandbox"}
            staged_inputs, missing_inputs = _stage_inputs(run_dir, userid, list(input_files))
            if not staged_inputs:
                yield {
                    "type": "code_result", "success": False, "engine": None,
                    "language": lang, "stdout": "", "stderr": "", "exit_code": None,
                    "inputs": [], "inputs_missing": missing_inputs,
                    "error": "None of the input_files could be loaded, so the code was NOT run: "
                             + "; ".join(missing_inputs[:5]),
                }
                return

        # Files the code names but the model forgot to pass in input_files.
        auto_placed = _auto_stage_referenced(run_dir, userid, code)

        # Pre-create the outputs folder (world-writable so the docker
        # 'nobody' user can write to it too).
        out_dir = os.path.join(run_dir, OUTPUT_SUBDIR)
        os.makedirs(out_dir, exist_ok=True)
        os.chmod(out_dir, 0o777)
        if auto_placed:
            _chown_tree(run_dir)
            for dp, dn, fn in os.walk(run_dir):
                os.chmod(dp, 0o777 if os.path.basename(dp) == OUTPUT_SUBDIR else 0o755)
                for n in fn:
                    os.chmod(os.path.join(dp, n), 0o644)
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            # the child runs as nobody (see _resource_limits) — it must own
            # its run folder to read main.py and write ./outputs/
            os.chmod(run_dir, 0o755)
            os.chown(run_dir, 65534, 65534)
            os.chown(code_path, 65534, 65534)

        chain = _build_sandbox_chain(prefer_isolation)
        for link in chain:
            yield {"type": "status", "text": f"Executing via {link['name']}...", "detail": None, "icon": "sandbox"}
            outcome = link["call"](runner, code_path, timeout_s, packages_dir)
            if outcome.get("ok"):
                print(f"[SANDBOX] ran {lang} via {link['name']} — exit={outcome.get('exit_code')}, "
                      f"timed_out={outcome.get('timed_out')}, userid={userid!r}")
                raw_err = outcome.get("stderr", "") or ""
                clean_err = _strip_runtime_noise(raw_err)
                noise_only = bool(raw_err.strip()) and not clean_err
                ran_ok = (outcome.get("exit_code") == 0) or (noise_only and outcome.get("exit_code") is not None)
                ran_ok = ran_ok and not outcome.get("timed_out")
                output_files = []
                if ran_ok:
                    _sweep_stray_files(run_dir, auto_placed)
                    output_files = _collect_outputs_into_workspace(run_dir, userid, auto_placed)
                yield {
                    "type": "code_result",
                    "success": ran_ok,
                    "engine": link["name"],
                    "language": lang,
                    "stdout": outcome.get("stdout", ""),
                    "stderr": clean_err,
                    "exit_code": 0 if (noise_only and ran_ok) else outcome.get("exit_code"),
                    "timed_out": outcome.get("timed_out", False),
                    "duration_s": outcome.get("duration_s"),
                    "inputs": staged_inputs,               # files copied into ./inputs/ for this run
                    "inputs_missing": missing_inputs,      # refs that could not be loaded (with reason)
                    "files": [],   # nothing is auto-delivered any more — see saved_to_workspace + output_files
                    "saved_to_workspace": output_files,   # [{"path","size"}] files the code wrote to ./outputs/
                    "next_step": ("These files are saved in your workspace but NOT yet shown to the user. "
                                  "Call output_files with the paths you want them to get.") if output_files else None,
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
