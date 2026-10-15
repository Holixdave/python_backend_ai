#!/usr/bin/env python3
# gpt2_workspace_run.py — workspace_run: run a shell command INSIDE the user's
# workspace project (run its tests, scripts, builds) — the "code can finally
# see the files" tool that run_code (one script, empty folder) can't be.
# ─────────────────────────────────────────────────────────────────────────────
# HOW IT STAYS SAFE (no Docker needed)
#   1. COPY-IN / COPY-OUT: the command runs in a temporary COPY of the chosen
#      workspace folder, never in the real workspace. Only that copy is visible
#      to the command — other users' workspaces and the app's folders aren't
#      in its working directory.
#   2. DROPS TO 'nobody' when the server runs as root (Render's Docker does):
#      the process can't read WORKSPACE_ROOT (mode 0700), other users' files,
#      or anything not world-readable. Env vars/secrets are scrubbed.
#   3. NO NETWORK when the host allows it (unshare -n). If it doesn't (common
#      on managed hosts), the result says network_isolated=false honestly.
#   4. CPU / memory / process / file-size limits and a hard timeout that kills
#      the whole process group.
#   5. Changes are synced back only as regular files (symlinks created by the
#      command are ignored), within the size/count caps, never outside the
#      folder it was started in.
#
# Python deps: pass packages=["pytest","requests"]; they're installed ONCE into
# the shared sandbox package dir (same one run_code uses) before the command.
# ─────────────────────────────────────────────────────────────────────────────

import os
import shutil
import subprocess
import tempfile
import time
from typing import Optional

from gpt2_workspace import (
    MAX_FILE_BYTES, MAX_USER_BYTES, _dir_size, _dirty, _rate_ok, _safe_path, _user_root,
)
from gpt2_sandbox import PACKAGE_INSTALL_DIR, _ensure_packages, _truncate

DEFAULT_TIMEOUT_S = 60
MAX_TIMEOUT_S = 180
MAX_COPY_BYTES = 100 * 1024 * 1024
MAX_SYNC_FILES = 500
MAX_STREAM_CHARS = 12_000
MEMORY_BYTES = 768 * 1024 * 1024
NOBODY = 65534
_COPY_SKIP = {".git", "__pycache__"}
_unshare_ok: Optional[bool] = None


def _can_unshare() -> bool:
    global _unshare_ok
    if _unshare_ok is None:
        try:
            _unshare_ok = subprocess.run(["unshare", "-n", "true"], capture_output=True, timeout=3).returncode == 0
        except Exception:
            _unshare_ok = False
    return _unshare_ok


def _snapshot(base: str) -> dict:
    snap = {}
    for dp, dn, fn in os.walk(base):
        dn[:] = [d for d in dn if d not in _COPY_SKIP and not os.path.islink(os.path.join(dp, d))]
        for f in fn:
            fp = os.path.join(dp, f)
            if os.path.islink(fp):
                continue
            st = os.stat(fp)
            snap[os.path.relpath(fp, base)] = (st.st_size, st.st_mtime_ns)
    return snap


def _chown_tree(path: str) -> None:
    os.chown(path, NOBODY, NOBODY)
    for dp, dn, fn in os.walk(path):
        for n in dn + fn:
            try:
                os.lchown(os.path.join(dp, n), NOBODY, NOBODY)
            except OSError:
                pass


def workspace_run(userid: str, command: str, cwd: str = "", timeout_s: int = DEFAULT_TIMEOUT_S,
                  packages: Optional[list] = None) -> dict:
    """
    Run a shell command inside a workspace folder and get real stdout/stderr/
    exit_code — e.g. "pytest -q", "python app.py", "node build.js", "ls".
    The command runs on a COPY of `cwd` (default: whole workspace); any files it
    creates or changes are copied back, and listed as changed_files. Default
    timeout 60s, max 180s. packages = pip packages to install first (python).
    Files it creates are NOT shown to the user until you call output_files.
    """
    if not command or not command.strip():
        return {"success": False, "error": "empty command"}
    if len(command) > 5000:
        return {"success": False, "error": "command too long (max 5000 chars)"}
    if not _rate_ok(userid, "run", 40):
        return {"success": False, "error": "Run limit reached (40 per hour) — try again later."}

    src = _safe_path(userid, cwd)
    if not os.path.isdir(src):
        return {"success": False, "error": f"'{cwd}' isn't a folder in the workspace."}
    if _dir_size(src) > MAX_COPY_BYTES:
        return {"success": False, "error": "That folder is over 100MB — pass a smaller cwd (a sub-project)."}
    timeout_s = max(1, min(int(timeout_s or DEFAULT_TIMEOUT_S), MAX_TIMEOUT_S))

    packages_dir = None
    if packages:
        ok, err = _ensure_packages(list(packages))
        if not ok:
            return {"success": False, "error": f"Couldn't install packages: {err}"}
        packages_dir = PACKAGE_INSTALL_DIR

    is_root = hasattr(os, "geteuid") and os.geteuid() == 0
    isolate_net = _can_unshare()

    run_dir = tempfile.mkdtemp(prefix="wsrun-")
    work = os.path.join(run_dir, "work")
    home = os.path.join(run_dir, "home")
    try:
        shutil.copytree(src, work, symlinks=False, ignore=shutil.ignore_patterns(*_COPY_SKIP), copy_function=shutil.copy2)
        os.makedirs(home, exist_ok=True)
        before = _snapshot(work)
        if is_root:
            os.chmod(run_dir, 0o755)       # nobody must be able to enter it (parent only holds our copy)
            _chown_tree(run_dir)

        env = {
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "HOME": home, "LANG": "C.UTF-8", "TMPDIR": home,
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        if packages_dir:
            env["PYTHONPATH"] = packages_dir

        bash = ["bash", "--noprofile", "--norc", "-c", command]
        if is_root and isolate_net:
            cmd = ["unshare", "-n", "setpriv", f"--reuid={NOBODY}", f"--regid={NOBODY}", "--clear-groups", *bash]
        elif isolate_net:
            cmd = ["unshare", "-n", *bash]
        else:
            cmd = bash

        def _limits():
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (timeout_s + 5, timeout_s + 5))
            resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))
            resource.setrlimit(resource.RLIMIT_NPROC, (128, 128))
            resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_FILE_BYTES, MAX_FILE_BYTES))
            os.setsid()
            if is_root and not isolate_net:   # (with unshare the setpriv wrapper drops privileges instead)
                os.setgroups([])
                os.setgid(NOBODY)
                os.setuid(NOBODY)

        start = time.time()
        timed_out = False
        proc = subprocess.Popen(cmd, cwd=work, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                preexec_fn=_limits, text=True, errors="replace")
        try:
            out, err = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, 9)
            except Exception:
                proc.kill()
            out, err = proc.communicate()
            err = (err or "") + f"\n[killed — exceeded {timeout_s}s timeout]"
        duration = round(time.time() - start, 2)

        # ── sync changes back into the real workspace ──
        after = _snapshot(work)
        src_real = os.path.realpath(src)
        changed, deleted, skipped = [], [], []
        room = MAX_USER_BYTES - _dir_size(_user_root(userid))
        for rel, meta in after.items():
            if before.get(rel) == meta:
                continue
            if len(changed) >= MAX_SYNC_FILES:
                skipped.append(f"{rel} (too many changed files)")
                continue
            if meta[0] > MAX_FILE_BYTES or meta[0] > room:
                skipped.append(f"{rel} (too big / no space)")
                continue
            dest = os.path.realpath(os.path.join(src_real, rel))
            if not dest.startswith(src_real + os.sep):
                skipped.append(f"{rel} (unsafe path)")
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copyfile(os.path.join(work, rel), dest)
            room -= meta[0]
            changed.append(rel)
        for rel in before:
            if rel not in after:
                dest = os.path.realpath(os.path.join(src_real, rel))
                if dest.startswith(src_real + os.sep) and os.path.isfile(dest) and not os.path.islink(dest):
                    os.remove(dest)
                    deleted.append(rel)
        if changed or deleted:
            _dirty(userid)

        prefix = (cwd.strip("/") + "/") if cwd.strip("/") else ""
        res = {
            "success": proc.returncode == 0 and not timed_out,
            "exit_code": proc.returncode, "timed_out": timed_out, "duration_s": duration,
            "stdout": _truncate(out, MAX_STREAM_CHARS), "stderr": _truncate(err, MAX_STREAM_CHARS),
            "changed_files": [prefix + c for c in changed[:50]],
            "isolation": {
                "runs_as": "nobody" if is_root else "server user (server isn't root, so no user switch)",
                "network_isolated": isolate_net,
                "ran_on": "a temporary copy of the folder; changes were synced back",
            },
        }
        if len(changed) > 50:
            res["changed_files_note"] = f"{len(changed)} files changed; first 50 listed"
        if deleted:
            res["deleted_files"] = [prefix + d for d in deleted[:50]]
        if skipped:
            res["not_synced"] = skipped[:20]
        if changed:
            res["next_step"] = "New/changed files are saved in the workspace but NOT shown to the user until you call output_files."
        return res
    except Exception as e:
        return {"success": False, "error": f"workspace_run failed: {e}"}
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
