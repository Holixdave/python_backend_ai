"""Account security that has to live on the SERVER (a reinstall or a modified app can't reset it).

  Find / reset password      POST /auth/find-account  /auth/send-reset  /auth/reset-with-old
  Login attempt lockout      POST /auth/login-precheck  /auth/login-failed  /auth/login-ok
  Banned accounts            POST /auth/ban-status  /auth/appeal  /auth/resolve
  Device limits              POST /auth/device-check  /auth/device-register

Firestore collections written here (Admin SDK only — add `allow read, write: if false;` for them in your rules):
  loginAttempts/{sha256(email)}   {count, lastFailAt, lockedUntil, permanent}
  devices/{deviceId}              {accounts: [{uid, at}], blockedUntil}
  appeals/{banId}                 {uid, text, status, createdAt}
and on users/{uid}: banned, bannedReason, bannedAt, banUntil (absent = permanent), banId, bannedBy, resolvable.

Reset codes: POST /auth/send-reset-code /auth/verify-reset-code  (resetCodes/{uid}, server-only; needs SMTP_PASS etc. like the notifier)

Env: FIREBASE_WEB_API_KEY  (the "Web API key" of the Firebase project; used to send the reset email and to check an old password)
"""
import hashlib
import os
import re
import secrets
import threading
import time
from datetime import datetime, timedelta, timezone

import firebase_config  # noqa: F401
import requests
from fastapi import APIRouter, Header, HTTPException, Request
from firebase_admin import auth as fb_auth, firestore
from pydantic import BaseModel

router = APIRouter(prefix="/auth")

# ── limits ──────────────────────────────────────────────────────────────
# failed logins -> lock. Counts decay after 24 h without a failure.
LOCK_TIERS = [(15, None), (10, 24 * 3600), (7, 3600), (5, 15 * 60)]   # (fails, seconds; None = permanent)
FAIL_DECAY_SECONDS = 24 * 3600
DEVICE_SIGNUPS_PER_DAY = 2          # a 3rd account on one phone inside 24 h is banned (temporary)
DEVICE_BAN_SECONDS = 24 * 3600
EVASION_BAN_SECONDS = 7 * 24 * 3600  # new account on a phone that holds a permanently banned account

_rate_lock = threading.Lock()
_rate: dict[str, list[float]] = {}


def _db():
    return firestore.client()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    return (fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "?")) or "?"


def _limit(key: str, n: int, seconds: int):
    """Tiny in-memory rate limit (one Render instance). Raises 429."""
    now = time.time()
    with _rate_lock:
        hits = [t for t in _rate.get(key, []) if now - t < seconds]
        if len(hits) >= n:
            _rate[key] = hits
            raise HTTPException(status_code=429, detail="Too many tries. Please wait a while.")
        hits.append(now)
        _rate[key] = hits


def _uid(authorization: str | None) -> dict:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Sign in first")
    try:
        return fb_auth.verify_id_token(authorization.split(" ", 1)[1].strip())
    except Exception:
        raise HTTPException(status_code=401, detail="Bad sign-in")


def _mask(email: str) -> str:
    if "@" not in email:
        return ""
    local, dom = email.split("@", 1)
    if len(local) <= 2:
        return local[:1] + "•••@" + dom
    return local[0] + "•" * min(6, max(2, len(local) - 2)) + local[-1] + "@" + dom


def _email_key(email: str, device: str = "") -> str:
    """Lock key = this email on THIS phone. Someone failing logins on their own phone locks only their phone, never the
    real owner on theirs. (A client that sends no device id falls back to the old email-only key.)"""
    base = email.strip().lower() + ("|" + device.strip()[:80] if device and device.strip() else "")
    return hashlib.sha256(base.encode()).hexdigest()


def _web_key() -> str:
    k = os.getenv("FIREBASE_WEB_API_KEY", "").strip()
    if not k:
        raise HTTPException(status_code=503, detail="Server is not set up for this yet")
    return k


class DeviceReq(BaseModel):
    deviceId: str


class QueryReq(BaseModel):
    query: str


# ── 2-step (email code) accounts: a code is needed before ANY password reset ──
import hmac, smtplib, ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr

RESET_CODE_SECONDS = 10 * 60
RESET_TOKEN_SECONDS = 15 * 60
RESET_MAX_ATTEMPTS = 5


def _twofa_on(prof: dict) -> bool:
    tf = prof.get("twoFactor") or {}
    return isinstance(tf, dict) and tf.get("enabled") is True


def _send_mail(to_email: str, subject: str, text: str, html_body: str) -> bool:
    """Same SMTP settings as the notifier service (SMTP_SERVER / SMTP_PORT / SMTP_USER / SMTP_PASS / SENDER_NAME)."""
    server_host = os.getenv("SMTP_SERVER", "mail.chadivavirtuals.name.ng")
    port = int(os.getenv("SMTP_PORT", "465"))
    user = os.getenv("SMTP_USER", "auth@chadivavirtuals.name.ng")
    pw = os.getenv("SMTP_PASS", "")
    if not pw:
        return False
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = formataddr((os.getenv("SENDER_NAME", "ZINDRYX"), user))
        msg["To"] = to_email
        msg.attach(MIMEText(text, "plain", "utf-8"))
        msg.attach(MIMEText(html_body, "html", "utf-8"))
        if port == 465:
            srv = smtplib.SMTP_SSL(server_host, port, timeout=15, context=ssl.create_default_context())
        else:
            srv = smtplib.SMTP(server_host, port, timeout=15)
            srv.starttls(context=ssl.create_default_context())
        try:
            srv.login(user, pw)
            srv.sendmail(user, [to_email], msg.as_string())
        finally:
            try:
                srv.quit()
            except Exception:
                pass
        return True
    except Exception as e:
        print(f"[auth] smtp error: {e}")
        return False


def _h(uid: str, v: str) -> str:
    return hashlib.sha256(f"{uid}:{v}".encode()).hexdigest()


def _require_reset_token(uid: str, prof: dict, token: str):
    """2-step accounts must present the token that /auth/verify-reset-code gave out."""
    if not _twofa_on(prof):
        return
    d = _db().collection("resetCodes").document(uid).get().to_dict() or {}
    ok = (
        token
        and d.get("tokenHash")
        and hmac.compare_digest(_h(uid, token), d["tokenHash"])
        and d.get("tokenExpires") is not None
        and d["tokenExpires"] > _now()
    )
    if not ok:
        raise HTTPException(status_code=403, detail="Enter the code we email you first.")


class ResetCodeReq(BaseModel):
    query: str
    code: str = ""


@router.post("/send-reset-code")
def send_reset_code(req: QueryReq, request: Request):
    _limit("rc-ip:" + _ip(request), 10, 3600)
    hit = _lookup(req.query)
    if not hit or not hit[1]:
        raise HTTPException(status_code=404, detail="No account found")
    uid, email, prof = hit
    if not _twofa_on(prof):
        return {"needed": False}
    _limit("rc-uid:" + uid, 4, 3600)
    ref = _db().collection("resetCodes").document(uid)
    code = f"{secrets.randbelow(1_000_000):06d}"
    ref.set({"codeHash": _h(uid, code), "codeExpires": _now() + timedelta(seconds=RESET_CODE_SECONDS), "attempts": 0})
    name = (prof.get("name") or prof.get("displayName") or "there").strip()[:60]
    name = name.replace("<", "").replace(">", "")
    mins = RESET_CODE_SECONDS // 60
    ok = _send_mail(
        email, "Your ZINDRYX password reset code",
        f"Hi {name},\n\nYour password reset code is: {code}\n\nIt expires in {mins} minutes. If you didn't ask for this, ignore this email and consider changing your password.\n\n- ZINDRYX",
        f'<div style="font-family:Arial,sans-serif;padding:20px;background:#f4f4f7"><div style="max-width:480px;margin:0 auto;background:#fff;padding:24px;border-radius:12px">'
        f'<h2 style="color:#7c3aed;margin-top:0">ZINDRYX Security</h2><p>Hi {name}, use this code to reset your password:</p>'
        f'<div style="background:#f3e8ff;color:#7c3aed;font-size:28px;font-weight:bold;text-align:center;padding:12px;border-radius:8px;letter-spacing:4px;margin:20px 0">{code}</div>'
        f'<p style="font-size:12px;color:#666">Expires in {mins} minutes. If you did not request this, ignore this email.</p></div></div>',
    )
    if not ok:
        ref.delete()
        raise HTTPException(status_code=502, detail="Couldn't send the code. Try again.")
    return {"needed": True, "sent": True, "maskedEmail": _mask(email)}


@router.post("/verify-reset-code")
def verify_reset_code(req: ResetCodeReq, request: Request):
    _limit("vc-ip:" + _ip(request), 30, 3600)
    hit = _lookup(req.query)
    if not hit:
        raise HTTPException(status_code=404, detail="No account found")
    uid, _, prof = hit
    ref = _db().collection("resetCodes").document(uid)
    d = ref.get().to_dict() or {}
    if not d.get("codeHash") or d.get("codeExpires") is None or d["codeExpires"] < _now():
        raise HTTPException(status_code=400, detail="That code has expired. Ask for a new one.")
    if int(d.get("attempts", 0)) >= RESET_MAX_ATTEMPTS:
        raise HTTPException(status_code=429, detail="Too many wrong codes. Ask for a new one.")
    if not hmac.compare_digest(_h(uid, req.code.strip()), d["codeHash"]):
        ref.update({"attempts": firestore.Increment(1)})
        raise HTTPException(status_code=400, detail="That code isn't right.")
    token = secrets.token_urlsafe(24)
    ref.set({"tokenHash": _h(uid, token), "tokenExpires": _now() + timedelta(seconds=RESET_TOKEN_SECONDS)})
    return {"ok": True, "token": token}


# ── find an account by email OR username ────────────────────────────────
def _lookup(query: str):
    """-> (uid, email, profile dict) or None."""
    q = (query or "").strip()
    if not q:
        return None
    db = _db()
    if "@" in q and not q.startswith("@"):
        try:
            rec = fb_auth.get_user_by_email(q)
        except Exception:
            return None
        prof = db.collection("users").document(rec.uid).get().to_dict() or {}
        return rec.uid, rec.email or q, prof
    uname = q.lstrip("@").strip()
    for cand in dict.fromkeys([uname, uname.lower()]):
        docs = list(db.collection("users").where("username", "==", cand).limit(1).stream())
        if docs:
            prof = docs[0].to_dict() or {}
            email = prof.get("email") or ""
            if not email:
                try:
                    email = fb_auth.get_user(docs[0].id).email or ""
                except Exception:
                    email = ""
            return docs[0].id, email, prof
    return None


@router.post("/find-account")
def find_account(req: QueryReq, request: Request):
    _limit("find:" + _ip(request), 25, 600)
    hit = _lookup(req.query)
    if not hit:
        return {"found": False}
    uid, email, prof = hit
    return {
        "found": True,
        "name": (prof.get("name") or prof.get("displayName") or "").strip(),
        "username": (prof.get("username") or "").strip(),
        "photoUrl": prof.get("photoUrl") or "",
        "maskedEmail": _mask(email),
        "twoFactor": _twofa_on(prof),
    }


class SendResetReq(BaseModel):
    query: str
    resetToken: str = ""


@router.post("/send-reset")
def send_reset(req: SendResetReq, request: Request):
    _limit("reset-ip:" + _ip(request), 10, 3600)
    hit = _lookup(req.query)
    if not hit or not hit[1]:
        raise HTTPException(status_code=404, detail="No account found")
    uid, email, prof = hit
    _require_reset_token(uid, prof, req.resetToken)
    _limit("reset-uid:" + uid, 3, 3600)
    r = requests.post(
        f"https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode?key={_web_key()}",
        json={"requestType": "PASSWORD_RESET", "email": email},
        timeout=15,
    )
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail="Couldn't send the email")
    return {"sent": True, "maskedEmail": _mask(email)}


class OldPwReq(BaseModel):
    query: str
    oldPassword: str
    newPassword: str
    deviceId: str = ""
    resetToken: str = ""


@router.post("/reset-with-old")
def reset_with_old(req: OldPwReq, request: Request):
    _limit("old-ip:" + _ip(request), 15, 3600)
    hit = _lookup(req.query)
    if not hit or not hit[1]:
        raise HTTPException(status_code=404, detail="No account found")
    uid, email, prof = hit
    _require_reset_token(uid, prof, req.resetToken)
    if len(req.newPassword) < 6:
        raise HTTPException(status_code=400, detail="The new password must be at least 6 characters.")
    st = _lock_state(email, req.deviceId)
    if st["locked"]:
        return {"ok": False, "locked": True, **st}
    r = requests.post(
        f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={_web_key()}",
        json={"email": email, "password": req.oldPassword, "returnSecureToken": False},
        timeout=15,
    )
    if r.status_code != 200:
        # a wrong old password counts like a wrong login
        st = _record_fail(email, req.deviceId)
        return {"ok": False, "wrongPassword": True, **st}
    fb_auth.update_user(uid, password=req.newPassword)
    _clear_fails(email, req.deviceId)
    return {"ok": True}


# ── login attempt lockout ───────────────────────────────────────────────
def _lock_state(email: str, device: str = "") -> dict:
    snap = _db().collection("loginAttempts").document(_email_key(email, device)).get()
    d = snap.to_dict() or {}
    if d.get("permanent"):
        return {"locked": True, "permanent": True, "untilMs": 0}
    until = d.get("lockedUntil")
    if until is not None and until > _now():
        return {"locked": True, "permanent": False, "untilMs": int(until.timestamp() * 1000)}
    return {"locked": False, "permanent": False, "untilMs": 0}


def _record_fail(email: str, device: str = "") -> dict:
    ref = _db().collection("loginAttempts").document(_email_key(email, device))
    d = ref.get().to_dict() or {}
    last = d.get("lastFailAt")
    count = int(d.get("count", 0))
    if last is not None and (_now() - last).total_seconds() > FAIL_DECAY_SECONDS:
        count = 0
    count += 1
    update = {"count": count, "lastFailAt": _now()}
    permanent = bool(d.get("permanent"))
    until = None
    for fails, secs in LOCK_TIERS:
        if count >= fails:
            if secs is None:
                permanent = True
            else:
                until = _now() + timedelta(seconds=secs)
            break
    if permanent:
        update["permanent"] = True
    if until is not None:
        update["lockedUntil"] = until
    ref.set(update, merge=True)
    left_before_lock = next((f - count for f, _ in sorted(LOCK_TIERS, key=lambda t: t[0]) if f > count), None)
    return {
        "locked": permanent or until is not None,
        "permanent": permanent,
        "untilMs": int(until.timestamp() * 1000) if until else 0,
        "attemptsLeft": left_before_lock,
        "count": count,
    }


def _clear_fails(email: str, device: str = ""):
    ref = _db().collection("loginAttempts").document(_email_key(email, device))
    d = ref.get().to_dict() or {}
    if d.get("permanent"):
        return  # a permanent lock is lifted by support only
    ref.delete()


class EmailReq(BaseModel):
    email: str
    deviceId: str = ""


@router.post("/login-precheck")
def login_precheck(req: EmailReq, request: Request):
    _limit("pre:" + _ip(request), 60, 600)
    return _lock_state(req.email, req.deviceId)


@router.post("/login-failed")
def login_failed(req: EmailReq, request: Request):
    _limit("fail-ip:" + _ip(request), 40, 600)
    if "@" not in req.email:
        return {"locked": False}
    return _record_fail(req.email, req.deviceId)


@router.post("/login-ok")
def login_ok(req: DeviceReq | None = None, authorization: str | None = Header(default=None)):
    tok = _uid(authorization)
    email = tok.get("email")
    if email:
        _clear_fails(email, req.deviceId if req else "")
    return {"ok": True}


# ── bans ────────────────────────────────────────────────────────────────
def _ban(uid: str, reason: str, seconds: int | None, by: str, resolvable: bool = False) -> dict:
    ref = _db().collection("users").document(uid)
    upd = {
        "banned": True,
        "bannedReason": reason,
        "bannedAt": firestore.SERVER_TIMESTAMP,
        "banId": "BN-" + secrets.token_hex(3).upper(),
        "bannedBy": by,
        "resolvable": resolvable,
    }
    if seconds is None:
        upd["banUntil"] = firestore.DELETE_FIELD
    else:
        upd["banUntil"] = _now() + timedelta(seconds=seconds)
    ref.set(upd, merge=True)
    return {"banned": True, "untilMs": 0 if seconds is None else int((_now() + timedelta(seconds=seconds)).timestamp() * 1000)}


def _active_ban(prof: dict) -> bool:
    if prof.get("banned") is not True:
        return False
    until = prof.get("banUntil")
    return until is None or until > _now()


@router.post("/ban-status")
def ban_status(authorization: str | None = Header(default=None)):
    uid = _uid(authorization)["uid"]
    ref = _db().collection("users").document(uid)
    prof = ref.get().to_dict() or {}
    if prof.get("banned") is True and not _active_ban(prof):
        ref.set({"banned": False, "banUntil": firestore.DELETE_FIELD}, merge=True)   # the temporary ban ran out
        return {"banned": False}
    if not _active_ban(prof):
        return {"banned": False}
    until = prof.get("banUntil")
    ban_id = prof.get("banId") or ""
    ap = _db().collection("appeals").document(ban_id).get().to_dict() if ban_id else None
    at = prof.get("bannedAt")
    return {
        "banned": True,
        "reason": prof.get("bannedReason") or "Breaking the Zindryx rules",
        "banId": ban_id,
        "bannedAtMs": int(at.timestamp() * 1000) if at else 0,
        "untilMs": int(until.timestamp() * 1000) if until else 0,
        "permanent": until is None,
        "resolvable": bool(prof.get("resolvable")),
        "appeal": ({"status": ap.get("status", "pending"), "reply": ap.get("reply", "")} if ap else None),
    }


class AppealReq(BaseModel):
    text: str


@router.post("/appeal")
def appeal(req: AppealReq, authorization: str | None = Header(default=None)):
    uid = _uid(authorization)["uid"]
    prof = _db().collection("users").document(uid).get().to_dict() or {}
    if not _active_ban(prof):
        raise HTTPException(status_code=400, detail="Your account is not banned")
    ban_id = prof.get("banId") or uid
    ref = _db().collection("appeals").document(ban_id)
    if ref.get().exists:
        raise HTTPException(status_code=409, detail="You already sent an appeal for this ban")
    text = (req.text or "").strip()[:1000]
    if len(text) < 10:
        raise HTTPException(status_code=400, detail="Please explain in a little more detail")
    ref.set({"uid": uid, "banId": ban_id, "text": text, "status": "pending", "createdAt": firestore.SERVER_TIMESTAMP})
    return {"ok": True}


@router.post("/resolve")
def resolve(authorization: str | None = Header(default=None)):
    uid = _uid(authorization)["uid"]
    ref = _db().collection("users").document(uid)
    prof = ref.get().to_dict() or {}
    if not _active_ban(prof):
        return {"ok": True}
    if prof.get("banUntil") is None or not prof.get("resolvable"):
        raise HTTPException(status_code=403, detail="This ban can't be resolved here. You can appeal it.")
    ref.set({"banned": False, "banUntil": firestore.DELETE_FIELD, "resolvable": False}, merge=True)
    return {"ok": True}


# ── device limits ───────────────────────────────────────────────────────


def _device_ref(device_id: str):
    d = re.sub(r"[^a-zA-Z0-9]", "", device_id)[:80]
    if len(d) < 16:
        raise HTTPException(status_code=400, detail="Bad device")
    return _db().collection("devices").document(d)


@router.post("/device-check")
def device_check(req: DeviceReq, request: Request):
    """Called BEFORE creating an account on this phone."""
    _limit("dev:" + _ip(request), 60, 600)
    d = _device_ref(req.deviceId).get().to_dict() or {}
    blocked = d.get("blockedUntil")
    if blocked is not None and blocked > _now():
        return {"allowed": False, "untilMs": int(blocked.timestamp() * 1000)}
    cutoff = _now() - timedelta(seconds=DEVICE_BAN_SECONDS)
    recent = [a for a in d.get("accounts", []) if a.get("at") and a["at"] > cutoff]
    if len(recent) >= DEVICE_SIGNUPS_PER_DAY:
        until = min(a["at"] for a in recent) + timedelta(seconds=DEVICE_BAN_SECONDS)
        return {"allowed": False, "untilMs": int(until.timestamp() * 1000)}
    return {"allowed": True}


@router.post("/device-register")
def device_register(req: DeviceReq, authorization: str | None = Header(default=None)):
    """Called when an account signs up or signs in on this phone. Brand-new accounts are counted; too many get a temporary ban."""
    uid = _uid(authorization)["uid"]
    ref = _device_ref(req.deviceId)
    d = ref.get().to_dict() or {}
    accounts = list(d.get("accounts", []))
    now = _now()
    try:
        created = datetime.fromtimestamp(fb_auth.get_user(uid).user_metadata.creation_timestamp / 1000, tz=timezone.utc)
    except Exception:
        created = None
    is_new = created is not None and (now - created).total_seconds() < 15 * 60
    if is_new and not any(a.get("uid") == uid for a in accounts):
        accounts.append({"uid": uid, "at": now})
    accounts = [a for a in accounts if a.get("uid")][-30:]
    cutoff = now - timedelta(seconds=DEVICE_BAN_SECONDS)
    recent = [a for a in accounts if a.get("at") and a["at"] > cutoff]
    out = {"banned": False}
    if is_new and len(recent) > DEVICE_SIGNUPS_PER_DAY:
        out = _ban(uid, "Too many accounts were created on one phone", DEVICE_BAN_SECONDS, "auto-device")
        ref.set({"accounts": accounts, "blockedUntil": now + timedelta(seconds=DEVICE_BAN_SECONDS)}, merge=True)
        return out
    ref.set({"accounts": accounts}, merge=True)
    # ban evasion: a NEW account on a phone that holds a permanently banned account
    if is_new:
        for a in accounts:
            other = a.get("uid")
            if other and other != uid:
                p = _db().collection("users").document(other).get().to_dict() or {}
                if p.get("banned") is True and p.get("banUntil") is None:
                    return _ban(uid, "This phone belongs to a banned account", EVASION_BAN_SECONDS, "auto-device")
    return out
