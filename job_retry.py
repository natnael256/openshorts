"""Manual retry of a failed clip job, and the per-job settings a resume needs.

Pure logic, stdlib only at import (plus ``jsonfile``), so it is unit-testable
without importing ``app``. Two dotfile sidecars live in the job directory
(``/videos`` never serves a dot segment, see ``media_auth.normalise_relpath``):

* ``.submit.json`` (``SUBMIT_SIDECAR``): what ``/api/process`` decided for the
  job, enough to run it again: the argv, the non-secret per-job settings
  (closed allowlist ``SETTING_KEYS``), priority, webhook URL, partial cut and
  the manual-retry counter. NEVER an API key, token, proxy URL, cookie, the
  webhook secret or the creator brief (that one stays in ``.brief.json``).
* ``.failed.json`` (``FAILED_SIDECAR``): the plain-language reason of the last
  failure (closed set of codes), so it survives a server restart.

A sidecar is data on disk: every read is shape-validated and the stored argv
is only ever run after ``cmd_is_safe`` and with the interpreter rebuilt from
``sys.executable``.
"""
import json
import os
import re
import threading
import time

from jsonfile import write_json_atomic

SUBMIT_SIDECAR = ".submit.json"
FAILED_SIDECAR = ".failed.json"

MANUAL_RETRY_LIMIT = int(os.environ.get("MANUAL_RETRY_LIMIT", "5"))

# --- per-job settings ---------------------------------------------------------
# What app.layout_env can return (AUTO_LAYOUT, the LAYOUT_ENV values and
# LAYOUT_IMPLIES extras) plus the generation controls /api/process forwards.
_FLAG_KEYS = frozenset({
    "AUTO_LAYOUT", "SPLIT_LAYOUT", "SCREENCAST_LAYOUT", "SPEAKER_CUT", "PUNCH_IN",
    "SPEAKER_SIGNAL", "AUTO_HOOK", "AUTO_TEASER", "AUTO_CAPTIONS",
})
# name -> (low, high, integer)
_NUMERIC_KEYS = {
    "CLIP_TARGET_MIN": (1, 15, True),
    "CLIP_TARGET_MAX": (1, 15, True),
    "CLIP_MIN_SECONDS": (5, 175, False),
    "CLIP_MAX_SECONDS": (10, 180, False),
}
SETTING_KEYS = frozenset(_FLAG_KEYS | set(_NUMERIC_KEYS) | {"AUTO_HOOK_STYLE", "GEMINI_MODEL"})
_MAX_VALUE = 64
_PRINTABLE = re.compile(r"^[\x21-\x7e]{1,64}$")   # no spaces, no control chars
_STYLE_RE = re.compile(r"^[a-z0-9_]{1,32}$")


def _short_value(value):
    return isinstance(value, str) and len(value) <= _MAX_VALUE and bool(_PRINTABLE.match(value))


def pick_settings(env, base=None) -> dict:
    """The allowlisted, non-secret per-job settings in ``env``.

    With ``base`` (the server's own environment) only the values this job
    changed are kept, so a deployment default is never frozen into the job.
    """
    out = {}
    if not isinstance(env, dict):
        return out
    for key in sorted(SETTING_KEYS):
        value = env.get(key)
        if not _short_value(value):
            continue
        if base is not None and base.get(key) == value:
            continue
        out[key] = value
    return out


def _valid_setting(key, value, model_choices, hook_styles):
    if key not in SETTING_KEYS or not _short_value(value):
        return False
    if key in _FLAG_KEYS:
        return value in ("0", "1")
    if key in _NUMERIC_KEYS:
        lo, hi, integer = _NUMERIC_KEYS[key]
        try:
            num = float(value)
        except (TypeError, ValueError):
            return False
        if num != num or not (lo <= num <= hi):
            return False
        return (num == int(num)) if integer else True
    if key == "GEMINI_MODEL":
        return value in tuple(model_choices or ())
    if key == "AUTO_HOOK_STYLE":
        return bool(_STYLE_RE.match(value)) and value in set(hook_styles or ())
    return False


def apply_settings(env, settings, model_choices, hook_styles=()) -> list:
    """Write stored ``settings`` back into ``env`` after re-validating each one.

    Returns the keys applied. Anything off the allowlist, malformed or out of
    range is dropped; ``GEMINI_MODEL`` only from ``model_choices`` and
    ``AUTO_HOOK_STYLE`` only from ``hook_styles``.
    """
    applied = []
    if not isinstance(settings, dict) or not isinstance(env, dict):
        return applied
    for key, value in settings.items():
        if _valid_setting(key, value, model_choices, hook_styles):
            env[key] = value
            applied.append(key)
    return applied


# --- the stored argv ----------------------------------------------------------
_VALUE_FLAGS = ("-u", "-i", "-o", "--format", "--transcript")
_FORMATS = ("vertical", "horizontal", "square")
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_URL_RE = re.compile(r"^https?://[^\s\x00-\x1f\x7f]{1,2048}$", re.IGNORECASE)


def _cmd_shape_ok(cmd):
    return (isinstance(cmd, list) and len(cmd) >= 3 and all(isinstance(a, str) for a in cmd)
            and bool(cmd[0]) and cmd[1] == "-u" and cmd[2] == "main.py")


def _parse_args(cmd):
    """{flag: value} for the arguments after ``main.py``, or None when the argv
    holds an unknown flag, a repeated flag or a flag without its value."""
    if not _cmd_shape_ok(cmd):
        return None
    args = {}
    i = 3
    while i < len(cmd):
        flag = cmd[i]
        if flag in args:
            return None
        if flag == "--keep-original":
            args[flag] = True
            i += 1
            continue
        if flag not in _VALUE_FLAGS or i + 1 >= len(cmd):
            return None
        args[flag] = cmd[i + 1]
        i += 2
    return args


def source_of(cmd) -> dict:
    """``{"url", "input_path"}`` from a stored argv (either may be None)."""
    args = _parse_args(cmd) or {}
    url = args.get("-u")
    path = args.get("-i")
    return {"url": url if isinstance(url, str) else None,
            "input_path": path if isinstance(path, str) else None}


def _real(path):
    return os.path.normcase(os.path.realpath(path))


def cmd_is_safe(cmd, output_dir, upload_dir, job_id) -> bool:
    """Whether ``cmd`` is exactly the shape /api/process builds for this job.

    Interpreter, ``-u``, ``main.py``, then only ``-u <url> --keep-original`` or
    ``-i <upload> [--transcript <file in the job dir>]``, ``-o <this job's
    directory>`` and an optional ``--format``. ``-i`` must be a file directly in
    ``upload_dir`` named ``<job_id>_*``. Anything else is refused.
    """
    if not isinstance(job_id, str) or not _UUID_RE.match(job_id):
        return False
    args = _parse_args(cmd)
    if not args:
        return False
    interp = os.path.basename(cmd[0]).lower()
    if not interp.startswith("python"):
        return False
    url, inp = args.get("-u"), args.get("-i")
    if (url is None) == (inp is None):
        return False
    if "-o" not in args:
        return False
    job_dir = _real(os.path.join(output_dir, job_id))
    if _real(args["-o"]) != job_dir:
        return False
    if "--format" in args and args["--format"] not in _FORMATS:
        return False
    if url is not None:
        if not _URL_RE.match(url) or "--transcript" in args:
            return False
    else:
        if "--keep-original" in args:
            return False
        if os.path.dirname(_real(inp)) != _real(upload_dir):
            return False
        if not os.path.basename(inp).startswith(f"{job_id}_"):
            return False
        if os.path.basename(_real(inp)) != os.path.normcase(os.path.basename(inp)):
            return False   # a symlink renamed to another job's upload
    if "--transcript" in args:
        if os.path.dirname(_real(args["--transcript"])) != job_dir:
            return False
    return True


# --- .submit.json -------------------------------------------------------------
_submit_lock = threading.Lock()


def _num_or_none(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value


def _clean_partial(raw):
    if not isinstance(raw, dict):
        return None
    out = {}
    for key in ("processed_minutes", "total_minutes", "start_seconds", "end_seconds"):
        if key in raw and raw[key] is not None:
            num = _num_or_none(raw[key])
            if num is None or num < 0:
                return None
            out[key] = num
    if not out.get("processed_minutes"):
        return None
    if raw.get("chosen") is True:
        out["chosen"] = True
    return out


def _short_str(value, cap=2048):
    return value if isinstance(value, str) and 0 < len(value) <= cap else None


def write_submit(job_dir, *, cmd, settings, priority, webhook_url=None, base_url=None,
                 partial=None, source_cap_minutes=None, skip_statics=False,
                 via_api_key=False, reservation_id=None) -> None:
    """Write the submit sidecar atomically (counter starts at 0)."""
    data = {
        "version": 1,
        "created_at": time.time(),
        "cmd": list(cmd),
        "settings": dict(settings or {}),
        "priority": int(priority),
        "webhook_url": webhook_url,
        "base_url": base_url,
        "partial": partial,
        "source_cap_minutes": source_cap_minutes,
        "skip_statics": bool(skip_statics),
        "via_api_key": bool(via_api_key),
        # Not a secret: lets a retry release a reservation that is somehow
        # still held (release is a no-op unless the row is still 'reserved').
        "reservation_id": None if reservation_id is None else str(reservation_id),
        "retries": 0,
    }
    write_json_atomic(os.path.join(job_dir, SUBMIT_SIDECAR), data)


def _read_raw(job_dir):
    try:
        with open(os.path.join(job_dir, SUBMIT_SIDECAR), "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def read_submit(job_dir) -> dict:
    """The validated submit sidecar, or ``{}`` when missing or malformed."""
    if not job_dir:
        return {}
    raw = _read_raw(job_dir)
    if raw is None or not _cmd_shape_ok(raw.get("cmd")):
        return {}
    retries = raw.get("retries", 0)
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        return {}
    priority = raw.get("priority", 2)
    if isinstance(priority, bool) or not isinstance(priority, int) or not 0 <= priority <= 9:
        priority = 2
    settings = raw.get("settings")
    settings = {k: v for k, v in settings.items() if k in SETTING_KEYS and _short_value(v)} \
        if isinstance(settings, dict) else {}
    cap = _num_or_none(raw.get("source_cap_minutes"))
    created = _num_or_none(raw.get("created_at"))
    return {
        "cmd": list(raw["cmd"]),
        "settings": settings,
        "priority": priority,
        "webhook_url": _short_str(raw.get("webhook_url")),
        "base_url": _short_str(raw.get("base_url")),
        "partial": _clean_partial(raw.get("partial")),
        "source_cap_minutes": cap if cap and cap > 0 else None,
        "skip_statics": raw.get("skip_statics") is True,
        "via_api_key": raw.get("via_api_key") is True,
        "reservation_id": _short_str(raw.get("reservation_id"), 128),
        "created_at": created,
        "retries": retries,
    }


_UPDATABLE = ("reservation_id", "settings", "priority")


def bump_retries(job_dir, **updates) -> int:
    """Add one to the retry counter (optionally updating ``reservation_id``,
    ``settings`` or ``priority``) and return the new count."""
    with _submit_lock:
        raw = _read_raw(job_dir)
        if raw is None:
            raise FileNotFoundError(SUBMIT_SIDECAR)
        current = raw.get("retries", 0)
        if isinstance(current, bool) or not isinstance(current, int) or current < 0:
            current = 0
        raw["retries"] = current + 1
        for key, value in updates.items():
            if key in _UPDATABLE:
                raw[key] = value
        write_json_atomic(os.path.join(job_dir, SUBMIT_SIDECAR), raw)
        return raw["retries"]


# --- failure reasons ----------------------------------------------------------
MESSAGES = {
    "missing_key": "No Gemini API key was available. Add your key in Settings, then retry.",
    "download_blocked": ("YouTube blocked the download of this video. Retrying later may "
                         "work; uploading the video file always does."),
    "video_unavailable": ("This video cannot be reached: it is private, removed, "
                          "members-only or not available in our region. Upload the file instead."),
    "no_clips": ("The AI did not find clip-worthy moments in this video. Try a longer "
                 "video or a different part of it."),
    "no_audio": "This video has no audio track to work from.",
    "content_blocked": "The AI provider declined to process this video's content.",
    "too_short": "This video is too short to cut clips from.",
    "model_overloaded": "The AI service was overloaded. Retrying in a few minutes usually works.",
    "model_unavailable": ("The selected AI model is not available. Pick another model in "
                          "Settings, then retry."),
    "out_of_memory": ("The server ran out of graphics memory while rendering. "
                      "Retrying usually helps."),
    "server_restarted": ("The server restarted while your video was processing. "
                         "Retrying usually helps."),
    "quota": "You are out of processing minutes for this period.",
    "unknown": "Something went wrong while processing this video. Retrying usually helps.",
}
FAILURE_CODES = tuple(MESSAGES)

# Ordered: the first matching rule wins. Markers come from the lines main.py
# and app.py actually print (main.get_viral_clips / get_visual_clips,
# main._content_block and the download banner, gemini_worker.raise_if_blocked,
# app._TRANSIENT_MARKERS / _PERMANENT_MARKERS, transcribe_backends).
_RULES = (
    ("missing_key", ("gemini_api_key not found", "missing x-gemini-key",
                     "add a gemini_api_key")),
    ("quota", ("quota_exceeded", "out of openshorts processing minutes")),
    ("content_blocked", ("prohibited_content", "blocked this video", "blocked its answer",
                         "geminiblockederror")),
    # Gemini's 404 for a retired or unknown model ("This model models/x is no
    # longer available to new users"). Before video_unavailable: a bare "no
    # longer available" there used to swallow it and blame the video.
    ("model_unavailable", ("no longer available to new users", "is not found for api version",
                           "is not supported for generatecontent")),
    # Phrased as YouTube words them ("This video is no longer available ...",
    # "This video has been removed by ..."), not the bare tails: an AI-side
    # error saying "no longer available" is not a video problem.
    ("video_unavailable", ("private video", "this video is private", "members-only",
                           "join this channel", "video has been removed",
                           "video is no longer available",
                           "not available in your country", "sign in to confirm your age",
                           "account associated with this video has been terminated")),
    ("download_blocked", ("not a bot", "http error 403", "http error 429",
                          "youtube blocked the request")),
    ("video_unavailable", ("video unavailable", "video is unavailable")),
    ("download_blocked", ("youtube download failed", "download failed (all strategies)")),
    ("no_audio", ("no audio track", "no_audio")),
    ("too_short", ("short-form content", "video is too short")),
    ("model_overloaded", ("resource_exhausted", "resource exhausted", "overloaded",
                          "too many requests", "exceeded your current quota",
                          "503 unavailable", "deadline exceeded")),
    ("no_clips", ("did not return usable clips", "clip detection failed")),
    ("out_of_memory", ("out of memory", "outofmemory", "cuda error", "cuda_error", "cublas",
                       "cudnn", "nvenc", "generic error in an external library",
                       "exit code 187", "cannot allocate memory", "memoryerror",
                       "exit code -9")),
    ("server_restarted", ("exit code -15", "resume attempts", "server restarted")),
)
_STATUS_RE = re.compile(r"\b(?:error|code|status)\W{0,3}(?:429|503)\b")
_MODEL_GONE_RE = re.compile(r"\bmodels/[a-z0-9._-]+ is (?:no longer available|not found)")


def classify_failure(logs_or_text) -> dict:
    """``{"code", "message"}`` for a failed job's error text (or log lines).

    The message is fixed per code: never a raw log line, path or URL."""
    if isinstance(logs_or_text, (list, tuple)):
        text = " ".join(str(x) for x in logs_or_text)
    else:
        text = str(logs_or_text or "")
    low = text.lower()
    for code, markers in _RULES:
        if code == "no_clips" and _STATUS_RE.search(low):
            # A Gemini 429/503 that ended in "Clip detection failed".
            return reason("model_overloaded")
        if code == "model_unavailable" and _MODEL_GONE_RE.search(low):
            return reason(code)
        if any(m in low for m in markers):
            return {"code": code, "message": MESSAGES[code]}
    if _STATUS_RE.search(low):
        return reason("model_overloaded")
    return reason("unknown")


def reason(code) -> dict:
    """The fixed ``{"code", "message"}`` for a known code (``unknown`` otherwise)."""
    code = code if code in MESSAGES else "unknown"
    return {"code": code, "message": MESSAGES[code]}


def write_failed(job_dir, failure) -> None:
    code = failure.get("code") if isinstance(failure, dict) else None
    write_json_atomic(os.path.join(job_dir, FAILED_SIDECAR),
                      {"code": code if code in MESSAGES else "unknown", "at": time.time()})


def read_failed(job_dir):
    """The stored reason (message rebuilt from the closed set), or None."""
    if not job_dir:
        return None
    try:
        with open(os.path.join(job_dir, FAILED_SIDECAR), "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError, TypeError):
        return None
    code = data.get("code") if isinstance(data, dict) else None
    if code not in MESSAGES:
        return None
    return reason(code)


def clear_failed(job_dir) -> None:
    try:
        os.remove(os.path.join(job_dir, FAILED_SIDECAR))
    except OSError:
        pass


def clear_for_manual_retry(job_dir) -> None:
    """What a manual retry drops on top of ``app._clean_for_retry``: the old
    failure reason and the brief directives the failed run recorded (main.py
    writes them again from ``.brief.json``)."""
    for name in (FAILED_SIDECAR, ".brief_applied.json"):
        try:
            os.remove(os.path.join(job_dir, name))
        except OSError:
            pass


# --- retry eligibility --------------------------------------------------------
RETRY_REASONS = {
    "limit": "This video was retried too many times. Please submit it again.",
    "not_retryable": "Please submit this video again.",
    "source_missing": ("The original upload is no longer on the server. "
                       "Please upload the video again."),
    "running": "This video is already being processed.",
}


def retry_view(job_id, output_dir, upload_dir, *, active=False, submit=None) -> dict:
    """``{"allowed", "used", "limit", "reason"}`` for a failed job."""
    job_dir = os.path.join(output_dir, job_id) if isinstance(job_id, str) else None
    submit = read_submit(job_dir) if submit is None else submit
    used = int(submit.get("retries", 0)) if submit else 0
    out = {"allowed": False, "used": used, "limit": MANUAL_RETRY_LIMIT, "reason": None}
    if active:
        out["reason"] = RETRY_REASONS["running"]
    elif not submit or not cmd_is_safe(submit["cmd"], output_dir, upload_dir, job_id):
        out["reason"] = RETRY_REASONS["not_retryable"]
    elif used >= MANUAL_RETRY_LIMIT:
        out["reason"] = RETRY_REASONS["limit"]
    else:
        src = source_of(submit["cmd"]).get("input_path")
        if src and not os.path.isfile(src):
            out["reason"] = RETRY_REASONS["source_missing"]
        else:
            out["allowed"] = True
    return out
