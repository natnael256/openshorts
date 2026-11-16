"""Manual retry of failed clip jobs (job_retry.py + POST /api/jobs/{id}/retry),
the settings a resume re-applies, and the failure reasons /api/status reports."""
import asyncio
import json
import os
import sys
import types
import uuid

import httpx
import pytest
from fastapi import HTTPException

import clip_history
import creator_brief
import job_retry

app_module = pytest.importorskip("app")

MARK = "ZEBRA-RULE-7731"
URL = "https://example.com/watch?v=abc"
CHOICES = ("gemini-3.5-flash", "gemini-3.1-flash-lite")


# --- pure: settings -----------------------------------------------------------
def test_pick_keeps_only_allowlisted_settings_and_no_secret():
    env = {"GEMINI_API_KEY": "sk-SECRET1", "PROXY_URL": "http://u:SECRET2@h:1",
           "YOUTUBE_COOKIES": "SECRET3", "UPLOAD_POST_API_KEY": "SECRET4",
           "AUTO_HOOK": "1", "AUTO_HOOK_STYLE": "pill", "GEMINI_MODEL": "gemini-3.5-flash",
           "CLIP_TARGET_MIN": "3", "CLIP_TARGET_MAX": "3", "SPLIT_LAYOUT": "1",
           "SPEAKER_SIGNAL": "1", "AUTO_CAPTIONS": "0", "PATH": "/usr/bin",
           "CLIP_MIN_SECONDS": "x" * 65}
    picked = job_retry.pick_settings(env)
    assert set(picked) == {"AUTO_HOOK", "AUTO_HOOK_STYLE", "GEMINI_MODEL", "CLIP_TARGET_MIN",
                           "CLIP_TARGET_MAX", "SPLIT_LAYOUT", "SPEAKER_SIGNAL", "AUTO_CAPTIONS"}
    assert set(picked) <= job_retry.SETTING_KEYS


def test_secrets_never_reach_the_sidecar(tmp_path):
    env = {"GEMINI_API_KEY": "sk-SECRET1", "PROXY_URL": "http://u:SECRET2@h:1",
           "YOUTUBE_COOKIES": "SECRET3", "AUTO_HOOK": "1"}
    d = tmp_path / "j"
    d.mkdir()
    job_retry.write_submit(str(d), cmd=["python", "-u", "main.py", "-u", URL, "-o", str(d)],
                           settings=job_retry.pick_settings(env), priority=2)
    raw = (d / job_retry.SUBMIT_SIDECAR).read_text()
    for secret in ("SECRET1", "SECRET2", "SECRET3", "GEMINI_API_KEY", "PROXY_URL", "COOKIES"):
        assert secret not in raw
    assert json.loads(raw)["settings"] == {"AUTO_HOOK": "1"}


def test_pick_with_base_keeps_only_what_the_job_changed():
    base = {"AUTO_LAYOUT": "1", "GEMINI_MODEL": "gemini-3.1-flash-lite"}
    env = dict(base, AUTO_HOOK="1")
    assert job_retry.pick_settings(env, base=base) == {"AUTO_HOOK": "1"}


@pytest.mark.parametrize("key,value", [
    ("GEMINI_MODEL", "evil-model"), ("AUTO_HOOK", "yes"), ("AUTO_HOOK", "2"),
    ("CLIP_TARGET_MIN", "99"), ("CLIP_TARGET_MIN", "3.5"), ("CLIP_MIN_SECONDS", "nan"),
    ("CLIP_MAX_SECONDS", "500"), ("AUTO_HOOK_STYLE", "nope"), ("AUTO_HOOK_STYLE", "../x"),
    ("GEMINI_API_KEY", "k"), ("PROXY_URL", "http://x"), ("LD_PRELOAD", "/x.so"),
    ("AUTO_HOOK", 1), ("AUTO_HOOK", "1 ; rm"),
])
def test_apply_rejects_bad_values(key, value):
    env = {}
    assert job_retry.apply_settings(env, {key: value}, CHOICES, hook_styles=("pill",)) == []
    assert env == {}


def test_apply_writes_valid_values():
    env = {}
    applied = job_retry.apply_settings(env, {
        "GEMINI_MODEL": "gemini-3.5-flash", "AUTO_HOOK": "1", "AUTO_HOOK_STYLE": "pill",
        "CLIP_TARGET_MIN": "4", "CLIP_MIN_SECONDS": "20.0", "AUTO_LAYOUT": "0"},
        CHOICES, hook_styles=("pill",))
    assert sorted(applied) == sorted(env) and env["CLIP_MIN_SECONDS"] == "20.0"
    assert job_retry.apply_settings({}, None, CHOICES) == []
    assert job_retry.apply_settings({}, ["AUTO_HOOK"], CHOICES) == []


# --- pure: sidecar + argv -----------------------------------------------------
def _uid():
    return str(uuid.uuid4())


def test_submit_round_trip_and_bump(tmp_path):
    d = tmp_path / "j"
    d.mkdir()
    cmd = ["python", "-u", "main.py", "-u", URL, "--keep-original", "-o", str(d)]
    partial = {"processed_minutes": 5, "total_minutes": 20}
    job_retry.write_submit(str(d), cmd=cmd, settings={"AUTO_HOOK": "1"}, priority=1,
                           webhook_url="https://hook.example.com/x", partial=partial,
                           source_cap_minutes=21, skip_statics=True, reservation_id="r1")
    got = job_retry.read_submit(str(d))
    assert got["cmd"] == cmd and got["priority"] == 1 and got["retries"] == 0
    assert got["partial"] == partial and got["source_cap_minutes"] == 21
    assert got["skip_statics"] is True and got["reservation_id"] == "r1"
    assert job_retry.bump_retries(str(d), reservation_id="r2") == 1
    assert job_retry.bump_retries(str(d)) == 2
    got = job_retry.read_submit(str(d))
    assert got["retries"] == 2 and got["reservation_id"] == "r2"


@pytest.mark.parametrize("payload", [
    "not json", "[]", json.dumps({"cmd": "python -u main.py"}),
    json.dumps({"cmd": ["python", "main.py"]}),
    json.dumps({"cmd": ["python", "-u", "other.py"]}),
    json.dumps({"cmd": ["python", "-u", "main.py"], "retries": -1}),
    json.dumps({"cmd": ["python", "-u", "main.py"], "retries": True}),
    json.dumps({"cmd": ["python", "-u", "main.py"], "retries": "0"}),
    json.dumps({"cmd": ["python", "-u", "main.py", 5]}),
])
def test_corrupt_submit_reads_empty(tmp_path, payload):
    (tmp_path / job_retry.SUBMIT_SIDECAR).write_text(payload)
    assert job_retry.read_submit(str(tmp_path)) == {}


def test_missing_submit_reads_empty(tmp_path):
    assert job_retry.read_submit(str(tmp_path)) == {}
    assert job_retry.read_submit(None) == {}


def _safe_fixture(tmp_path):
    out, up = tmp_path / "output", tmp_path / "uploads"
    out.mkdir()
    up.mkdir()
    job_id = _uid()
    d = out / job_id
    d.mkdir()
    return out, up, job_id, d


def test_cmd_is_safe_accepts_what_process_builds(tmp_path):
    out, up, job_id, d = _safe_fixture(tmp_path)
    url_cmd = [sys.executable, "-u", "main.py", "-u", URL, "--keep-original", "-o", str(d)]
    assert job_retry.cmd_is_safe(url_cmd, str(out), str(up), job_id)
    assert job_retry.cmd_is_safe(url_cmd + ["--format", "square"], str(out), str(up), job_id)
    src = up / f"{job_id}_talk.mp4"
    up_cmd = [sys.executable, "-u", "main.py", "-i", str(src), "-o", str(d),
              "--transcript", str(d / "source_transcript.json")]
    assert job_retry.cmd_is_safe(up_cmd, str(out), str(up), job_id)
    assert job_retry.source_of(up_cmd) == {"url": None, "input_path": str(src)}
    assert job_retry.source_of(url_cmd) == {"url": URL, "input_path": None}


def test_cmd_is_safe_refuses_every_variant(tmp_path):
    out, up, job_id, d = _safe_fixture(tmp_path)
    other = _uid()
    base = [sys.executable, "-u", "main.py"]
    bad = [
        base + ["-u", URL, "-o", str(d), "--skip-analysis"],               # extra flag
        base + ["-u", URL, "-o", str(out / other)],                        # -o elsewhere
        base + ["-u", URL, "-o", str(tmp_path)],
        base + ["-u", URL],                                                # no -o
        base + ["-i", str(tmp_path / f"{job_id}_x.mp4"), "-o", str(d)],     # -i outside uploads
        base + ["-i", str(up / f"{other}_x.mp4"), "-o", str(d)],           # another job's upload
        base + ["-i", str(up / "sub" / f"{job_id}_x.mp4"), "-o", str(d)],   # not directly inside
        base + ["-i", str(up / f"{job_id}_x.mp4"), "-u", URL, "-o", str(d)],  # both sources
        base + ["-i", str(up / f"{job_id}_x.mp4"), "-o", str(d), "--transcript", "/etc/passwd"],
        base + ["-u", "file:///etc/passwd", "-o", str(d)],
        base + ["-u", URL, "-o", str(d), "--format", "--evil"],
        base + ["-u", URL, "-o", str(d), "-o", str(d)],                    # repeated flag
        base + ["-u", URL, "-o"],                                          # flag without value
        [sys.executable, "-u", "evil.py", "-u", URL, "-o", str(d)],        # wrong script
        ["/bin/sh", "-u", "main.py", "-u", URL, "-o", str(d)],             # not an interpreter
        [sys.executable, "main.py", "-u", URL, "-o", str(d)],
    ]
    for cmd in bad:
        assert not job_retry.cmd_is_safe(cmd, str(out), str(up), job_id), cmd
    good = base + ["-u", URL, "-o", str(d)]
    assert not job_retry.cmd_is_safe(good, str(out), str(up), "../" + job_id)
    assert not job_retry.cmd_is_safe(good, str(out), str(up), other)


# --- pure: failure reasons ----------------------------------------------------
@pytest.mark.parametrize("lines,code", [
    (["❌ Error: GEMINI_API_KEY not found in environment variables (set it, or point "
      "LLM_BASE_URL at an OpenAI-compatible server).",
      "❌ RuntimeError: Clip detection failed — the AI model did not return usable clips "
      "for this video."], "missing_key"),
    (["ERROR: [youtube] abc: Sign in to confirm you're not a bot. Use --cookies",
      "❌ FATAL ERROR: YOUTUBE DOWNLOAD FAILED (all strategies)"], "download_blocked"),
    (["❌ FATAL ERROR: YOUTUBE DOWNLOAD FAILED (all strategies)",
      "Technical Details: ERROR: unable to download video data: HTTP Error 403: Forbidden"],
     "download_blocked"),
    (["❌ This video is private on YouTube. Set it to Unlisted (or Public) and try again, "
      "or upload the file instead."], "video_unavailable"),
    (["ERROR: [youtube] abc: Video unavailable. This video has been removed by the uploader"],
     "video_unavailable"),
    (["ERROR: [youtube] abc: Video unavailable",
      "❌ FATAL ERROR: YOUTUBE DOWNLOAD FAILED (all strategies)"], "video_unavailable"),
    (["❌ RuntimeError: Clip detection failed — the AI model did not return usable clips "
      "for this video."], "no_clips"),
    (["❌ NoAudioError: This video has no audio track. OpenShorts finds viral moments"],
     "no_audio"),
    (["❌ GeminiBlockedError: Gemini blocked this video's content (PROHIBITED_CONTENT)."],
     "content_blocked"),
    (["This video is only 12s long — clip generation needs at least 60s of material to cut "
      "from. It already is short-form content."], "too_short"),
    (["❌ Gemini Error: 503 UNAVAILABLE. {'error': {'code': 503, 'message': 'The model is "
      "overloaded. Please try again later.'}}", "❌ Clip detection failed"], "model_overloaded"),
    (["Error: 429 RESOURCE_EXHAUSTED"], "model_overloaded"),
    (["❌ Clip 1 failed: RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB"],
     "out_of_memory"),
    (["ERROR: Generic error in an external library", "Process failed with exit code 187"],
     "out_of_memory"),
    (["Process failed with exit code -15"], "server_restarted"),
    (["quota_exceeded: The user is out of OpenShorts processing minutes for this period."],
     "quota"),
    (["Process failed with exit code 1"], "unknown"),
    ([], "unknown"),
    # A retired Gemini model: the download worked, the AI call 404'd.
    (["❌ Gemini Error: 404 NOT_FOUND. {'error': {'code': 404, 'message': 'This model "
      "models/gemini-2.5-flash is no longer available to new users. Please update your "
      "code to use a newer model for the latest features and improvements.', "
      "'status': 'NOT_FOUND'}}",
      "❌ RuntimeError: Clip detection failed — the AI model did not return usable clips "
      "for this video."], "model_unavailable"),
    (["❌ Gemini vision error: 404 NOT_FOUND. models/gemini-9-x is not found for API "
      "version v1beta, or is not supported for generateContent."], "model_unavailable"),
    (["❌ Gemini Error: 404 models/gemini-2.5-flash-lite is no longer available"],
     "model_unavailable"),
    # Genuine unavailable videos still say so.
    (["ERROR: [youtube] abc: This video is no longer available because the YouTube "
      "account associated with this video has been terminated."], "video_unavailable"),
    (["ERROR: [youtube] abc: This video is no longer available due to a copyright claim"],
     "video_unavailable"),
])
def test_classify_failure(lines, code):
    got = job_retry.classify_failure(lines)
    assert got["code"] == code
    assert got["message"] == job_retry.MESSAGES[code]
    assert got == job_retry.classify_failure(" ".join(lines))


def test_messages_never_carry_log_text():
    got = job_retry.classify_failure(f"❌ Error: secret /srv/x {URL} GEMINI_API_KEY not found")
    assert URL not in got["message"] and "/srv" not in got["message"]
    assert set(job_retry.FAILURE_CODES) == {
        "missing_key", "download_blocked", "video_unavailable", "no_clips", "no_audio",
        "content_blocked", "too_short", "model_overloaded", "model_unavailable",
        "out_of_memory", "server_restarted", "quota", "unknown"}
    assert "Settings" in job_retry.MESSAGES["model_unavailable"]


RETIRED = ("gemini-2.5-flash", "gemini-2.5-flash-lite")


def test_retired_models_are_not_choices():
    for model in RETIRED:
        assert model not in app_module.GEMINI_MODEL_CHOICES
    assert "gemini-3.1-flash-lite" in app_module.GEMINI_MODEL_CHOICES


@pytest.mark.parametrize("model", RETIRED + ("", "made-up"))
def test_off_list_model_falls_back_to_the_default(model):
    env = {"AUTO_HOOK": "1"}
    applied = job_retry.apply_settings(env, {"GEMINI_MODEL": model, "AUTO_HOOK": "0"},
                                       app_module.GEMINI_MODEL_CHOICES)
    assert applied == ["AUTO_HOOK"] and "GEMINI_MODEL" not in env


def test_failed_sidecar_round_trip_and_validation(tmp_path):
    job_retry.write_failed(str(tmp_path), {"code": "no_clips", "message": "ignored"})
    assert job_retry.read_failed(str(tmp_path)) == job_retry.reason("no_clips")
    (tmp_path / job_retry.FAILED_SIDECAR).write_text(json.dumps(
        {"code": "<script>", "message": "x"}))
    assert job_retry.read_failed(str(tmp_path)) is None
    (tmp_path / job_retry.FAILED_SIDECAR).write_text("garbage")
    assert job_retry.read_failed(str(tmp_path)) is None
    job_retry.write_failed(str(tmp_path), {"code": "made-up"})
    assert job_retry.read_failed(str(tmp_path))["code"] == "unknown"


# --- app: fixtures --------------------------------------------------------------
class _User:
    def __init__(self, uid):
        self.id = uid


def _req(method, path, headers=None):
    async def _do():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://testserver") as client:
            return await client.request(method, path, headers=headers or {})
    return asyncio.run(_do())


def _retry(job_id, headers=None):
    h = {"X-Gemini-Key": "hdr-key"}
    h.update(headers or {})
    return _req("POST", f"/api/jobs/{job_id}/retry", h)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    out = tmp_path / "output"
    up = tmp_path / "uploads"
    (out / "thumbnails").mkdir(parents=True)
    up.mkdir()
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(out))
    monkeypatch.setattr(app_module, "THUMBNAILS_DIR", str(out / "thumbnails"))
    monkeypatch.setattr(app_module, "UPLOAD_DIR", str(up))
    monkeypatch.setattr(app_module, "jobs", {})
    monkeypatch.setattr(app_module, "_running_jobs", set())
    monkeypatch.setattr(app_module, "_retrying", set())
    monkeypatch.setattr(app_module, "BILLING_ENABLED", False)
    monkeypatch.setattr(job_retry, "MANUAL_RETRY_LIMIT", 5)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    calls = {"enqueued": []}
    monkeypatch.setattr(app_module, "_enqueue_job",
                        lambda job_id, priority=2: calls["enqueued"].append((job_id, priority)))

    async def _fine(url):
        return None
    monkeypatch.setattr(app_module, "_validate_source_url", _fine)
    clip_history.invalidate_failed()
    yield types.SimpleNamespace(out=out, up=up, calls=calls)
    clip_history.invalidate_failed()


SETTINGS = {"AUTO_HOOK": "1", "AUTO_HOOK_STYLE": "pill", "CLIP_TARGET_MIN": "3",
            "CLIP_TARGET_MAX": "3", "SPLIT_LAYOUT": "1", "GEMINI_MODEL": "gemini-3.5-flash"}


def _make(env, kind="upload", owner=None, in_memory=True, status="failed", retries=0,
          settings=SETTINGS, partial=None, submit=True, reservation_id=None):
    job_id = _uid()
    d = env.out / job_id
    d.mkdir()
    if kind == "url":
        cmd = [sys.executable, "-u", "main.py", "-u", URL, "--keep-original", "-o", str(d)]
        label = URL
    else:
        src = env.up / f"{job_id}_talk.mp4"
        src.write_bytes(b"src")
        cmd = [sys.executable, "-u", "main.py", "-i", str(src), "-o", str(d)]
        label = "talk.mp4"
    if submit:
        job_retry.write_submit(str(d), cmd=cmd, settings=settings, priority=2,
                               partial=partial, reservation_id=reservation_id)
        if retries:
            raw = json.loads((d / job_retry.SUBMIT_SIDECAR).read_text())
            raw["retries"] = retries
            (d / job_retry.SUBMIT_SIDECAR).write_text(json.dumps(raw))
    creator_brief.write_sidecar(str(d), MARK, source={"kind": kind if kind == "url" else "file",
                                                      "label": label})
    if owner is not None:
        (d / ".owner").write_text(str(owner))
    (d / "V_clip_1.mp4").write_bytes(b"half")
    (d / ".transcript_checkpoint.json").write_text("{}")
    (d / ".brief_applied.json").write_text("{}")
    job_retry.write_failed(str(d), {"code": "missing_key"})
    if in_memory:
        app_module.jobs[job_id] = {
            "status": status, "output_dir": str(d), "user_id": owner, "cmd": cmd, "env": {},
            "logs": app_module._TimedLog(["❌ Error: GEMINI_API_KEY not found."]),
            "reservation_id": reservation_id, "result": None}
    return job_id, d, cmd


# --- app: resume ----------------------------------------------------------------
def test_resume_reapplies_the_job_settings(env):
    job_id, d, cmd = _make(env, in_memory=False)
    (d / app_module._RESUME_FILE).write_text(json.dumps(
        {"cmd": cmd, "priority": 2, "attempts": 0}))
    raw = json.loads((d / job_retry.SUBMIT_SIDECAR).read_text())
    raw["settings"]["GEMINI_API_KEY"] = "smuggled"
    raw["settings"]["CLIP_MAX_SECONDS"] = "9999"
    (d / job_retry.SUBMIT_SIDECAR).write_text(json.dumps(raw))
    app_module._resume_interrupted_jobs()
    job_env = app_module.jobs[job_id]["env"]
    for key, value in SETTINGS.items():
        assert job_env[key] == value, key
    assert job_env.get("GEMINI_API_KEY") != "smuggled"
    assert "CLIP_MAX_SECONDS" not in job_env or job_env["CLIP_MAX_SECONDS"] != "9999"
    assert env.calls["enqueued"] == [(job_id, 2)]


def test_resume_without_sidecar_is_unchanged(env):
    d = env.out / "legacy"
    d.mkdir()
    (d / app_module._RESUME_FILE).write_text(json.dumps(
        {"cmd": ["python", "main.py"], "priority": 1, "attempts": 0}))
    app_module._resume_interrupted_jobs()
    assert app_module.jobs["legacy"]["status"] == "queued"


def test_process_writes_the_submit_sidecar(env, monkeypatch):
    monkeypatch.setattr(app_module, "QUALITY_GATE_MIN_HEIGHT", 0)
    monkeypatch.setattr(app_module, "MIN_SOURCE_SECONDS", 0)
    monkeypatch.setenv("PROXY_URL", "http://u:SECRETP@h:1")

    async def _do():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            return await c.post("/api/process", json={
                "url": URL, "acknowledged": True, "brief": MARK, "layouts": "split",
                "target_clips": 4, "auto_hook_style": "pill", "teaser": False},
                headers={"X-Gemini-Key": "SECRETK", "X-Gemini-Model": "gemini-3.5-flash"})
    r = asyncio.run(_do())
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    d = env.out / job_id
    raw = (d / job_retry.SUBMIT_SIDECAR).read_text()
    for secret in ("SECRETK", "SECRETP", MARK):
        assert secret not in raw
    submit = job_retry.read_submit(str(d))
    assert submit["cmd"] == app_module.jobs[job_id]["cmd"]
    assert submit["settings"]["SPLIT_LAYOUT"] == "1"
    assert submit["settings"]["CLIP_TARGET_MIN"] == "4"
    assert submit["settings"]["AUTO_TEASER"] == "0"
    assert submit["settings"]["GEMINI_MODEL"] == "gemini-3.5-flash"
    assert submit["settings"]["AUTO_HOOK_STYLE"] == "pill"
    assert job_retry.cmd_is_safe(submit["cmd"], str(env.out), str(env.up), job_id)


def test_process_ignores_a_retired_model_header(env, monkeypatch):
    monkeypatch.setattr(app_module, "QUALITY_GATE_MIN_HEIGHT", 0)
    monkeypatch.setattr(app_module, "MIN_SOURCE_SECONDS", 0)

    async def _do():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            return await c.post("/api/process", json={"url": URL, "acknowledged": True},
                                headers={"X-Gemini-Key": "k",
                                         "X-Gemini-Model": "gemini-2.5-flash"})
    r = asyncio.run(_do())
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    assert "GEMINI_MODEL" not in app_module.jobs[job_id]["env"]
    assert "GEMINI_MODEL" not in job_retry.read_submit(str(env.out / job_id))["settings"]


# --- app: wrapper writes the reason --------------------------------------------
def test_terminal_failure_records_its_reason(env, monkeypatch):
    job_id, d, _ = _make(env)
    job_retry.clear_failed(str(d))
    job = app_module.jobs[job_id]
    job["status"] = "processing"
    for name in ("_archive_managed_job", "_notify_job_webhook", "_record_job_alert",
                 "_track_proxy_usage", "_notify_clips_ready", "_notify_clip_activity",
                 "_autopilot_job_finished", "_settle_reservation"):
        async def _noop(*a, **k):
            return None
        monkeypatch.setattr(app_module, name, _noop)
    monkeypatch.setattr(app_module, "_draining", False)
    monkeypatch.setattr(app_module, "_stopping", False)

    async def fake_run_job(jid, j):
        j["status"] = "failed"
    monkeypatch.setattr(app_module, "run_job", fake_run_job)

    async def go():
        sem = asyncio.Semaphore(1)
        await sem.acquire()
        monkeypatch.setattr(app_module, "concurrency_semaphore", sem)
        q = asyncio.PriorityQueue()
        q.put_nowait((2, 0, job_id))
        await q.get()
        monkeypatch.setattr(app_module, "job_queue", q)
        await app_module.run_job_wrapper(job_id)
    asyncio.run(go())
    assert job_retry.read_failed(str(d))["code"] == "missing_key"
    job["status"] = "completed"
    app_module._record_failure_reason(job_id, job)
    assert job_retry.read_failed(str(d)) is None


def test_clean_for_retry_keeps_the_sidecars(env):
    job_id, d, _ = _make(env)
    app_module._clean_for_retry(str(d))
    for keep in (creator_brief.SIDECAR, job_retry.SUBMIT_SIDECAR, ".transcript_checkpoint.json"):
        assert (d / keep).exists(), keep
    assert not (d / "V_clip_1.mp4").exists()


# --- app: retry happy path ------------------------------------------------------
def test_retry_requeues_the_same_job_with_its_settings(env):
    job_id, d, cmd = _make(env)
    brief_before = (d / creator_brief.SIDECAR).read_bytes()
    r = _retry(job_id)
    assert r.status_code == 200, r.text
    assert r.json() == {"job_id": job_id, "status": "queued", "retry": {"used": 1, "limit": 5}}
    job = app_module.jobs[job_id]
    assert job["status"] == "queued" and job["manual_retries"] == 1 and job["auto_retries"] == 0
    assert list(job["logs"]) == ["Retrying your video (attempt 1 of 5)."]
    assert job["env"]["GEMINI_API_KEY"] == "hdr-key"
    for key, value in SETTINGS.items():
        assert job["env"][key] == value, key
    assert job["cmd"][0] == sys.executable and job["cmd"][1:] == cmd[1:]
    assert env.calls["enqueued"] == [(job_id, 2)]
    # brief untouched and nowhere else
    assert (d / creator_brief.SIDECAR).read_bytes() == brief_before
    assert MARK not in json.dumps(job["env"]) and MARK not in json.dumps(job["cmd"])
    manifest = app_module._read_manifest(job_id)
    assert manifest and MARK not in json.dumps(manifest)
    assert manifest["cmd"] == job["cmd"]
    assert "hdr-key" not in json.dumps(manifest)
    assert "hdr-key" not in (d / job_retry.SUBMIT_SIDECAR).read_text()
    # cleaned, counter bumped, stale sidecars gone
    assert not (d / "V_clip_1.mp4").exists()
    assert (d / ".transcript_checkpoint.json").exists()
    assert not (d / job_retry.FAILED_SIDECAR).exists()
    assert not (d / ".brief_applied.json").exists()
    assert job_retry.read_submit(str(d))["retries"] == 1


def test_header_model_wins_over_the_stored_one(env):
    job_id, d, _ = _make(env)
    r = _retry(job_id, {"X-Gemini-Model": "gemini-3.1-flash-lite"})
    assert r.status_code == 200
    assert app_module.jobs[job_id]["env"]["GEMINI_MODEL"] == "gemini-3.1-flash-lite"
    r2 = _make(env)[0]
    assert _retry(r2, {"X-Gemini-Model": "made-up"}).status_code == 200
    assert app_module.jobs[r2]["env"]["GEMINI_MODEL"] == "gemini-3.5-flash"


@pytest.mark.parametrize("dead", RETIRED)
def test_retry_of_a_job_stored_with_a_retired_model_runs_on_the_default(env, dead):
    job_id, d, _ = _make(env, settings=dict(SETTINGS, GEMINI_MODEL=dead))
    job_retry.write_failed(str(d), {"code": "model_unavailable"})
    # the browser may still send the same dead choice
    r = _retry(job_id, {"X-Gemini-Model": dead})
    assert r.status_code == 200, r.text
    job_env = app_module.jobs[job_id]["env"]
    assert "GEMINI_MODEL" not in job_env          # main.py then uses its default
    assert job_env["AUTO_HOOK_STYLE"] == "pill"   # the other settings still apply
    assert env.calls["enqueued"] == [(job_id, 2)]


def test_resume_drops_a_stored_retired_model(env):
    job_id, d, cmd = _make(env, in_memory=False,
                           settings=dict(SETTINGS, GEMINI_MODEL="gemini-2.5-flash"))
    (d / app_module._RESUME_FILE).write_text(json.dumps(
        {"cmd": cmd, "priority": 2, "attempts": 0}))
    app_module._resume_interrupted_jobs()
    assert "GEMINI_MODEL" not in app_module.jobs[job_id]["env"]


def test_url_job_retry(env):
    job_id, d, cmd = _make(env, kind="url")
    assert _retry(job_id).status_code == 200
    assert app_module.jobs[job_id]["cmd"][1:] == cmd[1:]


def test_missing_key_is_refused_and_changes_nothing(env):
    job_id, d, _ = _make(env)
    r = _req("POST", f"/api/jobs/{job_id}/retry")
    assert r.status_code == 400
    assert app_module.jobs[job_id]["status"] == "failed"
    assert (d / "V_clip_1.mp4").exists() and job_retry.read_submit(str(d))["retries"] == 0


def test_sixth_retry_is_429(env):
    job_id, d, _ = _make(env)
    for n in range(1, 6):
        r = _retry(job_id)
        assert r.status_code == 200 and r.json()["retry"]["used"] == n
        # the run fails again: terminal state, the wrapper drops the manifest
        app_module.jobs[job_id]["status"] = "failed"
        app_module._clear_resume_manifest(job_id)
    r = _retry(job_id)
    assert r.status_code == 429 and r.json()["detail"]["code"] == "limit"
    assert len(env.calls["enqueued"]) == 5
    assert job_retry.read_submit(str(d))["retries"] == 5


def test_double_submit_enqueues_once(env, monkeypatch):
    job_id, d, _ = _make(env)
    real = app_module.reserve_process_minutes

    async def slow(*a, **k):
        await asyncio.sleep(0.05)
        return await real(*a, **k)
    monkeypatch.setattr(app_module, "reserve_process_minutes", slow)

    async def _both():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            return await asyncio.gather(*[
                c.post(f"/api/jobs/{job_id}/retry", headers={"X-Gemini-Key": "k"})
                for _ in range(2)])
    results = asyncio.run(_both())
    assert sorted(r.status_code for r in results) == [200, 409]
    assert len(env.calls["enqueued"]) == 1
    assert job_retry.read_submit(str(d))["retries"] == 1


@pytest.mark.parametrize("status", ["queued", "processing", "completed"])
def test_running_or_completed_is_409(env, status):
    job_id, d, _ = _make(env, status=status)
    r = _retry(job_id)
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == ("completed" if status == "completed" else "running")
    assert env.calls["enqueued"] == [] and (d / "V_clip_1.mp4").exists()


def test_failed_but_still_running_here_is_409(env):
    job_id, d, _ = _make(env)
    app_module._running_jobs.add(job_id)
    assert _retry(job_id).status_code == 409
    assert env.calls["enqueued"] == []


@pytest.mark.parametrize("bad", ["not-a-uuid", "..", "..%2fsibling", "%2e%2e%2fsibling",
                                 "A" * 36, str(uuid.uuid4()).upper()])
def test_bad_or_unknown_ids_are_404_and_touch_nothing(env, bad):
    job_id, d, _ = _make(env)
    sibling = env.out / "sibling"
    sibling.mkdir()
    (sibling / "keep.txt").write_text("k")
    assert _retry(bad).status_code in (404, 405)
    assert _retry(str(uuid.uuid4())).status_code == 404
    assert (sibling / "keep.txt").exists() and (d / "V_clip_1.mp4").exists()
    assert env.calls["enqueued"] == []


def test_no_sidecar_or_unsafe_argv_is_not_retryable(env):
    job_id, d, _ = _make(env, submit=False)
    r = _retry(job_id)
    assert r.status_code == 409
    assert r.json()["detail"] == {"code": "not_retryable",
                                  "message": "Please submit this video again."}
    job2, d2, cmd2 = _make(env)
    raw = json.loads((d2 / job_retry.SUBMIT_SIDECAR).read_text())
    raw["cmd"] = [sys.executable, "-u", "main.py", "-u", URL, "-o", str(env.out)]
    (d2 / job_retry.SUBMIT_SIDECAR).write_text(json.dumps(raw))
    assert _retry(job2).json()["detail"]["code"] == "not_retryable"
    assert env.calls["enqueued"] == []


def test_missing_upload_is_source_missing(env):
    job_id, d, _ = _make(env)
    os.remove(env.up / f"{job_id}_talk.mp4")
    r = _retry(job_id)
    assert r.status_code == 409
    assert r.json()["detail"] == {
        "code": "source_missing",
        "message": "The original upload is no longer on the server. Please upload the video again."}
    assert app_module.jobs[job_id]["status"] == "failed" and env.calls["enqueued"] == []


# --- app: lost jobs and /api/status -----------------------------------------------
def test_lost_job_reports_failed_and_is_retryable(env):
    job_id, d, _ = _make(env, in_memory=False)
    job_retry.clear_failed(str(d))
    st = _req("GET", f"/api/status/{job_id}").json()
    assert st["status"] == "failed" and st["result"] is None
    assert st["error"] == job_retry.reason("server_restarted")
    assert st["retry"] == {"allowed": True, "used": 0, "limit": 5, "reason": None}
    assert job_id not in app_module.jobs
    assert _retry(job_id).status_code == 200
    assert app_module.jobs[job_id]["status"] == "queued"
    st = _req("GET", f"/api/status/{job_id}").json()
    assert st["status"] == "queued" and st["error"] is None and st["retry"] is None


def test_lost_job_keeps_its_recorded_reason(env):
    job_id, d, _ = _make(env, in_memory=False)
    job_retry.write_failed(str(d), {"code": "no_clips"})
    st = _req("GET", f"/api/status/{job_id}").json()
    assert st["error"]["code"] == "no_clips"


def test_dir_without_submit_sidecar_is_still_404(env):
    d = env.out / str(uuid.uuid4())
    d.mkdir()
    assert _req("GET", f"/api/status/{d.name}").status_code == 404


def test_status_of_a_failed_memory_job(env):
    job_id, d, _ = _make(env)
    st = _req("GET", f"/api/status/{job_id}").json()
    assert st["error"]["code"] == "missing_key"
    assert st["retry"]["allowed"] is True
    os.remove(env.up / f"{job_id}_talk.mp4")
    st = _req("GET", f"/api/status/{job_id}").json()
    assert st["retry"]["allowed"] is False and "upload" in st["retry"]["reason"]


def test_history_lists_the_failed_job(env):
    job_id, d, _ = _make(env)
    lost, _, _ = _make(env, in_memory=False)
    body = _req("GET", "/api/clip-history").json()
    by = {e["job_id"]: e for e in body["failed"]}
    assert set(by) == {job_id, lost}
    assert by[job_id]["error"]["code"] == "missing_key"
    assert by[job_id]["title"] == "talk.mp4" and by[job_id]["retry"]["allowed"] is True
    assert by[lost]["state"] == "failed"   # its reason was recorded before the restart
    assert MARK not in json.dumps(body["failed"])


# --- app: cloud -------------------------------------------------------------------
@pytest.fixture()
def cloud(env, monkeypatch):
    monkeypatch.setattr(app_module, "BILLING_ENABLED", True)
    released, reserved = [], []

    async def fake_user(request):
        auth = request.headers.get("authorization", "")
        return _User(auth.split()[-1]) if auth else None
    monkeypatch.setattr(app_module, "_user_from_request", fake_user)

    async def fake_key(request):
        return "managed-key"
    monkeypatch.setattr(app_module, "resolve_gemini", fake_key)

    async def fake_release(rid):
        released.append(rid)
    monkeypatch.setattr(app_module, "cloud", types.SimpleNamespace(
        metering=types.SimpleNamespace(release_reservation=fake_release)))

    async def fake_reserve(request, url, input_path, job_id, max_minutes=None,
                           source_range=None):
        reserved.append({"job_id": job_id, "max_minutes": max_minutes,
                         "source_range": source_range})
        user = await fake_user(request)
        partial = None
        if max_minutes:
            partial = {"processed_minutes": max_minutes, "total_minutes": 20}
        return user.id, 1, f"res-{len(reserved)}", "free", partial
    monkeypatch.setattr(app_module, "reserve_process_minutes", fake_reserve)
    env.released, env.reserved = released, reserved
    return env


def _auth(uid):
    return {"Authorization": f"Bearer {uid}"}


def test_cloud_owner_isolation(cloud):
    job_id, d, _ = _make(cloud, owner="alice")
    assert _retry(job_id).status_code == 401
    assert _retry(job_id, _auth("bob")).status_code == 404
    assert cloud.calls["enqueued"] == [] and (d / "V_clip_1.mp4").exists()
    r = _retry(job_id, _auth("alice"))
    assert r.status_code == 200, r.text
    job = app_module.jobs[job_id]
    assert job["user_id"] == "alice" and job["reservation_id"] == "res-1"
    assert job["env"]["GEMINI_API_KEY"] == "managed-key"
    assert job["env"]["WATERMARK"] == "1"                        # free plan now
    assert cloud.calls["enqueued"] == [(job_id, 1)]
    assert app_module._read_manifest(job_id)["reservation_id"] == "res-1"
    assert job_retry.read_submit(str(d))["reservation_id"] == "res-1"


def test_cloud_ownerless_is_404(cloud):
    job_id, d, _ = _make(cloud, owner=None)
    lost, _, _ = _make(cloud, owner=None, in_memory=False)
    for jid in (job_id, lost):
        for uid in ("alice", "bob"):
            assert _retry(jid, _auth(uid)).status_code == 404
    assert cloud.calls["enqueued"] == []


def test_cloud_lost_job_owner_from_marker(cloud):
    job_id, d, _ = _make(cloud, owner="alice", in_memory=False)
    assert _retry(job_id, _auth("bob")).status_code == 404
    assert _retry(job_id, _auth("alice")).status_code == 200


def test_cloud_stale_reservation_released_before_a_new_one(cloud):
    job_id, d, _ = _make(cloud, owner="alice", reservation_id="old-res")
    assert _retry(job_id, _auth("alice")).status_code == 200
    assert cloud.released == ["old-res"]
    assert app_module.jobs[job_id]["reservation_id"] == "res-1"


def test_cloud_partial_travels_with_the_retry(cloud):
    job_id, d, _ = _make(cloud, owner="alice",
                         partial={"processed_minutes": 5, "total_minutes": 20})
    assert _retry(job_id, _auth("alice")).status_code == 200
    assert cloud.reserved[-1]["max_minutes"] == 5 and cloud.reserved[-1]["source_range"] is None
    assert app_module.jobs[job_id]["env"]["MAX_SOURCE_MINUTES"] == "5"
    chosen, _, _ = _make(cloud, owner="alice", partial={
        "processed_minutes": 3, "total_minutes": 20, "start_seconds": 60.0,
        "end_seconds": 240.0, "chosen": True})
    assert _retry(chosen, _auth("alice")).status_code == 200
    assert cloud.reserved[-1]["source_range"] == (60.0, 240.0)


def test_cloud_quota_refusal_leaves_the_job_as_it_was(cloud, monkeypatch):
    job_id, d, _ = _make(cloud, owner="alice")

    async def refuse(*a, **k):
        raise HTTPException(status_code=402, detail={"error": "quota_exceeded"})
    monkeypatch.setattr(app_module, "reserve_process_minutes", refuse)
    r = _retry(job_id, _auth("alice"))
    assert r.status_code == 402 and r.json()["detail"]["error"] == "quota_exceeded"
    assert app_module.jobs[job_id]["status"] == "failed"
    assert job_retry.read_submit(str(d))["retries"] == 0
    assert (d / "V_clip_1.mp4").exists() and (d / job_retry.FAILED_SIDECAR).exists()
    assert (d / ".brief_applied.json").exists()
    assert app_module._read_manifest(job_id) is None
    assert cloud.calls["enqueued"] == []


def test_cloud_history_failed_is_scoped(cloud):
    mine, _, _ = _make(cloud, owner="alice")
    theirs, _, _ = _make(cloud, owner="bob", in_memory=False)
    body = _req("GET", "/api/clip-history", _auth("alice")).json()
    assert [e["job_id"] for e in body["failed"]] == [mine]
