"""POST /api/process with a creator brief: validation, the .brief.json sidecar,
and the rule that the text rides nowhere else (env, argv, resume manifest)."""
import asyncio
import json
import os

import httpx
import pytest

import brief_presets
import creator_brief

app_module = pytest.importorskip("app")

MARK = "ZEBRA-RULE-7731"


def _post(json_body):
    async def _do():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://testserver") as client:
            return await client.post("/api/process", json=json_body,
                                     headers={"X-Gemini-Key": "test-key"})
    return asyncio.run(_do())


@pytest.fixture()
def dirs(tmp_path, monkeypatch):
    out_root = tmp_path / "output"
    up_root = tmp_path / "uploads"
    out_root.mkdir()
    up_root.mkdir()
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(out_root))
    monkeypatch.setattr(app_module, "UPLOAD_DIR", str(up_root))
    saved = dict(brief_presets._cfg)
    brief_presets.configure(output_dir=str(out_root), billing_enabled=False,
                            user_resolver=None)
    video = up_root / "thumb_sess_source.mp4"
    video.write_bytes(b"fake-video-bytes")
    monkeypatch.setitem(app_module.thumbnail_sessions, "sessb", {
        "user_id": None, "video_path": str(video),
        "transcript_ready": False, "transcript": None})
    yield out_root
    brief_presets._cfg.update(saved)


def _submit(**extra):
    body = {"thumbnail_session_id": "sessb", "acknowledged": True}
    body.update(extra)
    return _post(body)


def _ok(resp):
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]
    return job_id, app_module.jobs[job_id]


def test_too_long_brief_is_400_and_leaves_no_job_dir(dirs):
    resp = _submit(brief="x" * (creator_brief.MAX_BRIEF_CHARS + 1))
    assert resp.status_code == 400
    assert str(creator_brief.MAX_BRIEF_CHARS) in resp.text
    assert os.listdir(dirs) == []


def test_non_string_brief_is_400(dirs):
    assert _submit(brief=["a"]).status_code == 400
    assert _submit(brief=5).status_code == 400
    assert os.listdir(dirs) == []


def test_brief_lands_sanitized_in_the_sidecar(dirs):
    resp = _submit(brief=f"  {MARK}\x00 keep it short‮  ")
    job_id, job = _ok(resp)
    side = creator_brief.read_sidecar(str(dirs / job_id))
    assert side["brief"] == creator_brief.sanitize(f"  {MARK}\x00 keep it short‮  ")
    assert MARK in side["brief"] and "\x00" not in side["brief"]
    assert side["preset_name"] is None
    assert side["source"]["kind"] == "thumbnail_session"


def test_absent_brief_writes_an_empty_sidecar(dirs):
    job_id, _ = _ok(_submit())
    side = creator_brief.read_sidecar(str(dirs / job_id))
    assert side["brief"] == ""
    assert os.path.isfile(dirs / job_id / creator_brief.SIDECAR)


def test_url_job_records_the_url_as_label(dirs, monkeypatch):
    async def _fine(url):
        return None
    monkeypatch.setattr(app_module, "_validate_source_url", _fine)
    monkeypatch.setattr(app_module, "QUALITY_GATE_MIN_HEIGHT", 0)
    monkeypatch.setattr(app_module, "MIN_SOURCE_SECONDS", 0)
    job_id, _ = _ok(_post({"url": "https://example.com/watch?v=abc", "acknowledged": True,
                           "brief": MARK}))
    side = creator_brief.read_sidecar(str(dirs / job_id))
    assert side["source"] == {"kind": "url", "label": "https://example.com/watch?v=abc"}


def test_brief_never_in_env_cmd_or_manifest(dirs):
    job_id, job = _ok(_submit(brief=MARK))
    assert MARK not in json.dumps(job["env"], default=str)
    assert MARK not in json.dumps(job["cmd"], default=str)
    for name in os.listdir(dirs / job_id):
        if name == creator_brief.SIDECAR:
            continue
        path = dirs / job_id / name
        if path.is_file():
            assert MARK.encode() not in path.read_bytes(), name
    manifest = app_module._read_manifest(job_id)
    assert manifest is not None
    assert MARK not in json.dumps(manifest, default=str)
    # and no other field of the job record carries it
    assert MARK not in json.dumps(
        {k: v for k, v in job.items() if k not in ("logs",)}, default=str)
    assert MARK not in "".join(str(x) for x in job["logs"])


def test_preset_supplies_text_and_name(dirs):
    async def _mk():
        return await brief_presets._file_store.create(None, "Camp X", MARK + " preset text")
    preset = asyncio.run(_mk())
    job_id, _ = _ok(_submit(brief_preset_id=preset["id"]))
    side = creator_brief.read_sidecar(str(dirs / job_id))
    assert MARK in side["brief"] and side["preset_name"] == "Camp X"


def test_explicit_brief_wins_over_preset_text(dirs):
    async def _mk():
        return await brief_presets._file_store.create(None, "Camp Y", "preset text")
    preset = asyncio.run(_mk())
    job_id, _ = _ok(_submit(brief="my own text", brief_preset_id=preset["id"]))
    side = creator_brief.read_sidecar(str(dirs / job_id))
    assert side["brief"] == "my own text" and side["preset_name"] == "Camp Y"


def test_unknown_or_malformed_preset_is_404(dirs):
    assert _submit(brief_preset_id="0" * 32).status_code == 404
    assert _submit(brief_preset_id="../etc").status_code == 404
    assert os.listdir(dirs) == [] or all(n.startswith(".") for n in os.listdir(dirs))


def test_sidecar_failure_does_not_fail_the_submit(dirs, monkeypatch):
    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(creator_brief, "write_sidecar", boom)
    resp = _submit(brief=MARK)
    assert resp.status_code == 200


def test_auto_hook_defaults_on_and_only_explicit_false_turns_it_off(dirs):
    _, job = _ok(_submit())
    assert job["env"]["AUTO_HOOK"] == "1"
    for off in (False, "false", "0", "no"):
        _, job = _ok(_submit(auto_hook=off))
        assert "AUTO_HOOK" not in job["env"], off
    _, job = _ok(_submit(auto_hook=True, auto_hook_style="yellow"))
    assert job["env"]["AUTO_HOOK"] == "1" and job["env"]["AUTO_HOOK_STYLE"] == "yellow"


def test_teaser_default_and_opt_out(dirs):
    _, job = _ok(_submit())
    assert "AUTO_TEASER" not in job["env"]
    _, job = _ok(_submit(teaser=True))
    assert "AUTO_TEASER" not in job["env"]
    for off in (False, "false", "0", "no"):
        _, job = _ok(_submit(teaser=off))
        assert job["env"]["AUTO_TEASER"] == "0", off
