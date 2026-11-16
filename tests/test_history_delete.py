"""DELETE /api/clip-history/...: owner-scoped, traversal-safe, tombstoning."""
import asyncio
import io
import json
import os
import uuid
import zipfile

import httpx
import pytest

app_module = pytest.importorskip("app")

BASE = "My Video"


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
    monkeypatch.setattr(app_module, "BILLING_ENABLED", False)
    return out, up


def _make_job(env, n_clips=3, user_id=None, status="completed"):
    out, up = env
    job_id = str(uuid.uuid4())
    d = out / job_id
    d.mkdir()
    shorts = []
    for i in range(n_clips):
        name = f"{BASE}_clip_{i + 1}.mp4"
        (d / name).write_bytes(b"clean")
        (d / f"{name}.layout.json").write_text("{}")
        shorts.append({"video_title_for_youtube_short": f"t{i}",
                       "video_url": f"/videos/{job_id}/{name}"})
    (d / f"{BASE}_metadata.json").write_text(json.dumps({"shorts": shorts}))
    (up / f"{job_id}_source.mp4").write_bytes(b"src")
    (up / "other_source.mp4").write_bytes(b"src")
    app_module.jobs[job_id] = {
        "status": status, "output_dir": str(d), "user_id": user_id,
        "logs": app_module._TimedLog([]),
        "result": {"clips": json.loads(json.dumps(shorts))}}
    return job_id, d


def _shorts(d):
    return json.loads((d / f"{BASE}_metadata.json").read_text())["shorts"]


def test_delete_whole_video(env):
    out, up = env
    job_id, d = _make_job(env)
    sibling = out / "sibling"
    sibling.mkdir()
    r = _req("DELETE", f"/api/clip-history/{job_id}")
    assert r.status_code == 200
    assert r.json() == {"success": True, "deleted": "job", "job_deleted": True}
    assert not d.exists() and job_id not in app_module.jobs
    assert not (up / f"{job_id}_source.mp4").exists()
    assert (up / "other_source.mp4").exists() and sibling.is_dir()
    assert (out / "thumbnails").is_dir()


def test_delete_one_clip_tombstones_and_removes_its_files(env):
    job_id, d = _make_job(env)
    for name in (f"subtitled_111_{BASE}_clip_2.mp4", f"hooked_222_{BASE}_clip_2.mp4",
                 f"recut_333_{BASE}_clip_2.mp4", f"hook_{BASE}_clip_2.mp4",
                 f"wm_{BASE}_clip_2.mp4", f"wm_subtitled_111_{BASE}_clip_2.mp4"):
        (d / name).write_bytes(b"d")
        (d / f"{name}.layout.json").write_text("{}")
    (d / f".hookrecipe_hooked_222_{BASE}_clip_2.mp4.json").write_text("{}")
    (d / f".hookrecipe_hooked_222_{BASE}_clip_2.mp4.png").write_bytes(b"p")
    r = _req("DELETE", f"/api/clip-history/{job_id}/clips/1")
    assert r.status_code == 200
    assert r.json() == {"success": True, "deleted": "clip", "job_deleted": False}
    left = sorted(os.listdir(d))
    assert not [n for n in left if "_clip_2" in n], left
    assert f"{BASE}_clip_1.mp4" in left and f"{BASE}_clip_3.mp4" in left
    assert f"{BASE}_clip_1.mp4.layout.json" in left
    shorts = _shorts(d)
    assert len(shorts) == 3
    assert shorts[1]["deleted"] is True and "video_url" not in shorts[1]
    assert "deleted" not in shorts[0] and "video_url" in shorts[2]
    mem = app_module.jobs[job_id]["result"]["clips"]
    assert mem[1]["deleted"] is True and "video_url" not in mem[1]
    assert len(mem) == 3
    st = _req("GET", f"/api/status/{job_id}").json()
    assert len(st["result"]["clips"]) == 3 and st["result"]["clips"][1]["deleted"]


def test_clip_1_never_touches_clip_11(env):
    job_id, d = _make_job(env, n_clips=12)
    (d / f"subtitled_9_{BASE}_clip_11.mp4").write_bytes(b"x")
    (d / f"wm_{BASE}_clip_11.mp4").write_bytes(b"x")
    (d / f"subtitled_9_{BASE}_clip_1.mp4").write_bytes(b"x")
    assert _req("DELETE", f"/api/clip-history/{job_id}/clips/0").status_code == 200
    left = os.listdir(d)
    assert f"{BASE}_clip_1.mp4" not in left and f"subtitled_9_{BASE}_clip_1.mp4" not in left
    for keep in (f"{BASE}_clip_11.mp4", f"subtitled_9_{BASE}_clip_11.mp4",
                 f"wm_{BASE}_clip_11.mp4", f"{BASE}_clip_12.mp4"):
        assert keep in left, keep
    # and the other way round: deleting clip 11 leaves clip 12 and clip 2
    assert _req("DELETE", f"/api/clip-history/{job_id}/clips/10").status_code == 200
    left = os.listdir(d)
    assert f"{BASE}_clip_11.mp4" not in left
    assert f"{BASE}_clip_12.mp4" in left and f"{BASE}_clip_2.mp4" in left


@pytest.mark.parametrize("bad", ["..", "../x", "not-a-uuid", "%2e%2e", "..%2fsibling",
                                 "%2e%2e%2fsibling", "A" * 36])
def test_bad_ids_are_404_and_touch_nothing(env, bad):
    out, up = env
    job_id, d = _make_job(env)
    sibling = out / "sibling"
    sibling.mkdir()
    (sibling / "keep.txt").write_text("k")
    for path in (f"/api/clip-history/{bad}", f"/api/clip-history/{bad}/clips/0"):
        assert _req("DELETE", path).status_code in (404, 405)
    assert (sibling / "keep.txt").exists() and d.is_dir()
    assert (up / f"{job_id}_source.mp4").exists()


def test_unknown_uuid_is_404(env):
    assert _req("DELETE", f"/api/clip-history/{uuid.uuid4()}").status_code == 404


@pytest.mark.parametrize("idx", ["-1", "3", "99", "abc"])
def test_clip_index_out_of_range(env, idx):
    job_id, d = _make_job(env)
    r = _req("DELETE", f"/api/clip-history/{job_id}/clips/{idx}")
    assert r.status_code in (404, 422)
    assert all("deleted" not in s for s in _shorts(d))


def test_already_deleted_clip_is_404(env):
    job_id, d = _make_job(env)
    assert _req("DELETE", f"/api/clip-history/{job_id}/clips/0").status_code == 200
    assert _req("DELETE", f"/api/clip-history/{job_id}/clips/0").status_code == 404


def test_active_or_unfinished_job_is_409(env):
    job_id, d = _make_job(env)
    app_module._running_jobs.add(job_id)
    try:
        assert _req("DELETE", f"/api/clip-history/{job_id}").status_code == 409
        assert _req("DELETE", f"/api/clip-history/{job_id}/clips/0").status_code == 409
    finally:
        app_module._running_jobs.discard(job_id)
    assert d.is_dir()
    other, od = _make_job(env, status="failed")
    assert _req("DELETE", f"/api/clip-history/{other}").status_code == 409
    queued, qd = _make_job(env, status="queued")
    assert _req("DELETE", f"/api/clip-history/{queued}").status_code == 409
    assert od.is_dir() and qd.is_dir()


def test_last_live_clip_deletes_the_job(env):
    out, up = env
    job_id, d = _make_job(env, n_clips=2)
    assert _req("DELETE", f"/api/clip-history/{job_id}/clips/0").json()["job_deleted"] is False
    r = _req("DELETE", f"/api/clip-history/{job_id}/clips/1")
    assert r.json() == {"success": True, "deleted": "clip", "job_deleted": True}
    assert not d.exists() and job_id not in app_module.jobs
    assert not (up / f"{job_id}_source.mp4").exists()


def test_tombstone_survives_recovery_from_disk(env):
    job_id, d = _make_job(env)
    assert _req("DELETE", f"/api/clip-history/{job_id}/clips/1").status_code == 200
    app_module.jobs.pop(job_id)
    app_module._recover_jobs_from_disk()
    clips = app_module.jobs[job_id]["result"]["clips"]
    assert len(clips) == 3
    assert clips[1]["deleted"] is True and "video_url" not in clips[1]
    assert clips[0]["video_url"].endswith("_clip_1.mp4")


def test_download_all_skips_tombstones(env):
    job_id, d = _make_job(env)
    assert _req("DELETE", f"/api/clip-history/{job_id}/clips/1").status_code == 200
    r = _req("GET", f"/api/jobs/{job_id}/download-all")
    assert r.status_code == 200
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert len(names) == 2 and not any("_clip_2" in n for n in names)


def test_history_skips_deleted_clips(env):
    job_id, d = _make_job(env, n_clips=2)
    assert _req("DELETE", f"/api/clip-history/{job_id}/clips/0").status_code == 200
    page = _req("GET", "/api/clip-history").json()
    assert [c["index"] for c in page["sources"][0]["clips"]] == [1]


# --- cloud: owner scoping ---------------------------------------------------
@pytest.fixture()
def cloud(env, monkeypatch):
    monkeypatch.setattr(app_module, "BILLING_ENABLED", True)

    async def fake_user(request):
        auth = request.headers.get("authorization", "")
        return _User(auth.split()[-1]) if auth else None
    monkeypatch.setattr(app_module, "_user_from_request", fake_user)
    return env


def _auth(uid):
    return {"Authorization": f"Bearer {uid}"}


def test_cloud_owner_can_delete_and_other_user_cannot(cloud):
    job_id, d = _make_job(cloud, user_id="alice")
    b = _auth("bob")
    assert _req("DELETE", f"/api/clip-history/{job_id}", b).status_code == 404
    assert _req("DELETE", f"/api/clip-history/{job_id}/clips/0", b).status_code == 404
    assert d.is_dir() and all("deleted" not in s for s in _shorts(d))
    a = _auth("alice")
    assert _req("DELETE", f"/api/clip-history/{job_id}/clips/0", a).status_code == 200
    assert _req("DELETE", f"/api/clip-history/{job_id}", a).status_code == 200
    assert not d.exists()


def test_cloud_signed_out_is_401_and_ownerless_is_404(cloud):
    job_id, d = _make_job(cloud, user_id="alice")
    assert _req("DELETE", f"/api/clip-history/{job_id}").status_code == 401
    ownerless, od = _make_job(cloud, user_id=None)
    for uid in ("alice", "bob"):
        assert _req("DELETE", f"/api/clip-history/{ownerless}", _auth(uid)).status_code == 404
        assert _req("DELETE", f"/api/clip-history/{ownerless}/clips/0", _auth(uid)).status_code == 404
    assert d.is_dir() and od.is_dir()
