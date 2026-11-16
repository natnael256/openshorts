"""Clip history: pure grouping/ordering/scoping logic over fake job records,
plus one round-trip through the self-host endpoint."""
import asyncio
import json
import os
import uuid

import httpx
import pytest

import clip_history
import creator_brief


@pytest.fixture(autouse=True)
def _fresh_cache():
    clip_history._created_cache.clear()
    yield
    clip_history._created_cache.clear()


def _clip(job_id, n=0, **extra):
    c = {"video_title_for_youtube_short": f"title {n}", "viral_hook_text": f"hook {n}",
         "start": 10.0 + n, "end": 40.0 + n, "video_url": f"/videos/{job_id}/clip_{n}.mp4"}
    c.update(extra)
    return c


def _job(tmp_path, job_id, mtime, clips=None, status="completed", user_id=None,
         sidecar=None, metadata=True):
    d = tmp_path / job_id
    d.mkdir()
    if metadata:
        (d / "My Video_metadata.json").write_text("{}", encoding="utf-8")
    if sidecar is not None:
        creator_brief.write_sidecar(str(d), sidecar.get("brief", ""),
                                    preset_name=sidecar.get("preset"),
                                    source=sidecar.get("source"))
    os.utime(d, (mtime, mtime))
    return {"status": status, "output_dir": str(d), "user_id": user_id,
            "result": {"clips": clips if clips is not None else [_clip(job_id)]}}


def test_groups_clips_under_their_source(tmp_path):
    job = _job(tmp_path, "j1", 1000,
               clips=[_clip("j1", 0), _clip("j1", 1)],
               sidecar={"brief": "No swearing", "preset": "Camp",
                        "source": {"kind": "url", "label": "https://example.com/v"}})
    page = clip_history.build({"j1": job}, sign=lambda rel: "http://x" + rel)
    assert page["total"] == 1 and page["limit"] == 20 and page["offset"] == 0
    src = page["sources"][0]
    assert src["job_id"] == "j1"
    assert src["title"] == "My Video"
    assert src["brief"] == "No swearing" and src["preset_name"] == "Camp"
    assert src["source"] == {"kind": "url", "label": "https://example.com/v"}
    assert [c["index"] for c in src["clips"]] == [0, 1]
    c0 = src["clips"][0]
    assert c0["title"] == "title 0" and c0["hook"] == "hook 0"
    assert c0["duration"] == 30.0
    assert c0["video_url"] == "/videos/j1/clip_0.mp4"
    assert c0["download_url"] == "http://x/videos/j1/clip_0.mp4"
    assert c0["brief_check"] is None  # no brief on the clip: nothing was checked
    assert clip_history.build({"j1": job}, retention_seconds=86400)["retention_seconds"] == 86400


def test_newest_first_and_pagination(tmp_path):
    jobs = {f"j{i}": _job(tmp_path, f"j{i}", 1000 + i) for i in range(5)}
    page = clip_history.build(jobs, limit=2, offset=0)
    assert [s["job_id"] for s in page["sources"]] == ["j4", "j3"]
    assert page["total"] == 5
    page = clip_history.build(jobs, limit=2, offset=4)
    assert [s["job_id"] for s in page["sources"]] == ["j0"]
    assert clip_history.build(jobs, limit=2, offset=99)["sources"] == []


def test_page_clamps(tmp_path):
    jobs = {"j": _job(tmp_path, "j", 1)}
    assert clip_history.build(jobs, limit=0)["limit"] == 1
    assert clip_history.build(jobs, limit=-5)["limit"] == 1
    assert clip_history.build(jobs, limit=10_000)["limit"] == 50
    assert clip_history.build(jobs, limit="abc")["limit"] == 20
    assert clip_history.build(jobs, offset=-3)["offset"] == 0
    assert clip_history.build(jobs, offset="x")["offset"] == 0


def test_owner_isolation(tmp_path):
    jobs = {
        "a": _job(tmp_path, "a", 3, user_id="user-a"),
        "b": _job(tmp_path, "b", 2, user_id="user-b"),
        "none": _job(tmp_path, "none", 1, user_id=None),
    }
    mine = clip_history.build(jobs, owner="user-a", scoped=True)
    assert [s["job_id"] for s in mine["sources"]] == ["a"] and mine["total"] == 1
    # an owner of None never sees ownerless jobs either
    assert clip_history.build(jobs, owner=None, scoped=True)["sources"] == []
    # self-host (unscoped) lists everything
    assert clip_history.build(jobs)["total"] == 3


def test_owner_compared_as_strings(tmp_path):
    uid = uuid.uuid4()
    jobs = {"a": _job(tmp_path, "a", 1, user_id=str(uid))}
    assert clip_history.build(jobs, owner=uid, scoped=True)["total"] == 1


def test_field_whitelist(tmp_path):
    secret = {"transcript": "SECRET WORDS", "words": [1], "path": "/srv/x",
              "brief_check": {"verdict": "pass", "violated": [], "reason": "ok",
                              "transcript": "SECRET WORDS"}}
    jobs = {"j": _job(tmp_path, "j", 1, clips=[_clip("j", 0, **secret)])}
    page = clip_history.build(jobs)
    dumped = json.dumps(page)
    assert "SECRET WORDS" not in dumped and "/srv/x" not in dumped
    clip = page["sources"][0]["clips"][0]
    assert set(clip) == {"index", "title", "hook", "start", "end", "duration",
                         "video_url", "download_url", "brief_check", "hook_score",
                         "hook_type", "teaser"}
    assert set(clip["brief_check"]) == {"verdict", "violated", "reason"}
    assert clip["brief_check"]["verdict"] == "pass"


@pytest.mark.parametrize("bad", [
    "pass", ["pass"], {"verdict": "great"}, {"verdict": "flag", "violated": "x"},
    {"verdict": "flag", "violated": [1]}, {"verdict": "flag", "violated": [], "reason": 3},
    {"verdict": None},
])
def test_malformed_brief_check_is_dropped(bad):
    assert clip_history.clean_brief_check(bad) == {
        "verdict": "unchecked", "violated": [], "reason": ""}


def test_brief_check_is_bounded():
    out = clip_history.clean_brief_check(
        {"verdict": "reject", "violated": ["r" * 500] * 30, "reason": "x" * 900})
    assert len(out["reason"]) == 200
    assert len(out["violated"]) == 10 and all(len(v) <= 120 for v in out["violated"])


def test_only_completed_jobs_with_clips(tmp_path):
    jobs = {
        "done": _job(tmp_path, "done", 5),
        "running": _job(tmp_path, "running", 4, status="processing"),
        "failed": _job(tmp_path, "failed", 3, status="failed"),
        "empty": _job(tmp_path, "empty", 2, clips=[]),
        "nourl": _job(tmp_path, "nourl", 1, clips=[{"viral_hook_text": "x"}]),
    }
    jobs["noresult"] = {"status": "completed", "output_dir": str(tmp_path), "result": None}
    page = clip_history.build(jobs)
    assert [s["job_id"] for s in page["sources"]] == ["done"]


def test_clip_urls_are_only_passed_through(tmp_path):
    clips = [_clip("j", 0), _clip("j", 1, video_url="http://evil/x.mp4"),
             _clip("j", 2, video_url="/etc/passwd")]
    jobs = {"j": _job(tmp_path, "j", 1, clips=clips)}
    got = clip_history.build(jobs)["sources"][0]["clips"]
    assert [c["video_url"] for c in got] == ["/videos/j/clip_0.mp4"]


def test_missing_sidecar_falls_back(tmp_path):
    jobs = {"old": _job(tmp_path, "old", 1234)}
    src = clip_history.build(jobs)["sources"][0]
    assert src["brief"] == "" and src["preset_name"] is None and src["source"] is None
    assert src["title"] == "My Video"
    meta = tmp_path / "old" / "My Video_metadata.json"
    assert abs(src["created_at"] - os.path.getmtime(str(meta))) < 1


def test_no_metadata_title_falls_back_to_job_id(tmp_path):
    jobs = {"j": _job(tmp_path, "j", 1, metadata=False)}
    assert clip_history.build(jobs)["sources"][0]["title"] == "j"


def test_sign_failure_gives_null_download_url(tmp_path):
    def boom(rel):
        raise RuntimeError("no")
    jobs = {"j": _job(tmp_path, "j", 1)}
    assert clip_history.build(jobs, sign=boom)["sources"][0]["clips"][0]["download_url"] is None


def test_disk_reads_only_for_the_page(tmp_path, monkeypatch):
    jobs = {f"j{i}": _job(tmp_path, f"j{i}", 100 + i) for i in range(6)}
    reads = []
    real = clip_history._read_sidecar
    monkeypatch.setattr(clip_history, "_read_sidecar",
                        lambda d: reads.append(d) or real(d))
    clip_history.build(jobs, limit=2)
    assert len(reads) == 2


def test_endpoint_in_self_host_mode(tmp_path, monkeypatch):
    app_module = pytest.importorskip("app")
    job = _job(tmp_path, "hist1", 50, sidecar={"brief": "rules"})
    monkeypatch.setitem(app_module.jobs, "hist1", job)

    async def _do():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            return await c.get("/api/clip-history", params={"limit": 1000})
    r = asyncio.run(_do())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["limit"] == 50
    mine = [s for s in body["sources"] if s["job_id"] == "hist1"]
    assert mine and mine[0]["brief"] == "rules"
    assert mine[0]["clips"][0]["download_url"] is None  # self-host: no media secret
    # Self-host keeps history until deleted (0); HISTORY_KEEP=0 gives the clock back.
    assert body["retention_seconds"] == (
        0 if app_module.KEEP_HISTORY else app_module.JOB_RETENTION_SECONDS)


# --- hook fields, directives, tombstones -------------------------------------
def test_clip_carries_whitelisted_hook_fields():
    job = _job_in_memory = {
        "status": "completed", "output_dir": None, "user_id": None,
        "result": {"clips": [
            _clip("j", 0, hook_score=87, hook_type="open_loop",
                  recipe={"teaser": True}, transcript="SECRET"),
            _clip("j", 1, hook_score=250, hook_type="made_up", recipe={"teaser": "yes"}),
            _clip("j", 2, hook_score=True, hook_type=5, recipe="x"),
            _clip("j", 3),
        ]}}
    page = clip_history.build({"j": job})
    clips = page["sources"][0]["clips"]
    assert (clips[0]["hook_score"], clips[0]["hook_type"], clips[0]["teaser"]) == (87, "open_loop", True)
    assert (clips[1]["hook_score"], clips[1]["hook_type"], clips[1]["teaser"]) == (None, None, False)
    assert (clips[2]["hook_score"], clips[2]["hook_type"], clips[2]["teaser"]) == (None, None, False)
    assert (clips[3]["hook_score"], clips[3]["hook_type"], clips[3]["teaser"]) == (None, None, False)
    assert "SECRET" not in json.dumps(page)


def test_directives_are_whitelisted(tmp_path, monkeypatch):
    job = _job(tmp_path, "jd", 1000)
    applied = {"captions": "off", "hook_style": "x" * 41, "n": 3, "bad": [1],
               "flag": True, "k" * 41: "v"}
    applied.update({f"d{i}": i for i in range(20)})
    monkeypatch.setattr(creator_brief, "read_applied", lambda d: applied, raising=False)
    src = clip_history.build({"jd": job})["sources"][0]
    d = src["directives"]
    assert len(d) <= 12 and d["captions"] == "off" and d["n"] == 3
    assert "hook_style" not in d and "bad" not in d and "flag" not in d
    assert all(len(k) <= 40 for k in d)


def test_directives_default_to_empty_without_read_applied(tmp_path, monkeypatch):
    job = _job(tmp_path, "je", 1000)
    monkeypatch.delattr(creator_brief, "read_applied", raising=False)
    assert clip_history.build({"je": job})["sources"][0]["directives"] == {}
    monkeypatch.setattr(creator_brief, "read_applied",
                        lambda d: (_ for _ in ()).throw(OSError("x")), raising=False)
    assert clip_history.build({"je": job})["sources"][0]["directives"] == {}


def test_deleted_clips_are_skipped_and_empty_sources_omitted(tmp_path):
    gone = _job(tmp_path, "g1", 1000, clips=[
        _clip("g1", 0, deleted=True), _clip("g1", 1, deleted=True)])
    some = _job(tmp_path, "g2", 900, clips=[
        _clip("g2", 0, deleted=True), _clip("g2", 1)])
    page = clip_history.build({"g1": gone, "g2": some})
    assert page["total"] == 1
    assert [c["index"] for c in page["sources"][0]["clips"]] == [1]


def test_clip_file_names_matching():
    names = ["V_clip_1.mp4", "V_clip_11.mp4", "subtitled_1_V_clip_1.mp4",
             "subtitled_1_V_clip_11.mp4", "wm_V_clip_1.mp4", "wm_V_clip_11.mp4",
             "hook_V_clip_1.mp4", "V_clip_1.mp4.layout.json", "other.mp4"]
    assert sorted(clip_history.clip_file_names(names, "V", 0)) == sorted(
        ["V_clip_1.mp4", "subtitled_1_V_clip_1.mp4", "wm_V_clip_1.mp4", "hook_V_clip_1.mp4"])


# --- failed / interrupted jobs (manual retry) ---------------------------------
def _uid():
    return str(uuid.uuid4())


def _fail_dir(root, job_id, *, owner=None, submit=True, metadata=False, failed=None,
              label=None, created=None):
    import job_retry
    d = root / job_id
    d.mkdir()
    if submit:
        job_retry.write_submit(str(d), cmd=["python", "-u", "main.py", "-u",
                                            "https://example.com/v", "--keep-original",
                                            "-o", str(d)],
                               settings={}, priority=2)
        if created is not None:
            raw = json.loads((d / job_retry.SUBMIT_SIDECAR).read_text())
            raw["created_at"] = created
            (d / job_retry.SUBMIT_SIDECAR).write_text(json.dumps(raw))
    if metadata:
        (d / "V_metadata.json").write_text("{}")
    if owner is not None:
        (d / ".owner").write_text(str(owner))
    if failed:
        job_retry.write_failed(str(d), {"code": failed})
    if label is not None:
        creator_brief.write_sidecar(str(d), "secret brief text",
                                    source={"kind": "url", "label": label})
        side = json.loads((d / creator_brief.SIDECAR).read_text())
        side.pop("created_at", None)
        if created is not None:
            side["created_at"] = created
        (d / creator_brief.SIDECAR).write_text(json.dumps(side))
    return d


def _failed(jobs, root, **kw):
    import job_retry

    def reason(jid, job):
        if job.get("lost"):
            return job_retry.read_failed(job["output_dir"]) or job_retry.reason("server_restarted")
        return job_retry.classify_failure(job.get("logs") or [])
    args = dict(owner=None, scoped=False, read_reason=reason,
                read_submit=job_retry.read_submit, read_sidecar=clip_history._read_sidecar,
                is_active=lambda jid: False, now=1000.0,
                retry_info=lambda jid, submit: {"allowed": True, "used": submit.get("retries", 0),
                                                "limit": 5, "reason": None})
    args.update(kw)
    return clip_history.failed_jobs(jobs, str(root), **args)


@pytest.fixture()
def froot(tmp_path):
    clip_history.invalidate_failed()
    root = tmp_path / "out"
    root.mkdir()
    yield root
    clip_history.invalidate_failed()


def test_failed_lists_memory_failures_and_disk_leftovers(froot):
    mem = _uid()
    d = _fail_dir(froot, mem, label="https://example.com/a", created=300)
    disk_failed = _uid()
    _fail_dir(froot, disk_failed, failed="no_clips", label="talk.mp4", created=200)
    disk_lost = _uid()
    _fail_dir(froot, disk_lost, created=100)
    _fail_dir(froot, _uid(), metadata=True)            # completed: not listed
    _fail_dir(froot, _uid(), submit=False)              # pre-feature dir: not listed
    (froot / "not-a-uuid").mkdir()
    jobs = {mem: {"status": "failed", "output_dir": str(d),
                  "logs": ["❌ Error: GEMINI_API_KEY not found."]},
            "q": {"status": "queued", "output_dir": None}}
    out = _failed(jobs, froot)
    ids = [e["job_id"] for e in out]
    assert ids[0] == mem and set(ids) == {mem, disk_failed, disk_lost}
    by = {e["job_id"]: e for e in out}
    assert by[mem]["state"] == "failed" and by[mem]["error"]["code"] == "missing_key"
    assert by[mem]["title"] == "https://example.com/a"
    assert by[disk_failed]["state"] == "failed" and by[disk_failed]["error"]["code"] == "no_clips"
    assert by[disk_failed]["title"] == "talk.mp4"
    assert by[disk_lost]["state"] == "interrupted"
    assert by[disk_lost]["error"]["code"] == "server_restarted"
    assert by[disk_lost]["title"] == "Untitled video"
    assert by[disk_lost]["retry"] == {"allowed": True, "used": 0, "limit": 5, "reason": None}
    assert "secret brief text" not in json.dumps(out)


def test_failed_excludes_active_and_in_memory_non_failed(froot):
    a, b = _uid(), _uid()
    _fail_dir(froot, a)
    _fail_dir(froot, b)
    out = _failed({b: {"status": "queued", "output_dir": str(froot / b)}}, froot,
                  is_active=lambda jid: jid == a)
    assert out == []


def test_failed_scope_in_cloud(froot):
    mine, theirs, ownerless = _uid(), _uid(), _uid()
    _fail_dir(froot, mine, owner="alice")
    _fail_dir(froot, theirs, owner="bob")
    _fail_dir(froot, ownerless)
    mem_mine, mem_theirs = _uid(), _uid()
    jobs = {mem_mine: {"status": "failed", "user_id": "alice", "logs": [],
                       "output_dir": str(froot / "x")},
            mem_theirs: {"status": "failed", "user_id": "bob", "logs": [],
                         "output_dir": str(froot / "y")}}
    out = _failed(jobs, froot, owner="alice", scoped=True)
    assert {e["job_id"] for e in out} == {mine, mem_mine}
    assert _failed(jobs, froot, owner=None, scoped=True) == []
    clip_history.invalidate_failed()
    assert len(_failed(jobs, froot)) == 5              # self-host lists all


def test_failed_scan_is_cached_for_30s(froot, monkeypatch):
    _fail_dir(froot, _uid())
    calls = []
    real = clip_history._scan_failed
    monkeypatch.setattr(clip_history, "_scan_failed",
                        lambda *a: calls.append(1) or real(*a))
    assert len(_failed({}, froot, now=1000.0)) == 1
    _fail_dir(froot, _uid())
    assert len(_failed({}, froot, now=1029.0)) == 1     # cached
    assert len(calls) == 1
    assert len(_failed({}, froot, now=1031.0)) == 2     # expired
    clip_history.invalidate_failed()
    _failed({}, froot, now=1032.0)
    assert len(calls) == 3


def test_failed_is_capped_at_20(froot):
    for i in range(25):
        _fail_dir(froot, _uid(), created=float(i + 1))
    out = _failed({}, froot)
    assert len(out) == 20 and out[0]["created_at"] == 25.0
