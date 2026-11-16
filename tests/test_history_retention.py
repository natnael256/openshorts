"""Self-host keeps completed clip jobs until the user deletes them; everything
else keeps aging out; cloud mode and HISTORY_KEEP=0 behave as before."""
import json
import os
import time
import uuid

import pytest

app_module = pytest.importorskip("app")

OLD = 10 * 24 * 3600


@pytest.fixture()
def out(tmp_path, monkeypatch):
    root = tmp_path / "output"
    (root / "thumbnails").mkdir(parents=True)
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(root))
    monkeypatch.setattr(app_module, "THUMBNAILS_DIR", str(root / "thumbnails"))
    monkeypatch.setattr(app_module, "jobs", {})
    monkeypatch.setattr(app_module, "BILLING_ENABLED", False)
    monkeypatch.setattr(app_module, "KEEP_HISTORY", True)
    monkeypatch.setattr(app_module, "JOB_RETENTION_SECONDS", 3600)
    return root


def _job(root, metadata=True, age=OLD):
    job_id = str(uuid.uuid4())
    d = root / job_id
    d.mkdir()
    (d / "clip_1.mp4").write_bytes(b"x" * 10)
    if metadata:
        (d / "Video_metadata.json").write_text(json.dumps({"shorts": [{}]}))
    t = time.time() - age
    os.utime(d, (t, t))
    return job_id, d


def _over_cap(monkeypatch):
    """OUTPUT_DIR looks full, every job dir weighs next to nothing."""
    root = app_module.OUTPUT_DIR
    monkeypatch.setattr(app_module, "_dir_size",
                        lambda p: 10 ** 13 if p == root else 10)


def test_old_kept_job_survives_the_age_sweep(out):
    kept, kept_dir = _job(out)
    scratch, scratch_dir = _job(out, metadata=False)
    app_module.jobs[kept] = {"status": "completed"}
    app_module._sweep_expired_job_dirs(time.time())
    assert kept_dir.is_dir() and kept in app_module.jobs
    assert not scratch_dir.exists()


def test_old_kept_job_survives_the_size_cap(out, monkeypatch):
    kept, kept_dir = _job(out)
    scratch, scratch_dir = _job(out, metadata=False)
    _over_cap(monkeypatch)
    app_module._enforce_output_size_cap()
    assert kept_dir.is_dir()
    assert not scratch_dir.exists()
    assert (out / "thumbnails").is_dir()


def test_kept_job_recovered_from_disk_is_also_kept(out):
    kept, kept_dir = _job(out)  # no in-memory record at all
    app_module._sweep_expired_job_dirs(time.time())
    assert kept_dir.is_dir()


def test_failed_job_with_metadata_still_ages_out(out):
    failed, failed_dir = _job(out)
    app_module.jobs[failed] = {"status": "failed"}
    app_module._sweep_expired_job_dirs(time.time())
    assert not failed_dir.exists()


def test_history_keep_off_restores_the_old_behaviour(out, monkeypatch):
    monkeypatch.setattr(app_module, "KEEP_HISTORY", False)
    j1, d1 = _job(out)
    j2, d2 = _job(out)
    app_module._sweep_expired_job_dirs(time.time())
    assert not d1.exists() and not d2.exists()
    j3, d3 = _job(out)
    _over_cap(monkeypatch)
    app_module._enforce_output_size_cap()
    assert not d3.exists()


def test_cloud_mode_has_history_keep_off():
    # KEEP_HISTORY is frozen at import: billing on can never keep history.
    src = open(app_module.__file__, encoding="utf-8").read()
    assert 'KEEP_HISTORY = (not BILLING_ENABLED) and' in src


def test_active_jobs_stay_protected(out, monkeypatch):
    monkeypatch.setattr(app_module, "KEEP_HISTORY", False)
    queued, queued_dir = _job(out, metadata=False)
    running, running_dir = _job(out)
    app_module.jobs[queued] = {"status": "queued"}
    app_module._running_jobs.add(running)
    try:
        app_module._sweep_expired_job_dirs(time.time())
        _over_cap(monkeypatch)
        app_module._enforce_output_size_cap()
    finally:
        app_module._running_jobs.discard(running)
    assert queued_dir.is_dir() and running_dir.is_dir()


def test_config_and_history_report_until_deleted(out):
    import asyncio
    cfg = asyncio.run(app_module.get_config())
    assert cfg["keepHistory"] is True
