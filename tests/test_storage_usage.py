"""GET /api/storage: numbers, the full flag, the 60 s cache, cloud off."""
import asyncio
import threading

import httpx
import pytest

import storage_usage

app_module = pytest.importorskip("app")

GB = 1024 ** 3


def _get():
    async def _do():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://testserver") as client:
            return await client.get("/api/storage")
    return asyncio.run(_do())


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    storage_usage.invalidate()
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(app_module, "BILLING_ENABLED", False)
    yield
    storage_usage.invalidate()


def _sizes(monkeypatch, value, calls=None):
    def fake(path):
        if calls is not None:
            calls.append(path)
        return value
    monkeypatch.setattr(app_module, "_dir_size", fake)


def test_numbers(monkeypatch):
    monkeypatch.setattr(app_module, "OUTPUT_MAX_GB", 10)
    monkeypatch.setattr(app_module, "KEEP_HISTORY", True)
    _sizes(monkeypatch, 5 * GB)
    body = _get().json()
    assert body == {"enabled": True, "used_bytes": 5 * GB, "cap_bytes": 10 * GB,
                    "percent": 50.0, "full": False, "keep_history": True}


def test_full_at_and_above_the_cap(monkeypatch):
    monkeypatch.setattr(app_module, "OUTPUT_MAX_GB", 10)
    _sizes(monkeypatch, 10 * GB)
    assert _get().json()["full"] is True
    storage_usage.invalidate()
    _sizes(monkeypatch, 12 * GB)
    body = _get().json()
    assert body["full"] is True and body["percent"] == 120.0
    storage_usage.invalidate()
    _sizes(monkeypatch, 10 * GB - 1)
    assert _get().json()["full"] is False


def test_cap_zero_is_never_full(monkeypatch):
    monkeypatch.setattr(app_module, "OUTPUT_MAX_GB", 0)
    _sizes(monkeypatch, 50 * GB)
    body = _get().json()
    assert body["full"] is False and body["cap_bytes"] == 0 and body["percent"] == 0


def test_cache_reuse_and_invalidation(monkeypatch):
    calls = []
    _sizes(monkeypatch, 1 * GB, calls)
    _get()
    _get()
    assert len(calls) == 1
    storage_usage.invalidate()
    _get()
    assert len(calls) == 2


def test_cache_expires_after_ttl():
    calls = []
    clock = [100.0]
    now = lambda: clock[0]
    size = lambda p: calls.append(p) or 7
    assert storage_usage.used_bytes("/x", size, now) == 7
    clock[0] += storage_usage.TTL_SECONDS - 1
    storage_usage.used_bytes("/x", size, now)
    assert len(calls) == 1
    clock[0] += 2
    storage_usage.used_bytes("/x", size, now)
    assert len(calls) == 2


def test_concurrent_requests_share_one_walk():
    calls = []
    gate = threading.Event()

    def slow(path):
        calls.append(path)
        gate.wait(2)
        return 3

    results = []
    threads = [threading.Thread(target=lambda: results.append(
        storage_usage.used_bytes("/y", slow))) for _ in range(5)]
    for t in threads:
        t.start()
    gate.set()
    for t in threads:
        t.join()
    assert results == [3] * 5 and len(calls) == 1


def test_cloud_mode_is_disabled(monkeypatch):
    monkeypatch.setattr(app_module, "BILLING_ENABLED", True)
    calls = []
    _sizes(monkeypatch, 1 * GB, calls)
    assert _get().json() == {"enabled": False}
    assert calls == []
