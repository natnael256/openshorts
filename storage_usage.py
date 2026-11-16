"""Self-host storage report for GET /api/storage.

Walking a directory of video is slow, so the byte count is cached for a minute
behind a lock: concurrent requests share one walk, and a delete invalidates it.
Reporting only; nothing here blocks or removes anything.
"""
import threading
import time

TTL_SECONDS = 60

_lock = threading.Lock()
_cache = {"path": None, "at": 0.0, "bytes": 0}


def used_bytes(path, size_fn, now=time.monotonic):
    """``size_fn(path)``, reused for ``TTL_SECONDS``. Blocking: run in a thread."""
    with _lock:
        t = now()
        if _cache["path"] == path and 0 <= t - _cache["at"] < TTL_SECONDS:
            return _cache["bytes"]
        total = int(size_fn(path))
        _cache.update(path=path, at=now(), bytes=total)
        return total


def invalidate():
    with _lock:
        _cache.update(path=None, at=0.0, bytes=0)


def report(used, cap, keep_history):
    used = max(0, int(used))
    cap = max(0, int(cap))
    return {
        "enabled": True,
        "used_bytes": used,
        "cap_bytes": cap,
        "percent": round(used * 100 / cap, 1) if cap > 0 else 0,
        "full": cap > 0 and used >= cap,
        "keep_history": bool(keep_history),
    }
