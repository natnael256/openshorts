"""Clip history: each source video as a parent with its clips underneath.

Pure logic over the in-memory ``jobs`` mapping (app.py rebuilds completed jobs
from disk after a restart, so there is no new store and no directory scan per
request). The endpoint in app.py is a thin wrapper: auth, then ``build``.

Rules this module keeps:
  * clip URLs are only ever passed through from the job result (the served
    ``video_url`` already honours the free-plan watermark copy); nothing here
    builds a path from file names;
  * a clip exposes a fixed whitelist of fields, never the transcript or
    anything else the metadata holds;
  * disk reads (sidecar, metadata name) happen for the returned page only; the
    sort key is one cached ``stat`` per job.

The one exception is ``failed_jobs``: a job that failed before a restart is no
longer in memory, so it scans the job directories for a submit sidecar
(job_retry) with no metadata. That scan reads only the small sidecars and is
cached for ``FAILED_SCAN_TTL`` seconds.
"""
import glob
import json
import os
import re
import threading

DEFAULT_LIMIT = 20
MAX_LIMIT = 50
VERDICTS = ("pass", "flag", "reject", "unchecked")
_SOURCE_KINDS = ("url", "file", "upload_id", "thumbnail_session")
_MAX_LABEL = 200
_MAX_REASON = 200
_MAX_VIOLATED = 10
_MAX_VIOLATED_CHARS = 120
_MAX_TEXT = 300
HOOK_TYPES = ("curiosity_gap", "bold_claim", "contrarian", "mistake", "open_loop")
_MAX_DIRECTIVES = 12
_MAX_DIRECTIVE_CHARS = 40

# job_id -> created_at used for ordering; pruned against the live jobs mapping.
_created_cache = {}


def clamp_page(limit, offset):
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = DEFAULT_LIMIT
    try:
        offset = int(offset)
    except (TypeError, ValueError):
        offset = 0
    return max(1, min(MAX_LIMIT, limit)), max(0, offset)


def _num(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return float(value)


def _text(value, cap=_MAX_TEXT):
    return value.strip()[:cap] if isinstance(value, str) else ""


def clean_brief_check(raw):
    """A verdict the dashboard can trust, or the neutral 'unchecked'."""
    neutral = {"verdict": "unchecked", "violated": [], "reason": ""}
    if not isinstance(raw, dict):
        return neutral
    verdict = raw.get("verdict")
    violated = raw.get("violated", [])
    reason = raw.get("reason", "")
    if verdict not in VERDICTS or not isinstance(violated, list) or not isinstance(reason, str):
        return neutral
    if not all(isinstance(v, str) for v in violated):
        return neutral
    return {
        "verdict": verdict,
        "violated": [v.strip()[:_MAX_VIOLATED_CHARS] for v in violated[:_MAX_VIOLATED] if v.strip()],
        "reason": reason.strip()[:_MAX_REASON],
    }


def _hook_score(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    value = int(round(value))
    return value if 0 <= value <= 100 else None


def _clean_clip(index, clip, sign):
    if not isinstance(clip, dict) or clip.get("deleted"):
        return None
    rel = clip.get("video_url")
    if not isinstance(rel, str) or not rel.startswith("/videos/"):
        return None
    start, end = _num(clip.get("start")), _num(clip.get("end"))
    duration = round(end - start, 3) if start is not None and end is not None and end >= start else None
    try:
        download = sign(rel)
    except Exception:
        download = None
    return {
        "index": index,
        "title": _text(clip.get("video_title_for_youtube_short")),
        "hook": _text(clip.get("viral_hook_text")),
        "start": start,
        "end": end,
        "duration": duration,
        "video_url": rel,
        "hook_score": _hook_score(clip.get("hook_score")),
        "hook_type": clip.get("hook_type") if clip.get("hook_type") in HOOK_TYPES else None,
        "teaser": isinstance(clip.get("recipe"), dict) and clip["recipe"].get("teaser") is True,
        "download_url": download if isinstance(download, str) else None,
        # None when the job had no brief (nothing was checked, so no badge).
        "brief_check": (clean_brief_check(clip["brief_check"])
                        if clip.get("brief_check") is not None else None),
    }


def _clips_of(job):
    result = job.get("result")
    clips = result.get("clips") if isinstance(result, dict) else None
    return clips if isinstance(clips, list) else []


def _owner_matches(job, owner):
    job_owner = job.get("user_id")
    return job_owner is not None and owner is not None and str(job_owner) == str(owner)


def _sort_key(job_id, job):
    cached = _created_cache.get(job_id)
    if cached is not None:
        return cached
    try:
        value = os.stat(job.get("output_dir")).st_mtime
    except (OSError, TypeError, ValueError):
        value = 0.0
    _created_cache[job_id] = value
    return value


def _read_sidecar(output_dir):
    try:
        import creator_brief
        data = creator_brief.read_sidecar(output_dir)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _directives(output_dir):
    """The directives a creator brief applied to this job (``{}`` when none)."""
    try:
        import creator_brief
        raw = creator_brief.read_applied(output_dir)
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    out = {}
    for key, value in raw.items():
        if len(out) >= _MAX_DIRECTIVES:
            break
        if not isinstance(key, str) or not key or len(key) > _MAX_DIRECTIVE_CHARS:
            continue
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            continue
        if isinstance(value, str) and len(value) > _MAX_DIRECTIVE_CHARS:
            continue
        out[key] = value
    return out


def _clean_source(raw):
    if not isinstance(raw, dict) or raw.get("kind") not in _SOURCE_KINDS:
        return None
    return {"kind": raw["kind"], "label": _text(raw.get("label"), _MAX_LABEL)}


def _metadata_title(output_dir, job_id):
    try:
        found = sorted(glob.glob(os.path.join(output_dir, "*_metadata.json")))
    except (TypeError, ValueError):
        found = []
    if found:
        return os.path.basename(found[0])[: -len("_metadata.json")]
    return job_id


def _metadata_mtime(output_dir):
    try:
        found = glob.glob(os.path.join(output_dir, "*_metadata.json"))
        return os.path.getmtime(found[0]) if found else None
    except (OSError, TypeError, ValueError):
        return None


def build(jobs, *, owner=None, scoped=False, limit=DEFAULT_LIMIT, offset=0,
          sign=lambda rel: None, retention_seconds=0):
    """The history page.

    ``scoped``: only jobs owned by ``owner`` (compared as strings); jobs with no
    owner are never listed then. Unscoped (self-host) lists everything.
    ``sign``: ``rel_video_url -> absolute signed url | None``.
    """
    limit, offset = clamp_page(limit, offset)
    eligible = []
    # list() because the event loop's other tasks mutate `jobs` between awaits.
    for job_id, job in list(jobs.items()):
        if not isinstance(job, dict) or job.get("status") != "completed":
            continue
        if scoped and not _owner_matches(job, owner):
            continue
        if not any(isinstance(c, dict) and not c.get("deleted")
                   and isinstance(c.get("video_url"), str)
                   for c in _clips_of(job)):
            continue
        eligible.append((_sort_key(job_id, job), job_id, job))
    if len(_created_cache) > 4 * len(jobs) + 64:
        for stale in set(_created_cache) - set(jobs):
            _created_cache.pop(stale, None)
    eligible.sort(key=lambda t: (t[0], t[1]), reverse=True)

    sources = []
    for sort_ts, job_id, job in eligible[offset:offset + limit]:
        out_dir = job.get("output_dir")
        side = _read_sidecar(out_dir) if out_dir else {}
        source = _clean_source(side.get("source"))
        created = _num(side.get("created_at"))
        if created is None:
            created = (_metadata_mtime(out_dir) if out_dir else None) or sort_ts
        title = _metadata_title(out_dir, job_id) if out_dir else job_id
        preset = side.get("preset_name")
        clips = []
        for i, clip in enumerate(_clips_of(job)):
            item = _clean_clip(i, clip, sign)
            if item:
                clips.append(item)
        sources.append({
            "job_id": job_id,
            "title": title,
            "created_at": created,
            "source": source,
            "brief": side.get("brief") if isinstance(side.get("brief"), str) else "",
            "directives": _directives(out_dir) if out_dir else {},
            "preset_name": _text(preset, 60) or None if isinstance(preset, str) else None,
            "clips": clips,
        })
    return {"sources": sources, "total": len(eligible), "limit": limit,
            "offset": offset, "retention_seconds": retention_seconds}


def forget(job_id):
    """Drop a deleted job from the ordering cache."""
    _created_cache.pop(job_id, None)


# --- Failed or interrupted jobs (manual retry) --------------------------------
FAILED_LIMIT = 20
FAILED_SCAN_TTL = 30.0
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_FAILED_SIDECAR = ".failed.json"
_failed_lock = threading.Lock()
_failed_cache = {"at": None, "dir": None, "entries": []}


def invalidate_failed():
    """Forget the cached directory scan (a retry or a delete changed it)."""
    with _failed_lock:
        _failed_cache["at"] = None


def _owner_marker(job_dir):
    try:
        with open(os.path.join(job_dir, ".owner"), "r", encoding="utf-8") as f:
            raw = f.read(128).strip()
    except OSError:
        return None
    return raw or None


def _scan_failed(output_dir, read_submit):
    """Job dirs holding a valid submit sidecar and no ``*_metadata.json``."""
    entries = []
    try:
        names = os.listdir(output_dir)
    except OSError:
        return entries
    for name in names:
        if not _UUID_RE.match(name):
            continue
        job_dir = os.path.join(output_dir, name)
        if os.path.islink(job_dir) or not os.path.isdir(job_dir):
            continue
        submit = read_submit(job_dir)
        if not submit:
            continue
        if glob.glob(os.path.join(glob.escape(job_dir), "*_metadata.json")):
            continue
        try:
            mtime = os.stat(job_dir).st_mtime
        except OSError:
            mtime = 0.0
        entries.append({"job_id": name, "dir": job_dir, "owner": _owner_marker(job_dir),
                        "submit": submit, "mtime": mtime})
    return entries


def _disk_candidates(output_dir, read_submit, now):
    with _failed_lock:
        fresh = (_failed_cache["at"] is not None and _failed_cache["dir"] == output_dir
                 and 0 <= now - _failed_cache["at"] < FAILED_SCAN_TTL)
        if not fresh:
            _failed_cache.update(at=now, dir=output_dir,
                                 entries=_scan_failed(output_dir, read_submit))
        return list(_failed_cache["entries"])


def _retry_summary(raw, submit):
    if isinstance(raw, dict):
        return {"allowed": raw.get("allowed") is True,
                "used": raw.get("used") if isinstance(raw.get("used"), int) else 0,
                "limit": raw.get("limit") if isinstance(raw.get("limit"), int) else None,
                "reason": raw.get("reason") if isinstance(raw.get("reason"), str) else None}
    used = submit.get("retries", 0) if isinstance(submit, dict) else 0
    return {"allowed": False, "used": used if isinstance(used, int) else 0,
            "limit": None, "reason": None}


def failed_jobs(jobs, output_dir, *, owner=None, scoped=False, read_reason, read_submit,
                read_sidecar, is_active, now, retry_info=None, limit=FAILED_LIMIT):
    """Failed or interrupted clip jobs, newest first, at most ``limit``.

    In memory: every job whose status is ``failed``. On disk: a job directory
    with a submit sidecar, no metadata, not in ``jobs`` and not active (the
    server restarted, or gave up resuming it): ``state`` ``"failed"`` when a
    failure reason was recorded, ``"interrupted"`` otherwise. Owner scoping as
    in ``build``. ``title`` / ``source`` come from the creator-brief sidecar's
    source label, never from log lines.

    ``read_reason(job_id, job) -> {code, message}``; ``read_submit(dir)``;
    ``read_sidecar(dir)``; ``is_active(job_id)``; ``retry_info(job_id,
    submit) -> {allowed, used, limit, reason}``.
    """
    picked = []
    for job_id, job in list(jobs.items()):
        if not isinstance(job, dict) or job.get("status") != "failed":
            continue
        if scoped and not _owner_matches(job, owner):
            continue
        out_dir = job.get("output_dir") or os.path.join(output_dir, job_id)
        picked.append((job_id, job, out_dir, "failed", None))
    for entry in _disk_candidates(output_dir, read_submit, now):
        job_id = entry["job_id"]
        if job_id in jobs or not os.path.isdir(entry["dir"]) or is_active(job_id):
            continue
        if scoped and (entry["owner"] is None or owner is None
                       or entry["owner"] != str(owner)):
            continue
        state = ("failed" if os.path.isfile(os.path.join(entry["dir"], _FAILED_SIDECAR))
                 else "interrupted")
        view = {"status": "failed", "lost": True, "output_dir": entry["dir"],
                "user_id": entry["owner"]}
        picked.append((job_id, view, entry["dir"], state, entry))

    items = []
    for job_id, job, out_dir, state, entry in picked:
        side = read_sidecar(out_dir) or {}
        source = _clean_source(side.get("source"))
        submit = entry["submit"] if entry else (read_submit(out_dir) or {})
        created = (_num(side.get("created_at")) or _num(submit.get("created_at"))
                   or (entry["mtime"] if entry else None))
        if created is None:
            try:
                created = os.stat(out_dir).st_mtime
            except (OSError, TypeError, ValueError):
                created = 0.0
        try:
            error = read_reason(job_id, job)
        except Exception:
            error = None
        if not (isinstance(error, dict) and isinstance(error.get("code"), str)
                and isinstance(error.get("message"), str)):
            error = {"code": "unknown", "message": ""}
        try:
            raw_retry = retry_info(job_id, submit) if retry_info else None
        except Exception:
            raw_retry = None
        items.append({
            "job_id": job_id,
            "title": (source or {}).get("label") or "Untitled video",
            "created_at": created,
            "source": source,
            "state": state,
            "error": {"code": error["code"], "message": error["message"]},
            "retry": _retry_summary(raw_retry, submit),
        })
    items.sort(key=lambda it: (it["created_at"], it["job_id"]), reverse=True)
    return items[:max(0, int(limit))]


# --- Deleting one clip --------------------------------------------------------
# The clip list is positional everywhere (file names carry index + 1 and every
# editor endpoint addresses clips by index), so a deleted clip stays in the list
# as a tombstone: ``deleted: true`` and no ``video_url``.

_delete_lock = threading.Lock()
_DERIVED_LAYERS = "wm_"


def live_indices(shorts):
    return [i for i, c in enumerate(shorts)
            if isinstance(c, dict) and not c.get("deleted")]


def _metadata_path(job_dir):
    found = sorted(glob.glob(os.path.join(job_dir, "*_metadata.json")))
    return found[0] if found else None


def clip_file_names(names, base_name, index):
    """The entries of ``names`` that are a version of clip ``index``.

    The clean file is ``<base>_clip_<n>.mp4`` (n = index + 1); every derived
    layer (``subtitled_<ts>_``, ``hooked_<ts>_``, ``recut_<ts>_``, ``hook_``,
    ``wm_``) keeps it as the tail, so the match is "is the clean name" or "ends
    with ``_<clean>``", after an optional ``wm_``. ``_clip_1.mp4`` can never
    match ``_clip_11.mp4``: the tail has to line up on the underscore.
    """
    clean = f"{base_name}_clip_{index + 1}.mp4"
    out = []
    for name in names:
        bare = name[len(_DERIVED_LAYERS):] if name.startswith(_DERIVED_LAYERS) else name
        if bare == clean or bare.endswith("_" + clean) or name.endswith("_" + clean):
            out.append(name)
    return out


def _remove_clip_files(job_dir, base_name, index):
    try:
        names = os.listdir(job_dir)
    except OSError:
        return 0
    removed = 0
    for name in clip_file_names(names, base_name, index):
        path = os.path.join(job_dir, name)
        if os.path.islink(path) or not os.path.isfile(path):
            continue
        victims = [path, path + ".layout.json"]
        try:
            from hooks import recipe_paths
            victims.extend(recipe_paths(job_dir, name))
        except Exception:
            victims.extend(os.path.join(job_dir, f".hookrecipe_{name}{ext}")
                           for ext in (".json", ".png"))
        for victim in victims:
            try:
                os.remove(victim)
                removed += 1
            except OSError:
                pass
    return removed


def delete_clip(job_dir, index):
    """Tombstone clip ``index`` of the job in ``job_dir`` and delete its files.

    Returns ``"clip"`` when done, ``"last"`` when it is the only live clip (the
    caller then deletes the whole job, nothing is touched here). Raises
    ``LookupError`` for an unknown, out-of-range or already deleted clip.
    """
    from jsonfile import write_json_atomic
    with _delete_lock:
        meta_path = _metadata_path(job_dir)
        if not meta_path:
            raise LookupError("no metadata")
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        shorts = data.get("shorts")
        if (not isinstance(shorts, list) or isinstance(index, bool)
                or not isinstance(index, int) or index not in live_indices(shorts)):
            raise LookupError("no such clip")
        if len(live_indices(shorts)) == 1:
            return "last"
        shorts[index]["deleted"] = True
        shorts[index].pop("video_url", None)
        write_json_atomic(meta_path, data, indent=2)
        base_name = os.path.basename(meta_path)[: -len("_metadata.json")]
        _remove_clip_files(job_dir, base_name, index)
        return "clip"
