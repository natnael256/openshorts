"""Saved creator-brief presets: named, reusable instruction sets.

Two backends behind one small interface, picked by ``configure``:

  * cloud (billing on): the ``brief_presets`` table, every query filtered by
    the signed-in user's id;
  * self-host: one dotfile JSON under OUTPUT_DIR (``.brief_presets.json``).
    ``/videos`` can never serve a dotfile and the retention sweeps only touch
    directories, so it outlives the job cleanup.

app.py cannot be imported from here (circular), so it hands in what it owns.
"""
import json
import os
import re
import threading
import unicodedata
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request

import jsonfile

MAX_PRESETS = 50
MAX_NAME_CHARS = 60
STORE_NAME = ".brief_presets.json"

router = APIRouter()

_cfg = {"output_dir": None, "billing": False, "resolver": None}
_lock = threading.Lock()
_ID_RE = re.compile(r"^[0-9a-fA-F]{32}$|^[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}$")


def configure(output_dir, billing_enabled, user_resolver):
    """``user_resolver``: ``async (request) -> user | None`` (cloud only)."""
    _cfg["output_dir"] = output_dir
    _cfg["billing"] = bool(billing_enabled)
    _cfg["resolver"] = user_resolver


def _brief_module():
    import creator_brief
    return creator_brief


def max_brief_chars():
    return _brief_module().MAX_BRIEF_CHARS


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def clean_name(raw):
    if not isinstance(raw, str):
        raise HTTPException(status_code=400, detail="name must be a string")
    text = unicodedata.normalize("NFKC", raw)
    text = "".join(" " if ch.isspace() else ch for ch in text
                   if ch.isspace() or not unicodedata.category(ch).startswith("C"))
    text = re.sub(r" {2,}", " ", text).strip()
    if not text:
        raise HTTPException(status_code=400, detail="name must not be empty")
    if len(text) > MAX_NAME_CHARS:
        raise HTTPException(status_code=400,
                            detail=f"name is longer than {MAX_NAME_CHARS} characters")
    return text


def clean_brief(raw):
    if not isinstance(raw, str):
        raise HTTPException(status_code=400, detail="brief must be a string")
    limit = max_brief_chars()
    if len(raw) > limit:
        raise HTTPException(status_code=400,
                            detail=f"brief is longer than {limit} characters")
    text = _brief_module().sanitize(raw)
    if not text:
        raise HTTPException(status_code=400, detail="brief must not be empty")
    return text


def valid_id(preset_id):
    return isinstance(preset_id, str) and bool(_ID_RE.match(preset_id))


def _check_id(preset_id):
    if not valid_id(preset_id):
        raise HTTPException(status_code=404, detail="Preset not found")
    return preset_id.replace("-", "").lower()


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _public(row):
    return {"id": row["id"], "name": row["name"], "brief": row["brief"],
            "updated_at": row["updated_at"]}


# --------------------------------------------------------------------------- #
# Self-host backend (JSON file)
# --------------------------------------------------------------------------- #
def _store_path():
    base = _cfg["output_dir"]
    if not base:
        raise HTTPException(status_code=503, detail="Presets are not configured")
    return os.path.join(base, STORE_NAME)


def _read_file():
    try:
        with open(_store_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
    except HTTPException:
        raise
    except Exception:
        return []  # absent or corrupt: treated as empty
    items = data.get("presets") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    out = []
    for it in items:
        if (isinstance(it, dict) and isinstance(it.get("id"), str)
                and isinstance(it.get("name"), str) and isinstance(it.get("brief"), str)):
            out.append({"id": it["id"], "name": it["name"], "brief": it["brief"],
                        "updated_at": str(it.get("updated_at") or "")})
    return out


def _write_file(items):
    path = _store_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    jsonfile.write_json_atomic(path, {"presets": items})


def _same_name(a, b):
    return a.casefold() == b.casefold()


class _FileStore:
    async def list(self, owner):
        with _lock:
            items = _read_file()
        return sorted(items, key=lambda r: r["updated_at"], reverse=True)

    async def get(self, owner, preset_id):
        with _lock:
            for it in _read_file():
                if it["id"] == preset_id:
                    return it
        return None

    async def create(self, owner, name, brief):
        with _lock:
            items = _read_file()
            if len(items) >= MAX_PRESETS:
                raise HTTPException(status_code=409,
                                    detail=f"You can keep at most {MAX_PRESETS} presets")
            if any(_same_name(i["name"], name) for i in items):
                raise HTTPException(status_code=409, detail="A preset with that name exists")
            row = {"id": uuid.uuid4().hex, "name": name, "brief": brief,
                   "updated_at": _now_iso()}
            items.append(row)
            _write_file(items)
        return row

    async def update(self, owner, preset_id, name, brief):
        with _lock:
            items = _read_file()
            row = next((i for i in items if i["id"] == preset_id), None)
            if row is None:
                return None
            if name is not None:
                if any(i is not row and _same_name(i["name"], name) for i in items):
                    raise HTTPException(status_code=409,
                                        detail="A preset with that name exists")
                row["name"] = name
            if brief is not None:
                row["brief"] = brief
            row["updated_at"] = _now_iso()
            _write_file(items)
            return row

    async def delete(self, owner, preset_id):
        with _lock:
            items = _read_file()
            kept = [i for i in items if i["id"] != preset_id]
            if len(kept) == len(items):
                return False
            _write_file(kept)
            return True


# --------------------------------------------------------------------------- #
# Cloud backend (SQLAlchemy)
# --------------------------------------------------------------------------- #
def _row_dict(r):
    ts = r.updated_at or r.created_at
    return {"id": r.id.hex, "name": r.name, "brief": r.brief,
            "updated_at": ts.isoformat() if ts else ""}


class _DbStore:
    @staticmethod
    def _uid(preset_id):
        return uuid.UUID(preset_id)

    async def list(self, owner):
        from sqlalchemy import select
        from cloud import database
        from cloud.models import BriefPreset
        async with database.session() as session:
            rows = (await session.execute(
                select(BriefPreset).where(BriefPreset.user_id == owner)
                .order_by(BriefPreset.updated_at.desc()))).scalars().all()
            return [_row_dict(r) for r in rows]

    async def get(self, owner, preset_id):
        from sqlalchemy import select
        from cloud import database
        from cloud.models import BriefPreset
        async with database.session() as session:
            r = (await session.execute(
                select(BriefPreset).where(BriefPreset.id == self._uid(preset_id),
                                          BriefPreset.user_id == owner))).scalar_one_or_none()
            return _row_dict(r) if r else None

    async def create(self, owner, name, brief):
        from sqlalchemy import select
        from cloud import database
        from cloud.models import BriefPreset
        async with database.session() as session:
            async with session.begin():
                names = (await session.execute(
                    select(BriefPreset.name).where(BriefPreset.user_id == owner))).scalars().all()
                if len(names) >= MAX_PRESETS:
                    raise HTTPException(status_code=409,
                                        detail=f"You can keep at most {MAX_PRESETS} presets")
                if any(_same_name(n, name) for n in names):
                    raise HTTPException(status_code=409,
                                        detail="A preset with that name exists")
                row = BriefPreset(user_id=owner, name=name, brief=brief,
                                  updated_at=datetime.now(timezone.utc))
                session.add(row)
                await session.flush()
                await session.refresh(row)
                return _row_dict(row)

    async def update(self, owner, preset_id, name, brief):
        from sqlalchemy import select
        from cloud import database
        from cloud.models import BriefPreset
        async with database.session() as session:
            async with session.begin():
                r = (await session.execute(
                    select(BriefPreset).where(BriefPreset.id == self._uid(preset_id),
                                              BriefPreset.user_id == owner))).scalar_one_or_none()
                if r is None:
                    return None
                if name is not None:
                    others = (await session.execute(
                        select(BriefPreset.name).where(BriefPreset.user_id == owner,
                                                       BriefPreset.id != r.id))).scalars().all()
                    if any(_same_name(n, name) for n in others):
                        raise HTTPException(status_code=409,
                                            detail="A preset with that name exists")
                    r.name = name
                if brief is not None:
                    r.brief = brief
                r.updated_at = datetime.now(timezone.utc)
                await session.flush()
                return _row_dict(r)

    async def delete(self, owner, preset_id):
        from sqlalchemy import delete
        from cloud import database
        from cloud.models import BriefPreset
        async with database.session() as session:
            async with session.begin():
                res = await session.execute(
                    delete(BriefPreset).where(BriefPreset.id == self._uid(preset_id),
                                              BriefPreset.user_id == owner))
                return res.rowcount > 0


_file_store = _FileStore()
_db_store = _DbStore()


def _store():
    return _db_store if _cfg["billing"] else _file_store


async def _owner(request):
    """The signed-in user's id (cloud; 401 without one), None in self-host."""
    if not _cfg["billing"]:
        return None
    resolver = _cfg["resolver"]
    user = await resolver(request) if resolver else None
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user.id


# --------------------------------------------------------------------------- #
# Helper for /api/process
# --------------------------------------------------------------------------- #
async def get_preset(owner, preset_id):
    """The caller's preset ``{id, name, brief, updated_at}``; 404 when it is
    malformed or not theirs. ``owner`` is the cloud user id (None in self-host)."""
    pid = _check_id(preset_id)
    row = await _store().get(owner, pid)
    if row is None:
        raise HTTPException(status_code=404, detail="Preset not found")
    return row


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
async def _json_body(request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Send a JSON body")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Send a JSON object")
    return body


@router.get("/api/brief-presets")
async def list_presets(request: Request):
    owner = await _owner(request)
    rows = await _store().list(owner)
    return {"presets": [_public(r) for r in rows], "max_presets": MAX_PRESETS,
            "max_brief_chars": max_brief_chars(), "max_name_chars": MAX_NAME_CHARS}


@router.post("/api/brief-presets")
async def create_preset(request: Request):
    owner = await _owner(request)
    body = await _json_body(request)
    name = clean_name(body.get("name"))
    brief = clean_brief(body.get("brief"))
    return _public(await _store().create(owner, name, brief))


@router.put("/api/brief-presets/{preset_id}")
async def update_preset(preset_id: str, request: Request):
    owner = await _owner(request)
    pid = _check_id(preset_id)
    body = await _json_body(request)
    name = clean_name(body["name"]) if body.get("name") is not None else None
    brief = clean_brief(body["brief"]) if body.get("brief") is not None else None
    row = await _store().update(owner, pid, name, brief)
    if row is None:
        raise HTTPException(status_code=404, detail="Preset not found")
    return _public(row)


@router.delete("/api/brief-presets/{preset_id}")
async def delete_preset(preset_id: str, request: Request):
    owner = await _owner(request)
    pid = _check_id(preset_id)
    if not await _store().delete(owner, pid):
        raise HTTPException(status_code=404, detail="Preset not found")
    return {"success": True}
