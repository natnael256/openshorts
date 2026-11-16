"""Saved brief presets: self-host JSON store through the real router, plus the
account-erasure registration of the cloud table."""
import asyncio
import json
import os

import httpx
import pytest
from fastapi import FastAPI

import brief_presets
import creator_brief

GOOD_ID = "0" * 32


@pytest.fixture()
def client(tmp_path):
    brief_presets.configure(output_dir=str(tmp_path), billing_enabled=False,
                            user_resolver=None)
    api = FastAPI()
    api.include_router(brief_presets.router)

    def call(method, path, body=None):
        async def _do():
            transport = httpx.ASGITransport(app=api)
            async with httpx.AsyncClient(transport=transport,
                                         base_url="http://testserver") as c:
                return await c.request(method, path, json=body)
        return asyncio.run(_do())
    call.dir = tmp_path
    return call


def _create(client, name="Campaign X", brief="No profanity. Hook in 3 seconds."):
    return client("POST", "/api/brief-presets", {"name": name, "brief": brief})


def test_crud_roundtrip(client):
    r = _create(client)
    assert r.status_code == 200, r.text
    preset = r.json()
    assert set(preset) == {"id", "name", "brief", "updated_at"}
    assert len(preset["id"]) == 32

    listing = client("GET", "/api/brief-presets").json()
    assert listing["max_presets"] == 50
    assert listing["max_brief_chars"] == creator_brief.MAX_BRIEF_CHARS
    assert listing["max_name_chars"] == 60
    assert [p["id"] for p in listing["presets"]] == [preset["id"]]

    u = client("PUT", f"/api/brief-presets/{preset['id']}", {"brief": "New rules"})
    assert u.status_code == 200 and u.json()["brief"] == "New rules"
    assert u.json()["name"] == "Campaign X"

    d = client("DELETE", f"/api/brief-presets/{preset['id']}")
    assert d.json() == {"success": True}
    assert client("GET", "/api/brief-presets").json()["presets"] == []


def test_newest_first(client):
    a = _create(client, "A").json()
    b = _create(client, "B").json()
    client("PUT", f"/api/brief-presets/{a['id']}", {"brief": "touched"})
    ids = [p["id"] for p in client("GET", "/api/brief-presets").json()["presets"]]
    assert ids == [a["id"], b["id"]]


@pytest.mark.parametrize("body", [
    {"name": "", "brief": "x"},
    {"name": "   ", "brief": "x"},
    {"name": "n" * 61, "brief": "x"},
    {"name": "ok", "brief": ""},
    {"name": "ok", "brief": "   "},
    {"name": "ok", "brief": "b" * (creator_brief.MAX_BRIEF_CHARS + 1)},
    {"name": 5, "brief": "x"},
    {"name": "ok", "brief": ["x"]},
    {"brief": "x"},
])
def test_validation_errors_are_400(client, body):
    assert client("POST", "/api/brief-presets", body).status_code == 400


def test_non_object_body_is_400(client):
    assert client("POST", "/api/brief-presets", ["x"]).status_code == 400


def test_name_is_one_clean_line(client):
    r = _create(client, "  Camp\naign\t\x00 X  ")
    assert r.status_code == 200
    assert r.json()["name"] == "Camp aign X"


def test_duplicate_name_is_409(client):
    assert _create(client, "Same").status_code == 200
    assert _create(client, "same").status_code == 409
    other = _create(client, "Other").json()
    assert client("PUT", f"/api/brief-presets/{other['id']}", {"name": "SAME"}).status_code == 409


def test_cap_of_50(client):
    for i in range(brief_presets.MAX_PRESETS):
        assert _create(client, f"p{i}").status_code == 200
    assert _create(client, "one too many").status_code == 409


@pytest.mark.parametrize("bad", ["nothex", "..%2F..", "1234", "g" * 32])
def test_bad_ids_are_404(client, bad):
    assert client("PUT", f"/api/brief-presets/{bad}", {"name": "x"}).status_code == 404
    assert client("DELETE", f"/api/brief-presets/{bad}").status_code == 404


def test_unknown_id_is_404(client):
    assert client("PUT", f"/api/brief-presets/{GOOD_ID}", {"name": "x"}).status_code == 404
    assert client("DELETE", f"/api/brief-presets/{GOOD_ID}").status_code == 404


def test_corrupt_file_is_empty_then_recovers(client):
    (client.dir / ".brief_presets.json").write_text("{not json", encoding="utf-8")
    assert client("GET", "/api/brief-presets").json()["presets"] == []
    assert _create(client).status_code == 200
    data = json.loads((client.dir / ".brief_presets.json").read_text(encoding="utf-8"))
    assert len(data["presets"]) == 1


def test_wrong_shape_file_is_empty(client):
    (client.dir / ".brief_presets.json").write_text('{"presets": [1, {"id": 3}]}',
                                                    encoding="utf-8")
    assert client("GET", "/api/brief-presets").json()["presets"] == []


def test_store_is_a_dotfile_in_output_dir(client):
    _create(client)
    assert os.path.isfile(os.path.join(str(client.dir), ".brief_presets.json"))


def test_get_preset_helper(client):
    created = _create(client).json()
    got = asyncio.run(brief_presets.get_preset(None, created["id"]))
    assert got["name"] == "Campaign X"
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as e:
        asyncio.run(brief_presets.get_preset(None, GOOD_ID))
    assert e.value.status_code == 404
    with pytest.raises(HTTPException) as e:
        asyncio.run(brief_presets.get_preset(None, "../x"))
    assert e.value.status_code == 404


def test_cloud_mode_requires_a_user(tmp_path):
    async def nobody(request):
        return None
    brief_presets.configure(output_dir=str(tmp_path), billing_enabled=True,
                            user_resolver=nobody)
    try:
        api = FastAPI()
        api.include_router(brief_presets.router)

        async def _do():
            transport = httpx.ASGITransport(app=api)
            async with httpx.AsyncClient(transport=transport,
                                         base_url="http://testserver") as c:
                return [(await c.get("/api/brief-presets")).status_code,
                        (await c.post("/api/brief-presets",
                                      json={"name": "a", "brief": "b"})).status_code,
                        (await c.put(f"/api/brief-presets/{GOOD_ID}",
                                     json={"name": "a"})).status_code,
                        (await c.delete(f"/api/brief-presets/{GOOD_ID}")).status_code]
        assert asyncio.run(_do()) == [401, 401, 401, 401]
    finally:
        brief_presets.configure(output_dir=None, billing_enabled=False, user_resolver=None)


def test_model_is_registered_for_account_erasure():
    pytest.importorskip("sqlalchemy")
    from cloud import account
    from cloud.models import BriefPreset
    assert BriefPreset in account.USER_OWNED_TABLES
    assert BriefPreset.__tablename__ == "brief_presets"
