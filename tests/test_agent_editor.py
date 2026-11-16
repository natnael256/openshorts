"""The agent-as-editor endpoints: GET /api/jobs/{id}/transcript and
POST /api/clip/create, plus their MCP tools and OAuth scope.

Same conventions as test_rerender_endpoint.py: a real ASGI round-trip against
the imported app (BILLING_ENABLED=0 via conftest), private OUTPUT_DIR /
UPLOAD_DIR under tmp_path, and the render stubbed at recut.perform_recut.
"""

import asyncio
import json
import os

import httpx
import pytest

app_module = pytest.importorskip("app")
import recut  # noqa: E402
from cloud import scopes  # noqa: E402

JOB_ID = "agent-editor-test-job"

TRANSCRIPT = {
    "language": "en",
    "segments": [
        {"start": 0.0, "end": 10.004, "text": " first thought ",
         "words": [{"word": "first", "start": 1.0, "end": 1.456},
                   {"word": "thought", "start": 2.0, "end": 2.5}]},
        {"start": 10.0, "end": 20.0, "text": "second thought",
         "words": [{"word": "second", "start": 12.0, "end": 12.5},
                   {"word": "thought", "start": 13.0, "end": 13.5}]},
        {"start": 50.0, "end": 60.0, "text": "third",
         "words": [{"word": "third", "start": 51.0, "end": 51.5}]},
    ],
}


def _request(method, path, json_body=None, params=None):
    async def _do():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://testserver") as client:
            return await client.request(method, path, json=json_body, params=params)
    return asyncio.run(_do())


@pytest.fixture()
def job(tmp_path, monkeypatch):
    out_root = tmp_path / "output"
    up_root = tmp_path / "uploads"
    job_dir = out_root / JOB_ID
    job_dir.mkdir(parents=True)
    up_root.mkdir()
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(out_root))
    monkeypatch.setattr(app_module, "UPLOAD_DIR", str(up_root))

    clip = {"start": 10.0, "end": 40.0,
            "video_title_for_youtube_short": "test clip",
            "video_url": f"/videos/{JOB_ID}/mytitle_clip_1.mp4"}
    meta = {"shorts": [clip], "transcript": TRANSCRIPT,
            "source_video": "src.mp4", "output_format": "9:16",
            "source_subtitle_band": [0.8, 1.0], "cost_analysis": {}}
    (job_dir / "mytitle_metadata.json").write_text(json.dumps(meta))
    (job_dir / "mytitle_clip_1.mp4").write_bytes(b"canonical")
    (job_dir / "src.mp4").write_bytes(b"source")

    app_module.jobs[JOB_ID] = {
        "status": "completed", "logs": [],
        "result": {"clips": [dict(clip)], "cost_analysis": {}},
        "user_id": None, "watermark": False,
    }
    try:
        yield {"dir": job_dir, "meta_path": job_dir / "mytitle_metadata.json"}
    finally:
        app_module.jobs.pop(JOB_ID, None)


@pytest.fixture()
def fake_recut(monkeypatch):
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        name = f"recut_1_{kwargs['clean_name']}"
        with open(os.path.join(kwargs["output_dir"], name), "wb") as f:
            f.write(b"recut")
        return name, name

    monkeypatch.setattr(recut, "perform_recut", fake)
    return calls


def _meta(job):
    return json.loads(job["meta_path"].read_text())


# --------------------------------------------------------------------------- #
# GET /api/jobs/{id}/transcript
# --------------------------------------------------------------------------- #
class TestTranscript:
    def test_whole_transcript_rounded(self, job):
        resp = _request("GET", f"/api/jobs/{JOB_ID}/transcript")
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == JOB_ID
        assert data["language"] == "en"
        assert "duration" in data
        assert data["segments"][0] == {"i": 0, "start": 0.0, "end": 10.0,
                                       "text": "first thought"}
        assert [s["i"] for s in data["segments"]] == [0, 1, 2]
        assert "words" not in data
        assert "note" not in data

    def test_slice_keeps_overlapping_segments_and_their_indices(self, job):
        data = _request("GET", f"/api/jobs/{JOB_ID}/transcript",
                        params={"start": 15, "end": 55}).json()
        assert [s["i"] for s in data["segments"]] == [1, 2]

    def test_words_for_the_range(self, job):
        data = _request("GET", f"/api/jobs/{JOB_ID}/transcript",
                        params={"words": "true", "start": 0, "end": 12.2}).json()
        assert data["words"] == [{"w": "first", "s": 1.0, "e": 1.46},
                                 {"w": "thought", "s": 2.0, "e": 2.5},
                                 {"w": "second", "s": 12.0, "e": 12.5}]

    def test_inverted_range_400(self, job):
        resp = _request("GET", f"/api/jobs/{JOB_ID}/transcript",
                        params={"start": 20, "end": 10})
        assert resp.status_code == 400

    def test_silent_source_has_a_note(self, job):
        meta = _meta(job)
        meta["transcript"] = {"language": "none", "segments": []}
        job["meta_path"].write_text(json.dumps(meta))
        data = _request("GET", f"/api/jobs/{JOB_ID}/transcript",
                        params={"words": "true"}).json()
        assert data["segments"] == []
        assert data["words"] == []
        assert data["language"] == "none"
        assert data["note"]

    def test_404_unknown_job_and_missing_metadata(self, job):
        assert _request("GET", "/api/jobs/nope/transcript").status_code == 404
        os.remove(job["meta_path"])
        assert _request("GET", f"/api/jobs/{JOB_ID}/transcript").status_code == 404

    def test_owner_check_applies(self, job, monkeypatch):
        seen = []

        async def deny(request, record):
            seen.append(record)
            raise app_module.HTTPException(status_code=404, detail="Job not found")

        monkeypatch.setattr(app_module, "_assert_job_owner", deny)
        resp = _request("GET", f"/api/jobs/{JOB_ID}/transcript")
        assert resp.status_code == 404
        assert seen and seen[0] is app_module.jobs[JOB_ID]


# --------------------------------------------------------------------------- #
# POST /api/clip/create
# --------------------------------------------------------------------------- #
class TestCreateClip:
    def _create(self, **body):
        payload = {"job_id": JOB_ID, "segments": [{"start": 45, "end": 55}],
                   "snap_to_words": False}
        payload.update(body)
        return _request("POST", "/api/clip/create", payload)

    def test_appends_a_new_clip_from_the_source(self, job, fake_recut):
        resp = self._create(segments=[{"start": 45, "end": 55},
                                      {"start": 12, "end": 14}],
                            title="Big idea")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["clip_index"] == 1
        assert data["new_video_url"] == \
            f"/videos/{JOB_ID}/recut_1_mytitle_clip_2.mp4"
        assert data["duration"] == 12.0

        call = fake_recut[0]
        assert call["input_path"].endswith("src.mp4")
        assert call["reframe"] is True
        assert call["clean_name"] == "mytitle_clip_2.mp4"
        assert call["output_format"] == "9:16"
        assert call["subtitle_band"] == [0.8, 1.0]
        assert call["force_strategy"] is None
        assert not call.get("watermark")
        assert call["captions_transcript"]["segments"]

        meta = _meta(job)
        assert len(meta["shorts"]) == 2
        new = meta["shorts"][1]
        assert new["created_by"] == "agent"
        assert new["title"] == new["video_title_for_youtube_short"] == "Big idea"
        assert (new["start"], new["end"]) == (12.0, 55.0)
        assert new["recipe"] == {
            "v": 1,
            "segments": [{"start": 45.0, "end": 55.0}, {"start": 12.0, "end": 14.0}],
            "canonical_range": {"start": 12.0, "end": 55.0}}
        assert new["video_url"] == data["new_video_url"]
        assert "layout_ranges" in new
        # The original clip is untouched.
        assert meta["shorts"][0]["video_url"].endswith("mytitle_clip_1.mp4")
        mem = app_module.jobs[JOB_ID]["result"]["clips"]
        assert len(mem) == 2 and mem[1]["video_url"] == data["new_video_url"]

    def test_framing_and_captions_flags(self, job, fake_recut):
        resp = self._create(framing="full", reapply_captions=False)
        assert resp.status_code == 200
        assert fake_recut[0]["force_strategy"] == "WIDE"
        assert fake_recut[0]["captions_transcript"] is None
        assert fake_recut[0]["subtitle_band"] is None
        assert _meta(job)["shorts"][1]["recipe"]["framing"] == "full"

    def test_indices_stay_stable_past_a_tombstone(self, job, fake_recut):
        meta = _meta(job)
        meta["shorts"].append({"deleted": True})
        job["meta_path"].write_text(json.dumps(meta))
        app_module.jobs[JOB_ID]["result"]["clips"].append({"deleted": True})

        data = self._create().json()
        assert data["clip_index"] == 2
        assert fake_recut[0]["clean_name"] == "mytitle_clip_3.mp4"
        shorts = _meta(job)["shorts"]
        assert shorts[1] == {"deleted": True}
        assert shorts[2]["created_by"] == "agent"
        assert shorts[2]["title"] == "Clip 3"

    def test_title_is_sanitised(self, job, fake_recut):
        raw = "  Hello\x00\x1b[31m‮ world\n\t" + "x" * 300
        self._create(title=raw)
        title = _meta(job)["shorts"][1]["title"]
        assert len(title) <= 120
        assert title.startswith("Hello [31m world x")
        assert not any(ord(c) < 32 or c == "‮" for c in title)

    def test_blank_title_falls_back(self, job, fake_recut):
        self._create(title="\x00 \n")
        assert _meta(job)["shorts"][1]["title"] == "Clip 2"

    def test_409_without_source(self, job, fake_recut):
        os.remove(job["dir"] / "src.mp4")
        resp = self._create()
        assert resp.status_code == 409
        assert fake_recut == []
        assert len(_meta(job)["shorts"]) == 1

    def test_400_bad_segments(self, job, fake_recut):
        for segments in ([], [{"start": 15, "end": 15.1}],
                         [{"start": 0, "end": 100}, {"start": 100, "end": 200}]):
            resp = self._create(segments=segments)
            assert resp.status_code == 400, segments
        assert self._create(framing="cinematic").status_code == 400
        assert fake_recut == []
        assert len(_meta(job)["shorts"]) == 1

    def test_clip_cap_enforced(self, job, fake_recut):
        meta = _meta(job)
        meta["shorts"] += [{"deleted": True}] * (app_module.MAX_CLIPS_PER_JOB - 1)
        job["meta_path"].write_text(json.dumps(meta))
        resp = self._create()
        assert resp.status_code == 409
        assert fake_recut == []

    def test_404_unknown_job(self, job, fake_recut):
        resp = _request("POST", "/api/clip/create", {
            "job_id": "nope", "segments": [{"start": 1, "end": 5}]})
        assert resp.status_code == 404

    def test_snap_to_words_is_the_default(self, job, fake_recut, monkeypatch):
        snapped = []

        def fake_snap(segments, transcript, bound):
            snapped.append(segments)
            return segments

        monkeypatch.setattr(recut, "snap_segments", fake_snap)
        resp = _request("POST", "/api/clip/create", {
            "job_id": JOB_ID, "segments": [{"start": 45, "end": 55}]})
        assert resp.status_code == 200
        assert snapped

    def test_created_clip_can_then_be_rerendered(self, job, fake_recut):
        idx = self._create().json()["clip_index"]
        resp = _request("POST", "/api/clip/rerender", {
            "job_id": JOB_ID, "clip_index": idx,
            "segments": [{"start": 46, "end": 54}]})
        assert resp.status_code == 200
        # No canonical file exists for an agent clip: always the source path.
        assert resp.json()["render_path"] == "source"
        assert fake_recut[1]["clean_name"] == "mytitle_clip_2.mp4"


# --------------------------------------------------------------------------- #
# MCP + scope
# --------------------------------------------------------------------------- #
class TestMcpAndScope:
    def test_tools_registered_with_hints(self):
        import mcp_server
        by_name = {t["name"]: t for t in mcp_server.TOOLS}
        for name in ("get_transcript", "create_clip"):
            assert name in by_name and name in mcp_server._TOOL_IMPLS
        assert by_name["get_transcript"]["annotations"]["readOnlyHint"] is True
        assert by_name["create_clip"]["annotations"]["readOnlyHint"] is False
        assert "boundaries" in by_name["get_transcript"]["description"]

    def test_routes_in_mcp_scope(self):
        assert scopes.allows("openshorts", "GET", "/api/jobs/abc-123/transcript")
        assert scopes.allows("openshorts", "POST", "/api/clip/create")
        assert not scopes.allows("openshorts", "GET", "/api/jobs/../keys/transcript")
        assert not scopes.allows("openshorts", "DELETE", "/api/clip/create")

    def test_create_clip_tool_returns_index_and_download_url(self, job, fake_recut):
        import mcp_server

        async def _do():
            transport = httpx.ASGITransport(app=app_module.app)
            async with httpx.AsyncClient(transport=transport,
                                         base_url="http://testserver") as client:
                return await mcp_server._tool_create_clip(client, {
                    "job_id": JOB_ID, "segments": [{"start": 45, "end": 55}],
                    "snap_to_words": False, "title": "From the agent"})
        out, is_error = asyncio.run(_do())
        assert is_error is False
        assert out["clip_index"] == 1
        assert out["download_url"].endswith("recut_1_mytitle_clip_2.mp4")

    def test_get_transcript_tool_passes_the_slice(self, job):
        import mcp_server

        async def _do():
            transport = httpx.ASGITransport(app=app_module.app)
            async with httpx.AsyncClient(transport=transport,
                                         base_url="http://testserver") as client:
                return await mcp_server._tool_get_transcript(client, {
                    "job_id": JOB_ID, "words": True, "start": 50, "end": 60})
        out, is_error = asyncio.run(_do())
        assert is_error is False
        assert [s["i"] for s in out["segments"]] == [2]
        assert [w["w"] for w in out["words"]] == ["third"]
