"""teaser: plan() skip rules, caption timing, render() with recut faked."""
import os

import pytest

import creator_brief
import recut
import teaser


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in ("AUTO_TEASER", "TEASER_MIN_CLIP_SECONDS", "TEASER_MIN_OFFSET"):
        monkeypatch.delenv(k, raising=False)
    creator_brief.activate("", None)
    yield
    creator_brief.activate("", None)


def _clip(ts=20.0, te=23.0, opening="slow"):
    return {"start": 10.0, "end": 40.0, "opening": opening,
            "teaser_pick": {"start": ts, "end": te}}


def test_plan_happy_path():
    segs = teaser.plan(_clip(), 10.0, 40.0)
    assert segs == [{"start": 20.0, "end": 23.0}, {"start": 10.0, "end": 40.0}]
    rec = teaser.recipe(segs, 10.0, 40.0)
    assert rec == {"v": 1, "segments": segs,
                   "canonical_range": {"start": 10.0, "end": 40.0}, "teaser": True}
    assert teaser.teaser_seconds({"recipe": rec}) == pytest.approx(3.0)
    assert teaser.teaser_seconds({"recipe": {"v": 1, "segments": segs}}) == 0.0
    assert teaser.teaser_seconds({}) == 0.0


def test_plan_env_off(monkeypatch):
    monkeypatch.setenv("AUTO_TEASER", "0")
    assert teaser.plan(_clip(), 10.0, 40.0) is None


def test_plan_brief_off():
    creator_brief.activate("no teaser please", {"teaser": "off"})
    assert teaser.plan(_clip(), 10.0, 40.0) is None


def test_plan_brief_max_length_covers_teaser():
    creator_brief.activate("max 31s", {"max_clip_seconds": 31})
    assert teaser.plan(_clip(), 10.0, 40.0) is None       # 30 + 3 > 31 + 1
    creator_brief.activate("max 40s", {"max_clip_seconds": 40})
    assert teaser.plan(_clip(), 10.0, 40.0) is not None


def test_plan_no_pick_or_garbage():
    c = _clip()
    del c["teaser_pick"]
    assert teaser.plan(c, 10.0, 40.0) is None
    assert teaser.plan({"teaser_pick": "20-23"}, 10.0, 40.0) is None
    assert teaser.plan({"teaser_pick": {"start": "x", "end": 3}}, 10.0, 40.0) is None
    assert teaser.plan(None, 10.0, 40.0) is None


def test_plan_strong_opening_skips():
    assert teaser.plan(_clip(opening="strong"), 10.0, 40.0) is None


def test_plan_short_clip(monkeypatch):
    assert teaser.plan(_clip(ts=18.0, te=21.0), 10.0, 24.0) is None    # 14 s < 15
    monkeypatch.setenv("TEASER_MIN_CLIP_SECONDS", "10")
    assert teaser.plan(_clip(ts=18.0, te=21.0), 10.0, 24.0) is not None


def test_plan_pick_outside_final_bounds():
    assert teaser.plan(_clip(ts=38.0, te=41.0), 10.0, 40.0) is None
    assert teaser.plan(_clip(ts=8.0, te=11.0), 10.0, 40.0) is None


def test_plan_too_soon_after_start(monkeypatch):
    assert teaser.plan(_clip(ts=14.0, te=17.0), 10.0, 40.0) is None    # 4 s < 5
    monkeypatch.setenv("TEASER_MIN_OFFSET", "3")
    assert teaser.plan(_clip(ts=14.0, te=17.0), 10.0, 40.0) is not None


def test_plan_teaser_length_bounds():
    assert teaser.plan(_clip(ts=20.0, te=21.0), 10.0, 40.0) is None    # 1 s
    assert teaser.plan(_clip(ts=20.0, te=25.0), 10.0, 40.0) is None    # 5 s
    assert teaser.plan(_clip(ts=20.0, te=21.5), 10.0, 40.0) is not None
    assert teaser.plan(_clip(ts=20.0, te=24.5), 10.0, 40.0) is not None


def test_plan_total_over_recut_max():
    end = 10.0 + recut.MAX_TOTAL_SECONDS - 1.0
    assert teaser.plan(_clip(), 10.0, end) is None


# --- captions ---------------------------------------------------------------

def _transcript():
    words = []
    t = 10.0
    for i in range(70):          # one word every 0.4 s from 10 s to 38 s
        words.append({"word": f" w{i}", "start": round(t, 3), "end": round(t + 0.3, 3)})
        t += 0.4
    return {"language": "en", "segments": [{"start": 10.0, "end": 40.0,
                                            "text": "", "words": words}]}


def test_caption_inputs_timing():
    tr = _transcript()
    segs = teaser.plan(_clip(), 10.0, 40.0)
    vt, s, e = teaser.caption_inputs(tr, segs)
    assert (s, e) == (0.0, pytest.approx(33.0))
    assert vt == recut.virtual_transcript(tr, segs)
    teaser_words, body_words = vt["segments"][0]["words"], vt["segments"][1]["words"]
    # teaser words: source 20..23 -> clip 0..3
    assert teaser_words[0]["word"] == " w25" and teaser_words[0]["start"] == pytest.approx(0.0)
    assert all(0.0 <= w["start"] < 3.0 for w in teaser_words)
    # body words shifted by the teaser length
    assert body_words[0]["word"] == " w0"
    assert body_words[0]["start"] == pytest.approx(3.0)
    src = {w["word"]: w["start"] for w in tr["segments"][0]["words"]}
    for w in body_words:
        assert w["start"] == pytest.approx(src[w["word"]] - 10.0 + 3.0, abs=1e-3)


# --- render -----------------------------------------------------------------

def test_render_success_shape(tmp_path, monkeypatch):
    import layout_ranges
    seen = {}

    def fake(**kw):
        seen.update(kw)
        name = "recut_1_abc_" + kw["clean_name"]
        path = os.path.join(kw["output_dir"], name)
        with open(path, "wb") as f:
            f.write(b"x")
        layout_ranges.write(path, [(0, 3, "split"), (3, 33, "track")])
        return name, name

    monkeypatch.setattr(recut, "perform_recut", fake)
    segs = teaser.plan(_clip(), 10.0, 40.0)
    out = teaser.render(str(tmp_path / "c.mp4"), _clip(), segs, 10.0, 40.0,
                        str(tmp_path), "c.mp4")
    assert out is not None
    path, ranges = out
    assert os.path.basename(path) == "recut_1_abc_c.mp4"
    assert [r["layout"] for r in ranges] == ["split", "track"]
    assert seen["reframe"] is False and seen["captions_transcript"] is None
    assert seen["segments"] == [{"start": 10.0, "end": 13.0}, {"start": 0.0, "end": 30.0}]
    assert seen["input_path"].endswith("c.mp4")


def test_render_exception_returns_none(tmp_path, monkeypatch, capsys):
    def boom(**kw):
        raise RuntimeError("ffmpeg failed: secret path /x/y")

    monkeypatch.setattr(recut, "perform_recut", boom)
    segs = teaser.plan(_clip(), 10.0, 40.0)
    assert teaser.render("c.mp4", _clip(), segs, 10.0, 40.0, str(tmp_path), "c.mp4") is None
    assert "secret path" not in capsys.readouterr().out


def test_render_missing_output_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(recut, "perform_recut", lambda **kw: ("gone.mp4", "gone.mp4"))
    segs = teaser.plan(_clip(), 10.0, 40.0)
    assert teaser.render("c.mp4", _clip(), segs, 10.0, 40.0, str(tmp_path), "c.mp4") is None
