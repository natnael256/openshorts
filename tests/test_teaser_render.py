"""Real-render evidence for the teaser cold open (needs ffmpeg + main.py's stack).

``run_synthetic(workdir)`` builds a 30 s 1080x1920 test clip with ffmpeg lavfi,
a fake word-level transcript, renders the teaser recut through recut.py
(real ffmpeg), captions it through main.auto_caption_clip with
teaser.caption_inputs, and checks durations with ffprobe and caption timing
in the generated .ass. The pytest wrapper skips where ffmpeg or ``import
main`` is unavailable (the CI image), so CI stays green; run it directly in
the real image:

    python -c "import sys; sys.path.insert(0, '/app'); \\
        from tests.test_teaser_render import run_synthetic; \\
        print(run_synthetic('/tmp/teaser_check'))"
"""
import glob
import os
import re
import shutil
import subprocess
import tempfile

try:
    import pytest
except ImportError:      # the real image ships without pytest; run_synthetic still works
    pytest = None

SOURCE_START, SOURCE_END = 100.0, 130.0        # the canonical clip's source range
TEASER = (112.0, 115.0)                        # a 3 s beat, 12 s into the clip
BODY_FIRST_WORD = "Zulu"
TEASER_WORDS = ("Tango", "Uniform", "Victor", "Whiskey", "Xray", "Yankee")


def _ffprobe_duration(path):
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path], timeout=60)
    return float(out.decode().strip())


def _make_clip(path, seconds):
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "testsrc2=size=1080x1920:rate=30",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
         "-t", str(seconds), "-c:v", "libx264", "-preset", "ultrafast",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", path],
        check=True, timeout=300)


def _transcript():
    """A word every 0.5 s over the source range; the teaser beat says
    TEASER_WORDS and the clip's very first word is BODY_FIRST_WORD."""
    words, t, i = [], SOURCE_START + 0.2, 0
    teaser_iter = iter(TEASER_WORDS)
    while t + 0.35 < SOURCE_END:
        if t == SOURCE_START + 0.2:
            text = BODY_FIRST_WORD
        elif TEASER[0] <= t and t + 0.35 <= TEASER[1]:
            text = next(teaser_iter, "Echo")
        else:
            text = f"word{i}"
        words.append({"word": " " + text, "start": round(t, 3), "end": round(t + 0.35, 3)})
        t = round(t + 0.5, 3)
        i += 1
    return {"language": "en", "segments": [
        {"start": SOURCE_START, "end": SOURCE_END,
         "text": "".join(w["word"] for w in words).strip(), "words": words}]}


def _ass_seconds(stamp):
    h, m, s = stamp.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _events(ass_path):
    out = []
    with open(ass_path, encoding="utf-8-sig") as f:
        for line in f:
            if line.startswith("Dialogue:"):
                parts = line.rstrip("\n").split(",", 9)
                out.append((_ass_seconds(parts[1]), _ass_seconds(parts[2]), parts[9]))
    return out


def _active(word, text):
    # the active word carries the highlight prefix ({\c...}) and ends with {\r}
    return re.search(r"\{\\c[^}]*\}" + re.escape(word) + r"\{\\r\}", text, re.IGNORECASE)


def run_synthetic(workdir):
    import layout_ranges
    import main
    import teaser

    os.makedirs(workdir, exist_ok=True)
    work = tempfile.mkdtemp(prefix="teaser_", dir=workdir)
    canonical = os.path.join(work, "synthetic_clip_1.mp4")
    _make_clip(canonical, SOURCE_END - SOURCE_START)
    # the reframe's layout sidecar: second half stacked (SPLIT)
    layout_ranges.write(canonical, [(0.0, 15.0, "track"), (15.0, 30.0, "split")])

    transcript = _transcript()
    clip = {"start": SOURCE_START, "end": SOURCE_END, "opening": "slow",
            "teaser_pick": {"start": TEASER[0], "end": TEASER[1]}}
    segments = teaser.plan(clip, SOURCE_START, SOURCE_END)
    assert segments, "plan() refused the synthetic teaser"

    rendered = teaser.render(canonical, clip, segments, SOURCE_START, SOURCE_END,
                             work, os.path.basename(canonical))
    assert rendered, "teaser.render failed"
    teased_path, ranges = rendered
    teaser_len = TEASER[1] - TEASER[0]
    body_len = SOURCE_END - SOURCE_START

    v_transcript, cap_start, cap_end = teaser.caption_inputs(transcript, segments)
    captioned = main.auto_caption_clip(
        teased_path, v_transcript, cap_start, cap_end,
        split_ranges=layout_ranges.split_ranges(ranges))
    assert captioned and os.path.exists(captioned), "captioning failed"

    ass_files = sorted(glob.glob(os.path.join(work, "autosubs_*.ass")), key=os.path.getmtime)
    assert ass_files, "no .ass written"
    events = _events(ass_files[-1])
    teaser_events = [e for e in events if _active(TEASER_WORDS[0], e[2])]
    body_events = [e for e in events if _active(BODY_FIRST_WORD, e[2])]

    # The hook over the teaser: shown for the teaser's length (min 2.5 s),
    # larger, burned in the same ffmpeg as the captions (as the pipeline does).
    clip["viral_hook_text"] = "Tango Uniform changes everything"
    clip["recipe"] = teaser.recipe(segments, SOURCE_START, SOURCE_END)
    clip["layout_ranges"] = ranges
    saved_secs = os.environ.pop("AUTO_HOOK_SECONDS", None)
    try:
        plan = main.auto_caption_clip(teased_path, v_transcript, cap_start, cap_end,
                                      split_ranges=layout_ranges.split_ranges(ranges),
                                      plan_only=True)
        hooked = main.auto_hook_clip(teased_path, clip, captions=plan)
    finally:
        if saved_secs is not None:
            os.environ["AUTO_HOOK_SECONDS"] = saved_secs
    assert hooked, "auto_hook_clip failed on the recut"
    hooked_path, hook_config = hooked
    hooked_captioned = hook_config.pop("_captioned", None)

    result = {
        "hook_duration_s": hook_config.get("duration_seconds"),
        "hook_size": hook_config.get("size"),
        # the hooked-only picture may be kept as a recipe (hooks.materialize)
        "hooked_s": (round(_ffprobe_duration(hooked_path), 3)
                     if os.path.exists(hooked_path) else "lazy (recipe)"),
        "hooked_captioned_s": (round(_ffprobe_duration(hooked_captioned), 3)
                               if hooked_captioned and os.path.exists(hooked_captioned)
                               else None),
        "canonical_s": round(_ffprobe_duration(canonical), 3),
        "teased_s": round(_ffprobe_duration(teased_path), 3),
        "captioned_s": round(_ffprobe_duration(captioned), 3),
        "expected_s": body_len + teaser_len,
        "first_event_s": round(min(e[0] for e in events), 3),
        "teaser_first_word_s": round(teaser_events[0][0], 3) if teaser_events else None,
        "body_first_word_s": round(body_events[0][0], 3) if body_events else None,
        "teaser_len_s": teaser_len,
        "events": len(events),
        "layout_ranges": ranges,
        "canonical_untouched": not os.path.basename(teased_path) == os.path.basename(canonical),
        "files": sorted(os.path.basename(p) for p in os.listdir(work)),
    }
    assert abs(result["teased_s"] - result["expected_s"]) <= 0.25, result
    assert abs(result["captioned_s"] - result["teased_s"]) <= 0.25, result
    assert abs(result["canonical_s"] - body_len) <= 0.25, result
    assert result["hook_duration_s"] == max(2.5, teaser_len) and result["hook_size"] == "L", result
    assert result["hooked_captioned_s"] is not None, result
    assert abs(result["hooked_captioned_s"] - result["teased_s"]) <= 0.25, result
    assert result["first_event_s"] <= 0.3, result
    assert result["teaser_first_word_s"] is not None and result["teaser_first_word_s"] <= 0.3, result
    assert result["body_first_word_s"] is not None, result
    assert abs(result["body_first_word_s"] - (teaser_len + 0.2)) <= 0.15, result
    # SPLIT stretch carried through the cut: clip 15..30 lands at 18..33
    splits = layout_ranges.split_ranges(ranges)
    assert splits and abs(splits[0][0] - (15.0 + teaser_len)) <= 0.05, result
    return result


def _stack_available():
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        return False
    try:
        import main  # noqa: F401
    except Exception:
        return False
    return True


def test_teaser_render_synthetic(tmp_path):
    if not _stack_available():
        pytest.skip("needs ffmpeg and main.py's ML stack (run in the real image)")
    result = run_synthetic(str(tmp_path))
    assert result["teased_s"] > result["canonical_s"]
