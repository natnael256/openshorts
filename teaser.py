"""Cold open: replay one strong line of a clip before it starts, as an
automatic recut.

The hook pass (auto_hooks) may pick one beat of a clip as a teaser
(``clip['teaser_pick']``, absolute source seconds). Here that becomes a clip
made of two source segments, ``[teaser, body]``, rendered by the clip
editor's own engine (``recut.perform_recut``, fast path: cut from the
already-reframed canonical clip, layout sidecar carried by
``layout_ranges.remap``). The clip's metadata then stores the same
``recipe`` the editor's rerender stores, so the editor shows the teaser as an
ordinary first segment the user can remove, and every caption path
(/api/subtitle, _reapply_captions, the EDL endpoint) captions through
``recut.virtual_transcript``. The canonical clip stays untouched and clean.

Fail-open everywhere: ``plan`` returns None and ``render`` returns None
instead of raising, and the caller keeps the canonical clip.
See docs/architecture/auto-hooks.md.
"""
import os

import creator_brief
import recut

TEASER_MIN_SECONDS = 1.5
TEASER_MAX_SECONDS = 4.5
_EPS = 1e-6


def _env_float(name, default):
    try:
        raw = os.environ.get(name, "")
        return float(raw) if str(raw).strip() else default
    except (TypeError, ValueError):
        return default


def enabled():
    if os.environ.get("AUTO_TEASER", "1").strip() == "0":
        return False
    try:
        return creator_brief.active_directives().get("teaser") != "off"
    except Exception:
        return True


def _brief_max_seconds():
    try:
        mx = creator_brief.active_directives().get("max_clip_seconds") or 0
        return float(mx) if mx > 0 else float("inf")
    except Exception:
        return float("inf")


def plan(clip, start, end):
    """``[teaser, {start, end}]`` (source-absolute) or None when the clip
    should not get a cold open. ``start``/``end`` are the FINAL bounds the
    canonical clip was cut from (after word snapping)."""
    try:
        if not enabled() or not isinstance(clip, dict):
            return None
        pick = clip.get("teaser_pick")
        if not isinstance(pick, dict):
            return None
        if clip.get("opening") == "strong":
            return None          # it already opens strongly
        start, end = float(start), float(end)
        ts, te = float(pick["start"]), float(pick["end"])
        if not (ts == ts and te == te):   # NaN
            return None
        if end - start < _env_float("TEASER_MIN_CLIP_SECONDS", 15.0) - _EPS:
            return None
        if ts < start - _EPS or te > end + _EPS:
            return None          # bounds moved (snap / lead-in trim)
        if ts - start < _env_float("TEASER_MIN_OFFSET", 5.0) - _EPS:
            return None          # a replay seconds later is awkward
        length = te - ts
        if not (TEASER_MIN_SECONDS - _EPS <= length <= TEASER_MAX_SECONDS + _EPS):
            return None
        if (end - start) + length > recut.MAX_TOTAL_SECONDS:
            return None
        if (end - start) + length > _brief_max_seconds() + 1.0:
            return None          # the brief's max length covers the teaser too
        return [{"start": round(ts, 3), "end": round(te, 3)},
                {"start": round(start, 3), "end": round(end, 3)}]
    except Exception:
        return None


def render(canonical_path, clip, segments, start, end, output_dir, clean_name):
    """Cut ``segments`` out of the canonical clip (fast recut path, no
    reframe, no captions). Returns (recut_path, layout_ranges of the recut)
    or None on any failure. The canonical file is never modified."""
    try:
        import layout_ranges
        _served, out_name = recut.perform_recut(
            input_path=canonical_path,
            segments=recut.rebase_segments(segments, start, end),
            output_dir=output_dir, clean_name=clean_name,
            reframe=False, captions_transcript=None)
        path = os.path.join(output_dir, out_name)
        if not os.path.exists(path) or os.path.getsize(path) <= 0:
            return None
        return path, layout_ranges.read(path)
    except Exception as e:
        print(f"   ⚠️ Teaser recut failed ({type(e).__name__}); keeping the clip as cut.")
        return None


def recipe(segments, start, end):
    """The metadata the clip records, the shape the rerender endpoint stores
    (plus ``teaser``)."""
    return {"v": 1, "segments": [dict(s) for s in segments],
            "canonical_range": {"start": round(float(start), 3), "end": round(float(end), 3)},
            "teaser": True}


def teaser_seconds(clip):
    """Length of the clip's teaser segment, 0.0 when it has none."""
    try:
        rec = clip.get("recipe") or {}
        if not rec.get("teaser"):
            return 0.0
        first = rec["segments"][0]
        return max(0.0, float(first["end"]) - float(first["start"]))
    except Exception:
        return 0.0


def caption_inputs(transcript, segments):
    """(transcript, clip_start, clip_end) to caption the recut: the teaser's
    own words at 0..T and the body shifted by T."""
    return (recut.virtual_transcript(transcript, segments), 0.0,
            recut.total_duration(segments))
