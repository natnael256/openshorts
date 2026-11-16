"""
Pure helpers for the Gemini clip-selection pipeline.

Standard-library only so both main.py and gemini_worker.py can import it and
the logic stays unit-testable without the heavy video dependencies.
"""
import math
import re

# A word that closes a sentence (shared with auto_hooks.build_beats).
SENTENCE_END = re.compile(r"[.!?…。！？؟।]['\")\]”’]*$")
_CLAUSE_END = re.compile(r"[,;:—–，、；：]['\")\]”’]*$")

# USD per 1M tokens (input, output incl. thinking), from ai.google.dev pricing.
MODEL_PRICES = {
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3-flash-preview": (0.50, 3.00),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.0-flash": (0.10, 0.40),  # deprecated (shut down 2026-06-01)
}


def lookup_model_prices(model_name):
    """Longest-prefix match against MODEL_PRICES; None if unknown."""
    name = str(model_name or "").lower()
    best_key = None
    for key in MODEL_PRICES:
        if name.startswith(key) and (best_key is None or len(key) > len(best_key)):
            best_key = key
    return MODEL_PRICES[best_key] if best_key else None


# Gemini models a job may run on (app.py validates X-Gemini-Model against this
# list; keep in sync with GEMINI_MODELS in dashboard/src/App.jsx). A value off
# this list (a retired model a browser or a stored .submit.json still carries)
# is ignored everywhere, so the job runs on the default model instead of
# failing. gemini-2.5-flash / -flash-lite are retired (404 for new users).
GEMINI_MODEL_CHOICES = (
    "gemini-3.1-flash-lite", "gemini-3.5-flash", "gemini-3-flash-preview",
)


def boundary_model_name(default_model):
    """``(model, note)`` for the stages that decide where a clip starts and
    ends (the idea pass and the context check). ``GEMINI_BOUNDARY_MODEL``
    picks a model from GEMINI_MODEL_CHOICES for those stages only; unset, or
    off the list, they use ``default_model`` (the job's GEMINI_MODEL), so
    behaviour and cost do not change unless it is set. ``note`` is a log line
    for an ignored value (the value itself is not echoed), else None."""
    import os

    raw = (os.environ.get("GEMINI_BOUNDARY_MODEL") or "").strip()
    if not raw or raw == default_model:
        return default_model, None
    if raw in GEMINI_MODEL_CHOICES:
        return raw, None
    return default_model, ("GEMINI_BOUNDARY_MODEL is not an allowed model; "
                           f"boundaries use {default_model}.")


def clip_count_targets(n_windows):
    """How many clips to ask the detail pass for, given the shortlist size.

    Measured on prod 3-ago-2026: 408 of 429 jobs (95%) delivered 3 clips or
    fewer, the mode being ONE, while the prompt was free to return one per
    shortlisted window. Users who received 1-3 clips came back a second day
    0.4% of the time; those who received 4-9 came back 16.1% — so the clip
    count, not the clip quality, is what the retention curve hangs on.

    The old prompt biased hard the other way ("prefer one great clip per
    candidate window") and handed the model two unbounded licences to drop
    clips (the 2-second rule and STANDS ALONE both end in "or skip it"), with
    no floor to stop it collapsing to a single clip. This puts a floor and a
    realistic ceiling on it instead.

    ``CLIP_TARGET_MIN`` / ``CLIP_TARGET_MAX`` override both for A/B runs
    without a deploy (the reframe-testing harness drives them).
    """
    import os

    n = max(1, int(n_windows or 1))
    # Floor grows with the material: 3 windows -> 3, 5 -> 4, 10+ -> 6.
    low = max(2, min(6, n // 2 + 2))
    # Ceiling allows a rich window to yield more than one without inviting padding.
    high = min(12, max(4, n * 2))
    low = min(low, high)

    def _override(name, current):
        raw = os.environ.get(name)
        if not raw:
            return current
        try:
            return max(1, int(raw))
        except ValueError:
            return current

    low = _override("CLIP_TARGET_MIN", low)
    high = _override("CLIP_TARGET_MAX", high)
    return low, max(low, high)


def shortlist_target(video_duration):
    """How many scored windows the detail pass should get.

    Scales with duration so a long video surfaces more candidates without
    exploding the detail call, floored at 3 so a short one still gets a
    choice. ``score_batches`` exists to make this number reachable.
    """
    try:
        seconds = float(video_duration or 0)
    except (TypeError, ValueError):
        seconds = 0.0
    return max(3, min(10, int(seconds // 90) + 2))


def score_batches(windows, batch_size):
    """Split ``windows`` into near-equal scoring batches.

    Two things were wrong with the old ``range(0, n, batch_size)`` walk plus a
    prompt that said "choose up to 3 windows from this batch".

    The cap made the shortlist unreachable. Its real ceiling was
    ``3 * n_batches``, not ``shortlist_target``: measured 22-sep-2026 on a
    9:10 source, 9 windows batched 8 + 1 returned 3 + 1 = 4 windows against a
    target of 8. ``clip_count_targets`` then derives the clip floor from that
    halved shortlist, so the detail pass was asked for 4-8 clips instead of
    6-12 and the job delivered 4 (3 in prod on the same video). Every source
    under ~30 minutes was starved the same way, which is the mechanism behind
    "95% of jobs deliver 3 clips or fewer" — the clip count the retention
    curve hangs on (see clip_count_targets). The prompt now scores every
    window it is given and the shortlist is the global top ``target``, so the
    selection happens where the scores can actually be compared.

    Per-batch selection also threw away the ranking it was computing: a batch
    holding five great moments could only contribute three, while a batch of
    filler still contributed its own three.

    The remainder was its own bug: a trailing batch of one window came back
    with that window whatever its score, because a cap of 3 cannot filter a
    batch of 1 — the tail of the video entered the shortlist by arithmetic
    rather than merit (``window_009`` did exactly that on the measured run).
    Near-equal batches keep that from happening and also keep each score
    comparable, since a window judged alone is judged against nothing.
    """
    windows = list(windows or [])
    n = len(windows)
    if not n:
        return []
    size = max(1, int(batch_size or 1))
    n_batches = max(1, -(-n // size))       # ceil
    base, extra = divmod(n, n_batches)

    out = []
    start = 0
    for index in range(n_batches):
        take = base + (1 if index < extra else 0)
        batch = windows[start:start + take]
        start += take
        if batch:
            out.append(batch)
    return out


def trim_to_best(shorts, max_clips):
    """Cut an over-long detail-pass result down to ``max_clips`` BY SCORE.

    The detail pass hands its clips back in transcript order, batch after
    batch, so slicing the list keeps the EARLIEST clips rather than the best
    ones. On a 9-minute walkthrough that quietly threw away everything past
    minute three: the model proposed clips across the whole video, and the
    ones covering the demo, the MCP walkthrough and the close were the tail
    that got dropped. Worse, the failure scales the wrong way — the more
    generous the model is, the more of the video disappears.

    That sabotages the windowing: get_viral_clips builds scoring windows
    precisely because "a single call over the whole transcript clusters picks
    near the start", and a positional slice puts the clustering right back.

    Ranking is by ``predicted_score`` (the detail prompt already asks for it,
    and nothing else was reading it here). Ties keep transcript order, and the
    survivors come back in transcript order too, so clip numbering still runs
    front to back the way every caller downstream expects.
    """
    max_clips = max(1, int(max_clips or 1))
    if len(shorts) <= max_clips:
        return list(shorts)

    def score(item):
        try:
            return float(item[1].get("predicted_score") or 0)
        except (TypeError, ValueError, AttributeError):
            return 0.0

    indexed = list(enumerate(shorts))
    best = sorted(indexed, key=score, reverse=True)[:max_clips]
    return [item for _, item in sorted(best, key=lambda pair: pair[0])]


def vision_segments(video_duration, segment_seconds, max_segments):
    """Split a silent source into near-equal ``(start, end)`` stretches for
    the vision pass, each at most ``segment_seconds`` long.

    The vision pass sends Gemini the footage itself at ~300 tokens/second, so
    an hour (~1.08M tokens) no longer fits a 1M-token window and a silent
    multi-hour capture failed at the model. Near-equal stretches, not fixed
    ones: a 41-minute source split at 40 minutes would leave a one-minute
    tail that the prompt has to fill with 3+ clips. A source that fits in one
    stretch returns one stretch, so short videos take the single-upload path
    unchanged. Past ``max_segments`` the source is truncated (the caller logs
    it) rather than spending without bound on one job.
    """
    duration = max(0.0, float(video_duration or 0))
    seg = max(60.0, float(segment_seconds))
    if duration <= seg:
        return [(0.0, round(duration, 3))]
    n = int(math.ceil(duration / seg))
    n_kept = max(1, min(n, int(max_segments)))
    step = duration / n
    return [(round(i * step, 3), round(min(duration, (i + 1) * step), 3))
            for i in range(n_kept)]


def clip_duration_bounds():
    """The clip length band (seconds) the selection prompts and word-snapping
    enforce. ``CLIP_MIN_SECONDS`` / ``CLIP_MAX_SECONDS`` override the classic
    15-60 — set per job by /api/process when the user asks for a specific
    length, or by hand for A/B runs. Values are clamped to platform-sane
    limits and re-ordered so bad input degrades instead of breaking the job.
    """
    import os

    def _read(name, default):
        try:
            return float(os.environ.get(name, ""))
        except ValueError:
            return default

    lo = _read("CLIP_MIN_SECONDS", 15.0)
    hi = _read("CLIP_MAX_SECONDS", 60.0)
    lo = min(max(lo, 5.0), 175.0)
    hi = min(max(hi, 10.0), 180.0)
    if hi < lo + 5.0:  # keep a real band: degenerate ranges starve the model
        hi = min(180.0, lo + 5.0)
    return round(lo, 3), round(hi, 3)


def clip_length_overrun():
    """Seconds a clip may run past the DEFAULT maximum to finish its idea.

    Measured: three of eleven clips on two videos hit the 60 s cap and were
    cut under a second before their point ("...without going | off the
    rails"); one complete idea ran 63 s. The 60 s default is an editorial
    target, not a platform limit (the band itself goes to 180 s), so when
    no maximum was asked for, an idea that lands a few seconds past it may
    keep its landing. The context check uses it only after the preferred
    fix (ending on the last complete thought inside the band) does not fit.

    0 whenever ``CLIP_MAX_SECONDS`` is set: a maximum the user asked for
    (request field, or a creator brief's ``max_clip_seconds``, both of which
    set that variable) is exact. ``CLIP_MAX_OVERRUN_SECONDS`` (default 5,
    0..15) tunes it; never past the band's 180 s ceiling."""
    import os

    if str(os.environ.get("CLIP_MAX_SECONDS", "") or "").strip():
        return 0.0
    try:
        raw = float(os.environ.get("CLIP_MAX_OVERRUN_SECONDS", "") or 5.0)
    except ValueError:
        raw = 5.0
    over = min(max(raw, 0.0), 15.0)
    _, hi = clip_duration_bounds()
    return round(max(0.0, min(over, 180.0 - hi)), 3)


def compact_words(words, precision=2):
    """Round word timestamps for prompts — full float precision wastes tokens."""
    return [
        {
            "w": w.get("w", ""),
            "s": round(float(w.get("s", 0)), precision),
            "e": round(float(w.get("e", 0)), precision),
        }
        for w in words
    ]


def build_transcript_windows(transcript_result, video_duration,
                             window_seconds=90, overlap_seconds=30):
    """
    Build scoring windows aligned to Whisper segment boundaries, so a sentence
    (and usually a viral moment) is never cut in half mid-window. Windows grow
    segment by segment to roughly window_seconds (up to 1.25x for the closing
    segment) and the next window starts ~overlap_seconds before the previous
    end, also snapped to a segment start.
    """
    segments = []
    for segment in transcript_result.get("segments", []):
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        segments.append((float(segment.get("start", 0)), float(segment.get("end", 0)), text))

    windows = []
    window_index = 1
    i = 0
    n = len(segments)
    while i < n:
        w_start = segments[i][0]
        j = i
        # Extend while the NEXT segment still fits within a tolerant cap, so the
        # window closes on a segment boundary near window_seconds.
        while j + 1 < n and segments[j + 1][1] - w_start <= window_seconds * 1.25:
            j += 1
            if segments[j][1] - w_start >= window_seconds:
                break
        w_end = segments[j][1]
        windows.append({
            "id": f"window_{window_index:03d}",
            "start": round(w_start, 3),
            "end": round(w_end, 3),
            "text": " ".join(seg[2] for seg in segments[i:j + 1]),
        })
        window_index += 1

        if j >= n - 1:
            break
        # Next window starts at the first segment beginning after (end - overlap),
        # but always makes progress.
        target = w_end - overlap_seconds
        k = i + 1
        while k <= j and segments[k][0] < target:
            k += 1
        i = max(k, i + 1)

    if not windows:
        windows.append({
            "id": "window_001",
            "start": 0.0,
            "end": round(float(video_duration), 3),
            "text": str(transcript_result.get("text", "") or ""),
        })
    return windows


def dedupe_overlapping(shorts, ratio=0.25):
    """Drop clips that overlap an already-kept clip by ``ratio`` of the
    shorter one, keeping the higher ``predicted_score`` (earlier on a tie).

    0.25, not 0.5: two clips sharing 25 s of 59 s (42%) both survived the old
    bar and the same punchline came out twice in one batch.

    The detail prompt's DIVERSITY rule is the only thing that stopped two
    clips from sharing the same seconds, and a rule is not a guarantee: two
    picks from one window can cover the same moment with different edges,
    and the user then downloads the same short twice. Survivors come back in
    transcript order, like ``trim_to_best``.
    """
    def _score(c):
        try:
            return float(c.get("predicted_score") or 0)
        except (TypeError, ValueError, AttributeError):
            return 0.0

    def _span(c):
        try:
            return float(c.get("start", 0)), float(c.get("end", 0))
        except (TypeError, ValueError, AttributeError):
            return 0.0, 0.0

    indexed = list(enumerate(shorts))
    kept = []
    for idx, clip in sorted(indexed, key=lambda p: (-_score(p[1]), p[0])):
        s, e = _span(clip)
        clash = False
        for _, other in kept:
            os_, oe = _span(other)
            overlap = min(e, oe) - max(s, os_)
            shorter = max(1e-6, min(e - s, oe - os_))
            if overlap > 0 and overlap / shorter >= ratio:
                clash = True
                break
        if not clash:
            kept.append((idx, clip))
    return [c for _, c in sorted(kept, key=lambda p: p[0])]


def snap_clip_to_words(start, end, words, video_duration,
                       min_duration=15.0, max_duration=60.0,
                       search_window=1.5, max_lead=0.35, max_tail=0.45):
    """
    Snap Gemini-proposed clip boundaries onto real word boundaries plus a bit
    of the surrounding silence. LLMs are bad at millisecond arithmetic; the
    word-level timestamps are ground truth, so cuts land in pauses instead of
    mid-word.

    words: [{'w','s','e'}, ...] for the whole video, sorted by start.
    Returns (start, end); falls back to the input if no words are nearby or
    snapping cannot satisfy the duration bounds.
    """
    original = (round(float(start), 3), round(float(end), 3))
    if not words:
        return original

    starts = [float(w.get("s", 0)) for w in words]
    ends = [float(w.get("e", 0)) for w in words]

    # START: snap to the nearest word start, then lead into the silence before it.
    new_start = float(start)
    candidates = [s for s in starts if abs(s - new_start) <= search_window]
    if candidates:
        word_start = min(candidates, key=lambda s: abs(s - new_start))
        prev_ends = [e for e in ends if e <= word_start]
        if prev_ends:
            gap = max(0.0, word_start - max(prev_ends))
            lead = min(max_lead, gap / 2)
        else:
            lead = max_lead
        new_start = max(0.0, word_start - lead)

    # END: snap to the nearest word end, then trail into the silence after it.
    new_end = float(end)
    candidates = [e for e in ends if abs(e - new_end) <= search_window]
    if candidates:
        word_end = min(candidates, key=lambda e: abs(e - new_end))
        next_starts = [s for s in starts if s >= word_end]
        if next_starts:
            gap = max(0.0, min(next_starts) - word_end)
            tail = min(max_tail, gap / 2)
        else:
            tail = max_tail
        new_end = min(float(video_duration), word_end + tail)

    # Repair duration bounds while staying on word boundaries.
    if new_end - new_start < min_duration:
        target = new_start + min_duration
        later = sorted(e for e in ends if e >= target)
        if later and later[0] - new_start <= max_duration:
            new_end = min(float(video_duration), later[0] + 0.2)
        else:
            return original
    if new_end - new_start > max_duration:
        target = new_start + max_duration
        earlier = [e for e in ends if new_start < e <= target]
        new_end = (max(earlier) + 0.2) if earlier else target
        new_end = min(new_end, new_start + max_duration, float(video_duration))

    if new_end <= new_start or new_end - new_start < min_duration:
        return original
    return (round(new_start, 3), round(new_end, 3))


# ---------------------------------------------------------------------------
# Pieces: how the detail pass and the context check name clip boundaries
# ---------------------------------------------------------------------------
#
# A clip is one complete idea: it opens where the idea starts being set up and
# ends where it lands. Where that is depends on meaning, not on punctuation:
# speech-to-text output often runs for minutes without a full stop, a pause can
# sit in the middle of a sentence, and a grammatically clean first sentence can
# still be an answer to a question the viewer never heard. So the transcript is
# split into short phrase-level PIECES (a few words to a few seconds) that are
# fine enough for an idea's true beginning and end to always be selectable, and
# the models pick pieces by index (closed choice, re-validated in code, never
# timestamp arithmetic): the detail pass picks the first and last piece of each
# clip, the context check (context_check.py) may move them to an earlier /
# later piece, and snap_clip_to_pieces only places the cut in the silence at
# the edge of the chosen piece. Punctuation, pauses and discourse words decide
# where pieces split and break ties; they never define a valid cut on their
# own. See docs/architecture/clip-selection.md.

PIECE_PAUSE_SECONDS = 0.5     # a silence this long always ends a piece
PIECE_MAX_SECONDS = 5.0       # a piece is split before it would pass this
PIECE_CLAUSE_SECONDS = 1.0    # a comma may end a piece at least this long
PIECE_SOFT_SECONDS = 1.8      # a discourse word may open a new piece past this
PIECE_MIN_SECONDS = 0.6       # a shorter piece with a weak end joins the next
PIECE_GAP_SPLIT = 0.15        # a forced split at a pause this long is cue "gap"

# Words that usually open a new phrase (English; other languages split on
# punctuation, pauses and the max length only).
DISCOURSE_WORDS = frozenset((
    "and", "but", "so", "because", "cause", "or", "then", "now", "well",
    "which", "when", "while", "if", "although", "though",
    "anyway", "basically", "okay", "ok", "except", "unless", "until", "whereas",
    "meanwhile", "however", "otherwise", "plus", "instead",
))
# Words that often open a clause: a forced split prefers to land before one.
STARTER_WORDS = DISCOURSE_WORDS | frozenset((
    "another", "this", "that", "these", "those", "there", "here", "it", "we",
    "i", "you", "they", "he", "she", "what", "how", "why", "who", "where",
    "my", "our", "your", "their", "let's", "imagine",
))

# cue: how a piece ends. A hint and a tie-breaker only (every piece edge is a
# valid cut); the cost of moving a cut onto that edge when the snap has to
# move one to fit the length band. "end" is the edge of the transcript.
CUE_PENALTY = {"sentence": 0.0, "end": 0.0, "pause": 0.5, "clause": 1.0,
               "discourse": 1.5, "gap": 2.5, "max": 4.0}
_EXACT_TOLERANCE = 0.05       # seconds: a cut this close to an edge IS that edge
_BARE_EDGE_COST = 0.05        # cost of dropping the lead/tail to fit the band
# Extra cost of SHORTENING a clip onto an edge that is probably mid-sentence
# (cue clause / discourse / gap / max) to fit the length cap: an end moved
# earlier, or a start moved later. Without it the cheapest cut dropped the
# last second of the point ("...without going | off the rails", cue max)
# instead of the last sentence. A sentence edge up to ~3 s further now wins;
# unpunctuated speech, where every edge is weak, still drops the least.
_MID_THOUGHT_END_PENALTY = 6.0
_WEAK_END_CUES = ("clause", "discourse", "gap", "max")
# Dropping speech at the START costs more than at the end (2 per second): the
# opening is where the subject is introduced. Measured on the cap case: at
# equal cost the snap dropped "Now this is where the harness and the model
# work together..." and opened on "So the model decides on... that goal".
_START_DROP_COST = 3.0


def _piece_word(w):
    text = str(w.get("w", w.get("word", "")) or "").strip()
    s = float(w.get("s", w.get("start")))
    e = float(w.get("e", w.get("end")))
    return text, s, e


def _clean_words(words):
    out = []
    for w in words or []:
        try:
            text, s, e = _piece_word(w)
        except (TypeError, ValueError, AttributeError):
            continue
        if text and e >= s and math.isfinite(s) and math.isfinite(e):
            out.append((text, s, e))
    out.sort(key=lambda t: t[1])
    return out


def _bare(token):
    return re.sub(r"^\W+|\W+$", "", token.lower())


def _capital_starter(token):
    """A capitalised clause-starter word ("What", "So", "They"; never "I"):
    speech-to-text often drops a full stop but keeps the capital of the next
    sentence ("...going to get to What scenario is it..."), and that is
    where an idea can begin."""
    text = re.sub(r"^\W+", "", str(token or ""))
    bare = _bare(token)
    return bool(text) and text[0].isupper() and bare in STARTER_WORDS and bare != "i" \
        and not text.isupper()


def _hidden_pause(word):
    """Seconds of silence a word's timing probably swallows: many ASR
    aligners give contiguous timings (zero gaps), stretching a word over the
    pause next to it, so a word far longer than its letters is a pause cue."""
    text, s, e = word
    return max(0.0, (e - s) - (0.12 + 0.06 * len(_bare(text) or text)))


def _forced_split(cur):
    """Where to split an over-long run of words: the index k to split after
    (best inter-word gap or hidden pause, a little extra before a clause
    starter word), keeping at least PIECE_CLAUSE_SECONDS before it; and its
    cue ("gap" for a real or hidden pause, "max" otherwise). None: no room."""
    best = None
    for k in range(len(cur) - 1):
        if cur[k][2] - cur[0][1] < PIECE_CLAUSE_SECONDS:
            continue
        pause = (cur[k + 1][1] - cur[k][2]) + 0.5 * (
            _hidden_pause(cur[k]) + _hidden_pause(cur[k + 1]))
        score = pause + (0.3 if _bare(cur[k + 1][0]) in STARTER_WORDS else 0.0)
        if best is None or score > best[0]:
            best = (score, k, pause)
    if best is None or best[0] <= 0:
        return None
    return best[1], ("gap" if best[2] >= PIECE_GAP_SPLIT else "max")


def build_pieces(words, pause=PIECE_PAUSE_SECONDS, max_seconds=PIECE_MAX_SECONDS,
                 min_seconds=PIECE_MIN_SECONDS):
    """The transcript's words as numbered phrase-level pieces
    ``[{index, start, end, text, cue}]`` (absolute seconds, word edges).

    A piece ends at sentence punctuation, at a silence of ``pause`` or more,
    at a comma once it is a real clause, before a capitalised clause starter
    once it is a clause (a full stop the ASR dropped), before a discourse word ("and",
    "but", "so", "because"...) once it is a phrase, and before it would pass
    ``max_seconds`` (then where a real or hidden pause is, preferring a spot
    before a clause starter word: ``cue`` "gap", else "max"). A piece shorter than
    ``min_seconds`` whose end is weak joins the next one; a short piece that
    ends a sentence or a pause stays ("Fox." / "Hell no, man." are answers a
    clip may need to start or end on). Bad words are skipped.
    """
    ws = _clean_words(words)
    if not ws:
        return []

    groups, cur = [], []     # groups: [(words, cue)]
    for i, w in enumerate(ws):
        cur.append(w)
        nxt = ws[i + 1] if i + 1 < len(ws) else None
        if nxt is None:
            groups.append((cur, "end"))
            cur = []
            break
        length = w[2] - cur[0][1]
        if SENTENCE_END.search(w[0]):
            cue = "sentence"
        elif nxt[1] - w[2] >= pause:
            cue = "pause"
        elif _CLAUSE_END.search(w[0]) and length >= PIECE_CLAUSE_SECONDS:
            cue = "clause"
        elif _capital_starter(nxt[0]) and length >= PIECE_CLAUSE_SECONDS:
            cue = "clause"            # a dropped full stop before a capital
        elif _bare(nxt[0]) in DISCOURSE_WORDS and length >= PIECE_SOFT_SECONDS:
            cue = "discourse"
        elif nxt[2] - cur[0][1] > max_seconds:
            split = _forced_split(cur + [nxt])
            if split is not None and split[0] < len(cur) - 1:
                k, cue = split
                groups.append((cur[:k + 1], cue))
                cur = cur[k + 1:]
                continue
            cue = split[1] if split is not None else "max"
        else:
            continue
        groups.append((cur, cue))
        cur = []

    i = 0
    while len(groups) > 1 and i < len(groups) - 1:
        g, cue = groups[i]
        if g[-1][2] - g[0][1] >= min_seconds or cue in ("sentence", "pause", "end"):
            i += 1
            continue
        groups[i:i + 2] = [(g + groups[i + 1][0], groups[i + 1][1])]
    return [{"index": n, "start": round(g[0][1], 3), "end": round(g[-1][2], 3),
             "text": " ".join(t for t, _, _ in g), "cue": cue}
            for n, (g, cue) in enumerate(groups)]


def window_pieces(pieces, start, end):
    """The pieces whose midpoint lies in [start, end]."""
    lo, hi = float(start), float(end)
    return [p for p in pieces or [] if lo <= (p["start"] + p["end"]) / 2 <= hi]


def pieces_payload(pieces, text_cap=300):
    """Compact prompt form: ``[[index, start_second, text], ...]``. A piece
    ends where the next one starts, so its end is not sent; whole seconds
    are enough to judge the length band."""
    return [[p["index"], int(round(float(p["start"]))), str(p["text"])[:text_cap]]
            for p in pieces or []]


def clip_piece_span(start, end, pieces):
    """``(first, last)`` piece indices of the clip [start, end] (the pieces
    whose midpoint lies inside it), or None."""
    try:
        inside = window_pieces(pieces, float(start) - _EXACT_TOLERANCE,
                               float(end) + _EXACT_TOLERANCE)
    except (TypeError, ValueError):
        return None
    if not inside:
        return None
    return inside[0]["index"], inside[-1]["index"]


def _pick_index(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if f != f or f in (float("inf"), float("-inf")) or f != int(f):
        return None
    return int(f)


def resolve_piece_pick(first, last, pieces, allowed=None,
                       min_duration=15.0, max_duration=60.0):
    """``(start, end)`` in seconds for the model's ``first``/``last`` piece,
    or None when the pick is malformed: not integers, out of range, reversed,
    outside ``allowed`` (the indices the prompt showed) or a span far outside
    the length band (under half the minimum or over 1.5x the maximum: the
    model then clearly meant something else and its seconds are kept). A
    span slightly outside the band is kept: the final snap fits it."""
    f, l = _pick_index(first), _pick_index(last)
    if f is None or l is None or not pieces:
        return None
    if not (0 <= f <= l < len(pieces)):
        return None
    if allowed is not None and (f not in allowed or l not in allowed):
        return None
    start, end = float(pieces[f]["start"]), float(pieces[l]["end"])
    span = end - start
    if span <= 0 or span < 0.5 * float(min_duration) or span > 1.5 * float(max_duration):
        return None
    return round(start, 3), round(end, 3)


def clean_about(value, cap=300):
    """The model's one-sentence ``about`` as one plain line, or ""."""
    if not isinstance(value, str):
        return ""
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", value)
    return re.sub(r"\s+", " ", text).strip()[:cap]


def apply_piece_picks(shorts, pieces, allowed=None, min_duration=15.0, max_duration=60.0):
    """Turn each clip's ``first_piece``/``last_piece`` into ``start``/``end``
    (in place), record ``piece_pick`` and keep the model's ``about`` line. A
    missing or malformed pick keeps the model's own seconds, so the clip
    degrades to the old behaviour and is never dropped. Returns
    ``(used, kept_seconds)``."""
    used = kept = 0
    for clip in shorts or []:
        if not isinstance(clip, dict):
            continue
        first, last = clip.pop("first_piece", None), clip.pop("last_piece", None)
        about = clean_about(clip.get("about"))
        if about:
            clip["about"] = about
        else:
            clip.pop("about", None)
        span = resolve_piece_pick(first, last, pieces, allowed, min_duration, max_duration)
        if span is None:
            kept += 1
            continue
        clip["start"], clip["end"] = span
        clip["piece_pick"] = [_pick_index(first), _pick_index(last)]
        used += 1
    return used, kept


def snap_clip_to_pieces(start, end, pieces, words, video_duration,
                        min_duration=15.0, max_duration=60.0,
                        max_lead=0.35, max_tail=0.45):
    """Place the cuts of a clip whose edges were chosen by meaning:
    ``(start, end, moved)``, or None when no placement fits the band (the
    caller then falls back to ``snap_clip_to_words``).

    A requested edge that is a piece edge or a word edge (a hook-pass trim
    lands on a word) is kept exactly, plus a lead/tail into the silence
    around it (half the gap, at most ``max_lead``/``max_tail``) so the cut
    never lands mid-word. Only when that misses the length band, or when the
    request is not on any edge (the model answered in seconds), does an edge
    move to another piece edge: taking in speech costs 1 per second, dropping
    it 2 at the end and 3 at the start (the setup), plus the cue penalty of
    the new edge (a sentence end is free, a forced split costs most) and a
    mid-thought penalty for a shortened edge on a weak cue. The cheapest pair inside
    [min_duration, max_duration] wins; dropping the lead/tail is almost free.
    ``moved`` is True when an edge left the requested one."""
    try:
        p_s, p_e = float(start), float(end)
        duration = float(video_duration)
    except (TypeError, ValueError):
        return None
    if not pieces or p_e <= p_s:
        return None
    ws = _clean_words(words)
    reach = max(float(max_duration), 20.0)

    def gap_lead(t, prev_end):
        return max_lead if prev_end is None else min(max_lead, max(0.0, t - prev_end) / 2)

    def gap_tail(t, next_start):
        return max_tail if next_start is None else min(max_tail, max(0.0, next_start - t) / 2)

    starts, ends = [], []   # (cut, cost, edge time)

    def add_start(t, cost, prev_end):
        lead = gap_lead(t, prev_end)
        starts.append((max(0.0, t - lead), cost, t))
        if lead > 0:
            starts.append((t, cost + _BARE_EDGE_COST, t))

    def add_end(t, cost, next_start):
        tail = gap_tail(t, next_start)
        ends.append((min(duration, t + tail), cost, t))
        if tail > 0:
            ends.append((min(duration, t), cost + _BARE_EDGE_COST, t))

    n = len(pieces)
    for i, p in enumerate(pieces):
        t = float(p["start"])
        d = t - p_s
        if abs(d) <= reach:
            prev_cue = pieces[i - 1]["cue"] if i else "end"
            cost = 0.0 if abs(d) <= _EXACT_TOLERANCE else (
                (-d if d < 0 else _START_DROP_COST * d) + CUE_PENALTY.get(prev_cue, 4.0)
                + (_MID_THOUGHT_END_PENALTY if d > 0 and prev_cue in _WEAK_END_CUES else 0.0))
            add_start(t, cost, float(pieces[i - 1]["end"]) if i else None)
        t = float(p["end"])
        d = t - p_e
        if abs(d) <= reach:
            cost = 0.0 if abs(d) <= _EXACT_TOLERANCE else (
                (d if d > 0 else 2.0 * -d) + CUE_PENALTY.get(p["cue"], 4.0)
                + (_MID_THOUGHT_END_PENALTY if d < 0 and p["cue"] in _WEAK_END_CUES else 0.0))
            add_end(t, cost, float(pieces[i + 1]["start"]) if i + 1 < n else None)

    # A requested edge on a word (not a piece edge) is a model choice too.
    for k, (_, s, e) in enumerate(ws):
        if abs(s - p_s) <= _EXACT_TOLERANCE:
            add_start(s, 0.0, ws[k - 1][2] if k else None)
        if abs(e - p_e) <= _EXACT_TOLERANCE:
            add_end(e, 0.0, ws[k + 1][1] if k + 1 < len(ws) else None)

    best = None
    want = p_e - p_s
    for s, cs, ts in starts:
        for e, ce, te in ends:
            length = e - s
            if length < min_duration - 1e-6 or length > max_duration + 1e-6:
                continue
            key = (round(cs + ce, 6), abs(length - want))
            if best is None or key < best[0]:
                best = (key, s, e, ts, te)
    if best is None:
        return None
    _, s, e, ts, te = best
    moved = abs(ts - p_s) > _EXACT_TOLERANCE or abs(te - p_e) > _EXACT_TOLERANCE
    return round(s, 3), round(e, 3), moved


# Verdicts a context_check record may carry (context_check.START_VERDICTS /
# END_VERDICTS, plus the names older records used).
_KNOWN_VERDICTS = ("ok", "orphan_answer", "unresolved_reference", "mid_thought", "unfinished",
                   "orphan_head", "complete", "mid_sentence", "point_not_landed", "runs_on")


def boundary_label(clip, on_pieces=True):
    """How a clip's cuts were decided (``clip["boundary"]``), truthfully:

    - "checked": the context check judged both edges and found them sound or
      repaired them;
    - "flagged": the context check found a problem it could not repair
      (``context_check.kept`` says why; ``opening`` "unresolved" / ``ending``
      "unfinished" say which edge);
    - "chosen": the model picked the edges by meaning (``piece_pick``) and no
      check judged them;
    - "fallback": the model answered in seconds only; the cut went to the
      nearest piece edges;
    - "word": no pieces, or nothing fitted the band: the plain word snap.
    """
    if not on_pieces:
        return "word"
    cc = clip.get("context_check") if isinstance(clip, dict) else None
    if isinstance(cc, dict):
        if cc.get("kept") or cc.get("opening") == "unresolved" or cc.get("ending") == "unfinished":
            return "flagged"
        if cc.get("start") in _KNOWN_VERDICTS and cc.get("end") in _KNOWN_VERDICTS:
            return "checked"
    return "chosen" if isinstance(clip, dict) and clip.get("piece_pick") else "fallback"
