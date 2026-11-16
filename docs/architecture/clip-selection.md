# Clip selection

## How many clips a job returns (`clip_selection.py`, `main.get_viral_clips`)

The count is not a setting, it is derived: `shortlist_target` (`min(10,
duration//90 + 2)`, floor 3) feeds `clip_count_targets`, which gives the floor
`min_clips` and the ceiling `max_clips`. The floor matters: jobs that return
only 1-3 clips retain far worse than jobs that return 4-9. `get_viral_clips`
picks clips **among ideas** (`idea_segments.py`, next section); the old
window pipeline is the fallback.

Two invariants keep that chain from leaking. They were written for scoring
windows; the idea pipeline keeps the same guarantees for ideas:

- **The scoring stage ranks, it does not select.** Every candidate it is given
  gets a score and the shortlist is the global top N. Ideas: the idea pass
  scores every idea it segments in every chunk, and `idea_segments.top_ideas`
  takes the global top `max_clips + spare` over the whole video, never a quota
  per chunk. Windows: the score prompt scores every window and the shortlist
  is the global top `shortlist_target`. A prompt that says "choose up to 3 from
  this batch" caps the shortlist at `3 * n_batches` and starves every source
  under ~30 min. Batches and chunks are near-equal (`score_batches`,
  `idea_segments.chunk_pieces`): a trailing batch of one would be judged
  against nothing.
- **The clip floor is enforced in code, not just in the prompt.** When the copy
  (detail) pass comes back under `min_clips`, the candidates that produced
  nothing get one more call for the difference: on the idea path the
  shortlisted ideas it skipped plus the spare candidates, on the window path
  the shortlist windows. An empty answer is accepted: padding is worse than a
  short list.

`target_clips` on `/api/process` (dashboard: advanced options) pins both ends via
`CLIP_TARGET_MIN` / `CLIP_TARGET_MAX` when the user wants an exact number.

## How clip boundaries are chosen (ideas first, then a concrete check)

A clip is **one complete idea**: a viewer who has seen nothing else understands
it. It opens where the idea's subject is introduced (the question, the claim,
the piece that names who or what is discussed) and ends where it lands (the
answer, the punchline, the conclusion). That is a question of meaning, not
punctuation.

Why ideas first. The window pipeline scored fixed ~90 s windows and the detail
pass had to pick each clip inside one, then a context check could move the
start back by up to 25 s. Reading failing clips against their surrounding
speech showed three causes, in this order of frequency: (1) the setup was in
the window and in reach, but the model judged a dependent opening "ok" ("Hell
no." two seconds after the question it answers; "Minnesota is not Philly.
Let's not do that."; "and they look in the place..."), and the check's
"smallest repair" instruction on ~1.2 s dialogue pieces moved starts one piece
back, still mid-sentence; (2) the check could only move a start back, never
forward past the tail of the previous topic ("...and we've got the agentic loop
so let's take a look at these one at a time we'll start with tools | so
through tools..."); (3) a window wall or the 25 s look-back cut the clip off
from its setup ("And in addition to this cycle..." whose cycle was introduced
37 s earlier, in the previous window). Separately, ideas a little over 60 s
were cut at the cap a second before their point.

The pipeline (`main.get_viral_clips` → `_clips_from_ideas`):

1. **Pieces** (`clip_selection.build_pieces`). The words become numbered
   phrase-level pieces, a few words to a few seconds (median ~2.7 s on a talk,
   ~1.2 s on rapid dialogue), fine enough that an idea's true beginning and end
   are always selectable. A piece ends at sentence punctuation, at a silence of
   0.5 s, at a comma once it is a clause (1 s), before a capitalised clause
   starter once it is a clause (a full stop the ASR dropped: "...get to What
   scenario is it..."), before a discourse word ("and", "but", "so",
   "because"...; English list) once it is a phrase (1.8 s), and before 5 s at
   the best spot: a real or *hidden* pause (aligners that give contiguous
   timings stretch a word over the pause next to it), preferring a spot before
   a clause-starter word. Short weak fragments join the next piece; short
   answers that end a sentence ("Fox.") stay. Each piece records its `cue`
   (sentence / pause / clause / discourse / gap / max / end): a hint and a
   tie-breaker, never the definition of a valid cut.
2. **Idea pass** (`idea_segments.run_pass`, boundary model). The pieces are
   read in transcript order in near-equal overlapping chunks
   (`chunk_pieces`: 8 min on Gemini, 5 min on a local LLM; the overlap is at
   least 1.25x the longest clip + 15 s, so an idea cut by one chunk's edge is
   whole in its neighbour; a 7-minute source is one call). Each chunk is
   segmented into consecutive ideas covering it (filler too, scored low): per
   idea `about`, `first_piece` (where its subject is introduced; never a piece
   that leans on something earlier), `last_piece` (where it lands, last
   sentence complete), a 0-100 `score` (the 2-second test of the old score
   prompt) and a short `reason`. The prompt states whether the chunk starts or
   ends mid-video and asks to leave out leading/trailing pieces that belong to
   the neighbour. Validated in code: indices inside the chunk, in order, span
   within 0.5x min .. 1.5x (max + overrun); scores clamped. `merge` keeps one
   entry per idea (two entries sharing half the shorter span): the copy not
   cut by a chunk edge, then the higher score. Twitch `viewer_moments` are
   listed per chunk as strong evidence (scores only).
3. **Rank, then check before copy.** `top_ideas` takes the global top
   `max_clips + spare` (spare = `max(2, min_clips // 2)`), skipping one that
   overlaps a better one by 25%. The **context check** (below) runs on these
   candidates *before* any copy is written, so titles and hooks describe the
   final cut and selection can prefer a self-contained idea: `rerank` orders
   by `score - 25` for an opening the check could not resolve and `- 10` for
   an ending left unfinished, drops a candidate a repair made overlap a
   better one, and the shortlist is the top `max_clips` of that order (hot
   Twitch ideas are promoted as before, `twitch_signals.promote_hot_windows`).
4. **Copy pass** (`gemini_worker.IDEA_DETAIL_PROMPT_TEMPLATE`, job model): one
   entry per candidate (`source_window_id` = idea id), edges unchanged; it may
   skip only what it would not publish (and keeps the DIVERSITY rule). Gemini:
   one call; a local LLM: 4 ideas per call. Floor top-up as above.
   `idea_segments.attach` gives each clip its idea's `start`/`end`,
   `piece_pick`, `about`, `idea_id`, `idea_score` and the check's record; a
   clip naming no known idea is matched by overlap, else keeps its seconds.
5. Hook pass → `trim_to_best` → context check for clips **without** a record
   (on this path only clips whose earlier check failed or matched no idea) →
   final snap → `dedupe_overlapping`.

Too few ideas (fewer than `min_clips`: the pass is off, every chunk failed,
or a small model could not follow the format) falls back to the **window
pipeline** (`_clips_from_windows`, unchanged): windows scored in batches,
global top `shortlist_target`, the detail pass picks each clip inside them
by `first_piece` / `last_piece` (`apply_piece_picks`), floor top-up, hook pass,
`trim_to_best`, then the context check on every clip.

**The context check** (`context_check.py`, one batched call: `CONTEXT_BATCH`
12, 2 on a local LLM; boundary model). Each clip is shown with up to 40 s / 24
pieces of speech before it and 20 s / 10 pieces after it, marked as not heard
by the viewer, indices relative to the clip's first piece (look-back
negative), its ends in full, its middle as one capped text, and its `about`.
The question is concrete, not "is it ok":

- `unresolved`: **every** reference in the first lines a cold viewer cannot
  resolve from the clip itself, each as `phrase` (≤ 6 words), `kind`
  (pronoun / back_reference / answer_to_unseen_question /
  reaction_to_unseen_claim / list_item / undefined_term / continues_sentence)
  and `introduced_at` (the look-back piece where it is introduced, 0 when not
  shown). A reference the clip explains itself is not listed.
- `idea_starts_at`: 0, or a clip piece (1-5) where the idea really begins when
  everything before it is greeting, housekeeping or the previous topic's tail.
- `end_verdict` (complete / mid_sentence / point_not_landed / runs_on),
  `lands_at` (where the point lands with its sentence complete: in the clip or
  the look-ahead) and `last_complete_piece` (the latest clip piece that would
  still end on a complete thought).

Code validates and decides (`judge`): references introduced inside the clip
are dropped; the start moves to the **earliest** introducing piece (an
opening with an unresolved reference is never "ok"); a forward move only
without references and only if the minimum still holds. Ends, in order: the
landing inside the band; when no maximum was asked for, the landing up to
`clip_length_overrun()` seconds past the default maximum; the last complete
thought inside the band; for the repaired start first, then the original
start. A clip never ends at a cut the cap forced. What cannot be repaired is
recorded, not passed: `kept` = reach (introduction not shown: too far back),
index (bad index), shift (too far), band (the repair breaks the length band).
Each judged clip records `context_check = {start, end, opening, ending,
moved, start_shift, end_shift[, refs, end_at, over_max, kept]}` (`opening`:
ok / repaired / unresolved; `ending`: ok / repaired / unfinished; `refs`:
kind, a ≤ 40-char phrase and the look-back index) and `opening_checked`. It
never adds, drops or fails a clip; any exception restores the clips exactly.
The job log gets counts only (references named, repairs, unresolved
openings/endings, overruns), never transcript text. No creator brief block:
the brief steers which moments and which tone, not where an idea begins (its
duration directives reach the check through the band).

**The length cap.** `clip_length_overrun()` (default 5 s,
`CLIP_MAX_OVERRUN_SECONDS`, 0..15) applies only when `CLIP_MAX_SECONDS` is
unset: a maximum from the request or from a creator brief's
`max_clip_seconds` (both set that variable) is exact. The 60 s default is an
editorial target, not a platform limit (the band goes to 180 s), and the
measured cap cuts were under a second short; the one complete idea measured
ran 63 s, and ending on its last complete thought inside 60 s would have
dropped 27 s. The snap gives a clip the overrun only when the check recorded
`over_max`.

**The hook pass after the check.** A clip with `opening_checked` keeps its
start: the hook pass may still write the hook and pick a teaser, but never
trims a lead-in into the setup the check decided (`auto_hooks.apply`). On the
window path the check still runs after the hook pass, so a lead-in trim that
cut the setup is undone there (`lead_in_trimmed` shrinks or goes).

**The final snap follows the chosen pieces** (`snap_clip_to_pieces`). A start /
end on a piece edge or on a word edge is kept exactly, plus the lead/tail into
the silence around it (half the gap, at most 0.35 / 0.45 s), never mid-word.
An edge moves only to fit the length band (or when the model answered in
seconds): to another piece edge, taking in speech at cost 1 per second,
dropping it at 2 at the end and 3 at the start (the setup), plus the new
edge's cue penalty, plus 6 when a shortened edge lands on a weak cue (clause /
discourse / gap / max: probably mid-sentence), so the cap ends a clip on the
last sentence, not a second before its point. No pieces or nothing fitting
falls back to `snap_clip_to_words`.

**Labels say how the cut was decided** (`clip_selection.boundary_label`):
`boundary` = `checked` (the check judged both edges sound or repaired them),
`flagged` (it found a problem it could not repair: `kept`, an `opening`
"unresolved" or an `ending` "unfinished"), `chosen` (picked by piece,
unchecked), `fallback` (seconds only, cut at the nearest piece edges) or
`word` (no pieces / nothing fitted). `band_fit: true` marks a chosen edge moved
to fit the band; `piece_pick` and `proposed` (before the snap) are kept.

**Model per stage.** `GEMINI_BOUNDARY_MODEL` runs the boundary-critical
stages (the idea pass and the context check) on another model from the
app's allowed list (`clip_selection.GEMINI_MODEL_CHOICES`, the list `app.py`
validates `X-Gemini-Model` against); unset, or off the list (one log line,
value not echoed), they use `GEMINI_MODEL`, so behaviour and cost do not
change. The copy pass, the hook pass and the window fallback stay on
`GEMINI_MODEL`. The cost line names the boundary model and its share
(`cost_analysis.boundary_model` / `boundary_cost`). On a local LLM the
override does not apply (one log line). Boundary calls time out after 120 s.

**Fail-open, step by step.** A failed idea chunk is skipped; too few ideas →
window pipeline; a failed check batch leaves its clips as cut (they are
checked again after the hook pass); no word timings → seconds + word snap.
No stage fails the job or drops a clip on a model error.

**Calls and prompt size.** Measured with the prompt builders on an 8-minute
talk and on the same transcript repeated to 58 minutes (input tokens ≈ prompt
characters / 4; the hook pass is not counted in the job's cost total, as
before; +1 call when the floor top-up fires):

| Source | Before (windows) | After (ideas) |
| --- | --- | --- |
| 8 min, Gemini | 5 calls: score 1 (~2.9k), detail 1 (~4.5k), hook 2, check 1 (~3.6k); ~11k counted input | 7 calls: ideas 2 (~2.1k each; a source up to 8 min is 1), check 2 (~8k for 14 candidates), copy 1 (~3.6k), hook 2; ~16k counted input, ~12k of it on the boundary model |
| 58 min, Gemini | 11 calls: score 7 (~3k each), detail 1 (~5.8k), hook 2, check 1 (~4.8k); ~31k | 14 calls: ideas 9 (~3k each), check 2 (~8.6k), copy 1 (~3.6k), hook 2; ~39k, ~36k on the boundary model |
| 8 min, local | 12 calls: score 3, detail 1 (~4.5k), hook 4, check 4; ~15k | 18 calls: ideas 2 (~2.1k), check 7 (~1.9k each), copy 3 (~1.8k), hook 6; ~22k |
| 58 min, local | 29 calls: score 18 (~1.4k), detail 1 (~5.8k: over a 4k context), hook 5, check 5; ~39k | 34 calls: ideas 17 (~2.2k), check 8 (~1.9k), copy 3 (~1.8k), hook 6; ~56k, no prompt over ~2.3k |

The idea pass answers more (one entry per idea, ~70 tokens: ~1k output
tokens for 8 minutes, ~6k for an hour, against ~40 per scored window). No
stage makes one call per clip on Gemini. The check grows because it runs on
the candidates (up to `max_clips + spare`) with a 40 s look-back, before the
copy, instead of on the kept clips with 25 s.

Tests: `tests/test_idea_segments.py`, `tests/test_clip_boundaries.py`,
`tests/test_context_check.py`, `tests/test_clip_floor.py` (window fallback).

| Variable | Default | Effect |
| --- | --- | --- |
| `IDEA_PASS` | `1` | `0` skips the idea pass (window pipeline). |
| `IDEA_CHUNK_SECONDS` | `480` (`300` local) | Seconds of speech per idea-pass call (at least 2x the overlap). |
| `IDEA_OVERLAP_SECONDS` | 1.25x longest clip + 15 (≥ 90) | Overlap between chunks (never below that floor). |
| `IDEA_PAUSE_SECONDS` | `0.5` | Pause between idea-pass calls. |
| `GEMINI_BOUNDARY_MODEL` | (unset = `GEMINI_MODEL`) | Model for the idea pass and the context check; must be in `GEMINI_MODEL_CHOICES`. |
| `CLIP_MAX_OVERRUN_SECONDS` | `5` | Seconds past the DEFAULT max a clip may run to land its point (0 with an explicit max). |
| `CONTEXT_CHECK` | `1` | `0` skips the context check (labels then say `chosen` / `fallback`). |
| `CONTEXT_BATCH` | `12` | Clips per context-check call (2 on a local LLM). |
| `CONTEXT_MAX_CLIPS` | `30` | Clips checked per call of the pass. |
| `CONTEXT_PAUSE_SECONDS` | `0.5` | Pause between context-check calls. |

## Silent footage: the vision fallback (`main.get_visual_clips`)

`transcribe_video` raising `NoAudioError`, **or** `speech_is_sparse()` returning
true, both set `transcript = None` and route to `get_visual_clips`. Sparse means
under `MIN_SPEECH_WORDS` (8) in total or under `MIN_SPEECH_WORDS_PER_MIN` (5).
Music-only footage and mic-muted recordings transcribe to a handful of stray
words; without the second test the picker scores those words and cuts around
them.

The vision pass uploads the video, Gemini returns the same `{"shorts"}` shape
(`gemini_worker.VisualResponse`) in the same 15-60 s band, so every later stage
is identical. `CLIP_TARGET_MIN`/`MAX` apply directly (there are no scoring
windows). The transcript is stored as `{"language": "none", "segments": []}`, so
the clips have no subtitles, which is correct.

**This is the one stage that sends Gemini the video instead of frames, on
purpose**: frames can say what kind of video it is, not which 40 seconds to cut.
Gemini counts video at ~300 tokens/second, so an hour is ~1.08M tokens, past a 1M
window. A source longer than `VISION_SEGMENT_SECONDS` (2400) is therefore split by
`clip_selection.vision_segments` into near-equal stretches (no one-minute tail),
each re-encoded to a 360p / 1 fps proxy (`_vision_proxy`: re-encoded, not
stream-copied, or every timestamp would be off by up to a GOP), watched on its
own, offset back to source time, then merged with `dedupe_overlapping` +
`trim_to_best` to `CLIP_TARGET_MAX`. One failed stretch is logged and skipped.
`VISION_MAX_SEGMENTS` (8, ~5 h 20) bounds the spend of one job; past it only the
first stretches are watched. A source that fits one stretch is one stretch: it
goes through the same proxy, never as the raw file (a broken capture, such as an
HLS VOD whose audio will not decode, made Gemini answer 400 INVALID_ARGUMENT).
When the proxy's audio cannot be encoded it is retried without audio. Do not
port the frame trick here. Gemini-only: a
text-only `LLM_BASE_URL` server cannot see footage, and without `GEMINI_API_KEY`
the function logs one line and returns None, which fails the job.

The public `/gta-5-clips`, `/gta-6-clips`, `/wow-forever-clips` and `/platforms/twitch` pages state
these thresholds and the stretch sizes; if the behaviour changes, change
`dashboard/seo/pages.js` and `dashboard/seo/clip-tools.js` too.

## Local LLM for the moment picker (`llm_backend.py`)

`LLM_BASE_URL` (+ `LLM_MODEL`, `LLM_API_KEY`) routes the transcript passes of
`get_viral_clips` (idea pass, context check, copy, hook pass; the window
fallback's score and detail passes) to any OpenAI-compatible
`/chat/completions` instead of Gemini; the response is validated with the same
pydantic schemas Gemini enforces server-side, so `main.py` sees one shape.
Batches shrink there (local contexts are 4-8k; a truncated prompt scores
garbage silently): idea chunks of 5 min, 2 clips per check / hook call, 4
ideas per copy call, 3 windows per score call (`main.score_batch_size`).
`GEMINI_BOUNDARY_MODEL` does not apply. Self-host `/api/process` then accepts a request without
`X-Gemini-Key` and `/api/config.localLlm` tells the dashboard not to demand one.
Frame-based stages (`layout_picker`, `screencast_layout`, `get_visual_clips`)
stay on Gemini. Never wired in cloud mode: `BILLING_ENABLED` ignores it.
