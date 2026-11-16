# Automatic hooks

The user never picks a hook. After the detail pass, every transcript job runs one
**hook pass** (`auto_hooks.py`) that judges each clip's opening, trims a slow
lead-in, writes a grounded hook and may pick a **teaser** line that the render
plays first as a cold open (`teaser.py`). All of it is on by default and
fail-open: any failure leaves the clip exactly as the pipeline made it before.
Silent-footage jobs (`get_visual_clips`, no transcript) skip all of it.

## The hook pass (`auto_hooks.run_pass`, called from `main.get_viral_clips`)

Runs after the copy (detail) pass and the `min_clips` floor top-up, **before**
`trim_to_best`, the final snap (`snap_clip_to_pieces`) and `dedupe_overlapping`.
On the idea path (see `clip-selection.md`) the context check has already decided
each clip's opening before the copy was written: such a clip carries
`opening_checked` and the hook pass never trims its lead-in (it would cut into
the setup a cold viewer needs; greetings and the previous topic's tail are
already handled by the check's `idea_starts_at`). Hook text, scores and the
teaser still apply. On the window path the context check runs after the hook
pass, as before: when a lead-in trim cut away the setup a clip needs, the check
moves the start back and `lead_in_trimmed` shrinks accordingly, so the trim
never undoes it.

- **Beats.** `build_beats` splits a clip's words into numbered beats: a beat ends
  at sentence punctuation, at a pause over 0.45 s or before 4.5 s; beats under
  1.2 s merge into a neighbour. Beat bounds are word bounds, absolute seconds.
- **Closed choices.** One prompt per batch of clips (`HOOK_BATCH`, 2 on a local
  LLM) lists each clip's beats (index, seconds from clip start, text), current
  hook and title. The answer (`HookAnswer`: string Literals, plain ints, no
  defaults, so it converts under google-genai) per clip: `opening`
  (strong/ok/slow), `lead_in_beats`, `teaser_beat` (-1 for none), `hook_type`
  (curiosity_gap/bold_claim/contrarian/mistake/open_loop), `hook_text`,
  `hook_score`, `opening_score` (`lead_in_beats` is ignored for a clip with
  `opening_checked`). The model picks beat indices; it never does
  timestamp arithmetic (the `layout_picker` philosophy).
- **Untrusted text.** Transcript, hook and title are cleaned with the creator
  brief's sanitiser and sit between `<clip_transcript id="NONCE">` tags with
  guard text (`creator_brief.new_nonce` / `clean_untrusted` / `wrap_untrusted`).
- **Validated in code** (`validate`): unknown or duplicate ids dropped, enums
  checked, scores clamped 0-100, `lead_in_beats` clamped to 0..3 and to what keeps
  the clip at or above the minimum length, `teaser_beat` must exist, be 1.5-4.5 s,
  come after the trimmed opening (not one of its first beats) by at least
  `TEASER_MIN_OFFSET` seconds and end inside the clip, else -1.
- **Hook text guards** (`accept_hook_text`): one sanitised line, at most 10 words
  and 90 characters, no greeting / "welcome back" / "in this video" opener, and
  every digit sequence must occur in the clip's own words (no invented numbers).
  A rejected hook keeps the detail pass's `viral_hook_text`; it is never blanked.
- **Apply.** Per clip: `hook_type`, `opening`, `hook_score` (0.6 x hook + 0.4 x
  opening), the accepted hook (previous one under `viral_hook_text_detail`),
  `start` moved to the first kept beat (`lead_in_trimmed` = seconds cut), and
  `teaser_pick = {start, end}` (absolute). Ranking: the detail score moves to
  `content_score` and `predicted_score = round(0.65 x content + 0.35 x
  hook_score)`, so `trim_to_best`, `dedupe_overlapping` and `clip_render_order`
  rank by the blend. A second run blends from `content_score`, never from a blend.
- **The clip count is untouched.** The pass never adds or removes clips; the
  floor logic runs before it and is not changed.
- **Rate limiting.** Sequential calls, `HOOK_PAUSE_SECONDS` apart, at most
  `HOOK_MAX_CLIPS` clips (best `predicted_score` first). A failed batch (timeout,
  policy block, bad JSON) is skipped and its clips stay as they were. The job log
  gets a one-line summary, never transcript text.

## The teaser is an automatic recut (`teaser.py`, called from `_process_one_clip`)

After the canonical render and `hook_grounding.reground`, `teaser.plan` turns
`teaser_pick` into two source segments `[teaser, body]` and `teaser.render` cuts
them with `recut.perform_recut` on the fast path (from the already-reframed
canonical clip, `reframe=False`, no captions). Why a recut and not a new concat
engine: the clip editor, every caption path (`/api/subtitle`, `_reapply_captions`,
the EDL endpoint, through `recut.virtual_transcript`) and the layout sidecar
(`layout_ranges.remap`) already understand multi-segment clips.

The clip then records exactly what the rerender endpoint stores:
`recipe = {"v": 1, "segments": [teaser, body], "canonical_range": {start, end},
"teaser": true}` and `layout_ranges` = the recut's ranges. `start`/`end` stay
the body's range (the covering range, as the rerender computes it). The editor
shows the teaser as an ordinary first segment the user can delete. The
canonical `<base>_clip_<n>.mp4` stays clean and untouched.

Hook and captions then run on the `recut_...` file: captions from
`teaser.caption_inputs` (virtual transcript, 0..teaser+body: the teaser's own
words at 0..T, the body shifted by T), split ranges and hook placement from the
recut's ranges. With a teaser the hook shows for the teaser's length (min 2.5 s)
at `font_scale` 1.3 (`auto_hook.size = "L"`, the editor's L); a brief's
`hook_seconds` (`AUTO_HOOK_SECONDS`) still wins. Served file chain:
`[wm_]subtitled_<ts>_hooked_<ts>_recut_<ts>_<hex>_<clean>`, all matched by
`_canonical_clip_file`; `CLIP_READY` names the final served file.

**Skip rules** (`plan` returns None): `AUTO_TEASER=0` or the brief's `teaser: off`;
no `teaser_pick`; `opening == "strong"`; clip shorter than
`TEASER_MIN_CLIP_SECONDS`; the pick not fully inside the final (snapped) bounds
or less than `TEASER_MIN_OFFSET` after the start; teaser outside 1.5-4.5 s; total
over `recut.MAX_TOTAL_SECONDS` or over the brief's `max_clip_seconds` + 1 s.

**Fallbacks.** A failed recut, or any exception in the teaser step, keeps the
canonical clip, its layout ranges and the existing text-only hook; a retry
re-plans from scratch. `teaser_pick` stays in the metadata as a documented hint
(absolute seconds of the picked beat), so a retried clip can still use it.

## Creator brief

The hook prompt appends `creator_brief.prompt_block(brief, "hook")`: tone and
wording only; the closed answer and the grounding rules stay authoritative. The
`teaser: off` directive (a brief asking for no cold open / teaser / replayed
intro) sets `AUTO_TEASER=0`. See `creator-brief.md`.

## Env knobs

| Variable | Default | Effect |
| --- | --- | --- |
| `AUTO_HOOKS` | `1` | `0` skips the hook pass (no trims, no teasers, detail hooks). |
| `AUTO_TEASER` | `1` | `0` never renders a cold open. |
| `TEASER_MIN_CLIP_SECONDS` | `15` | Shorter clips get no teaser. |
| `TEASER_MIN_OFFSET` | `5` | Seconds the teaser must start after the clip start. |
| `HOOK_BATCH` | `6` | Clips per hook call (2 on a local LLM). |
| `HOOK_MAX_CLIPS` | `30` | Clips judged per job. |
| `HOOK_PAUSE_SECONDS` | `0.5` | Pause between hook calls. |

## Tests

`tests/test_auto_hooks.py`, `tests/test_teaser.py` (CI) and
`tests/test_teaser_render.py`, a real ffmpeg render that skips without ffmpeg or
main.py's stack; run `run_synthetic(workdir)` in the real image for evidence.
