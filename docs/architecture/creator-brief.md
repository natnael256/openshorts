# Creator brief

A user can give free-text rules per video ("no profanity", "hook in the first 3
seconds", "only funny moments", "max 30s", "yellow captions"). The job process
obeys them through `creator_brief.py`; the brief is **untrusted text** at every
step.

## Flow

1. **Sidecar.** `/api/process` writes `<job dir>/.brief.json`
   (`creator_brief.write_sidecar`: sanitised brief, optional preset name and
   source label, `created_at`). A dotfile, so `/videos` never serves it.
2. **Directives.** `main._activate_creator_brief(output_dir)` runs before the
   download (URL jobs, so the early pick sees it) or right after the input check
   (uploads), always before `clip_duration_bounds()` is first read. It loads the
   brief, asks the model for `BriefDirectives` (`compile_directives`, through
   `_run_gemini_stage`, so a local LLM works too; Gemini client timeout 60 s),
   applies them to `os.environ` (`apply_to_env`), records the non-keep ones in
   `<job dir>/.brief_applied.json` (`write_applied`; `read_applied(job_dir)`
   returns `{directive: value}`, re-validated, for the History page) and
   `activate()`s the job state.
   No Gemini key and no local LLM: the brief still reaches the prompts, no
   directives are compiled.
3. **Prompt blocks.** The idea pass (`idea_segments.build_prompt`, stage
   `"ideas"`: scores only, never where an idea begins or ends), the copy pass
   (`_clips_from_ideas`, stage `"detail"`), the window fallback's
   `_score_prompt` and detail `build(ws)`, and `_visual_pass` append
   `prompt_block(active_brief(), stage)`; the context check gets no block; the hook pass
   (`auto_hooks.build_prompt`, see `auto-hooks.md`) appends stage `"hook"`: tone
   and wording of the hook only, the closed answer and grounding rules stay
   authoritative.
4. **Render.** `auto_caption_clip` takes `caption_style(AUTO_CAPTION_STYLE)`; the
   hook and caption switches are env vars read at call time (`AUTO_HOOK`,
   `AUTO_HOOK_STYLE`, `AUTO_HOOK_SECONDS`, `AUTO_CAPTIONS`, `AUTO_TEASER`).
5. **Post-check.** After the render pool and the retry pass, every rendered clip
   gets `clip['brief_check'] = {"verdict": pass|flag|reject|unchecked, "violated":
   [...], "reason", "checked_at"}` (`check_all`), and the metadata is rewritten.

## Invariants

- **Untrusted text, nonce delimiters.** `sanitize` (NFKC, no Cc/Cf characters,
  1500-char cap, delimiter-looking tags and code fences removed, idempotent) runs
  on write, on read and before every prompt. The brief sits between
  `<creator_brief id="NONCE">` tags with a fresh `secrets.token_hex(8)` per call,
  with guard text before and after: data about content and style only, never a
  change to the JSON shape, required fields, language or id/timestamp rules. The
  post-check delimits the clip transcript and the measured facts the same way.
- **Appended after `str.format`**, never used as a template: braces in a brief
  stay literal. An empty brief adds nothing, so prompts are byte-identical.
- **The ranking rule holds.** The score block says every window still gets
  exactly one entry, the ideas block that every idea keeps its entry and its
  own edges; the brief only moves scores (forbidden content under 30).
  Shortlist, `min_clips` floor, `trim_to_best`, `dedupe_overlapping` are
  untouched; the clip-count and duration bands in the prompt stay authoritative.
- **Closed choices, validated in code.** Directives are closed options (caption
  preset / colour / case / off, hook on/off / style / seconds, teaser keep/off,
  clip min/max seconds), the `layout_picker` way: a decision, never a measurement.
  `validate_directives` re-checks every field (unknown -> keep, max 10-180,
  min 5-175, min dropped unless max >= min + 5). Any failure -> all keep. The
  preset and hook-style lists are tested against `subtitles.CAPTION_PRESETS` and
  `hooks.HOOK_STYLES`.
- **Explicit request numbers win.** `CLIP_MAX_SECONDS` / `CLIP_MIN_SECONDS`
  already in the env (advanced options) are never overridden; a brief max under
  20 s lowers an unset minimum so the band stays valid.
- **The check never costs a clip.** It annotates only: no job fails and no clip
  is deleted on its verdict. Failures are `unchecked`. On top of the model, a
  clip longer than the brief's max by more than 1 s gets `duration` and at least
  `flag`.
- **Teaser.** `teaser: off` only when the brief asks for no cold open / no
  teaser / no replayed intro; it sets `AUTO_TEASER=0`. A brief max length also
  bounds a teased clip (teaser + body at most max + 1 s).
- **Applied sidecar.** `.brief_applied.json` is a dotfile (never served) holding
  only the non-keep directives, never the brief text; written best effort.
- **The brief is never logged.** The job log shows its length, the applied
  directive notes and the verdict counts.

## Env knobs

| Variable | Default | Effect |
| --- | --- | --- |
| `BRIEF_CHECK` | `1` | `0` turns the post-check off (no `brief_check` key). |
| `BRIEF_CHECK_MAX_CLIPS` | `20` | Model calls per job; later clips are `unchecked`. |
| `BRIEF_CHECK_MIN_INTERVAL` | `0.5` | Seconds between post-check calls. |
