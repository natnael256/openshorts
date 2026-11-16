# Job concurrency, failure handling and deploy handover

## Concurrency model

Async job queue with semaphore-based concurrency control, sized by
`MAX_CONCURRENT_JOBS` (default 5). Jobs auto-clean after 1 hour.

The real limit on a GPU host is VRAM, not CPU: each `main.py` job holds Parakeet
(onnxruntime CUDA), TransNetV2 (torch, ~2 GB at peak) and an NVENC session. When
the card is full a cut fails as "Generic error in an external library" / exit
187 with 0 bytes, TransNetV2 as CUDA OOM, and the reframe writer as a broken
pipe. The retry in `ffmpeg_utils.cut_clip` waits for a **busy** GPU; it cannot
help with a **full** one. Size `MAX_CONCURRENT_JOBS` against free VRAM first,
then check the CPU.

Guards that keep the card from filling:

- **Release the ASR model after transcribing.** `main.transcribe_video` calls
  `transcribe_backends.release_models()`; so do the in-process transcriptions in
  the API (thumbnail studio, `/api/subtitle` on a dubbed clip), otherwise the
  ASR singletons live in uvicorn for good. TransNetV2 `empty_cache()`s after each
  pass.
- **Parakeet runs lean** (`transcribe_backends.parakeet_providers`,
  `PARAKEET_VAD_BATCH`=4): 4 VAD segments per encoder batch, CUDA arena
  `kSameAsRequested`, no max cuDNN workspace. Same words, lower peak. Batch 2
  dropped a sentence and int8 was much slower with many words different: both
  rejected.
- **Host-wide transcription slots** (`transcribe_backends.host_asr_slot`,
  `ASR_HOST_SLOTS`, default 2): flock files `output/.asr-gpu-N.lock`, shared by
  every job process and by both containers of a deploy.
- **Queue admission** (`app._wait_for_shared_gpu`): a job starts only when
  running-here + running-on-the-draining-instance < `MAX_CONCURRENT_JOBS` and
  `nvidia-smi` reports at least `GPU_MIN_FREE_MB` (4500) free; an idle card
  always starts, and the wait is bounded by the drain timeout.

CPU savings that must not be undone (output verified identical by decoded-frame
MD5 + audio):

- `parakeet_session_options()` turns onnxruntime thread spinning off for the ASR
  session only; the VAD must keep `load_vad`'s defaults (see its docstring).
- `frame_sampler.read_at` reads forward once instead of seeking per sampled
  frame in `analyze_scenes_strategy` and `split_layout` (each seek re-decoded the
  GOP).

Speed at equal quality:

- **Cut on the card** (`ffmpeg_utils.cut_clip`, NVDEC → NVENC with
  `-hwaccel_output_format cuda`) for 8-bit 4:2:0 sources; byte-identical to the
  CPU cut. Other sources and a failed GPU cut decode on the CPU.
- **Silero VAD on one CPU thread** (`vad_load_kwargs`): on CUDA it ran one 32 ms
  chunk per launch.
- **Blur at quarter size** (`ffmpeg_utils.blurred_backdrop`).
- **Watermark as a served copy** (`main.mark_delivery`, see `watermark.md`):
  after captions. It must not ride the reframe encode, which would make it
  permanent. The `wm_` copy is one more output of the ffmpeg that burns the
  captions (`add_hook_to_video(mark=)` with a hook, `burn_subtitles(mark=)`
  without); only a clip with no captions gets its own `mark_delivery` pass.
  Those multi-output passes can segfault on exit after every file is written
  (`ffmpeg_utils.finished_before_crash` keeps the files then).
- **hooked_ + subtitled_ from one ffmpeg** (`hooks.add_hook_to_video(also=)`):
  the editor still needs both files (re-caption walks back to hooked_, hook
  replace to the canonical), so nothing is skipped, only one decode.
- **Cheap cut** (`ffmpeg_utils.INTERMEDIATE`): the cut is decoded again by the
  reframe at once, so on NVENC it uses p1 at a lower `-cq` (`CUT_NVENC_PRESET`,
  `CUT_NVENC_CQ`) instead of p4 + hq + AQ. libx264 keeps its old settings.
- **Thread cap** (`bin/ffmpeg`, first on the image's `PATH`): every ffmpeg gets
  `-threads FFMPEG_THREADS` per input and the same for its filter graphs.
  Uncapped, each one sized its pools to the whole machine, so a few jobs kept
  more runnable threads than cores. Decoding and filtering are deterministic:
  the clips are byte-identical. Unset or 0 = uncapped.
- **Jobs run niced** (`JOB_NICE`, default 10, `app._lower_priority`): under CPU
  pressure the API and the editor endpoints, which run in the API process, go
  first. Containers compete by their CPU weight, not by nice, so this only
  orders work inside the instance.
- **NVENC encode slots** (`JOB_NVENC_SLOTS`, default 4, `app._job_encode_env`
  → `bin/ffmpeg`): at most that many job ffmpegs encode with NVENC at once,
  host-wide (flock files `output/.nvenc-N.lock`, shared by both containers of
  a deploy). A few encodes already keep the card's NVENC engines busy; with
  5 jobs × `CLIP_WORKERS` uncapped, dozens of sessions split the same engines
  and starved any other encoder on the card without finishing jobs sooner.
  Exempt: the nvenc probe (`NVENC_SLOTS=0` in its env), pipe-fed encoders,
  and the API process (editor renders). A queued encode runs anyway after
  `NVENC_SLOT_WAIT` (600 s). 0 turns it off.
- **NVENC on p3** (QUALITY, QUALITY_FAST, DELIVERY; was p5, then p4, oct-2026):
  same hq tune and AQs. p3 against p4 on real clips: ~25% less engine time,
  files ~13% larger, SSIM to the source within 0.001; p2 bought little more
  speed for ~21% larger files and p1 is no faster than p2.
- **`CLIP_WORKERS=6`** by default.
- **Decode on the card, keep the CPU's pixels** (`gpu_decode`): the scene
  pass, the sampled frames of the classifiers (`frame_sampler.read_at_path`),
  the trajectory, every reframe segment and the hook/caption/watermark burns
  decode with NVDEC. Frames come down as nv12 and are repacked to yuv420p
  before any other filter (`gpu_decode.DOWNLOAD`): nv12 converted straight to
  BGR differs from the CPU decode, the repacked frame does not. Only 8-bit
  4:2:0 H.264/HEVC without rotation, and only with `GPU_DECODE_MIN_FREE_MB`
  free on the card; any ffmpeg that fails on the card runs again on the CPU.
  Kill switches: `GPU_DECODE=0` (all), `GPU_DECODE_RENDER=0` (encodes only).
- **One-pass GPU render** (`gpu_render` + `osrender/`, GPU image only): the
  reframe render (`reframe_v2.encode_scenes`) runs as a single process that
  keeps the frames on the card (NVDEC -> CUDA kernels -> NVENC) instead of one
  ffmpeg per scene downloading every frame for crop/scale/blur on the CPU and
  a concat. The layers come from the scenes' own filtergraphs
  (`gpu_render.layers_from_graph`, which mirrors ffmpeg's even-coordinate
  overlay/pad/crop and sendcmd's frame timing); output matches the ffmpeg path
  at SSIM >= 0.97 per layout (`tests/test_gpu_render_equivalence.py`). It takes
  an NVENC slot like `bin/ffmpeg`. Unreadable graph, unsupported source, no
  binary or any failure: the scene-by-scene ffmpeg render, unchanged.
  `GPU_RENDER=0` turns it off. Punch-in only renders whole here: ffmpeg's crop
  ends the stream at the first size change a sendcmd makes.
- **Hooked-only picture on demand** (`hooks.materialize`, `HOOK_LAZY`): the
  job's hook + captions pass encodes only what is served (captioned, plus the
  free plan's `wm_` copy). The hooked-only `hooked_<ts>_<clean>` is needed
  only when the editor re-captions or removes the captions, so the pass keeps
  a recipe (`.hookrecipe_<name>.json` + `.png`: overlay image, position,
  window) and `app._strip_burned_captions` encodes it the first time it is
  asked for, off the event loop; `/api/hook` strips past it without encoding.
  Same filter and encoder as before, so the file is the same. No recipe or a
  failed encode: the walk-back stops as it always did. `HOOK_LAZY=0` encodes
  it in the pass again.
- **Sparse trajectory** (`reframe_v2._sparse_trajectory`, `SPARSE_TRAJECTORY`):
  the detectors look at one frame in `DETECT_STRIDE` plus scene starts, and only
  in TRACK scenes, so only those are scaled and piped (ffmpeg `select`). It
  needs the clip's frame count from the scene pass, kept only when that
  rawvideo pipe repeated or dropped no frame (`dup=`/`drop=` in its report):
  the trajectory counts the frames of a constant-frame-rate pipe, `select`
  counts decoded frames, and the two agree only then. Any mismatch restores the
  camera and runs the full pass.
- **Per-job CPU** (`JOB_CPU` marker at main.py exit, logged by the API as
  `[job-cpu]`): CPU-seconds of the job process plus every ffmpeg it waited for,
  wall time and the GPU decode counters, to compare pipeline versions per job.

Tried and dropped: `scale_cuda`/`scale_npp` for the analysis frames (their
pixels differ from swscale's, so the detectors would see other frames); yt-dlp chunking/concurrent fragments (no
gain). Method for such changes: record Gemini decisions once and replay them,
run old and new code side by side, compare clips by decoded-frame MD5 / SSIM.

## Parakeet writes some English in Cyrillic

`transcribe_backends._repair_script_drift`. Parakeet v3 picks the language per
VAD segment, so on short segments it can spell English phonetically in
Ukrainian/Russian ("Тудей Марк із гана такі сторі") while the file still reads
`en` and the whisper fallback never fires. Segments whose script cannot belong to the file's language are
re-transcribed with whisper forced to that language; any failure keeps the
parakeet text.

## Failures the user should never see (`app.run_job_wrapper`)

- **Auto-retry**: a failed job whose error text is transient (CUDA/OOM, NVENC
  "Generic error in an external library", cublas, Gemini 5xx, "No clips could be
  rendered") is re-queued once after `AUTO_RETRY_DELAY_SECONDS` (30), keeping its
  reservation and transcript checkpoint (`AUTO_RETRY_LIMIT`=1). Content failures
  (no audio, private video, no clips found, policy block) are final. Inside a
  job, `main.py` retries each failed clip once, alone, after a pause, and renders
  clips best-score first.
- **Shutdown is not failure**: when the drain timeout cancels a running job, the
  child is killed and the manifest + reservation are kept, so the next instance
  resumes it. A manifest next to a metadata file means "stopped mid-render" and
  is resumed, not recovered as completed.
- `/api/status` returns `queue: {position, ahead, eta_seconds}` while queued; the
  dashboard shows it with a "paid plans skip the line" upsell (paid plans do
  dispatch first: `PLAN_PRIORITY`).

## Manual retry (`job_retry.py`, `POST /api/jobs/{job_id}/retry`)

A failed (or lost) clip job can be run again by the user under the same job id,
without re-entering anything. Two dotfile sidecars in the job directory make it
possible (`/videos` never serves dot segments):

- **`.submit.json`** (`job_retry.write_submit`, written by `/api/process` right
  after the resume manifest): the argv, the allow-listed per-job settings
  (`job_retry.SETTING_KEYS`: the `layout_env` flags, `AUTO_HOOK`,
  `AUTO_HOOK_STYLE`, `AUTO_TEASER`, `AUTO_CAPTIONS`, `CLIP_TARGET_MIN/MAX`,
  `CLIP_MIN/MAX_SECONDS`, an allow-listed `GEMINI_MODEL`; only the values the
  job changed from the server's own env), priority, webhook URL, partial cut,
  source cap, skip-statics, the reservation id and the manual-retry counter.
- **`.failed.json`** (`run_job_wrapper`, terminal path): the failure's code from
  `job_retry.classify_failure` (closed set: `missing_key`, `download_blocked`,
  `video_unavailable`, `no_clips`, `no_audio`, `content_blocked`, `too_short`,
  `model_overloaded`, `model_unavailable`, `out_of_memory`, `server_restarted`,
  `quota`, `unknown`), so the reason survives a restart. The message is fixed per
  code, never a log line. A completed run removes it. `model_unavailable` is
  Gemini's 404 for a retired model; it is matched before `video_unavailable`,
  whose markers are phrased as YouTube words them so an AI-side "no longer
  available" never blames the video.
- A `GEMINI_MODEL` off `app.GEMINI_MODEL_CHOICES` (request header, stored
  `.submit.json`, or the dashboard's remembered choice) is dropped and the job
  runs on the default model, so a job stored with a retired model succeeds on
  retry.

**Never persisted** in either: the Gemini key (the retry request brings it, as
`/api/process` does: `resolve_gemini`), proxy URLs, cookies, any other
credential, the webhook secret (a retried job keeps its HMAC secret only while
the failed record is still in memory; after a restart it is delivered unsigned),
and the creator brief (it stays in `.brief.json` and main.py reads it there, as
on the first run; the retry endpoint accepts no brief).

**Retry rules.** Only a `failed` job (in memory) or a lost one (a directory
with `.submit.json`, no manifest, no metadata, not in memory: `/api/status`
answers it as `failed` with `lost: true` instead of 404) that is not active
(`_job_is_active`). The stored argv runs only if `job_retry.cmd_is_safe`
accepts it (exactly what `/api/process` builds; `-o` this job's directory, `-i`
a `<job_id>_*` file directly in `uploads/`) and with the interpreter rebuilt
as `sys.executable`. At most `MANUAL_RETRY_LIMIT` (5) retries per job, counted
in `.submit.json` (429 past it). An in-process guard makes two simultaneous
requests for one job enqueue it once (the second gets 409). Upload jobs need
their file in `uploads/` (409 `source_missing` once the upload sweep took it).
URL jobs download again: main.py deletes and re-downloads an existing
`<title>.mp4` (`download_youtube_video` removes `expected` before yt-dlp, with
`overwrites`), so a kept source is not reused; the transcript checkpoint is.

**What is reused**: the job id and directory, the source upload, the transcript
checkpoint, `.owner`, `.brief.json`, `.submit.json`. `_clean_for_retry` drops
the half-made outputs, and the manual retry also drops `.failed.json` and
`.brief_applied.json`. The new record starts with `auto_retries` 0, so a
transient failure of the retry still gets its one automatic retry.

**Billing.** A failed job's reservation is released by `_settle_reservation`
(it commits only on `completed`). The retry reserves again through the same
`reserve_process_minutes` as `/api/process` (job limit, balance, probe), with
the stored partial (`max_minutes`) or chosen range (`source_range`), so a cloud
user is neither charged twice nor given a free run. Before reserving it calls
`release_reservation` on the old id (memory record or `.submit.json`): a no-op
for a row that is not `reserved` any more, and the guarantee that a reservation
somehow still held is returned rather than doubled. A 402/429 from the
reservation leaves the job untouched (still failed, counter not bumped,
nothing cleaned). The watermark follows the user's current plan. The quality
gate does not run again.

**Resume now re-applies settings.** `_resume_interrupted_jobs` used to rebuild
the child env from `child_env()` and the manifest only, so a resumed job lost
its layouts, hook, clip controls and model. It now applies `.submit.json`'s
settings first (`_apply_submit_settings`, re-validated). It still cannot bring
back a BYOK key that came in a header: a resumed self-host job without a
server key fails with `missing_key`, and the manual retry is the way out.

## Deploys and running jobs (handover + drain)

Every push to `main` redeploys the API container as a rolling update: the NEW
container starts before the old one stops and both share `output/`, so `app.py`
coordinates them:

- Each instance writes its id to `output/.instance` at startup. An instance that
  sees another id there is the old one and **drains**: it finishes the jobs it is
  running, starts none, and leaves queued manifests on disk.
- A running job heartbeats its `.resume.json` every 10 s. The resume scan
  (startup + every 30 s) re-enqueues only manifests nobody heartbeated for 60 s,
  so no job runs twice and none is lost. Max 2 resume attempts.
- SIGTERM drains too, up to `DRAIN_TIMEOUT_SECONDS` (840), then hands the signal
  to uvicorn. **Keep it below the deployment platform's stop grace period.**
  `--timeout-graceful-shutdown 15` (Dockerfile) caps the wait for in-flight
  connections; uvicorn's default is unbounded and one open range download can
  keep a drained container alive for the whole grace period.
- `/health/ready` + the Dockerfile `HEALTHCHECK` keep the reverse proxy off a
  dying container (it only routes to `healthy` containers). An instance answers
  503 from the moment it gets SIGTERM, and a booting one gets no traffic until it
  answers. Only SIGTERM flips it, not the marker drain (at that point the new
  container is still booting). The platform health check replaces the
  Dockerfile one with a curl/wget command, so **the image must ship `curl`** or
  every deploy rolls back as unhealthy. The drain keeps serving for
  `PROXY_DRAIN_SECONDS` (20) after the jobs are done so the proxy notices the 503
  before the socket closes, and `HARD_EXIT_SECONDS` (30) after that the process
  is ended outright (an executor thread hung in a network probe would otherwise
  keep the interpreter alive). `/health` stays a plain liveness probe.
- `/api/status` answers from disk for a job this instance never held, so a poll
  landing on either container during the handover is fine.
- `main.py` leaves `.transcript_checkpoint.json` in the job dir so a re-run job
  skips the paid transcription (download and Gemini repeat).

Every deploy is a full build plus a handover, so batch small commits (tests,
docs) with the next real change.

## History retention

Self-host keeps every completed clip job (a job directory holding a
`*_metadata.json`, what History lists) until the user deletes it:
`_sweep_expired_job_dirs` skips it and `_enforce_output_size_cap` never picks it
(`KEEP_HISTORY`, on when `BILLING_ENABLED` is off; `HISTORY_KEEP=0` restores the
old `JOB_RETENTION_SECONDS` clock). The `OUTPUT_MAX_GB` cap stays the only cap:
for kept jobs it becomes the "storage full" warning of `GET /api/storage`
instead of a trim. Still aging out as before: failed jobs (no metadata, or a
`failed` record), SaaSShorts and scratch directories, and source uploads in
`uploads/` (`_sweep_expired_uploads` and its cap are unchanged: an uploaded
source file still ages out, while a URL job's downloaded source lives in the job
directory and is kept with it). Cloud is unchanged (1 h clock, clips in R2).
Deleting from History removes the local copy only.
