# API endpoints and agent access (MCP, API keys, OAuth, webhooks)

## Main endpoints

- `POST /api/process`: submit a video for processing. Optional `source_start` /
  `source_end` (seconds) clip and bill only that stretch.
  Optional `brief` (creator instructions, max 1500 characters, 400 above that)
  and `brief_preset_id` (use a saved preset's text; an explicit `brief` wins and
  the preset only supplies its name; 404 when the preset is not the caller's).
  Form field or JSON key, like `captions`. The text is sanitised and written
  only to the job's `.brief.json` sidecar, never to the child environment, the
  command line, the resume manifest or a log line. The pipeline side is in
  `docs/architecture/creator-brief.md`.
  Optional `teaser` (form or JSON, like `captions`): clips open with a ~3 s
  cold open by default; an explicit false-like value (`0/false/no`) sets
  `AUTO_TEASER=0`. `auto_hook` is ON when absent (hooks are automatic, for the
  dashboard, API, MCP and webhook callers alike); only an explicit false-like
  value turns it off. The MCP `process_video` tool forwards both (`teaser`, and
  `auto_hook=false` is sent explicitly).
- `GET /api/status/{job_id}`: poll job status and logs. A failed job also
  carries `error: {code, message}` (closed set of codes, plain message, never a
  log line; see `jobs-and-deploys.md`, "Manual retry") and
  `retry: {allowed, used, limit, reason}`; both are `null` in every other
  state. A job no instance holds any more that has a `.submit.json` (it failed
  before a restart, or its resume gave up) answers `failed` instead of 404,
  with `error.code` from `.failed.json` or `server_restarted`.
- `POST /api/jobs/{job_id}/retry`: run a failed or lost clip job again under
  the same id. No body; the Gemini key and `X-Gemini-Model` are resolved as on
  `/api/process`; everything else (argv, settings, partial, webhook URL) comes
  from the job's `.submit.json`, and the creator brief from its `.brief.json`.
  `200 {"job_id", "status": "queued", "retry": {"used", "limit"}}`. Errors
  carry `detail: {code, message}`: 404 (malformed or unknown id, another
  user's or, in cloud, an ownerless job), 401 signed out (cloud), 409
  `running` (queued/processing, or a second simultaneous request), 409
  `completed`, 409 `not_retryable` ("Please submit this video again.": a job
  from before this feature or an argv that fails the safety check), 409
  `source_missing` (the upload is gone), 429 `limit` (`MANUAL_RETRY_LIMIT`,
  default 5). Cloud reserves minutes again via `reserve_process_minutes`, so
  its 402/429 pass through unchanged. Not in the OAuth (`openshorts`) scope.
- `POST /api/edit`: apply AI video effects.
- `POST /api/subtitle`: generate and apply subtitles (auto-transcribes dubbed videos).
- `POST /api/hook`: add text hook overlays.
- `POST /api/translate`: AI voice dubbing via ElevenLabs.
- `GET /api/translate/languages`: supported dubbing languages.
- `POST /api/social/post`: post to social media (async upload). Cloud posts
  through one managed Upload-Post account, one `os_<hex>` profile per user,
  created only by `/api/social/connect` (reads never create it). Capped at
  `SOCIAL_POSTS_PER_HOUR` per user and `SOCIAL_POSTS_PER_DAY_FREE` on the free
  plan (429 with the reason); thumbnail and SaaSShorts publishing count too.
- `GET/POST /api/brief-presets`, `PUT/DELETE /api/brief-presets/{id}`: named,
  reusable briefs (`brief_presets.py`). Up to 50 per user, unique names
  (409 otherwise), name at most 60 characters on one line. Cloud stores them in
  the `brief_presets` table (every query filtered by `user_id`, erased with the
  account); self-host keeps one dotfile, `OUTPUT_DIR/.brief_presets.json`,
  which `/videos` cannot serve and the retention sweeps (directories only)
  never delete.
- `GET /api/clip-history?limit=20&offset=0`: each source video with its clips
  underneath (`clip_history.py`), newest first, `limit` clamped to 1..50. Built
  from the in-memory `jobs` (completed jobs with clips; recovered from disk
  after a restart), so there is no directory scan: only the returned page reads
  its sidecar. Cloud lists only the caller's jobs (401 without a user; ownerless
  jobs are never listed). Clips expose a fixed whitelist of fields; `video_url`
  is passed through from the job result (so the free-plan `wm_` copy rule
  holds) and `download_url` is signed the way `/api/status` does it. Clips also
  carry `hook_score` (0-100 or null), `hook_type` (closed list or null) and
  `teaser` (bool); each source carries `directives` (what a creator brief
  applied, at most 12 short str/int entries). Deleted clips are skipped and a
  source with only deleted clips is omitted. `retention_seconds` is `0` when
  history is kept until deleted (self-host default, see `jobs-and-deploys.md`).
  `failed` (not paginated, at most 20, newest first) lists the caller's failed
  or interrupted jobs: `{job_id, title, created_at, source, state:
  "failed"|"interrupted", error: {code, message}, retry: {allowed, used, limit,
  reason}}`; `title`/`source` come from the brief sidecar's source label. Jobs
  failed in memory plus job directories holding a `.submit.json` and no
  metadata that nobody holds or runs; that directory scan reads only the small
  sidecars and is cached for 30 s. Same owner scoping as `sources`.
- `DELETE /api/clip-history/{job_id}`: delete a whole source video with all its
  clips. `DELETE /api/clip-history/{job_id}/clips/{clip_index}`: delete one clip.
  Both answer `{"success": true, "deleted": "job"|"clip", "job_deleted": bool}`.
  The id must be a uuid (404 before any disk access) and the job directory is
  checked with `realpath` to be a direct child of `OUTPUT_DIR`; the job must be
  known, completed and idle (409 while it runs), and in cloud the caller must
  own it (401 signed out, 404 for another user's or an ownerless job). A single
  clip is tombstoned, not removed: its slot stays in the list with
  `deleted: true` and no `video_url` (metadata JSON and the in-memory result),
  so every other clip keeps its index; its files (clean, `subtitled_`,
  `hooked_`, `recut_`, `hook_`, `wm_` versions, `.layout.json` and hook-recipe
  sidecars) are deleted by exact name match (`_clip_1` never matches
  `_clip_11`). Restart recovery, the restore path and download-all skip
  tombstones. Deleting the last live clip deletes the job. These deletes remove
  the local working copy only (job directory, `uploads/<job_id>_*`, memory
  records); R2 and the database are untouched, the cloud library has its own
  lifecycle. Confirmation lives in the dashboard.
- `GET /api/storage`: self-host disk usage `{enabled, used_bytes, cap_bytes,
  percent, full, keep_history}` (`full` when the existing `OUTPUT_MAX_GB` cap is
  reached; cap 0 is never full). The directory walk runs in a thread and is
  cached for 60 s (`storage_usage.py`), invalidated by a delete. Cloud answers
  `{"enabled": false}`. It only reports: jobs are never blocked by it.
  `/api/config` exposes `keepHistory`.
- `POST /mcp`: MCP server (JSON-RPC), the pipeline as agent tools.
- `GET /api/jobs/{id}/transcript`: the whole-source transcript in source seconds
  (`?start=&end=` slice, `?words=true` adds word timings); a silent source
  returns empty segments and a `note`.
- `POST /api/clip/create`: render a NEW clip from source segments and append it
  (`created_by: "agent"`). Always the rerender SOURCE path (`_source_recut`),
  same lock, billing and delivery; 409 without a source or past
  `MAX_CLIPS_PER_JOB` (tombstones count), 400 on a bad EDL.
- `POST/GET/DELETE /api/keys`: user API keys (cloud mode, session JWT only).
- `DELETE /api/account`: erase the account and everything in it (GDPR art. 17).
- `GET/PUT /api/autopilot`: Autopilot settings, connected accounts, recent runs (cloud only).
- `GET /api/autopilot/videos`: latest uploads of the connected YouTube channel + what Autopilot did.
- `POST /api/autopilot/run`: clip one channel video now (`{video_id}`).

## API keys (`cloud/api_keys.py`)

`osk_...` tokens, sha256-stored, created in the dashboard account page.
`cloud/auth.get_current_user_optional` accepts them (`Bearer osk_...` or
`X-API-Key`) and resolves the owner, so metering, entitlement, plan priority and
job ownership apply to agents with zero endpoint changes. Key management, OAuth
consent, account erasure and billing (cancel, resume, retention offer, end
trial, Stripe portal, invoices) refuse API-key auth
(`cloud/auth.get_session_user_required`): a leaked key cannot mint
replacements nor touch money.

## MCP server (`mcp_server.py`, always mounted)

Stateless Streamable-HTTP JSON-RPC at `/mcp`, no SDK dependency, ~3 methods + 10
tools (`process_video`, `create_upload`, `get_job_status`, `list_clips`,
`get_quota`, `add_subtitles`, `recut_clip`, `get_transcript`, `create_clip`,
`publish_clip`). `get_transcript` + `create_clip` let the agent be the editor:
read the transcript, pick moments by meaning, render them as new clips. Each tool calls back into this same app in-process
(`httpx.ASGITransport`) forwarding the caller's auth headers, so it can never
drift from the REST behaviour. Cloud mode 401s without a resolvable user;
self-host stays BYOK-open.

## stdio transport (`mcp_stdio.py`)

The same `handle_message` / `call_tool` as a subprocess, for hosts that only
launch MCP servers as a command (e.g. Glama's Dockerfile deployments; a local
client can skip the web server). Two invariants:

- `sys.stdout` is swapped for stderr **before `app` is imported**: the pipeline
  prints everywhere and one stray line corrupts the JSON-RPC stream.
- The app's lifespan is entered (`router.lifespan_context`), which
  `ASGITransport` does not do on its own.

## OAuth for MCP clients (`cloud/mcp_oauth.py`, cloud mode only)

claude.ai and ChatGPT connect by URL, so the server publishes RFC 9728/8414
metadata under `/.well-known/`, accepts dynamic client registration
(`POST /oauth/register`, public clients, PKCE S256 mandatory) and bounces
`GET /oauth/authorize` to the dashboard consent screen (`#/oauth/authorize`),
because the session JWT lives in localStorage on the frontend host. `POST
/api/oauth/authorize` (session auth, never a key) mints the code, and Cancel asks
`POST /api/oauth/deny` for the redirect: the consent page's own URL is whatever
the link said, so it never navigates to a `redirect_uri` the API did not check
against the client's registered ones. `POST /oauth/token` redeems
it by **minting an ordinary `osk_` key** named after the client and returning it
as the access token. No new auth path: the key shows up in Account → API keys
and revoking it disconnects the app. The key never expires, but a refresh token
(`osr_`, table `oauth_refresh_tokens`, hashed) comes with it for clients that
want one (Gemini connected apps require "persistent linking"): `grant_type=
refresh_token` swaps the key's secret in place and returns the same refresh
token, and a revoked key refuses it. `POST /oauth/revoke` (RFC 7009, always 200)
revokes the OAuth key behind an access or refresh token; hand-made keys are
never revocable there. The `/mcp` 401 carries
`WWW-Authenticate: Bearer resource_metadata=...` so clients find the flow.
The key carries the granted scope (`api_keys.scope`, `cloud/scopes.py`). The only
scope is `openshorts`, the default every client asks for (unknown scopes are
dropped, not refused); it reaches `/mcp`, the endpoints the MCP tools call
in-process and the download paths, and anything else is a 403 naming the
scope. OAuth keys minted before the column are recognised by their name
suffix. Hand-made keys have no scope and keep the whole API. A new MCP tool
calling a new endpoint must add it to `scopes._MCP_ROUTES` (a test reads the
tool code and fails otherwise).
`oauth_codes` and `oauth_refresh_tokens` are in `USER_OWNED_TABLES`; `oauth_clients` deliberately not.
Registration is capped per client IP (`OAUTH_REGISTRATIONS_PER_HOUR`, 429
`slow_down`): a client registers once per connection, the cap only stops one
stuck in a loop. Agent upload slots (`POST /api/uploads`) are capped at
`MAX_PENDING_UPLOADS_PER_USER` open at once.

## Webhooks

`POST /api/process` takes `webhook_url` + optional `webhook_secret`
(HMAC-SHA256, `X-OpenShorts-Signature`). Validated with
`security_utils.assert_public_url` at submit AND at delivery (DNS rebinding).
Fired once per job from `run_job_wrapper` after the R2 archive so the payload can
carry durable download links; survives redeploys via the resume manifest.
`PUBLIC_API_URL` sets the absolute-URL base when behind a proxy.

## Media links (`/videos`, `/thumbnails`; `media_auth.py`, `app._authorize_media`)

Cloud mode serves a job's files only to a request that has a path signature
(`?exp=&sig=`, 24 h, `MEDIA_URL_TTL_SECONDS`), the owner's media token (`?mt=`
or the `os_media` HttpOnly cookie, both from `POST /api/media-token`, 12 h) or
the owner's own `Authorization` / `X-API-Key`. Owner: live job record, the
`.owner` sidecar (clip, SaaSShorts and thumbnail directories all write one),
then the `projects` row. Everything handed outside the dashboard is signed:
MCP `video_url` / `download_url`, webhook `video_url`, and a per-clip
`download_url` in `GET /api/status`. The dashboard appends `mt` in
`getApiUrl`. Bare links to jobs created before `MEDIA_LEGACY_CUTOFF` keep
working until `MEDIA_LEGACY_UNTIL` (links sent before signing existed).
Self-host is open. The R2 library links are separate: public objects behind an
unguessable key.

Jobs sent with an `osk_` key (REST, MCP, OAuth clients) do not trigger the
clips-ready email: the caller polls or gets the webhook. The job records
`via_api_key` and the resume manifest carries it.

## Payment-required errors

Every 402 (`quota_exceeded`, `no_plan`) goes through `app.payment_required`: it adds `upgrade_url` (`/#/pricing?plan=starter&src=api`, which opens the Starter checkout) and repeats the link inside `message`, since MCP clients and SDKs often show only the message. The MCP `get_quota` tool returns the same `upgrade_url` for free or unentitled users, and the server instructions tell the agent to hand that link to the user.

