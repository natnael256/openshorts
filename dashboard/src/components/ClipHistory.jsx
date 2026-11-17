import React, { useState, useEffect, useCallback } from 'react';
import { Loader2, Download, Film, ChevronDown, Trash2, RotateCcw, Copy, Check } from 'lucide-react';
import { copyText } from '../lib/clipboard';
import { apiJson, apiFetch, QuotaError, openQuotaWall } from '../lib/api';
import { retryJob } from '../lib/jobRetry';
import { getApiUrl } from '../config';

const PAGE = 20;

const VERDICTS = {
    pass: { label: 'pass', cls: 'text-ok border-ok' },
    flag: { label: 'flagged', cls: 'text-brass border-brass' },
    reject: { label: 'rejected', cls: 'text-danger border-danger' },
    unchecked: { label: 'not checked', cls: 'text-muted border-rule2' },
};

// Verdict as a text label plus colour. `check` may be null (no brief).
export function VerdictBadge({ check }) {
    if (!check || !check.verdict) return null;
    const v = VERDICTS[check.verdict] || VERDICTS.unchecked;
    const violated = Array.isArray(check.violated) ? check.violated : [];
    return (
        <div className="mt-2">
            <span className={`inline-block text-[11px] font-mono lowercase px-2 py-0.5 rounded-full border ${v.cls}`}>
                {v.label}
            </span>
            {violated.length > 0 && (
                <span className="text-[11px] text-muted ml-2 lowercase">{violated.join(' · ')}</span>
            )}
            {check.reason ? (
                <p className="text-xs text-muted mt-1 leading-relaxed break-words">{check.reason}</p>
            ) : null}
        </div>
    );
}

const HOOK_TYPES = {
    curiosity_gap: 'curiosity gap',
    bold_claim: 'bold claim',
    contrarian: 'contrarian',
    mistake: 'mistake',
    open_loop: 'open loop',
};

// Hook score, hook type and cold-open tag. Renders nothing for missing values.
export function HookInfo({ score, type, teaser }) {
    const n = score == null || score === '' ? NaN : Number(score);
    const hasScore = Number.isFinite(n);
    const typeLabel = type ? (HOOK_TYPES[type] || String(type).replace(/_/g, ' ')) : '';
    if (!hasScore && !typeLabel && !teaser) return null;
    const tone = n >= 75 ? 'text-ok' : n >= 50 ? 'text-brass' : 'text-muted';
    return (
        <div className="mt-2 flex flex-wrap items-center gap-1.5">
            {hasScore && (
                <span className={`inline-block text-[11px] font-mono lowercase px-2 py-0.5 rounded-full border border-rule2 ${tone}`} title="how strong the opening hook is, out of 100">
                    hook {Math.round(n)}
                </span>
            )}
            {typeLabel && <span className="text-[11px] text-muted lowercase">{typeLabel}</span>}
            {teaser && (
                <span className="inline-block text-[11px] font-mono lowercase px-2 py-0.5 rounded-full border border-rule2 text-ink2" title="opens with its most gripping moment, then plays from the start">
                    cold open
                </span>
            )}
        </div>
    );
}

const formatBytes = (b) => {
    const n = Number(b) || 0;
    if (n < 1024) return `${n} B`;
    const units = ['KB', 'MB', 'GB', 'TB'];
    let v = n / 1024;
    let i = 0;
    while (v >= 1024 && i < units.length - 1) { v /= 1024; i += 1; }
    return `${v >= 100 ? Math.round(v) : v.toFixed(1)} ${units[i]}`;
};

const DIRECTIVE_LABELS = {
    caption_preset: 'caption style',
    caption_color: 'caption colour',
    caption_case: 'caption case',
    captions: 'captions',
    hook_overlay: 'hook overlay',
    hook_style: 'hook style',
    hook_seconds: 'hook length',
    max_clip_seconds: 'max clip length',
    min_clip_seconds: 'min clip length',
    teaser: 'cold open',
};
const SECONDS_KEYS = ['hook_seconds', 'max_clip_seconds', 'min_clip_seconds'];

function directiveChips(d) {
    if (!d || typeof d !== 'object') return [];
    return Object.entries(d).map(([k, v]) => {
        let val;
        if (typeof v === 'boolean') val = v ? 'on' : 'off';
        else if (SECONDS_KEYS.includes(k) && v != null && v !== '' && !Number.isNaN(Number(v))) val = `${v}s`;
        else val = String(v).replace(/_/g, ' ');
        return `${DIRECTIVE_LABELS[k] || k}: ${val}`;
    });
}

const errDetail = async (res) => {
    try {
        const j = await res.json();
        if (j && typeof j.detail === 'string') return j.detail;
    } catch { /* ignore */ }
    return res.status === 401 ? 'Sign in again to delete.' : 'Could not delete.';
};

// Inline confirm (no window.confirm): `ask` is the question, `onYes` runs the delete.
function ConfirmDelete({ ask, busy, error, onYes, onCancel }) {
    return (
        <div className="mt-2" role="group" aria-label="confirm delete">
            <p className="text-xs text-ink2 lowercase break-words">{ask}</p>
            <div className="mt-1.5 flex flex-wrap gap-2">
                <button type="button" className="btn-ghost px-3 py-1 text-xs text-danger" disabled={busy} onClick={onYes}>
                    {busy ? <><Loader2 size={12} className="animate-spin" /> deleting…</> : 'yes, delete'}
                </button>
                <button type="button" className="btn-ghost px-3 py-1 text-xs" disabled={busy} onClick={onCancel}>cancel</button>
            </div>
            {error ? <p className="text-danger text-xs mt-1.5 break-words" role="alert">{error}</p> : null}
        </div>
    );
}

const fmtDate = (epoch) => (epoch
    ? new Date(epoch * 1000).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })
    : '');

const fmtTime = (s) => {
    const n = Math.max(0, Math.round(Number(s) || 0));
    return `${Math.floor(n / 60)}:${String(n % 60).padStart(2, '0')}`;
};

const retentionText = (secs) => {
    const s = Number(secs);
    if (!s || s <= 0) return '';
    const hours = Math.round(s / 3600);
    if (hours < 48) return `${Math.max(1, hours)} hour${hours === 1 ? '' : 's'}`;
    const days = Math.round(s / 86400);
    return `${days} days`;
};

function summarize(clips) {
    const c = { pass: 0, flag: 0, reject: 0 };
    for (const clip of clips) {
        const v = clip.brief_check && clip.brief_check.verdict;
        if (v in c) c[v] += 1;
    }
    const parts = [];
    if (c.pass) parts.push(`${c.pass} pass`);
    if (c.flag) parts.push(`${c.flag} flagged`);
    if (c.reject) parts.push(`${c.reject} rejected`);
    return parts.join(' · ');
}

// One failed or stuck job: why it stopped and a retry. The server re-runs the same
// video with the same settings and instructions, so only the key/model travel.
function FailedRow({ f, apiKey, geminiModel, onRetried }) {
    const [busy, setBusy] = useState(false);
    const [started, setStarted] = useState(false);
    const [problem, setProblem] = useState('');
    const retry = f.retry || null;
    const used = retry ? Number(retry.used) || 0 : 0;
    const limit = retry ? Number(retry.limit) || 0 : 0;
    const label = (f.source && f.source.label) || '';
    const reason = (f.error && f.error.message) || '';

    const doRetry = async () => {
        if (busy) return;
        setBusy(true);
        setProblem('');
        try {
            await retryJob(f.job_id, { apiKey, geminiModel });
            // 'background': another job is running, so the row stays, marked.
            // Otherwise the parent removes this row.
            const outcome = onRetried(f.job_id);
            if (outcome === 'background') { setStarted(true); setBusy(false); }
        } catch (e) {
            if (e instanceof QuotaError) openQuotaWall(e, 'history-retry');
            else setProblem((e && e.message) || 'Could not retry.');
            setBusy(false);
        }
    };

    return (
        <li className="card p-3 flex flex-col sm:flex-row sm:items-start gap-3">
            <div className="min-w-0 flex-1">
                <p className="text-sm text-ink font-medium truncate" title={f.title}>{f.title || label || 'Project'}</p>
                <p className="readout mt-0.5">
                    {fmtDate(f.created_at)} · <span className={f.state === 'interrupted' ? 'text-brass' : 'text-danger'}>{f.state === 'interrupted' ? 'interrupted' : 'failed'}</span>
                    {used > 0 ? ` · retry ${used}${limit > 0 ? ` of ${limit}` : ''} used` : ''}
                </p>
                {label && label !== f.title ? <p className="text-xs text-muted truncate mt-0.5" title={label}>{label}</p> : null}
                {reason ? <p className="text-xs text-ink2 mt-1 break-words">{reason}</p> : null}
                {retry && retry.allowed === false ? (
                    <p className="text-xs text-muted mt-1 break-words">{retry.reason || 'This video cannot be retried.'}</p>
                ) : null}
                {started ? <p className="text-xs text-ok mt-1 lowercase" role="status">retry started</p> : null}
                {problem ? <p className="text-xs text-danger mt-1 break-words" role="alert">{problem}</p> : null}
            </div>
            {retry && retry.allowed !== false && !started ? (
                <button type="button" className="btn-ghost px-3 py-1.5 text-xs inline-flex items-center gap-1.5 self-start shrink-0" disabled={busy} onClick={doRetry}>
                    {busy ? <><Loader2 size={12} className="animate-spin" /> retrying...</> : <><RotateCcw size={12} /> retry</>}
                </button>
            ) : null}
        </li>
    );
}

// One click to copy the clip's title as the post caption.
function CopyCaption({ text }) {
    const [copied, setCopied] = useState(false);
    const copy = async () => {
        if (await copyText(text)) {
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
        }
    };
    return (
        <button type="button" onClick={copy} className="mt-2 mr-3 text-micro font-mono uppercase text-brass hover:text-ink inline-flex items-center gap-1 transition-colors" title="Copy the title to paste as the caption">
            {copied ? <><Check size={14} /> Copied</> : <><Copy size={14} /> Caption</>}
        </button>
    );
}

function SourceEntry({ s, onJobDeleted, onClipDeleted }) {
    const [open, setOpen] = useState(false);
    // 'job' | clip index | null: which delete is awaiting confirmation.
    const [confirm, setConfirm] = useState(null);
    const [busy, setBusy] = useState(false);
    const [delError, setDelError] = useState('');
    const clips = [...(s.clips || [])].sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
    const chips = directiveChips(s.directives);

    const doDelete = async (clipIndex) => {
        setBusy(true);
        setDelError('');
        try {
            const base = `/api/clip-history/${encodeURIComponent(s.job_id)}`;
            const path = clipIndex == null ? base : `${base}/clips/${encodeURIComponent(clipIndex)}`;
            const res = await apiFetch(path, { method: 'DELETE' });
            if (!res.ok) {
                setDelError(await errDetail(res));
                setBusy(false);
                return;
            }
            let body = {};
            try { body = await res.json(); } catch { /* ignore */ }
            setBusy(false);
            setConfirm(null);
            if (clipIndex == null) onJobDeleted(s.job_id);
            else onClipDeleted(s.job_id, clipIndex, !!body.job_deleted);
        } catch {
            setDelError('Could not delete.');
            setBusy(false);
        }
    };
    const summary = summarize(clips);
    const label = s.source && s.source.label;
    const panelId = `brief-${s.job_id}`;

    return (
        <section>
            <div className="mb-3 pb-2 border-b border-rule">
                <p className="text-sm text-ink font-medium truncate" title={s.title}>{s.title || 'Project'}</p>
                <p className="readout mt-0.5">
                    {fmtDate(s.created_at)} · {clips.length} clip{clips.length === 1 ? '' : 's'}
                    {summary ? ` · ${summary}` : ''}
                </p>
                {label ? (
                    <p className="text-xs text-muted truncate mt-0.5" title={label}>{label}</p>
                ) : null}
                <button
                    type="button"
                    onClick={() => setOpen((v) => !v)}
                    aria-expanded={open}
                    aria-controls={panelId}
                    className="mt-2 flex items-center gap-1.5 text-xs text-muted hover:text-ink2 lowercase transition-colors"
                >
                    <ChevronDown size={14} className={`transition-transform ${open ? 'rotate-180' : ''}`} />
                    instructions used
                </button>
                {confirm === 'job' ? (
                    <ConfirmDelete
                        ask={`delete this video and its ${clips.length} clip${clips.length === 1 ? '' : 's'}?`}
                        busy={busy}
                        error={delError}
                        onYes={() => doDelete(null)}
                        onCancel={() => { setConfirm(null); setDelError(''); }}
                    />
                ) : (
                    <button
                        type="button"
                        disabled={busy}
                        onClick={() => { setConfirm('job'); setDelError(''); }}
                        className="mt-2 flex items-center gap-1.5 text-xs text-muted hover:text-danger lowercase transition-colors disabled:opacity-45"
                    >
                        <Trash2 size={13} /> delete video
                    </button>
                )}
                {open && (
                    <div id={panelId} className="mt-2 animate-fade">
                        {s.preset_name ? <p className="readout mb-1">preset: {s.preset_name}</p> : null}
                        {s.brief ? (
                            <p className="whitespace-pre-wrap break-words text-xs text-ink2 leading-relaxed bg-paper3 rounded-input p-3">{s.brief}</p>
                        ) : (
                            <p className="text-xs text-muted lowercase">no instructions</p>
                        )}
                        {chips.length > 0 && (
                            <div className="mt-2">
                                <p className="text-xs text-muted lowercase mb-1">set by your instructions:</p>
                                <div className="flex flex-wrap gap-1.5">
                                    {chips.map((c) => (
                                        <span key={c} className="text-[11px] font-mono lowercase px-2 py-0.5 rounded-full bg-paper3 text-ink2 break-words">{c}</span>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4 sm:gap-5">
                {clips.map((c) => {
                    const src = getApiUrl(c.video_url || '');
                    const dl = c.download_url ? getApiUrl(c.download_url) : src;
                    return (
                        <div key={`${s.job_id}-${c.index}`} className="card overflow-hidden">
                            <div className="aspect-[9/16] bg-black">
                                {c.video_url ? (
                                    <video src={src} controls preload="metadata" playsInline className="w-full h-full object-contain" />
                                ) : null}
                            </div>
                            <div className="p-3">
                                <p className="text-sm text-ink font-medium line-clamp-2 break-words" title={c.title}>{c.title || 'Short'}</p>
                                {c.hook ? <p className="text-xs text-ink2 mt-1 line-clamp-2 break-words" title={c.hook}>{c.hook}</p> : null}
                                <p className="readout mt-1">
                                    {fmtTime(c.start)}–{fmtTime(c.end)} · {Math.round(Number(c.duration) || 0)}s
                                </p>
                                <VerdictBadge check={c.brief_check} />
                                <HookInfo score={c.hook_score} type={c.hook_type} teaser={c.teaser} />
                                {c.title ? <CopyCaption text={c.title} /> : null}
                                {c.video_url ? (
                                    <a
                                        href={dl}
                                        download
                                        className="mt-2 text-micro font-mono uppercase text-brass hover:text-ink inline-flex items-center gap-1 transition-colors"
                                        title="Download"
                                    >
                                        <Download size={14} /> Download
                                    </a>
                                ) : null}
                                {confirm === c.index ? (
                                    <ConfirmDelete
                                        ask="delete this clip?"
                                        busy={busy}
                                        error={delError}
                                        onYes={() => doDelete(c.index)}
                                        onCancel={() => { setConfirm(null); setDelError(''); }}
                                    />
                                ) : (
                                    <button
                                        type="button"
                                        disabled={busy}
                                        onClick={() => { setConfirm(c.index); setDelError(''); }}
                                        className="mt-2 flex items-center gap-1 text-xs text-muted hover:text-danger lowercase transition-colors disabled:opacity-45"
                                    >
                                        <Trash2 size={13} /> delete
                                    </button>
                                )}
                            </div>
                        </div>
                    );
                })}
            </div>
        </section>
    );
}

// History grouped by source video: each source with the clips cut from it, the
// instructions used and the per-clip brief verdicts.
export default function ClipHistory({ apiKey, geminiModel, onRetried }) {
    const [sources, setSources] = useState(null);
    const [failed, setFailed] = useState([]);
    const [total, setTotal] = useState(0);
    const [retention, setRetention] = useState(0);
    const [storage, setStorage] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const load = useCallback(async (offset) => {
        setLoading(true);
        setError('');
        try {
            const d = await apiJson(`/api/clip-history?limit=${PAGE}&offset=${offset}`);
            setSources((cur) => (offset === 0 ? (d.sources || []) : [...(cur || []), ...(d.sources || [])]));
            if (offset === 0) setFailed(Array.isArray(d.failed) ? d.failed : []);
            setTotal(Number(d.total) || 0);
            setRetention(Number(d.retention_seconds) || 0);
        } catch (e) {
            setError(e && e.status === 401 ? 'Sign in to see your clip history.' : 'Could not load your clip history.');
            setSources((cur) => cur || []);
        }
        setLoading(false);
    }, []);

    const loadStorage = useCallback(async () => {
        try {
            const d = await apiJson('/api/storage');
            setStorage(d && d.enabled ? d : null);
        } catch { setStorage(null); }
    }, []);

    useEffect(() => { load(0); loadStorage(); }, [load, loadStorage]);

    const storageChanged = () => {
        loadStorage();
        try { window.dispatchEvent(new Event('os:storage-changed')); } catch { /* ignore */ }
    };
    const onJobDeleted = (jobId) => {
        setSources((cur) => (cur || []).filter((x) => x.job_id !== jobId));
        setTotal((t) => Math.max(0, t - 1));
        storageChanged();
    };
    const onClipDeleted = (jobId, clipIndex, jobGone) => {
        if (jobGone) { onJobDeleted(jobId); return; }
        setSources((cur) => (cur || []).map((x) => (x.job_id === jobId
            ? { ...x, clips: (x.clips || []).filter((c) => c.index !== clipIndex) }
            : x)));
        storageChanged();
    };

    const handleRetried = (jobId) => {
        const outcome = typeof onRetried === 'function' ? onRetried(jobId) : 'attached';
        if (outcome !== 'background') setFailed((cur) => cur.filter((x) => x.job_id !== jobId));
        return outcome;
    };

    const rt = retentionText(retention);
    const keepForever = sources !== null && (!retention || retention <= 0 || !!(storage && storage.keep_history));
    const cap = storage ? Number(storage.cap_bytes) || 0 : 0;
    const pct = storage && cap > 0 ? Math.min(100, Math.max(0, (Number(storage.used_bytes) / cap) * 100)) : 0;
    const full = !!(storage && storage.full);

    return (
        <div className="animate-fade">
            <p className="eyebrow mb-1.5">07 · HISTORY</p>
            <h1 className="font-display lowercase text-2xl text-ink mb-2">Clips by source video</h1>
            <p className="text-muted text-sm mb-8 lowercase">
                Every source you processed with the clips cut from it and the instructions used.
                {keepForever
                    ? ' Videos are kept until you delete them.'
                    : (rt ? ` Clips stay here for ${rt} after they are generated.` : '')}
            </p>
            {storage && (
                <div className="mb-8 -mt-5">
                    {cap > 0 && (
                        <div className="h-1.5 rounded-full bg-paper3 overflow-hidden max-w-xs" role="progressbar" aria-valuenow={Math.round(pct)} aria-valuemin={0} aria-valuemax={100} aria-label="storage used">
                            <div className={`h-full ${full ? 'bg-danger' : 'bg-brass'}`} style={{ width: `${pct}%` }} />
                        </div>
                    )}
                    <p className={`readout mt-1 ${full ? 'text-danger' : ''}`}>
                        {cap > 0 ? `${formatBytes(storage.used_bytes)} of ${formatBytes(cap)} used` : `${formatBytes(storage.used_bytes)} used, no limit`}
                        {full ? ' · storage full' : ''}
                    </p>
                </div>
            )}

            {error && <p className="text-danger text-sm mb-4" role="alert">{error}</p>}

            {sources === null && (
                <div className="flex justify-center py-20"><Loader2 className="animate-spin text-brass" /></div>
            )}

            {sources && sources.length === 0 && !error && (
                <div className="text-center py-20 text-muted">
                    <Film size={40} className="mx-auto mb-4 text-muted" />
                    <p className="lowercase">No clips yet. Generate your first short from the Clip Generator.</p>
                </div>
            )}

            {failed.length > 0 && (
                <section className="mb-10" aria-label="needs attention">
                    <p className="eyebrow mb-2">needs attention</p>
                    <ul className="space-y-2">
                        {failed.map((f) => (
                            <FailedRow key={f.job_id} f={f} apiKey={apiKey} geminiModel={geminiModel} onRetried={handleRetried} />
                        ))}
                    </ul>
                </section>
            )}

            <div className="space-y-10">
                {(sources || []).map((s) => <SourceEntry key={s.job_id} s={s} onJobDeleted={onJobDeleted} onClipDeleted={onClipDeleted} />)}
            </div>

            {sources && sources.length > 0 && sources.length < total && (
                <div className="flex justify-center mt-8">
                    <button type="button" className="btn-ghost px-4 py-2 text-xs" disabled={loading} onClick={() => load(sources.length)}>
                        {loading ? <><Loader2 size={14} className="animate-spin" /> loading…</> : 'load more'}
                    </button>
                </div>
            )}
        </div>
    );
}
