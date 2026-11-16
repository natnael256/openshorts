import React, { useState, useEffect, useCallback } from 'react';
import { Loader2 } from 'lucide-react';
import { apiJson } from '../lib/api';

const DRAFT_KEY = 'os_brief_draft';
const DEFAULT_LIMITS = { brief: 1500, name: 60 };

const errText = (e, fallback) => (e && typeof e.detail === 'string' && e.detail ? e.detail : fallback);

// Creator instructions ("brief") for a run, plus saved presets. The parent owns
// the text and the selected preset id; editing the text never clears the id
// (the parent compares text to the preset to decide what to send).
export default function BriefField({ value, onChange, presetId, onPresetChange, onPresetText, disabled }) {
    const [presets, setPresets] = useState(null); // null = loading
    const [unavailable, setUnavailable] = useState(false);
    const [limits, setLimits] = useState(DEFAULT_LIMITS);
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState('');
    const [naming, setNaming] = useState(false);
    const [newName, setNewName] = useState('');
    const [confirmDelete, setConfirmDelete] = useState(false);

    useEffect(() => {
        let cancelled = false;
        apiJson('/api/brief-presets')
            .then((d) => {
                if (cancelled) return;
                setPresets(Array.isArray(d.presets) ? d.presets : []);
                setLimits({
                    brief: Number(d.max_brief_chars) || DEFAULT_LIMITS.brief,
                    name: Number(d.max_name_chars) || DEFAULT_LIMITS.name,
                });
            })
            .catch(() => { if (!cancelled) { setUnavailable(true); setPresets([]); } });
        return () => { cancelled = true; };
    }, []);

    const selected = presets && presetId ? presets.find((p) => p.id === presetId) : null;
    // A preset id that no longer exists (deleted elsewhere) is dropped.
    useEffect(() => {
        if (presets && presetId && !selected) onPresetChange(null);
    }, [presets, presetId, selected, onPresetChange]);

    // Tell the parent the selected preset's saved text (for the unmodified check).
    const savedText = selected ? selected.brief : null;
    useEffect(() => { if (onPresetText) onPresetText(savedText); }, [savedText, onPresetText]);

    const modified = !!selected && selected.brief !== value;

    const edit = useCallback((text) => {
        onChange(text);
        try { localStorage.setItem(DRAFT_KEY, text); } catch { /* ignore */ }
    }, [onChange]);

    const choose = (id) => {
        setError('');
        setConfirmDelete(false);
        if (!id) { onPresetChange(null); return; }
        const p = presets.find((x) => x.id === id);
        if (!p) return;
        onPresetChange(p.id);
        edit(p.brief || '');
    };

    const run = async (fn) => {
        setBusy(true);
        setError('');
        try { await fn(); } catch (e) { setError(errText(e, 'Something went wrong. Please try again.')); }
        setBusy(false);
    };

    const savePreset = () => run(async () => {
        const name = newName.trim();
        if (!name) { setError('Give the preset a name.'); return; }
        const p = await apiJson('/api/brief-presets', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, brief: value.trim() }),
        });
        setPresets((cur) => [...(cur || []), p]);
        onPresetChange(p.id);
        setNaming(false);
        setNewName('');
    });

    const updatePreset = () => run(async () => {
        const p = await apiJson(`/api/brief-presets/${encodeURIComponent(selected.id)}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ brief: value.trim() }),
        });
        setPresets((cur) => (cur || []).map((x) => (x.id === p.id ? p : x)));
    });

    const deletePreset = () => run(async () => {
        const id = selected.id;
        await apiJson(`/api/brief-presets/${encodeURIComponent(id)}`, { method: 'DELETE' });
        setPresets((cur) => (cur || []).filter((x) => x.id !== id));
        onPresetChange(null);
        setConfirmDelete(false);
    });

    const linkBtn = 'text-xs lowercase text-muted hover:text-brass underline underline-offset-2 transition-colors disabled:opacity-45 disabled:cursor-not-allowed disabled:no-underline';
    const locked = disabled || busy;

    return (
        <div className="mt-5" data-tutorial="brief">
            <div className="flex items-baseline justify-between gap-3 mb-2">
                <label htmlFor="brief-text" className="eyebrow">
                    creator instructions <span className="text-muted normal-case">(optional)</span>
                </label>
                {value && (
                    <button type="button" onClick={() => { edit(''); onPresetChange(null); setConfirmDelete(false); }} disabled={disabled} className={linkBtn}>
                        clear
                    </button>
                )}
            </div>
            <textarea
                id="brief-text"
                value={value}
                onChange={(e) => edit(e.target.value)}
                maxLength={limits.brief}
                rows={3}
                disabled={disabled}
                placeholder="no profanity · hook in the first 3 seconds · only funny moments · max 30s · yellow captions"
                className="input-field resize-y min-h-[5rem]"
            />
            <div className="flex items-start justify-between gap-3 mt-1.5">
                <p className="text-[11px] leading-relaxed text-muted">
                    Instructions steer which moments are picked, the hook and caption look, and each clip is checked against them afterwards.
                </p>
                <span className="readout shrink-0">{value.length}/{limits.brief}</span>
            </div>

            <div className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-2">
                {presets === null ? (
                    <Loader2 size={14} className="animate-spin text-muted" aria-label="loading presets" />
                ) : unavailable ? (
                    <span className="text-xs text-muted lowercase">presets unavailable</span>
                ) : (
                    <>
                        <select
                            value={selected ? selected.id : ''}
                            onChange={(e) => choose(e.target.value)}
                            disabled={locked}
                            aria-label="saved instruction presets"
                            className="input-field !w-auto max-w-full text-xs py-1.5"
                        >
                            <option value="">no preset</option>
                            {presets.map((p) => (
                                <option key={p.id} value={p.id}>{p.name}</option>
                            ))}
                        </select>
                        {modified && <span className="text-xs text-brass lowercase">modified</span>}
                        {!naming && (
                            <button type="button" className={linkBtn} disabled={locked || !value.trim()} onClick={() => { setNaming(true); setError(''); setConfirmDelete(false); }}>
                                save as preset
                            </button>
                        )}
                        {selected && modified && (
                            <button type="button" className={linkBtn} disabled={locked || !value.trim()} onClick={updatePreset}>
                                update preset
                            </button>
                        )}
                        {selected && !confirmDelete && (
                            <button type="button" className={linkBtn} disabled={locked} onClick={() => { setConfirmDelete(true); setNaming(false); setError(''); }}>
                                delete
                            </button>
                        )}
                        {selected && confirmDelete && (
                            <span className="flex items-center gap-2 text-xs lowercase text-ink2">
                                delete &ldquo;<span className="truncate max-w-[10rem]">{selected.name}</span>&rdquo;?
                                <button type="button" className="text-danger underline underline-offset-2 disabled:opacity-45" disabled={locked} onClick={deletePreset}>yes, delete</button>
                                <button type="button" className={linkBtn} disabled={busy} onClick={() => setConfirmDelete(false)}>cancel</button>
                            </span>
                        )}
                    </>
                )}
                {busy && <Loader2 size={14} className="animate-spin text-muted" aria-label="working" />}
            </div>

            {naming && !unavailable && (
                <div className="mt-2 flex flex-wrap items-center gap-2">
                    <input
                        type="text"
                        value={newName}
                        onChange={(e) => setNewName(e.target.value)}
                        onKeyDown={(e) => {
                            if (e.key === 'Enter') { e.preventDefault(); if (!locked) savePreset(); }
                            if (e.key === 'Escape') { setNaming(false); setNewName(''); }
                        }}
                        maxLength={limits.name}
                        placeholder="preset name"
                        aria-label="preset name"
                        className="input-field !w-auto flex-1 min-w-[10rem] text-xs py-1.5"
                        autoFocus
                    />
                    <button type="button" className="btn-ghost px-3 py-1.5 text-xs" disabled={locked || !newName.trim()} onClick={savePreset}>save</button>
                    <button type="button" className={linkBtn} disabled={busy} onClick={() => { setNaming(false); setNewName(''); setError(''); }}>cancel</button>
                </div>
            )}

            {error && <p className="text-danger text-xs mt-2" role="alert">{error}</p>}
        </div>
    );
}
