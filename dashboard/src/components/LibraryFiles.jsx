import React, { useState, useEffect, useMemo } from 'react';
import { Download, Copy, Check, CheckCircle2, Circle, FolderOpen } from 'lucide-react';
import { apiJson } from '../lib/api';
import { getApiUrl } from '../config';
import { copyText } from '../lib/clipboard';

const fileUrl = (path, download = false) =>
    getApiUrl(`/api/library/file?path=${encodeURIComponent(path)}${download ? '&download=true' : ''}`);

const fmtSize = (b) => `${Math.max(1, Math.round((Number(b) || 0) / 1e6))} MB`;

function FileCard({ f, onPosted }) {
    const [copied, setCopied] = useState(false);
    const [busy, setBusy] = useState(false);
    const posted = !!f.posted_at;

    const copy = async () => {
        if (await copyText(f.caption)) {
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
        }
    };

    const toggle = async () => {
        setBusy(true);
        try {
            await apiJson('/api/library/posted', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: f.path, posted: !posted }),
            });
            onPosted(f.path, posted ? null : Math.floor(Date.now() / 1000));
        } catch (_) { /* leave the tick as it was */ }
        setBusy(false);
    };

    return (
        <div className={`card overflow-hidden ${posted ? 'opacity-60' : ''}`}>
            <div className="aspect-[9/16] bg-black">
                <video src={fileUrl(f.path)} controls preload="metadata" playsInline className="w-full h-full object-contain" />
            </div>
            <div className="p-3 space-y-2">
                <p className="text-sm text-ink font-medium break-words" title={f.name}>{f.name.replace(/\.[^.]+$/, '').replace(/_/g, ' ')}</p>
                <p className="readout">{fmtSize(f.size)}</p>
                {f.caption ? (
                    <p className="text-xs text-ink2 line-clamp-3 break-words" title={f.caption}>{f.caption}</p>
                ) : null}
                <div className="flex flex-wrap gap-2">
                    {f.caption ? (
                        <button type="button" onClick={copy} className="btn-ghost px-2.5 py-1.5 text-xs inline-flex items-center gap-1">
                            {copied ? <><Check size={13} /> copied</> : <><Copy size={13} /> caption</>}
                        </button>
                    ) : null}
                    <a href={fileUrl(f.path, true)} download className="btn-ghost px-2.5 py-1.5 text-xs inline-flex items-center gap-1">
                        <Download size={13} /> video
                    </a>
                </div>
                <button
                    type="button"
                    disabled={busy}
                    onClick={toggle}
                    className={`flex items-center gap-1.5 text-xs lowercase transition-colors ${posted ? 'text-ok' : 'text-muted hover:text-ink'}`}
                    title={posted ? 'Marked as posted. Click to undo.' : 'Tick once you have posted it'}
                >
                    {posted ? <CheckCircle2 size={14} /> : <Circle size={14} />}
                    {posted ? 'posted' : 'mark posted'}
                </button>
            </div>
        </div>
    );
}

// "My files": finished clips in the library folder, ready to post by hand.
// Self-host only (the API answers 404 in cloud mode and this renders nothing).
export default function LibraryFiles() {
    const [data, setData] = useState(null);

    useEffect(() => {
        apiJson('/api/library').then(setData).catch(() => setData({ available: false, files: [] }));
    }, []);

    const folders = useMemo(() => {
        const map = new Map();
        for (const f of data?.files || []) {
            const key = f.folder || '';
            if (!map.has(key)) map.set(key, []);
            map.get(key).push(f);
        }
        return [...map.entries()];
    }, [data]);

    const onPosted = (path, ts) => setData((d) => ({
        ...d, files: d.files.map((f) => (f.path === path ? { ...f, posted_at: ts } : f)),
    }));

    if (!data || !data.available || data.files.length === 0) return null;

    return (
        <section className="mb-12" aria-label="my files">
            <p className="eyebrow mb-1.5">my files</p>
            <p className="text-muted text-sm mb-6 lowercase">
                Finished clips ready to post: copy the caption, download the video, then tick it once it's up.
            </p>
            <div className="space-y-8">
                {folders.map(([folder, files]) => {
                    const left = files.filter((f) => !f.posted_at).length;
                    return (
                        <div key={folder || '.'}>
                            <div className="flex items-center gap-2 mb-3 pb-2 border-b border-rule">
                                <FolderOpen size={15} className="text-brass" />
                                <p className="text-sm text-ink font-medium">{folder || 'library'}</p>
                                <span className="readout ml-auto">{left} of {files.length} to post</span>
                            </div>
                            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4 sm:gap-5">
                                {files.map((f) => <FileCard key={f.path} f={f} onPosted={onPosted} />)}
                            </div>
                        </div>
                    );
                })}
            </div>
        </section>
    );
}
