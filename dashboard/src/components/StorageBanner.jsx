import React, { useState, useEffect, useCallback } from 'react';
import { HardDrive } from 'lucide-react';
import { apiJson } from '../lib/api';

const formatBytes = (b) => {
    const n = Number(b) || 0;
    if (n < 1024) return `${n} B`;
    const units = ['KB', 'MB', 'GB', 'TB'];
    let v = n / 1024;
    let i = 0;
    while (v >= 1024 && i < units.length - 1) { v /= 1024; i += 1; }
    return `${v >= 100 ? Math.round(v) : v.toFixed(1)} ${units[i]}`;
};

const REFRESH_MS = 5 * 60 * 1000;

// Shown when self-host storage is full. Never blocks anything: jobs still start.
export default function StorageBanner({ onManage }) {
    const [st, setSt] = useState(null);

    const load = useCallback(async () => {
        try {
            const d = await apiJson('/api/storage');
            setSt(d && d.enabled && d.full ? d : null);
        } catch {
            setSt(null);
        }
    }, []);

    useEffect(() => {
        load();
        const id = setInterval(load, REFRESH_MS);
        window.addEventListener('os:storage-changed', load);
        return () => {
            clearInterval(id);
            window.removeEventListener('os:storage-changed', load);
        };
    }, [load]);

    if (!st) return null;
    const cap = Number(st.cap_bytes) || 0;
    return (
        <div role="status" className="shrink-0 flex flex-wrap items-center gap-x-3 gap-y-2 px-4 py-2.5 border-b border-rule bg-paper3 text-sm">
            <HardDrive size={16} className="text-danger shrink-0" />
            <span className="text-danger">Storage full, remove older videos</span>
            <span className="readout">
                {formatBytes(st.used_bytes)}{cap > 0 ? ` of ${formatBytes(cap)} used` : ' used'}
            </span>
            <button type="button" className="btn-ghost px-3 py-1 text-xs ml-auto" onClick={onManage}>
                open history
            </button>
        </div>
    );
}
