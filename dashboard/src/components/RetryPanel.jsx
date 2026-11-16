import React from 'react';
import { Loader2, RotateCcw } from 'lucide-react';

// Why a job failed in plain words, plus a retry button. Server text is rendered
// as plain React text. `retry` null (an older job with no record) hides the button.
export default function RetryPanel({ error, fallbackText, retry, busy, onRetry, problem }) {
    const reason = (error && error.message) || fallbackText || '';
    const used = retry ? Number(retry.used) || 0 : 0;
    const limit = retry ? Number(retry.limit) || 0 : 0;
    const blocked = !!retry && retry.allowed === false;
    const canRetry = !!retry && retry.allowed !== false && typeof onRetry === 'function';
    const missingKey = !!error && error.code === 'missing_key';

    if (!reason && !retry && !problem) return null;

    return (
        <div className="w-full max-w-md text-left space-y-2">
            {reason ? <p className="text-sm text-danger break-words" role="alert">{reason}</p> : null}
            {missingKey ? (
                <p className="text-xs text-muted lowercase">add your key in settings, then retry.</p>
            ) : null}
            {blocked ? (
                <p className="text-xs text-muted break-words">
                    {retry.reason || 'This video cannot be retried.'}
                </p>
            ) : null}
            {canRetry ? (
                <button
                    type="button"
                    className="btn-primary px-4 py-2 text-sm inline-flex items-center gap-2"
                    disabled={!!busy}
                    onClick={onRetry}
                >
                    {busy
                        ? <><Loader2 size={14} className="animate-spin" /> retrying...</>
                        : <><RotateCcw size={14} /> retry</>}
                </button>
            ) : null}
            {retry && used > 0 ? (
                <p className="readout">retry {used}{limit > 0 ? ` of ${limit}` : ''} used</p>
            ) : null}
            {problem ? <p className="text-xs text-danger break-words" role="alert">{problem}</p> : null}
        </div>
    );
}
