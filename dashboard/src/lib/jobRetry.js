// Retry a failed clip job. The server re-runs the same video with the same
// settings and creator instructions from its own records, so nothing about them
// travels from here: no body, only the key/model headers /api/process also takes.
import { apiFetch } from './api';

// Job ids with a retry request in flight: a second call for the same job while
// one is pending is rejected here, so one job never gets two requests.
const inFlight = new Set();

// FastAPI's `detail` is a string or an object {code, message}.
function parseDetail(body) {
  const d = body && typeof body === 'object' ? body.detail : null;
  if (typeof d === 'string' && d) return { message: d, code: null };
  if (d && typeof d === 'object') {
    return {
      message: typeof d.message === 'string' ? d.message : '',
      code: typeof d.code === 'string' ? d.code : null,
    };
  }
  return { message: '', code: null };
}

export async function retryJob(jobId, { apiKey, geminiModel } = {}) {
  if (!jobId) throw Object.assign(new Error('No job to retry.'), { code: 'no_job', status: 0 });
  if (inFlight.has(jobId)) {
    throw Object.assign(new Error('A retry for this video is already being sent.'), { code: 'in_flight', status: 0 });
  }
  inFlight.add(jobId);
  try {
    const headers = {};
    if (apiKey) headers['X-Gemini-Key'] = apiKey;
    if (geminiModel) headers['X-Gemini-Model'] = geminiModel;
    // apiFetch throws QuotaError on a 402: let it through untouched.
    const res = await apiFetch(`/api/jobs/${encodeURIComponent(jobId)}/retry`, { method: 'POST', headers });
    if (!res.ok) {
      let body = null;
      try { body = await res.json(); } catch { /* not JSON */ }
      const { message, code } = parseDetail(body);
      const fallback = res.status === 429
        ? 'The retry limit for this video has been reached.'
        : 'Could not retry this video.';
      const err = new Error((message || fallback).slice(0, 300));
      err.status = res.status;
      err.code = code;
      throw err;
    }
    return await res.json();
  } finally {
    inFlight.delete(jobId);
  }
}
