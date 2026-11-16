// Which controls the user's instruction text already speaks to. Pure keyword
// checks, used only to show a "set by your instructions" note: the server
// decides what actually wins when the job runs.
const has = (t, re) => re.test(t);

export function briefHints(text) {
    const t = typeof text === 'string' ? text.toLowerCase() : '';
    if (!t.trim()) {
        return { hook: false, hookStyle: false, captions: false, teaser: false, clipLength: false };
    }
    const hookStyle = has(t, /hook style|\b(outline|pill)\b|yellow (card|box|hook)|red (card|box|hook)|dark (card|box|hook)/);
    return {
        hook: hookStyle || has(t, /\bhooks?\b|title card|text overlay/),
        hookStyle,
        captions: has(t, /\bcaptions?\b|subtitles?/),
        teaser: has(t, /cold open|teaser|intro replay/),
        clipLength: has(t, /\b(max|min|maximum|minimum|under|over|at least|at most|up to|no longer than|shorter than|longer than)\s*(of\s*)?\d+\s*(s|sec|secs|seconds?)\b/)
            || has(t, /\b\d+\s*(s|sec|secs|seconds?)\s*(max|min|maximum|minimum|or (less|more|shorter|longer))\b/),
    };
}
