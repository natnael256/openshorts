"""Read-only "My files" library for self-host, built for posting by hand.

Lists video files under LIBRARY_DIR (default ``library/``), one folder level
deep, so finished clips made outside the app can be played, downloaded and
posted from the History tab. A caption written next to a video as
``<video>.txt`` is offered as a one-click copy. The only state kept is which
files the user ticked as posted, in a small JSON file outside the library
(the library itself may be mounted read-only).

Paths from the client are relative names; every one is resolved and must stay
inside the library root (no ``..``, no absolute paths, no symlinks out).
"""

import json
import os
import threading
import time
from typing import Dict, List

from jsonfile import write_json_atomic

VIDEO_EXTS = (".mp4", ".mov", ".webm", ".m4v")
MAX_FILES = 500
MAX_CAPTION = 4000

_state_lock = threading.Lock()


def root() -> str:
    return os.path.realpath(os.environ.get("LIBRARY_DIR") or "library")


def _state_path() -> str:
    return os.environ.get("LIBRARY_STATE_PATH") or os.path.join("data", "library_state.json")


def resolve(rel: str) -> str:
    """Absolute path of library video ``rel``; ValueError when it is not one."""
    if not isinstance(rel, str) or not rel or "\x00" in rel or len(rel) > 512:
        raise ValueError("bad path")
    if os.path.isabs(rel) or rel.replace("\\", "/").startswith("/"):
        raise ValueError("bad path")
    base = root()
    full = os.path.realpath(os.path.join(base, rel))
    if full == base or os.path.commonpath([full, base]) != base:
        raise ValueError("outside library")
    if not full.lower().endswith(VIDEO_EXTS) or not os.path.isfile(full):
        raise ValueError("not a video")
    return full


def _caption(full: str) -> str:
    txt = os.path.splitext(full)[0] + ".txt"
    try:
        with open(txt, "r", encoding="utf-8") as f:
            return f.read(MAX_CAPTION).strip()
    except (OSError, UnicodeDecodeError):
        return ""


def _read_state() -> Dict:
    try:
        with open(_state_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
        posted = data.get("posted") if isinstance(data, dict) else None
        return {"posted": posted if isinstance(posted, dict) else {}}
    except (OSError, ValueError):
        return {"posted": {}}


def set_posted(rel: str, posted: bool) -> None:
    resolve(rel)  # only real library videos get an entry
    with _state_lock:
        state = _read_state()
        if posted:
            state["posted"][rel] = int(time.time())
        else:
            state["posted"].pop(rel, None)
        path = _state_path()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        write_json_atomic(path, state, indent=2)


def _entry(base: str, full: str, posted: Dict) -> Dict:
    st = os.stat(full)
    rel = os.path.relpath(full, base).replace(os.sep, "/")
    return {"path": rel, "name": os.path.basename(full), "folder": os.path.dirname(rel),
            "size": st.st_size, "modified": int(st.st_mtime),
            "caption": _caption(full), "posted_at": posted.get(rel)}


def _videos_in(directory: str) -> List[str]:
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return []
    out = []
    for name in names:
        full = os.path.join(directory, name)
        if (not name.startswith(".") and name.lower().endswith(VIDEO_EXTS)
                and not os.path.islink(full) and os.path.isfile(full)):
            out.append(full)
    return out


def listing() -> Dict:
    """Videos in the library root and in each of its direct sub-folders."""
    base = root()
    if not os.path.isdir(base):
        return {"available": False, "files": []}
    posted = _read_state()["posted"]
    dirs = [base]
    try:
        dirs += [os.path.join(base, d) for d in sorted(os.listdir(base))
                 if not d.startswith(".") and os.path.isdir(os.path.join(base, d))]
    except OSError:
        pass
    files: List[Dict] = []
    for d in dirs:
        for full in _videos_in(d):
            try:
                files.append(_entry(base, full, posted))
            except OSError:
                continue
            if len(files) >= MAX_FILES:
                return {"available": True, "files": files, "truncated": True}
    return {"available": True, "files": files}
