import os

import pytest

import media_library


@pytest.fixture
def lib(tmp_path, monkeypatch):
    root = tmp_path / "library"
    (root / "Channel").mkdir(parents=True)
    (root / "Channel" / "deep").mkdir()
    (root / "Channel" / "01_clip.mp4").write_bytes(b"x" * 10)
    (root / "Channel" / "01_clip.txt").write_text("Caption one #tag", encoding="utf-8")
    (root / "Channel" / "notes.md").write_text("not a video", encoding="utf-8")
    (root / "Channel" / "deep" / "hidden.mp4").write_bytes(b"x")
    (root / "top.mov").write_bytes(b"y")
    (tmp_path / "secret.mp4").write_bytes(b"s")
    monkeypatch.setenv("LIBRARY_DIR", str(root))
    monkeypatch.setenv("LIBRARY_STATE_PATH", str(tmp_path / "state" / "library_state.json"))
    return root


def test_listing_one_level_with_caption(lib):
    out = media_library.listing()
    paths = {f["path"]: f for f in out["files"]}
    assert set(paths) == {"top.mov", "Channel/01_clip.mp4"}
    assert paths["Channel/01_clip.mp4"]["caption"] == "Caption one #tag"
    assert paths["Channel/01_clip.mp4"]["folder"] == "Channel"
    assert paths["top.mov"]["caption"] == ""


def test_missing_library_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setenv("LIBRARY_DIR", str(tmp_path / "nope"))
    assert media_library.listing() == {"available": False, "files": []}


@pytest.mark.parametrize("bad", [
    "", "../secret.mp4", "Channel/../../secret.mp4", "/etc/passwd",
    "Channel/notes.md", "Channel", "Channel/missing.mp4", "a\x00b.mp4",
])
def test_resolve_refuses_outside_or_non_video(lib, bad):
    with pytest.raises(ValueError):
        media_library.resolve(bad)


def test_resolve_refuses_symlink_out(lib, tmp_path):
    link = lib / "Channel" / "link.mp4"
    try:
        os.symlink(tmp_path / "secret.mp4", link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not available")
    with pytest.raises(ValueError):
        media_library.resolve("Channel/link.mp4")


def test_posted_tick_round_trip(lib):
    media_library.set_posted("Channel/01_clip.mp4", True)
    f = {x["path"]: x for x in media_library.listing()["files"]}["Channel/01_clip.mp4"]
    assert isinstance(f["posted_at"], int)
    media_library.set_posted("Channel/01_clip.mp4", False)
    f = {x["path"]: x for x in media_library.listing()["files"]}["Channel/01_clip.mp4"]
    assert f["posted_at"] is None


def test_posted_refuses_unknown_file(lib):
    with pytest.raises(ValueError):
        media_library.set_posted("../secret.mp4", True)
