"""Clip boundaries are chosen by meaning (clip_selection pieces).

The transcript is split into short phrase-level pieces; the idea pass
(idea_segments.py, tests in test_idea_segments.py) names each idea by its
first and last piece (the window path's detail pass does the same inside a
window); the context check (context_check.py, tests in
test_context_check.py) may move them; the final snap only places the cut in
the silence at the chosen edge and keeps the length band. Punctuation and pauses only decide where pieces split and break
ties: they never drag a cut the model chose, and a pause is never reported
as a sentence end.
"""
import ast
import json
import os
import re

import pytest

from clip_selection import (CUE_PENALTY, apply_piece_picks, boundary_label,
                            build_pieces, clean_about, clip_piece_span,
                            pieces_payload, resolve_piece_pick,
                            snap_clip_to_pieces, snap_clip_to_words,
                            window_pieces)


def words_from(sentences, t0=0.0, step=0.4, gap=0.0):
    """Contiguous word timings (like real ASR output), ``gap`` s between sentences."""
    out, t = [], float(t0)
    for k, sentence in enumerate(sentences):
        if k:
            t += gap
        for w in sentence.split():
            out.append({"w": w, "s": round(t, 3), "e": round(t + step, 3)})
            t += step
    return out


def marked_words(text, step=0.3, pause=1.6, stretch=0.9, t0=0.0):
    """Words from ``text`` where ``||`` is a real silence of ``pause`` seconds
    and ``~`` stretches the previous word by ``stretch`` seconds (a pause an
    aligner with contiguous timings swallows into the word)."""
    out, t = [], float(t0)
    for tok in text.split():
        if tok == "||":
            t += pause
        elif tok == "~":
            out[-1]["e"] = round(out[-1]["e"] + stretch, 3)
            t += stretch
        else:
            out.append({"w": tok, "s": round(t, 3), "e": round(t + step, 3)})
            t += step
    return out


def piece_starting(pieces, first_words):
    hits = [p for p in pieces if p["text"].startswith(first_words)]
    assert hits, f"no piece starts with {first_words!r}: {[p['text'] for p in pieces]}"
    return hits[0]


def piece_ending(pieces, last_words):
    hits = [p for p in pieces if p["text"].endswith(last_words)]
    assert hits, f"no piece ends with {last_words!r}: {[p['text'] for p in pieces]}"
    return hits[0]


# --------------------------------------------------------------------------
# build_pieces
# --------------------------------------------------------------------------

UNPUNCTUATED = (
    "so it can read and write to them ~ another thing we can do with tools is run code "
    "and that code might run in a sandbox as well we can go and get information on the "
    "internet and ~ we can also often actually drive a computer screen itself ~ this is "
    "called computer use and here the model can move a cursor around and operate any "
    "software similar to the way that a person would do it")


class TestBuildPieces:
    def test_pieces_split_at_sentence_punctuation(self):
        words = words_from(["One two three four five six.", "Seven eight nine ten eleven?",
                            "Twelve thirteen fourteen fifteen!"])
        pieces = build_pieces(words)
        assert [p["text"] for p in pieces] == [
            "One two three four five six.", "Seven eight nine ten eleven?",
            "Twelve thirteen fourteen fifteen!"]
        assert [p["index"] for p in pieces] == [0, 1, 2]
        assert [p["cue"] for p in pieces] == ["sentence", "sentence", "end"]
        assert pieces[0]["start"] == 0.0 and pieces[0]["end"] == 2.4

    def test_unpunctuated_speech_still_gets_short_pieces_at_idea_edges(self):
        # Measured failure 1: no punctuation for minutes, so sentence units
        # were 15 s long and the cut fell back to a mid-thought word.
        words = marked_words(UNPUNCTUATED, step=0.4)     # ~150 words/min, like the source
        pieces = build_pieces(words)
        assert len(pieces) >= 8
        assert all(p["end"] - p["start"] <= 5.0 + 1e-6 for p in pieces)
        # The idea's true beginnings and ends are selectable piece edges.
        piece_starting(pieces, "another thing we can do")
        piece_starting(pieces, "this is called computer use")
        piece_ending(pieces, "computer use")
        piece_ending(pieces, "on the internet")
        assert not any(p["text"].endswith("information") for p in pieces)
        assert {p["cue"] for p in pieces} <= set(CUE_PENALTY)

    def test_a_pause_inside_a_sentence_is_a_piece_edge_not_a_sentence_end(self):
        # Measured failure 2: a 1.6 s pause mid-sentence.
        words = marked_words("They are going to get the fourth because any smart team "
                             "or coach, || them last two weeks, and they look in the place.")
        pieces = build_pieces(words)
        after = piece_starting(pieces, "them last two weeks")
        before = pieces[after["index"] - 1]
        assert before["cue"] == "pause"
        assert before["cue"] != "sentence"
        # The sentence's real start is selectable too.
        assert pieces[0]["text"].startswith("They are going")

    def test_discourse_words_open_a_new_piece_once_a_phrase_is_long_enough(self):
        words = marked_words("the harness runs another process where it compacts the window "
                             "so it summarizes what happened earlier so the important parts "
                             "stay around but stuff like redundant tool outputs gets pruned")
        pieces = build_pieces(words)
        piece_ending(pieces, "stay around")
        piece_starting(pieces, "but stuff like")
        assert any(p["cue"] == "discourse" for p in pieces)

    def test_a_forced_split_prefers_a_hidden_pause_before_a_starter_word(self):
        words = marked_words("alpha beta gamma delta epsilon zeta eta theta ~ another "
                             "point is made here today with more words")
        pieces = build_pieces(words)
        piece_starting(pieces, "another point")
        assert pieces[0]["cue"] == "gap"

    def test_a_dropped_full_stop_before_a_capital_is_a_piece_edge(self):
        # Measured: "I forgot he's just going to get to What scenario is it
        # that you're actually taking LaMelo" was one piece, so the question
        # that opens the idea could not be chosen as a start.
        words = marked_words("I forgot he's just going to get to What scenario is it that "
                             "you're actually taking LaMelo if LaMelo doesn't average turnovers")
        pieces = build_pieces(words)
        piece_starting(pieces, "What scenario is it")
        assert pieces[0]["text"] == "I forgot he's just going to get to"
        # "I", proper names and short runs never split.
        words = marked_words("so I think LaMelo And Fox are close")
        assert not any(p["text"].startswith("I think") for p in build_pieces(words))

    def test_short_answers_that_end_a_sentence_stay_their_own_piece(self):
        words = words_from(["Who has the best roster right now in the West?", "Hell no, man.",
                            "Yes.", "There is a reason they got him this year."])
        pieces = build_pieces(words)
        texts = [p["text"] for p in pieces]
        assert "Hell no, man." in texts and "Yes." in texts

    def test_a_short_weak_fragment_joins_the_next_piece(self):
        words = [{"w": "and", "s": 0.0, "e": 0.2}, {"w": "um", "s": 0.2, "e": 0.4},
                 {"w": "then,", "s": 0.4, "e": 0.5}]
        words += marked_words("we go on to the next part of the story", t0=0.5)
        pieces = build_pieces(words)
        assert pieces[0]["text"].startswith("and um then, we go")

    def test_bad_input_yields_no_pieces(self):
        assert build_pieces([]) == []
        assert build_pieces(None) == []
        assert build_pieces([{"w": "x", "s": "bad", "e": 1}, {"w": "", "s": 0, "e": 1},
                             {"w": "y", "s": float("nan"), "e": 1}]) == []

    def test_accepts_whisper_word_keys(self):
        pieces = build_pieces([{"word": " Hello", "start": 0.0, "end": 0.5},
                               {"word": " there.", "start": 0.5, "end": 1.0}])
        assert pieces[0]["text"] == "Hello there."


class TestWindowPiecesAndPayload:
    def test_window_pieces_by_midpoint(self):
        pieces = [{"index": i, "start": i * 10.0, "end": i * 10.0 + 10, "text": t, "cue": "sentence"}
                  for i, t in enumerate("abc")]
        assert [p["index"] for p in window_pieces(pieces, 6, 26)] == [1, 2]
        assert clip_piece_span(10.0, 30.0, pieces) == (1, 2)
        assert clip_piece_span(100.0, 130.0, pieces) is None

    def test_payload_is_compact(self):
        pieces = [{"index": 7, "start": 12.6, "end": 15.678, "text": "Hi there.", "cue": "sentence"}]
        assert pieces_payload(pieces) == [[7, 13, "Hi there."]]


# --------------------------------------------------------------------------
# Piece pick -> seconds
# --------------------------------------------------------------------------

def _pieces(n=20, length=2.5, cue="discourse"):
    return [{"index": i, "start": i * length, "end": (i + 1) * length,
             "text": f"phrase {i}", "cue": cue} for i in range(n)]


class TestResolvePiecePick:
    def test_valid_pick_becomes_piece_edges(self):
        assert resolve_piece_pick(2, 9, _pieces()) == (5.0, 25.0)
        assert resolve_piece_pick("2", 9.0, _pieces()) == (5.0, 25.0)

    @pytest.mark.parametrize("first,last", [
        (-1, -1), (9, 2), (0, 99), (None, 3), (True, 9), ("x", 9), (2.5, 9),
        (float("nan"), 9), (float("inf"), 9)])
    def test_malformed_picks_are_refused(self, first, last):
        assert resolve_piece_pick(first, last, _pieces()) is None

    def test_pick_outside_the_shown_pieces_is_refused(self):
        assert resolve_piece_pick(2, 9, _pieces(), allowed=set(range(2, 9))) is None
        assert resolve_piece_pick(2, 8, _pieces(), allowed=set(range(2, 9))) == (5.0, 22.5)

    def test_span_far_outside_the_band_is_refused(self):
        assert resolve_piece_pick(0, 1, _pieces(), min_duration=15) is None    # 5 s
        assert resolve_piece_pick(0, 19, _pieces(), max_duration=30) is None   # 50 s
        assert resolve_piece_pick(0, 3, _pieces(), min_duration=15) == (0.0, 10.0)

    def test_no_pieces_no_pick(self):
        assert resolve_piece_pick(0, 1, []) is None


class TestApplyPiecePicks:
    def test_malformed_pick_keeps_seconds_and_never_drops_a_clip(self):
        shorts = [
            {"start": 11.0, "end": 29.0, "first_piece": 2, "last_piece": 9,
             "about": "  Why the harness\nmatters  "},
            {"start": 3.3, "end": 21.7, "first_piece": 9, "last_piece": 1},
            {"start": 4.0, "end": 24.0, "about": 42},
            {"start": 7.0, "end": 27.0, "first_piece": "junk", "last_piece": None},
        ]
        used, kept = apply_piece_picks(shorts, _pieces(), None, 15, 60)
        assert (used, kept) == (1, 3)
        assert len(shorts) == 4
        assert (shorts[0]["start"], shorts[0]["end"]) == (5.0, 25.0)
        assert shorts[0]["piece_pick"] == [2, 9]
        assert shorts[0]["about"] == "Why the harness matters"
        assert (shorts[1]["start"], shorts[1]["end"]) == (3.3, 21.7)
        assert "piece_pick" not in shorts[1] and "about" not in shorts[2]
        assert all("first_piece" not in s and "last_piece" not in s for s in shorts)

    def test_clean_about(self):
        assert clean_about("a\x00b\tc") == "a b c"
        assert clean_about(None) == ""
        assert len(clean_about("x" * 1000)) == 300


# --------------------------------------------------------------------------
# The final snap follows the chosen pieces
# --------------------------------------------------------------------------

def gapped_pieces(n=30, length=2.0, gap=0.4, cues=None):
    """Pieces with ``gap`` s of silence between them, and matching words."""
    pieces, words, t = [], [], 0.0
    for i in range(n):
        cue = (cues or {}).get(i, "discourse")
        pieces.append({"index": i, "start": round(t, 3), "end": round(t + length, 3),
                       "text": f"w{i}a w{i}b", "cue": cue})
        words.append({"w": f"w{i}a", "s": round(t, 3), "e": round(t + length / 2, 3)})
        words.append({"w": f"w{i}b", "s": round(t + length / 2, 3), "e": round(t + length, 3)})
        t += length + gap
    return pieces, words


class TestSnapClipToPieces:
    def test_a_chosen_piece_edge_is_kept_exactly_plus_silence(self):
        pieces, words = gapped_pieces()
        s, e, moved = snap_clip_to_pieces(pieces[3]["start"], pieces[10]["end"],
                                          pieces, words, 500.0, 15, 60)
        assert not moved
        assert s == pytest.approx(pieces[3]["start"] - 0.2)      # half the 0.4 s gap
        assert e == pytest.approx(pieces[10]["end"] + 0.2)

    def test_never_drags_a_chosen_cut_to_a_nearby_sentence_end(self):
        # The old unit snap moved a cut seconds away to a full stop the model
        # had not chosen. A discourse edge the model chose stays.
        cues = {9: "sentence", 11: "sentence"}
        pieces, words = gapped_pieces(cues=cues)
        s, e, moved = snap_clip_to_pieces(pieces[2]["start"], pieces[10]["end"],
                                          pieces, words, 500.0, 15, 60)
        assert not moved
        assert e == pytest.approx(pieces[10]["end"] + 0.2)

    def test_a_word_edge_inside_a_piece_is_a_choice_too(self):
        # A hook-pass lead-in trim lands on a beat, i.e. on a word.
        pieces, words = gapped_pieces()
        mid = pieces[4]["start"] + 1.0                       # w4b starts here
        s, e, moved = snap_clip_to_pieces(mid, pieces[12]["end"], pieces, words, 500.0, 15, 60)
        assert not moved
        assert s == pytest.approx(mid)                       # no silence before a mid-piece word

    def test_too_long_moves_the_cheapest_edge_and_reports_it(self):
        pieces, words = gapped_pieces(cues={14: "sentence"})
        s, e, moved = snap_clip_to_pieces(pieces[0]["start"], pieces[15]["end"],
                                          pieces, words, 500.0, 15, 30)
        assert moved
        assert 15 - 1e-6 <= e - s <= 30 + 1e-6

    def test_too_short_takes_in_speech(self):
        pieces, words = gapped_pieces()
        s, e, moved = snap_clip_to_pieces(pieces[5]["start"], pieces[8]["end"],
                                          pieces, words, 500.0, 15, 60)
        assert moved and e - s >= 15 - 1e-6
        assert s <= pieces[5]["start"] and e >= pieces[8]["end"]

    @pytest.mark.parametrize("lo,hi", [(15, 60), (20, 30), (10, 15), (40, 60)])
    def test_band_holds(self, lo, hi):
        pieces, words = gapped_pieces(n=60)
        got = snap_clip_to_pieces(pieces[10]["start"], pieces[19]["end"],
                                  pieces, words, 500.0, lo, hi)
        assert got is not None
        assert lo - 1e-6 <= got[1] - got[0] <= hi + 1e-6

    def test_the_lead_and_tail_give_way_before_an_edge_moves(self):
        pieces, words = gapped_pieces()
        # Exactly 60.0 s on piece edges: the silence would make it 60.4.
        start = pieces[0]["start"]
        end_piece = next(p for p in pieces if p["end"] - start >= 59.0)
        hi = end_piece["end"] - start
        s, e, moved = snap_clip_to_pieces(start, end_piece["end"], pieces, words, 500.0, 15, hi)
        assert not moved
        assert e - s <= hi + 1e-6

    def test_seconds_only_requests_go_to_the_nearest_edges(self):
        pieces, words = gapped_pieces()
        s, e, moved = snap_clip_to_pieces(pieces[3]["start"] + 0.7, pieces[11]["end"] - 0.6,
                                          pieces, words, 500.0, 15, 60)
        assert moved
        assert s == pytest.approx(pieces[3]["start"] - 0.2)
        assert e == pytest.approx(pieces[11]["end"] + 0.2)

    def test_the_cap_ends_on_a_sentence_not_mid_thought(self):
        # Measured: an idea 0.9 s over a 60 s max lost the end of its point
        # ("...without going | off the rails"): the cheapest cut was
        # mid-sentence. A sentence edge a few seconds earlier now wins, and
        # the opening is not shaved mid-sentence instead.
        cues = {k: "max" for k in range(40)}
        cues.update({27: "sentence", 30: "sentence"})
        pieces, words = gapped_pieces(n=40, length=2.0, gap=0.0, cues=cues)
        s, e, moved = snap_clip_to_pieces(pieces[0]["start"], pieces[30]["end"],
                                          pieces, words, 500.0, 15, 60)       # 62 s
        assert moved and s == pieces[0]["start"]
        assert e == pytest.approx(pieces[27]["end"])
        # With the overrun the check granted, the idea is kept whole.
        s, e, moved = snap_clip_to_pieces(pieces[0]["start"], pieces[30]["end"],
                                          pieces, words, 500.0, 15, 65)
        assert not moved and e == pytest.approx(pieces[30]["end"])

    def test_nothing_fits_returns_none(self):
        pieces, words = gapped_pieces(n=5)                     # 11.6 s of speech in all
        assert snap_clip_to_pieces(0.0, 8.0, pieces, words, 12.0, 15, 60) is None
        assert snap_clip_to_pieces(0.0, 8.0, [], words, 100.0) is None
        assert snap_clip_to_pieces("x", 8.0, pieces, words, 100.0) is None

    def test_bounds_clamp_to_the_source(self):
        pieces, words = gapped_pieces()
        end_of_speech = pieces[-1]["end"]
        s, e, _ = snap_clip_to_pieces(pieces[20]["start"], end_of_speech, pieces, words,
                                      end_of_speech + 0.1, 15, 60)
        assert e <= end_of_speech + 0.1 + 1e-6 and s >= 0


# --------------------------------------------------------------------------
# Truthful labels
# --------------------------------------------------------------------------

class TestBoundaryLabel:
    def test_labels_say_how_the_cut_was_decided(self):
        assert boundary_label({"piece_pick": [1, 4]}, on_pieces=False) == "word"
        assert boundary_label({"piece_pick": [1, 4]}) == "chosen"
        assert boundary_label({}) == "fallback"
        ok = {"start": "ok", "end": "ok", "moved": False}
        assert boundary_label({"piece_pick": [1, 4], "context_check": ok}) == "checked"
        fixed = {"start": "orphan_answer", "end": "ok", "moved": True}
        assert boundary_label({"context_check": fixed}) == "checked"
        stuck = {"start": "mid_thought", "end": "ok", "moved": False, "kept": "band"}
        assert boundary_label({"piece_pick": [1, 4], "context_check": stuck}) == "flagged"
        half = {"start": "ok", "end": "unknown", "moved": False}
        assert boundary_label({"piece_pick": [1, 4], "context_check": half}) == "chosen"
        new = {"start": "unresolved_reference", "end": "complete", "opening": "repaired",
               "ending": "ok", "moved": True}
        assert boundary_label({"piece_pick": [1, 4], "context_check": new}) == "checked"
        head = dict(new, start="orphan_head")
        assert boundary_label({"piece_pick": [1, 4], "context_check": head}) == "checked"
        # An edge left unresolved is flagged even without a `kept` reason.
        for bad in (dict(new, opening="unresolved"), dict(new, end="mid_sentence",
                                                          ending="unfinished")):
            assert boundary_label({"piece_pick": [1, 4], "context_check": bad}) == "flagged"

    def test_no_label_claims_a_sentence(self):
        for clip in ({}, {"piece_pick": [0, 1]},
                     {"context_check": {"start": "ok", "end": "ok"}}):
            assert boundary_label(clip) != "sentence"


# --------------------------------------------------------------------------
# Prompt and schema
# --------------------------------------------------------------------------

def _detail_template():
    mod = ast.parse(open(os.path.join(os.path.dirname(__file__), "..", "gemini_worker.py"),
                         encoding="utf-8").read())
    for node in mod.body:
        if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "DETAIL_PROMPT_TEMPLATE"):
            return node.value.value
    raise AssertionError("DETAIL_PROMPT_TEMPLATE not found")


class TestDetailPrompt:
    def test_prompt_asks_for_an_idea_by_piece_and_still_formats(self):
        text = _detail_template().format(
            video_duration=100, language="en", min_clips=2, max_clips=4,
            min_secs=15.0, max_secs=60.0, windows_json="[]")
        for needle in ("first_piece", "last_piece", '"about"', "ONE COMPLETE IDEA",
                       "MEANING", "STANDS ALONE", "skip it"):
            assert needle in text
        assert "first_unit" not in text
        assert re.findall(r"\{[a-z_]+\}", text) == []

    def test_the_idea_copy_prompt_formats_and_keeps_the_edges(self):
        gw = pytest.importorskip("gemini_worker")
        text = gw.IDEA_DETAIL_PROMPT_TEMPLATE.format(
            video_duration=100, language="en", min_clips=2, max_clips=4, ideas_json="[]")
        for needle in ("ONE complete idea", "Do not change", "source_window_id",
                       "CANDIDATE_IDEAS_JSON", "DIVERSITY", "Skip a candidate only"):
            assert needle in text
        assert re.findall(r"\{[a-z_]+\}", text) == []

    def test_schema_defaults_the_piece_pick(self):
        gw = pytest.importorskip("gemini_worker")
        clip = gw.DetailClipModel.model_validate({
            "start": 1, "end": 20, "source_window_id": "window_001", "predicted_score": 50,
            "video_description_for_tiktok": "d", "video_description_for_instagram": "d",
            "video_title_for_youtube_short": "t", "viral_hook_text": "h"})
        assert clip.first_piece == -1 and clip.last_piece == -1 and clip.about == ""


# --------------------------------------------------------------------------
# get_viral_clips end to end (canned model)
# --------------------------------------------------------------------------

DIALOGUE = [
    "Okay let me ask you straight, which team in the West has the best roster right now?",  # Q
    "Hell no, man.",                                                                       # orphan
    "There is a reason they got the big man with the first pick of the draft.",
    "They have a solid bench, they have solid guards and a coach who knows the system.",
    "If they stay healthy they win the conference and nobody is even close to them.",
    "That is the whole argument and I will stand by it all season long.",
    "Now let us move on to the East where things look very different this year.",
    "The top three teams are separated by a couple of games and that is about it.",
]


def _dialogue_transcript():
    words = words_from(["Welcome back to the show everybody, good to have you here."]
                       + DIALOGUE * 4, t0=0.0, gap=0.3)
    segments = []
    for k in range(0, len(words), 20):
        chunk = words[k:k + 20]
        segments.append({"start": chunk[0]["s"], "end": chunk[-1]["e"],
                         "text": " ".join(w["w"] for w in chunk),
                         "words": [{"word": w["w"], "start": w["s"], "end": w["e"]}
                                   for w in chunk]})
    return {"language": "en", "segments": segments}, words


def _between_tags(prompt, tag="clip_transcript"):
    m = re.search(r'<%s id="([0-9a-f]+)">\n(.*?)\n</%s id="\1">' % (tag, tag), prompt, re.S)
    return [json.loads(line) for line in m.group(2).splitlines() if line.strip()]


COST = {"input_tokens": 10, "output_tokens": 5, "total_cost": 0.001}
BASE_COPY = {"predicted_score": 70, "video_description_for_tiktok": "d",
             "video_description_for_instagram": "d", "video_title_for_youtube_short": "t",
             "viral_hook_text": "h"}


def _hook_answer(prompt, lead=1):
    return {"clips": [{"id": c["id"], "opening": "slow", "lead_in_beats": lead,
                       "teaser_beat": -1, "hook_type": "bold_claim",
                       "hook_text": "Best roster in the West", "hook_score": 60,
                       "opening_score": 60} for c in _between_tags(prompt)]}


def _ctx_answer(prompt, overrides=None):
    """Complete endings; an opening on "Hell no" names its question."""
    out = []
    for c in _between_tags(prompt):
        entry = {"id": c["id"], "unresolved": [], "idea_starts_at": 0,
                 "end_verdict": "complete", "lands_at": c["last_clip_piece"],
                 "last_complete_piece": c["last_clip_piece"]}
        if c["clip_start"][0][1].startswith("Hell no"):
            q = [k for k, t in c["before"] if t.startswith("Okay let me ask")]
            entry["unresolved"] = [{"phrase": "Hell no, man.", "kind": "answer_to_unseen_question",
                                    "introduced_at": q[-1]}]
        if overrides:
            entry.update(overrides(c) or {})
        out.append(entry)
    return {"clips": out}


def _idea_answer(prompt):
    """Segment each round of DIALOGUE into the West idea (opening on the
    orphan answer, as the measured failures did) and the East idea."""
    rows = _between_tags(prompt)
    ideas = []
    hells = [r[0] for r in rows if r[2].startswith("Hell no")]
    for n, h in enumerate(hells):
        close = next(r[0] for r in rows if r[0] > h and r[2].startswith("That is the whole"))
        while not next(r for r in rows if r[0] == close)[2].endswith("long."):
            close += 1
        east = next(r[0] for r in rows if r[0] > close and r[2].startswith("Now let us"))
        end_east = next(r[0] for r in rows if r[0] > east and r[2].endswith("about it."))
        ideas.append({"about": "Why one team has the best roster in the West",
                      "first_piece": h, "last_piece": close, "score": 80 - 2 * n, "reason": "r"})
        ideas.append({"about": "The East is close", "first_piece": east,
                      "last_piece": end_east, "score": 30 - 2 * n, "reason": "r"})
    return {"ideas": ideas}


def _detail_answer(prompt, take=None):
    items = json.loads(prompt.split("CANDIDATE_IDEAS_JSON:")[-1].split("Return only:")[0])
    items = items if take is None else items[:take]
    return {"shorts": [dict(BASE_COPY, source_window_id=i["id"], start=0, end=0,
                            first_piece=-1, last_piece=-1, about="copy about")
                       for i in items]}


def _drive(monkeypatch, handlers, env=None):
    """get_viral_clips on the dialogue transcript with canned stages.
    ``handlers``: {schema name: fn(prompt, model) -> parsed}. Returns
    (out, calls, words) with calls = [(schema name, model)]."""
    main = pytest.importorskip("main")
    for k in ("IDEA_PAUSE_SECONDS", "CONTEXT_PAUSE_SECONDS", "HOOK_PAUSE_SECONDS"):
        monkeypatch.setenv(k, "0")
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    transcript, words = _dialogue_transcript()
    calls = []

    def fake_stage(client, model, prompt, schema):
        name = schema.__name__
        calls.append((name, model))
        if name not in handlers:
            raise AssertionError(f"unexpected schema {name}")
        return handlers[name](prompt, model), dict(COST, model=model)

    monkeypatch.setattr(main, "_run_gemini_stage", fake_stage)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(main.genai, "Client", lambda *a, **k: object())
    out = main.get_viral_clips(transcript, words[-1]["e"] + 1)
    return out, calls, words


IDEA_HANDLERS = {
    "IdeaResponse": lambda p, m: _idea_answer(p),
    "ContextAnswer": lambda p, m: _ctx_answer(p),
    "DetailResponse": lambda p, m: _detail_answer(p),
    "HookAnswer": lambda p, m: _hook_answer(p),
}


def test_get_viral_clips_by_ideas_moves_an_orphan_answer_back_to_its_question(monkeypatch):
    out, calls, words = _drive(monkeypatch, IDEA_HANDLERS)
    clips = out["shorts"]
    names = [c[0] for c in calls]
    # Ideas first, the check BEFORE the copy, the hook pass after; the check
    # runs once (every candidate in one batch) and not again after the hooks.
    assert names.index("IdeaResponse") < names.index("ContextAnswer") < \
        names.index("DetailResponse") < names.index("HookAnswer")
    assert names.count("ContextAnswer") == 1 and "ScoreResponse" not in names
    # Idea pass, check and copy count toward the job's cost (the hook pass
    # runs through _brief_runner, which does not report costs).
    counted = [n for n in names if n != "HookAnswer"]
    assert out["cost_analysis"]["total_cost"] == pytest.approx(0.001 * len(counted))
    assert "boundary_model" not in out["cost_analysis"]

    pieces = build_pieces(words)
    west = [c for c in clips if c.get("about", "").startswith("Why one team")]
    assert west, "the orphan-answer ideas must survive as clips"
    for c in west:
        rec = c["context_check"]
        assert rec["opening"] == "repaired" and rec["moved"] is True
        assert rec["refs"][0]["kind"] == "answer_to_unseen_question"
        q = [p for p in pieces if p["text"].startswith("Okay let me ask")
             and p["start"] <= c["proposed"][0] + 0.01]
        assert c["proposed"][0] == pytest.approx(q[-1]["start"])
        assert c["start"] == pytest.approx(q[-1]["start"] - 0.15)    # half the 0.3 s gap
        assert c["boundary"] == "checked" and c["idea_id"].startswith("idea_")
        # The checked opening is locked: the hook pass asked to trim a beat.
        assert "lead_in_trimmed" not in c and c["opening_checked"] is True
        assert 15 <= c["end"] - c["start"] <= 60
    for c in clips:
        assert "first_piece" not in c and "proposed" in c
        assert c["boundary"] in ("checked", "flagged", "chosen", "fallback", "word")


def test_a_failed_idea_pass_falls_back_to_the_window_pipeline(monkeypatch):
    def detail(prompt, model):
        w = json.loads(prompt.split("CANDIDATE_WINDOWS_JSON:")[-1].split("Return only:")[0])[0]
        pcs = w["pieces"]
        hell = next(k for k, p in enumerate(pcs) if p[2].startswith("Hell no"))
        last = next(k for k, p in enumerate(pcs) if k > hell and p[2].endswith("long."))
        good = dict(BASE_COPY, source_window_id=w["id"], about="Why one team has the best roster",
                    first_piece=pcs[hell][0], last_piece=pcs[last][0], start=0, end=0)
        later = pcs[-8][1] + 0.7
        broken = dict(BASE_COPY, source_window_id=w["id"], first_piece=10 ** 6, last_piece=-3,
                      start=later, end=later + 20)
        return {"shorts": [good, broken]}

    def score(prompt, model):
        payload = json.loads(prompt.split("WINDOWS_JSON:")[-1].split("Return only:")[0])
        return {"windows": [{"id": w["id"], "start": w["start"], "end": w["end"],
                             "score": 90 - n, "reason": "r"} for n, w in enumerate(payload)]}

    details = []

    def detail_once(prompt, model):
        details.append(1)
        return detail(prompt, model) if len(details) == 1 else {"shorts": []}

    def boom(prompt, model):
        raise RuntimeError("model down")

    handlers = {"IdeaResponse": boom, "ScoreResponse": score, "DetailResponse": detail_once,
                "ContextAnswer": lambda p, m: _ctx_answer(p),
                "HookAnswer": lambda p, m: _hook_answer(p, lead=0)}
    out, calls, words = _drive(monkeypatch, handlers)
    names = [c[0] for c in calls]
    clips = out["shorts"]
    assert len(clips) == 2, "a broken pick must not drop its clip"
    assert "ScoreResponse" in names
    # Window path: the check runs after the hook pass, once for all clips.
    assert names.index("ContextAnswer") > names.index("HookAnswer")
    assert names.count("ContextAnswer") == 1
    fixed = next(c for c in clips if c.get("piece_pick"))
    assert fixed["context_check"]["opening"] == "repaired"
    assert fixed["boundary"] == "checked"
    broken = next(c for c in clips if not c.get("piece_pick"))
    assert broken["context_check"]["moved"] is False


def test_the_idea_pass_can_be_turned_off(monkeypatch):
    handlers = dict(IDEA_HANDLERS)
    handlers["ScoreResponse"] = lambda p, m: {"windows": []}
    handlers["DetailResponse"] = lambda p, m: {"shorts": []}
    out, calls, _ = _drive(monkeypatch, handlers, env={"IDEA_PASS": "0"})
    assert "IdeaResponse" not in [c[0] for c in calls]
    assert "ScoreResponse" in [c[0] for c in calls]


def test_the_boundary_model_runs_only_the_boundary_stages(monkeypatch, capsys):
    out, calls, _ = _drive(monkeypatch, IDEA_HANDLERS,
                           env={"GEMINI_MODEL": "gemini-3.1-flash-lite",
                                "GEMINI_BOUNDARY_MODEL": "gemini-3.5-flash"})
    by_stage = {}
    for name, model in calls:
        by_stage.setdefault(name, set()).add(model)
    assert by_stage["IdeaResponse"] == {"gemini-3.5-flash"}
    assert by_stage["ContextAnswer"] == {"gemini-3.5-flash"}
    assert by_stage["DetailResponse"] == {"gemini-3.1-flash-lite"}
    assert by_stage["HookAnswer"] == {"gemini-3.1-flash-lite"}
    cost = out["cost_analysis"]
    assert cost["model"] == "gemini-3.1-flash-lite"
    assert cost["boundary_model"] == "gemini-3.5-flash"
    n_boundary = sum(1 for n, m in calls if m == "gemini-3.5-flash")
    assert cost["boundary_cost"] == pytest.approx(0.001 * n_boundary)
    log = capsys.readouterr().out
    assert "boundaries: gemini-3.5-flash" in log and "boundaries on gemini-3.5-flash" in log


def test_a_boundary_model_off_the_list_is_ignored(monkeypatch, capsys):
    _, calls, _ = _drive(monkeypatch, IDEA_HANDLERS,
                         env={"GEMINI_MODEL": "gemini-3.1-flash-lite",
                              "GEMINI_BOUNDARY_MODEL": "gemini-9-ultra"})
    assert {m for _, m in calls} == {"gemini-3.1-flash-lite"}
    log = capsys.readouterr().out
    assert "not an allowed model" in log and "gemini-9-ultra" not in log


def test_on_a_local_llm_the_override_does_not_apply(monkeypatch, capsys):
    main = pytest.importorskip("main")
    monkeypatch.setattr(main.llm_backend, "active", lambda: True)
    monkeypatch.setattr(main.llm_backend, "model_name", lambda: "qwen2.5:14b")
    monkeypatch.setattr(main.llm_backend, "base_url", lambda: "http://llm.invalid/v1")
    out, calls, _ = _drive(monkeypatch, IDEA_HANDLERS,
                           env={"GEMINI_BOUNDARY_MODEL": "gemini-3.5-flash"})
    assert {m for _, m in calls} == {"qwen2.5:14b"}
    assert out["shorts"]
    assert "does not apply to a local LLM" in capsys.readouterr().out
    # Local batches: the check takes 2 clips per call, the copy 4 ideas.
    names = [c[0] for c in calls]
    assert names.count("ContextAnswer") >= 2 and names.count("DetailResponse") >= 2


def test_the_clip_floor_asks_the_skipped_and_spare_ideas(monkeypatch):
    seen = []

    def detail(prompt, model):
        items = json.loads(prompt.split("CANDIDATE_IDEAS_JSON:")[-1].split("Return only:")[0])
        seen.append([i["id"] for i in items])
        return _detail_answer(prompt, take=2 if len(seen) == 1 else None)

    handlers = dict(IDEA_HANDLERS, DetailResponse=detail)
    out, calls, _ = _drive(monkeypatch, handlers)
    assert len(seen) == 2, "the floor never triggered a second call"
    assert not set(seen[0][:2]) & set(seen[1])          # only ideas not yet used
    assert len(out["shorts"]) >= 4


def _span_idea_answer(target_lo, target_hi):
    """One idea per stretch whose span lies in (target_lo, target_hi) s."""
    def run(prompt, model):
        rows = _between_tags(prompt)
        starts = {r[0]: r[1] for r in rows}
        q = next(r[0] for r in rows if r[2].startswith("Okay let me ask"))
        ideas = []
        for last in range(q + 1, rows[-1][0]):
            # pieces end where the next starts (+ a 0.3 s gap at most)
            span = starts.get(last + 1, 0) - starts[q]
            if target_lo < span < target_hi:
                ideas.append({"about": "The West, the East and the whole argument",
                              "first_piece": q, "last_piece": last, "score": 90, "reason": "r"})
                break
        # Filler ideas so the floor of 4 is reachable.
        tail = [r[0] for r in rows if r[1] > starts[q] + target_hi + 2]
        for k in range(0, min(len(tail) - 6, 40), 6):
            ideas.append({"about": "filler", "first_piece": tail[k], "last_piece": tail[k + 5],
                          "score": 10, "reason": "r"})
        return {"ideas": ideas}
    return run


@pytest.mark.parametrize("explicit", [False, True])
def test_the_cap_never_cuts_a_finished_idea_short_unless_the_max_is_explicit(monkeypatch,
                                                                              explicit):
    handlers = dict(IDEA_HANDLERS, IdeaResponse=_span_idea_answer(61.5, 64.5))
    env = {"CLIP_MAX_SECONDS": "60"} if explicit else {}
    if not explicit:
        monkeypatch.delenv("CLIP_MAX_SECONDS", raising=False)
    out, _, _ = _drive(monkeypatch, handlers, env=env)
    (big,) = [c for c in out["shorts"] if c.get("about", "").startswith("The West, the East")]
    length = big["end"] - big["start"]
    if explicit:
        assert length <= 60 + 1e-6                       # the user's max is exact
        assert big["context_check"].get("over_max") is None
    else:
        # Default max: the idea keeps its landing a few seconds past 60.
        assert 60 < length <= 65 + 1e-6
        assert big["context_check"]["over_max"] > 0 and big.get("band_fit") is None


def test_viewer_moments_reach_the_idea_pass_and_promote_their_idea(monkeypatch):
    """Twitch: the stretches viewers clipped are shown to the idea pass and
    the hottest idea is guaranteed a shortlist slot, even a low-scored one."""
    main = pytest.importorskip("main")
    transcript, words = _dialogue_transcript()
    pieces = build_pieces(words)
    east = [p for p in pieces if p["text"].startswith("Now let us")][-1]   # lowest idea
    viewer = [{"start": east["start"] + 0.5, "end": east["start"] + 6.0, "views": 500}]
    prompts = {}

    def remember(name, fn):
        def run(prompt, model):
            prompts.setdefault(name, []).append(prompt)
            return fn(prompt, model)
        return run

    handlers = {k: remember(k, v) for k, v in IDEA_HANDLERS.items()}
    for k, v in {"IDEA_PAUSE_SECONDS": "0", "CONTEXT_PAUSE_SECONDS": "0",
                 "HOOK_PAUSE_SECONDS": "0", "CLIP_TARGET_MIN": "2",
                 "CLIP_TARGET_MAX": "2"}.items():
        monkeypatch.setenv(k, v)

    def fake_stage(client, model, prompt, schema):
        return handlers[schema.__name__](prompt, model), dict(COST, model=model)

    monkeypatch.setattr(main, "_run_gemini_stage", fake_stage)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(main.genai, "Client", lambda *a, **k: object())
    out = main.get_viral_clips(transcript, words[-1]["e"] + 1, viewer_clips=viewer)
    assert "VIEWER_MOMENTS" in prompts["IdeaResponse"][0]
    copy = json.loads(prompts["DetailResponse"][0].split("CANDIDATE_IDEAS_JSON:")[-1]
                      .split("Return only:")[0])
    hot = [i for i in copy if i.get("viewer_moments")]
    assert hot and hot[0]["start"] <= east["start"] + 0.01 <= hot[0]["end"]
    assert any(c.get("idea_id") == hot[0]["id"] for c in out["shorts"])
