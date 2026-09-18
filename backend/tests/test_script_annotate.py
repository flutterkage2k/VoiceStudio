"""The guard that keeps script annotation from rewriting the manuscript.

Annotation asks an LLM to insert tags into the user's book. The failure that
would actually cost them something is not a bad pause — it is a silently
reworded or truncated paragraph discovered after the render. So every response
is verified by stripping the insertable tags and comparing what is left to what
went in, and these tests pin that check.

No CLI and no model: `annotate_chunk` is driven through a stubbed
`subprocess.run`, so the whole contract is exercised offline.
"""
from __future__ import annotations

import subprocess

import pytest

from services import script_annotate as sa


def _stub_run(monkeypatch, stdout="", returncode=0, exc=None):
    def fake(cmd, **kw):
        if exc:
            raise exc
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr="boom")
    monkeypatch.setattr(sa.subprocess, "run", fake)


SRC = "本を読む。しかし、それは間違いです。"


# ── strip_tags: only OUR tags disappear ─────────────────────────────────────
def test_inserted_tags_round_trip():
    tagged = "本を読む。[pause 500ms] しかし、[sigh] それは間違いです。"
    assert sa.strip_tags(tagged) == sa.strip_tags(SRC)


@pytest.mark.parametrize("tag", list(sa.REACTIONS) + [
    "[pause 300ms]", "[pause 1.5s]", "[pause 1s]",
])
def test_every_insertable_tag_is_stripped(tag):
    assert sa.strip_tags(f"あ{tag}い") == "あい"


def test_headings_voice_and_emphasis_are_not_stripped():
    """The author's structure — a model that drops one must be caught.

    [emphasis] joined this set when it was dropped from what the tool inserts:
    the engine's rendering of it stood out oddly in narration.
    """
    assert sa.strip_tags("# 章\n[voice:me] あ") == "#章[voice:me]あ"
    assert sa.strip_tags("[emphasis]あ[/emphasis]") == "[emphasis]あ[/emphasis]"


# ── annotate_chunk: anything but a verified insertion keeps the original ────
def test_emphasis_the_model_added_is_rejected(monkeypatch):
    """The tool no longer inserts emphasis, so adding one changes the
    performance and must not slip through as a verified insertion."""
    _stub_run(monkeypatch, stdout="本を読む。[emphasis]しかし[/emphasis]、それは間違いです。")
    text, reason = sa.annotate_chunk(SRC)
    assert text == SRC and reason


def test_emphasis_already_in_the_script_is_preserved(monkeypatch):
    src = "本を読む。[emphasis]しかし[/emphasis]、それは間違いです。"
    good = "本を読む。[pause 500ms][emphasis]しかし[/emphasis]、それは間違いです。"
    _stub_run(monkeypatch, stdout=good)
    assert sa.annotate_chunk(src) == (good, None)


def test_emphasis_the_model_dropped_is_rejected(monkeypatch):
    src = "本を読む。[emphasis]しかし[/emphasis]、それは間違いです。"
    _stub_run(monkeypatch, stdout="本を読む。しかし、それは間違いです。")
    text, reason = sa.annotate_chunk(src)
    assert text == src and reason


def test_verified_insertion_is_returned(monkeypatch):
    good = "本を読む。[pause 500ms]しかし、それは間違いです。"
    _stub_run(monkeypatch, stdout=good)
    assert sa.annotate_chunk(SRC) == (good, None)


def test_reworded_response_is_rejected(monkeypatch):
    _stub_run(monkeypatch, stdout="本を読んだ。[pause 500ms]しかし、それは間違いです。")
    text, reason = sa.annotate_chunk(SRC)
    assert text == SRC and "words changed" in reason


def test_truncated_response_is_rejected(monkeypatch):
    _stub_run(monkeypatch, stdout="本を読む。[pause 500ms]")
    text, reason = sa.annotate_chunk(SRC)
    assert text == SRC and reason


def test_dropped_heading_is_rejected(monkeypatch):
    src = "# 章\n本を読む。"
    _stub_run(monkeypatch, stdout="本を読む。[pause 500ms]")
    text, reason = sa.annotate_chunk(src)
    assert text == src and reason


def test_code_fence_is_unwrapped_not_rejected(monkeypatch):
    inner = "本を読む。[pause 500ms]しかし、それは間違いです。"
    _stub_run(monkeypatch, stdout=f"```\n{inner}\n```")
    assert sa.annotate_chunk(SRC) == (inner, None)


@pytest.mark.parametrize("kw,expect", [
    (dict(stdout="", returncode=0), "empty response"),
    (dict(stdout="x", returncode=2), "claude exited 2"),
    (dict(exc=FileNotFoundError()), "claude CLI not found"),
    (dict(exc=subprocess.TimeoutExpired("claude", 1)), "timed out"),
])
def test_failures_keep_the_original_text(monkeypatch, kw, expect):
    _stub_run(monkeypatch, **kw)
    text, reason = sa.annotate_chunk(SRC)
    assert text == SRC
    assert expect in reason


# ── whole-script pass ───────────────────────────────────────────────────────
def test_split_preserves_every_word():
    body = "\n\n".join(f"段落{i}。" * 200 for i in range(6))
    chunks = sa.split_chunks(body, max_chars=2000)
    assert len(chunks) > 1
    assert sa.strip_tags("".join(chunks)) == sa.strip_tags(body)


def test_annotate_script_reports_skipped_chunks(monkeypatch):
    """A partial pass must not read as a clean one."""
    _stub_run(monkeypatch, stdout="まったく別の文章です。")
    body = "\n\n".join(f"段落{i}。" * 200 for i in range(3))
    out = sa.annotate_script(body)
    assert out["annotated"] == 0
    assert len(out["skipped"]) == out["chunks"] > 1
    assert sa.strip_tags(out["text"]) == sa.strip_tags(body)  # nothing lost


# ── streaming: the UI's only signal that the pass is alive ──────────────────
def test_stream_reports_progress_for_every_chunk(monkeypatch):
    """start, one event per chunk, then done — a UI with no per-chunk event
    cannot tell a long pass from a hung one."""
    _stub_run(monkeypatch, stdout="まったく別の文章です。")   # every chunk skipped
    body = "\n\n".join(f"段落{i}。" * 200 for i in range(3))
    events = list(sa.iter_annotate_script(body))

    assert events[0]["type"] == "start"
    n = events[0]["chunks"]
    assert n > 1

    chunks = [e for e in events if e["type"] == "chunk"]
    assert [e["index"] for e in chunks] == list(range(1, n + 1))
    assert all(e["chunks"] == n for e in chunks)
    assert all(e["skipped"] and e["reason"] for e in chunks)

    assert events[-1]["type"] == "done"
    assert events[-1]["chunks"] == n


def test_blocking_form_returns_the_final_event(monkeypatch):
    good = "本を読む。[pause 500ms]しかし、それは間違いです。"
    _stub_run(monkeypatch, stdout=good)
    assert sa.annotate_script(SRC) == {
        "type": "done", "text": good, "chunks": 1, "annotated": 1, "skipped": [],
    }


def test_cli_available_reports_a_reason_when_missing(monkeypatch):
    monkeypatch.setattr(sa.shutil, "which", lambda _: None)
    ok, reason = sa.cli_available()
    assert ok is False and reason


# ── Lexicon suggestions ─────────────────────────────────────────────────────
# The words are found locally; only the READINGS come from the model, and the
# caller writes them into the user's lexicon — so nothing the model did not get
# asked about may come back.
class TestFindWords:
    JA = "「apple りんご」と読み上げる。紙に Apple を書く。output も。"

    def test_finds_latin_words_in_a_cjk_script(self):
        assert sa.find_unregistered_latin_words(self.JA) == ["apple", "output"]

    def test_dedupes_case_insensitively_keeping_first_spelling(self):
        assert "Apple" not in sa.find_unregistered_latin_words(self.JA)

    def test_skips_words_the_lexicon_already_covers(self):
        assert sa.find_unregistered_latin_words(self.JA, known=["APPLE"]) == ["output"]

    def test_all_latin_script_yields_nothing(self):
        """Every word would match, and the misread this addresses needs CJK."""
        assert sa.find_unregistered_latin_words("An English book about apples.") == []

    def test_markup_is_never_offered(self):
        text = "あ [pause 500ms] い [voice:narrator] う apple"
        assert sa.find_unregistered_latin_words(text) == ["apple"]

    def test_single_letters_are_not_words(self):
        assert sa.find_unregistered_latin_words("あ a い bc") == ["bc"]

    def test_result_is_capped(self):
        # Alphabetic-only: a digit ends the match, so "word1"/"word2" would both
        # dedupe to "word" and the cap would never be reached.
        import string
        words = [a + b for a in string.ascii_lowercase for b in string.ascii_lowercase]
        many = "あ " + " ".join(words[: sa.MAX_SUGGESTIONS + 20])
        assert len(sa.find_unregistered_latin_words(many)) == sa.MAX_SUGGESTIONS


class TestSuggestReadings:
    def test_parses_tab_separated_readings(self, monkeypatch):
        _stub_run(monkeypatch, stdout="apple\tアップル\noutput\tアウトプット")
        out = sa.suggest_readings(["apple", "output"], "Japanese")
        assert out["readings"] == {"apple": "アップル", "output": "アウトプット"}
        assert out["reason"] is None

    def test_words_we_did_not_ask_about_are_dropped(self, monkeypatch):
        """The caller writes these into the lexicon — nothing invented gets in."""
        _stub_run(monkeypatch, stdout="apple\tアップル\nbanana\tバナナ")
        assert sa.suggest_readings(["apple"], "Japanese")["readings"] == {"apple": "アップル"}

    def test_blank_reading_is_dropped_not_shown_as_a_guess(self, monkeypatch):
        _stub_run(monkeypatch, stdout="apple\t\noutput\tアウトプット")
        assert sa.suggest_readings(["apple", "output"], "Japanese")["readings"] == {
            "output": "アウトプット"
        }

    def test_case_differences_still_match_the_asked_word(self, monkeypatch):
        _stub_run(monkeypatch, stdout="APPLE\tアップル")
        assert sa.suggest_readings(["apple"], "Japanese")["readings"] == {"apple": "アップル"}

    def test_commentary_lines_are_ignored(self, monkeypatch):
        _stub_run(monkeypatch, stdout="Sure! Here you go:\napple\tアップル")
        assert sa.suggest_readings(["apple"], "Japanese")["readings"] == {"apple": "アップル"}

    @pytest.mark.parametrize("kw,expect", [
        (dict(exc=FileNotFoundError()), "claude CLI not found"),
        (dict(exc=subprocess.TimeoutExpired("claude", 1)), "timed out"),
        (dict(stdout="x", returncode=2), "claude exited 2"),
    ])
    def test_failures_report_a_reason_and_no_readings(self, monkeypatch, kw, expect):
        _stub_run(monkeypatch, **kw)
        out = sa.suggest_readings(["apple"], "Japanese")
        assert out["readings"] == {} and expect in out["reason"]

    def test_no_words_makes_no_cli_call(self, monkeypatch):
        def boom(*a, **k):
            raise AssertionError("must not shell out for an empty word list")
        monkeypatch.setattr(sa.subprocess, "run", boom)
        assert sa.suggest_readings([], "Japanese") == {"readings": {}, "reason": None}
