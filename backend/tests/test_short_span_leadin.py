"""A very short audiobook span must not open with the engine's breathy lead-in.

OmniVoice stretches the planned length of a short text (a 0.47 s word is given
1.24 s) and fills the surplus with a breath before the word. Measured on a
two-character span: 0.54-0.74 s of noise ahead of the voice, every seed.
"""
from __future__ import annotations

import math

import torch

from services.audio_dsp import is_short_span, trim_short_span_leadin

SR = 24000


def _clip(lead_s=0.6, lead_db=-30.0, word_s=0.4):
    torch.manual_seed(0)
    lead = torch.randn(int(SR * lead_s)) * 10 ** (lead_db / 20.0)
    t = torch.arange(int(SR * word_s)) / SR
    word = 0.5 * torch.sin(2 * math.pi * 120 * t)
    return torch.cat([lead, word])[None]


def test_breathy_leadin_is_cut_to_the_kept_lead():
    out = trim_short_span_leadin(_clip(), SR)
    cut = (_clip().shape[-1] - out.shape[-1]) / SR
    assert 0.45 < cut < 0.55          # 0.6 s lead-in minus the 0.1 s kept


def test_the_word_itself_is_untouched():
    src = _clip()
    out = trim_short_span_leadin(src, SR)
    n = int(SR * 0.4)
    assert torch.equal(out[..., -n:], src[..., -n:])


def test_kept_lead_fades_in_from_zero():
    assert float(trim_short_span_leadin(_clip(), SR)[0, 0]) == 0.0


def test_clean_onset_is_returned_as_is():
    src = _clip(lead_s=0.05)
    assert trim_short_span_leadin(src, SR) is src


def test_silent_and_empty_clips_are_returned_as_is():
    for src in (torch.zeros(1, SR), torch.zeros(0)):
        assert trim_short_span_leadin(src, SR) is src


def test_one_dimensional_input_keeps_its_shape():
    assert trim_short_span_leadin(_clip()[0], SR).ndim == 1


def test_only_very_short_spans_qualify():
    assert is_short_span("ab")
    assert is_short_span("[ab]")                       # punctuation is not counted
    assert not is_short_span("a" * 7)
    # Two words: the first may be far quieter than the second, and was cut
    # out of a real render as if it were the lead-in.
    assert not is_short_span("ab.\ncd")
    assert not is_short_span("ab cd")
    assert not is_short_span("")
    assert not is_short_span("...")


def test_chapter_synthesis_trims_short_spans_only():
    from services.audiobook import Span, synthesize_chapter

    clip = _clip()
    spans = [Span(voice_id=None, text="ab"), Span(voice_id=None, text="a much longer sentence")]
    audio, _ = synthesize_chapter(spans, lambda *_: clip, SR, crossfade_ms=0)
    trimmed = trim_short_span_leadin(clip, SR).shape[-1]
    assert audio.shape[-1] == trimmed + clip.shape[-1]


def test_a_segment_cached_before_the_trim_is_cleaned_on_load():
    from services.audiobook import Span, synthesize_chapter

    clip = _clip()

    class _Cache:
        def load(self, span, nonce=0):
            return clip

        def store(self, *a, **k):
            raise AssertionError("a cache hit must not be re-stored")

    def _no_synth(*_):
        raise AssertionError("a cache hit must not synthesize")

    audio, _ = synthesize_chapter([Span(voice_id=None, text="ab")], _no_synth, SR,
                                  segment_cache=_Cache())
    assert audio.shape[-1] == trim_short_span_leadin(clip, SR).shape[-1]


def test_punctuation_only_spans_are_not_synthesized():
    """Quote marks left around a voice switch cost an engine call each."""
    from services.audiobook import Span, synthesize_chapter

    said = []

    def synth(text, *_):
        said.append(text)
        return _clip()

    spans = [Span(voice_id=None, text="["), Span(voice_id="jp", text="ab"),
             Span(voice_id=None, text="] - ["), Span(voice_id=None, text="...", pause_ms_after=300)]
    audio, _ = synthesize_chapter(spans, synth, SR, crossfade_ms=0)
    assert said == ["ab"]
    assert audio.shape[-1] == trim_short_span_leadin(_clip(), SR).shape[-1] + int(SR * 0.3)


def test_a_multiword_fragment_cached_under_the_old_rule_is_not_replayed():
    """Its stored WAV may have lost its first word to the too-wide trim."""
    from services.audiobook import Span, synthesize_chapter

    class _Cache:
        asked, stored = [], []

        def load(self, span, nonce=0):
            self.asked.append(span.text)
            return None

        def store(self, span, audio, nonce=0):
            self.stored.append(span.text)

    cache = _Cache()
    clip = _clip()
    audio, _ = synthesize_chapter([Span(voice_id=None, text="ab.\ncd")], lambda *_: clip, SR,
                                  crossfade_ms=0, segment_cache=cache)
    assert cache.asked == cache.stored and cache.asked[0] != "ab.\ncd"
    assert audio.shape[-1] == clip.shape[-1]          # and it is no longer trimmed
