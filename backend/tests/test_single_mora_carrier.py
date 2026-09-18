"""A lone kana mora is rendered behind a carrier sentence and cut back out.

OmniVoice cannot voice one mora on its own: 2 of 6 seeds, barely, and in a
real book the span came out as breath only. Behind "kore wa, X." the mora was
intelligible in 5 of 6. Padding with punctuation or forcing a longer duration
did not help (0 of 18). Kana are written as code points — repo rule.
"""
from __future__ import annotations

import math

import torch

from services.audio_dsp import cut_last_island, single_mora_carrier
from services.audiobook import Span, synthesize_chapter

SR = 24000
TA, KYA, UE = "た", "きゃ", "上に"   # ta / kya / ue-ni


def _tone(s, amp=0.5):
    t = torch.arange(int(SR * s)) / SR
    return amp * torch.sin(2 * math.pi * 120 * t)


def _carrier_take(mora_s=0.18, gap_s=0.25):
    return torch.cat([torch.zeros(2400), _tone(0.5), torch.zeros(int(SR * gap_s)),
                      _tone(mora_s), torch.zeros(2400)])[None]


def test_only_a_single_kana_mora_gets_a_carrier():
    assert TA in single_mora_carrier(TA)
    assert KYA in single_mora_carrier(KYA)
    assert TA in single_mora_carrier(f"[{TA}]")            # brackets do not count
    for text in (UE, TA + TA, "a", "", "..."):
        assert single_mora_carrier(text) is None


def test_retries_use_a_different_carrier():
    assert len({single_mora_carrier(TA, i) for i in range(3)}) == 3


def test_the_mora_is_cut_out_and_the_carrier_is_gone():
    out = cut_last_island(_carrier_take(), SR)
    assert 0.18 <= out.shape[-1] / SR <= 0.18 + 0.08       # mora + padding only
    assert float(out[0, 0]) == 0.0 and float(out[0, -1]) == 0.0


def test_no_pause_or_an_oversized_tail_is_refused():
    assert cut_last_island(_tone(0.6)[None], SR) is None            # ran together
    assert cut_last_island(_carrier_take(mora_s=0.9), SR) is None   # not a mora
    assert cut_last_island(torch.zeros(1, SR), SR) is None


def test_chapter_renders_the_mora_through_the_carrier():
    said = []

    def synth(text, *_):
        said.append(text)
        return _carrier_take()

    audio, _ = synthesize_chapter([Span(voice_id="jp", text=TA)], synth, SR, crossfade_ms=0)
    assert said == [single_mora_carrier(TA)]
    assert audio.shape[-1] / SR < 0.3


def test_a_take_with_no_clean_cut_is_retried_then_falls_back_to_the_bare_mora():
    said = []

    def synth(text, *_):
        said.append(text)
        return _tone(0.6)[None]                                     # never a pause

    class _Cache:
        stored = []

        def load(self, span, nonce=0):
            return None

        def store(self, span, audio, nonce=0):
            self.stored.append(span.text)

    cache = _Cache()
    synthesize_chapter([Span(voice_id="jp", text=TA)], synth, SR, segment_cache=cache)
    assert said == [single_mora_carrier(TA, i) for i in range(3)] + [TA]
    assert cache.stored == []              # so the next render tries again


def test_the_cut_mora_is_cached_under_the_carrier_key():
    class _Cache:
        stored = []

        def load(self, span, nonce=0):
            return None

        def store(self, span, audio, nonce=0):
            self.stored.append(span.text)

    cache = _Cache()
    synthesize_chapter([Span(voice_id="jp", text=TA)], lambda *_: _carrier_take(), SR,
                       segment_cache=cache)
    assert cache.stored == [single_mora_carrier(TA)]
