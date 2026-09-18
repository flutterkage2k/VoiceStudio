"""The MPS sidecar must encode a reference clip once, not once per span.

The in-process backend has had an encoded-reference LRU for a long time
(`services.tts_backend._get_clone_prompt`); the sidecar that Apple Silicon
hosts actually run OmniVoice in handed ``ref_audio`` straight to
``model.generate`` every time. Measured on an M4 that was ~15 s per voice
switch, which turned a two-voice audiobook's 10 minutes of synthesis into 17.

These tests drive the sidecar's ``_generate`` with a counting stub model —
no torch, no weights — and pin the contract: one encode per (clip, text),
the encoded prompt is what reaches ``generate``, and every failure mode falls
back to the exact inline call the sidecar always made.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SIDE = Path(__file__).resolve().parents[1] / "engines" / "omnivoice_subprocess" / "main.py"


@pytest.fixture
def side(monkeypatch):
    spec = importlib.util.spec_from_file_location("omnivoice_sidecar_under_test", _SIDE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._prompt_cache.clear()
    return mod


class _Model:
    """Counts encodes and records what generate() was called with."""

    def __init__(self, prompt_generate_raises=False):
        self.encodes = []
        self.calls = []
        self._raises = prompt_generate_raises

    def create_voice_clone_prompt(self, ref_audio, ref_text=None, preprocess_prompt=True):
        self.encodes.append((ref_audio, ref_text, preprocess_prompt))
        return f"prompt:{ref_audio}:{ref_text}"

    def generate(self, **kw):
        self.calls.append(kw)
        if self._raises and "voice_clone_prompt" in kw:
            raise RuntimeError("model rejects prompt")
        return ["audio"]


class _ModelWithoutPromptApi(_Model):
    """An engine that only knows the inline ``ref_audio`` call."""

    create_voice_clone_prompt = None  # shadows the method: hasattr() is False below


def _refs(tmp_path):
    a = tmp_path / "kr.wav"; a.write_bytes(b"kr")
    b = tmp_path / "jp.wav"; b.write_bytes(b"jp")
    return str(a), str(b)


def test_alternating_voices_encode_each_reference_once(side, tmp_path):
    """The bug: two voices alternating → an encode on every span."""
    kr, jp = _refs(tmp_path)
    m = _Model()
    for i in range(10):
        ref = kr if i % 2 == 0 else jp
        side._generate(m, text=f"line {i}", ref_audio=ref, ref_text="t", gen_kw={})
    assert len(m.encodes) == 2                       # not 10
    assert {e[0] for e in m.encodes} == {kr, jp}
    assert all("voice_clone_prompt" in c and "ref_audio" not in c for c in m.calls)


def test_cached_prompt_is_what_reaches_generate(side, tmp_path):
    kr, _ = _refs(tmp_path)
    m = _Model()
    side._generate(m, text="a", ref_audio=kr, ref_text="t", gen_kw={"speed": 1.0})
    (call,) = m.calls
    assert call["voice_clone_prompt"] == f"prompt:{kr}:t"
    assert call["text"] == "a" and call["speed"] == 1.0
    assert "ref_text" not in call                    # mutually exclusive with the prompt


def test_different_ref_text_is_a_different_prompt(side, tmp_path):
    """The parent keys on (clip, text): a re-typed transcript must re-encode."""
    kr, _ = _refs(tmp_path)
    m = _Model()
    side._generate(m, text="a", ref_audio=kr, ref_text="one", gen_kw={})
    side._generate(m, text="b", ref_audio=kr, ref_text="two", gen_kw={})
    assert len(m.encodes) == 2


def test_edited_clip_is_re_encoded(side, tmp_path):
    """Same path, new bytes (the user replaced the file) → new prompt."""
    kr, _ = _refs(tmp_path)
    m = _Model()
    side._generate(m, text="a", ref_audio=kr, ref_text="t", gen_kw={})
    Path(kr).write_bytes(b"kr but longer")          # size changes → key changes
    side._generate(m, text="b", ref_audio=kr, ref_text="t", gen_kw={})
    assert len(m.encodes) == 2


def test_no_reference_takes_the_inline_path_untouched(side):
    m = _Model()
    side._generate(m, text="a", ref_audio=None, ref_text=None, gen_kw={"num_step": 8})
    assert m.encodes == []
    assert m.calls == [{"text": "a", "ref_audio": None, "ref_text": None, "num_step": 8}]


def test_model_without_prompt_api_falls_back_inline(side, tmp_path):
    kr, _ = _refs(tmp_path)
    m = _ModelWithoutPromptApi()
    side._generate(m, text="a", ref_audio=kr, ref_text="t", gen_kw={})
    assert m.calls == [{"text": "a", "ref_audio": kr, "ref_text": "t"}]


def test_prompt_rejected_by_generate_falls_back_inline(side, tmp_path):
    """Same retry the parent does: a prompt the model refuses must not fail the span."""
    kr, _ = _refs(tmp_path)
    m = _Model(prompt_generate_raises=True)
    out = side._generate(m, text="a", ref_audio=kr, ref_text="t", gen_kw={})
    assert out == ["audio"]
    assert m.calls[-1] == {"text": "a", "ref_audio": kr, "ref_text": "t"}


def test_missing_clip_falls_back_inline_instead_of_raising_in_the_cache(side, tmp_path):
    m = _Model()
    gone = str(tmp_path / "nope.wav")
    side._generate(m, text="a", ref_audio=gone, ref_text="t", gen_kw={})
    assert m.encodes == []                           # the cache never stat-crashed
    assert m.calls == [{"text": "a", "ref_audio": gone, "ref_text": "t"}]


def test_cache_is_bounded_like_the_parents(side, tmp_path):
    m = _Model()
    for i in range(side._PROMPT_CACHE_MAX + 3):
        p = tmp_path / f"v{i}.wav"; p.write_bytes(b"x" * (i + 1))
        side._generate(m, text="a", ref_audio=str(p), ref_text="t", gen_kw={})
    assert len(side._prompt_cache) == side._PROMPT_CACHE_MAX
