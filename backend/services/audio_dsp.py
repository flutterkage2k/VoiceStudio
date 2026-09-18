"""
Audio DSP pipeline — broadcast-grade mastering + configurable effects chain.

`apply_mastering()` is the shared pre-stage that runs before the user's
effect preset: highpass + gentle compression only (see `MASTERING_CHAIN`).
Reverb is deliberately NOT part of it — it is preset-declared only (e.g.
cinematic, warm); a hidden reverb here used to bake echo into every non-raw
synthesis, which field reports flagged. `apply_effects_chain()` lets callers
build custom pipelines from a list of named effects.

All effects use Spotify's `pedalboard` library. When pedalboard isn't
installed, every function degrades gracefully (returns audio unmodified).
"""
import logging
import torch
from typing import Optional

logger = logging.getLogger("omnivoice.dsp")

# ── Effect presets ──────────────────────────────────────────────────────

EFFECT_PRESETS = {
    "broadcast": {
        "label": "Broadcast",
        "icon": "📻",
        "description": "Radio/podcast standard — warm, compressed, clear.",
        "chain": [
            {"type": "highpass", "cutoff_hz": 80},
            {"type": "compressor", "threshold_db": -18, "ratio": 3.0, "attack_ms": 5, "release_ms": 80},
            {"type": "eq", "low_gain_db": 1.5, "mid_gain_db": 0, "high_gain_db": 2.0},
            {"type": "limiter", "threshold_db": -1.0},
        ],
    },
    "cinematic": {
        "label": "Cinematic",
        "icon": "🎬",
        "description": "Film-quality — spacious reverb, gentle compression.",
        "chain": [
            {"type": "highpass", "cutoff_hz": 60},
            {"type": "compressor", "threshold_db": -15, "ratio": 1.8, "attack_ms": 10, "release_ms": 150},
            {"type": "reverb", "room_size": 0.35, "wet_level": 0.15, "dry_level": 0.85},
            {"type": "limiter", "threshold_db": -1.5},
        ],
    },
    "podcast": {
        "label": "Podcast",
        "icon": "🎙️",
        "description": "Close-mic, intimate — heavy compression, no reverb.",
        "chain": [
            {"type": "highpass", "cutoff_hz": 100},
            {"type": "noise_gate", "threshold_db": -40, "release_ms": 200},
            {"type": "compressor", "threshold_db": -20, "ratio": 4.0, "attack_ms": 2, "release_ms": 60},
            {"type": "eq", "low_gain_db": -1.0, "mid_gain_db": 2.0, "high_gain_db": 1.5},
            {"type": "limiter", "threshold_db": -0.5},
        ],
    },
    "raw": {
        "label": "Raw",
        "icon": "🔇",
        "description": "No processing — model output as-is.",
        "chain": [],
    },
    "warm": {
        "label": "Warm",
        "icon": "☀️",
        "description": "Boosted low-mids, subtle saturation, cozy feel.",
        "chain": [
            {"type": "highpass", "cutoff_hz": 60},
            {"type": "eq", "low_gain_db": 3.0, "mid_gain_db": 1.0, "high_gain_db": -1.0},
            {"type": "compressor", "threshold_db": -16, "ratio": 2.0, "attack_ms": 8, "release_ms": 120},
            {"type": "reverb", "room_size": 0.15, "wet_level": 0.06, "dry_level": 0.94},
        ],
    },
    "bright": {
        "label": "Bright",
        "icon": "✨",
        "description": "Crisp high-end, presence boost, airy feel.",
        "chain": [
            {"type": "highpass", "cutoff_hz": 80},
            {"type": "eq", "low_gain_db": -1.0, "mid_gain_db": 0, "high_gain_db": 4.0},
            {"type": "compressor", "threshold_db": -14, "ratio": 2.5, "attack_ms": 3, "release_ms": 80},
            {"type": "limiter", "threshold_db": -1.0},
        ],
    },
}


def list_effect_presets() -> list[dict]:
    """Return presets for the frontend UI picker."""
    return [
        {"id": k, "label": v["label"], "icon": v["icon"], "description": v["description"]}
        for k, v in EFFECT_PRESETS.items()
    ]


def get_effect_chain(preset_id: str) -> list[dict]:
    """Return the effect chain for a preset. Falls back to empty chain."""
    p = EFFECT_PRESETS.get(preset_id)
    return p["chain"] if p else []


# ── Core DSP functions ──────────────────────────────────────────────────

#: Shared pre-preset mastering stage: highpass + gentle compression ONLY.
#: Reverb must never live here — a hidden Reverb in this chain baked echo
#: into every non-raw synthesis regardless of the chosen preset (field
#: reports of echoey voices; the podcast preset even promises "no reverb").
#: Reverb is preset-declared only (see EFFECT_PRESETS: cinematic, warm).
MASTERING_CHAIN = [
    {"type": "highpass", "cutoff_hz": 60},
    {"type": "compressor", "threshold_db": -15, "ratio": 1.5, "attack_ms": 2.0, "release_ms": 100},
]


def apply_mastering(audio_tensor, sample_rate=24000):
    """Applies the broadcast pre-stage (highpass + gentle compression) to the clone voice.

    Reverb is intentionally absent — only user-chosen effect presets declare
    it. Degrades gracefully: pedalboard missing or any DSP error returns the
    input unmodified.
    """
    try:
        return apply_effects_chain(audio_tensor, sample_rate, MASTERING_CHAIN)
    except Exception as e:
        logger.warning("Mastering DSP Error: %s", e)
        return audio_tensor


def normalize_audio(audio_tensor, target_dBFS=-2.0):
    """Peak-normalizes the audio to a standard broadcasting level (-2 dB) to fix F5TTS volume fluctuations.

    Never amplifies a near-silent signal. A failed/empty render sits at the
    noise floor; blindly scaling its peak up to -2 dBFS applies thousands of
    times of gain and turns silence into full-scale hiss — the "blank noise"
    some generated voices exhibited. Below a -50 dBFS silence floor we leave the
    audio untouched so it stays inaudible (and downstream guards can treat it as
    a dead render) instead of shipping amplified noise. Real speech — even a
    whisper — peaks well above this floor, so normal output is unaffected.
    """
    if audio_tensor.numel() == 0:
        return audio_tensor
    max_val = torch.abs(audio_tensor).max()
    # -50 dBFS ≈ 0.00316 linear. Anything at/below this is silence / noise floor.
    silence_floor = 10 ** (-50.0 / 20.0)
    if max_val > silence_floor:
        target_amp = 10 ** (target_dBFS / 20.0)
        audio_tensor = audio_tensor * (target_amp / max_val)
    return audio_tensor


def trim_trailing_silence(
    audio_tensor: torch.Tensor,
    sample_rate: int,
    keep_tail_s: float = 0.3,
) -> torch.Tensor:
    """Trim trailing near-silence from a generated clip, keeping a short
    natural tail of ``keep_tail_s`` seconds after the last voiced sample.

    Amplitude-based SILENCE trim only — no content analysis of any kind.
    Uses the same -50 dBFS silence floor as :func:`normalize_audio`: the last
    sample above that floor marks the end of speech, and everything more than
    ``keep_tail_s`` past it is dropped.

    Guaranteed no-op cases (input returned as-is, same object):
      • the trailing quiet span is already ≤ ``keep_tail_s`` (clean output);
      • the entire clip sits below the floor (dead render — downstream
        dead-render guards own that case, we must not shrink their evidence);
      • empty input.

    Accepts ``(n,)`` or ``(channels, n)`` tensors; the returned tensor keeps
    the input's shape convention.
    """
    if audio_tensor.numel() == 0:
        return audio_tensor
    # -50 dBFS ≈ 0.00316 linear — matches normalize_audio's silence floor.
    floor = 10 ** (-50.0 / 20.0)
    envelope = torch.abs(audio_tensor)
    if envelope.ndim > 1:
        envelope = envelope.amax(dim=tuple(range(envelope.ndim - 1)))
    voiced = torch.nonzero(envelope > floor)
    if voiced.numel() == 0:
        return audio_tensor
    last_voiced = int(voiced[-1].item())
    end = last_voiced + 1 + int(keep_tail_s * sample_rate)
    if end >= audio_tensor.shape[-1]:
        return audio_tensor
    return audio_tensor[..., :end]


#: A span this short gets its planned length stretched by the engine's
#: short-text boost (omnivoice/utils/duration.py: below 50 frames the estimate
#: is raised on a cube-root curve — a 0.47 s word is given 1.24 s). The model
#: fills the surplus with a breathy lead-in copied from the reference's
#: delivery, which is what a listener hears as an inhale before the word.
SHORT_SPAN_MAX_CHARS = 6
#: Lead kept ahead of the detected onset, so an unvoiced initial consonant
#: (fricative / affricate / stop burst) that sits under the threshold survives.
_LEADIN_KEEP_S = 0.10
_LEADIN_FADE_S = 0.03
#: Onset = first frame within this many dB of the loudest frame. Measured on
#: the lead-in this removes: the breath sat 26-35 dB under the loudest frame,
#: and at -24 its own peaks already read as the onset on 7 of 20 samples.
_LEADIN_ONSET_DB = -20.0


def is_speakable(text: str) -> bool:
    """False for a span with nothing to say — brackets, quotes, dashes only.

    A voice switch inside quotes leaves the quote marks as spans of their own;
    synthesizing one costs a full engine call and yields a stray noise.
    """
    return any(ch.isalnum() for ch in (text or ""))


def is_short_span(text: str) -> bool:
    """True for a span short enough to pick up the boosted-length lead-in."""
    # One unbroken word only. A fragment such as "X.\nY" is short too, but its
    # first word can sit 20 dB under the second — the trim took it for lead-in
    # and a whole word went missing from a rendered book.
    words = "".join(ch if ch.isalnum() else " " for ch in (text or "")).split()
    return len(words) == 1 and len(words[0]) <= SHORT_SPAN_MAX_CHARS


def was_trimmed_as_short(text: str) -> bool:
    """The first, too-wide rule: any span of up to 6 characters, however many
    words. Kept only so segments cached under it can be told apart."""
    return 0 < sum(1 for ch in (text or "") if ch.isalnum()) <= SHORT_SPAN_MAX_CHARS


def trim_short_span_leadin(audio_tensor: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Drop the breathy lead-in ahead of a very short span's first loud frame.

    Level-based only. Keeps ``_LEADIN_KEEP_S`` before the onset and fades it
    in, so nothing starts on a click. No-op (same object back) when the onset
    is already within the kept lead, or the clip is empty / has no loud frame.
    ponytail: a fixed -20 dB onset; a word that opens with a long quiet
    fricative loses what lies beyond the 100 ms lead — raise _LEADIN_KEEP_S if
    that is ever heard.
    """
    n = audio_tensor.shape[-1] if audio_tensor.numel() else 0
    frame = max(1, int(sample_rate * 0.02))
    if n < frame * 4:
        return audio_tensor
    mono = audio_tensor.reshape(-1, n).abs().amax(dim=0) if audio_tensor.ndim > 1 else audio_tensor.abs()
    rms = mono[: n - n % frame].reshape(-1, frame).pow(2).mean(dim=1).sqrt()
    top = float(rms.max())
    if top <= 0:
        return audio_tensor
    loud = torch.nonzero(rms > top * 10 ** (_LEADIN_ONSET_DB / 20.0))
    start = int(loud[0].item()) * frame - int(_LEADIN_KEEP_S * sample_rate)
    if start <= 0:
        return audio_tensor
    out = audio_tensor[..., start:].clone()
    k = min(int(_LEADIN_FADE_S * sample_rate), out.shape[-1])
    out[..., :k] *= torch.linspace(0.0, 1.0, k, dtype=out.dtype, device=out.device)
    return out


#: One kana, optionally with a small ya/yu/yo — a single mora. Code points, not
#: literals: hiragana U+3041-3096, katakana U+30A1-30FA.
_SMALL_YA_YU_YO = "\u3083\u3085\u3087\u30e3\u30e5\u30e7"
#: "kore wa, X." — the engine cannot voice a lone mora (2 of 6 seeds, and
#: those barely), but does once real speech precedes it (5 of 6 behind this
#: carrier). Padding with punctuation or a forced duration did not help (0 of 18).
#: The first is the measured one and the segment-cache key; the others exist so
#: a retry differs even under a pinned seed (the seed is derived from the text).
#: "kore wa, X." / "sore wa, X." / "tsugi wa, X."
_MORA_CARRIERS = (
    "\u3053\u308c\u306f\u3001{}\u3002",
    "\u305d\u308c\u306f\u3001{}\u3002",
    "\u3064\u304e\u306f\u3001{}\u3002",
)


def _is_kana(ch: str) -> bool:
    return "\u3041" <= ch <= "\u3096" or "\u30a1" <= ch <= "\u30fa"


def single_mora_carrier(text: str, attempt: int = 0) -> Optional[str]:
    """The carrier sentence for a span that is one kana mora, else ``None``."""
    said = [ch for ch in (text or "") if ch.isalnum()]
    if not (len(said) == 1 or (len(said) == 2 and said[1] in _SMALL_YA_YU_YO)):
        return None
    if not _is_kana(said[0]):
        return None
    return _MORA_CARRIERS[attempt % len(_MORA_CARRIERS)].format("".join(said))


#: A citation-form mora cut from the carrier measured 0.12-0.24 s.
_MORA_MIN_S, _MORA_MAX_S = 0.06, 0.45
_ISLAND_GAP_S = 0.12
_ISLAND_PAD_S = 0.03


def cut_last_island(audio_tensor: torch.Tensor, sample_rate: int) -> Optional[torch.Tensor]:
    """The last stretch of sound after a pause — the mora behind its carrier.

    ``None`` when there is no pause to cut at, or what follows it is not
    mora-sized; the caller re-renders rather than keep carrier words.
    """
    n = audio_tensor.shape[-1] if audio_tensor.numel() else 0
    frame = max(1, int(sample_rate * 0.02))
    if n < frame * 4:
        return None
    mono = audio_tensor.reshape(-1, n).abs().amax(dim=0) if audio_tensor.ndim > 1 else audio_tensor.abs()
    rms = mono[: n - n % frame].reshape(-1, frame).pow(2).mean(dim=1).sqrt()
    active = (rms > float(mono.max()) * 0.03).tolist()
    gap = max(1, round(_ISLAND_GAP_S / 0.02))
    end = max((i for i, a in enumerate(active) if a), default=-1)
    start = end
    while start > 0 and any(active[max(0, start - gap):start]):
        start -= 1
    if end < 0 or not any(active[:start]):
        return None                      # nothing before the pause: no carrier
    if not _MORA_MIN_S <= (end - start + 1) * 0.02 <= _MORA_MAX_S:
        return None
    pad = int(_ISLAND_PAD_S * sample_rate)
    out = audio_tensor[..., max(0, start * frame - pad): min(n, (end + 1) * frame + pad)].clone()
    k = min(int(0.01 * sample_rate), out.shape[-1] // 2)
    ramp = torch.linspace(0.0, 1.0, k, dtype=out.dtype, device=out.device)
    out[..., :k] *= ramp
    out[..., -k:] *= ramp.flip(0)
    return out


def apply_effects_chain(audio_tensor, sample_rate: int, chain: list[dict]) -> torch.Tensor:
    """Apply a chain of named effects to an audio tensor.

    Each item in `chain` is a dict with a `type` key and effect-specific
    parameters. Unknown types are silently skipped.

    Supported types:
        highpass    — cutoff_hz (default 80)
        lowpass     — cutoff_hz (default 8000)
        compressor  — threshold_db, ratio, attack_ms, release_ms
        reverb      — room_size, wet_level, dry_level
        noise_gate  — threshold_db, release_ms
        eq          — low_gain_db, mid_gain_db, high_gain_db
        limiter     — threshold_db
    """
    if not chain:
        return audio_tensor

    try:
        from pedalboard import (
            Pedalboard,
            Compressor,
            Reverb,
            HighpassFilter,
            LowpassFilter,
            NoiseGate,
            Limiter,
            LowShelfFilter,
            HighShelfFilter,
            PeakFilter,
        )
        import numpy as np
    except ImportError:
        logger.debug("pedalboard not installed — effects chain skipped")
        return audio_tensor

    plugins = []
    for fx in chain:
        t = fx.get("type", "").lower()
        try:
            if t == "highpass":
                plugins.append(HighpassFilter(cutoff_frequency_hz=fx.get("cutoff_hz", 80)))
            elif t == "lowpass":
                plugins.append(LowpassFilter(cutoff_frequency_hz=fx.get("cutoff_hz", 8000)))
            elif t == "compressor":
                plugins.append(Compressor(
                    threshold_db=fx.get("threshold_db", -15),
                    ratio=fx.get("ratio", 2.0),
                    attack_ms=fx.get("attack_ms", 5),
                    release_ms=fx.get("release_ms", 100),
                ))
            elif t == "reverb":
                plugins.append(Reverb(
                    room_size=fx.get("room_size", 0.2),
                    wet_level=fx.get("wet_level", 0.1),
                    dry_level=fx.get("dry_level", 0.9),
                ))
            elif t == "noise_gate":
                plugins.append(NoiseGate(
                    threshold_db=fx.get("threshold_db", -40),
                    release_ms=fx.get("release_ms", 200),
                ))
            elif t == "limiter":
                plugins.append(Limiter(threshold_db=fx.get("threshold_db", -1.0)))
            elif t == "eq":
                low = fx.get("low_gain_db", 0)
                mid = fx.get("mid_gain_db", 0)
                high = fx.get("high_gain_db", 0)
                if low:
                    plugins.append(LowShelfFilter(cutoff_frequency_hz=250, gain_db=low))
                if mid:
                    plugins.append(PeakFilter(cutoff_frequency_hz=1500, gain_db=mid, q=1.0))
                if high:
                    plugins.append(HighShelfFilter(cutoff_frequency_hz=4000, gain_db=high))
            else:
                logger.debug("Unknown effect type: %s — skipped", t)
        except Exception as e:
            logger.warning("Failed to create %s effect: %s", t, e)

    if not plugins:
        return audio_tensor

    board = Pedalboard(plugins)
    audio_np = audio_tensor.cpu().numpy()
    if audio_np.ndim == 1:
        audio_np = audio_np[None, :]
    try:
        effected = board(audio_np, sample_rate, reset=False)
        return torch.from_numpy(effected).to(audio_tensor.device)
    except Exception as e:
        logger.warning("Effects chain failed: %s — returning unmodified audio", e)
        return audio_tensor

