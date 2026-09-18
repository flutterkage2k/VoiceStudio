"""omnivoice-subprocess sidecar entry point (#730/#1190).

Runs the resident OmniVoice TTS model in a child process under the parent's own
interpreter (same pins), so a wedged generate can be hard-killed by the parent
to reclaim VRAM/device, the thing the in-process ``ThreadPoolExecutor`` worker
structurally cannot do.

Wire protocol: length-prefixed JSON over stdin/stdout, byte-identical to
``services/subprocess_backend.py`` and ``engines/dots_tts/main.py``::

    [ 4-byte big-endian uint32 length ][ N bytes UTF-8 JSON ]

Op flow:
    1. sidecar -> parent: {"op":"ready","engine":"omnivoice-subprocess",
                            "sample_rate":24000}
    2. parent -> sidecar: {"op":"ping"} -> {"op":"pong","vram_mb":N}
    3. parent -> sidecar: {"op":"synthesize","text":"...",
                            "ref_audio":"/path","ref_text":"...",
                            "language":"...","num_step":16,
                            "guidance_scale":2.0,"speed":1.0,...}
       -> {"op":"progress",...} (cold load) then
       -> {"op":"audio","audio_pcm_b64":"...","sample_rate":24000,
           "n_samples":N}
    4. parent -> sidecar: {"op":"shutdown"} -> exit 0

Stdlib-only at import time; torch + the OmniVoice model are imported lazily on
the first synthesize so the ``ready`` frame fits the parent's 30s spawn
handshake even on a cold filesystem.
"""
from __future__ import annotations

import base64
import json
import os
import struct
import sys
import traceback

# Mirrors backend/services/subprocess_backend.py::MAX_FRAME_BYTES (T-02-01).
MAX_FRAME_BYTES = 64 * 1024 * 1024

#: OmniVoice's canonical output rate. The real value is re-read from the loaded
#: model on each generate.
OMNIVOICE_SAMPLE_RATE = 24000

#: kwargs model.generate accepts. The parent forwards JSON-safe kwargs from
#: backend.generate(); allowlist the known surface so an unexpected key never
#: reaches model.generate (it has an explicit signature and TypeErrors on
#: unknown kwargs). cache_ref is a parent-side cache marker, not a model param.
_GEN_KW_ALLOWLIST = (
    "language", "instruct", "duration", "num_step", "guidance_scale",
    "speed", "denoise", "postprocess_output", "preprocess_prompt",
    "t_shift", "layer_penalty_factor", "position_temperature",
    "class_temperature", "audio_chunk_duration", "audio_chunk_threshold",
)

_model = None

# ── Encoded-reference cache ──────────────────────────────────────────────────
# The in-process backend never re-encodes a reference clip it has already seen
# (services.tts_backend._get_clone_prompt, an 8-entry LRU). This sidecar used
# to hand ``ref_audio`` straight to ``model.generate`` on every request, so on
# the hosts that run OmniVoice in a sidecar (Apple Silicon) every span paid the
# reference encode again — and a script that alternates two voices paid it on
# EVERY span, because the model's own single-slot memory only helps when the
# same voice repeats. Measured on an M4: ~15 s per voice switch, which turned a
# two-voice audiobook's 10 minutes of synthesis into 17.
#
# Kept local rather than importing the parent's cache: the sidecar is a
# minimal process by design (crash isolation), and pulling in
# services.tts_backend would drag the whole engine registry along with it.
# Same key, same bound, same fallback as the parent, so behaviour matches
# across hosts.
from collections import OrderedDict

_PROMPT_CACHE_MAX = 8
_prompt_cache: "OrderedDict[tuple, object]" = OrderedDict()


def _prompt_key(ref_audio: str, ref_text, preprocess_prompt: bool) -> tuple:
    st = os.stat(ref_audio)
    return (os.path.abspath(ref_audio), st.st_mtime_ns, st.st_size,
            ref_text or "", bool(preprocess_prompt))


def _clone_prompt(model, ref_audio: str, ref_text, preprocess_prompt: bool):
    """The encoded reference for ``ref_audio``, encoded once per (clip, text).

    Returns ``None`` when the model has no prompt API or the encode fails, so
    the caller falls back to the inline path — never a hard error for a cache.
    """
    if not callable(getattr(model, "create_voice_clone_prompt", None)):
        return None
    try:
        key = _prompt_key(ref_audio, ref_text, preprocess_prompt)
    except OSError:
        return None
    hit = _prompt_cache.get(key)
    if hit is not None:
        _prompt_cache.move_to_end(key)
        return hit
    try:
        prompt = model.create_voice_clone_prompt(
            ref_audio, ref_text=ref_text, preprocess_prompt=preprocess_prompt
        )
    except Exception:  # noqa: BLE001 — a cache must not turn a working path into a failure
        return None
    _prompt_cache[key] = prompt
    while len(_prompt_cache) > _PROMPT_CACHE_MAX:
        _prompt_cache.popitem(last=False)
    return prompt


def _generate(model, *, text: str, ref_audio, ref_text, gen_kw: dict):
    """``model.generate`` with the reference encoded once, not once per span.

    ``voice_clone_prompt`` and ``ref_audio``/``ref_text`` are mutually
    exclusive on the model, so a cached prompt goes in alone; anything else —
    no reference, no prompt API, encode failure, a prompt the model rejects —
    takes the exact inline call this sidecar always made.
    """
    if ref_audio:
        prompt = _clone_prompt(model, ref_audio, ref_text,
                               bool(gen_kw.get("preprocess_prompt", True)))
        if prompt is not None:
            try:
                return model.generate(text=text, voice_clone_prompt=prompt, **gen_kw)
            except Exception:  # noqa: BLE001 — same retry the parent does
                pass
    return model.generate(text=text, ref_audio=ref_audio, ref_text=ref_text, **gen_kw)


# ── wire protocol ─────────────────────────────────────────────────────────


def _send(stream, obj: dict) -> None:
    body = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    stream.write(struct.pack("!I", len(body)))
    stream.write(body)
    stream.flush()


def _recv(stream):
    header = stream.read(4)
    if len(header) < 4:
        return None  # EOF
    (n,) = struct.unpack("!I", header)
    if n > MAX_FRAME_BYTES:
        raise IOError(f"frame too large: {n}")
    body = bytearray()
    while len(body) < n:
        chunk = stream.read(n - len(body))
        if not chunk:
            raise IOError("short read")
        body.extend(chunk)
    return json.loads(bytes(body).decode("utf-8"))


def _measure_vram_mb() -> float:
    """This sidecar's own GPU memory in MB. 0 on CPU. Never raises."""
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.memory_allocated() / (1024 ** 2), 1)
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return round(torch.mps.driver_allocated_memory() / (1024 ** 2), 1)
    except Exception:
        pass
    return 0.0


# ── model loading (lazy, on first synthesize) ─────────────────────────────


def _ensure_backend_on_path() -> None:
    """The sidecar is launched as ``<python> main.py``, so sys.path[0] is this
    script's directory, not ``backend/``. Add ``backend/`` so ``services`` is
    importable, letting us reuse the parent's load primitives verbatim."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root and root not in sys.path:
        sys.path.insert(0, root)


def _load_model(stdout):
    """Cold-construct the OmniVoice model, reusing the parent's load path."""
    global _model
    if _model is not None:
        return _model

    _send(stdout, {"op": "progress", "stage": "loading_model", "percent": 0})

    _ensure_backend_on_path()
    # Reuse the parent's own load primitives (same interpreter): the checkpoint
    # resolver, device probe, and ASR-preload policy are exactly what the
    # in-process engine uses, so this sidecar loads the identical model.
    from services.model_manager import (  # noqa: PLC0415
        _lazy_omnivoice,
        _lazy_torch,
        get_best_device,
        resolve_omnivoice_checkpoint,
        should_preload_tts_asr,
    )
    from utils.hf_progress import register_listener, unregister_listener  # noqa: PLC0415

    # Forward real HF download/weight progress so the parent's recv loop keeps
    # its watchdog alive across a slow cold load (the parent consumes these
    # {"op": "progress"} frames and re-arms its deadline on each one).
    def _on_progress(ev):
        pct = ev.get("pct", 0.0)
        if pct:
            _send(stdout, {"op": "progress", "stage": "loading_model",
                           "percent": min(round(pct * 100), 99)})

    torch = _lazy_torch()
    OmniVoice = _lazy_omnivoice()
    checkpoint = resolve_omnivoice_checkpoint()
    device = get_best_device()
    preload_asr = should_preload_tts_asr()

    lid = register_listener(_on_progress)
    try:
        _model = OmniVoice.from_pretrained(
            checkpoint, device_map=device, dtype=torch.float16, load_asr=preload_asr,
        )
    finally:
        unregister_listener(lid)
    _send(stdout, {"op": "progress", "stage": "loading_model", "percent": 100})
    return _model


def _tensor_to_pcm_b64(audio, sample_rate: int) -> tuple[str, int, int]:
    """Convert a torch waveform tensor (1, N) in [-1, 1] to base64 int16 PCM."""
    import numpy as np

    arr = audio.detach().to("cpu").float().numpy()
    arr = np.asarray(arr, dtype=np.float32).squeeze()
    while arr.ndim > 1:
        # Downmix along whichever axis is the channel axis. Hardcoded to axis 0
        # this averaged across TIME for a channels-last (N, 2) array -- every
        # output sample became the mean of two neighbouring samples, which is
        # not a downmix but a destroyed waveform. (#1328)
        arr = arr.mean(axis=int(np.argmin(arr.shape)))
    arr = np.clip(arr, -1.0, 1.0)
    pcm = (arr * 32767.0).astype(np.int16).tobytes()
    return base64.b64encode(pcm).decode("ascii"), int(sample_rate), int(arr.shape[0])


def _handle_synthesize(msg: dict, stdout) -> None:
    """Dispatch one synthesize request. Emits the audio frame or raises."""
    text = msg.get("text")
    if not text or not isinstance(text, str):
        raise ValueError("synthesize: missing or non-string 'text'")

    model = _load_model(stdout)

    ref_audio = msg.get("ref_audio") or None
    ref_text = msg.get("ref_text") or None
    gen_kw = {k: msg[k] for k in _GEN_KW_ALLOWLIST if k in msg}

    seed = msg.get("seed")
    if seed is not None:
        import torch

        torch.manual_seed(int(seed))

    audios = _generate(model, text=text, ref_audio=ref_audio, ref_text=ref_text,
                       gen_kw=gen_kw)
    audio = audios[0] if isinstance(audios, (list, tuple)) else audios
    sample_rate = int(getattr(model, "sampling_rate", OMNIVOICE_SAMPLE_RATE))

    pcm_b64, sr, n_samples = _tensor_to_pcm_b64(audio, sample_rate)
    _send(stdout, {
        "op": "audio",
        "audio_pcm_b64": pcm_b64,
        "sample_rate": sr,
        "n_samples": n_samples,
    })


# ── main loop ─────────────────────────────────────────────────────────────


def main() -> int:
    stdin = sys.stdin.buffer
    # Frames go down a PRIVATE fd, and fd 1 is pointed at stderr (#1428).
    #
    # This sidecar's protocol is length-prefixed binary on stdout, but it is
    # not the only thing writing there: the libraries it loads print freely to
    # fd 1 — wetextprocessing's FST logs, tqdm bars, native prints from torch
    # and ONNX runtime. Those bytes interleave with frames, and the parent
    # then reads four bytes of log text as a length prefix, which is how a
    # generation dies with `OSError: frame too large: 1044258881` (that number
    # is ASCII). Worse, it desyncs the stream, so every later request on the
    # same sidecar reads stale bytes and no retry can recover.
    #
    # Duplicating fd 1 first keeps a clean channel only this module can write
    # to; redirecting fd 1 to fd 2 sends the library noise to stderr, which
    # the parent already drains into its own log (through the HF-token
    # redactor). Nothing is lost and the frame stream cannot be corrupted.
    _frame_fd = os.dup(1)
    os.dup2(2, 1)
    stdout = os.fdopen(_frame_fd, "wb")

    # Ready handshake fires BEFORE any heavy import.
    _send(stdout, {
        "op": "ready",
        "engine": "omnivoice-subprocess",
        "sample_rate": OMNIVOICE_SAMPLE_RATE,
    })

    while True:
        try:
            msg = _recv(stdin)
        except Exception as exc:
            _send(stdout, {
                "op": "error",
                "stage": "recv",
                "message": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            })
            return 1
        if msg is None:
            return 0

        op = msg.get("op") if isinstance(msg, dict) else None
        try:
            if op == "ping":
                _send(stdout, {"op": "pong", "vram_mb": _measure_vram_mb()})
            elif op == "synthesize":
                _handle_synthesize(msg, stdout)
            elif op == "shutdown":
                return 0
            else:
                _send(stdout, {
                    "op": "error",
                    "stage": "dispatch",
                    "message": f"unknown op: {op!r}",
                })
        except Exception as exc:
            _send(stdout, {
                "op": "error",
                "stage": op or "unknown",
                "message": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            })


if __name__ == "__main__":
    sys.exit(main())
