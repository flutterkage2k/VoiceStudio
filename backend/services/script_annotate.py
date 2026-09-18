"""Annotate an audiobook script with pause / emphasis / reaction markup.

Drives the local ``claude`` CLI (`claude -p`), so it runs on the user's own
Claude Code subscription — no API key, no per-call billing, and no new network
endpoint the app has to be trusted with. That also makes it a **local
capability, not a feature**: the button is disabled with a reason wherever the
CLI is absent, exactly like the engines that report themselves unavailable.

The one rule the model is never allowed to break is that the words stay put.
Every response is checked by stripping the tags this module is allowed to
insert and comparing what is left to what went in; a chunk that fails the
check is discarded and the original text is kept. Silent rewrites are the
failure mode that would actually cost a user a finished render, so a chunk we
cannot verify is never returned.
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess

logger = logging.getLogger("omnivoice.script_annotate")

#: Reaction tokens the engine understands (mirrors the Markup toolbar's list).
REACTIONS = (
    "[laughter]", "[sigh]", "[confirmation-en]",
    "[question-en]", "[question-ah]", "[question-oh]",
    "[question-ei]", "[question-yi]",
    "[surprise-ah]", "[surprise-oh]", "[surprise-wa]", "[surprise-yo]",
    "[dissatisfaction-hnn]",
)

#: Per-request chunk size. Paragraph-bounded, so a chunk is always whole
#: thoughts; small enough that one bad response costs little.
DEFAULT_MAX_CHARS = 2000
#: One CLI call's ceiling. A wedged CLI must not hold the request forever.
CHUNK_TIMEOUT_S = 300.0

PROMPT = """You are annotating a script that will be read aloud by a text-to-speech engine (VoiceStudio).

Insert performance tags at natural places. Return the SAME TEXT with tags added.

ALLOWED TAGS — you may add only these:
- [pause 300ms] [pause 500ms] [pause 800ms] [pause 1s] [pause 1.5s]
  Use after a sentence that lands a point, at paragraph breaks, and before a
  turn in the argument. Longer pauses at bigger structural breaks.
- Reactions, each as a standalone token: {reactions}
  Only where the writing genuinely calls for a vocal reaction. In expository
  or argumentative prose that is usually nowhere. Do not sprinkle them.

HARD RULES:
1. Do NOT change, translate, reorder, add or delete a single word of the text.
   Only insert tags between the existing words.
2. Do NOT add `#` headings, [voice:...] switches or [emphasis]...[/emphasis].
   If they are already there, leave them exactly as they are — character for
   character.
3. Do NOT wrap the answer in code fences, and do not explain anything.
   Output the annotated text and nothing else.
4. Keep the original line breaks.

TEXT:
{text}"""

#: Only the tags this module inserts. `#` headings, [voice:NAME] and
#: [emphasis] are the author's structure, so they are deliberately NOT
#: stripped — a model that adds or drops one fails the comparison instead of
#: quietly changing the performance.
#:
#: [emphasis] used to be inserted here and was dropped on listening: the
#: engine's rendering of it stood out oddly in narration. Adding it back needs
#: a listening pass, not a code change.
_TAG_RE = re.compile(
    r"\[pause\s+[0-9.]+m?s\]"
    r"|\[(?:laughter|sigh|confirmation-en|question-(?:en|ah|oh|ei|yi)"
    r"|surprise-(?:ah|oh|wa|yo)|dissatisfaction-hnn)\]"
)


def cli_available() -> tuple[bool, str]:
    """``(ok, reason)`` — whether the ``claude`` CLI can be invoked."""
    if shutil.which("claude"):
        return True, ""
    return False, "claude CLI not found on PATH"


def strip_tags(text: str) -> str:
    """`text` with insertable tags removed and whitespace dropped.

    Whitespace is dropped rather than normalised because inserting a tag
    legitimately shifts spacing around it; the words are what must survive.
    """
    return re.sub(r"\s+", "", _TAG_RE.sub("", text))


def split_chunks(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[str]:
    """Split on blank lines, packing paragraphs up to ``max_chars``."""
    parts = re.split(r"(\n\s*\n)", text)
    chunks: list[str] = []
    buf = ""
    for part in parts:
        if buf and len(buf) + len(part) > max_chars:
            chunks.append(buf)
            buf = part.lstrip("\n")
        else:
            buf += part
    if buf.strip():
        chunks.append(buf)
    return chunks or [text]


def _unfence(out: str) -> str:
    """Drop a stray markdown fence — the words inside are still intact."""
    if out.startswith("```"):
        return re.sub(r"^```[a-z]*\n?|\n?```$", "", out).strip()
    return out


def annotate_chunk(chunk: str, *, model: str | None = None,
                   timeout_s: float = CHUNK_TIMEOUT_S) -> tuple[str, str | None]:
    """One CLI round trip. Returns ``(text, skip_reason)``.

    ``skip_reason`` is ``None`` on success; otherwise the original chunk comes
    back unchanged and the reason says why, so the caller can report how much
    of the script was left alone instead of pretending it all worked.
    """
    prompt = PROMPT.format(reactions=" ".join(REACTIONS), text=chunk)
    cmd = ["claude", "-p", prompt]
    if model:
        cmd[1:1] = ["--model", model]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except FileNotFoundError:
        return chunk, "claude CLI not found"
    except subprocess.TimeoutExpired:
        return chunk, "timed out"
    if proc.returncode != 0:
        first = (proc.stderr or "").strip().splitlines()
        return chunk, f"claude exited {proc.returncode}" + (f": {first[0][:120]}" if first else "")

    out = _unfence((proc.stdout or "").strip())
    if not out:
        return chunk, "empty response"
    if strip_tags(out) != strip_tags(chunk):
        return chunk, "the words changed under the tags"
    return out, None


def iter_annotate_script(text: str, *, model: str | None = None,
                        max_chars: int = DEFAULT_MAX_CHARS):
    """Annotate a whole script, yielding progress as it goes.

    A long script is minutes of CLI calls, so this is a generator rather than
    one blocking call: the caller streams each event to the UI, which is the
    difference between "working" and "frozen" on screen.

    Yields ``{"type": "start", "chunks": N}``, then one
    ``{"type": "chunk", "index", "chunks", "skipped", "reason"}`` per chunk,
    then a final ``{"type": "done", "text", "chunks", "annotated", "skipped"}``.
    Never raises on a per-chunk failure — a chunk that cannot be verified
    keeps its original text and is reported in ``skipped``.
    """
    chunks = split_chunks(text, max_chars)
    yield {"type": "start", "chunks": len(chunks)}

    out: list[str] = []
    skipped: list[dict] = []
    for i, chunk in enumerate(chunks, start=1):
        annotated, reason = annotate_chunk(chunk, model=model)
        if reason:
            # Static message + index only: the script is the user's manuscript
            # and must not reach the log; the reason strings are our own.
            logger.warning("script annotate: chunk %d/%d kept as-is (%s)",
                           i, len(chunks), reason)
            skipped.append({"index": i, "reason": reason})
        else:
            logger.info("script annotate: chunk %d/%d annotated", i, len(chunks))
        out.append(annotated)
        yield {"type": "chunk", "index": i, "chunks": len(chunks),
               "skipped": bool(reason), "reason": reason}

    yield {
        "type": "done",
        "text": "\n\n".join(c.strip("\n") for c in out),
        "chunks": len(chunks),
        "annotated": len(chunks) - len(skipped),
        "skipped": skipped,
    }


def annotate_script(text: str, *, model: str | None = None,
                    max_chars: int = DEFAULT_MAX_CHARS) -> dict:
    """Blocking form of :func:`iter_annotate_script` — the final event."""
    last = None
    for ev in iter_annotate_script(text, model=model, max_chars=max_chars):
        last = ev
    return last

# ── Lexicon suggestions ─────────────────────────────────────────────────────
# A Latin word inside CJK narration is the reliable misread: OmniVoice said
# "アポラン語" for "apple りんご". The lexicon fixes it, but only if the user
# notices the word in the first place — so find the candidates for them.

#: Only offer suggestions for a script that actually mixes scripts. In an
#: all-Latin script every word would match, and the failure this addresses
#: does not happen there.
_CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힣]")
#: Markup spans are ours, not narration — a [voice:NAME] must never be offered.
_MARKUP_SPAN_RE = re.compile(r"\[[^\]\[\n]{0,128}\]")
#: A Latin run of 2+ letters. Apostrophes/hyphens stay inside the word.
_LATIN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]{1,}")
#: Bound the CLI prompt (and the dialog) for a book-length script.
MAX_SUGGESTIONS = 50


def find_unregistered_latin_words(text: str, known=()) -> list[str]:
    """Latin words in a CJK script that the lexicon doesn't cover yet.

    Returns first-seen order, case-folded-deduplicated, capped at
    ``MAX_SUGGESTIONS``. Empty when the script has no CJK at all.
    """
    if not text or not _CJK_RE.search(text):
        return []
    seen = {str(k).strip().casefold() for k in (known or ()) if str(k).strip()}
    out: list[str] = []
    for word in _LATIN_WORD_RE.findall(_MARKUP_SPAN_RE.sub(" ", text)):
        key = word.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(word)
        if len(out) >= MAX_SUGGESTIONS:
            break
    return out


_READING_PROMPT = """These words appear inside a script that will be read aloud by a \
text-to-speech engine in {language}. Give the reading for each, written in \
{language}'s own script, as the narrator should pronounce it.

Reply with one line per word, exactly `word\treading`, nothing else. No \
numbering, no commentary, no code fences. If you are unsure of a word, give an \
empty reading (`word` then a tab then nothing) rather than guessing.

WORDS:
{words}"""


def suggest_readings(words, language: str, *, model: str | None = None,
                     timeout_s: float = CHUNK_TIMEOUT_S) -> dict:
    """Propose a reading per word. Returns ``{"readings": {...}, "reason": str|None}``.

    Only words that were ASKED about come back — a reading for anything else is
    dropped rather than shown, because the caller is about to write these into
    the user's lexicon. A blank reading is kept out too: the prompt asks for one
    when unsure, and an empty row is the honest answer, not a guess to display.
    """
    words = [w for w in (words or []) if w]
    if not words:
        return {"readings": {}, "reason": None}
    prompt = _READING_PROMPT.format(language=language or "the script's language",
                                    words="\n".join(words))
    cmd = ["claude", "-p", prompt]
    if model:
        cmd[1:1] = ["--model", model]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except FileNotFoundError:
        return {"readings": {}, "reason": "claude CLI not found"}
    except subprocess.TimeoutExpired:
        return {"readings": {}, "reason": "timed out"}
    if proc.returncode != 0:
        return {"readings": {}, "reason": f"claude exited {proc.returncode}"}

    asked = {w.casefold(): w for w in words}
    readings: dict[str, str] = {}
    for line in _unfence((proc.stdout or "").strip()).splitlines():
        if "\t" not in line:
            continue
        raw, say = line.split("\t", 1)
        original = asked.get(raw.strip().casefold())
        if original and say.strip():
            readings[original] = say.strip()
    return {"readings": readings, "reason": None}
