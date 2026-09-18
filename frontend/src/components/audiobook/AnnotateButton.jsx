import React, { useEffect, useRef, useState } from 'react';
import { Sparkles } from 'lucide-react';
import { annotateScript, annotateStatus } from '../../api/audiobook';
import { parseSSELine, splitSSEBuffer } from '../../utils/sseParse';

const PILL =
  'inline-flex h-[26px] items-center justify-center gap-[4px] border border-transparent bg-[var(--chrome-bg)] text-[var(--chrome-fg-muted)] px-[7px] py-0 rounded-[var(--chrome-radius-pill)] [font-family:var(--chrome-font-mono)] text-[0.62rem] whitespace-nowrap cursor-pointer transition-colors duration-[120ms] hover:bg-[var(--chrome-hover-bg)] hover:text-[var(--chrome-fg)] focus-visible:[outline:2px_solid_var(--chrome-accent)] focus-visible:[outline-offset:1px] disabled:opacity-50 disabled:cursor-not-allowed';

const mmss = (s) =>
  `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;

/**
 * Annotate the script with pause / emphasis / reaction markup (local `claude`
 * CLI, no API key).
 *
 * Reads the backend's SSE stream so a minutes-long pass shows real progress —
 * chunk N of M plus elapsed time. Without that, the only signal is a button
 * that changed label, which reads exactly like a hang.
 *
 * The result is a PREVIEW the user applies explicitly: the script is their
 * manuscript, and overwriting it silently is not recoverable.
 */
export default function AnnotateButton({ t, text, setText, onBusyChange }) {
  const [status, setStatus] = useState(null);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(null); // {index, chunks}
  const [elapsed, setElapsed] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const startedAt = useRef(0);

  useEffect(() => {
    let alive = true;
    annotateStatus()
      .then((s) => alive && setStatus(s))
      .catch(() => alive && setStatus({ available: false, reason: '' }));
    return () => {
      alive = false;
    };
  }, []);

  // Elapsed clock. A chunk can take a minute, so "3/7" alone still looks
  // stuck between events — the seconds are what show it is alive.
  useEffect(() => {
    if (!busy) return undefined;
    const id = setInterval(
      () => setElapsed(Math.round((Date.now() - startedAt.current) / 1000)),
      1000,
    );
    return () => clearInterval(id);
  }, [busy]);

  useEffect(() => {
    onBusyChange?.(busy);
  }, [busy, onBusyChange]);

  const disabled = busy || !text.trim() || !(status && status.available);
  const title =
    status && !status.available && status.reason
      ? status.reason
      : t('audiobook.annotate_hint');

  const run = async () => {
    setBusy(true);
    setError('');
    setProgress(null);
    setElapsed(0);
    startedAt.current = Date.now();
    try {
      const res = await annotateScript({ text });
      if (!res.ok || !res.body) throw new Error(await res.text());
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = '';
      let done = null;
      for (;;) {
        const { value, done: finished } = await reader.read();
        if (finished) break;
        buf += dec.decode(value, { stream: true });
        const { lines, rest } = splitSSEBuffer(buf);
        buf = rest;
        for (const line of lines) {
          const ev = parseSSELine(line);
          if (!ev) continue;
          if (ev.type === 'chunk') setProgress({ index: ev.index, chunks: ev.chunks });
          else if (ev.type === 'start') setProgress({ index: 0, chunks: ev.chunks });
          else if (ev.type === 'done') done = ev;
          else if (ev.type === 'error') throw new Error(ev.detail);
        }
      }
      if (!done) throw new Error(t('audiobook.annotate_no_result'));
      setResult(done);
    } catch (e) {
      setError(e?.message || String(e));
    } finally {
      setBusy(false);
      setProgress(null);
    }
  };

  const apply = () => {
    setText(result.text);
    setResult(null);
  };

  const label = busy
    ? progress && progress.chunks
      ? t('audiobook.annotate_progress', {
          current: progress.index,
          total: progress.chunks,
          elapsed: mmss(elapsed),
        })
      : t('audiobook.annotate_running')
    : t('audiobook.annotate');

  return (
    <>
      <button
        type="button"
        className={PILL}
        onClick={run}
        disabled={disabled}
        title={title}
        aria-busy={busy || undefined}
        data-testid="annotate-run"
      >
        <Sparkles size={11} aria-hidden="true" className={busy ? 'animate-pulse' : undefined} />
        {label}
      </button>

      {/* Screen readers get the same progress the label shows. */}
      <span className="sr-only" role="status" aria-live="polite">
        {busy ? label : ''}
      </span>

      {error ? (
        <span className="text-[var(--text-sm)] text-[var(--color-danger)]" role="alert">
          {error}
        </span>
      ) : null}

      {result ? (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-[24px]"
          role="dialog"
          aria-modal="true"
          aria-label={t('audiobook.annotate_preview')}
        >
          <div className="flex max-h-full w-full max-w-[720px] flex-col gap-[10px] rounded-[14px] bg-[var(--color-bg-elev-2)] p-[16px]">
            <h2 className="m-0 text-[var(--text-md)]">{t('audiobook.annotate_preview')}</h2>
            <p className="m-0 text-[var(--text-sm)] text-fg-muted">
              {t('audiobook.annotate_summary', {
                annotated: result.annotated,
                chunks: result.chunks,
              })}
            </p>
            {result.skipped.length ? (
              <p className="m-0 text-[var(--text-sm)] text-[var(--color-warn)]">
                {t('audiobook.annotate_skipped', {
                  count: result.skipped.length,
                  reason: result.skipped[0].reason,
                })}
              </p>
            ) : null}
            <textarea
              className="input-base min-h-[320px] flex-1"
              readOnly
              value={result.text}
              aria-label={t('audiobook.annotate_preview')}
              data-testid="annotate-preview"
            />
            <div className="flex justify-end gap-[8px]">
              <button type="button" className={PILL} onClick={() => setResult(null)}>
                {t('audiobook.dismiss')}
              </button>
              <button
                type="button"
                className={PILL}
                onClick={apply}
                data-testid="annotate-apply"
              >
                {t('audiobook.annotate_apply')}
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </>
  );
}
