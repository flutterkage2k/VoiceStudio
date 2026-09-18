import React, { useState } from 'react';
import { Search } from 'lucide-react';
import { Button } from '../../ui';
import { suggestLexicon } from '../../api/audiobook';

/**
 * Find Latin words in a CJK script that the lexicon does not cover yet.
 *
 * The readings come back as a DRAFT the user confirms row by row — a wrong
 * respelling written silently is the same class of bug the lexicon exists to
 * fix, and how a name or loanword should be read is not something the model
 * can settle. Unchecked rows and blank readings are simply not added.
 *
 * Adds to the "every book" list: these are recurring words (an English term
 * inside Japanese narration), which is exactly what that list is for.
 */
export default function LexiconScanButton({ t, text, knownWords, language, onAdd }) {
  const [busy, setBusy] = useState(false);
  const [rows, setRows] = useState(null); // [{word, say, keep}]
  const [reason, setReason] = useState('');
  const [error, setError] = useState('');

  const scan = async () => {
    setBusy(true);
    setError('');
    setReason('');
    try {
      // Read the lists at scan time, not render time — a word added a
      // moment ago must not be offered again.
      const res = await suggestLexicon({ text, known: knownWords(), language });
      setReason(res.reason || '');
      setRows(res.words.map((w) => ({ ...w, keep: true })));
    } catch (e) {
      setError(e?.message || String(e));
    } finally {
      setBusy(false);
    }
  };

  const setRow = (i, k) => (e) =>
    setRows((rs) => rs.map((r, j) => (j === i ? { ...r, [k]: e.target.value } : r)));
  const toggle = (i) => () =>
    setRows((rs) => rs.map((r, j) => (j === i ? { ...r, keep: !r.keep } : r)));

  const add = () => {
    onAdd(
      rows
        .filter((r) => r.keep && r.word.trim() && r.say.trim())
        .map((r) => ({ word: r.word.trim(), say: r.say.trim() })),
    );
    setRows(null);
  };

  const addable = rows ? rows.filter((r) => r.keep && r.say.trim()).length : 0;

  return (
    <>
      <Button
        variant="subtle"
        onClick={scan}
        disabled={busy || !text.trim()}
        leading={<Search size={14} />}
        style={{ alignSelf: 'flex-start' }}
        data-testid="lexicon-scan"
      >
        {busy ? t('audiobook.lex_scan_running') : t('audiobook.lex_scan')}
      </Button>

      {error ? (
        <span className="text-[var(--text-sm)] text-[var(--color-danger)]" role="alert">
          {error}
        </span>
      ) : null}

      {rows ? (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-[24px]"
          role="dialog"
          aria-modal="true"
          aria-label={t('audiobook.lex_scan_found')}
        >
          <div className="flex max-h-full w-full max-w-[560px] flex-col gap-[10px] overflow-hidden rounded-[14px] bg-[var(--color-bg-elev-2)] p-[16px]">
            <h2 className="m-0 text-[var(--text-md)]">{t('audiobook.lex_scan_found')}</h2>

            {rows.length === 0 ? (
              <p className="m-0 text-[var(--text-sm)] text-fg-muted">
                {t('audiobook.lex_scan_none')}
              </p>
            ) : (
              <>
                <p className="m-0 text-[var(--text-sm)] text-fg-muted">
                  {t('audiobook.lex_scan_hint', { count: rows.length })}
                </p>
                {reason ? (
                  <p className="m-0 text-[var(--text-sm)] text-[var(--color-warn)]">
                    {t('audiobook.lex_scan_no_readings', { reason })}
                  </p>
                ) : null}
                <div className="flex flex-col gap-[6px] overflow-y-auto">
                  {rows.map((r, i) => (
                    <div key={r.word} className="flex items-center gap-[6px]">
                      <input
                        type="checkbox"
                        checked={r.keep}
                        onChange={toggle(i)}
                        aria-label={r.word}
                      />
                      <input
                        className="input-base"
                        value={r.word}
                        onChange={setRow(i, 'word')}
                        aria-label={t('audiobook.lex_word')}
                        style={{ flex: 1, minWidth: 0 }}
                      />
                      <input
                        className="input-base"
                        value={r.say}
                        onChange={setRow(i, 'say')}
                        placeholder={t('audiobook.lex_say')}
                        aria-label={t('audiobook.lex_say')}
                        style={{ flex: 1, minWidth: 0 }}
                      />
                    </div>
                  ))}
                </div>
              </>
            )}

            <div className="flex justify-end gap-[8px]">
              <Button variant="subtle" onClick={() => setRows(null)}>
                {t('audiobook.dismiss')}
              </Button>
              {rows.length ? (
                <Button
                  variant="subtle"
                  onClick={add}
                  disabled={!addable}
                  data-testid="lexicon-scan-add"
                >
                  {t('audiobook.lex_scan_add', { count: addable })}
                </Button>
              ) : null}
            </div>
          </div>
        </div>
      ) : null}
    </>
  );
}
