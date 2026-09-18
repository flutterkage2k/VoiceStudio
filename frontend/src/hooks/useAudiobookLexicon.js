import { useEffect, useRef, useState } from 'react';
import { useAppStore } from '../store';

/** Rows → the {word → say} dict, dropping half-typed rows. */
const toDict = (rows) =>
  Object.fromEntries(
    rows.filter((r) => r.word.trim() && r.say.trim()).map((r) => [r.word.trim(), r.say.trim()]),
  );

const toRows = (dict) => Object.entries(dict || {}).map(([word, say]) => ({ word, say }));

/**
 * Pronunciation-lexicon editor state for the Audiobook tab (extracted from
 * AudiobookTab so the page stays under the max-lines lint, #1217).
 *
 * Two lists, because they have different lifetimes:
 *
 * * **book** — travels with the project. `loadProject`/`newProject` replace it,
 *   which is right for a respelling that only makes sense in one book.
 * * **global** — survives every project switch. A respelling the user will
 *   always want (an English word inside Japanese narration, a recurring proper
 *   noun) used to have to be retyped for every new book.
 *
 * `lexDict()` merges them with the book winning on a conflicting word, so a
 * book can still override a global respelling.
 *
 * Rows stay LOCAL (half-typed rows aren't junk-persisted); each filtered dict
 * flushes to the store, and hydrates back on mount.
 */
export function useAudiobookLexicon() {
  const setLexiconStore = useAppStore((s) => s.setLexicon);
  const storeLexicon = useAppStore((s) => s.lexicon);
  const setGlobalStore = useAppStore((s) => s.setGlobalLexicon);
  const storeGlobal = useAppStore((s) => s.globalLexicon);

  const [lex, setLex] = useState([]); // [{ word, say }] — this book
  const [globalLex, setGlobalLex] = useState([]); // [{ word, say }] — every book
  const hydrated = useRef(false);

  useEffect(() => {
    if (hydrated.current) return;
    hydrated.current = true;
    const rows = toRows(storeLexicon);
    if (rows.length) setLex(rows);
    const grows = toRows(storeGlobal);
    if (grows.length) setGlobalLex(grows);
  }, [storeLexicon, storeGlobal]);

  /** What the render actually gets: global first, book overrides it. */
  const lexDict = () => ({ ...toDict(globalLex), ...toDict(lex) });

  useEffect(() => {
    if (!hydrated.current) return;
    setLexiconStore(toDict(lex));
  }, [lex]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!hydrated.current) return;
    setGlobalStore(toDict(globalLex));
  }, [globalLex]); // eslint-disable-line react-hooks/exhaustive-deps

  const rowSetter = (setRows) => (i, k) => (e) =>
    setRows((rows) => rows.map((r, j) => (j === i ? { ...r, [k]: e.target.value } : r)));

  /** Bulk-add confirmed rows to the global list, skipping words it already has. */
  const addGlobalLexRows = (rows) =>
    setGlobalLex((cur) => {
      const have = new Set(cur.map((r) => r.word.trim().toLowerCase()));
      const fresh = rows.filter((r) => !have.has(r.word.trim().toLowerCase()));
      return [...cur, ...fresh.map((r) => ({ word: r.word, say: r.say }))];
    });

  /** Every word already covered by either list — the scan skips these. */
  const knownWords = () => [
    ...Object.keys(toDict(lex)),
    ...Object.keys(toDict(globalLex)),
  ];

  return {
    lex,
    globalLex,
    addGlobalLexRows,
    knownWords,
    lexDict,
    setLexRow: rowSetter(setLex),
    addLexRow: () => setLex((rows) => [...rows, { word: '', say: '' }]),
    removeLexRow: (i) => setLex((rows) => rows.filter((_, j) => j !== i)),
    setGlobalLexRow: rowSetter(setGlobalLex),
    addGlobalLexRow: () => setGlobalLex((rows) => [...rows, { word: '', say: '' }]),
    removeGlobalLexRow: (i) => setGlobalLex((rows) => rows.filter((_, j) => j !== i)),
  };
}
