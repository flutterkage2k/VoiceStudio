import { Plus, X } from 'lucide-react';

import { Button } from '../../ui';
import LexiconScanButton from './LexiconScanButton';

/**
 * Pronunciation lexicon editor — the editable {word → respelling} rows plus the
 * Add button. Extracted from AudiobookTab so the block can live inside a
 * collapsible Section and the tab stays under the max-lines lint. Behaviour is
 * identical: the parent still owns the `lex` rows and the row mutators.
 */
function Rows({ t, lex, setLexRow, addLexRow, removeLexRow, idPrefix }) {
  return (
    <>
      {lex.map((row, i) => (
        <div key={i} className="flex gap-[6px]">
          <input
            className="input-base"
            name={`${idPrefix}-word-${i}`}
            autoComplete="off"
            placeholder={t('audiobook.lex_word')}
            value={row.word}
            onChange={setLexRow(i, 'word')}
            aria-label={t('audiobook.lex_word')}
            style={{ flex: 1, minWidth: 0 }}
          />
          <input
            className="input-base"
            name={`${idPrefix}-pronunciation-${i}`}
            autoComplete="off"
            placeholder={t('audiobook.lex_say')}
            value={row.say}
            onChange={setLexRow(i, 'say')}
            aria-label={t('audiobook.lex_say')}
            style={{ flex: 1, minWidth: 0 }}
          />
          <Button
            variant="icon"
            iconSize="sm"
            onClick={() => removeLexRow(i)}
            aria-label={t('audiobook.lex_remove')}
          >
            <X size={14} />
          </Button>
        </div>
      ))}
      <Button
        variant="subtle"
        onClick={addLexRow}
        leading={<Plus size={14} />}
        style={{ alignSelf: 'flex-start' }}
      >
        {t('audiobook.lex_add')}
      </Button>
    </>
  );
}

const GROUP_LABEL =
  'm-0 text-[var(--text-sm)] font-medium text-fg-muted';
const GROUP_NOTE = 'm-0 text-[0.66rem] leading-[1.5] text-fg-muted';

/**
 * Pronunciation lexicon editor — two groups, because they have different
 * lifetimes. "This book" travels with the project; "Every book" survives a
 * project switch, so a respelling the user always wants is typed once. The
 * book's entry wins when both name the same word.
 */
export default function LexiconEditor({
  t,
  lex,
  setLexRow,
  addLexRow,
  removeLexRow,
  globalLex,
  setGlobalLexRow,
  addGlobalLexRow,
  removeGlobalLexRow,
  scriptText,
  knownWords,
  addGlobalLexRows,
  language,
}) {
  return (
    <div className="flex flex-col gap-[10px]">
      <div className="flex flex-col gap-[6px]">
        <p className={GROUP_LABEL}>{t('audiobook.lex_scope_book')}</p>
        <Rows
          t={t}
          lex={lex}
          setLexRow={setLexRow}
          addLexRow={addLexRow}
          removeLexRow={removeLexRow}
          idPrefix="lexicon"
        />
      </div>
      <div className="flex flex-col gap-[6px] rounded-[10px] bg-[var(--color-bg-elev-1)] p-[8px]">
        <p className={GROUP_LABEL}>{t('audiobook.lex_scope_global')}</p>
        <p className={GROUP_NOTE}>{t('audiobook.lex_scope_global_hint')}</p>
        <Rows
          t={t}
          lex={globalLex}
          setLexRow={setGlobalLexRow}
          addLexRow={addGlobalLexRow}
          removeLexRow={removeGlobalLexRow}
          idPrefix="lexicon-global"
        />
        <LexiconScanButton
          t={t}
          text={scriptText}
          knownWords={knownWords}
          language={language}
          onAdd={addGlobalLexRows}
        />
      </div>
    </div>
  );
}
