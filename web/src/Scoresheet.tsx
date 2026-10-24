import type { Ply } from "./types";

interface Props {
  plies: Ply[];
  sans: string[]; // current san for each ply (after reviewer changes)
  selected: number;
  decided: Set<number>;
  illegalFrom: number | null;
  onSelect: (ply: number) => void;
}

function Cell({ p, san, selected, decided, bad, onSelect }: { p: Ply; san: string; selected: boolean; decided: boolean; bad: boolean; onSelect: () => void }) {
  const open = p.status === "needs_review" && !decided;
  const cls = ["cell", selected ? "is-selected" : "", open ? "is-open" : "", bad ? "is-bad" : ""].join(" ");
  return (
    <button type="button" className={cls} onClick={onSelect} aria-current={selected ? "true" : undefined}>
      <span className="san">{san}</span>
      <span className={`mark mark-${bad ? "bad" : open ? "open" : p.status === "asked" ? "asked" : p.status === "needs_review" ? "done" : "auto"}`} aria-hidden />
      <span className="conf" aria-hidden>
        <span style={{ width: `${Math.round(p.confidence * 100)}%` }} />
      </span>
    </button>
  );
}

export function Scoresheet({ plies, sans, selected, decided, illegalFrom, onSelect }: Props) {
  const rows: Ply[][] = [];
  for (let i = 0; i < plies.length; i += 2) rows.push(plies.slice(i, i + 2));
  return (
    <div className="sheet" role="group" aria-label="Scoresheet">
      <div className="sheet-head">
        <span />
        <span>White</span>
        <span>Black</span>
      </div>
      {rows.map((row, i) => (
        <div className="sheet-row" key={i}>
          <span className="no">{i + 1}</span>
          {row.map((p) => (
            <Cell
              key={p.ply}
              p={p}
              san={sans[p.ply - 1] ?? p.san ?? "?"}
              selected={p.ply === selected}
              decided={decided.has(p.ply)}
              bad={illegalFrom !== null && p.ply >= illegalFrom}
              onSelect={() => onSelect(p.ply)}
            />
          ))}
          {row.length === 1 && <span />}
        </div>
      ))}
    </div>
  );
}
