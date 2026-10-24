import type { Ply } from "./types";

interface Props {
  plies: Ply[];
  selected: number;
  decided: Set<number>;
  illegalFrom: number | null;
  onSelect: (ply: number) => void;
}

function label(p: Ply, decided: boolean, bad: boolean): string {
  if (bad) return `Move ${p.ply}: no longer legal`;
  if (p.status === "needs_review") return `Move ${p.ply}: ${decided ? "decided" : "needs a decision"}`;
  if (p.status === "asked") return `Move ${p.ply}: you answered a question`;
  return `Move ${p.ply}: accepted automatically`;
}

/** One tick per move. Shows at a glance how far the record can be trusted. */
export function TrustStrip({ plies, selected, decided, illegalFrom, onSelect }: Props) {
  return (
    <ol className="strip" aria-label="Moves by trust">
      {plies.map((p) => {
        const bad = illegalFrom !== null && p.ply >= illegalFrom;
        const isDecided = decided.has(p.ply);
        const kind = bad ? "bad" : p.status === "needs_review" && !isDecided ? "open" : p.status === "needs_review" ? "done" : p.status;
        return (
          <li key={p.ply}>
            <button
              type="button"
              className={`tick tick-${kind}${p.ply === selected ? " is-selected" : ""}`}
              aria-label={label(p, isDecided, bad)}
              aria-current={p.ply === selected ? "true" : undefined}
              onClick={() => onSelect(p.ply)}
            />
          </li>
        );
      })}
    </ol>
  );
}
