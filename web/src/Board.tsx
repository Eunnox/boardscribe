import { Chess } from "chess.js";
import { parseUci } from "./review";

const GLYPH: Record<string, string> = { k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟" };

interface Props {
  fen: string;
  move?: string | null; // uci of the move that led to this position
  illegal?: boolean;
}

/** White at the bottom. Squares are 10 units so the SVG scales cleanly. */
export function Board({ fen, move, illegal }: Props) {
  const grid = new Chess(fen).board();
  const m = move ? parseUci(move) : null;
  const sq = (name: string) => ({ x: (name.charCodeAt(0) - 97) * 10, y: (8 - Number(name[1])) * 10 });
  const from = m ? sq(m.from) : null;
  const to = m ? sq(m.to) : null;

  return (
    <svg className="board" viewBox="0 0 90 90" role="img" aria-label={`Board position ${fen.split(" ")[0]}`}>
      <g transform="translate(5 2)">
        {grid.map((row, r) =>
          row.map((_cell, c) => {
            const dark = (r + c) % 2 === 1;
            return <rect key={`${r}${c}`} x={c * 10} y={r * 10} width={10} height={10} className={dark ? "sq-dark" : "sq-light"} />;
          }),
        )}
        {from && <rect x={from.x} y={from.y} width={10} height={10} className="hl-from" />}
        {to && <rect x={to.x} y={to.y} width={10} height={10} className={illegal ? "hl-bad" : "hl-to"} />}
        {grid.map((row, r) =>
          row.map((cell, c) =>
            cell ? (
              <text
                key={`p${r}${c}`}
                x={c * 10 + 5}
                y={r * 10 + 7.6}
                textAnchor="middle"
                className={cell.color === "w" ? "piece-w" : "piece-b"}
              >
                {GLYPH[cell.type]}
                {"︎"}
              </text>
            ) : null,
          ),
        )}
        {"abcdefgh".split("").map((f, i) => (
          <text key={f} x={i * 10 + 5} y={83.5} textAnchor="middle" className="coord">
            {f}
          </text>
        ))}
        {[8, 7, 6, 5, 4, 3, 2, 1].map((n, i) => (
          <text key={n} x={-2.4} y={i * 10 + 6} textAnchor="middle" className="coord">
            {n}
          </text>
        ))}
      </g>
    </svg>
  );
}
