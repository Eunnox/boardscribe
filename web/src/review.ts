import { Chess } from "chess.js";
import type { ApprovedRecord, ApprovedPly, RecordFile } from "./types";

export interface Replay {
  /** fens[k] is the position after k plies; it stops before the first illegal ply. */
  fens: string[];
  sans: string[];
  /** 1-based ply number of the first move that is not legal, or null. */
  firstIllegal: number | null;
}

export function parseUci(uci: string): { from: string; to: string; promotion?: string } {
  return { from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci.length > 4 ? uci[4] : undefined };
}

export function replay(ucis: (string | null)[]): Replay {
  const chess = new Chess();
  const fens = [chess.fen()];
  const sans: string[] = [];
  for (let i = 0; i < ucis.length; i++) {
    const u = ucis[i];
    if (!u) return { fens, sans, firstIllegal: i + 1 };
    try {
      const mv = chess.move(parseUci(u));
      sans.push(mv.san);
      fens.push(chess.fen());
    } catch {
      return { fens, sans, firstIllegal: i + 1 };
    }
  }
  return { fens, sans, firstIllegal: null };
}

export interface LegalMove {
  san: string;
  uci: string;
}

export function legalMovesAfter(ucis: (string | null)[], plyCount: number): LegalMove[] {
  const r = replay(ucis.slice(0, plyCount));
  if (r.firstIllegal !== null) return [];
  const chess = new Chess(r.fens[r.fens.length - 1]);
  return chess.moves({ verbose: true }).map((m) => ({ san: m.san, uci: m.from + m.to + (m.promotion ?? "") }));
}

export interface Decisions {
  /** ply -> uci chosen by the reviewer */
  choices: Record<number, string>;
  /** plies flagged for review that the reviewer looked at and kept */
  confirmed: number[];
}

export function currentUcis(record: RecordFile, d: Decisions): (string | null)[] {
  return record.plies.map((p) => d.choices[p.ply] ?? p.uci);
}

export function unresolved(record: RecordFile, d: Decisions): number[] {
  return record.plies
    .filter((p) => p.status === "needs_review" && !d.confirmed.includes(p.ply) && d.choices[p.ply] === undefined)
    .map((p) => p.ply);
}

export function corrections(record: RecordFile, d: Decisions): number {
  return record.plies.filter((p) => d.choices[p.ply] !== undefined && d.choices[p.ply] !== p.uci).length;
}

export type Verdict =
  | { kind: "ready" }
  | { kind: "undecided"; plies: number[] }
  | { kind: "illegal"; ply: number };

export function verdict(record: RecordFile, d: Decisions): Verdict {
  const r = replay(currentUcis(record, d));
  if (r.firstIllegal !== null) return { kind: "illegal", ply: r.firstIllegal };
  const open = unresolved(record, d);
  if (open.length) return { kind: "undecided", plies: open };
  return { kind: "ready" };
}

export function buildApproved(record: RecordFile, d: Decisions, seconds: number): ApprovedRecord {
  const ucis = currentUcis(record, d);
  const r = replay(ucis);
  if (r.firstIllegal !== null) throw new Error(`ply ${r.firstIllegal} is not legal`);
  const plies: ApprovedPly[] = record.plies.map((p, i) => {
    const changed = d.choices[p.ply] !== undefined && d.choices[p.ply] !== p.uci;
    let source: ApprovedPly["source"];
    if (p.status === "needs_review") source = changed ? "review_changed" : "review_kept";
    else source = changed ? "corrected" : p.status;
    return { ply: p.ply, uci: ucis[i] as string, san: r.sans[i], source };
  });
  return {
    schema: "boardscribe/approved@1",
    synthetic: record.synthetic,
    plies,
    review: {
      seconds: Math.round(seconds),
      corrections: corrections(record, d),
      flagged_at_open: record.plies.filter((p) => p.status === "needs_review").length,
    },
  };
}

export function toPgn(ucis: string[]): string {
  const chess = new Chess();
  chess.header("Event", "BoardScribe record", "Site", "?", "Date", "????.??.??", "Result", "*");
  for (const u of ucis) chess.move(parseUci(u));
  return chess.pgn();
}
