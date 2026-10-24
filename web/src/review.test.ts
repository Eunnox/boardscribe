import { describe, expect, it } from "vitest";
import { buildApproved, corrections, legalMovesAfter, replay, toPgn, unresolved, verdict } from "./review";
import type { Ply, RecordFile } from "./types";

const GAME = ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"];

function ply(n: number, uci: string, status: Ply["status"] = "auto"): Ply {
  return { ply: n, t: n, san: null, uci, status, confidence: 0.99, alternatives: [], question: null, evidence: [], degraded: false };
}

function record(statuses: Record<number, Ply["status"]> = {}): RecordFile {
  return {
    schema: "boardscribe/record@1",
    synthetic: true,
    meta: {},
    stats: { total_frames: 0, decoded_frames: 0, selected_frames: 0, degraded_events: [] },
    plies: GAME.map((u, i) => ply(i + 1, u, statuses[i + 1] ?? "auto")),
    pgn: "",
  };
}

describe("replay", () => {
  it("replays a legal game", () => {
    const r = replay(GAME);
    expect(r.firstIllegal).toBeNull();
    expect(r.sans.at(-1)).toBe("Qxf7#");
    expect(r.fens).toHaveLength(GAME.length + 1);
  });

  it("names the first illegal ply", () => {
    const r = replay(["e2e4", "e7e5", "d1h6"]);
    expect(r.firstIllegal).toBe(3);
    expect(r.fens).toHaveLength(3);
  });

  it("treats a missing move as illegal", () => {
    expect(replay(["e2e4", null]).firstIllegal).toBe(2);
  });
});

describe("chain of trust", () => {
  it("changing an early move reports the later move it breaks", () => {
    const rec = record();
    const v = verdict(rec, { choices: { 1: "d2d4" }, confirmed: [] });
    expect(v).toEqual({ kind: "illegal", ply: 3 });
  });

  it("flagged plies block approval until decided", () => {
    const rec = record({ 4: "needs_review", 6: "needs_review" });
    expect(verdict(rec, { choices: {}, confirmed: [] })).toEqual({ kind: "undecided", plies: [4, 6] });
    expect(unresolved(rec, { choices: { 4: "b8c6" }, confirmed: [6] })).toEqual([]);
  });

  it("a legal change to a late move keeps the game valid and counts as a correction", () => {
    const rec = record({ 6: "needs_review" });
    const d = { choices: { 6: "g8h6" }, confirmed: [] };
    expect(verdict(rec, d)).toEqual({ kind: "ready" });
    expect(corrections(rec, d)).toBe(1);
  });
});

describe("legalMovesAfter", () => {
  it("lists the moves available at a ply", () => {
    const moves = legalMovesAfter(GAME, 0).map((m) => m.uci);
    expect(moves).toHaveLength(20);
    expect(moves).toContain("e2e4");
  });

  it("is empty when the earlier moves are illegal", () => {
    expect(legalMovesAfter(["e2e5"], 1)).toEqual([]);
  });
});

describe("approved record", () => {
  it("records where each move came from", () => {
    const rec = record({ 4: "needs_review", 6: "needs_review" });
    const d = { choices: { 6: "g8h6" }, confirmed: [4] };
    const a = buildApproved(rec, d, 41.6);
    expect(a.plies.map((p) => p.source)).toEqual(["auto", "auto", "auto", "review_kept", "auto", "review_changed", "auto"]);
    expect(a.review).toEqual({ seconds: 42, corrections: 1, flagged_at_open: 2 });
    expect(a.plies[5].san).toBe("Nh6");
  });

  it("refuses to approve an illegal game", () => {
    expect(() => buildApproved(record(), { choices: { 1: "d2d4" }, confirmed: [] }, 5)).toThrow(/ply 3/);
  });

  it("produces a PGN with headers", () => {
    const pgn = toPgn(GAME);
    expect(pgn).toContain('[Event "BoardScribe record"]');
    expect(pgn).toContain("4. Qxf7#");
  });
});
