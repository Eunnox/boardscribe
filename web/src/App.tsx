import { useEffect, useMemo, useState } from "react";
import { Board } from "./Board";
import { Scoresheet } from "./Scoresheet";
import { TrustStrip } from "./TrustStrip";
import { buildApproved, corrections, currentUcis, legalMovesAfter, parseUci, replay, toPgn, verdict, type Decisions } from "./review";
import type { ApprovedRecord, Ply, RecordFile } from "./types";

const BASE = "demo/";

function download(name: string, text: string, type: string) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

const pct = (x: number) => `${Math.round(x * 1000) / 10}%`;

function statusLine(p: Ply, decided: boolean, changed: boolean): string {
  if (changed) return "You changed this move.";
  if (p.status === "needs_review") {
    if (decided) return "You checked this move.";
    const why = (p.note ?? "").trim();
    return why ? `Needs your decision. ${why.charAt(0).toUpperCase()}${why.slice(1)}.` : "Needs your decision.";
  }
  if (p.status === "asked") return "A question was asked and answered for this move.";
  return `Accepted without asking, ${pct(p.confidence)} sure.`;
}

export function App() {
  const [record, setRecord] = useState<RecordFile | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState(1);
  const [dec, setDec] = useState<Decisions>({ choices: {}, confirmed: [] });
  const [frame, setFrame] = useState(0);
  const [openedAt] = useState(() => Date.now());
  const [approved, setApproved] = useState<ApprovedRecord | null>(null);

  useEffect(() => {
    fetch(`${BASE}record.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`record.json: ${r.status}`))))
      .then((rec: RecordFile) => {
        setRecord(rec);
        const first = rec.plies.find((p) => p.status === "needs_review");
        setSelected(first ? first.ply : 1);
      })
      .catch((e) => setError(String(e.message ?? e)));
  }, []);

  useEffect(() => setFrame(0), [selected]);

  const ucis = useMemo(() => (record ? currentUcis(record, dec) : []), [record, dec]);
  const rep = useMemo(() => replay(ucis), [ucis]);
  const v = useMemo(() => (record ? verdict(record, dec) : { kind: "ready" as const }), [record, dec]);
  const illegalFrom = rep.firstIllegal;

  useEffect(() => {
    if (!record) return;
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t.tagName === "SELECT" || t.tagName === "INPUT") return;
      if (e.key === "ArrowRight") setSelected((s) => Math.min(record.plies.length, s + 1));
      if (e.key === "ArrowLeft") setSelected((s) => Math.max(1, s - 1));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [record]);

  if (error) {
    return (
      <main className="app">
        <h1>Review a recorded game</h1>
        <p className="error">Could not load the record. {error}. Run <code>python -m boardscribe demo</code> first, then reload.</p>
      </main>
    );
  }
  if (!record) return <main className="app"><p>Loading the record.</p></main>;

  const p = record.plies[selected - 1];
  const decidedSet = new Set<number>([...dec.confirmed, ...Object.keys(dec.choices).map(Number)]);
  const nAuto = record.plies.filter((x) => x.status === "auto").length;
  const nAsked = record.plies.filter((x) => x.status === "asked").length;
  const nFlag = record.plies.filter((x) => x.status === "needs_review").length;
  const sans = record.plies.map((x, i) => rep.sans[i] ?? x.san ?? "?");

  const currentUci = ucis[selected - 1];
  const changed = dec.choices[p.ply] !== undefined && dec.choices[p.ply] !== p.uci;
  const posIndex = Math.min(selected, rep.fens.length - 1);
  const fen = rep.fens[posIndex];
  const shownMove = selected <= rep.fens.length - 1 ? currentUci : null;
  const here = illegalFrom !== null && selected >= illegalFrom;

  const candidates: { uci: string; san: string; p: number | null }[] = [];
  const addCand = (uci: string | null, san: string, prob: number | null) => {
    if (uci && !candidates.some((c) => c.uci === uci)) candidates.push({ uci, san, p: prob });
  };
  addCand(p.uci, p.san ?? "?", p.first_guess && p.first_guess.uci === p.uci ? p.first_guess.p : p.confidence);
  p.alternatives.forEach((a) => addCand(a.uci, a.san, a.p));
  if (p.first_guess) addCand(p.first_guess.uci, p.first_guess.san, p.first_guess.p);

  const legal = legalMovesAfter(ucis, selected - 1);
  const frames = p.evidence;
  const recorded = p.uci ? parseUci(p.uci) : null;

  const choose = (uci: string) => setDec((d) => ({ ...d, choices: { ...d.choices, [p.ply]: uci }, confirmed: d.confirmed.filter((x) => x !== p.ply) }));
  const undo = () => setDec((d) => { const c = { ...d.choices }; delete c[p.ply]; return { ...d, choices: c }; });
  const keep = () => setDec((d) => ({ ...d, confirmed: d.confirmed.includes(p.ply) ? d.confirmed : [...d.confirmed, p.ply] }));

  const approve = () => setApproved(buildApproved(record, dec, (Date.now() - openedAt) / 1000));
  const blocked = v.kind !== "ready";
  const blockedText =
    v.kind === "undecided" ? `${v.plies.length} ${v.plies.length === 1 ? "move needs" : "moves need"} a decision` : v.kind === "illegal" ? `Move ${v.ply} is no longer legal` : "";

  return (
    <main className="app">
      <header className="top">
        <div>
          <h1>Review a recorded game</h1>
          <p className="sub">
            {record.plies.length} moves: {nAuto} accepted, {nAsked} answered, {nFlag} flagged for you.{" "}
            <span className="demo-note">Rendered demo game, not real footage.</span>
          </p>
        </div>
        <div className="approve">
          <button type="button" className="primary" disabled={blocked || approved !== null} onClick={approve}>
            Approve game
          </button>
          {blocked && <span className="why" role="status">{blockedText}</span>}
        </div>
      </header>

      <TrustStrip plies={record.plies} selected={selected} decided={decidedSet} illegalFrom={illegalFrom} onSelect={setSelected} />

      {approved && (
        <section className="approved" aria-live="polite">
          <p>
            Approved after {approved.review.seconds} s with {approved.review.corrections} {approved.review.corrections === 1 ? "correction" : "corrections"}. Every move was replayed
            by the rules engine.
          </p>
          <div className="row">
            <button type="button" onClick={() => download("game.pgn", toPgn(approved.plies.map((x) => x.uci)) + "\n", "application/x-chess-pgn")}>Download PGN</button>
            <button type="button" onClick={() => download("approved.json", JSON.stringify(approved, null, 2), "application/json")}>Download approved.json</button>
          </div>
          <p className="hint">Command line check: <code>boardscribe export approved.json -o game.pgn</code></p>
        </section>
      )}

      <div className="grid">
        <section className="col-sheet" aria-label="Moves">
          <Scoresheet plies={record.plies} sans={sans} selected={selected} decided={decidedSet} illegalFrom={illegalFrom} onSelect={setSelected} />
          <ul className="legend">
            <li><span className="mark mark-auto" />Accepted</li>
            <li><span className="mark mark-asked" />Answered</li>
            <li><span className="mark mark-open" />Needs a decision</li>
            <li><span className="mark mark-done" />Decided</li>
          </ul>
        </section>

        <section className="col-board" aria-label="Board">
          <Board fen={fen} move={shownMove} illegal={here} />
          <div className="row nav">
            <button type="button" onClick={() => setSelected((s) => Math.max(1, s - 1))} disabled={selected === 1}>Previous move</button>
            <button type="button" onClick={() => setSelected((s) => Math.min(record.plies.length, s + 1))} disabled={selected === record.plies.length}>Next move</button>
          </div>
        </section>

        <section className="col-evidence" aria-label="Evidence">
          <h2>
            <span className="who">{p.ply % 2 ? "White" : "Black"}, move {Math.ceil(p.ply / 2)}</span>
            <span className="big">{sans[p.ply - 1]}</span>
          </h2>
          <p className={`status ${p.status === "needs_review" && !decidedSet.has(p.ply) ? "is-open" : ""}`}>{statusLine(p, decidedSet.has(p.ply), changed)}</p>
          {p.first_guess && p.first_guess.uci !== p.uci && <p className="note">First guess was {p.first_guess.san} ({pct(p.first_guess.p)}).</p>}
          {p.question && (
            <p className="note">
              Question: {p.question.options.join(" or ")}. {p.question.answer ? `Answer: ${p.question.answer}.` : "No answer yet."}
            </p>
          )}
          {p.degraded && <p className="note">No steady frame near this move; the calmest one was used.</p>}

          <figure className="frames">
            {frames.length ? (
              <>
                <img className="main-frame" src={`${BASE}${frames[Math.min(frame, frames.length - 1)].frame}`} alt={`Evidence frame ${frame + 1} for move ${p.ply}`} />
                <figcaption>
                  Frame at {frames[Math.min(frame, frames.length - 1)].t.toFixed(1)} s, head speed {frames[Math.min(frame, frames.length - 1)].head_speed.toFixed(2)} rad/s.
                  {recorded && ` Outlined: ${recorded.from} and ${recorded.to} (recorded move).`}
                </figcaption>
                <div className="thumbs">
                  {frames.map((f, i) => (
                    <button key={f.frame} type="button" className={i === frame ? "is-on" : ""} onClick={() => setFrame(i)} aria-label={`Show frame ${i + 1}`}>
                      <img src={`${BASE}${f.frame}`} alt="" />
                    </button>
                  ))}
                </div>
              </>
            ) : (
              <p className="note">No frame was usable for this move.</p>
            )}
          </figure>

          <div className="decide">
            <h3>Which move was it?</h3>
            <div className="choices" role="radiogroup" aria-label="Candidate moves">
              {candidates.map((c) => (
                <button key={c.uci} type="button" role="radio" aria-checked={currentUci === c.uci} className={currentUci === c.uci ? "is-on" : ""} onClick={() => choose(c.uci)}>
                  <span>{c.san}</span>
                  {c.p !== null && <small>{pct(c.p)}</small>}
                </button>
              ))}
            </div>
            <label className="other">
              Another legal move
              <select value="" onChange={(e) => e.target.value && choose(e.target.value)}>
                <option value="">Choose</option>
                {legal.map((m) => <option key={m.uci} value={m.uci}>{m.san}</option>)}
              </select>
            </label>
            <div className="row">
              {p.status === "needs_review" && !decidedSet.has(p.ply) && <button type="button" className="primary" onClick={keep}>Keep {sans[p.ply - 1]}</button>}
              {dec.choices[p.ply] !== undefined && <button type="button" onClick={undo}>Undo my change</button>}
            </div>
            <p className={`consequence ${illegalFrom !== null ? "is-bad" : ""}`} role="status">
              {illegalFrom !== null
                ? `Move ${illegalFrom} (${record.plies[illegalFrom - 1].san ?? "?"}) is not legal in this game. Undo the change or fix that move.`
                : corrections(record, dec) > 0
                  ? "The corrected game is still legal."
                  : "The record replays as a legal game."}
            </p>
          </div>
        </section>
      </div>
    </main>
  );
}
