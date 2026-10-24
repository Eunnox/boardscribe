"""Measure the pipeline on synthetic games. Nothing here touches real footage.

Metrics (all per ply unless stated):
  frames        video frames -> frames decoded -> frames kept
  first guess   top-ranked move equals the true move, before any question
  auto          moves accepted without asking; how many were wrong
  questions     asked per game (never more than 1 per move)
  final         moves equal to the truth after the user answered
  rules check   single-move corruptions that the rules engine alone would catch
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import chess
import numpy as np

from .session import SessionConfig, run_session
from .synth import SceneConfig, SyntheticGame, random_game

BASELINE = SceneConfig()
STRESS = SceneConfig(p_shake=0.9, noise_sigma=7.0)
HEAVY_BLUR = SceneConfig(p_shake=0.9, noise_sigma=4.0, blur_px_per_rad_s=45.0)
SCENES = {"baseline": BASELINE, "stress": STRESS, "heavy_blur": HEAVY_BLUR}


def _oracle(truth):
    def ask(q):
        b = chess.Board()
        for m in truth[: q.ply - 1]:
            b.push(m)
        san = b.san(truth[q.ply - 1])
        return san if san in q.options else None

    return ask


def _one(args):
    seed, plies, scene, naive = args
    g = SyntheticGame(random_game(seed, plies), seed=seed, config=scene)
    truth = list(g.moves)
    r = run_session(g, g.imu_t, g.imu_gyro, g.event_times, asker=_oracle(truth), config=SessionConfig(naive_frames=naive))
    plies_rec = r.record["plies"]
    n = len(truth)
    got = [p.get("uci") for p in plies_rec] + [None] * (n - len(plies_rec))
    tru = [m.uci() for m in truth]
    first = sum(1 for p, t in zip(plies_rec, tru) if p.get("first_guess") and p["first_guess"]["uci"] == t)
    auto = [(p, t) for p, t in zip(plies_rec, tru) if p["status"] == "auto"]
    return {
        "seed": seed,
        "plies": n,
        "total_frames": r.stats.total_frames,
        "decoded_frames": r.stats.decoded_frames,
        "selected_frames": r.stats.selected_frames,
        "degraded": len(r.stats.degraded_events),
        "first_guess_correct": first,
        "auto": len(auto),
        "auto_wrong": sum(1 for p, t in auto if p["uci"] != t),
        "asked": sum(1 for p in plies_rec if p["status"] == "asked"),
        "needs_review": sum(1 for p in plies_rec if p["status"] == "needs_review"),
        "final_correct": sum(1 for a, b in zip(got, tru) if a == b),
        "game_exact": got == tru,
    }


def rules_engine_only(seeds, plies: int) -> dict:
    """If one recorded move is wrong, how often does replaying the rules alone expose it?"""
    from .verify import replay_uci

    caught = silent = 0
    delays: list[int] = []
    for seed in seeds:
        truth = [m.uci() for m in random_game(seed, plies)]
        board = chess.Board()
        rng = np.random.default_rng(seed)
        for i, m in enumerate(truth):
            alts = [x.uci() for x in board.legal_moves if x.uci() != m]
            if alts:
                wrong = alts[int(rng.integers(len(alts)))]
                v = replay_uci(truth[:i] + [wrong] + truth[i + 1 :])
                if v.ok:
                    silent += 1
                else:
                    caught += 1
                    delays.append(v.first_bad_ply - (i + 1))
            board.push(chess.Move.from_uci(m))
    total = caught + silent
    return {
        "single_move_errors": total,
        "caught_by_rules": caught,
        "caught_rate": caught / total if total else 0.0,
        "mean_plies_until_caught": float(np.mean(delays)) if delays else 0.0,
        "silent": silent,
    }


def _summarise(rows: list[dict]) -> dict:
    n = sum(r["plies"] for r in rows)
    games = len(rows)
    auto = sum(r["auto"] for r in rows)
    sel = sum(r["selected_frames"] for r in rows)
    return {
        "games": games,
        "plies": n,
        "frames_total": sum(r["total_frames"] for r in rows),
        "frames_decoded": sum(r["decoded_frames"] for r in rows),
        "frames_kept": sel,
        "frames_per_kept": sum(r["total_frames"] for r in rows) / max(sel, 1),
        "degraded_moves": sum(r["degraded"] for r in rows),
        "first_guess_accuracy": sum(r["first_guess_correct"] for r in rows) / n,
        "auto_rate": auto / n,
        "auto_wrong": sum(r["auto_wrong"] for r in rows),
        "questions_per_game": sum(r["asked"] + r["needs_review"] for r in rows) / games,
        "questions_per_100_moves": 100 * sum(r["asked"] + r["needs_review"] for r in rows) / n,
        "needs_review_left": sum(r["needs_review"] for r in rows),
        "final_move_accuracy": sum(r["final_correct"] for r in rows) / n,
        "games_exactly_right": sum(1 for r in rows if r["game_exact"]),
    }


def run_benchmark(games: int = 20, plies: int = 30, seed0: int = 100, workers: int = 4) -> dict:
    out: dict = {"games": games, "plies_per_game": plies, "synthetic": True}
    seeds = range(seed0, seed0 + games)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for name, scene in SCENES.items():
            for mode, naive in (("selected_frames", False), ("nearest_frames", True)):
                rows = list(ex.map(_one, [(s, plies, scene, naive) for s in seeds]))
                out[f"{name}/{mode}"] = _summarise(rows)
    out["rules_engine_only"] = rules_engine_only(seeds, plies)
    return out


def format_report(res: dict) -> str:
    lines = [f"BoardScribe benchmark: {res['games']} synthetic games x {res['plies_per_game']} moves (rendered, not real footage)", ""]
    head = f"{'':28}{'selected frames':>18}{'nearest frames':>18}"
    for scene in SCENES:
        a_, b_ = res[f"{scene}/selected_frames"], res[f"{scene}/nearest_frames"]
        a, b = a_, b_
        lines += [f"[{scene}]", head]

        def row(label, key, fmt):
            lines.append(f"{label:28}{fmt(a[key]):>18}{fmt(b[key]):>18}")

        pct = lambda x: f"{100 * x:.1f}%"
        row("first guess correct", "first_guess_accuracy", pct)
        row("accepted without asking", "auto_rate", pct)
        row("  of those, wrong", "auto_wrong", lambda x: str(x))
        row("questions / 100 moves", "questions_per_100_moves", lambda x: f"{x:.1f}")
        row("final move accuracy", "final_move_accuracy", pct)
        row("games exactly right", "games_exactly_right", lambda x: f"{x}/{a['games']}")
        row("moves with no steady frame", "degraded_moves", lambda x: str(x))
        lines.append(
            f"{'frames decoded':28}{a_['frames_decoded']:>18}{b_['frames_decoded']:>18}"
        )
        lines.append(f"({a_['frames_total']} video frames, {a_['frames_kept']} kept: {a_['frames_per_kept']:.0f} video frames per kept frame)")
        lines.append("")
    r = res["rules_engine_only"]
    lines += [
        "[rules engine alone]",
        f"single wrong moves exposed by replay: {r['caught_by_rules']}/{r['single_move_errors']} ({100 * r['caught_rate']:.1f}%), "
        f"mean {r['mean_plies_until_caught']:.1f} plies later; the rest stay legal, so only evidence or a question can catch them",
    ]
    return "\n".join(lines)


def main_json(res: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
