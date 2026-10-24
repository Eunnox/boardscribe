"""Command line: demo, run, verify, export, benchmark."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Optional

import chess

from .infer import Question
from .session import SessionConfig, run_session
from .synth import SceneConfig, SyntheticGame, random_game
from .verify import replay_uci, verify_record
from .video import VideoFileSource, read_imu_csv, write_imu_csv


def _interactive_asker(q: Question) -> Optional[str]:
    print(f"\nMove {q.ply}: {q.text}")
    for i, o in enumerate(q.options, 1):
        print(f"  {i}) {o}")
    raw = input("number (Enter to skip): ").strip()
    if raw.isdigit() and 1 <= int(raw) <= len(q.options):
        return q.options[int(raw) - 1]
    return None


def cmd_demo(a: argparse.Namespace) -> int:
    scene = SceneConfig(p_shake=a.shake, noise_sigma=a.noise, blur_px_per_rad_s=a.blur)
    g = SyntheticGame(random_game(a.seed, a.plies), seed=a.seed, config=scene)
    truth = list(g.moves)
    answered = {"n": 0}

    def asker(q: Question) -> Optional[str]:
        """A demo user who answers the first `--answer-first` questions truthfully, then stops answering."""
        if answered["n"] >= a.answer_first:
            return None
        b = chess.Board()
        for m in truth[: q.ply - 1]:
            b.push(m)
        san = b.san(truth[q.ply - 1])
        if san in q.options:
            answered["n"] += 1
            return san
        return None

    out = a.out
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    r = run_session(g, g.imu_t, g.imu_gyro, g.event_times, asker=asker, out_dir=out,
                    meta={"generator": "boardscribe.synth", "seed": a.seed, "plies": a.plies, "note": "rendered, not recorded",
                          "demo_user": f"answers the first {a.answer_first} question(s) truthfully, then stops answering"})
    with open(os.path.join(out, "record.json"), "w") as f:
        json.dump(r.record, f, indent=2)
    with open(os.path.join(out, "truth.json"), "w") as f:
        json.dump({"uci": [m.uci() for m in g.moves]}, f)
    if a.mp4:
        os.makedirs(a.mp4, exist_ok=True)
        g.write_mp4(os.path.join(a.mp4, "game.mp4"))
        write_imu_csv(os.path.join(a.mp4, "imu.csv"), g.imu_t, g.imu_gyro)
        with open(os.path.join(a.mp4, "events.json"), "w") as f:
            json.dump(g.event_times, f)
        print(f"wrote a rendered video, IMU and move times to {a.mp4} (large; not for git)")
    st = [p["status"] for p in r.record["plies"]]
    print(f"wrote {out}: {len(st)} moves, auto={st.count('auto')} asked={st.count('asked')} needs_review={st.count('needs_review')}")
    return 0


def cmd_run(a: argparse.Namespace) -> int:
    src = VideoFileSource(a.video)
    imu_t, gyro = read_imu_csv(a.imu)
    with open(a.events) as f:
        events = json.load(f)
    os.makedirs(a.out, exist_ok=True)
    r = run_session(src, imu_t, gyro, events, asker=_interactive_asker if a.interactive else None, out_dir=a.out,
                    config=SessionConfig(), meta={"video": os.path.basename(a.video)})
    r.record["synthetic"] = False
    with open(os.path.join(a.out, "record.json"), "w") as f:
        json.dump(r.record, f, indent=2)
    print(r.record["pgn"])
    return 0


def cmd_verify(a: argparse.Namespace) -> int:
    with open(a.record) as f:
        rec = json.load(f)
    v = verify_record(rec, allow_unresolved=a.allow_unresolved)
    print("OK" if v.ok else "NOT OK")
    for i in v.issues:
        print(" -", i)
    return 0 if v.ok else 1


def cmd_export(a: argparse.Namespace) -> int:
    """approved.json from the review UI -> PGN, only if the rules engine accepts every move."""
    with open(a.approved) as f:
        data = json.load(f)
    ucis = [p["uci"] for p in data["plies"]]
    v = replay_uci(ucis)
    if not v.ok:
        print("refusing to export:", *v.issues, sep="\n  ", file=sys.stderr)
        return 1
    if a.output:
        with open(a.output, "w") as f:
            f.write(v.pgn + "\n")
        print(f"wrote {a.output} ({len(ucis)} moves)")
    else:
        print(v.pgn)
    return 0


def cmd_benchmark(a: argparse.Namespace) -> int:
    from .benchmark import format_report, main_json, run_benchmark

    res = run_benchmark(a.games, a.plies, a.seed0, a.workers)
    print(format_report(res))
    if a.json:
        main_json(res, a.json)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="boardscribe", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="render a synthetic game and write a review record for the UI")
    d.add_argument("--out", default="web/public/demo")
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--plies", type=int, default=24)
    d.add_argument("--shake", type=float, default=0.45)
    d.add_argument("--noise", type=float, default=2.5)
    d.add_argument("--blur", type=float, default=14.0, help="motion blur in pixels per rad/s of head speed")
    d.add_argument("--answer-first", type=int, default=1, help="how many questions the demo user answers")
    d.add_argument("--mp4", metavar="DIR", help="also write game.mp4, imu.csv, events.json into DIR (the video is large)")
    d.set_defaults(fn=cmd_demo)

    r = sub.add_parser("run", help="process a video file")
    r.add_argument("--video", required=True)
    r.add_argument("--imu", required=True, help="CSV with t,gx,gy,gz")
    r.add_argument("--events", required=True, help="JSON list of 'move done' times in seconds")
    r.add_argument("--out", required=True)
    r.add_argument("--interactive", action="store_true")
    r.set_defaults(fn=cmd_run)

    v = sub.add_parser("verify", help="replay a record through the rules engine")
    v.add_argument("record")
    v.add_argument("--allow-unresolved", action="store_true")
    v.set_defaults(fn=cmd_verify)

    e = sub.add_parser("export", help="approved.json from the review UI -> PGN")
    e.add_argument("approved")
    e.add_argument("-o", "--output")
    e.set_defaults(fn=cmd_export)

    b = sub.add_parser("benchmark", help="measure on synthetic games")
    b.add_argument("--games", type=int, default=20)
    b.add_argument("--plies", type=int, default=30)
    b.add_argument("--seed0", type=int, default=100)
    b.add_argument("--workers", type=int, default=4)
    b.add_argument("--json")
    b.set_defaults(fn=cmd_benchmark)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
