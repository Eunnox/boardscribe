import json
import os

import pytest

from boardscribe.benchmark import _one, format_report, rules_engine_only
from boardscribe.cli import main
from boardscribe.synth import SceneConfig

DEMO = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web", "public", "demo", "record.json")

PLY_KEYS = {"ply", "t", "san", "uci", "fen_after", "status", "confidence", "note", "first_guess",
            "alternatives", "question", "evidence", "degraded"}


def test_demo_command_writes_a_record_the_ui_can_read(tmp_path, capsys):
    out = tmp_path / "demo"
    assert main(["demo", "--out", str(out), "--seed", "3", "--plies", "6"]) == 0
    rec = json.loads((out / "record.json").read_text())
    assert rec["schema"] == "boardscribe/record@1" and rec["synthetic"] is True
    assert len(rec["plies"]) == 6
    assert (out / "frames").is_dir() and any((out / "frames").iterdir())
    assert main(["verify", str(out / "record.json"), "--allow-unresolved"]) == 0


def test_shipped_demo_record_matches_what_the_ui_expects():
    rec = json.load(open(DEMO))
    assert rec["synthetic"] is True
    assert {"schema", "synthetic", "meta", "stats", "plies", "pgn"} <= set(rec)
    for p in rec["plies"]:
        assert PLY_KEYS <= set(p), PLY_KEYS - set(p)
        assert p["status"] in {"auto", "asked", "needs_review"}
        for e in p["evidence"]:
            assert os.path.exists(os.path.join(os.path.dirname(DEMO), e["frame"]))
    assert any(p["status"] == "needs_review" for p in rec["plies"])  # the demo has something to review


def test_export_writes_pgn_for_a_legal_game_and_refuses_an_illegal_one(tmp_path, capsys):
    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps({"plies": [{"uci": u} for u in ["e2e4", "e7e5", "g1f3"]]}))
    out = tmp_path / "game.pgn"
    assert main(["export", str(ok), "-o", str(out)]) == 0
    assert "1. e4 e5 2. Nf3" in out.read_text()

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"plies": [{"uci": "e2e4"}, {"uci": "e7e5"}, {"uci": "d1h6"}]}))
    assert main(["export", str(bad), "-o", str(tmp_path / "x.pgn")]) == 1
    assert not (tmp_path / "x.pgn").exists()
    assert "ply 3" in capsys.readouterr().err


def test_verify_exit_codes(tmp_path):
    rec = tmp_path / "r.json"
    rec.write_text(json.dumps({"plies": [{"ply": 1, "uci": "e2e4", "status": "needs_review"}]}))
    assert main(["verify", str(rec)]) == 1
    assert main(["verify", str(rec), "--allow-unresolved"]) == 0


def test_rules_engine_only_counts_are_consistent():
    r = rules_engine_only(range(3), 12)
    assert r["caught_by_rules"] + r["silent"] == r["single_move_errors"] > 0
    assert 0.0 < r["caught_rate"] < 1.0


def test_one_benchmark_game_and_report_format():
    row = _one((11, 6, SceneConfig(), False))
    assert row["plies"] == 6 and row["final_correct"] <= 6 and row["selected_frames"] <= 21
    summary = {k: row[k] for k in row}
    assert summary["total_frames"] > summary["selected_frames"]
    rows = [row]
    from boardscribe.benchmark import _summarise

    s = _summarise(rows)
    res = {"games": 1, "plies_per_game": 6, "synthetic": True, "rules_engine_only": rules_engine_only([11], 6)}
    for scene in ("baseline", "stress", "heavy_blur"):
        res[f"{scene}/selected_frames"] = s
        res[f"{scene}/nearest_frames"] = s
    text = format_report(res)
    assert "rendered, not real footage" in text and "[heavy_blur]" in text


def test_interactive_asker_picks_option_or_skips(monkeypatch):
    from boardscribe.cli import _interactive_asker
    from boardscribe.infer import Question

    q = Question(5, "Which move was that?", ("Nbd2", "Nfd2"))
    monkeypatch.setattr("builtins.input", lambda _="": "2")
    assert _interactive_asker(q) == "Nfd2"
    monkeypatch.setattr("builtins.input", lambda _="": "")
    assert _interactive_asker(q) is None
