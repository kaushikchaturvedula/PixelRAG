#!/usr/bin/env python3
"""Offline tests for the VLM reader + QA grader + qa_eval pipeline. ZERO API spend.

Everything runs through the `mock` reader provider and an injected mock judge, so no key is read
and no network call is made. Validates: reader mock contract, grader exact + judge (mock) +
aggregate, tile selection (flat vs hier-expand), and an end-to-end qa_eval run via subprocess
(`--reader mock --grade-method exact`, plus `--dry-run`).

Run: PYTHONPATH=research python tests/test_reader_grader.py   (or via pytest)
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # repo root → import qa_eval
sys.path.insert(0, str(ROOT / "research"))
import grader as G  # noqa: E402
import reader as R  # noqa: E402
import tree as T  # noqa: E402

import qa_eval  # noqa: E402


def _ca_chunks():
    return [
        {"tile_index": 0, "chunk_index": 0, "heading_path": "", "reading_order": 0,
         "width": 800, "height": 600},
        {"tile_index": 0, "chunk_index": 1, "heading_path": "Sec A", "reading_order": 1,
         "width": 800, "height": 600},
        {"tile_index": 0, "chunk_index": 2, "heading_path": "Sec A", "reading_order": 2,
         "width": 800, "height": 600},
        {"tile_index": 0, "chunk_index": 3, "heading_path": "Sec B", "reading_order": 3,
         "width": 800, "height": 600},
    ]


def _mk_tiles(tiles_dir: Path, aid: int, chunks):
    d = tiles_dir / f"{aid}.png.tiles"
    d.mkdir(parents=True)
    (d / "chunks.json").write_text(json.dumps({"chunks": chunks}))


# --------------------------------------------------------------------------- reader (mock)
def test_reader_mock():
    # mock must NOT read the image files (paths don't exist) and must report zero usage.
    answer, usage = R.read("What plant is this?", ["/nope/a.png", "/nope/b.png"], provider="mock")
    assert "2 tiles" in answer, answer
    assert usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    print(f"  reader mock OK -> {answer}")


# --------------------------------------------------------------------------- grader
def test_grader_exact():
    # exact-match against reference_list (normalization: case/articles/punct-insensitive)
    v_ok = G.grade("q", "When raw.", ["when raw"], method="exact")
    v_no = G.grade("q", "when cooked", ["when raw"], method="exact")
    assert v_ok["verdict"] == "correct", v_ok
    assert v_no["verdict"] == "incorrect", v_no
    print("  grader exact OK (correct + incorrect)")


def test_grader_judge_mock():
    # judge path with an INJECTED judge_fn — never calls the API.
    calls = []

    def mock_judge(question, prediction, ground_truth, model):
        calls.append((question, prediction, ground_truth, model))
        return "correct" if "artichoke" in prediction.lower() else "incorrect"

    v = G.grade("What species?", "It is a Jerusalem artichoke.",
                ["Helianthus tuberosus"], answer="Helianthus tuberosus",
                method="judge", judge_fn=mock_judge, judge_model="mock-judge")
    assert v["verdict"] == "correct", v
    assert v["ground_truth"] == "Any of: Helianthus tuberosus", v
    assert calls and calls[0][3] == "mock-judge"
    # <think> is stripped before judging
    v2 = G.grade("q", "<think>hmm</think>artichoke", ["x"], method="judge", judge_fn=mock_judge)
    assert v2["prediction"] == "artichoke", v2
    # judge_fn may also return (label, usage) like judge_openai — usage must be threaded through
    def mock_judge_with_usage(question, prediction, ground_truth, model):
        return "correct", {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}

    v3 = G.grade("q", "artichoke", ["artichoke"], method="judge", judge_fn=mock_judge_with_usage)
    assert v3["verdict"] == "correct" and v3["judge_usage"]["total_tokens"] == 14, v3
    print("  grader judge(mock) OK + <think> stripped + judge_usage threaded")


def test_grader_aggregate():
    agg = G.aggregate(["correct", "incorrect", "unattempted", "__error__"])
    assert agg == {"n": 3, "correct": 1, "incorrect": 1, "unattempted": 1,
                   "errors": 1, "accuracy": round(1 / 3, 4)}, agg
    print(f"  grader aggregate OK -> {agg}")


# --------------------------------------------------------------------------- tile selection
def test_select_tiles_flat_vs_hier():
    with tempfile.TemporaryDirectory() as d:
        tiles = Path(d)
        _mk_tiles(tiles, 0, _ca_chunks())
        tree = T.build_tree(str(tiles))
        hits = [
            {"article_id": 0, "tile_index": 0, "chunk_index": 1, "score": 0.9},
            {"article_id": 0, "tile_index": 0, "chunk_index": 3, "score": 0.5},
        ]
        flat = qa_eval._select_tiles(hits, {}, "flat", 1, "section+neighbors", 1, 8)
        assert flat == [(0, 0, 1)], flat
        hier = qa_eval._select_tiles(hits, tree, "hier-expand", 1, "section+neighbors", 1, 8)
        assert hier[0] == (0, 0, 1), "seed must come first"
        assert (0, 0, 2) in hier, "same-section sibling missing"
        assert (0, 0, 0) in hier, "reading-order neighbor missing"
        assert len(hier) > len(flat), "hier-expand should add context tiles"
        print(f"  select tiles OK -> flat={flat} hier={hier}")


# --------------------------------------------------------------------------- qa_eval (subprocess)
def _mk_fixture(tmp: Path):
    tiles = tmp / "tiles"
    _mk_tiles(tiles, 0, _ca_chunks())
    qid = "q1"
    results = {
        "chunker": "content_aware",
        "per_query": [
            {
                "qid": qid,
                "question": "What plant is shown?",
                "hits_image": [
                    {"article_id": 0, "tile_index": 0, "chunk_index": 1, "score": 0.9,
                     "url": "https://en.wikipedia.org/wiki/X"},
                    {"article_id": 0, "tile_index": 0, "chunk_index": 3, "score": 0.4,
                     "url": "https://en.wikipedia.org/wiki/X"},
                ],
                "hits_text": [
                    {"article_id": 0, "tile_index": 0, "chunk_index": 0, "score": 0.3,
                     "url": "https://en.wikipedia.org/wiki/X"},
                ],
            }
        ],
    }
    (tmp / "results.json").write_text(json.dumps(results))
    (tmp / "gold.jsonl").write_text(
        json.dumps({"id": qid, "question": "What plant is shown?",
                    "answer": "artichoke", "reference_list": ["artichoke"]}) + "\n"
    )
    return tiles


def _run_qa(tmp: Path, tiles: Path, extra: list[str]) -> dict:
    out = tmp / "out.json"
    cmd = [
        sys.executable, str(ROOT / "qa_eval.py"),
        "--results", str(tmp / "results.json"),
        "--tiles-dir", str(tiles),
        "--gold", str(tmp / "gold.jsonl"),
        "--reader", "mock",
        "--out", str(out),
        *extra,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0, f"qa_eval failed:\n{r.stdout}\n{r.stderr}"
    return json.loads(out.read_text())


def test_qa_eval_mock_flat_and_hier():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        tiles = _mk_fixture(tmp)
        # flat: mock reader + exact grader, fully offline
        flat = _run_qa(tmp, tiles, ["--retrieval", "flat", "--grade-method", "exact",
                                    "--reader-top-k", "2"])
        assert flat["n_graded"] == 1, flat
        assert flat["per_query"][0]["n_tiles"] == 2, flat
        assert flat["per_query"][0]["verdict"] in G.VERDICTS, flat
        assert isinstance(flat["accuracy"], float)
        # hier-expand: reader sees MORE tiles than flat (section + reading-order context)
        hier = _run_qa(tmp, tiles, ["--retrieval", "hier-expand", "--grade-method", "exact",
                                    "--reader-top-k", "1", "--expand-cap", "8"])
        assert hier["per_query"][0]["n_tiles"] > 1, hier
        assert hier["config"]["expand_mode"] == "section+neighbors", hier
        print(f"  qa_eval mock OK -> flat tiles={flat['per_query'][0]['n_tiles']} "
              f"hier tiles={hier['per_query'][0]['n_tiles']}")


def test_qa_eval_dry_run():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        tiles = _mk_fixture(tmp)
        out = _run_qa(tmp, tiles, ["--retrieval", "hier-expand", "--dry-run", "--reader-top-k", "1"])
        # dry-run selects tiles but does not grade
        assert out["per_query"][0]["n_tiles"] >= 1, out
        assert "answer" not in out["per_query"][0], "dry-run must not call the reader"
        print(f"  qa_eval dry-run OK -> tiles={out['per_query'][0]['tiles']}")


if __name__ == "__main__":
    test_reader_mock()
    test_grader_exact()
    test_grader_judge_mock()
    test_grader_aggregate()
    test_select_tiles_flat_vs_hier()
    test_qa_eval_mock_flat_and_hier()
    test_qa_eval_dry_run()
    print("ALL reader/grader/qa_eval tests PASSED")
