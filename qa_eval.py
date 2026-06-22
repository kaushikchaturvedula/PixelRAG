#!/usr/bin/env python3
"""End-to-end QA eval: retrieval (flat | hier-expand) → VLM reader → grade. API-only (no GPU/serve).

Reads a `run_eval.py` results JSON (which stores flat top-k hit coords per query), resolves the
retrieved chunk **PNGs** from `--tiles-dir`, optionally expands them with the Phase-3 tree
(`research/tree.py`), sends the tile images + question to a hosted VLM reader
(`research/reader.py`), and grades the answer against gold (`research/grader.py`). Writes
`results/qa_<chunker>_<retrieval>.json` with per-query answers/verdicts + QA accuracy + token usage.

Pixel-native throughout: the reader sees only chunk images, never parsed text. Decoupled from
retrieval on purpose — retrieval runs on a GPU box (run_eval.py), QA runs anywhere and is cost-gated
separately. Validate offline first: `--reader mock --grade-method exact` (zero API spend), or
`--dry-run` to print the selected tiles per query without reader/grader at all.

The 4-cell iNat sweep (isolates chunking vs expansion):
  fixed-flat · fixed-hier-expand (reading-order neighbors) ·
  content_aware-flat · content_aware-hier-expand (section + reading-order)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "research"))
import grader as G  # noqa: E402
import reader as R  # noqa: E402
import tree as T  # noqa: E402


def _resolve_path(tiles_dir: str, article_id: int, tile_index: int, chunk_index: int) -> str:
    """Flat tile layout (mirrors serve _resolve_path): {aid}.png.tiles/chunk_XXXX_YY.png."""
    return os.path.join(
        tiles_dir,
        f"{article_id}.png.tiles",
        f"chunk_{tile_index:04d}_{chunk_index:02d}.png",
    )


def _select_tiles(hits, tree, retrieval, reader_top_k, mode, neighbors, cap):
    """Pick the (article_id, tile_index, chunk_index) tiles the reader will see.

    Both arms start from the same top `reader_top_k` flat seed hits; hier-expand then adds context
    (section siblings + reading-order neighbors, seeds-first, capped) around them via the tree.
    """
    seeds = hits[:reader_top_k]
    seed_triples = [(h["article_id"], h["tile_index"], h["chunk_index"]) for h in seeds]
    if retrieval == "hier-expand":
        return T.expand(seed_triples, tree, mode=mode, neighbors=neighbors, cap=cap)
    return seed_triples


def main() -> int:
    ap = argparse.ArgumentParser(description="PixelRAG end-to-end QA eval (reader + grader).")
    ap.add_argument("--results", required=True, help="run_eval.py results JSON (with hit coords).")
    ap.add_argument("--tiles-dir", required=True, help="Tiles dir (resolves chunk PNGs).")
    ap.add_argument("--gold", required=True, help="gold.jsonl (question/answer/reference_list).")
    ap.add_argument("--reader", default="openai", choices=list(R.PROVIDERS))
    ap.add_argument("--reader-model", default=None, help="Override reader model (default per provider).")
    ap.add_argument("--reader-max-tokens", type=int, default=512)
    ap.add_argument("--reader-detail", default="low", choices=["low", "high", "auto"],
                    help="OpenAI/qwen per-image token cap (default low ~85 tok/img; high = full "
                         "multi-tile cost). Ignored by claude/gemini — use --reader-image-maxdim there.")
    ap.add_argument("--reader-image-maxdim", type=int, default=None,
                    help="PIL-downscale each tile to this long-side max before encoding (default off; "
                         "for when low detail blurs dense tables but full-res is overkill).")
    ap.add_argument("--modality", default="image", choices=["image", "text"],
                    help="Which retrieval to read over (image = primary for iNat).")
    ap.add_argument("--retrieval", default="flat", choices=["flat", "hier-expand"])
    ap.add_argument("--reader-top-k", type=int, default=4, help="# flat seed hits the reader starts from.")
    ap.add_argument("--expand-neighbors", type=int, default=1, help="±N reading-order neighbors (hier-expand).")
    ap.add_argument("--expand-cap", type=int, default=8, help="Max tiles after expansion.")
    ap.add_argument("--expand-mode", default="auto", choices=["auto", "section+neighbors", "neighbors"],
                    help="auto = section+neighbors (degrades to neighbors-only per-article when a page "
                         "has no headings, e.g. the fixed arm).")
    ap.add_argument("--grade-method", default="judge", choices=["judge", "exact"])
    ap.add_argument("--judge-model", default=G.DEFAULT_JUDGE_MODEL)
    ap.add_argument("--limit", type=int, default=None, help="Grade only the first N queries (smoke).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print selected tiles per query; no reader/grader calls (zero spend).")
    ap.add_argument("--out", default=None, help="Default: results/qa_<chunker>_<retrieval>.json")
    args = ap.parse_args()

    results = json.loads(Path(args.results).read_text())
    chunker = results.get("chunker", "unknown")
    per_query = results.get("per_query", [])
    gold = {
        r["id"]: r
        for r in (json.loads(l) for l in Path(args.gold).read_text().splitlines() if l.strip())
    }

    tree = T.build_tree(args.tiles_dir) if args.retrieval == "hier-expand" else {}
    mode = "section+neighbors" if args.expand_mode == "auto" else args.expand_mode
    hits_key = f"hits_{args.modality}"

    rows = per_query if args.limit is None else per_query[: args.limit]
    out_rows: list[dict] = []
    verdicts: list[str] = []
    usage_tot = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    judge_usage_tot = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    n_skipped = 0
    n_tiles_total = 0

    for pq in rows:
        qid = pq.get("qid")
        hits = pq.get(hits_key)
        g = gold.get(qid, {})
        question = pq.get("question") or g.get("question", "")
        if not hits:  # no retrieval for this modality (e.g. no image query) → skip
            n_skipped += 1
            continue

        triples = _select_tiles(
            hits, tree, args.retrieval, args.reader_top_k, mode,
            args.expand_neighbors, args.expand_cap,
        )
        paths = [_resolve_path(args.tiles_dir, a, t, c) for (a, t, c) in triples]
        n_tiles_total += len(paths)

        row = {"qid": qid, "question": question, "n_tiles": len(triples),
               "tiles": [list(t) for t in triples]}

        if args.dry_run:
            out_rows.append(row)
            continue

        # real readers need the PNGs on disk; mock ignores them. Don't crash the whole sweep on a
        # missing file — record an error verdict for that query and move on.
        read_paths = paths if args.reader == "mock" else [p for p in paths if os.path.exists(p)]
        if args.reader != "mock" and not read_paths:
            row["error"] = "no tile PNGs found on disk"
            row["verdict"] = "__error__"
            out_rows.append(row)
            verdicts.append("__error__")
            continue

        # don't let one query (reader/judge rate-limit, timeout, transient 5xx) discard the whole
        # sweep — record an error verdict and keep going. aggregate() excludes __error__ from N.
        try:
            answer, usage = R.read(
                question, read_paths, provider=args.reader,
                model=args.reader_model, max_tokens=args.reader_max_tokens,
                detail=args.reader_detail, image_maxdim=args.reader_image_maxdim,
            )
            verdict = G.grade(
                question, answer, g.get("reference_list"), answer=g.get("answer"),
                method=args.grade_method, judge_model=args.judge_model,
            )
        except Exception as e:
            row["error"] = str(e)
            row["verdict"] = "__error__"
            out_rows.append(row)
            verdicts.append("__error__")
            continue

        for k in usage_tot:
            usage_tot[k] += int(usage.get(k, 0) or 0)
        ju = verdict.get("judge_usage") or {}
        for k in judge_usage_tot:
            judge_usage_tot[k] += int(ju.get(k, 0) or 0)
        row["answer"] = answer
        row["verdict"] = verdict["verdict"]
        row["ground_truth"] = verdict["ground_truth"]
        out_rows.append(row)
        verdicts.append(verdict["verdict"])

    agg = G.aggregate(verdicts)
    out = {
        "chunker": chunker,
        "retrieval": args.retrieval,
        "modality": args.modality,
        "reader": {"provider": args.reader, "model": args.reader_model or R.DEFAULT_MODELS.get(args.reader),
                   "detail": args.reader_detail, "image_maxdim": args.reader_image_maxdim},
        "grade_method": args.grade_method,
        "judge_model": args.judge_model if args.grade_method == "judge" else None,
        "config": {
            "reader_top_k": args.reader_top_k,
            "expand_neighbors": args.expand_neighbors,
            "expand_cap": args.expand_cap,
            "expand_mode": mode if args.retrieval == "hier-expand" else None,
            # effective mode: section+neighbors degrades to neighbors-only when no page has headings
            # (the fixed arm), so the recorded label matches what actually ran.
            "expand_mode_effective": (
                ("neighbors" if (mode == "section+neighbors"
                                 and not any(a.get("has_sections") for a in tree.values()))
                 else mode)
                if args.retrieval == "hier-expand" else None
            ),
        },
        "n_graded": agg["n"],
        "n_skipped": n_skipped,
        "n_tiles_total": n_tiles_total,
        "mean_tiles_per_query": round(n_tiles_total / max(1, len(out_rows)), 2),
        "accuracy": agg["accuracy"],
        "verdict_counts": {k: agg[k] for k in ("correct", "incorrect", "unattempted", "errors")},
        "reader_usage": usage_tot,
        "judge_usage": judge_usage_tot if args.grade_method == "judge" else None,
        "results_src": str(args.results),
        "per_query": out_rows,
    }

    out_path = Path(args.out) if args.out else Path(__file__).parent / "results" / f"qa_{chunker}_{args.retrieval}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"\n===== QA eval — chunker={chunker} retrieval={args.retrieval} modality={args.modality} =====")
    print(f" reader={args.reader}({out['reader']['model']})  grade={args.grade_method}"
          f"{'' if args.grade_method == 'exact' else '(' + args.judge_model + ')'}")
    if args.dry_run:
        print(f" DRY RUN — selected tiles for {len(out_rows)} queries (no reader/grader calls)")
        for r in out_rows[:5]:
            print(f"   {r['qid']}: {r['n_tiles']} tiles {r['tiles']}")
    else:
        print(f" accuracy = {agg['accuracy']}  ({agg['correct']}/{agg['n']} correct; "
              f"I={agg['incorrect']} U={agg['unattempted']} err={agg['errors']})")
        print(f" tiles/query (mean) = {out['mean_tiles_per_query']}  "
              f"reader tokens: prompt={usage_tot['prompt_tokens']} completion={usage_tot['completion_tokens']}")
        if args.grade_method == "judge":
            print(f" judge tokens: prompt={judge_usage_tot['prompt_tokens']} "
                  f"completion={judge_usage_tot['completion_tokens']}")
    print(f" skipped (no {args.modality} retrieval) = {n_skipped}")
    print(f"[qa] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
