#!/usr/bin/env python3
"""Boundary-violation rate: how often a chunk boundary slices through a content region.

The headline mechanism metric for Phase 2 (content_aware vs fixed). For each page we have ONE
regions.json (DOM boxes, captured once at render) and each arm has its own chunks.json (different
cut lines). A region is "split" if a chunk content-boundary falls strictly inside it. We report:
  * block_split_rate — non-table blocks (p / heading / figure / li / …) split, per arm,
  * row_split_rate   — table <tr> rows split (a row is the atomic table unit), per arm,
  * answer_split_rate — answer-bearing regions split (from answers.jsonl), per arm (the paper metric).

Expectation: fixed splits many blocks/rows; content_aware ≈ 0 (it cuts at block + <tr> boundaries).
Stdlib only. Input: two tiles dirs (fixed + content_aware), each with per-article regions.json +
that arm's chunks.json (regions.json is identical in both; copied at render time).

Usage:
    python research/probe_set/boundary_violation.py \
        --fixed-dir <fixed_tiles> --ca-dir <content_aware_tiles> \
        [--answers research/probe_set/answers.jsonl] [--out results/boundary_violation.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

NON_TABLE_KINDS = {"block", "heading", "figure"}


def _scaled_regions(regions: dict) -> list[dict]:
    dpr = float(regions.get("device_pixel_ratio", 1) or 1)
    out = []
    for r in regions.get("regions", []):
        rr = {"kind": r.get("kind"), "y": r["y"] * dpr, "height": r["height"] * dpr, "id": r.get("id")}
        if r.get("rows"):
            rr["rows"] = [{"y": row["y"] * dpr, "height": row["height"] * dpr} for row in r["rows"]]
        out.append(rr)
    return out


def _tile_boundaries(chunks: list[dict], tile_height: int) -> dict[int, list[float]]:
    """Per tile_index, the sorted set of chunk content edges (tile-local px)."""
    by_tile: dict[int, set] = {}
    for c in chunks:
        ti = int(c["tile_index"])
        top = float(c["y_offset"])
        bot = top + float(c.get("bbox", {}).get("height", c["height"]))
        by_tile.setdefault(ti, set()).update((top, bot))
    return {ti: sorted(b) for ti, b in by_tile.items()}


def _is_split(y0: float, y1: float, boundaries: dict[int, list[float]], tile_height: int) -> bool:
    """True if a chunk boundary falls strictly inside [y0,y1] (absolute page px)."""
    ti = int(y0 // tile_height)
    # region spans a tile boundary → forced cut (counts as split for both arms equally)
    if int((y1 - 1) // tile_height) != ti:
        return True
    local0, local1 = y0 - ti * tile_height, y1 - ti * tile_height
    return any(local0 + 0.5 < b < local1 - 0.5 for b in boundaries.get(ti, []))


def _arm_rates(tiles_dir: Path, answers: dict) -> dict:
    blocks = blocks_split = rows = rows_split = ans = ans_split = 0
    pages = 0
    for art in sorted(tiles_dir.glob("*.png.tiles")):
        rj, cj = art / "regions.json", art / "chunks.json"
        if not (rj.exists() and cj.exists()):
            continue
        pages += 1
        regions = _scaled_regions(json.loads(rj.read_text()))
        manifest = json.loads(cj.read_text())
        th = int(manifest.get("tile_height", 8192))
        bounds = _tile_boundaries(manifest.get("chunks", []), th)
        for r in regions:
            if r.get("kind") in NON_TABLE_KINDS:
                blocks += 1
                blocks_split += _is_split(r["y"], r["y"] + r["height"], bounds, th)
            if r.get("kind") == "table":
                for row in r.get("rows", []):
                    rows += 1
                    rows_split += _is_split(row["y"], row["y"] + row["height"], bounds, th)
        # answer-bearing (keyed by article_id == dir stem)
        try:
            aid = int(art.name.split(".")[0])
        except ValueError:
            aid = None
        a = answers.get(aid)
        if a and a.get("answer_bbox"):
            bb = a["answer_bbox"]
            ti = int(bb.get("tile_index", 0))
            y0 = ti * th + float(bb["y"])
            ans += 1
            ans_split += _is_split(y0, y0 + float(bb["height"]), bounds, th)

    def rate(n, d):
        return round(n / d, 4) if d else None
    return {
        "pages": pages,
        "block_split_rate": rate(blocks_split, blocks), "blocks": blocks, "blocks_split": blocks_split,
        "row_split_rate": rate(rows_split, rows), "rows": rows, "rows_split": rows_split,
        "answer_split_rate": rate(ans_split, ans), "answers": ans, "answers_split": ans_split,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Boundary-violation rate: fixed vs content_aware.")
    ap.add_argument("--fixed-dir", required=True, help="Tiles dir with fixed chunks.json + regions.json.")
    ap.add_argument("--ca-dir", required=True, help="Tiles dir with content_aware chunks.json + regions.json.")
    ap.add_argument("--answers", default=None, help="answers.jsonl with answer_bbox (answer-bearing metric).")
    ap.add_argument("--out", default=str(Path("results") / "boundary_violation.json"))
    args = ap.parse_args()

    answers = {}
    if args.answers and Path(args.answers).exists():
        for line in Path(args.answers).read_text().splitlines():
            if line.strip():
                a = json.loads(line)
                if a.get("article_id") is not None:
                    answers[int(a["article_id"])] = a

    result = {
        "fixed": _arm_rates(Path(args.fixed_dir).resolve(), answers),
        "content_aware": _arm_rates(Path(args.ca_dir).resolve(), answers),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print("===== boundary-violation rate (lower = better) =====")
    for arm in ("fixed", "content_aware"):
        r = result[arm]
        print(f" {arm:14} block={r['block_split_rate']} ({r['blocks_split']}/{r['blocks']})  "
              f"row={r['row_split_rate']} ({r['rows_split']}/{r['rows']})  "
              f"answer={r['answer_split_rate']} ({r['answers_split']}/{r['answers']})")
    print(f"\n[boundary] wrote {out}")


if __name__ == "__main__":
    main()
