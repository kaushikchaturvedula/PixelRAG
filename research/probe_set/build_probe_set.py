#!/usr/bin/env python3
"""Build the table/layout-stress probe set (Phase-2 required deliverable).

~15-20 deliberately table-heavy Wikipedia pages, where the fixed 1024px cut frequently slices
through tables/rows and content-aware should not. Emits:
  urls.txt              one URL per line (consumed by the `web` source / render_local.py).
  answers_template.jsonl one row per page to hand-annotate AFTER rendering: a question whose answer
                         sits in a specific table cell/row + that row's pixel bbox (answer_bbox),
                         for the answer-bearing boundary-violation metric.

Then (gated, not here): render with regions, chunk BOTH arms, and run boundary_violation.py.
Stdlib only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

WIKI = "https://en.wikipedia.org/wiki/"

# Curated table-heavy pages (lists / standings / climate / comparison / element tables).
PAGES = [
    "List_of_countries_by_GDP_(nominal)",
    "List_of_countries_and_dependencies_by_population",
    "List_of_countries_by_area",
    "List_of_chemical_elements",
    "Comparison_of_programming_languages",
    "Comparison_of_file_systems",
    "List_of_S%26P_500_companies",
    "List_of_largest_companies_by_revenue",
    "List_of_highest-grossing_films",
    "List_of_tallest_buildings",
    "List_of_Nobel_laureates_in_Physics",
    "List_of_metro_systems",
    "2022_FIFA_World_Cup",
    "List_of_Olympic_Games_host_cities",
    "Demographics_of_India",
    "List_of_countries_by_life_expectancy",
    "List_of_United_States_cities_by_population",
    "Tokyo",  # has a monthly climate table
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the table-stress probe set (urls + answer template).")
    ap.add_argument("--out", default=str(Path(__file__).parent), help="Output dir (default: research/probe_set).")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    urls = [WIKI + p for p in PAGES]
    (out / "urls.txt").write_text(
        f"# PixelRAG table/layout-stress probe set — {len(urls)} table-heavy pages\n"
        + "\n".join(urls) + "\n"
    )
    # answer-bearing template: fill `question`, `answer`, and `answer_bbox` (tile-local px of the
    # answer's table row/cell) after rendering — used by boundary_violation.py's answer-bearing metric.
    with open(out / "answers_template.jsonl", "w") as f:
        for i, p in enumerate(PAGES):
            f.write(json.dumps({
                "qid": f"probe_{i:02d}", "article_id": i, "gold_url": WIKI + p,
                "question": "", "answer": "",
                "answer_bbox": None,  # {"tile_index": int, "y": int, "height": int} — fill after render
            }, ensure_ascii=False) + "\n")

    print(f"[probe] wrote {out/'urls.txt'} ({len(urls)} pages) and {out/'answers_template.jsonl'}")
    print("[probe] next (gated): render with regions, chunk fixed + content_aware, run boundary_violation.py")


if __name__ == "__main__":
    main()
