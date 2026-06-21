#!/usr/bin/env python3
"""Build a small *labeled* EncyclopedicVQA mini-corpus for the PixelRAG paper extension.

Why: the six PixelRAG benchmarks normally need the 217G+ hosted FAISS indexes. For a
self-contained, CPU-runnable baseline we instead render a handful of gold Wikipedia pages
(plus distractors) ourselves and build a tiny local index. EncyclopedicVQA is the only
"six-benchmark" loader that exposes a clean ``(question -> gold wikipedia_url)`` mapping
(eval/lib/benchmarks.py:131-158), so every query has a renderable, checkable gold page.

This script is **stdlib-only** (no pandas/datasets) so it runs anywhere. Default subset is the
EVQA **inaturalist** species pages: their query images come from iNaturalist (independent of
Wikipedia/Commons), so image-query retrieval is leak-free — unlike the landmarks subset, whose
GLDv2 query images are Wikimedia Commons photos that can overlap the gold page. (landmarks
remains available via ``--dataset-filter landmarks``.)

Outputs (under ``--out``):
  urls.txt       one URL per line (gold + distractors), consumed by the ``web`` source.
  gold.jsonl     one row per QUERY: {qid, question, answer, reference_list, gold_url, ...}
  manifest.json  full provenance: seed, counts, split, csv size, corpus URLs, params.
  pixelrag.yaml  ready for ``pixelrag index build -c <out>/pixelrag.yaml`` (chunker + cpu embed).

Download disclosed: EncyclopedicVQA ``val.csv`` is ~1.1 MB (storage.googleapis.com).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
import urllib.request
from pathlib import Path

DATASET_URLS = {
    "val": "https://storage.googleapis.com/encyclopedic-vqa/val.csv",
    "test": "https://storage.googleapis.com/encyclopedic-vqa/test.csv",
}

# Only render English Wikipedia article pages — uniform, renderable, and how EVQA gold is given.
WIKI_PREFIX = "https://en.wikipedia.org/wiki/"

SEED = 42


def _download_csv(split: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / f"encyclopedic_vqa_{split}.csv"
    if not dst.exists():
        url = DATASET_URLS[split]
        print(f"[build] downloading EVQA {split}.csv (~1.1 MB) from {url}", flush=True)
        urllib.request.urlretrieve(url, dst)
    else:
        print(f"[build] using cached {dst}", flush=True)
    return dst


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _load_candidates(csv_path: Path, dataset_filter: str, qtype_filter: set[str]) -> list[dict]:
    """Parse the CSV into candidate examples with a valid en.wikipedia gold URL.

    Mirrors eval/lib/benchmarks.py:load_encyclopedic_vqa_data's field extraction, minus the
    pandas/datasets dependency. Answers are pipe-separated; we keep the raw string + a list.
    """
    csv.field_size_limit(10_000_000)  # evidence columns are long
    out: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            url = (row.get("wikipedia_url") or "").strip()
            question = (row.get("question") or "").strip()
            if not question or not url.startswith(WIKI_PREFIX):
                continue
            dsname = (row.get("dataset_name") or "").strip().lower()
            qtype = (row.get("question_type") or "automatic").strip().lower()
            if dataset_filter and dsname != dataset_filter.lower():
                continue
            if qtype_filter and qtype not in qtype_filter:
                continue
            answer_raw = (row.get("answer") or "").strip()
            # Parse the query-image ids (pipe-separated). For landmarks these are GLDv2 ids
            # resolved to actual photos by research/fetch_query_images.py — needed for the
            # image-query (primary) retrieval metric.
            img_ids = [i.strip() for i in (row.get("dataset_image_ids") or "").split("|") if i.strip()]
            out.append(
                {
                    "id": hashlib.md5(f"{question}|{url}|{idx}".encode()).hexdigest(),
                    "question": question,
                    "question_original": (row.get("question_original") or "").strip(),
                    "answer": answer_raw,
                    "reference_list": [a.strip() for a in answer_raw.split("|") if a.strip()],
                    "gold_url": url,
                    "gold_title": (row.get("wikipedia_title") or "").strip(),
                    "dataset_name": dsname,
                    "question_type": qtype,
                    "dataset_image_ids": img_ids,
                }
            )
    return out


def build(args: argparse.Namespace) -> None:
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    csv_path = _download_csv(args.split, out / "cache")

    qtypes = {q.strip().lower() for q in args.question_types.split(",") if q.strip()}
    cands = _load_candidates(csv_path, args.dataset_filter, qtypes)
    print(
        f"[build] {len(cands)} candidate Qs "
        f"(dataset={args.dataset_filter or 'any'}, qtype={sorted(qtypes) or 'any'})",
        flush=True,
    )

    # One question per gold page → Recall is well defined. Deterministic (seed 42).
    rng = random.Random(SEED)
    by_url: dict[str, dict] = {}
    for ex in cands:
        by_url.setdefault(ex["gold_url"], ex)  # first question wins per URL
    unique = list(by_url.values())
    rng.shuffle(unique)

    if len(unique) < args.n_gold + args.n_distractors:
        sys.exit(
            f"[build] ERROR: only {len(unique)} distinct gold pages available, need "
            f"{args.n_gold + args.n_distractors}. Relax --dataset-filter/--question-types "
            f"or lower --n-gold/--n-distractors."
        )

    gold = unique[: args.n_gold]
    distractors = unique[args.n_gold : args.n_gold + args.n_distractors]

    gold_urls = [g["gold_url"] for g in gold]
    distractor_urls = [d["gold_url"] for d in distractors]
    corpus_urls = gold_urls + distractor_urls
    rng.shuffle(corpus_urls)  # avoid all-gold-first id clustering in the index

    # --- write urls.txt (consumed by the web source: one URL/line, '#' comments skipped) ---
    urls_txt = out / "urls.txt"
    with open(urls_txt, "w", encoding="utf-8") as f:
        f.write(f"# PixelRAG EVQA mini-corpus — {len(corpus_urls)} pages "
                f"({len(gold_urls)} gold + {len(distractor_urls)} distractors), seed={SEED}\n")
        for u in corpus_urls:
            f.write(u + "\n")

    # --- write gold.jsonl (one row per query; the labeled eval set) ---
    gold_jsonl = out / "gold.jsonl"
    with open(gold_jsonl, "w", encoding="utf-8") as f:
        for g in gold:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    # --- write provenance manifest ---
    manifest = {
        "dataset": "encyclopedic_vqa",
        "split": args.split,
        "seed": SEED,
        "dataset_filter": args.dataset_filter,
        "question_types": sorted(qtypes),
        "n_gold": len(gold),
        "n_distractors": len(distractor_urls),
        "n_corpus_pages": len(corpus_urls),
        "csv_bytes": csv_path.stat().st_size,
        "git_commit": _git_commit(),
        "gold_urls": gold_urls,
        "distractor_urls": distractor_urls,
        "corpus_urls": corpus_urls,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    # --- write pixelrag.yaml for `pixelrag index build` ---
    index_out = out / "output"
    yaml_text = (
        "# Auto-generated by research/build_mini_corpus.py — PixelRAG EVQA mini-corpus.\n"
        "source:\n"
        "  type: web\n"
        f"  urls_file: {urls_txt}\n"
        "ingest:\n"
        "  backend: cdp\n"
        "  quality: 85\n"
        "chunk:\n"
        f"  chunker: {args.chunker}\n"
        "embed:\n"
        "  model: Qwen/Qwen3-VL-Embedding-2B\n"
        f"  device: {args.device}\n"
        f"output: {index_out}\n"
    )
    (out / "pixelrag.yaml").write_text(yaml_text)

    print(
        f"[build] wrote:\n"
        f"  {urls_txt}  ({len(corpus_urls)} pages)\n"
        f"  {gold_jsonl}  ({len(gold)} labeled queries)\n"
        f"  {out / 'manifest.json'}\n"
        f"  {out / 'pixelrag.yaml'}  (chunker={args.chunker}, device={args.device})\n"
        f"\nNext: pixelrag index build -c {out / 'pixelrag.yaml'}",
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Build the EVQA labeled mini-corpus (stdlib only).")
    p.add_argument("--out", default=str(Path(__file__).parent / "mini_corpus"),
                   help="Output directory (default: research/mini_corpus).")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--dataset-filter", default="inaturalist",
                   help="EVQA dataset_name filter (inaturalist|landmarks|''). Default inaturalist "
                        "(leak-free query images via the iNaturalist API).")
    p.add_argument("--question-types", default="automatic",
                   help="Comma list of question_type to keep (default 'automatic'; "
                        "'' = any). 'templated' Qs rely on the query image, so are excluded.")
    p.add_argument("--n-gold", type=int, default=25, help="Number of labeled gold queries.")
    p.add_argument("--n-distractors", type=int, default=15,
                   help="Extra corpus pages that are never gold (make Recall non-trivial).")
    p.add_argument("--chunker", default="fixed", choices=["fixed"],
                   help="Chunker written into pixelrag.yaml (Phase 1: fixed baseline).")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    build(p.parse_args())


if __name__ == "__main__":
    main()
