#!/usr/bin/env python3
"""Build a small *labeled* NQ-Tables closed mini-corpus for the PixelRAG paper extension.

Phase-4 pilot. Generalizes research/build_mini_corpus.py (the iNat EVQA builder) to the paper's
NQ-Tables benchmark: render a set of gold Wikipedia pages (plus distractors) ourselves and build a
tiny local index, instead of needing the ~570G hosted FAISS indexes. NQ-Tables exposes a clean
(question -> gold Wikipedia URL) mapping via the table's `documentUrl`, so every query has a
renderable, checkable gold page — the same property build_mini_corpus.py relied on for EVQA.

Reuses the gold-QA + URL-resolution logic of eval/lib/simpleqa_data.py:load_nq_tables_data
(replicated stdlib-only here, with attribution, so the builder runs anywhere — same pattern as
build_mini_corpus.py replicating the EVQA loader). NQ-Tables questions are **self-contained text**
("What … in <entity>?"), so unlike iNat's deictic image queries they suit text-query retrieval —
which Phase-4 needs (all benchmarks are text-only).

Outputs (under --out, default research/benchmarks/nq_tables/):
  cache/dev.jsonl   raw NQ-Tables interactions dev split (gitignored; regenerate, ~9 MB GCS)
  urls.txt          one URL/line (gold + distractors), consumed by the `web` source
  gold.jsonl        one row per QUERY: {id, question, answer, reference_list, gold_url, ...}
                    (same schema as research/mini_corpus/gold.jsonl, dataset_name=nq_tables)
  manifest.json     provenance + counts (candidates / dropped / gold / distractor; render filled later)
  pixelrag.yaml     fixed-chunker index config   (-> output/)
  pixelrag_ca.yaml  content_aware-chunker config (-> output_ca/)

Download disclosed: NQ-Tables dev.jsonl is ~9 MB (storage.googleapis.com, public GCS).

Grading note: NQ-Tables uses EXACT-MATCH (SQuAD-normalize), NOT the VLM judge — see
eval/lib/grader.py EXACT_MATCH_TASKS; qa_eval.py --grade-method exact at Step 5.
"""

from __future__ import annotations

import argparse
import hashlib
import html as _html
import json
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

# Public GCS interactions dev split (table subset of Natural Questions). Source matches
# eval/lib/simpleqa_data.py:load_nq_tables_data.
DATASET_URL = "https://storage.googleapis.com/tapas_models/2021_07_22/nq_tables/interactions/dev.jsonl"
WIKI_PREFIX = "https://en.wikipedia.org/wiki/"
SEED = 42


def _download(url: str, dst: Path, approx: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print(f"[build] using cached {dst}", flush=True)
        return
    print(f"[build] downloading NQ-Tables dev.jsonl ({approx}) from {url}", flush=True)
    urllib.request.urlretrieve(url, dst)


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _resolve_url(raw: str) -> str | None:
    """NQ documentUrl -> clean https://en.wikipedia.org/wiki/Title, or None if not resolvable.

    Replicates eval/lib/simpleqa_data.py:load_nq_tables_data URL normalization:
    html-unescape, then map /w/index.php?title=Foo&oldid=123 -> /wiki/Foo. Keep only en.wikipedia
    article pages (uniform + renderable, same constraint as the EVQA builder's WIKI_PREFIX).
    """
    if not raw:
        return None
    url = _html.unescape(raw)
    m = re.search(r"[?&]title=([^&]+)", url)
    if m:
        url = WIKI_PREFIX + urllib.parse.quote(m.group(1), safe="/:(),-")
    # collapse accidental double slashes in the host part (NQ urls sometimes have en.wikipedia.org//w/)
    url = re.sub(r"(?<!:)//+", "/", url)
    if not url.startswith(WIKI_PREFIX):
        return None
    # reject non-article namespaces that won't render as a normal page
    tail = url[len(WIKI_PREFIX):]
    if not tail or tail.startswith(("File:", "Special:", "Help:", "Template:", "Category:")):
        return None
    return url


def _load_candidates(dev_path: Path) -> tuple[list[dict], dict]:
    """Parse dev.jsonl into candidates with a resolvable gold URL. Returns (candidates, drop_stats)."""
    drops = {"no_question": 0, "no_answer": 0, "no_url": 0, "total_lines": 0}
    out: list[dict] = []
    with open(dev_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            drops["total_lines"] += 1
            ex = json.loads(line)
            questions = ex.get("questions", [])
            if not questions:
                drops["no_question"] += 1
                continue
            q = questions[0]
            question_text = (q.get("originalText") or "").strip()
            answer_texts = q.get("answer", {}).get("answerTexts", []) or []
            gold_answers = [a.strip() for a in answer_texts if a and a.strip()]
            if not question_text:
                drops["no_question"] += 1
                continue
            if not gold_answers:
                drops["no_answer"] += 1
                continue
            table = ex.get("table", {})
            url = _resolve_url(table.get("documentUrl", ""))
            if not url:
                drops["no_url"] += 1
                continue
            out.append(
                {
                    "id": str(ex.get("id") or hashlib.md5(f"{question_text}|{url}".encode()).hexdigest()),
                    "question": question_text,
                    "answer": " | ".join(gold_answers),
                    "reference_list": gold_answers,
                    "gold_url": url,
                    "gold_title": (table.get("documentTitle") or "").strip(),
                    "dataset_name": "nq_tables",
                    "question_type": "table",
                    "table_id": (table.get("tableId") or "").strip(),
                }
            )
    return out, drops


def _yaml(out: Path, urls_txt: Path, chunker: str, index_out: Path) -> str:
    return (
        "# Auto-generated by research/benchmarks/build_nqt_corpus.py — PixelRAG NQ-Tables mini-corpus.\n"
        "source:\n  type: web\n"
        f"  urls_file: {urls_txt}\n"
        "ingest:\n  backend: cdp\n  quality: 85\n"
        f"chunk:\n  chunker: {chunker}\n"
        "embed:\n  model: Qwen/Qwen3-VL-Embedding-2B\n  device: cuda\n"
        f"output: {index_out}\n"
    )


def build(args: argparse.Namespace) -> None:
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    dev_path = out / "cache" / "dev.jsonl"
    _download(DATASET_URL, dev_path, approx="~9 MB")

    cands, drops = _load_candidates(dev_path)
    print(
        f"[build] {drops['total_lines']} lines -> {len(cands)} candidates with resolvable gold URL "
        f"(dropped: no_question={drops['no_question']} no_answer={drops['no_answer']} no_url={drops['no_url']})",
        flush=True,
    )

    # One question per gold page (Recall well-defined), deterministic (seed 42) — mirrors build_mini_corpus.
    import random

    rng = random.Random(SEED)
    by_url: dict[str, dict] = {}
    for ex in cands:
        by_url.setdefault(ex["gold_url"], ex)  # first question per page wins
    unique = list(by_url.values())
    rng.shuffle(unique)
    print(f"[build] {len(unique)} unique gold pages (deduped by URL)", flush=True)

    need = args.n_gold + args.n_distractors
    if len(unique) < need:
        sys.exit(
            f"[build] ERROR: only {len(unique)} unique resolvable pages, need {need} "
            f"(n_gold {args.n_gold} + n_distractors {args.n_distractors}). Lower the counts."
        )

    gold = unique[: args.n_gold]
    distractors = unique[args.n_gold : need]
    gold_urls = [g["gold_url"] for g in gold]
    distractor_urls = [d["gold_url"] for d in distractors]
    assert not (set(gold_urls) & set(distractor_urls)), "distractor/gold URL overlap"
    corpus_urls = gold_urls + distractor_urls
    rng.shuffle(corpus_urls)  # avoid all-gold-first id clustering

    urls_txt = out / "urls.txt"
    with open(urls_txt, "w", encoding="utf-8") as f:
        f.write(
            f"# PixelRAG NQ-Tables mini-corpus — {len(corpus_urls)} pages "
            f"({len(gold_urls)} gold + {len(distractor_urls)} distractors), seed={SEED}\n"
        )
        for u in corpus_urls:
            f.write(u + "\n")

    gold_jsonl = out / "gold.jsonl"
    with open(gold_jsonl, "w", encoding="utf-8") as f:
        for g in gold:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    manifest = {
        "dataset": "nq_tables",
        "split": "dev",
        "seed": SEED,
        "grader": "exact_match",  # eval/lib/grader.py EXACT_MATCH_TASKS — NOT the VLM judge
        "source_url": DATASET_URL,
        "dev_bytes": dev_path.stat().st_size,
        "n_candidates_resolvable": len(cands),
        "n_unique_pages": len(unique),
        "dropped": drops,
        "n_gold": len(gold),
        "n_distractors": len(distractor_urls),
        "n_corpus_pages": len(corpus_urls),
        "render": {"status": "pending", "n_rendered": None, "n_skipped": None, "skipped_reasons": []},
        "git_commit": _git_commit(),
        "gold_urls": gold_urls,
        "distractor_urls": distractor_urls,
        "corpus_urls": corpus_urls,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    (out / "pixelrag.yaml").write_text(_yaml(out, urls_txt, "fixed", out / "output"))
    (out / "pixelrag_ca.yaml").write_text(_yaml(out, urls_txt, "content_aware", out / "output_ca"))

    print(
        f"[build] wrote:\n"
        f"  {urls_txt}  ({len(corpus_urls)} pages)\n"
        f"  {gold_jsonl}  ({len(gold)} labeled queries)\n"
        f"  {out / 'manifest.json'}\n"
        f"  {out / 'pixelrag.yaml'} (fixed) + {out / 'pixelrag_ca.yaml'} (content_aware)\n"
        f"\nNext (both arms): pixelrag index build -c {out / 'pixelrag.yaml'}  and  -c {out / 'pixelrag_ca.yaml'}",
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Build the NQ-Tables labeled mini-corpus (stdlib only).")
    p.add_argument("--out", default=str(Path(__file__).parent / "nq_tables"),
                   help="Output dir (default: research/benchmarks/nq_tables).")
    p.add_argument("--n-gold", type=int, default=150, help="Number of labeled gold queries/pages.")
    p.add_argument("--n-distractors", type=int, default=60,
                   help="Extra corpus pages that are never gold (mirrors the iNat ~2.5:1 gold:distractor ratio).")
    build(p.parse_args())


if __name__ == "__main__":
    main()
