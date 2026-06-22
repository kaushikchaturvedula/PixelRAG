#!/usr/bin/env python3
"""Build a small *labeled* Natural Questions (NQ) closed mini-corpus for the PixelRAG paper extension.

Phase-4, second benchmark — the **prose** contrast to NQ-Tables (whose answers live in HTML tables).
Mirrors research/build_nqt_corpus.py exactly (closed-corpus design + the portable `_yaml`), so the NQ
yamls are correct out-of-the-box: repo-root-relative paths, embed.backend=direct_gpu, batch_size=4.

Gold QA comes from the existing NQ loader logic (eval/lib/simpleqa_data.py:load_nq_data), replicated
here so the builder is self-contained — EXCEPT the fetch itself needs HuggingFace `datasets`:
NQ is read by streaming `google-research-datasets/natural_questions` (the full dataset; rows embed the
entire tokenized Wikipedia document, so this is a real download even when streamed). We stop streaming
as soon as enough unique resolvable gold pages are collected, to bound the transfer.

Per question we keep: the question text, the SHORT answer span(s) (answer/reference_list), and the gold
Wikipedia page URL derived from `document.url` (NQ's `/w/index.php?title=Foo&oldid=123` -> /wiki/Foo).
Validation keeps examples where >=2 of 5 annotators marked a non-null short answer (the NQ eval protocol).

Outputs (under --out, default research/benchmarks/nq/):
  urls.txt          gold + distractor page URLs (web source input)
  gold.jsonl        one row per QUERY (same schema as nq_tables/gold.jsonl; dataset_name=nq)
  manifest.json     provenance + counts (scanned / dropped / gold / distractor)
  pixelrag.yaml     fixed-chunker config      (-> output/)
  pixelrag_ca.yaml  content_aware-chunker cfg (-> output_ca/)

DEP: requires `datasets` (HF). Grading: EXACT-MATCH (eval/lib/grader.py EXACT_MATCH_TASKS), like NQ-Tables.
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
from pathlib import Path

NQ_HF_DATASET = "google-research-datasets/natural_questions"
WIKI_PREFIX = "https://en.wikipedia.org/wiki/"
SEED = 42


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _repo_rel(p: Path) -> str:
    """Path relative to the repo root (portable across machines); absolute if outside the repo."""
    repo = Path(__file__).resolve().parent.parent.parent  # research/benchmarks/<file> -> repo root
    p = Path(p).resolve()
    try:
        return p.relative_to(repo).as_posix()
    except ValueError:
        return str(p)


def _resolve_url(raw: str) -> str | None:
    """NQ document.url -> clean https://en.wikipedia.org/wiki/Title, or None if not resolvable.

    Identical normalization to research/build_nqt_corpus.py / load_nq_data: html-unescape, map
    /w/index.php?title=Foo&oldid=123 -> /wiki/Foo, keep only en.wikipedia article pages.
    """
    if not raw:
        return None
    url = _html.unescape(raw)
    m = re.search(r"[?&]title=([^&]+)", url)
    if m:
        url = WIKI_PREFIX + urllib.parse.quote(m.group(1), safe="/:(),-")
    url = re.sub(r"(?<!:)//+", "/", url)  # collapse en.wikipedia.org//w/ style double slashes
    if not url.startswith(WIKI_PREFIX):
        return None
    tail = url[len(WIKI_PREFIX):]
    if not tail or tail.startswith(("File:", "Special:", "Help:", "Template:", "Category:")):
        return None
    return url


def _parse_example(ex: dict, min_nn: int) -> tuple[dict | None, str | None]:
    """One NQ HF example -> candidate dict, or (None, drop_reason).

    Pure (no I/O), so it is testable without `datasets`. Mirrors load_nq_data: short answer requires
    >= min_nn of 5 annotators with a non-null span; gold URL from document.url (?title=Foo -> /wiki/Foo).
    """
    ann = ex["annotations"]
    texts: set[str] = set()
    non_null = 0
    for i in range(len(ann["id"])):
        spans = ann["short_answers"][i].get("text", []) or []
        if spans:
            non_null += 1
            for s in spans:
                if s.strip():
                    texts.add(s.strip())
    if non_null < min_nn or not texts:
        return None, "no_short_answer"
    url = _resolve_url(ex["document"]["url"])
    if not url:
        return None, "no_url"
    question = ex["question"]["text"]
    refs = sorted(texts)
    return {
        "id": hashlib.md5(f"{question}|{url}".encode()).hexdigest(),
        "question": question,
        "answer": " | ".join(refs),
        "reference_list": refs,
        "gold_url": url,
        "gold_title": (ex["document"].get("title") or "").strip(),
        "dataset_name": "nq",
        "question_type": "short_answer",
    }, None


def _load_candidates(n_target: int, max_scan: int, split: str) -> tuple[list[dict], dict]:
    """Stream NQ, keep questions with a short answer AND a resolvable wiki URL, dedup by page.

    >=2 annotators (validation) / >=1 (train) with a non-null short answer. Stops as soon as n_target
    unique gold pages are found, to bound the streaming download. Returns (unique_candidates, drops).
    """
    from datasets import load_dataset  # heavy HF dep — only needed at fetch time

    ds = load_dataset(NQ_HF_DATASET, split=split, streaming=True)
    min_nn = 2 if split == "validation" else 1
    drops = {"scanned": 0, "no_short_answer": 0, "no_url": 0, "dup_page": 0}
    seen: set[str] = set()
    out: list[dict] = []
    for ex in ds:
        if drops["scanned"] >= max_scan:
            break
        drops["scanned"] += 1
        cand, reason = _parse_example(ex, min_nn)
        if cand is None:
            drops[reason] += 1
            continue
        if cand["gold_url"] in seen:  # one question per gold page (Recall well-defined)
            drops["dup_page"] += 1
            continue
        seen.add(cand["gold_url"])
        out.append(cand)
        if len(out) >= n_target:  # enough unique gold pages — stop streaming
            break
    return out, drops


def _yaml(out: Path, urls_txt: Path, chunker: str, index_out: Path) -> str:
    # Reused verbatim from research/build_nqt_corpus.py: repo-root-relative paths (portable),
    # embed.backend=direct_gpu (Kaggle/T4 default sglang isn't installed), batch_size caps GPU memory.
    return (
        "# Auto-generated by research/benchmarks/build_nq_corpus.py — PixelRAG NQ mini-corpus.\n"
        "# Paths are repo-root-relative; run `pixelrag index build -c <this file>` from the repo root.\n"
        "source:\n  type: web\n"
        f"  urls_file: {_repo_rel(urls_txt)}\n"
        "ingest:\n  backend: cdp\n  quality: 85\n"
        f"chunk:\n  chunker: {chunker}\n"
        "embed:\n  model: Qwen/Qwen3-VL-Embedding-2B\n  device: cuda\n"
        "  backend: direct_gpu\n  batch_size: 4\n"
        f"output: {_repo_rel(index_out)}\n"
    )


def build(args: argparse.Namespace) -> None:
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    import random

    need = args.n_gold + args.n_distractors
    cands, drops = _load_candidates(need, args.max_scan, args.split)
    print(
        f"[build] scanned {drops['scanned']} NQ {args.split} examples -> {len(cands)} unique gold pages "
        f"(dropped: no_short_answer={drops['no_short_answer']} no_url={drops['no_url']} dup_page={drops['dup_page']})",
        flush=True,
    )
    if len(cands) < need:
        sys.exit(
            f"[build] ERROR: only {len(cands)} unique resolvable pages, need {need}. "
            f"Raise --max-scan (currently {args.max_scan}) or lower --n-gold/--n-distractors."
        )

    rng = random.Random(SEED)
    rng.shuffle(cands)
    gold = cands[: args.n_gold]
    distractors = cands[args.n_gold : need]
    gold_urls = [g["gold_url"] for g in gold]
    distractor_urls = [d["gold_url"] for d in distractors]
    assert not (set(gold_urls) & set(distractor_urls)), "distractor/gold URL overlap"
    corpus_urls = gold_urls + distractor_urls
    rng.shuffle(corpus_urls)

    urls_txt = out / "urls.txt"
    with open(urls_txt, "w", encoding="utf-8") as f:
        f.write(
            f"# PixelRAG NQ mini-corpus — {len(corpus_urls)} pages "
            f"({len(gold_urls)} gold + {len(distractor_urls)} distractors), seed={SEED}\n"
        )
        for u in corpus_urls:
            f.write(u + "\n")

    gold_jsonl = out / "gold.jsonl"
    with open(gold_jsonl, "w", encoding="utf-8") as f:
        for g in gold:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    manifest = {
        "dataset": "nq",
        "split": args.split,
        "seed": SEED,
        "grader": "exact_match",  # eval/lib/grader.py EXACT_MATCH_TASKS — NOT the VLM judge
        "source": NQ_HF_DATASET,
        "filter": ">=2 annotators with non-null short answer" if args.split == "validation" else "non-null short answer",
        "n_scanned": drops["scanned"],
        "n_unique_pages": len(cands),
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
        f"\nNext (both arms, from repo root): pixelrag index build -c {_repo_rel(out / 'pixelrag.yaml')}  and  -c {_repo_rel(out / 'pixelrag_ca.yaml')}",
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Build the NQ labeled mini-corpus (needs HF `datasets`).")
    p.add_argument("--out", default=str(Path(__file__).parent / "nq"),
                   help="Output dir (default: research/benchmarks/nq).")
    p.add_argument("--split", default="validation", choices=["validation", "train"])
    p.add_argument("--n-gold", type=int, default=150, help="Number of labeled gold queries/pages.")
    p.add_argument("--n-distractors", type=int, default=60,
                   help="Extra corpus pages never gold (mirrors NQ-Tables' ~2.5:1 gold:distractor ratio).")
    p.add_argument("--max-scan", type=int, default=8000,
                   help="Safety cap on examples streamed (stops earlier once enough unique pages are found).")
    build(p.parse_args())


if __name__ == "__main__":
    main()
