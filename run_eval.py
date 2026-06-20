#!/usr/bin/env python3
"""PixelRAG evaluation harness — Phase 1 baseline (retrieval + cost only; no reader).

Scores a locally-built mini-corpus index against the labeled gold queries and writes
``results/baseline.json``. Stdlib-only: it drives the existing FAISS search API
(``pixelrag serve``) over HTTP — the same retriever that scales to the full corpus — rather
than re-implementing retrieval, then derives cost metrics from the on-disk chunk manifests.

Metrics
  Retrieval (article-level binary relevance — a tile is relevant iff it comes from the gold
  Wikipedia page; mirrors run_bench.py's ``gt_url in retrieved_url`` hit-test). Scored for TWO
  query modalities so every later arm is comparable:
    * IMAGE query (PRIMARY) — the EVQA query photo (base64) sent to serve. EVQA questions are
      deictic ("this castle"), so the image is the real retrieval signal. Scored over the queries
      whose photo fetch_query_images.py resolved; coverage reported in ``query_modality``.
    * TEXT query (secondary floor) — the question text; scored over ALL queries.
    * Recall@k  — fraction of queries with >=1 gold-page tile in the top-k (arm-invariant).
    * nDCG@k    — binary-relevance nDCG (see _ndcg_at_k for the exact DCG/IDCG formula).
    * DCG@k     — raw DCG (same rels, same log discount, NO per-arm normalization).

CROSS-ARM COMPARABILITY: nDCG@k normalizes by IDCG@k where R = #gold-page tiles in THIS arm's
index. Phase-2 content-aware chunking changes R for the same page, so each arm is normalized to
its OWN ideal and nDCG@k is not strictly comparable fixed-vs-content-aware. Use raw DCG@k (and
Recall@k) for the cross-arm comparison — both are arm-invariant. nDCG@k stays for within-arm
ranking quality. (Mirrored in EXPERIMENTS.md and stamped into baseline.json as metric_notes.)
  Cost (no model needed — read from chunks.json):
    * tiles_per_page        — mean retrieval-unit (chunk) count per page.
    * chunk_height_dist     — min/mean/median/max/std of chunk pixel heights (the fixed-height
                              reference the Phase 2 content-aware chunker is compared against).
    * vision_tokens_per_page — mean Qwen3-VL vision tokens per page (formula stored in output).
    * index_size_gb         — index.faiss (+ metadata.npz) size.
    * retrieval_latency_ms  — mean/median/p95 wall time per /search call.
  Provenance: git commit, seed, chunker label, model, k-list, nprobe, counts, timestamp.

Phase-1 note: fixed-vs-content-aware deltas on Recall@k / article-level nDCG@k are EXPECTED to
be ~flat — these metrics count any gold-page tile as relevant, split or not. The chunking signal
lives in the Phase-2 boundary-violation rate and Phase-4 answer-bearing-tile nDCG@k.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SEED = 42
# One Qwen3-VL merged visual token covers a 28x28 px cell (patch_size 14 * spatial_merge 2);
# this is the same 28px unit chunk.py calls "one Qwen3-VL patch" (MIN_CHUNK_HEIGHT = 28).
VISION_TOKEN_CELL_PX = 28
VISION_TOKEN_FORMULA = "ceil(width/28) * ceil(height/28) per chunk, summed per page"

# Travels with every baseline.json so the comparison rule is unambiguous in the results file.
NDCG_COMPARABILITY_NOTE = (
    "nDCG@k normalizes by IDCG@k with R = #gold-page tiles in THIS arm's index; content-aware "
    "chunking changes R for the same page, so nDCG@k is normalized per-arm and is NOT strictly "
    "comparable fixed-vs-content-aware. Use raw DCG@k (same rels + log discount, no per-arm "
    "normalization) and Recall@k for cross-arm comparison — both are arm-invariant."
)


# --------------------------------------------------------------------------- metrics
def _dcg(relevances: list[int], k: int) -> float:
    # rank i is 1-indexed: gain / log2(rank + 1)
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))


def _ndcg_at_k(relevances: list[int], n_relevant_in_corpus: int, k: int) -> float:
    """Binary-relevance nDCG@k.

    relevances: 0/1 per retrieved rank (1 = tile from the gold page).
    n_relevant_in_corpus: total gold-page tiles in the index (R) — defines the ideal ranking
        where the first min(k, R) slots are all relevant.
    """
    dcg = _dcg(relevances, k)
    ideal = _dcg([1] * min(k, n_relevant_in_corpus), k)
    return dcg / ideal if ideal > 0 else 0.0


# --------------------------------------------------------------------------- paired comparison
def _avg_ranks(xs: list[float]) -> list[float]:
    """Average (tie-corrected) ranks of xs, 1-based."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # average of ranks i+1..j+1
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    return ranks


def _wilcoxon(diffs: list[float]) -> dict:
    """Two-sided Wilcoxon signed-rank (normal approximation w/ continuity correction).

    Self-contained (no scipy). For small n the normal approx is rough — use scipy.stats.wilcoxon for
    exact p-values in the paper; this gives a quick directional signal alongside median diff + W±.
    """
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    if n == 0:
        return {"n": 0, "note": "all paired diffs zero"}
    ranks = _avg_ranks([abs(d) for d in nz])
    w_plus = sum(r for r, d in zip(ranks, nz) if d > 0)
    w_minus = sum(r for r, d in zip(ranks, nz) if d < 0)
    w = min(w_plus, w_minus)
    mean = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24
    z = (w - mean + 0.5) / math.sqrt(var) if var > 0 else 0.0
    p = math.erfc(abs(z) / math.sqrt(2))  # two-sided
    return {"n": n, "W": round(w, 1), "W_plus": round(w_plus, 1), "W_minus": round(w_minus, 1),
            "z": round(z, 3), "p_approx": round(p, 4)}


def _paired_compare(per_query: list[dict], baseline_path: str, ks: list[int]) -> dict:
    """Per-query PAIRED comparison vs a baseline results JSON, matched by qid.

    Uses arm-invariant per-query DCG@k (image primary + text floor) so the comparison is valid across
    chunkers (nDCG is per-arm-normalized — see metric_notes). Reports median diff, win/loss counts,
    and a Wilcoxon signed-rank test per k.
    """
    base = json.loads(Path(baseline_path).read_text())
    base_pq = {q.get("qid"): q for q in base.get("per_query", [])}
    out = {"baseline": str(baseline_path), "baseline_chunker": base.get("chunker"), "by_k": {}}
    for k in ks:
        block = {}
        for key in ("rels_image", "rels_text"):
            diffs = []
            for pq in per_query:
                b = base_pq.get(pq.get("qid"))
                if not b or pq.get(key) is None or b.get(key) is None:
                    continue
                diffs.append(_dcg(pq[key], k) - _dcg(b[key], k))
            if diffs:
                block[key.replace("rels_", "")] = {
                    "n": len(diffs),
                    "median_dcg_diff": round(statistics.median(diffs), 4),
                    "mean_dcg_diff": round(statistics.fmean(diffs), 4),
                    "n_better": sum(1 for d in diffs if d > 0),
                    "n_worse": sum(1 for d in diffs if d < 0),
                    "n_tie": sum(1 for d in diffs if d == 0),
                    "wilcoxon": _wilcoxon(diffs),
                }
        out["by_k"][k] = block
    return out


# --------------------------------------------------------------------------- serve
def _wait_health(port: int, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
            return
        except Exception:
            time.sleep(2)
    raise TimeoutError(f"serve API did not become healthy on :{port} within {timeout_s}s")


def _search(port: int, query: dict, n_docs: int, nprobe: int | None) -> tuple[list[dict], float]:
    """query is a serve Query dict: {"text": ...} or {"image": "<base64>"}."""
    body: dict = {"queries": [query], "n_docs": n_docs}
    if nprobe is not None:
        body["nprobe"] = nprobe
    req = urllib.request.Request(
        f"http://localhost:{port}/search",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return data.get("results", [{}])[0].get("hits", []), dt_ms


# --------------------------------------------------------------------------- cost (chunks.json)
def _iter_chunk_manifests(tiles_dir: Path):
    for cj in sorted(tiles_dir.rglob("chunks.json")):
        try:
            yield cj, json.loads(cj.read_text())
        except (json.JSONDecodeError, OSError):
            continue


def _cost_metrics(tiles_dir: Path) -> dict:
    per_page_chunks: list[int] = []
    per_page_tokens: list[int] = []
    chunk_heights: list[int] = []
    for _cj, meta in _iter_chunk_manifests(tiles_dir):
        chunks = meta.get("chunks", [])
        if not chunks:
            continue
        per_page_chunks.append(len(chunks))
        tokens = 0
        for c in chunks:
            w, h = int(c.get("width", 0)), int(c.get("height", 0))
            chunk_heights.append(h)
            tokens += math.ceil(w / VISION_TOKEN_CELL_PX) * math.ceil(h / VISION_TOKEN_CELL_PX)
        per_page_tokens.append(tokens)

    def _stats(xs: list[float]) -> dict:
        if not xs:
            return {"n": 0}
        return {
            "n": len(xs),
            "min": min(xs),
            "mean": round(statistics.fmean(xs), 2),
            "median": statistics.median(xs),
            "max": max(xs),
            "std": round(statistics.pstdev(xs), 2) if len(xs) > 1 else 0.0,
        }

    return {
        "pages": len(per_page_chunks),
        "total_chunks": sum(per_page_chunks),
        "tiles_per_page": _stats([float(x) for x in per_page_chunks]),
        "vision_tokens_per_page": _stats([float(x) for x in per_page_tokens]),
        "chunk_height_px_dist": _stats([float(x) for x in chunk_heights]),
        "vision_token_cell_px": VISION_TOKEN_CELL_PX,
        "vision_token_formula": VISION_TOKEN_FORMULA,
    }


def _gold_chunk_count(tiles_dir: Path, article_id: int) -> int:
    cj = tiles_dir / f"{article_id}.png.tiles" / "chunks.json"
    if cj.exists():
        try:
            return len(json.loads(cj.read_text()).get("chunks", []))
        except (json.JSONDecodeError, OSError):
            pass
    return 0


def _index_size_gb(index_dir: Path) -> dict:
    sizes = {}
    for name in ("index.faiss", "metadata.npz"):
        p = index_dir / name
        sizes[name] = p.stat().st_size if p.exists() else 0
    total = sum(sizes.values())
    return {"bytes": sizes, "total_gb": round(total / 1e9, 6)}


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent
        ).decode().strip()
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description="PixelRAG Phase-1 baseline eval (retrieval + cost).")
    p.add_argument("--index-dir", required=True, help="Built index dir (has index.faiss, articles.json, tiles/).")
    p.add_argument("--gold", required=True, help="gold.jsonl from build_mini_corpus.py.")
    p.add_argument("--query-images-dir", default=None,
                   help="Dir with {qid}.jpg query photos from fetch_query_images.py (image queries are "
                        "PRIMARY). Default: <gold-dir>/query_images. Missing image → text fallback.")
    p.add_argument("--tiles-dir", default=None, help="Tiles dir (default: <index-dir>/tiles).")
    p.add_argument("--articles-json", default=None, help="Default: <index-dir>/articles.json.")
    p.add_argument("--k", default="1,3,5,10", help="Comma list of cutoffs for Recall@k / nDCG@k.")
    p.add_argument("--model", default="Qwen/Qwen3-VL-Embedding-2B")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--nprobe", type=int, default=None)
    p.add_argument("--port", type=int, default=31999)
    p.add_argument("--serve-timeout", type=float, default=600.0, help="Seconds to wait for serve health.")
    p.add_argument("--chunker", default="fixed", help="Chunker label recorded in the output (provenance).")
    p.add_argument("--region-source", default=None, help="Region source label (dom|vision) for content_aware (provenance).")
    p.add_argument("--retrieval-mode", default="flat", help="Retrieval mode label (flat|hier-expand) (provenance).")
    p.add_argument("--baseline", default=None,
                   help="Optional prior results JSON (e.g. fixed baseline) for a per-query PAIRED comparison "
                        "(Wilcoxon signed-rank on per-query DCG@k, matched by qid).")
    p.add_argument("--out", default=str(Path(__file__).parent / "results" / "baseline.json"))
    p.add_argument("--reuse-serve", action="store_true", help="Assume a serve is already up on --port.")
    args = p.parse_args()

    index_dir = Path(args.index_dir).resolve()
    tiles_dir = Path(args.tiles_dir).resolve() if args.tiles_dir else index_dir / "tiles"
    articles_json = Path(args.articles_json).resolve() if args.articles_json else index_dir / "articles.json"
    ks = [int(x) for x in args.k.split(",") if x.strip()]
    max_k = max(ks)

    gold_path = Path(args.gold).resolve()
    query_images_dir = (Path(args.query_images_dir).resolve() if args.query_images_dir
                        else gold_path.parent / "query_images")
    gold = [json.loads(l) for l in gold_path.read_text().splitlines() if l.strip()]
    articles = json.loads(articles_json.read_text())  # list: id -> {title, url}
    url_to_aid = {a.get("url", ""): i for i, a in enumerate(articles) if a.get("url")}

    # --- start serve (unless reusing one) ---
    serve_proc = None
    if not args.reuse_serve:
        env = os.environ.copy()
        cmd = [
            sys.executable, "-m", "pixelrag_serve.api",
            "--index-dir", str(index_dir),
            "--tiles-dir", str(tiles_dir),
            "--articles-json", str(articles_json),
            "--model", args.model,
            "--device", args.device,
            "--port", str(args.port),
        ]
        print(f"[eval] starting serve: {' '.join(cmd)}", flush=True)
        serve_proc = subprocess.Popen(cmd, env=env)
    try:
        _wait_health(args.port, args.serve_timeout)
        print(f"[eval] serve healthy on :{args.port}; running {len(gold)} queries", flush=True)

        # article-level binary relevance per rank (gt_url substring of hit url == same article)
        def _rels(hits: list[dict], gold_url: str) -> list[int]:
            return [1 if (gold_url and gold_url in (h.get("url") or "")) else 0 for h in hits]

        # --- per-query retrieval: IMAGE query (primary) + TEXT query (secondary floor) ---
        img_lat: list[float] = []
        txt_lat: list[float] = []
        per_query = []
        for q in gold:
            qid = q.get("id")
            gold_url = (q.get("gold_url") or "").strip()
            gold_aid = url_to_aid.get(gold_url)
            n_rel_corpus = _gold_chunk_count(tiles_dir, gold_aid) if gold_aid is not None else 0

            # TEXT query — always (conservative floor; deictic EVQA questions retrieve weakly)
            t_hits, t_ms = _search(args.port, {"text": q["question"]}, max_k, args.nprobe)
            txt_lat.append(t_ms)
            t_rels = _rels(t_hits, gold_url)

            # IMAGE query — primary, when the EVQA query photo was resolved by fetch_query_images.py
            img_path = query_images_dir / f"{qid}.jpg"
            has_image = img_path.exists() and img_path.stat().st_size > 0
            i_rels = None
            i_hits: list[dict] = []
            if has_image:
                b64 = base64.b64encode(img_path.read_bytes()).decode()
                i_hits, i_ms = _search(args.port, {"image": b64}, max_k, args.nprobe)
                img_lat.append(i_ms)
                i_rels = _rels(i_hits, gold_url)

            per_query.append({
                "qid": qid,
                "question": q["question"],
                "gold_url": gold_url,
                "gold_article_id": gold_aid,
                "n_relevant_in_corpus": n_rel_corpus,
                "has_image_query": has_image,
                "rels_image": i_rels,
                "rels_text": t_rels,
                "top_urls_image": [h.get("url") for h in i_hits[:max_k]],
                "top_urls_text": [h.get("url") for h in t_hits[:max_k]],
            })

        # --- aggregate retrieval metrics per modality (same gt_url hit-test for both) ---
        def _agg(rels_key: str, subset: list[dict]) -> dict:
            out = {"n_queries": len(subset)}
            for k in ks:
                if subset:
                    recall = statistics.fmean([1.0 if any(pq[rels_key][:k]) else 0.0 for pq in subset])
                    ndcg = statistics.fmean(
                        [_ndcg_at_k(pq[rels_key], max(pq["n_relevant_in_corpus"], 1), k) for pq in subset]
                    )
                    # raw DCG@k: NO per-arm IDCG normalization → arm-invariant cross-arm comparison
                    dcg = statistics.fmean([_dcg(pq[rels_key], k) for pq in subset])
                else:
                    recall = ndcg = dcg = 0.0
                out[f"recall@{k}"] = round(recall, 4)
                out[f"ndcg@{k}"] = round(ndcg, 4)
                out[f"dcg@{k}"] = round(dcg, 4)
            return out

        img_subset = [pq for pq in per_query if pq["rels_image"] is not None]
        retrieval = {
            "primary": "image",
            "image": _agg("rels_image", img_subset),
            "text": _agg("rels_text", per_query),
        }
        query_modality = {
            "n_total": len(per_query),
            "n_image_queries": len(img_subset),
            "n_text_fallback": len(per_query) - len(img_subset),
            "fallback_qids": [pq["qid"] for pq in per_query if pq["rels_image"] is None],
            "query_images_dir": str(query_images_dir),
        }

        def _lat(xs: list[float]) -> dict:
            if not xs:
                return {"n": 0}
            s = sorted(xs)
            return {
                "mean_ms": round(statistics.fmean(xs), 1),
                "median_ms": round(statistics.median(xs), 1),
                "p95_ms": round(s[max(0, math.ceil(0.95 * len(s)) - 1)], 1),
                "n": len(xs),
            }
        latency = {"image": _lat(img_lat), "text": _lat(txt_lat)}
    finally:
        if serve_proc is not None:
            serve_proc.terminate()
            try:
                serve_proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                serve_proc.kill()

    # --- cost + provenance (no model needed) ---
    cost = _cost_metrics(tiles_dir)
    cost["index_size"] = _index_size_gb(index_dir)
    cost["retrieval_latency"] = latency

    paired = _paired_compare(per_query, args.baseline, ks) if args.baseline else None

    result = {
        "phase": "1-baseline" if args.chunker == "fixed" else "2-content-aware",
        "chunker": args.chunker,
        "retrieval_mode": args.retrieval_mode,
        "region_source": args.region_source,
        "paired_vs_baseline": paired,
        "provenance": {
            "git_commit": _git_commit(),
            "seed": SEED,
            "model": args.model,
            "device": args.device,
            "k": ks,
            "nprobe": args.nprobe,
            "n_queries": len(gold),
            "n_corpus_pages": cost["pages"],
            "index_dir": str(index_dir),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "retrieval": retrieval,
        "query_modality": query_modality,
        "metric_notes": NDCG_COMPARABILITY_NOTE,
        "cost": cost,
        "per_query": per_query,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    # --- console summary ---
    def _fmt(m: dict) -> str:
        return "  ".join(f"{key}={m[key]}" for key in m if key != "n_queries")

    print(f"\n===== PixelRAG eval — chunker={args.chunker} region_source={args.region_source} =====")
    print(f" corpus_pages={cost['pages']}  commit={result['provenance']['git_commit'][:8]}")
    print(f" query modality: image(primary)={query_modality['n_image_queries']}  "
          f"text(floor)={query_modality['n_total']}  text-fallback={query_modality['n_text_fallback']}")
    print(f" retrieval[IMAGE, primary, n={retrieval['image']['n_queries']}]: {_fmt(retrieval['image'])}")
    print(f" retrieval[TEXT, floor,   n={retrieval['text']['n_queries']}]: {_fmt(retrieval['text'])}")
    print(f" tiles/page (mean): {cost['tiles_per_page'].get('mean')}   total_chunks: {cost['total_chunks']}")
    print(f" vision-tokens/page (mean): {cost['vision_tokens_per_page'].get('mean')}")
    print(f" chunk-height px: {cost['chunk_height_px_dist']}")
    print(f" index size: {cost['index_size']['total_gb']} GB   "
          f"latency image={latency['image'].get('mean_ms')}ms text={latency['text'].get('mean_ms')}ms")
    if paired:
        kmax = max(ks)
        pim = paired["by_k"].get(kmax, {}).get("image")
        if pim:
            print(f" paired vs {paired.get('baseline_chunker')} (DCG@{kmax}, image): "
                  f"median_diff={pim['median_dcg_diff']}  better/worse={pim['n_better']}/{pim['n_worse']}  "
                  f"wilcoxon p~{pim['wilcoxon'].get('p_approx')}")
    print(f"\n[eval] wrote {out}")


if __name__ == "__main__":
    main()
