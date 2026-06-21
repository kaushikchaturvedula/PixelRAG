# EXPERIMENTS

Living record of how to reproduce every number/figure for the PixelRAG paper extension. Expanded
in Phase 4; for now it pins the **metric definitions and their cross-arm comparability**, which
must stay fixed across all arms (chunker × retrieval × region-source).

## Phase 1 — baseline
Reproduce: `research/REPRODUCE_PHASE1.md` (EVQA mini-corpus, 25 gold + 15 distractors, seed 42).
Output: `results/baseline.json` (written by `run_eval.py`). No reader in Phase 1.

## Metric definitions (fixed across arms)
- **Relevance**: article-level binary — a retrieved tile is relevant iff it comes from the gold
  Wikipedia page (`gold_url in hit.url`, the same test as `eval/run_bench.py:792-810`).
- **Recall@k**: fraction of queries with ≥1 gold-page tile in the top-k. **Arm-invariant.**
- **DCG@k**: `Σ_{i=1..k} rel_i / log2(i+1)`, raw, no normalization. **Arm-invariant.**
- **nDCG@k**: `DCG@k / IDCG@k`, where `IDCG@k = Σ_{i=1..min(k,R)} 1/log2(i+1)` and
  `R = #gold-page tiles in THIS arm's index`.
- **Cost**: tiles/page; mean vision-tokens/page = `ceil(w/28)·ceil(h/28)` per chunk summed per page
  (28px = one Qwen3-VL merged-patch cell); chunk-height distribution; index size (GB); latency.

### ⚠️ Cross-arm comparability of nDCG@k
nDCG@k is normalized by **R**, which is the number of gold-page tiles in that arm's index.
Content-aware chunking (Phase 2) changes R for the *same* page (different tiling → different tile
count), so **each arm is normalized to its own ideal and nDCG@k is NOT strictly comparable
fixed-vs-content-aware**. For the cross-arm comparison in the paper use **raw DCG@k** and
**Recall@k** (both arm-invariant); keep nDCG@k as a within-arm ranking-quality measure.
`run_eval.py` reports all three (`recall@k`, `dcg@k`, `ndcg@k`) per modality and stamps this note
into `baseline.json` as `metric_notes`.

### Corpus & query modality
Corpus: EVQA **iNaturalist** subset (25 gold species pages + 15 distractors, seed 42, gold-set md5
`b3418aa1…`). iNaturalist is chosen over landmarks for **source independence**: landmark (GLDv2)
query images are Wikimedia Commons photos that can overlap the gold page (retrieval leakage), whereas
iNaturalist query photos come from iNaturalist.

Image queries are the **primary** retrieval metric (EVQA questions are deictic — "this plant"); text
queries are a **secondary floor**. Query photos are same-species **representative photos from the
iNaturalist API** (`fetch_query_images.py --inat-source api`) — leak-free and no large download, but
**not EVQA-exact** (the task is "photo of species X → find X's page"). The EVQA-exact alternative
(`--inat-source val-tar`) needs the iNat2021 `val.tar.gz` (~8.93GB). Leakage is verified by
`check_leakage.py` (perceptual-hash query vs gold-page images; Phase-1 run: 0/25 leaks).
`baseline.json → query_modality` records image coverage (n_image_queries / n_text_fallback).
