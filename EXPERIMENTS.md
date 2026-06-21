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

---

## Phase 2 — content-aware chunking: results

**Setup.** Same 40-page iNaturalist mini-corpus + 25 image-primary queries as Phase 1. Two arms on the
**identical corpus + queries**: `fixed` = clean baseline (`results/baseline_clean.json`, untruncated
tiles), `content_aware` (`results/content_aware.json`, DOM-region cuts, `--region-source dom`). Embed
`Qwen/Qwen3-VL-Embedding-2B` → FAISS IVF → `run_eval.py`. Mechanism metric on a separate 18-page
table-stress probe set (`results/boundary_violation_probe.json`). Render fixes (page-target,
truncation, scroll-0 regions) on branch `feat/content-aware-chunker`.

**Clean-baseline note.** The truncation fix recovered 15 chunks on 6 tall pages (320→335). The
re-run fixed baseline is ≈ the original: image recall@{1,3,5,10} = 0.88 / 0.96 / 1.00 / 1.00
(identical), DCG@10 2.053 vs 2.051. → the original Phase-1 numbers were sound; all Phase-2
comparisons use the **clean** baseline (`baseline_clean.json`).

### Mechanism — boundary-violation (table-stress probe set, 18 pages)
A region is "split" if a chunk boundary slices its interior (lower = better).
| arm | block-split | table-row-split |
|---|---|---|
| fixed | 3.82% (387/10130) | 5.42% (387/7140) |
| content_aware | **0.77%** (78/10130) | **0.95%** (68/7140) |

→ ~**5× fewer** block splits, ~**5.7× fewer** table-row splits — the chunker's intended effect.

### Accuracy — 40-page iNat, image-primary, equal vision-token budget
| image metric | fixed (clean) | content_aware | Δ |
|---|---|---|---|
| recall@1 | 0.88 | **0.96** | +0.08 |
| recall@3 | 0.96 | 0.96 | 0 |
| recall@5 | 1.00 | 0.96 | −0.04¹ |
| recall@10 | 1.00 | 1.00 | 0 |
| DCG@3 | 1.589 | 1.740 | +0.151 |
| DCG@5 | 1.856 | 1.987 | +0.131 |
| DCG@10 | 2.053 | **2.284** | +0.231 |

**Paired Wilcoxon** (per-query DCG, image, vs clean baseline): k=3 p=0.022 (9/1 better/worse), k=5
p=0.047 (14/2), k=10 **p=0.0016** (18/2). Significance grows with k (k=1 n.s. — only 2 queries differ).
→ content_aware ranks the gold page's tiles higher, increasingly so deeper in the list; recall@1
0.88→0.96. ¹recall@5 is a single-query reorder that recovers by @10.

### Cost-neutrality
Mean **vision-tokens/page: fixed 9834 vs content_aware 9805 (−0.3%)** — the reader's token budget is
equal (same total pixels chunked). content_aware uses **more, smaller, content-aligned** chunks
(417 vs 335; mean height 809px σ=329 vs 1015px σ=82) → ~24% more retrieval vectors and a marginally
larger index (3.5MB vs 2.8MB at this scale). So: reader cost equal; retrieval granularity finer
(plausibly part of why DCG improves).

### Limitations (honest)
- **Text-query floor**: content_aware is slightly *lower* on text queries (recall@1 0.48→0.44, @10
  0.80→0.68) but **not significant at any k** (paired Wilcoxon p ≥ 0.23). The gain is on the primary
  (image) metric only.
- **Multi-column infobox residual**: content_aware still splits ~0.8% of blocks / ~0.95% of rows —
  almost all the Wikipedia infobox/taxobox-beside-text case, where one horizontal cut can't isolate a
  right-column table from left-column text. Single-column flow is cut cleanly.
- **Scale**: 40 pages / 25 queries — directional and significant on DCG@k, but small-n; full-corpus
  confirmation is future work.

### Reproduce
`research/REPRODUCE_PHASE1.md` (corpus + render) then, on a GPU box, the 3-step sequence: clean
`fixed` baseline → `content_aware` embed/index → `run_eval.py --chunker content_aware --baseline
results/baseline_clean.json` (paired Wilcoxon). Mechanism: `research/probe_set/boundary_violation.py`.
