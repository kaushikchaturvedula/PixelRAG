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

---

<!-- PHASE3:BEGIN (generated by research/gen_phase3_experiments.py — do not hand-edit) -->
## Phase 3 — hierarchical retrieve-then-expand: results

End-to-end QA accuracy (the gating dependency for a publishable PixelRAG result; retrieval-only
metrics are not). The reader was deferred since Phase 1; this phase wires a pixel-native hosted
VLM reader + an automatic VLM-judge grader and runs the 2×2 chunker×retrieval sweep.

### Setup
**Corpus.** Same 40-page iNaturalist mini-corpus, **25** image-primary queries, image modality
(EVQA questions are deictic — "this plant"). Reader and grader share the gold answers, so grading
needs no manual labeling.

**Reader.** OpenAI `gpt-4o`, `detail=low` — pixel-native: retrieved chunk tiles
are passed as **images**, never parsed text. **Judge.** `gpt-4.1-2025-04-14` (VLM-as-judge, the
paper grader prompt + Correct/Incorrect/Unattempted parse reused from `eval/lib/grader.py`).

**Retrieval modes.** `flat` = the top-4 retrieved tiles. `hier-expand` = each retrieved tile
plus its same-section siblings and ±1 reading-order neighbors (deduped, capped at 8), via the
Phase-3 page→section→tile tree. The fixed arm has no headings, so its hier-expand degrades to
reading-order-neighbors only (`expand_mode_effective=neighbors`) — section expansion is the
content_aware-specific lever.

**4-cell design.** `{fixed,content_aware}` × `{flat,hier-expand}`. The 2×2 separates the **chunking**
effect (flat: fixed vs content_aware) from the **expansion** effect (flat vs hier-expand within an
arm) and exposes their **interaction**, so "content_aware-hier beats fixed-flat" is not confounded by
only one arm receiving expansion.

### Results (4-cell, image-primary, n=25)
| chunker | retrieval | accuracy | mean tiles/query |
|---|---|---|---|
| fixed | flat | 0.44 | 4.00 |
| fixed | hier-expand | 0.40 | 6.84 |
| content_aware | flat | 0.56 | 4.00 |
| content_aware | hier-expand | **0.68** | 7.84 |

> **⚠️ Caveat — this +0.24 headline is IMAGE-QUERY-SPECIFIC.** It does **not** survive a
> text-query modality flip on the *same* corpus and content_aware tiles (disambiguation experiment,
> `research/CHUNKING_ANALYSIS.md`). With text queries the content_aware advantage is **-0.08 flat / +0.00 hier**
> (vs **+0.12 flat / +0.28 hier** with image queries). Mechanism: under image queries
> retrieval is saturated (the gold page reaches the reader ~96–100% either way), so content_aware's gain
> is a **reader-side** effect of finer tiles; under text queries content_aware's finer tiles **reduce**
> gold-page retrieval **68%→56%**, and the advantage vanishes. So content-aware chunking
> does **not** improve QA on its own — the gain here is contingent on image-query retrieval.

### Decomposition
- **Chunking** (fixed-flat → content_aware-flat): 0.44 → 0.56 (**+0.12**) — better boundaries help even with no expansion.
- **Expansion** (content_aware-flat → content_aware-hier): 0.56 → 0.68 (**+0.12**) — adding section + reading-order context to the same retrieved tiles.
- **Best vs baseline** (fixed-flat → content_aware-hier): 0.44 → 0.68 (**+0.24**, ≈55% relative).
- **Synergy, not addition.** Expansion on the **fixed** arm does **not** help (0.44 → 0.40, **-0.04**): reading-order neighbors
  around fixed-height cuts add noise as often as signal. The same expansion machinery only pays off
  once chunk boundaries follow content. The fixed-hier cell is the **control** that establishes this —
  the expansion gain is contingent on content-aware boundaries; the two techniques are synergistic.

### Significance — exact two-sided McNemar (binomial, paired, n=25)
Per query: 1 iff the judge verdict is `correct` (incorrect and unattempted both 0). `b` = baseline-arm
correct / other-arm wrong; `c` = the reverse. Exact binomial only — n is too small for the chi-square
or normal approximation (`research/qa_significance.py`).

| contrast | acc | Δacc | b (a✓/b✗) | c (a✗/b✓) | p (exact) |
|---|---|---|---|---|---|
| C1 chunking (fixed-flat → ca-flat) | 0.44→0.56 | +0.12 | 0 | 3 | 0.250 |
| C2 expansion (ca-flat → ca-hier) | 0.56→0.68 | +0.12 | 0 | 3 | 0.250 |
| C3 full (fixed-flat → ca-hier) | 0.44→0.68 | +0.24 | 0 | 6 | **0.031** |
| C4 expansion-on-fixed (fixed-flat → fixed-hier) | 0.44→0.40 | -0.04 | 1 | 0 | 1.000 |

- **Full method** (0.44 → 0.68, +0.24) is significant (**p=0.031**); all 6 discordant pairs favor the full method (b=0).
- **Components are individually monotone** — no query regresses under either step (b=0 for C1, b=0 for C2) — but neither clears significance alone at this n (p=0.250 each). This is what motivates the Phase-4 larger sweep.
- **Expansion on fixed chunks** shows no improvement (-0.04, p=1.000) → the gain
  is contingent on content-aware boundaries.

The **strict monotonicity** — zero regressions (b=0) across C1, C2, **and** C3 — is the key structural
evidence at this pilot scale: every contrast moves one way only. The better-powered companion sits one
level down, at retrieval: the Phase-2 paired Wilcoxon on per-query DCG@10 (p=0.0016, 18/2
better/worse) is what the QA numbers rest on.

### Cost
**Equal vision-token budget.** content_aware and fixed are compared at matched reader budget: mean
vision-tokens/page **9834** (fixed) vs **9805** (content_aware), **-0.3%** (Phase 2). The
QA gain is therefore not bought with more reader tokens — content_aware uses more, smaller,
content-aligned chunks at the same total pixels.

**Measured sweep cost (this run).** Reader at `detail=low`: ~426 prompt tokens/query
at 4 flat tiles, ~752 at hier-expand; **56,787** reader prompt tokens total across the
4 cells, plus **181,012** judge prompt tokens (text-only VLM-judge). Per-cell token usage is recorded in
each `results/qa_*.json`.

**Detail-mode token fix.** OpenAI defaults image inputs to `detail=high`, which tiles a tall full-page
screenshot into many crops. A 3-query smoke probe on **`gpt-4o-mini`** (a per-image-detail
probe — *not* the sweep model) measured **109,644** prompt tokens/query at `detail=high` vs
**11,416** at `detail=low` — a ≈10× (9.6×) reduction. The **ratio** is
the transferable finding; the absolute mini-smoke counts are not the gpt-4o sweep rate (the committed
totals above report the gpt-4o sweep at ~426 flat / ~752 hier tokens/query). At
list prices this is the difference between an illustrative ≈$37 (`detail=high`) and
≈$4 (`detail=low`) full sweep (gpt-4o input pricing × projected tokens — illustrative,
**not** a measured/logged cost). Answers were observed identical to `detail=high` on the smoke (a manual
spot-check, not a logged metric).

### Built vs. not built
- **Built.** Hierarchical retrieve-then-expand: a retrieved tile → same-section siblings + ±1 reading-order neighbors (deduped, capped), over a page→section→tile tree derived from the
  chunker's existing metadata. No re-embedding, no re-render.
- **Not built (future work).** Cross-reference / dependency-graph tracing — following anchors like
  "see Table 3" or DOM internal links to pull in referenced regions. No claim of full dependency
  tracing or complete context is made.

### Limitations (honest)
- **Pilot scale.** n=25 queries, single corpus; QA significance is limited by n (the C1/C2 p=0.250). The retrieval-level result is better powered; a larger QA sweep is Phase 4.
- **Single configuration.** Image modality only; one hosted reader (gpt-4o); judge grading on short
  descriptive answers.

### Positioning
A controlled **port + ablation + efficiency study**, not a new method. Content-aware visual chunking
and retrieval hierarchy already exist (M3DocDep; ColChunk / visual late chunking; MHier-RAG), and
PixelRAG established pixel-native RAG. The contribution here is a cost-neutral ablation showing that
content-aware chunking and section-expansion each contribute and are **synergistic**, and that the
Phase-2 retrieval gain **translates end-to-end to QA**.

### Reproduce
Per cell: `python qa_eval.py --results results/<arm>.json --tiles-dir <tiles> --gold
research/mini_corpus/gold.jsonl --retrieval {flat,hier-expand} --reader openai --reader-detail low`
→ `results/qa_<chunker>_<retrieval>.json`. Significance: `python research/qa_significance.py` →
`results/qa_significance.json`. This section: `python research/gen_phase3_experiments.py`.
<!-- PHASE3:END -->
