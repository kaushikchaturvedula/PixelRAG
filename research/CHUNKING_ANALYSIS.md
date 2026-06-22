# Why did content-aware (CA) chunking help on iNat but not NQ / NQ-Tables?

Zero-compute forensics on existing `results/*.json` (no rendering / embedding / API / new corpus).
Numbers reproduced by `research/analyze_chunking.py`. Reader = gpt-4o (detail=low); grading =
VLM-judge for iNat, exact-match for NQ/NQ-Tables.

## TL;DR
**The CA chunker was a near-exact no-op on NQ and NQ-Tables** — it produced byte-identical tile
counts and heights to the fixed chunker (1680 = 1680 chunks; NQ-Tables `fixed_flat` == `ca_flat`
accuracy to 4 decimals). So their "CA is neutral" result carries **no signal about content-aware
chunking** — the input never changed. **iNat is the only benchmark where CA actually altered the
tiles** (+24.5% chunks, heights 4× more variable), but it is *also* the only image-primary,
image-query benchmark, so chunk-difference, document type, and query modality are **perfectly
confounded**. Separately, all three best-CA cells are **reader-bound** (78–100% of misses had the
gold tile in front of the reader), so QA headroom is limited by the reader, not retrieval/chunking.

---

## Q1 — Did the chunker actually do anything? **(the key question)**

| benchmark | pages | total chunks fixed→CA (Δ) | tiles/page fixed→CA | chunk-height px mean±std fixed→CA | CA vs fixed |
|---|---|---|---|---|---|
| **iNat** | 40 | 335 → **417 (+82, +24.5%)** | 8.38±1.02 → 10.43±1.43 (**+2.05**) | 1015±82 → **809±329** (Δmean −206, Δstd +247) | **DIFFERS materially** |
| **NQ-Tables** | 210 | 1680 → 1680 (**+0, +0.0%**) | 8.00±0.00 → 8.00±0.00 (+0.00) | 1023.7±2.1 → 1023.7±2.1 (Δ −0.0) | **NO-OP (identical)** |
| **NQ** | 210 | 1680 → 1680 (**+0, +0.0%**) | 8.00±0.00 → 8.00±0.00 (+0.00) | 1023.8±1.7 → 1023.8±1.7 (Δ +0.0) | **NO-OP (identical)** |

**Answer: hypothesis confirmed.** On NQ and NQ-Tables the two chunkers are identical — same chunk
count, same per-page tile count (every page = **exactly 8 tiles of ~1024 px = 8192 px**, std 0), same
height distribution. CA changed nothing. On iNat CA differs materially: pages render to varying
heights (8–12 tiles), so CA cuts +24.5% more, smaller, content-aligned chunks (height std jumps
82→329 px).

**Likely mechanism (hypothesis — not verifiable from result JSONs alone):** every NQ/NQ-Tables page
rendered to a uniform 8192 px capture (the default `tile_height`), giving CA no heterogeneity to
exploit — and/or DOM regions were unavailable for those web renders, so CA degraded to fixed 1024 px
strips. Either way, **CA was never meaningfully exercised on the text corpora.** "CA neutral on
NQ/NQ-Tables" therefore means "CA did not run," **not** "CA doesn't help on Wikipedia text."

---

## Q2 — Reader ceiling vs retrieval ceiling on the misses

Best-CA cell per benchmark; each non-correct query classified by whether the gold page
(`gold_article_id`) was among the tiles the reader actually saw (`tiles` in the QA JSON).
(a) = gold tile **was** read but answered wrong (reader ceiling); (b) = gold tile **not** in the
reader's tiles (retrieval ceiling).

| benchmark | best CA cell (acc) | misses | (a) reader ceiling | (b) retrieval ceiling | (a):(b) | bound by |
|---|---|---|---|---|---|---|
| **iNat** | qa_ca_hier (0.68) | 8 | **8** | 0 | **8:0** | reader (**100%**) |
| **NQ-Tables** | qa_nqt_ca_hier (0.52) | 72 | **56** | 16 | **56:16** | reader (**78%**) |
| **NQ** | qa_nq_ca_flat (0.613) | 58 | **46** | 12 | **46:12** | reader (**79%**) |

**Answer: all three are reader-bound.** On iNat, retrieval was *perfect* on the misses (gold tile
always read) yet the reader still got 8/8 wrong — 0 retrieval-ceiling misses, so better
retrieval/expansion **cannot** raise iNat QA at all. On the text benchmarks ~78–79% of misses already
had the gold tile in front of the reader; only ~21% are retrieval-ceiling (16/16 and 11/12 of those
weren't even in the stored top-k, i.e. genuinely not retrieved, not merely cut by `reader_top_k`).
**The QA bottleneck is the reader, not chunking/retrieval** — which caps how much any chunking change
could move these numbers.

---

## Q3 — The image-query vs text-query confound

| benchmark | image queries | text queries | CA engaged? (Q1) |
|---|---|---|---|
| **iNat** | **25** | 0 | **yes** (tiles differ) |
| **NQ-Tables** | 0 | **150** | no (no-op) |
| **NQ** | 0 | **150** | no (no-op) |

**Answer: confirmed and confounded — triply so.** iNat QA is 100% image-primary queries; NQ/NQ-Tables
are 100% text queries (0 image). So iNat differs from the Wikipedia-text benchmarks on **three axes
that all co-vary**: (1) image-primary / heterogeneous-layout documents, (2) image-based retrieval, and
(3) CA actually altered the tiles. We **cannot** tell whether the iNat gain comes from image-primary
documents, from image-query retrieval, or from the chunker doing something — these are perfectly
aliased across our benchmark set.

---

## Q4 — Where expansion (flat→hier) helped vs hurt

| benchmark | arm | flat | hier | Δ (hier − flat) |
|---|---|---|---|---|
| iNat | fixed | 0.4400 | 0.4000 | **−0.0400** |
| iNat | **ca** | 0.5600 | 0.6800 | **+0.1200** |
| NQ-Tables | fixed | 0.4933 | 0.5400 | +0.0467 |
| NQ-Tables | ca | 0.4933 | 0.5200 | +0.0267 |
| NQ | fixed | 0.6400 | 0.5933 | −0.0467 |
| NQ | ca | 0.6133 | 0.6067 | −0.0066 |

fixed-vs-CA **at flat** (same depth — isolates whether CA changed anything): iNat **+0.1200**,
NQ-Tables **+0.0000** (identical → no-op smoking gun), NQ −0.0267 (CA slightly *worse* despite
identical tiles ⇒ embedding/index nondeterminism, not a chunking effect).

**Answer: no consistent pattern; the iNat CA-synergy is unique and does not replicate.** Expansion
helps a lot only for iNat-CA (+0.12); it is mildly positive on NQ-Tables (both arms), mildly negative
on NQ (both arms). The Phase-3 "CA enables expansion synergy" story (section+neighbors ≫
neighbors-only) is actively *contradicted* on NQ-Tables, where neighbors-only `fixed` expansion
(+0.047) beat section+neighbors `ca` expansion (+0.027) — opposite ordering. Because CA == fixed tiles
on both text benchmarks, their fixed-vs-CA splits are comparisons of essentially the same input and
should not be read as chunking effects.

---

## Q5 — Verdict

**Most defensible single claim (closest to (C), with (A) as the mechanism):**

> We cannot attribute the iNat QA gain to content-aware chunking. The two text-query benchmarks that
> could have decoupled chunking from query modality were **no-ops** — CA produced tiles identical to
> fixed (1680 = 1680 chunks; NQ-Tables `fixed_flat` == `ca_flat` to 4 decimals) — so they provide
> **zero evidence** about CA. iNat is the only benchmark where CA changed the input, but it is
> simultaneously the only image-primary, image-query benchmark, so chunk-difference, document type,
> and query modality are perfectly confounded. On top of that, every best-CA cell is reader-bound
> (78–100% of misses already had the gold tile read), so the QA ceiling is the reader, not chunking.

Plainly: **(A) is necessary but untested on text** (CA never engaged there), and **(B) cannot be
ruled out** (iNat is the lone image-query case) — so the honest verdict is **(C): the iNat result is
confounded and not yet attributable.**

**The one experiment that would disambiguate (do NOT run — just named):** **re-run the existing iNat
QA cells with TEXT queries** (`--modality text`; iNat text-query retrieval `hits_text` already exists
in `content_aware.json` / `baseline_clean.json`). This holds the iNat corpus *and* the CA-vs-fixed
chunk difference constant and flips **only** the query modality image→text. If CA's advantage
survives under text queries, the gain is the chunking/document (supports A); if it vanishes, the gain
was a modality / image-query effect (supports B). It is the cheapest decisive test — it reuses the
built iNat index and needs only a reader pass over the already-retrieved text-query tiles, no
rendering or re-embedding. (A costlier complement: re-render NQ/NQ-Tables **uncapped** so CA actually
differs from fixed, then re-run text-query QA — tests CA-on-text when the chunker genuinely engages.)
