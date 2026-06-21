# Phase 1 — reproduce the baseline (GPU/Linux box)

Self-contained **EncyclopedicVQA (iNaturalist subset)** mini-corpus baseline: render gold Wikipedia
species pages → chunk (fixed 1024px) → embed → local FAISS → score retrieval + cost.
**Image queries are the primary retrieval metric** (EVQA questions are deictic — "this plant" — so
the query *photo* is the real signal); text queries are a secondary floor. **No reader** (Phase 4).
Seeded (42); provenance stamped into `results/baseline.json`.

Why iNaturalist (not landmarks): landmark query images (GLDv2) are Wikimedia Commons photos that can
overlap the gold page → image-retrieval **leakage**. iNaturalist query photos come from iNaturalist,
independent of Wikipedia/Commons (verified by a perceptual-hash leakage check, step 2b).

The corpus selection (`research/mini_corpus/{urls.txt,gold.jsonl,manifest.json}`) is committed and
deterministic — re-running step 1 reproduces it (gold-set md5 `b3418aa1…`).

## 0. Prereqs
```bash
uv sync --extra index --extra serve     # render + chunk + embed + build-index + serve
# first embed/serve run downloads Qwen/Qwen3-VL-Embedding-2B (~5GB) to the HF cache
```
Plus a Chrome/Chromium binary for rendering (`CHROME_PATH` if not auto-discovered).

## 1. Build the labeled mini-corpus (stdlib; downloads EVQA val.csv ~1.1MB)
```bash
uv run python research/build_mini_corpus.py --device cuda   # default --dataset-filter inaturalist
#   → research/mini_corpus/{urls.txt(40 pages), gold.jsonl(25 species queries), pixelrag.yaml}
```

## 2. Query images (PRIMARY metric) — iNaturalist API representative photos
```bash
# 2a. one representative photo per gold SPECIES via the iNaturalist API (leak-free, no big download;
#     auto-downloads iNat2021 val.json ~9.8MB for the species names). NOT the EVQA-exact photo.
uv run python research/fetch_query_images.py \
  --gold research/mini_corpus/gold.jsonl --inat-source api
#   → research/mini_corpus/query_images/{qid}.jpg + manifest.json   (dev box: 25/25 resolved)

# 2b. leakage check — query photos must NOT be near-duplicates of any gold-page Wikipedia image
uv run python research/check_leakage.py --gold research/mini_corpus/gold.jsonl
#   → must print "PASS"; writes query_images/leakage.json (dev box: 0/25 leaks, min Hamming ≥ 12)
```
> EVQA-exact alternative: `--inat-source val-tar` uses the *exact* EVQA photo, but needs the iNat2021
> `val.tar.gz` (~8.93GB) extracted under `--inat-data-dir` (no per-image URLs exist for the
> competition set). The open-data S3 per-photo URL is a **different id space** and is intentionally
> NOT used (it would return wrong-species photos).

## 3. Build the index (render → chunk[--chunker fixed] → embed → FAISS)
```bash
uv run pixelrag index build -c research/mini_corpus/pixelrag.yaml
#   → research/mini_corpus/output/{tiles/, embeddings/, index.faiss, metadata.npz, articles.json}
```

## 4. HARD GATE — `--chunker fixed` byte-identical to unflagged (run BEFORE eval)
```bash
uv run python research/check_chunker_noop.py --tiles-dir research/mini_corpus/output/tiles
#   → must print "PASS:"; exits non-zero on any byte/manifest diff. (Proven on synthetic tiles: PASS.)
```

## 5. Score → results/baseline.json (image primary + text floor)
```bash
uv run python run_eval.py \
  --index-dir research/mini_corpus/output \
  --gold research/mini_corpus/gold.jsonl \
  --query-images-dir research/mini_corpus/query_images \
  --device cuda --k 1,3,5,10 --chunker fixed \
  --out results/baseline.json
```
Records `retrieval.image.{recall@k,dcg@k,ndcg@k}` (primary) + `retrieval.text.{...}` (floor),
`query_modality` (image coverage), and `cost` (tiles/page, vision-tokens/page, chunk-height dist,
index size, latency). Use **DCG@k + Recall@k** for cross-arm comparison (arm-invariant); nDCG@k is
per-arm normalized — see `EXPERIMENTS.md`.

## Notes
- This is the *fixed* (baseline) control. `--chunker content-aware` (Phase 2) is scored on the SAME
  corpus + same image/text queries.
- Query photos are same-species iNaturalist representatives (leak-free), **not** EVQA-exact; the
  retrieval task is "photo of species X → find X's Wikipedia page". For EVQA-exact photos use
  `--inat-source val-tar` (8.93GB).
- CPU fallback: drop `--device cuda` (uses `embed_cpu`); expect 30–90+ min on the embed.
