# PixelRAG — Architecture Notes (Phase 0 orientation)

Map of the pixel-native pipeline, written before any code changes. Goal: pin down exactly where
**tiling**, **index build**, **retrieval**, and the **reader** live, plus the eval/datasets and the
smallest end-to-end run. All paths are relative to the repo root; `file:line` references were read
directly from the code.

---

## Full flow

```
render (pixelshot)      chunk            embed              build-index        serve            read
URL/PDF ─► tiles  ──►  1024px chunks ─► vectors (.npz) ──► FAISS IVF + meta ─► /search ──► VLM reads images
   8192px strips        (RETRIEVAL UNIT)  Qwen3-VL-2B        index.faiss        top-k Hits     (never text)
```

Orchestrated end-to-end by `pixelrag index build`
([index/src/pixelrag_index/pipelines.py:14 `build`](index/src/pixelrag_index/pipelines.py#L14)):
**source → render → chunk → embed → build-index**.

### ⚠️ Where the "fixed pixel-height" cut actually is
- The renderer's **8192px "tiles"** are just a capture-size limit — a handful of big strips covering the page.
- **`pixelrag chunk` slices each tile into fixed 1024px strips**, and *those chunks are the unit that gets
  embedded, retrieved, and read.* **This 1024px cut is what splits tables/paragraphs mid-tile** and is the
  target of the content-aware chunker — not the 8192px render step.

---

## (1) Tiling / chunking

### Render → tiles (capture strips, 8192px)
| Path | File · function | Notes |
| --- | --- | --- |
| CDP standard | [render/src/pixelrag_render/backends/cdp.py:188 `capture_url`](render/src/pixelrag_render/backends/cdp.py#L188) | Fixed-height slice loop at **L230–283**; `tile_h: int = 8192` (L194). JS via `Runtime.evaluate` for readiness/scroll (L211, L243). |
| CDP turbo | [render/src/pixelrag_render/backends/fast_cdp.py](render/src/pixelrag_render/backends/fast_cdp.py) | `TILE_HEIGHT = 8192`; same per-tile clip logic. |
| PDF | [render/src/pixelrag_render/backends/pdf.py:16 `render_pdf`](render/src/pixelrag_render/backends/pdf.py#L16) | `pdf2image`, one image per page → JPEG. |
| CLI entry | [render/src/pixelrag_render/render.py `main`](render/src/pixelrag_render/render.py) → `render_urls` | `pixelshot` (pyproject `project.scripts`). `--tile-height` default 8192. |

**On disk:** `{stem}.png.tiles/tile_NNNN.jpg` (JPEG q85) + `tiles.json`
(`{url, page_height, tiles[], complete}`), written at [cdp.py:285](render/src/pixelrag_render/backends/cdp.py#L285).

### Chunk → 1024px strips (the retrieval/reading unit)
- [embed/src/pixelrag_embed/chunk.py:63 `chunk_article`](embed/src/pixelrag_embed/chunk.py#L63) — slice loop
  **L168–200**. Constants: `CHUNK_HEIGHT = 1024` ([L45](embed/src/pixelrag_embed/chunk.py#L45)),
  `MIN_CHUNK_HEIGHT = 28` (one Qwen3-VL patch; tiny tails dropped).
- **On disk:** `chunk_TTTT_CC.png` + `chunks.json`
  ([written L208–221](embed/src/pixelrag_embed/chunk.py#L208)).
- CLI: `pixelrag chunk` → [chunk.py:329 `main`](embed/src/pixelrag_embed/chunk.py#L329)
  (`--shard-dir | --tiles-dir`, `--workers`, `--force`, …).
- **No chunker abstraction today** — the 1024px height is hardcoded. (The `render/.../strategies/` dir is
  about *how to drive Chrome*, not how to slice — unrelated.)

### chunks.json contract (the key compatibility surface)
[embed.py:288 `scan_shard_chunks`](embed/src/pixelrag_embed/embed.py#L288) reads **only** these per-chunk
fields — `file`, `tile_index`, `chunk_index`, `y_offset`, `height` — plus manifest `page_height`,
`viewport_width`. **Any extra keys are ignored.** → A content-aware chunker can add
`heading_path / region_ids / bbox / reading_order` and keep the chunk **image format identical**, so
**embed / index / serve require zero changes**.

---

## (2) Index build

| Step | File · function | Output |
| --- | --- | --- |
| Embed (GPU) | [embed/src/pixelrag_embed/embed.py](embed/src/pixelrag_embed/embed.py) (vllm/sglang/direct_gpu) | `shard_NNN.npz` |
| Embed (CPU) | [embed/src/pixelrag_embed/embed_cpu.py](embed/src/pixelrag_embed/embed_cpu.py) — used by the demo on CPU | `shard_NNN.npz` |
| Build | [embed/src/pixelrag_embed/index.py:142 `build_ivf`](embed/src/pixelrag_embed/index.py#L142) | `index.faiss`, `metadata.npz`, `summary.json` |

- Model `Qwen/Qwen3-VL-Embedding-2B`, **2048-d**, **L2-normalized** vectors.
- FAISS `IndexIVFFlat`, metric **inner product** (cosine on unit vectors); `nlist` auto-set for small corpora
  ([pipelines.py:175](index/src/pixelrag_index/pipelines.py#L175): `min(4096, max(1, n//40))`).
- `metadata.npz` arrays (index order): `article_ids, tile_indices, chunk_indices, y_offsets, tile_heights`.
- CLI: `pixelrag embed`, `pixelrag build-index`, `pixelrag index` — dispatched by
  [src/pixelrag/cli.py:15 `STAGES`](src/pixelrag/cli.py#L15).

---

## (3) Retrieval / serve

[serve/src/pixelrag_serve/api.py](serve/src/pixelrag_serve/api.py) — FastAPI (`pixelrag serve`):
- `POST /search` (`search()` ~L398): encode query (`_encode_queries` ~L251 — **text or base64 image**, same
  Qwen3-VL model, last-token pool + L2 norm) → `index.search(q, k)` → `Hit`s with
  `score, vector_id, article_id, tile_index, chunk_index, y_offset, tile_height, path, url`. `n_docs`
  controls k; `nprobe`, `min_tile_height`, `include_images`, `articles_only` are options.
- `GET /tile/{article_id}/{tile_index}/{chunk_index}` serves a chunk PNG (`_resolve_path` ~L330).
- Loads `index.faiss` + `metadata.npz` + `articles.json` at startup (`load()` ~L571).

---

## (4) Reader (VLM) — reads images, never parsed text

| Surface | File | Reader |
| --- | --- | --- |
| Web chat agent | [web/agent-server.mjs](web/agent-server.mjs) | Claude (`sonnet`) calls `pixelrag_search` + `pixelrag_tile`, reads the **tile images** (system prompt ~L54). |
| Eval reader | [eval/lib/llm.py](eval/lib/llm.py) `build_messages` | OpenAI-compatible / Gemini VLM; paper reader `Qwen/Qwen3.5-4B`. Builds *question + retrieved tile images*. |

---

## Eval scripts, metrics & datasets

### Scripts ([eval/](eval/))
- [eval/run_bench.py](eval/run_bench.py) — universal QA evaluator; retrieval modes incl. `--local-api`
  (pixel search API), `--text-api`, ground-truth screenshot, etc. Output: JSONL per example.
- [eval/run_livevqa.py](eval/run_livevqa.py) (news MCQ), [eval/run_monaco.py](eval/run_monaco.py) (multi-hop ReAct).
- [eval/reproduce.sh](eval/reproduce.sh) — single-cell Table-1 runner: `bash reproduce.sh <bench> <retrieval>`
  (needs reader + search serves up).

### Metrics
- **Retrieval = article-level Recall@k**: a hit if the gold URL appears in the top-k tiles' URLs —
  [run_bench.py:784–815](eval/run_bench.py#L784); gold via `gt_url = extract_url_from_metadata(...)` then
  `gt_url in retrieved_url` ([L792, L810](eval/run_bench.py#L792)).
- **QA accuracy**: [eval/lib/grader.py](eval/lib/grader.py) — exact-match (NQ/TriviaQA) or LLM judge
  (SimpleQA/WorldVQA), seed 42.
- **No nDCG@k / qrels exist today.** (Plan: article-level *binary* nDCG@k in Phase 1; answer-bearing-tile
  nDCG@k in Phase 4.) **No boundary-violation metric exists** — new in Phase 4.

### Datasets ("six benchmarks")
SimpleQA, NQ, NQ-Tables, MMSearch, **EncyclopedicVQA (EVQA)**, LiveVQA — loaders in
[eval/lib/benchmarks.py](eval/lib/benchmarks.py). Full runs need the hosted FAISS indexes (217G+) — out of
scope here. **EVQA is the only loader exposing a clean `(question → gold wikipedia_url)`**
([benchmarks.py:131–158](eval/lib/benchmarks.py#L131)), so it is the renderable, self-contained baseline.
**No tile/zim fixtures are committed**; the small-subset path renders fresh.

---

## Running the baseline end-to-end on a SMALL subset (CPU)

**Mandatory download:** `Qwen/Qwen3-VL-Embedding-2B` (~5GB) — required to embed and to encode queries.

**Existing demo path** ([demos/e2e/run.py](demos/e2e/run.py) +
[demos/e2e/pixelrag.yaml](demos/e2e/pixelrag.yaml)):
```bash
uv sync --extra index --extra serve            # CPU; torch-CUDA is Linux-pinned, so this box uses embed_cpu
uv run python demos/e2e/run.py --limit 30 --device cpu
# renders 30 Simple-Wikipedia pages (~1GB ZIM auto-download) → chunk → embed_cpu → FAISS → serves :31337
# → runs 8 sample queries. NOTE: those sample queries have NO gold labels.
```

**Planned labeled baseline (Phase 1, this paper):** instead of the unlabeled demo queries, build a small
**EVQA mini-corpus** — render ~20–50 EVQA gold `wikipedia_url`s + a few distractors via the `web` source
([index/src/pixelrag_index/sources/web.py](index/src/pixelrag_index/sources/web.py)), build a local index,
and score article-level **Recall@k** + **binary nDCG@k** + cost metrics. A table/layout-stress probe set
(Phase 2) provides the chunking-quality signal that EVQA, being table-sparse, cannot.

---

## What changes for the paper (preview — gated per phase)
- **Phase 1:** `--chunker fixed` (no-op alias) + EVQA mini-corpus + `run_eval.py` (retrieval + cost only).
- **Phase 2:** `embed/src/pixelrag_embed/chunking/content_aware.py` (`--chunker content-aware`,
  `--region-source {dom,vision}`); DOM regions captured additively in render → `regions.json`; greedy packer
  never splits a region (table → split at `<tr>` + repeat header; figure → own tile); **+ table-stress probe
  set** and a **CSS-px → render-px reconciliation guardrail**.
- **Phase 3:** `--retrieval hier-expand` (page→section→tile tree + reading-order/cross-ref edges; expand after
  flat top-k). **Phase 4:** sweep + boundary-violation rate + figures + `EXPERIMENTS.md`.

All new behavior is **OFF by default behind flags**; the unmodified baseline stays runnable; the reader is
**only** ever shown images.
