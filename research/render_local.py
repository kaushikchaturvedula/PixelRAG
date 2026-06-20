#!/usr/bin/env python3
"""Render + chunk the mini-corpus LOCALLY (e.g. on macOS, where the standard CDP path works
because there is a real GPU/display), producing a self-contained ``output/`` that another
machine (e.g. Kaggle GPU) can embed → build-index → run_eval with NO re-rendering.

Reproduces pipeline stages 1-2 of ``pixelrag_index.pipelines.build`` (render → chunk): it assigns
integer article ids (0..N-1) in urls.txt order and writes ``articles.json`` exactly like the
orchestrator, then stops before embed. Reuses ``pixelrag_render.render.render_urls`` and the
``pixelrag chunk`` stage unchanged.

Run (from the repo root, with the render deps on the path):
    PYTHONPATH=render/src:embed/src .venv-mac/bin/python research/render_local.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Render+chunk the mini-corpus locally (no embed).")
    ap.add_argument("--corpus", default="research/mini_corpus", help="Dir holding urls.txt + gold.jsonl.")
    ap.add_argument("--out", default=None, help="Output dir (default: <corpus>/output).")
    ap.add_argument("--tile-height", type=int, default=8192)
    ap.add_argument("--quality", type=int, default=85)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunker", default="fixed", choices=["fixed"])
    args = ap.parse_args()

    from pixelrag_render.render import render_urls

    corpus = Path(args.corpus).resolve()
    out = Path(args.out).resolve() if args.out else corpus / "output"
    tiles = out / "tiles"
    tiles.mkdir(parents=True, exist_ok=True)

    # urls.txt in file order == the article-id order pipelines.build assigns (web source, '#' skipped)
    urls = [
        l.strip()
        for l in (corpus / "urls.txt").read_text().splitlines()
        if l.strip() and not l.startswith("#")
    ]
    stems = [str(i) for i in range(len(urls))]
    print(f"[render] {len(urls)} corpus pages → {tiles}", flush=True)

    # Stage 1: render (skip already-rendered, like pipelines.build) → {id}.png.tiles/
    todo = [
        (u, s) for u, s in zip(urls, stems)
        if not (tiles / f"{s}.png.tiles" / "tiles.json").exists()
    ]
    if todo:
        render_urls(
            [u for u, _ in todo],
            str(tiles),
            stems=[s for _, s in todo],
            tile_height=args.tile_height,
            quality=args.quality,
            workers=args.workers,
        )
    print(f"[render] rendered {len(todo)} (skipped {len(urls) - len(todo)} already present)", flush=True)

    # articles.json: id -> {title, url} in id order (mirror pipelines.build's mapping)
    articles = [
        {"title": u.split("/")[-1].replace("_", " ").replace("%20", " "), "url": u}
        for u in urls
    ]
    (out / "articles.json").write_text(json.dumps(articles))
    print(f"[render] wrote {out / 'articles.json'} ({len(articles)} entries)", flush=True)

    # Stage 2: chunk (fixed 1024px strips) → chunk_*.png + chunks.json (what embed consumes)
    subprocess.run(
        [sys.executable, "-m", "pixelrag_embed.chunk",
         "--shard-dir", str(tiles), "--workers", "8", "--chunker", args.chunker],
        check=True,
    )

    # sanity report
    dirs = sorted(tiles.glob("*.png.tiles"))
    n_chunks = sum(
        len(json.loads((d / "chunks.json").read_text()).get("chunks", []))
        for d in dirs if (d / "chunks.json").exists()
    )
    missing = [s for s in stems if not (tiles / f"{s}.png.tiles" / "chunks.json").exists()]
    print(f"\n[render] DONE: {len(dirs)}/{len(urls)} tile dirs, {n_chunks} chunks → {out}")
    if missing:
        print(f"[render] WARNING: {len(missing)} pages have no chunks.json (render failed): {missing}")
    else:
        print("[render] all pages rendered + chunked. Ready to bundle for Kaggle.")


if __name__ == "__main__":
    main()
