#!/usr/bin/env python3
"""Augment an already-rendered corpus with DOM region boxes — WITHOUT re-rendering the images.

For the content-aware chunker we need block-element bounding boxes (regions.json) beside each
page's tiles. Re-rendering the screenshots is expensive and unnecessary: this does a navigate-only
CDP pass (no screenshots) at the SAME viewport width the tiles were rendered at, runs
``cdp.REGIONS_EXPR``, and writes ``regions.json`` next to the existing ``tiles.json``. Page layout
is deterministic at a fixed width, so the captured boxes line up with the rendered tiles.

Reuses the (fixed) renderer: ``cdp._connect_cdp`` (page-target selection), ``cdp.BROWSER_ARGS``,
``cdp.capture_regions``, ``cdp._readiness_expr``. Does NOT modify the renderer.

Run (repo root):
    PYTHONPATH=render/src .venv-mac/bin/python research/regions_only.py \
        --corpus research/mini_corpus --out research/mini_corpus/output --workers 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import subprocess
from pathlib import Path

VIEWPORT_W = 875  # must match the width the tiles were rendered at


async def _capture_one(ws, msg_id_ref, url, viewport_w, wait_network_idle):
    from pixelrag_render.backends import cdp

    await cdp._cdp_send(ws, msg_id_ref, "Page.enable")
    await cdp._cdp_send(
        ws, msg_id_ref, "Emulation.setDeviceMetricsOverride",
        {"width": viewport_w, "height": 8192, "deviceScaleFactor": 1, "mobile": False},
    )
    await cdp._cdp_send(ws, msg_id_ref, "Page.navigate", {"url": url})
    # Same settle as the renderer (load + fonts + capped rAF), then capture regions at scroll 0.
    await cdp._cdp_send(
        ws, msg_id_ref, "Runtime.evaluate",
        {"expression": cdp._readiness_expr(wait_network_idle), "awaitPromise": True, "returnByValue": True},
    )
    return await cdp.capture_regions(ws, msg_id_ref)


async def _worker(chrome, port, queue, viewport_w, wait_network_idle, stats):
    from pixelrag_render.backends import cdp

    proc = subprocess.Popen(
        [chrome, f"--remote-debugging-port={port}", "--headless=new", f"--user-data-dir=/tmp/regions_{port}"]
        + cdp.BROWSER_ARGS + ["about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        await asyncio.sleep(3)
        ws = await cdp._connect_cdp(port)
        mid = [0]
        while True:
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            tile_dir, url = item["tile_dir"], item["url"]
            try:
                regions = await _capture_one(ws, mid, url, viewport_w, wait_network_idle)
                regions.setdefault("url", url)
                (tile_dir / "regions.json").write_text(json.dumps(regions))
                stats["ok"] += 1
            except Exception as e:
                stats["fail"] += 1
                print(f"  FAIL {tile_dir.name}: {str(e)[:140]}")
        await ws.close()
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> None:
    ap = argparse.ArgumentParser(description="Add regions.json to an already-rendered corpus (no re-render).")
    ap.add_argument("--corpus", default="research/mini_corpus", help="Dir with urls.txt (id order).")
    ap.add_argument("--out", default=None, help="Output dir with tiles/ (default: <corpus>/output).")
    ap.add_argument("--viewport-width", type=int, default=VIEWPORT_W)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--wait-network-idle", action="store_true")
    ap.add_argument("--force", action="store_true", help="Recapture even if regions.json exists.")
    args = ap.parse_args()

    from pixelrag_render.chrome import find_chrome

    corpus = Path(args.corpus).resolve()
    out = Path(args.out).resolve() if args.out else corpus / "output"
    tiles = out / "tiles"
    urls = [
        l.strip() for l in (corpus / "urls.txt").read_text().splitlines()
        if l.strip() and not l.startswith("#")
    ]
    # id == position in urls.txt (matches render_local / pipelines.build)
    queue: asyncio.Queue = asyncio.Queue()
    todo = 0
    for i, url in enumerate(urls):
        tile_dir = tiles / f"{i}.png.tiles"
        if not tile_dir.exists():
            print(f"  skip id={i}: no tiles dir")
            continue
        if (tile_dir / "regions.json").exists() and not args.force:
            continue
        queue.put_nowait({"tile_dir": tile_dir, "url": url})
        todo += 1
    print(f"[regions] capturing regions for {todo}/{len(urls)} pages (viewport={args.viewport_width}) → {tiles}")

    chrome = find_chrome()
    stats = {"ok": 0, "fail": 0}

    async def run():
        n = max(1, min(args.workers, todo)) if todo else 0
        await asyncio.gather(
            *[_worker(chrome, 9500 + w, queue, args.viewport_width, args.wait_network_idle, stats)
              for w in range(n)],
            return_exceptions=True,
        )

    if todo:
        asyncio.run(run())
    print(f"[regions] DONE: {stats['ok']} ok, {stats['fail']} failed.")


if __name__ == "__main__":
    main()
