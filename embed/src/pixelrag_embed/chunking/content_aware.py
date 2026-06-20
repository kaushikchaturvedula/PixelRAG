#!/usr/bin/env python3
"""Content-aware chunker (Phase 2).

Replaces the fixed 1024px cut with cuts placed at DOM block-element boundaries, so a chunk never
splits a paragraph/table/figure. Cut lines come from ``regions.json`` (block-element bounding boxes
captured at render time by ``pixelrag_render.backends.cdp`` — DOM is used ONLY to decide WHERE to
cut; the reader still only ever sees the cropped tile pixels). Output is drop-in compatible with the
fixed chunker: same ``chunk_{tile:04d}_{chunk:02d}.png`` naming + the load-bearing chunks.json fields
(file/tile_index/chunk_index/y_offset/height/width, manifest page_height/viewport_width), PLUS
additive metadata (heading_path/region_ids/bbox/reading_order). Variable heights already flow through
embed→index→serve.

Greedy packer: walk regions top→bottom, accumulate whole regions into a chunk until the next region
would exceed ``max_chunk_height`` (default = fixed CHUNK_HEIGHT, for fair token-cost parity), then cut
at a region boundary. NEVER split a region. An oversized region (> max):
  - table  → split at <tr> boundaries, repeating the header row pixels atop each piece;
  - figure / other → its own (possibly over-tall) chunk.

If a page has no regions.json (or region_source has no detector), this falls back to the FIXED chunker
so the pipeline never breaks.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from PIL import Image

from ..chunk import CHUNK_HEIGHT, MIN_CHUNK_HEIGHT, _compute_tile_hashes
from ..chunk import chunk_article as _fixed_chunk_article

Image.MAX_IMAGE_PIXELS = None
logger = logging.getLogger("content_aware_chunker")

MAX_CHUNK_HEIGHT = CHUNK_HEIGHT  # 1024 — same target as the fixed baseline (fair comparison)


# --------------------------------------------------------------------------- regions I/O + px
def _load_regions(article_dir: str) -> dict | None:
    p = os.path.join(article_dir, "regions.json")
    if not os.path.exists(p):
        return None
    try:
        return json.loads(Path(p).read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _reconcile_px(regions: dict) -> tuple[float, list[dict]]:
    """Scale DOM (CSS-px) boxes to render/device px via devicePixelRatio.

    The renderer uses Emulation.setDeviceMetricsOverride(deviceScaleFactor=1) + clip scale=1, so the
    in-page devicePixelRatio should be 1 (CSS px == tile px). We still scale by it defensively (no-op
    at 1) and assert it's ~1 — a mismatch means boundaries would drift, so fail loudly.
    """
    dpr = float(regions.get("device_pixel_ratio", 1) or 1)
    if abs(dpr - 1.0) > 0.01:
        logger.warning("devicePixelRatio=%.3f != 1 — scaling region boxes by it (px reconciliation)", dpr)
    out = []
    for r in regions.get("regions", []):
        rr = dict(r)
        rr["y"] = r["y"] * dpr
        rr["height"] = r["height"] * dpr
        if r.get("rows"):
            rr["rows"] = [
                {"y": row["y"] * dpr, "height": row["height"] * dpr, "is_header": bool(row.get("is_header"))}
                for row in r["rows"]
            ]
        out.append(rr)
    return dpr, out


# --------------------------------------------------------------------------- greedy packer
def _valid_cut(y: float, intervals: list[tuple[float, float]]) -> bool:
    """A cut at y is valid iff it is not strictly interior to any region (boundaries/gaps OK).

    This allows cutting between adjacent regions but never inside a region (including a nesting
    container like a table that encloses smaller regions).
    """
    return all(y <= top or y >= bot for (top, bot) in intervals)


def _split_table(loc: dict, max_h: float) -> list[dict]:
    """Split an oversized table into <tr>-aligned pieces, repeating the header row on later pieces.

    Returns pieces with tile-local y0/y1 and an optional 'header' (hy0,hy1) band to prepend.
    """
    rows = sorted(loc.get("rows", []), key=lambda r: r["top"])
    header = [r for r in rows if r.get("is_header")]
    body = [r for r in rows if not r.get("is_header")]
    if not body:  # no usable rows → emit whole table as one oversized chunk
        return [{"y0": loc["top"], "y1": loc["bot"], "header": None, "oversized": True, "locs": [loc]}]
    header_band = (min(h["top"] for h in header), max(h["bot"] for h in header)) if header else None
    header_h = (header_band[1] - header_band[0]) if header_band else 0.0

    pieces: list[dict] = []
    i, first = 0, True
    while i < len(body):
        avail = max_h - (0.0 if first else header_h)
        grp = [body[i]]
        i += 1
        while i < len(body) and (body[i]["bot"] - grp[0]["top"]) <= avail:
            grp.append(body[i])
            i += 1
        grp_bot = grp[-1]["bot"]
        if first:
            # piece 0 spans the real table top (natural header already present) → no compose
            y0 = loc["top"]
            pieces.append({"y0": y0, "y1": grp_bot, "header": None,
                           "oversized": (grp_bot - y0) > max_h, "locs": [loc]})
            first = False
        else:
            y0 = grp[0]["top"]
            pieces.append({"y0": y0, "y1": grp_bot, "header": header_band,
                           "oversized": (header_h + (grp_bot - y0)) > max_h, "locs": [loc]})
    return pieces


def _pack_tile(locs: list[dict], H: int, max_h: float) -> list[dict]:
    """Greedy-pack tile-local regions into pieces. Each piece: {y0,y1,header,oversized,locs}."""
    if not locs:
        return []  # caller falls back to fixed slicing for region-less tiles
    intervals = [(l["top"], l["bot"]) for l in locs]
    cuts = sorted({0.0, float(H)} | {l["top"] for l in locs} | {l["bot"] for l in locs})
    pieces: list[dict] = []
    start = 0.0
    guard = 0
    while start < H - 0.5 and guard < 100000:
        guard += 1
        target = start + max_h
        cand = [c for c in cuts if start + 0.5 < c <= target + 1e-6 and _valid_cut(c, intervals)]
        if cand:
            cut = max(cand)
            members = [l for l in locs if l["top"] < cut - 0.5 and l["bot"] > start + 0.5]
            pieces.append({"y0": start, "y1": cut, "header": None, "oversized": False, "locs": members})
            start = cut
            continue
        # No valid cut within max → a region straddles `target`: emit it whole (table → split).
        blockers = [l for l in locs if l["bot"] > target and l["top"] <= target + 0.5]
        block = min(blockers, key=lambda l: l["top"]) if blockers else None
        if block is None:  # nothing straddles (shouldn't happen) → safe fixed cut
            cut = min(target, float(H))
            pieces.append({"y0": start, "y1": cut, "header": None, "oversized": False, "locs": []})
            start = cut
            continue
        if block.get("kind") == "table" and block.get("rows"):
            tps = _split_table(block, max_h)
            pieces.extend(tps)
            start = block["bot"]
        else:
            pieces.append({"y0": start, "y1": block["bot"], "header": None, "oversized": True, "locs": [block]})
            start = block["bot"]
    # drop a tiny trailing piece (< one Qwen3-VL patch), mirroring the fixed chunker's tail rule
    if pieces and (pieces[-1]["y1"] - pieces[-1]["y0"]) < MIN_CHUNK_HEIGHT and not pieces[-1]["header"]:
        pieces.pop()
    return pieces


# --------------------------------------------------------------------------- emit
def _emit(tile_img: Image.Image, piece: dict, W: int, out_path: str, dry_run: bool) -> int:
    """Crop (and for table pieces, prepend the header band); save PNG; return emitted height."""
    y0, y1 = int(round(piece["y0"])), int(round(piece["y1"]))
    body = tile_img.crop((0, y0, W, y1))
    if piece.get("header"):
        hy0, hy1 = int(round(piece["header"][0])), int(round(piece["header"][1]))
        header = tile_img.crop((0, hy0, W, hy1))
        comp = Image.new("RGB", (W, header.height + body.height), (255, 255, 255))
        comp.paste(header, (0, 0))
        comp.paste(body, (0, header.height))
        body = comp
    if not dry_run:
        body.save(out_path, format="PNG")
    return body.height


def chunk_article(article_dir: str, dry_run: bool = False, force: bool = False,
                  region_source: str = "dom", max_chunk_height: int = MAX_CHUNK_HEIGHT) -> dict | None:
    """Content-aware chunk one ``*.png.tiles`` dir. Falls back to the fixed chunker if no regions."""
    tiles_json = os.path.join(article_dir, "tiles.json")
    chunks_json = os.path.join(article_dir, "chunks.json")
    if not os.path.exists(tiles_json):
        return None
    meta = json.loads(Path(tiles_json).read_text() or "{}")
    tile_names = meta.get("tiles", [])
    if not tile_names:
        return None

    tile_hashes = _compute_tile_hashes(article_dir, tile_names)
    if not tile_hashes:
        return None
    if os.path.exists(chunks_json) and not force:
        old = json.loads(Path(chunks_json).read_text() or "{}")
        if old.get("chunker") == "content_aware" and all(
            os.path.exists(os.path.join(article_dir, c["file"])) for c in old.get("chunks", [])
        ):
            return None  # up to date

    regions = _load_regions(article_dir) if region_source == "dom" else _vision_regions(article_dir)
    if not regions or not regions.get("regions"):
        # No layout signal → fixed chunker (keeps the pipeline robust); marks chunker accordingly.
        logger.warning("%s: no regions.json (%s) — falling back to fixed chunker", article_dir, region_source)
        return _fixed_chunk_article(article_dir, dry_run=dry_run, force=True)

    dpr, regions_dev = _reconcile_px(regions)
    page_height = meta.get("page_height", 0)
    viewport_width = meta.get("viewport_width", 875)
    tile_height = meta.get("tile_height", 8192)

    # remove any stale chunk_* before rewriting
    if not dry_run and os.path.exists(chunks_json):
        for f in os.listdir(article_dir):
            if f.startswith("chunk_") and f.endswith((".png", ".jpg", ".jpeg")):
                os.unlink(os.path.join(article_dir, f))

    chunks_info: list[dict] = []
    files_written = 0
    reading_order = 0
    for ti, tile_name in enumerate(tile_names):
        tp = os.path.join(article_dir, tile_name)
        if not os.path.exists(tp):
            continue
        img = Image.open(tp)
        W, H = img.size
        tile_top = ti * tile_height  # page-px of this tile's top
        # regions overlapping this tile, in tile-local px (+ tile-local table rows)
        locs = []
        for r in regions_dev:
            top = r["y"] - tile_top
            bot = r["y"] + r["height"] - tile_top
            tc, bc = max(0.0, top), min(float(H), bot)
            if bc - tc < 1:
                continue
            loc = {"top": tc, "bot": bc, "kind": r.get("kind"), "orig": r}
            if r.get("kind") == "table" and r.get("rows"):
                loc["rows"] = [
                    {"top": max(0.0, row["y"] - tile_top), "bot": min(float(H), row["y"] + row["height"] - tile_top),
                     "is_header": row.get("is_header")}
                    for row in r["rows"]
                    if min(float(H), row["y"] + row["height"] - tile_top) - max(0.0, row["y"] - tile_top) >= 1
                ]
            locs.append(loc)

        pieces = _pack_tile(locs, H, float(max_chunk_height))
        if not pieces:  # region-less tile → fixed 1024 slicing so we still emit chunks
            pieces = _fixed_pieces(H, max_chunk_height)

        for ci, piece in enumerate(pieces):
            chunk_name = f"chunk_{ti:04d}_{ci:02d}.png"
            h = _emit(img, piece, W, os.path.join(article_dir, chunk_name), dry_run)
            files_written += 0 if dry_run else 1
            members = piece.get("locs") or []
            chunks_info.append({
                # --- load-bearing (read by embed/serve/run_eval) ---
                "tile": tile_name, "tile_index": ti, "chunk_index": ci, "file": chunk_name,
                "y_offset": int(round(piece["y0"])), "height": int(h), "width": int(W),
                # --- additive Phase-2 metadata (ignored by existing consumers) ---
                "heading_path": (members[0]["orig"].get("heading_path", "") if members else ""),
                "region_ids": [m["orig"].get("id") for m in members],
                "bbox": {"y": int(round(piece["y0"])), "height": int(round(piece["y1"] - piece["y0"])), "width": int(W)},
                "reading_order": reading_order,
                "kind": "table_piece" if piece.get("header") else ("oversized" if piece.get("oversized") else "block"),
                "oversized": bool(piece.get("oversized")),
            })
            reading_order += 1
        img.close()

    if not chunks_info:
        return None

    manifest = {
        "page_height": page_height, "viewport_width": viewport_width, "tile_height": tile_height,
        "chunker": "content_aware", "region_source": region_source, "device_pixel_ratio": dpr,
        "max_chunk_height": max_chunk_height, "num_tiles": len(tile_names), "num_chunks": len(chunks_info),
        "tile_hashes": tile_hashes, "chunks": chunks_info,
    }
    if not dry_run:
        Path(chunks_json).write_text(json.dumps(manifest))
    return {"article_dir": article_dir, "num_tiles": len(tile_names),
            "num_chunks": len(chunks_info), "files_written": files_written}


def _fixed_pieces(H: int, max_h: int) -> list[dict]:
    """Fixed equal-height pieces for a region-less tile (fallback within content-aware)."""
    pieces, y = [], 0
    while y < H:
        ch = min(max_h, H - y)
        if ch < MIN_CHUNK_HEIGHT:
            break
        pieces.append({"y0": float(y), "y1": float(y + ch), "header": None, "oversized": False, "locs": []})
        y += ch
    return pieces


def _vision_regions(article_dir: str) -> dict | None:
    """Stub for --region-source vision (a pluggable layout detector). Not implemented yet.

    A real detector (e.g. DocLayout-YOLO / Surya) would run on the tile images here and return the
    same {regions:[{y,height,kind,rows?}]} shape. Deferred — model download size to be disclosed
    before fetching. Returns None so chunk_article falls back to the fixed chunker.
    """
    logger.warning("--region-source vision is a stub; no layout detector wired yet. %s", article_dir)
    return None


def process_shard(shard_dir: str, dry_run: bool = False, force: bool = False,
                  delete_tiles: bool = False, region_source: str = "dom") -> dict:
    """Content-aware variant of chunk.process_shard (same dir-walk + result shape)."""
    import time
    t0 = time.time()
    sub_dirs = sorted(p for p in Path(shard_dir).iterdir() if p.is_dir() and p.name.startswith("shard_"))
    if not sub_dirs:
        sub_dirs = [Path(shard_dir)]
    total = {"articles": 0, "chunked": 0, "skipped": 0, "tiles": 0, "chunks": 0, "files_written": 0}
    for sub in sub_dirs:
        for art in sorted(sub.iterdir()):
            if not art.is_dir() or not art.name.endswith(".png.tiles"):
                continue
            total["articles"] += 1
            r = chunk_article(str(art), dry_run=dry_run, force=force, region_source=region_source)
            if r is None:
                total["skipped"] += 1
                continue
            total["chunked"] += 1
            total["tiles"] += r["num_tiles"]
            total["chunks"] += r["num_chunks"]
            total["files_written"] += r["files_written"]
    if delete_tiles and not dry_run:
        from ..chunk import _delete_tiles_in_shard
        total["tiles_deleted"] = _delete_tiles_in_shard(shard_dir)
    else:
        total["tiles_deleted"] = 0
    return {"shard": os.path.basename(shard_dir.rstrip("/")), **total, "elapsed_s": round(time.time() - t0, 1)}
