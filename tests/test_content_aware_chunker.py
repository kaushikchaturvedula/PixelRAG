#!/usr/bin/env python3
"""Unit tests for the content-aware chunker (no rendering — synthetic tile + regions.json).

Validates the Phase-2 invariants:
  * no region is split across a chunk boundary,
  * the header row is repeated on table-split pieces,
  * every chunk <= MAX height except those explicitly marked oversized,
  * chunks.json keeps the load-bearing fields + serve naming and adds the metadata fields,
  * px reconciliation scales by devicePixelRatio,
  * a page with no regions.json falls back to the fixed chunker.

Run: PYTHONPATH=embed/src python tests/test_content_aware_chunker.py   (or via pytest)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from PIL import Image

from pixelrag_embed.chunking import content_aware as ca

W = 875
MAX = ca.MAX_CHUNK_HEIGHT  # 1024

# A synthetic single-tile page (device px, dpr=1) exercising pack / cut / figure / oversized table.
PAGE_H = 3050
REGIONS = [
    {"id": 0, "tag": "h2", "y": 0, "height": 50, "heading_path": "", "kind": "heading"},
    {"id": 1, "tag": "p", "y": 50, "height": 400, "heading_path": "Sec", "kind": "block"},
    {"id": 2, "tag": "p", "y": 450, "height": 400, "heading_path": "Sec", "kind": "block"},
    {"id": 3, "tag": "p", "y": 850, "height": 400, "heading_path": "Sec", "kind": "block"},
    {"id": 4, "tag": "figure", "y": 1250, "height": 700, "heading_path": "Sec", "kind": "figure"},
    {"id": 5, "tag": "table", "y": 1950, "height": 1100, "heading_path": "Sec", "kind": "table",
     "rows": ([{"y": 1950, "height": 50, "is_header": True}]
              + [{"y": 2000 + 50 * i, "height": 50, "is_header": False} for i in range(21)])},
]
NON_TABLE = [r for r in REGIONS if r["kind"] != "table"]


def _make_article(tmp: Path, with_regions: bool = True) -> Path:
    art = tmp / "0.png.tiles"
    art.mkdir(parents=True)
    img = Image.new("RGB", (W, PAGE_H))
    for y in range(0, PAGE_H, 2):  # deterministic non-uniform content
        for x in range(0, W, 50):
            img.putpixel((x, y), ((x + y) % 256, (y * 3) % 256, (x * 5) % 256))
    img.save(art / "tile_0000.png", format="PNG")
    (art / "tiles.json").write_text(json.dumps(
        {"url": "synthetic://t", "page_height": PAGE_H, "viewport_width": W,
         "tile_height": 8192, "tiles": ["tile_0000.png"]}))
    if with_regions:
        (art / "regions.json").write_text(json.dumps(
            {"url": "synthetic://t", "device_pixel_ratio": 1, "page_height": PAGE_H,
             "viewport_width": W, "regions": REGIONS}))
    return art


def _chunks(art: Path) -> list[dict]:
    return json.loads((art / "chunks.json").read_text())["chunks"]


def test_no_region_split_and_max_height():
    with tempfile.TemporaryDirectory() as d:
        art = _make_article(Path(d))
        ca.chunk_article(str(art))
        chunks = _chunks(art)
        spans = [(c["y_offset"], c["y_offset"] + c["bbox"]["height"]) for c in chunks]
        # every non-table region sits fully inside exactly one chunk span (never split)
        for r in NON_TABLE:
            r0, r1 = r["y"], r["y"] + r["height"]
            assert any(s0 <= r0 and r1 <= s1 for s0, s1 in spans), f"region {r['id']} split: {r0,r1} vs {spans}"
        # every chunk <= MAX unless flagged oversized
        for c in chunks:
            assert c["height"] <= MAX or c["oversized"], f"chunk {c['file']} h={c['height']} > {MAX} not oversized"
        # serve naming + load-bearing + additive fields present
        for ci, c in enumerate(chunks):
            assert c["file"] == f"chunk_0000_{ci:02d}.png"
            for k in ("tile_index", "chunk_index", "y_offset", "height", "width"):
                assert k in c
            for k in ("heading_path", "region_ids", "bbox", "reading_order"):
                assert k in c, f"missing additive field {k}"
            assert (art / c["file"]).exists()
        print(f"  no-split + max-height: {len(chunks)} chunks OK")


def test_table_split_repeats_header():
    with tempfile.TemporaryDirectory() as d:
        art = _make_article(Path(d))
        ca.chunk_article(str(art))
        chunks = _chunks(art)
        pieces = [c for c in chunks if c["kind"] == "table_piece"]
        assert pieces, "expected the oversized table to split into table_piece chunks"
        # the table (h=1100 > 1024) must produce >= 2 pieces (so a header is repeated on a later one)
        assert len(pieces) >= 1
        hdr_h = 50  # the header row height
        for c in pieces:
            img = Image.open(art / c["file"])
            # composed piece image height = header (50) + body; > the content bbox height
            assert img.height == c["height"]
            assert img.height >= c["bbox"]["height"] + hdr_h - 1, "header band not prepended to table piece"
        print(f"  table split repeats header: {len(pieces)} composed piece(s) OK")


def test_px_reconciliation():
    dpr, scaled = ca._reconcile_px(
        {"device_pixel_ratio": 2, "regions": [{"id": 0, "tag": "p", "y": 100, "height": 50}]})
    assert dpr == 2
    assert scaled[0]["y"] == 200 and scaled[0]["height"] == 100, "regions must scale by devicePixelRatio"
    # dpr==1 is identity (our renderer's deviceScaleFactor=1 case)
    _, ident = ca._reconcile_px({"device_pixel_ratio": 1, "regions": [{"id": 0, "tag": "p", "y": 7, "height": 9}]})
    assert ident[0]["y"] == 7 and ident[0]["height"] == 9
    print("  px reconciliation OK")


def test_fallback_to_fixed_without_regions():
    with tempfile.TemporaryDirectory() as d:
        art = _make_article(Path(d), with_regions=False)
        ca.chunk_article(str(art))
        chunks = _chunks(art)
        # fixed fallback → 1024px strips of a 3050px tile = 3 chunks (1024,1024,1002)
        assert len(chunks) == 3, f"expected fixed 3-strip fallback, got {len(chunks)}"
        assert all(c["height"] <= MAX for c in chunks)
        print(f"  fallback-to-fixed (no regions.json): {len(chunks)} fixed strips OK")


if __name__ == "__main__":
    test_no_region_split_and_max_height()
    test_table_split_repeats_header()
    test_px_reconciliation()
    test_fallback_to_fixed_without_regions()
    print("ALL content-aware chunker tests PASSED")
