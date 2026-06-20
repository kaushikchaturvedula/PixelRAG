#!/usr/bin/env python3
"""Synthetic test for boundary_violation: fixed splits a region; content_aware does not. Stdlib only.

Run: PYTHONPATH=research/probe_set python tests/test_boundary_violation.py  (or via pytest)
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research" / "probe_set"))
import boundary_violation as bv  # noqa: E402

# One block spanning a fixed 1024 boundary, plus a 2-row table whose 2nd row also crosses 1024.
REGIONS = {
    "device_pixel_ratio": 1,
    "regions": [
        {"id": 0, "kind": "block", "y": 900, "height": 300},   # 900..1200 — crosses 1024
        {"id": 1, "kind": "table", "y": 1500, "height": 200,
         "rows": [{"y": 1500, "height": 100}, {"y": 1600, "height": 100}]},
    ],
}
# fixed: 1024px strips → boundaries 1024, 2048 → block [900,1200] split; rows ok (1500/1600 not at 1024).
FIXED_CHUNKS = {"tile_height": 8192, "chunks": [
    {"tile_index": 0, "y_offset": 0, "height": 1024, "bbox": {"height": 1024}},
    {"tile_index": 0, "y_offset": 1024, "height": 1024, "bbox": {"height": 1024}},
]}
# content_aware: cut at region boundaries → block whole; row-aligned.
CA_CHUNKS = {"tile_height": 8192, "chunks": [
    {"tile_index": 0, "y_offset": 0, "height": 900, "bbox": {"height": 900}},
    {"tile_index": 0, "y_offset": 900, "height": 300, "bbox": {"height": 300}},   # the block, whole
    {"tile_index": 0, "y_offset": 1200, "height": 500, "bbox": {"height": 500}},
]}


def _mk(tmp: Path, chunks: dict) -> Path:
    art = tmp / "0.png.tiles"
    art.mkdir(parents=True)
    (art / "regions.json").write_text(json.dumps(REGIONS))
    (art / "chunks.json").write_text(json.dumps(chunks))
    return tmp


def test_fixed_splits_block_content_aware_does_not():
    with tempfile.TemporaryDirectory() as df, tempfile.TemporaryDirectory() as dc:
        fixed = _mk(Path(df), FIXED_CHUNKS)
        ca = _mk(Path(dc), CA_CHUNKS)
        rf = bv._arm_rates(fixed, {})
        rc = bv._arm_rates(ca, {})
        assert rf["block_split_rate"] == 1.0, rf      # fixed splits the block
        assert rc["block_split_rate"] == 0.0, rc      # content_aware does not
        print(f"  fixed block_split={rf['block_split_rate']}  content_aware block_split={rc['block_split_rate']} OK")


if __name__ == "__main__":
    test_fixed_splits_block_content_aware_does_not()
    print("boundary_violation test PASSED")
